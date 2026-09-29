"""Historical nflverse calibration helpers.

The functions in this module are deliberately network-free.  The command-line
pipeline in ``scripts/calibrate_nflfastr.py`` handles downloads and passes
weekly stats/rosters into these helpers.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

from fantasy_draft.config import PositionProfile
from fantasy_draft.models import POSITIONS, Player, Position

HISTORICAL_POOL_COUNTS: dict[Position, int] = {
    Position.QB: 24,
    Position.RB: 36,
    Position.WR: 42,
    Position.TE: 18,
}
ACTIVE_STATUS = "ACT"


def _column(frame: pd.DataFrame, *candidates: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"none of the required columns are present: {candidates}")


def aggregate_player_seasons(
    stats: pd.DataFrame,
    rosters: pd.DataFrame,
    *,
    min_active_weeks: int = 4,
) -> pd.DataFrame:
    """Estimate healthy scoring and unavailability by player-season.

    Team-game weeks observed in player stats are joined to weekly rosters,
    removing bye weeks.  Active roster weeks without a stats row are genuine
    zero-point observations; inactive/reserve weeks estimate unavailability.
    """

    if min_active_weeks < 2:
        raise ValueError("min_active_weeks must be at least 2")

    stats_team = _column(stats, "team", "recent_team")
    stats_player = _column(stats, "player_id", "gsis_id")
    roster_player = _column(rosters, "gsis_id", "player_id")
    points = _column(stats, "fantasy_points_ppr")
    stat_name = next(
        (name for name in ("player_display_name", "player_name", "full_name") if name in stats),
        None,
    )
    roster_name = next(
        (name for name in ("full_name", "football_name") if name in rosters),
        None,
    )

    stat = stats.copy()
    roster = rosters.copy()
    if "season_type" in stat:
        stat = stat[stat["season_type"].eq("REG")]
    if "game_type" in roster:
        roster = roster[roster["game_type"].eq("REG")]
    stat = stat[stat["position"].isin([position.value for position in POSITIONS])]
    roster = roster[roster["position"].isin([position.value for position in POSITIONS])]

    stat = stat.rename(columns={stats_team: "team_key", stats_player: "player_key"})
    roster = roster.rename(columns={roster_player: "player_key", "team": "team_key"})
    stat["player_key"] = stat["player_key"].astype("string")
    roster["player_key"] = roster["player_key"].astype("string")

    game_weeks = stat[["season", "week", "team_key"]].dropna().drop_duplicates()
    roster = roster.merge(game_weeks, on=["season", "week", "team_key"], how="inner")

    roster["_available"] = roster["status"].fillna("").eq(ACTIVE_STATUS)
    roster_cols = ["season", "week", "team_key", "player_key", "position"]
    if roster_name:
        roster_cols.append(roster_name)
    roster_week = (
        roster[roster_cols + ["_available"]]
        .sort_values("_available", ascending=False)
        .drop_duplicates(["season", "week", "player_key"])
    )

    stat_cols = ["season", "week", "player_key", points]
    if stat_name:
        stat_cols.append(stat_name)
    stat_week = stat[stat_cols].copy()
    stat_week[points] = pd.to_numeric(stat_week[points], errors="coerce").fillna(0.0)
    aggregations: dict[str, str] = {points: "sum"}
    if stat_name:
        aggregations[stat_name] = "first"
    stat_week = stat_week.groupby(
        ["season", "week", "player_key"], as_index=False
    ).agg(aggregations)

    weekly = roster_week.merge(
        stat_week,
        on=["season", "week", "player_key"],
        how="left",
    )
    weekly["_points"] = weekly[points].fillna(0.0)
    weekly.loc[~weekly["_available"], "_points"] = np.nan
    name_candidates = [name for name in (stat_name, roster_name) if name and name in weekly]
    if name_candidates:
        weekly["_name"] = weekly[name_candidates].bfill(axis=1).iloc[:, 0]
    else:
        weekly["_name"] = weekly["player_key"]

    records: list[dict[str, Any]] = []
    group_cols = ["season", "player_key"]
    for (season, player_key), group in weekly.groupby(group_cols, sort=False):
        active_points = group.loc[group["_available"], "_points"].dropna()
        if len(active_points) < min_active_weeks:
            continue
        records.append(
            {
                "season": int(season),
                "player_id": str(player_key),
                "player_name": str(group["_name"].dropna().iloc[0]),
                "position": str(group["position"].mode().iloc[0]),
                "team": str(group["team_key"].mode().iloc[0]),
                "eligible_weeks": int(len(group)),
                "active_weeks": int(group["_available"].sum()),
                "unavailable_weeks": int((~group["_available"]).sum()),
                "mu": float(active_points.mean()),
                "sigma": float(active_points.std(ddof=1)),
                "injury_probability": float((~group["_available"]).mean()),
            }
        )

    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    result["sigma"] = result["sigma"].fillna(0.0)
    result = result[result["mu"] > 0].copy()
    result["position_rank"] = (
        result.groupby(["season", "position"])["mu"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return result.sort_values(["season", "position", "position_rank"]).reset_index(drop=True)


def _fit_rank_curve(rank_values: pd.DataFrame, count: int) -> tuple[float, float, float]:
    means = (
        rank_values[rank_values["position_rank"].le(count)]
        .groupby("position_rank")["mu"]
        .median()
        .reindex(range(1, count + 1))
        .interpolate(limit_direction="both")
        .to_numpy(dtype=float)
    )
    fraction = np.arange(count, dtype=float) / max(count - 1, 1)
    best: tuple[float, float, float, float] | None = None
    for power in np.linspace(0.15, 1.75, 321):
        basis = 1.0 - fraction**power
        design = np.column_stack([np.ones(count), basis])
        floor, spread = np.linalg.lstsq(design, means, rcond=None)[0]
        prediction = floor + spread * basis
        error = float(np.mean((means - prediction) ** 2))
        candidate = (error, float(floor + spread), float(floor), float(power))
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    _, elite, floor, power = best
    return max(elite, floor), max(0.0, floor), power


def _beta_moments(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    variance = float(np.var(values))
    max_variance = mean * (1.0 - mean)
    if mean <= 0.0 or mean >= 1.0 or variance <= 1e-8 or variance >= max_variance:
        concentration = 30.0
    else:
        concentration = max(2.0, max_variance / variance - 1.0)
    return max(0.1, mean * concentration), max(0.1, (1.0 - mean) * concentration)


def calibrate_position_profiles(
    player_seasons: pd.DataFrame,
    counts: dict[Position, int] | None = None,
) -> tuple[dict[Position, PositionProfile], dict[str, Any]]:
    """Fit ``PositionProfile`` values from historical player-seasons."""

    counts = counts or HISTORICAL_POOL_COUNTS
    profiles: dict[Position, PositionProfile] = {}
    report: dict[str, Any] = {}
    for position in POSITIONS:
        position_data = player_seasons[player_seasons["position"].eq(position.value)].copy()
        count = counts[position]
        ranked = position_data[position_data["position_rank"].le(count)].copy()
        if ranked.empty or ranked["position_rank"].nunique() < count // 2:
            raise ValueError(f"insufficient historical observations for {position.value}")
        elite, floor, power = _fit_rank_curve(ranked, count)
        rank_median = ranked.groupby("position_rank")["mu"].transform("median")
        mu_noise = float(np.std(ranked["mu"] - rank_median, ddof=1))
        sigma_low, sigma_high = np.quantile(ranked["sigma"], [0.10, 0.90])
        injury_values = ranked["injury_probability"].clip(0.0, 0.999).to_numpy(float)
        alpha, beta = _beta_moments(injury_values)
        injury_max = float(np.quantile(injury_values, 0.95))
        profile = PositionProfile(
            count=count,
            elite_mu=round(elite, 4),
            floor_mu=round(floor, 4),
            decay_power=round(power, 4),
            mu_noise=round(max(mu_noise, 0.05), 4),
            sigma_low=round(float(sigma_low), 4),
            sigma_high=round(float(sigma_high), 4),
            injury_alpha=round(alpha, 4),
            injury_beta=round(beta, 4),
            injury_max=round(max(injury_max, 0.01), 4),
        )
        profiles[position] = profile
        report[position.value] = {
            "profile": asdict(profile),
            "player_seasons": int(len(position_data)),
            "ranked_player_seasons": int(len(ranked)),
            "mu_quantiles": [
                round(float(value), 4)
                for value in np.quantile(ranked["mu"], [0.1, 0.5, 0.9])
            ],
            "sigma_quantiles": [
                round(float(value), 4)
                for value in np.quantile(ranked["sigma"], [0.1, 0.5, 0.9])
            ],
            "unavailable_probability_quantiles": [
                round(float(value), 4)
                for value in np.quantile(
                    ranked["injury_probability"], [0.1, 0.5, 0.9]
                )
            ],
        }
    return profiles, report


def historical_player_pool(
    player_seasons: pd.DataFrame,
    season: int,
    counts: dict[Position, int] | None = None,
) -> tuple[Player, ...]:
    """Convert one season's ex-post distributions into a draft player pool."""

    counts = counts or HISTORICAL_POOL_COUNTS
    season_data = player_seasons[player_seasons["season"].eq(season)]
    team_names = sorted(season_data["team"].dropna().unique())
    team_ids = {name: index for index, name in enumerate(team_names)}
    players: list[Player] = []
    next_id = 0
    for position in POSITIONS:
        rows = (
            season_data[season_data["position"].eq(position.value)]
            .sort_values(["mu", "player_id"], ascending=[False, True])
            .head(counts[position])
        )
        if len(rows) < counts[position]:
            raise ValueError(
                f"{season} has only {len(rows)} eligible {position.value} players"
            )
        for row in rows.itertuples(index=False):
            players.append(
                Player(
                    player_id=next_id,
                    position=position,
                    mu=round(float(row.mu), 4),
                    sigma=round(float(row.sigma), 4),
                    injury_probability=round(float(row.injury_probability), 5),
                    team_id=team_ids[str(row.team)],
                )
            )
            next_id += 1
    return tuple(players)


def profile_report(profiles: Iterable[PositionProfile]) -> list[dict[str, Any]]:
    """Serialize profiles for simple command-line diagnostics."""

    return [asdict(profile) for profile in profiles]
