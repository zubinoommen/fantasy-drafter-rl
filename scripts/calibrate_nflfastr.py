#!/usr/bin/env python3
"""Calibrate the synthetic DGP and benchmark policies on nflverse data.

The upstream data are nflfastR/nflverse weekly player summaries and weekly
rosters.  Downloads are cached locally so reruns are deterministic and cheap.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fantasy_draft.config import LeagueConfig, RosterConfig
from fantasy_draft.draft import DraftEnvironment, DraftState
from fantasy_draft.historical import (
    aggregate_player_seasons,
    calibrate_position_profiles,
    historical_player_pool,
)
from fantasy_draft.models import Position
from fantasy_draft.policies import RosterAwareSoftmaxPolicy
from fantasy_draft.season import SeasonSimulator, mean_weekly_score
from fantasy_draft.state import StateEncoder

PLAYER_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.parquet"
)
ROSTERS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "weekly_rosters/roster_weekly_{season}.parquet"
)


class ValueMLP(nn.Module):
    """Checkpoint-compatible value network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class HistoricalMLPAgent:
    """Greedy checkpoint policy used by the historical benchmark."""

    def __init__(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model = ValueMLP(**checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.feature_mean = checkpoint["feature_mean"].float()
        self.feature_std = checkpoint["feature_std"].float()
        self.encoder = StateEncoder()

    def select_player(self, state: DraftState, environment: DraftEnvironment) -> int:
        legal = environment.legal_actions(state)
        post_pick = [
            environment.step(state, action, validate=False) for action in legal
        ]
        vectors = np.stack([self.encoder.encode_state(item) for item in post_pick])
        features = torch.from_numpy(vectors).float()
        normalized = (features - self.feature_mean) / self.feature_std
        with torch.no_grad():
            predictions = self.model(normalized).squeeze(1).numpy()
        return int(legal[int(np.argmax(predictions))])


def _download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "fantasy-draft-mdp historical calibration"},
    )
    print(f"Downloading {url}")
    with urllib.request.urlopen(request, timeout=120) as response:
        temporary.write_bytes(response.read())
    temporary.replace(destination)


def download_frames(
    seasons: list[int],
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_frames: list[pd.DataFrame] = []
    roster_frames: list[pd.DataFrame] = []
    for season in seasons:
        stats_path = cache_dir / f"stats_player_week_{season}.parquet"
        roster_path = cache_dir / f"roster_weekly_{season}.parquet"
        _download(PLAYER_STATS_URL.format(season=season), stats_path)
        _download(ROSTERS_URL.format(season=season), roster_path)
        stats_frames.append(pd.read_parquet(stats_path))
        roster_frames.append(pd.read_parquet(roster_path))
    return (
        pd.concat(stats_frames, ignore_index=True),
        pd.concat(roster_frames, ignore_index=True),
    )


def _league(slot: int) -> LeagueConfig:
    return LeagueConfig(
        n_teams=10,
        controlled_team=slot,
        roster=RosterConfig(
            required={
                Position.QB: 1,
                Position.RB: 2,
                Position.WR: 2,
                Position.TE: 1,
            },
            flex_slots=1,
            bench_slots=3,
        ),
    )


def _complete_draft(
    players: tuple,
    league: LeagueConfig,
    *,
    controlled_policy: HistoricalMLPAgent | None,
    seed: int,
) -> DraftState:
    environment = DraftEnvironment(league)
    state = environment.initial_state(players)
    baseline = RosterAwareSoftmaxPolicy(temperature=3.0)
    rng = np.random.default_rng(seed)
    while not state.is_terminal:
        if (
            controlled_policy is not None
            and state.current_team == league.controlled_team
        ):
            action = controlled_policy.select_player(state, environment)
        else:
            action = baseline.select_player(state, environment, rng)
        state = environment.step(state, action, validate=False)
    return state


def _expected_score(
    state: DraftState,
    team: int,
    seeds: tuple[int, ...],
) -> float:
    simulator = SeasonSimulator(
        roster_config=state.config.roster,
        n_weeks=state.config.season_weeks,
    )
    results = simulator.simulate_many(
        state.roster_players(team),
        state.available_players(),
        seeds,
        player_universe=state.players,
    )
    return float(np.mean([mean_weekly_score(result.weekly_scores) for result in results]))


def run_hindsight_benchmark(
    player_seasons: pd.DataFrame,
    checkpoint_path: Path,
    *,
    seasons: list[int],
    slots: list[int],
    drafts_per_slot: int,
    season_simulations: int,
    seed: int,
) -> pd.DataFrame:
    """Compare MLP and baseline on ex-post season distributions."""

    agent = HistoricalMLPAgent(checkpoint_path)
    records: list[dict[str, Any]] = []
    for season in seasons:
        players = historical_player_pool(player_seasons, season)
        for slot in slots:
            league = _league(slot)
            for replicate in range(drafts_per_slot):
                draft_seed = seed + season * 10_000 + slot * 100 + replicate
                simulation_rng = np.random.default_rng(draft_seed + 1)
                simulation_seeds = tuple(
                    int(value)
                    for value in simulation_rng.integers(
                        0, 1 << 31, size=season_simulations
                    )
                )
                mlp_state = _complete_draft(
                    players,
                    league,
                    controlled_policy=agent,
                    seed=draft_seed,
                )
                baseline_state = _complete_draft(
                    players,
                    league,
                    controlled_policy=None,
                    seed=draft_seed,
                )
                mlp_score = _expected_score(
                    mlp_state, league.controlled_team, simulation_seeds
                )
                baseline_score = _expected_score(
                    baseline_state, league.controlled_team, simulation_seeds
                )
                opponent_scores = [
                    _expected_score(mlp_state, team, simulation_seeds)
                    for team in range(league.n_teams)
                    if team != league.controlled_team
                ]
                records.append(
                    {
                        "season": season,
                        "draft_slot": slot,
                        "replicate": replicate,
                        "draft_seed": draft_seed,
                        "season_simulations": season_simulations,
                        "mlp_weekly_points": mlp_score,
                        "baseline_weekly_points": baseline_score,
                        "opponent_field_weekly_points": float(np.mean(opponent_scores)),
                        "paired_advantage": mlp_score - baseline_score,
                        "field_advantage": mlp_score - float(np.mean(opponent_scores)),
                    }
                )
        print(f"Benchmarked ex-post {season} player distributions")
    return pd.DataFrame.from_records(records)


def _parse_int_list(value: str) -> list[int]:
    if ":" in value:
        start, end = (int(part) for part in value.split(":", maxsplit=1))
        return list(range(start, end + 1))
    return [int(part) for part in value.split(",")]


def _confidence_interval(values: pd.Series) -> list[float]:
    mean = float(values.mean())
    if len(values) < 2:
        return [mean, mean]
    half_width = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return [mean - half_width, mean + half_width]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default="2018:2025")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "nflfastr")
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "data" / "nflverse_cache"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "models" / "value_mlp.pt"
    )
    parser.add_argument("--slots", default="0,4,9")
    parser.add_argument("--drafts-per-slot", type=int, default=3)
    parser.add_argument("--season-simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    seasons = _parse_int_list(args.seasons)
    slots = _parse_int_list(args.slots)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats, rosters = download_frames(seasons, args.cache_dir)
    player_seasons = aggregate_player_seasons(stats, rosters)
    if set(player_seasons["season"].unique()) != set(seasons):
        raise RuntimeError("one or more requested seasons produced no player distributions")
    player_path = args.output_dir / "player_seasons.parquet"
    player_seasons.to_parquet(player_path, index=False, compression="zstd")

    profiles, positional_report = calibrate_position_profiles(player_seasons)
    rank_summary = (
        player_seasons.groupby(["position", "position_rank"])
        .agg(
            seasons=("season", "nunique"),
            mu_median=("mu", "median"),
            sigma_median=("sigma", "median"),
            unavailable_probability_median=("injury_probability", "median"),
        )
        .reset_index()
    )
    rank_summary.to_parquet(
        args.output_dir / "position_rank_summary.parquet",
        index=False,
        compression="zstd",
    )
    calibration = {
        "schema_version": 1,
        "seasons": seasons,
        "scoring": "full_ppr",
        "season_type": "REG",
        "availability_definition": (
            "weekly roster status ACT on observed team-game weeks; "
            "active weeks without stats are zero points"
        ),
        "minimum_active_weeks": 4,
        "sources": {
            "player_stats": PLAYER_STATS_URL,
            "weekly_rosters": ROSTERS_URL,
        },
        "rows": int(len(player_seasons)),
        "positions": positional_report,
        "profiles": {
            position.value: asdict(profile) for position, profile in profiles.items()
        },
    }
    with open(args.output_dir / "calibration.json", "w", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=2)
    print(f"Wrote {len(player_seasons):,} player-seasons to {player_path}")
    print(json.dumps(calibration["profiles"], indent=2))

    if args.skip_benchmark:
        return
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {args.checkpoint}; use --skip-benchmark to calibrate only"
        )
    benchmark = run_hindsight_benchmark(
        player_seasons,
        args.checkpoint,
        seasons=seasons,
        slots=slots,
        drafts_per_slot=args.drafts_per_slot,
        season_simulations=args.season_simulations,
        seed=args.seed,
    )
    benchmark.to_parquet(
        args.output_dir / "hindsight_benchmark.parquet",
        index=False,
        compression="zstd",
    )
    summary = {
        "rows": int(len(benchmark)),
        "mean_mlp_weekly_points": float(benchmark["mlp_weekly_points"].mean()),
        "mean_baseline_weekly_points": float(
            benchmark["baseline_weekly_points"].mean()
        ),
        "mean_paired_advantage": float(benchmark["paired_advantage"].mean()),
        "paired_advantage_95_ci": _confidence_interval(
            benchmark["paired_advantage"]
        ),
        "paired_win_rate": float((benchmark["paired_advantage"] > 0).mean()),
        "mean_field_advantage": float(benchmark["field_advantage"].mean()),
        "field_win_rate": float((benchmark["field_advantage"] > 0).mean()),
        "interpretation": (
            "Hindsight distribution benchmark on simulated drafts, not a "
            "reconstruction of historical fantasy drafts or ADP."
        ),
    }
    with open(
        args.output_dir / "hindsight_benchmark_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
