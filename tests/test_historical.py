import numpy as np
import pandas as pd

from fantasy_draft.historical import (
    aggregate_player_seasons,
    calibrate_position_profiles,
    historical_player_pool,
)
from fantasy_draft.models import POSITIONS, Position


def test_aggregate_player_seasons_excludes_byes_and_keeps_active_zeroes() -> None:
    stats = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "season_type": "REG",
                "team": "AAA",
                "player_id": "p1",
                "player_display_name": "Player One",
                "position": "RB",
                "fantasy_points_ppr": 10.0,
            },
            # Dummy team rows establish real games in weeks 2 and 3.
            *[
                {
                    "season": 2024,
                    "week": week,
                    "season_type": "REG",
                    "team": "AAA",
                    "player_id": f"q{week}",
                    "player_display_name": "Dummy",
                    "position": "QB",
                    "fantasy_points_ppr": 1.0,
                }
                for week in (2, 3)
            ],
        ]
    )
    rosters = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": week,
                "game_type": "REG",
                "team": "AAA",
                "gsis_id": "p1",
                "full_name": "Player One",
                "position": "RB",
                "status": status,
            }
            for week, status in ((1, "ACT"), (2, "ACT"), (3, "INA"), (4, "ACT"))
        ]
    )

    result = aggregate_player_seasons(stats, rosters, min_active_weeks=2)
    player = result.iloc[0]
    assert player["eligible_weeks"] == 3
    assert player["active_weeks"] == 2
    assert player["mu"] == 5.0
    assert player["sigma"] == np.std([10.0, 0.0], ddof=1)
    assert player["injury_probability"] == 1 / 3


def _fake_player_seasons() -> pd.DataFrame:
    rows = []
    counts = {Position.QB: 6, Position.RB: 8, Position.WR: 8, Position.TE: 5}
    for season in (2023, 2024):
        for position in POSITIONS:
            for rank in range(1, counts[position] + 1):
                rows.append(
                    {
                        "season": season,
                        "player_id": f"{season}-{position.value}-{rank}",
                        "player_name": "Test Player",
                        "position": position.value,
                        "team": f"T{rank % 4}",
                        "eligible_weeks": 17,
                        "active_weeks": 16,
                        "unavailable_weeks": 1,
                        "mu": 25.0 - rank - 0.1 * (season - 2023),
                        "sigma": 4.0 + rank / 10,
                        "injury_probability": rank / 100,
                        "position_rank": rank,
                    }
                )
    return pd.DataFrame(rows)


def test_calibration_and_historical_pool_are_deterministic() -> None:
    data = _fake_player_seasons()
    counts = {Position.QB: 6, Position.RB: 8, Position.WR: 8, Position.TE: 5}
    profiles, report = calibrate_position_profiles(data, counts)

    assert set(profiles) == set(POSITIONS)
    assert profiles[Position.QB].count == 6
    assert profiles[Position.WR].elite_mu > profiles[Position.WR].floor_mu
    assert report["TE"]["ranked_player_seasons"] == 10

    first = historical_player_pool(data, 2024, counts)
    second = historical_player_pool(data, 2024, counts)
    assert first == second
    assert len(first) == sum(counts.values())
    assert first[0].position is Position.QB
