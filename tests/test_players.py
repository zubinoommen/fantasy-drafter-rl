import numpy as np

from fantasy_draft.config import default_position_profiles
from fantasy_draft.models import POSITIONS, Position
from fantasy_draft.players import generate_players


def test_generation_is_reproducible_and_has_default_size() -> None:
    first = generate_players(123)
    second = generate_players(123)
    assert first == second
    assert len(first) == 120
    assert len({player.player_id for player in first}) == 120


def test_position_profiles_differ_and_values_are_sensible() -> None:
    players = generate_players(7)
    means = {
        position: np.mean([player.mu for player in players if player.position == position])
        for position in POSITIONS
    }
    assert len({round(value, 1) for value in means.values()}) > 2
    assert means[Position.QB] > means[Position.TE]
    assert all(0 < player.mu < 30 for player in players)
    assert all(0 < player.sigma < 12 for player in players)
    profiles = default_position_profiles()
    assert all(
        0 <= player.injury_probability <= profiles[player.position].injury_max
        for player in players
    )


def test_position_depth_counts_are_distinct() -> None:
    players = generate_players(1)
    counts = {
        position: sum(player.position == position for player in players)
        for position in POSITIONS
    }
    assert counts == {
        Position.QB: 24,
        Position.RB: 36,
        Position.WR: 42,
        Position.TE: 18,
    }
