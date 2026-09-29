import numpy as np
import pytest

from fantasy_draft.config import LeagueConfig, RosterConfig
from fantasy_draft.draft import DraftEnvironment
from fantasy_draft.models import Position
from fantasy_draft.players import generate_players
from fantasy_draft.policies import (
    RosterAwareSoftmaxPolicy,
    projected_replacement_levels,
)


def test_roster_aware_policy_always_selects_legal_action() -> None:
    environment = DraftEnvironment()
    state = environment.initial_state(generate_players(6))
    rng = np.random.default_rng(10)
    policy = RosterAwareSoftmaxPolicy()
    for _ in range(30):
        legal = environment.legal_actions(state)
        action = policy.select_player(state, environment, rng)
        assert action in legal
        state = environment.step(state, action)


def test_temperature_must_be_positive() -> None:
    with pytest.raises(ValueError):
        RosterAwareSoftmaxPolicy(temperature=0.0)


def test_replacement_value_weight_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        RosterAwareSoftmaxPolicy(value_above_replacement_weight=-1.0)


def test_replacement_value_responds_to_two_qb_demand() -> None:
    players = generate_players(9)
    standard = DraftEnvironment(LeagueConfig(n_teams=10)).initial_state(players)
    two_qb_roster = RosterConfig(
        required={
            Position.QB: 2,
            Position.RB: 2,
            Position.WR: 2,
            Position.TE: 1,
        },
        flex_slots=1,
        bench_slots=3,
    )
    two_qb = DraftEnvironment(
        LeagueConfig(n_teams=10, roster=two_qb_roster)
    ).initial_state(players)
    standard_levels = projected_replacement_levels(standard)
    two_qb_levels = projected_replacement_levels(two_qb)
    assert two_qb_levels[Position.QB] < standard_levels[Position.QB]
    assert two_qb_levels[Position.RB] == standard_levels[Position.RB]


def test_replacement_value_reduces_one_qb_first_pick_bias() -> None:
    environment = DraftEnvironment(LeagueConfig(n_teams=10))
    state = environment.initial_state(generate_players(42))
    policy = RosterAwareSoftmaxPolicy()
    rng = np.random.default_rng(123)
    positions = [
        state.player_map[policy.select_player(state, environment, rng)].position
        for _ in range(2_000)
    ]
    qb_share = positions.count(Position.QB) / len(positions)
    assert 0.07 < qb_share < 0.20


def test_starter_constraint_forces_unfilled_starter_positions_at_end_of_draft() -> None:
    """Verify that policies guarantee every team drafts all required starting positions."""
    environment = DraftEnvironment()
    state = environment.initial_state(generate_players(55))
    policy = RosterAwareSoftmaxPolicy()
    rng = np.random.default_rng(42)

    while not state.is_terminal:
        action = policy.select_player(state, environment, rng)
        state = environment.step(state, action)

    required = state.config.roster.required
    for team in range(state.config.n_teams):
        counts = state.roster_position_counts(team)
        for position, req_count in required.items():
            assert counts[position] >= req_count, f"Team {team} missing required starter {position}"


