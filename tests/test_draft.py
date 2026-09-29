import numpy as np
import pytest

from fantasy_draft.config import LeagueConfig, RosterConfig
from fantasy_draft.draft import DraftEnvironment
from fantasy_draft.models import Position
from fantasy_draft.players import generate_players
from fantasy_draft.policies import RosterAwareSoftmaxPolicy


def test_snake_order() -> None:
    environment = DraftEnvironment()
    order = [environment.team_at_pick(index) for index in range(24)]
    assert order[:8] == list(range(8))
    assert order[8:16] == list(reversed(range(8)))
    assert order[16:24] == list(range(8))


@pytest.mark.parametrize("n_teams", [8, 10, 12])
def test_snake_order_and_next_turn_for_multiple_league_sizes(n_teams: int) -> None:
    environment = DraftEnvironment(LeagueConfig(n_teams=n_teams))
    order = [environment.team_at_pick(index) for index in range(n_teams * 2)]
    assert order[:n_teams] == list(range(n_teams))
    assert order[n_teams:] == list(reversed(range(n_teams)))
    state = environment.initial_state(generate_players(100 + n_teams))
    assert environment.picks_until_team(state, 0) == 0
    state = environment.step(state, environment.legal_actions(state)[0])
    assert environment.picks_until_team(state, 0) == 2 * n_teams - 2


@pytest.mark.parametrize(
    ("qb", "flex", "bench", "expected_rounds"),
    [(1, 1, 3, 10), (1, 2, 3, 11), (2, 1, 3, 11), (2, 2, 4, 13)],
)
def test_draft_length_derives_from_roster_configuration(
    qb: int, flex: int, bench: int, expected_rounds: int
) -> None:
    roster = RosterConfig(
        required={
            Position.QB: qb,
            Position.RB: 2,
            Position.WR: 2,
            Position.TE: 1,
        },
        flex_slots=flex,
        bench_slots=bench,
    )
    config = LeagueConfig(n_teams=10, roster=roster)
    assert config.n_rounds == expected_rounds
    assert config.total_picks == 10 * expected_rounds


def test_step_removes_player_and_updates_current_roster() -> None:
    environment = DraftEnvironment()
    state = environment.initial_state(generate_players(2))
    action = environment.legal_actions(state)[0]
    next_state = environment.step(state, action)
    assert action not in next_state.available_ids
    assert next_state.rosters[0] == (action,)
    assert action in state.available_ids  # transitions do not mutate their input


def test_draft_terminates_at_expected_size_with_feasible_rosters() -> None:
    environment = DraftEnvironment()
    state = environment.initial_state(generate_players(3))
    policy = RosterAwareSoftmaxPolicy()
    rng = np.random.default_rng(99)
    while not state.is_terminal:
        state = environment.step(state, policy.select_player(state, environment, rng))
    assert state.pick_index == 80
    assert all(len(roster) == 10 for roster in state.rosters)
    assert len(state.available_ids) == 40
    assert environment.legal_actions(state) == ()


def test_physical_legality_remains_nonempty_until_draft_completion() -> None:
    environment = DraftEnvironment()
    state = environment.initial_state(generate_players(4))
    rng = np.random.default_rng(5)
    policy = RosterAwareSoftmaxPolicy()
    # Physical legality remains available even when roster composition is poor.
    while not state.is_terminal:
        legal = environment.legal_actions(state)
        assert legal
        state = environment.step(state, policy.select_player(state, environment, rng))


def test_strategically_bad_extra_qb_remains_legal() -> None:
    environment = DraftEnvironment()
    state = environment.initial_state(generate_players(44))
    while len(state.rosters[0]) < 4 or state.current_team != 0:
        legal = environment.legal_actions(state)
        lookup = state.player_map
        if state.current_team == 0:
            qbs = [player_id for player_id in legal if lookup[player_id].position == Position.QB]
            action = qbs[0]
        else:
            non_qbs = [
                player_id
                for player_id in legal
                if lookup[player_id].position != Position.QB
            ]
            action = non_qbs[0]
        state = environment.step(state, action)
    lookup = state.player_map
    remaining_qbs = [
        player_id
        for player_id in state.available_ids
        if lookup[player_id].position == Position.QB
    ]
    assert remaining_qbs
    assert set(remaining_qbs).issubset(environment.legal_actions(state))


def test_illegal_duplicate_pick_is_rejected() -> None:
    environment = DraftEnvironment()
    state = environment.initial_state(generate_players(5))
    action = environment.legal_actions(state)[0]
    state = environment.step(state, action)
    with pytest.raises(ValueError, match="illegal"):
        environment.step(state, action)
