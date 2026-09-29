import numpy as np

from fantasy_draft.dataset import (
    ValueObservation,
    load_generated_states,
    load_value_observations,
    save_generated_states,
    save_value_observations,
)
from fantasy_draft.draft import DraftEnvironment
from fantasy_draft.evaluation import EvaluationSeedPlan, evaluate_state
from fantasy_draft.generation import generate_states
from fantasy_draft.players import generate_players
from fantasy_draft.policies import RosterAwareSoftmaxPolicy
from fantasy_draft.state import StateEncoder


def test_evaluation_is_reproducible_and_uses_nested_averaging() -> None:
    state = DraftEnvironment().initial_state(generate_players(20))
    policy = RosterAwareSoftmaxPolicy()
    first = evaluate_state(state, policy, 3, 4, 99)
    second = evaluate_state(state, policy, 3, 4, 99)
    assert first == second
    assert first.mean_value == np.mean(first.per_rollout_means)
    assert first.per_rollout_means == tuple(np.mean(row) for row in first.utilities)
    expected_std = np.std(first.per_rollout_means, ddof=1)
    assert np.isclose(first.std, expected_std)
    assert np.isclose(first.standard_error, expected_std / np.sqrt(3))


def test_seed_plan_supports_matched_comparisons() -> None:
    state = DraftEnvironment().initial_state(generate_players(21))
    plan = EvaluationSeedPlan.create(5, 2, 3)
    policy = RosterAwareSoftmaxPolicy()
    first = evaluate_state(
        state, policy, 2, 3, 0, seed_plan=plan
    )
    second = evaluate_state(
        state, policy, 2, 3, 999, seed_plan=plan
    )
    assert first.utilities == second.utilities
    assert first.terminal_rosters == second.terminal_rosters


def test_continuation_policy_parameter_variants_produce_different_values() -> None:
    state = DraftEnvironment().initial_state(generate_players(22))
    conservative = evaluate_state(
        state, RosterAwareSoftmaxPolicy(temperature=1.0, starter_need_bonus=8.0), 3, 2, 2
    )
    diffuse = evaluate_state(
        state, RosterAwareSoftmaxPolicy(temperature=6.0, starter_need_bonus=1.0), 3, 2, 2
    )
    assert conservative.terminal_rosters != diffuse.terminal_rosters
    assert conservative.mean_value != diffuse.mean_value


def test_generated_state_and_value_dataset_round_trips(tmp_path) -> None:
    generated = generate_states(3, 30)
    state_path = tmp_path / "states.jsonl"
    save_generated_states(state_path, generated)
    restored = load_generated_states(state_path)
    assert restored == generated

    result = evaluate_state(restored[0].state, RosterAwareSoftmaxPolicy(), 2, 2, 31)
    observation = ValueObservation.from_result(
        restored[0],
        StateEncoder().encode_state(restored[0].state),
        result,
    )
    value_path = tmp_path / "values.npz"
    save_value_observations(value_path, [observation])
    loaded = load_value_observations(value_path)
    assert len(loaded) == 1
    assert loaded[0].state_id == observation.state_id
    assert np.array_equal(loaded[0].state_vector, observation.state_vector)
    assert loaded[0].estimated_value == observation.estimated_value


def test_state_generation_is_reproducible_and_stratified() -> None:
    first = generate_states(18, 40)
    second = generate_states(18, 40)
    assert first == second
    assert {item.phase for item in first} == {"early", "middle", "late"}
    assert len({item.league_config_id for item in first}) >= 6
    assert {item.state.config.n_teams for item in first}.issubset({8, 10, 12})
    assert all(item.player_pool_seed != item.draft_history_seed for item in first)
    assert all(item.encoder_version == 2 for item in first)
    assert all(
        item.state.current_team == item.state.config.controlled_team for item in first
    )
