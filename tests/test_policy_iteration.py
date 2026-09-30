from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from fantasy_draft.config import LeagueConfig
from fantasy_draft.draft import DraftEnvironment
from fantasy_draft.generation import LeagueConfigDistribution
from fantasy_draft.players import generate_players
from fantasy_draft.policy_iteration import (
    CollectionResult,
    ControlledVsOpponentPolicy,
    GreedyValuePolicy,
    PolicyIterationConfig,
    TrainingResult,
    ValueMLP,
    collect_policy_trajectories,
    relabel_replay_states,
    run_fitted_policy_iteration,
)
from fantasy_draft.state import StateEncoder


class _FirstPolicy:
    name = "first"
    is_deterministic = True

    def __init__(self) -> None:
        self.calls = 0

    def select_player(self, state, environment, rng) -> int:
        del rng
        self.calls += 1
        return environment.legal_actions(state)[0]


class _LastPolicy:
    name = "last"
    is_deterministic = True

    def __init__(self) -> None:
        self.calls = 0

    def select_player(self, state, environment, rng) -> int:
        del rng
        self.calls += 1
        return environment.legal_actions(state)[-1]


def _zero_checkpoint() -> dict:
    model_config = {
        "input_dim": len(StateEncoder().feature_names()),
        "hidden_dims": (8,),
        "dropout": 0.0,
    }
    model = ValueMLP(**model_config)
    for parameter in model.parameters():
        parameter.data.zero_()
    return {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "feature_mean": torch.zeros(model_config["input_dim"]),
        "feature_std": torch.ones(model_config["input_dim"]),
        "target_mean": 0.0,
        "target_std": 1.0,
        "iteration": 0,
    }


def test_controlled_policy_routes_only_controlled_turns() -> None:
    league = LeagueConfig(n_teams=2, controlled_team=0)
    environment = DraftEnvironment(league)
    state = environment.initial_state(generate_players(1))
    controlled = _FirstPolicy()
    opponent = _LastPolicy()
    policy = ControlledVsOpponentPolicy(controlled, opponent)
    rng = np.random.default_rng(2)

    first_action = policy.select_player(state, environment, rng)
    state = environment.step(state, first_action)
    second_action = policy.select_player(state, environment, rng)

    assert controlled.calls == 1
    assert opponent.calls == 1
    assert first_action == 0
    assert second_action == max(state.available_ids)


def test_greedy_checkpoint_policy_is_deterministic() -> None:
    environment = DraftEnvironment()
    state = environment.initial_state(generate_players(3))
    policy = GreedyValuePolicy(_zero_checkpoint())

    first = policy.select_player(state, environment, np.random.default_rng(1))
    second = policy.select_player(state, environment, np.random.default_rng(99))

    assert first == second == min(environment.legal_actions(state))


def test_trajectory_rows_share_terminal_return_by_draft(tmp_path: Path) -> None:
    output = tmp_path / "iteration.parquet"
    distribution = LeagueConfigDistribution(
        team_counts=(2,),
        qb_requirements=(1,),
        flex_requirements=(1,),
        bench_slots=(1,),
    )
    result = collect_policy_trajectories(
        n_samples=12,
        controlled_policy=_FirstPolicy(),
        output_path=output,
        iteration=0,
        season_simulations=2,
        epsilon=0.0,
        seed=4,
        league_distribution=distribution,
        verbose=False,
    )
    frame = pq.read_table(output).drop(["state_vector"]).to_pandas()

    assert result.rows == 12
    assert frame.groupby("draft_id")["reward"].nunique().max() == 1
    assert frame.groupby("draft_id")["season_seed"].nunique().max() == 1
    assert set(frame["iteration"]) == {0}
    assert frame["draft_id"].nunique() == result.drafts


def test_replay_states_are_relabelled_for_current_iteration(tmp_path: Path) -> None:
    output = tmp_path / "iteration_00.parquet"
    distribution = LeagueConfigDistribution(
        team_counts=(2,),
        qb_requirements=(1,),
        flex_requirements=(1,),
        bench_slots=(1,),
    )
    collect_policy_trajectories(
        n_samples=12,
        controlled_policy=_FirstPolicy(),
        output_path=output,
        iteration=0,
        season_simulations=2,
        epsilon=0.0,
        seed=9,
        league_distribution=distribution,
        verbose=False,
    )

    replay = relabel_replay_states(
        [output],
        n_samples=3,
        controlled_policy=_LastPolicy(),
        iteration=1,
        season_simulations=1,
        epsilon=0.0,
        replay_decay=0.5,
        seed=10,
        verbose=False,
    )

    assert len(replay) == 3
    assert all(row["is_replay"] for row in replay)
    assert {row["iteration"] for row in replay} == {1}
    assert {row["source_iteration"] for row in replay} == {0}
    assert all("last" in row["evaluation_policy"] for row in replay)
    assert all(row["season_simulations"] == 1 for row in replay)


def test_run_loop_resumes_completed_iteration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import fantasy_draft.policy_iteration as module

    calls = {"collect": 0, "train": 0, "benchmark": 0}
    checkpoint_data = _zero_checkpoint()

    def fake_collect(**kwargs):
        calls["collect"] += 1
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("dataset", encoding="utf-8")
        return CollectionResult(10, 2, output, 100.0, 1.0)

    def fake_train(dataset_path, checkpoint_path, **kwargs):
        del dataset_path
        calls["train"] += 1
        checkpoint_data["iteration"] = kwargs["iteration"]
        torch.save(checkpoint_data, checkpoint_path)
        return TrainingResult(
            Path(checkpoint_path),
            ({"epoch": 1.0, "train_loss": 1.0, "val_loss": 1.0},),
            {"MAE": 1.0, "RMSE": 1.0, "R2": 0.0},
            {"train": 6, "val": 2, "test": 2},
        )

    def fake_benchmark(*args, **kwargs):
        del args, kwargs
        calls["benchmark"] += 1
        return pd.DataFrame(
            {
                "candidate_value": [101.0, 102.0],
                "baseline_value": [100.0, 100.0],
                "paired_advantage": [1.0, 2.0],
                "candidate_won": [True, True],
            }
        )

    monkeypatch.setattr(module, "collect_policy_trajectories", fake_collect)
    monkeypatch.setattr(module, "train_value_model", fake_train)
    monkeypatch.setattr(module, "benchmark_controlled_policy", fake_benchmark)
    config = PolicyIterationConfig(
        iterations=1,
        samples_per_iteration=10,
        season_simulations=1,
        benchmark_drafts=2,
        benchmark_seasons=1,
        max_epochs=1,
        patience=1,
        batch_size=2,
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        resume=True,
    )

    first = run_fitted_policy_iteration(config, verbose=False)
    second = run_fitted_policy_iteration(config, verbose=False)

    assert calls == {"collect": 1, "train": 1, "benchmark": 1}
    assert first.to_dict("records") == second.to_dict("records")
    manifest = pd.read_json(config.data_dir / "manifest.json", typ="series")
    assert manifest["recommended_iteration"] == 0
    assert (config.model_dir / "best_value.pt").exists()
    assert asdict(config)["iterations"] == 1
