#!/usr/bin/env python3
"""Generate post-pick draft states and Monte Carlo value targets in Parquet."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from fantasy_draft.draft import DraftEnvironment, DraftState
from fantasy_draft.evaluation import evaluate_state
from fantasy_draft.generation import (
    LeagueConfigDistribution,
    _sample_player_pool_config,
    draft_phase,
    league_config_id,
)
from fantasy_draft.models import Position
from fantasy_draft.players import generate_players
from fantasy_draft.policies import RosterAwareSoftmaxPolicy
from fantasy_draft.state import ENCODER_VERSION, StateEncoder


@dataclass(frozen=True, slots=True)
class PostPickSample:
    """One exact transition after the controlled team selects a player."""

    sample_id: str
    decision_id: str
    state: DraftState
    action_player_id: int
    action_position: Position
    action_mu: float
    round_number: int
    phase: str
    league_config_id: str
    player_pool_seed: int
    draft_history_seed: int
    generation_policy_name: str


def _round_weights(n_rounds: int) -> np.ndarray:
    """Weight rounds 1-3 most heavily while retaining full-draft coverage."""

    rounds = np.arange(n_rounds)
    weights = np.ones(n_rounds, dtype=float)
    weights[rounds < min(3, n_rounds)] = 4.0
    weights[(rounds >= 3) & (rounds < max(3, 2 * n_rounds // 3))] = 2.0
    return weights / weights.sum()


def generate_post_pick_samples(
    n_samples: int,
    seed: int,
    *,
    samples_per_draft: int = 4,
) -> tuple[PostPickSample, ...]:
    """Generate diverse reachable states immediately after controlled-team picks."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if samples_per_draft < 1:
        raise ValueError("samples_per_draft must be positive")

    root_rng = np.random.default_rng(seed)
    configs = list(LeagueConfigDistribution().configurations())
    root_rng.shuffle(configs)
    samples: list[PostPickSample] = []
    draft_number = 0

    while len(samples) < n_samples:
        template = configs[draft_number % len(configs)]
        controlled_team = int(root_rng.integers(template.n_teams))
        league = replace(template, controlled_team=controlled_team)
        player_seed = int(root_rng.integers(0, np.iinfo(np.int64).max))
        draft_seed = int(root_rng.integers(0, np.iinfo(np.int64).max))

        pool = _sample_player_pool_config(
            league,
            np.random.default_rng(player_seed),
        )
        players = generate_players(player_seed, pool)
        environment = DraftEnvironment(league)
        state = environment.initial_state(players)
        draft_rng = np.random.default_rng(draft_seed)
        generation_policy = RosterAwareSoftmaxPolicy(
            temperature=float(draft_rng.uniform(2.0, 5.0)),
            starter_need_bonus=float(draft_rng.uniform(3.0, 8.0)),
            flex_need_bonus=float(draft_rng.uniform(1.0, 4.0)),
            value_above_replacement_weight=float(draft_rng.uniform(0.7, 1.3)),
        )

        target_count = min(samples_per_draft, league.n_rounds)
        target_rounds = set(
            int(value)
            for value in draft_rng.choice(
                league.n_rounds,
                size=target_count,
                replace=False,
                p=_round_weights(league.n_rounds),
            )
        )

        while not state.is_terminal:
            is_target = (
                state.current_team == controlled_team
                and state.round_index in target_rounds
                and len(samples) < n_samples
            )
            pre_pick_index = state.pick_index
            pre_round = state.round_number
            pre_phase = draft_phase(state)
            pre_roster_size = len(state.rosters[controlled_team])
            action = generation_policy.select_player(state, environment, draft_rng)
            selected = state.player_map[action]
            state = environment.step(state, action, validate=False)

            if is_target:
                if (
                    state.pick_index != pre_pick_index + 1
                    or len(state.rosters[controlled_team]) != pre_roster_size + 1
                    or state.rosters[controlled_team][-1] != action
                ):
                    raise AssertionError("captured state is not the exact post-pick transition")
                decision_id = f"draft-{draft_number:06d}-pick-{pre_pick_index:03d}"
                samples.append(
                    PostPickSample(
                        sample_id=f"{decision_id}-action-{action:04d}",
                        decision_id=decision_id,
                        state=state,
                        action_player_id=action,
                        action_position=selected.position,
                        action_mu=selected.mu,
                        round_number=pre_round,
                        phase=pre_phase,
                        league_config_id=league_config_id(league),
                        player_pool_seed=player_seed,
                        draft_history_seed=draft_seed,
                        generation_policy_name=generation_policy.name,
                    )
                )
        draft_number += 1

    return tuple(samples)


def _evaluate_sample(
    payload: tuple[PostPickSample, int, int, int],
) -> dict[str, Any]:
    sample, draft_rollouts, season_simulations, evaluation_seed = payload
    # One canonical policy controls every remaining pick by every team.
    evaluation_policy = RosterAwareSoftmaxPolicy()
    result = evaluate_state(
        sample.state,
        evaluation_policy,
        draft_rollouts,
        season_simulations,
        evaluation_seed,
    )
    vector = StateEncoder().encode_state(sample.state)
    roster = sample.state.config.roster
    return {
        "sample_id": sample.sample_id,
        "decision_id": sample.decision_id,
        "state_vector": vector,
        "reward": result.mean_value,
        "reward_standard_error": result.standard_error,
        "action_player_id": sample.action_player_id,
        "action_position": sample.action_position.value,
        "action_mu": sample.action_mu,
        "round_number": sample.round_number,
        "phase": sample.phase,
        "pick_index_after_action": sample.state.pick_index,
        "league_config_id": sample.league_config_id,
        "num_teams": sample.state.config.n_teams,
        "controlled_team": sample.state.config.controlled_team,
        "roster_size": roster.roster_size,
        "qb_starters": roster.required[Position.QB],
        "rb_starters": roster.required[Position.RB],
        "wr_starters": roster.required[Position.WR],
        "te_starters": roster.required[Position.TE],
        "flex_slots": roster.flex_slots,
        "bench_slots": roster.bench_slots,
        "player_pool_seed": sample.player_pool_seed,
        "draft_history_seed": sample.draft_history_seed,
        "evaluation_seed": evaluation_seed,
        "generation_policy": sample.generation_policy_name,
        "evaluation_policy": result.policy_name,
        "draft_rollouts": draft_rollouts,
        "season_simulations": season_simulations,
        "encoder_version": ENCODER_VERSION,
    }


def save_parquet(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write typed columns and a fixed-size float32 state vector."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            'Parquet output requires pyarrow; install with pip install -e ".[data]"'
        ) from exc

    if not rows:
        raise ValueError("cannot save an empty dataset")

    vectors = np.stack([row.pop("state_vector") for row in rows]).astype(
        np.float32,
        copy=False,
    )
    columns = {key: [row[key] for row in rows] for key in rows[0]}
    columns["evaluation_seed"] = pa.array(
        columns["evaluation_seed"],
        type=pa.uint64(),
    )
    columns["state_vector"] = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.reshape(-1), type=pa.float32()),
        vectors.shape[1],
    )
    table = pa.table(columns)
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"dataset_type": b"post_pick_state_value",
            b"reward_semantics": b"monte_carlo_mean_weekly_score_V_pi_of_post_action_state",
            b"state_semantics": b"exact_DraftEnvironment_step_state_after_controlled_pick",
            b"feature_names": json.dumps(StateEncoder().feature_names()).encode("utf-8"),
        }
    )
    table = table.replace_schema_metadata(metadata)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output, compression="zstd")


def run_pipeline(
    samples: int = 10_000,
    output: str | Path = "data/post_pick_values.parquet",
    seed: int = 42,
    samples_per_draft: int = 4,
    draft_rollouts: int = 5,
    season_simulations: int = 10,
    workers: int | None = None,
) -> Path:
    """Generate post-pick states, compute Monte Carlo rewards in parallel, and save Parquet.

    Configurable directly in Python so you can adjust parameters and hit Run.
    """
    total_start = time.perf_counter()
    max_workers = workers if workers is not None else min(8, max(1, (os.cpu_count() or 2) - 1))
    output_path = Path(output)

    print(f"=== Generating {samples:,} Post-Pick Training Samples ===")
    print(f"Output path: {output_path}")
    print(f"Rollouts per state: {draft_rollouts} drafts x {season_simulations} seasons")
    print(f"Parallel worker processes: {max_workers}")

    gen_start = time.perf_counter()
    post_pick_samples = generate_post_pick_samples(
        samples,
        seed,
        samples_per_draft=samples_per_draft,
    )
    gen_time = time.perf_counter() - gen_start
    print(f"State generation completed in {gen_time:.2f}s ({len(post_pick_samples):,} states)")

    seed_sequences = np.random.SeedSequence(seed + 1).spawn(len(post_pick_samples))
    payloads = [
        (
            sample,
            draft_rollouts,
            season_simulations,
            int(sequence.generate_state(1, dtype=np.uint64)[0]),
        )
        for sample, sequence in zip(post_pick_samples, seed_sequences, strict=True)
    ]

    eval_start = time.perf_counter()
    if max_workers == 1:
        rows = [_evaluate_sample(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            rows = list(executor.map(_evaluate_sample, payloads, chunksize=16))
    eval_time = time.perf_counter() - eval_start
    print(f"Monte Carlo evaluation completed in {eval_time:.2f}s ({len(rows)/eval_time:.1f} states/sec)")

    save_start = time.perf_counter()
    save_parquet(output_path, rows)
    save_time = time.perf_counter() - save_start
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    total_time = time.perf_counter() - total_start
    print(f"Parquet saved in {save_time:.2f}s ({file_size_mb:.2f} MB)")
    print(f"=== Total Elapsed Time: {total_time:.2f}s ({total_time / 60:.2f} minutes) ===")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create post-pick state/value training data in Parquet.",
    )
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--output", default="data/post_pick_values.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-draft", type=int, default=4)
    parser.add_argument("--draft-rollouts", type=int, default=5)
    parser.add_argument("--season-simulations", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    run_pipeline(
        samples=args.samples,
        output=args.output,
        seed=args.seed,
        samples_per_draft=args.samples_per_draft,
        draft_rollouts=args.draft_rollouts,
        season_simulations=args.season_simulations,
        workers=args.workers,
    )


if __name__ == "__main__":
    import sys

    # If CLI arguments were passed (e.g. `python script.py --samples 500`), use main().
    # If run directly with no arguments (e.g. hitting 'Run Python File' in IDE),
    # it executes the default function call below with configurable parameters:
    if len(sys.argv) > 1:
        main()
    else:
        run_pipeline(
            samples=50_000,
            output="data/post_pick_values.parquet",
            seed=42,
            samples_per_draft=4,
            draft_rollouts=3,
            season_simulations=5,
            workers=None,  # Set to an integer (e.g. 7) or None for auto-detection
        )
