#!/usr/bin/env python3
"""Evaluate a saved state collection under a baseline continuation policy."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from fantasy_draft.dataset import (
    ValueObservation,
    load_generated_states,
    save_value_observations,
)
from fantasy_draft.evaluation import evaluate_state
from fantasy_draft.policies import (
    RosterAwareSoftmaxPolicy,
)
from fantasy_draft.state import StateEncoder


def _evaluate_one(payload: tuple[Any, Any, int, int, int]) -> ValueObservation:
    generated, policy, draft_rollouts, season_simulations, seed = payload
    result = evaluate_state(
        generated.state,
        policy,
        draft_rollouts,
        season_simulations,
        seed,
    )
    return ValueObservation.from_result(
        generated,
        StateEncoder().encode_state(generated.state),
        result,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="values.npz")
    parser.add_argument(
        "--policy",
        choices=("roster-softmax",),
        default="roster-softmax",
    )
    parser.add_argument("--draft-rollouts", type=int, default=10)
    parser.add_argument("--season-simulations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 2) - 1)),
    )
    args = parser.parse_args()
    policy = {
        "roster-softmax": RosterAwareSoftmaxPolicy(),
    }[args.policy]
    states = load_generated_states(args.input)
    state_seeds = np.random.SeedSequence(args.seed).spawn(len(states))
    payloads = [
        (
            generated,
            policy,
            args.draft_rollouts,
            args.season_simulations,
            int(state_seed.generate_state(1, dtype=np.uint64)[0]),
        )
        for generated, state_seed in zip(states, state_seeds, strict=True)
    ]
    if args.workers == 1:
        observations = [_evaluate_one(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            observations = list(executor.map(_evaluate_one, payloads, chunksize=1))
    save_value_observations(args.output, observations)
    print(f"saved {len(observations)} observations to {args.output}")


if __name__ == "__main__":
    main()
