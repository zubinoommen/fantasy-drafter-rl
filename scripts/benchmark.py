#!/usr/bin/env python3
"""Benchmark end-to-end nested state evaluation."""

from __future__ import annotations

import argparse
import time

import numpy as np

from fantasy_draft.evaluation import evaluate_state
from fantasy_draft.generation import generate_states
from fantasy_draft.policies import RosterAwareSoftmaxPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--draft-rollouts", type=int, default=10)
    parser.add_argument("--season-simulations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.perf_counter()
    states = generate_states(args.states, args.seed)
    generation_seconds = time.perf_counter() - started
    seeds = np.random.SeedSequence(args.seed + 1).spawn(len(states))
    values: list[float] = []
    evaluation_started = time.perf_counter()
    policy = RosterAwareSoftmaxPolicy()
    for generated, child in zip(states, seeds, strict=True):
        result = evaluate_state(
            generated.state,
            policy,
            args.draft_rollouts,
            args.season_simulations,
            int(child.generate_state(1, dtype=np.uint64)[0]),
        )
        values.append(result.mean_value)
    evaluation_seconds = time.perf_counter() - evaluation_started
    total_samples = args.states * args.draft_rollouts * args.season_simulations
    print(f"state generation: {generation_seconds:.3f}s")
    print(f"evaluation: {evaluation_seconds:.3f}s")
    print(f"season samples: {total_samples}")
    print(f"season samples/second: {total_samples / evaluation_seconds:.1f}")
    print(f"mean estimated value: {np.mean(values):.3f}")


if __name__ == "__main__":
    main()
