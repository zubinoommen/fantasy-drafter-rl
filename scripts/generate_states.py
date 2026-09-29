#!/usr/bin/env python3
"""Generate reconstructable partial draft states."""

from __future__ import annotations

import argparse

from fantasy_draft.dataset import save_generated_states
from fantasy_draft.generation import generate_states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="states.jsonl")
    args = parser.parse_args()
    states = generate_states(args.states, args.seed)
    save_generated_states(args.output, states)
    print(f"saved {len(states)} states to {args.output}")


if __name__ == "__main__":
    main()
