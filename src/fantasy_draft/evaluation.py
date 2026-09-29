"""Nested Monte Carlo evaluation of arbitrary draft states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fantasy_draft.draft import DraftState
from fantasy_draft.policies import DraftPolicy, continue_draft
from fantasy_draft.season import SeasonSimulator, UtilityFunction, mean_weekly_score
from fantasy_draft.waivers import PoolReplacementLevelModel, ReplacementModel


def _seed(child: np.random.SeedSequence) -> int:
    return int(child.generate_state(1, dtype=np.uint64)[0])


@dataclass(frozen=True, slots=True)
class EvaluationSeedPlan:
    """Reusable hierarchical seeds for matched-policy/state comparisons."""

    root_seed: int
    draft_seeds: tuple[int, ...]
    season_seeds: tuple[tuple[int, ...], ...]

    @classmethod
    def create(
        cls,
        seed: int,
        n_draft_rollouts: int,
        n_season_simulations: int,
    ) -> EvaluationSeedPlan:
        root = np.random.SeedSequence(seed)
        rollout_children = root.spawn(n_draft_rollouts)
        draft_seeds: list[int] = []
        season_seeds: list[tuple[int, ...]] = []
        for child in rollout_children:
            descendants = child.spawn(n_season_simulations + 1)
            draft_seeds.append(_seed(descendants[0]))
            season_seeds.append(tuple(_seed(item) for item in descendants[1:]))
        return cls(seed, tuple(draft_seeds), tuple(season_seeds))


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    mean_value: float
    std: float
    standard_error: float
    n_draft_rollouts: int
    n_season_simulations: int
    per_rollout_means: tuple[float, ...]
    utilities: tuple[tuple[float, ...], ...]
    terminal_rosters: tuple[tuple[int, ...], ...]
    policy_name: str
    seed_plan: EvaluationSeedPlan


def evaluate_state(
    state: DraftState,
    continuation_policy: DraftPolicy,
    n_draft_rollouts: int,
    n_season_simulations: int,
    seed: int,
    *,
    utility_function: UtilityFunction = mean_weekly_score,
    replacement_model: ReplacementModel | None = None,
    seed_plan: EvaluationSeedPlan | None = None,
) -> EvaluationResult:
    """Estimate V^pi(s) with M draft continuations and K seasons each.

    The reported sample standard deviation is across the M per-draft means;
    standard error is that quantity divided by sqrt(M).
    """

    if n_draft_rollouts < 1 or n_season_simulations < 1:
        raise ValueError("Monte Carlo sample counts must be positive")
    plan = seed_plan or EvaluationSeedPlan.create(
        seed, n_draft_rollouts, n_season_simulations
    )
    if (
        len(plan.draft_seeds) != n_draft_rollouts
        or any(len(row) != n_season_simulations for row in plan.season_seeds)
    ):
        raise ValueError("seed_plan dimensions do not match requested sample counts")

    all_utilities: list[tuple[float, ...]] = []
    terminal_rosters: list[tuple[int, ...]] = []
    controlled_team = state.config.controlled_team
    model = replacement_model or PoolReplacementLevelModel()
    simulator = SeasonSimulator(
        state.config.roster,
        state.config.season_weeks,
        model,
    )
    deterministic_terminal = None
    if getattr(continuation_policy, "is_deterministic", False):
        deterministic_terminal = continue_draft(
            state,
            continuation_policy,
            np.random.default_rng(plan.draft_seeds[0]),
        )

    for rollout_index in range(n_draft_rollouts):
        terminal = deterministic_terminal or continue_draft(
            state,
            continuation_policy,
            np.random.default_rng(plan.draft_seeds[rollout_index]),
        )
        roster = terminal.roster_players(controlled_team)
        free_agents = terminal.available_players()
        terminal_rosters.append(terminal.rosters[controlled_team])
        seasons = simulator.simulate_many(
            roster,
            free_agents,
            plan.season_seeds[rollout_index],
            player_universe=terminal.players,
        )
        all_utilities.append(
            tuple(utility_function(season.weekly_scores) for season in seasons)
        )

    per_rollout = np.asarray([np.mean(values) for values in all_utilities], dtype=float)
    std = float(np.std(per_rollout, ddof=1)) if n_draft_rollouts > 1 else 0.0
    return EvaluationResult(
        mean_value=float(np.mean(per_rollout)),
        std=std,
        standard_error=std / np.sqrt(n_draft_rollouts),
        n_draft_rollouts=n_draft_rollouts,
        n_season_simulations=n_season_simulations,
        per_rollout_means=tuple(float(value) for value in per_rollout),
        utilities=tuple(all_utilities),
        terminal_rosters=tuple(terminal_rosters),
        policy_name=continuation_policy.name,
        seed_plan=plan,
    )
