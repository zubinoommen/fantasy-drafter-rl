"""Interchangeable baseline draft policies."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from typing import Protocol

import numpy as np

from fantasy_draft.draft import DraftEnvironment, DraftState
from fantasy_draft.models import POSITIONS, Player, Position


class DraftPolicy(Protocol):
    """Policy interface used for controlled and opponent selections."""

    @property
    def name(self) -> str: ...

    @property
    def is_deterministic(self) -> bool: ...

    def select_player(
        self,
        state: DraftState,
        environment: DraftEnvironment,
        rng: np.random.Generator,
    ) -> int: ...


def required_starter_actions(
    state: DraftState,
    legal: tuple[int, ...],
) -> tuple[int, ...] | None:
    """O(1) lookup to ensure required starters are drafted by the end of the draft.

    When remaining picks for the current team equals the number of unfilled required
    starter slots, restricts selectable actions strictly to those unfilled positions.
    Returns None if no restriction is currently binding.
    """
    team = state.current_team
    if team is None or not legal:
        return None

    roster_ids = state.rosters[team]
    roster_size = state.config.roster.roster_size
    picks_left = roster_size - len(roster_ids)
    if picks_left <= 0:
        return None

    counts = state.roster_position_counts(team)
    required = state.config.roster.required
    needed_positions = {
        pos for pos in POSITIONS
        if counts[pos] < required[pos]
    }
    needed_slots = sum(max(0, required[pos] - counts[pos]) for pos in POSITIONS)

    if picks_left <= needed_slots and needed_positions:
        lookup = state.player_map
        filtered = tuple(
            pid for pid in legal
            if lookup[pid].position in needed_positions
        )
        if filtered:
            return filtered
    return None


@lru_cache(maxsize=256)
def _cached_replacement_levels(
    players: tuple[Player, ...],
    n_teams: int,
    required: tuple[int, ...],
    flex_slots: int,
    flex_eligible: tuple[Position, ...],
) -> tuple[float, ...]:
    flex_share = (
        flex_slots / len(flex_eligible)
        if flex_eligible
        else 0.0
    )
    levels: list[float] = []
    for position in POSITIONS:
        expected_per_team = required[POSITIONS.index(position)]
        if position in flex_eligible:
            expected_per_team += flex_share
        expected_drafted = ceil(n_teams * expected_per_team)
        position_means = sorted(
            (
                player.mu
                for player in players
                if player.position == position
            ),
            reverse=True,
        )
        replacement_index = min(expected_drafted, len(position_means) - 1)
        levels.append(position_means[replacement_index])
    return tuple(levels)


def projected_replacement_levels(state: DraftState) -> dict[Position, float]:
    """Estimate each position's first-undrafted ``mu`` from league demand.

    Required starters consume their actual positions. FLEX demand is divided
    evenly among configured eligible positions. The estimate deliberately
    remains a soft policy feature and never affects draft legality.
    """

    roster = state.config.roster
    levels = _cached_replacement_levels(
        state.players,
        state.config.n_teams,
        tuple(roster.required[position] for position in POSITIONS),
        roster.flex_slots,
        roster.flex_eligible,
    )
    return dict(zip(POSITIONS, levels, strict=True))


@dataclass(frozen=True, slots=True)
class RosterAwareSoftmaxPolicy:
    """Configuration-driven draft policy balancing mean, VOR, and roster needs."""

    temperature: float = 3.0
    starter_need_bonus: float = 5.0
    flex_need_bonus: float = 2.0
    value_above_replacement_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.value_above_replacement_weight < 0:
            raise ValueError("value_above_replacement_weight must be non-negative")

    @property
    def name(self) -> str:
        return (
            f"roster_softmax_t{self.temperature:g}"
            f"_s{self.starter_need_bonus:g}_f{self.flex_need_bonus:g}"
            f"_vor{self.value_above_replacement_weight:g}"
        )

    @property
    def is_deterministic(self) -> bool:
        return False

    def select_player(
        self,
        state: DraftState,
        environment: DraftEnvironment,
        rng: np.random.Generator,
    ) -> int:
        legal = environment.legal_actions(state)
        if not legal:
            raise ValueError("policy called with no legal actions")
        restricted = required_starter_actions(state, legal)
        pool = restricted if restricted is not None else legal
        lookup = state.player_map
        team = state.current_team
        assert team is not None
        counts = state.roster_position_counts(team)
        roster_config = state.config.roster
        eligible_surplus = sum(
            max(0, counts[position] - roster_config.required[position])
            for position in roster_config.flex_eligible
        )
        flex_needed = eligible_surplus < roster_config.flex_slots
        replacement_levels = projected_replacement_levels(state)
        scores = []
        for player_id in pool:
            player = lookup[player_id]
            value_above_replacement = player.mu - replacement_levels[player.position]
            score = (
                player.mu
                + self.value_above_replacement_weight * value_above_replacement
            )
            if counts[player.position] < roster_config.required[player.position]:
                score += self.starter_need_bonus
            elif flex_needed and player.position in roster_config.flex_eligible:
                score += self.flex_need_bonus
            scores.append(score)
        logits = np.asarray(scores) / self.temperature
        weights = np.exp(logits - logits.max())
        return int(rng.choice(pool, p=weights / weights.sum()))


def continue_draft(
    state: DraftState,
    policy: DraftPolicy,
    rng: np.random.Generator,
) -> DraftState:
    """Complete a draft from an arbitrary reachable state."""

    environment = DraftEnvironment(state.config)
    current = state
    while not current.is_terminal:
        action = policy.select_player(current, environment, rng)
        current = environment.step(current, action, validate=False)
    return current
