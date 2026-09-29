"""Weekly season simulation, lineup optimization, and utility functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from fantasy_draft.config import RosterConfig
from fantasy_draft.models import POSITIONS, Player, Position
from fantasy_draft.waivers import (
    PoolReplacementLevelModel,
    ReplacementModel,
    ReplacementRequest,
)

UtilityFunction = Callable[[np.ndarray], float]


@dataclass(frozen=True, slots=True)
class ScoredPlayer:
    player: Player
    score: float
    is_replacement: bool = False


@dataclass(frozen=True, slots=True)
class WeeklyLineup:
    selected_ids: tuple[int, ...]
    replacement_ids: tuple[int, ...]
    score: float


@dataclass(frozen=True, slots=True)
class SeasonResult:
    weekly_scores: np.ndarray
    lineups: tuple[WeeklyLineup, ...]


@dataclass(frozen=True, slots=True)
class _PreparedSeason:
    universe: tuple[Player, ...]
    injury_probabilities: np.ndarray
    index: dict[int, int]
    roster: tuple[Player, ...]
    roster_indices: tuple[int, ...]
    free_agents: tuple[Player, ...]
    free_agent_indices: tuple[int, ...]


def optimal_lineup(
    candidates: Sequence[ScoredPlayer],
    roster_config: RosterConfig,
) -> WeeklyLineup:
    """Choose the maximum-score legal lineup; empty required slots score zero."""

    chosen: list[ScoredPlayer] = []
    used: set[int] = set()
    for position in POSITIONS:
        eligible = sorted(
            (item for item in candidates if item.player.position == position),
            key=lambda item: (-item.score, item.player.player_id),
        )
        count = roster_config.required[position]
        for item in eligible[:count]:
            chosen.append(item)
            used.add(item.player.player_id)

    flex_candidates = sorted(
        (
            item
            for item in candidates
            if item.player.position in roster_config.flex_eligible
            and item.player.player_id not in used
        ),
        key=lambda item: (-item.score, item.player.player_id),
    )
    for item in flex_candidates[: roster_config.flex_slots]:
        chosen.append(item)
        used.add(item.player.player_id)

    return WeeklyLineup(
        selected_ids=tuple(item.player.player_id for item in chosen),
        replacement_ids=tuple(
            item.player.player_id for item in chosen if item.is_replacement
        ),
        score=float(sum(item.score for item in chosen)),
    )


def _replacement_request(
    healthy_roster: Sequence[Player],
    roster_config: RosterConfig,
) -> ReplacementRequest:
    counts = {position: 0 for position in POSITIONS}
    for player in healthy_roster:
        counts[player.position] += 1
    missing: list[Position] = []
    for position in POSITIONS:
        missing.extend(
            [position] * max(0, roster_config.required[position] - counts[position])
        )
    eligible_surplus = sum(
        max(0, counts[position] - roster_config.required[position])
        for position in roster_config.flex_eligible
    )
    flex_deficit = max(0, roster_config.flex_slots - eligible_surplus)
    return ReplacementRequest(tuple(missing), flex_deficit, roster_config.flex_eligible)


@dataclass(slots=True)
class SeasonSimulator:
    """Simulate independent weekly player outcomes and optimal lineups."""

    roster_config: RosterConfig
    n_weeks: int = 14
    replacement_model: ReplacementModel | None = None

    def __post_init__(self) -> None:
        if self.n_weeks < 1:
            raise ValueError("n_weeks must be positive")
        if self.replacement_model is None:
            self.replacement_model = PoolReplacementLevelModel()

    def simulate(
        self,
        roster: Sequence[Player],
        free_agents: Sequence[Player],
        seed: int,
        player_universe: Iterable[Player] | None = None,
    ) -> SeasonResult:
        """Simulate one season.

        Supplying the same full ``player_universe`` and seed aligns player/week
        shocks across candidate rosters (common random numbers).
        """

        prepared = self._prepare(roster, free_agents, player_universe)
        return self._simulate_prepared(prepared, seed)

    def simulate_many(
        self,
        roster: Sequence[Player],
        free_agents: Sequence[Player],
        seeds: Sequence[int],
        player_universe: Iterable[Player] | None = None,
    ) -> tuple[SeasonResult, ...]:
        """Simulate many independent seasons while reusing static roster indexing."""

        prepared = self._prepare(roster, free_agents, player_universe)
        return tuple(self._simulate_prepared(prepared, seed) for seed in seeds)

    def _prepare(
        self,
        roster: Sequence[Player],
        free_agents: Sequence[Player],
        player_universe: Iterable[Player] | None,
    ) -> _PreparedSeason:
        universe = tuple(
            sorted(
                player_universe or (*roster, *free_agents),
                key=lambda player: player.player_id,
            )
        )
        if len({player.player_id for player in universe}) != len(universe):
            raise ValueError("player_universe must contain unique IDs")
        index = {player.player_id: offset for offset, player in enumerate(universe)}
        roster_tuple = tuple(roster)
        free_agent_tuple = tuple(free_agents)
        return _PreparedSeason(
            universe=universe,
            injury_probabilities=np.asarray(
                [player.injury_probability for player in universe], dtype=float
            ),
            index=index,
            roster=roster_tuple,
            roster_indices=tuple(index[player.player_id] for player in roster_tuple),
            free_agents=free_agent_tuple,
            free_agent_indices=tuple(
                index[player.player_id] for player in free_agent_tuple
            ),
        )

    def _simulate_prepared(
        self,
        prepared: _PreparedSeason,
        seed: int,
    ) -> SeasonResult:
        rng = np.random.default_rng(seed)
        uniforms = rng.random((self.n_weeks, len(prepared.universe)))
        normals = rng.standard_normal((self.n_weeks, len(prepared.universe)))
        lineups: list[WeeklyLineup] = []

        for week in range(self.n_weeks):
            available = uniforms[week] >= prepared.injury_probabilities
            healthy_roster = [
                player
                for player, player_index in zip(
                    prepared.roster, prepared.roster_indices, strict=True
                )
                if available[player_index]
            ]
            available_free_agent_ids = frozenset(
                player.player_id
                for player, player_index in zip(
                    prepared.free_agents,
                    prepared.free_agent_indices,
                    strict=True,
                )
                if available[player_index]
            )
            request = _replacement_request(healthy_roster, self.roster_config)
            assert self.replacement_model is not None
            replacements = self.replacement_model.select_replacements(
                request,
                prepared.free_agents,
                available_free_agent_ids,
                rng,
            )
            replacement_ids = {player.player_id for player in replacements}
            candidates = (*healthy_roster, *replacements)
            # Set the lineup from information available before games begin.
            projected = [
                ScoredPlayer(
                    player,
                    player.mu,
                    player.player_id in replacement_ids,
                )
                for player in candidates
            ]
            selected = optimal_lineup(projected, self.roster_config)
            candidate_map = {player.player_id: player for player in candidates}
            realized_score = 0.0
            for player_id in selected.selected_ids:
                player = candidate_map[player_id]
                player_index = prepared.index[player.player_id]
                realized_score += max(
                    0.0,
                    player.mu + player.sigma * normals[week, player_index],
                )
            lineups.append(
                WeeklyLineup(
                    selected_ids=selected.selected_ids,
                    replacement_ids=selected.replacement_ids,
                    score=float(realized_score),
                )
            )

        return SeasonResult(
            weekly_scores=np.asarray([lineup.score for lineup in lineups], dtype=float),
            lineups=tuple(lineups),
        )


def mean_weekly_score(scores: np.ndarray) -> float:
    return float(np.mean(scores))


def median_weekly_score(scores: np.ndarray) -> float:
    return float(np.median(scores))


def lower_quantile_utility(quantile: float) -> UtilityFunction:
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    return lambda scores: float(np.quantile(scores, quantile))


def mean_minus_risk_penalty(penalty: float) -> UtilityFunction:
    if penalty < 0:
        raise ValueError("penalty must be non-negative")
    return lambda scores: float(np.mean(scores) - penalty * np.std(scores))


def probability_above(threshold: float) -> UtilityFunction:
    return lambda scores: float(np.mean(scores > threshold))
