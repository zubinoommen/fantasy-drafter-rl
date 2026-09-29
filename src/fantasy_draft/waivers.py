"""Cheap, injectable waiver and replacement-level models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from fantasy_draft.models import Player, Position


@dataclass(frozen=True, slots=True)
class ReplacementRequest:
    """Lineup holes after accounting for healthy rostered players."""

    required_positions: tuple[Position, ...]
    flex_slots: int
    flex_eligible: tuple[Position, ...]


class ReplacementModel(Protocol):
    """Weekly replacement interface; future implementations may retain claims."""

    @property
    def name(self) -> str: ...

    def select_replacements(
        self,
        request: ReplacementRequest,
        free_agents: Sequence[Player],
        available_ids: frozenset[int],
        rng: np.random.Generator,
    ) -> tuple[Player, ...]: ...


@dataclass(frozen=True, slots=True)
class NoReplacementModel:
    @property
    def name(self) -> str:
        return "none"

    def select_replacements(
        self,
        request: ReplacementRequest,
        free_agents: Sequence[Player],
        available_ids: frozenset[int],
        rng: np.random.Generator,
    ) -> tuple[Player, ...]:
        del request, free_agents, available_ids, rng
        return ()


def _weighted_choice(
    candidates: list[Player],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> Player | None:
    if not candidates:
        return None
    probabilities = weights / weights.sum()
    return candidates[int(rng.choice(len(candidates), p=probabilities))]


@dataclass(frozen=True, slots=True)
class PoolReplacementLevelModel:
    """Default weekly replacement approximation derived from undrafted players.

    Candidate weight decays by expected-production rank, so good waiver players
    are more accessible without granting the best option deterministically.
    Claims reset each week; a player is used at most once within a week.
    """

    rank_scale: float = 4.0

    def __post_init__(self) -> None:
        if self.rank_scale <= 0:
            raise ValueError("rank_scale must be positive")

    @property
    def name(self) -> str:
        return f"pool_replacement_rank{self.rank_scale:g}"

    def _choose(
        self,
        positions: tuple[Position, ...],
        free_agents: Sequence[Player],
        available_ids: frozenset[int],
        used: set[int],
        rng: np.random.Generator,
    ) -> Player | None:
        candidates = sorted(
            (
                player
                for player in free_agents
                if player.position in positions
                and player.player_id in available_ids
                and player.player_id not in used
            ),
            key=lambda player: (-player.mu, player.player_id),
        )
        ranks = np.arange(len(candidates), dtype=float)
        return _weighted_choice(candidates, np.exp(-ranks / self.rank_scale), rng)

    def select_replacements(
        self,
        request: ReplacementRequest,
        free_agents: Sequence[Player],
        available_ids: frozenset[int],
        rng: np.random.Generator,
    ) -> tuple[Player, ...]:
        selected: list[Player] = []
        used: set[int] = set()
        for position in request.required_positions:
            player = self._choose((position,), free_agents, available_ids, used, rng)
            if player is not None:
                selected.append(player)
                used.add(player.player_id)
        for _ in range(request.flex_slots):
            player = self._choose(request.flex_eligible, free_agents, available_ids, used, rng)
            if player is not None:
                selected.append(player)
                used.add(player.player_id)
        return tuple(selected)


@dataclass(frozen=True, slots=True)
class StochasticWaiverPolicy:
    """Alternative explicit weekly-access policy with noisy market availability."""

    temperature: float = 2.5
    market_access_probability: float = 0.65

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= self.market_access_probability <= 1:
            raise ValueError("market_access_probability must be in [0, 1]")

    @property
    def name(self) -> str:
        return "stochastic_waiver"

    def _choose(
        self,
        positions: tuple[Position, ...],
        free_agents: Sequence[Player],
        available_ids: frozenset[int],
        used: set[int],
        rng: np.random.Generator,
    ) -> Player | None:
        candidates = [
            player
            for player in free_agents
            if player.position in positions
            and player.player_id in available_ids
            and player.player_id not in used
            and rng.random() < self.market_access_probability
        ]
        if not candidates:
            return None
        logits = np.asarray([player.mu / self.temperature for player in candidates])
        return _weighted_choice(candidates, np.exp(logits - logits.max()), rng)

    def select_replacements(
        self,
        request: ReplacementRequest,
        free_agents: Sequence[Player],
        available_ids: frozenset[int],
        rng: np.random.Generator,
    ) -> tuple[Player, ...]:
        selected: list[Player] = []
        used: set[int] = set()
        for position in request.required_positions:
            player = self._choose((position,), free_agents, available_ids, used, rng)
            if player is not None:
                selected.append(player)
                used.add(player.player_id)
        for _ in range(request.flex_slots):
            player = self._choose(request.flex_eligible, free_agents, available_ids, used, rng)
            if player is not None:
                selected.append(player)
                used.add(player.player_id)
        return tuple(selected)
