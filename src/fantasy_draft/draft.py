"""Deterministic snake-draft state and transition model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from fantasy_draft.config import LeagueConfig
from fantasy_draft.models import POSITIONS, Player, Position


@dataclass(frozen=True, slots=True)
class DraftState:
    """Complete reconstructable draft state.

    Random opponent decisions live in policies; applying a selected player ID is
    deterministic.
    """

    config: LeagueConfig
    players: tuple[Player, ...]
    available_ids: frozenset[int]
    rosters: tuple[tuple[int, ...], ...]
    pick_index: int = 0

    def __post_init__(self) -> None:
        ids = {player.player_id for player in self.players}
        if len(ids) != len(self.players):
            raise ValueError("player IDs must be unique")
        if len(self.rosters) != self.config.n_teams:
            raise ValueError("one roster is required per fantasy team")
        if not self.available_ids.issubset(ids):
            raise ValueError("available_ids contains an unknown player")
        drafted = {player_id for roster in self.rosters for player_id in roster}
        if drafted & self.available_ids:
            raise ValueError("a drafted player cannot remain available")
        if len(drafted) != sum(map(len, self.rosters)):
            raise ValueError("a player cannot appear on multiple rosters")
        if drafted | self.available_ids != ids:
            raise ValueError("every player must be drafted or available")
        if sum(map(len, self.rosters)) != self.pick_index:
            raise ValueError("pick_index must equal the number of drafted players")
        if any(len(roster) > self.config.roster.roster_size for roster in self.rosters):
            raise ValueError("a roster exceeds physical capacity")
        if not 0 <= self.pick_index <= self.config.total_picks:
            raise ValueError("pick_index is out of range")

    @property
    def is_terminal(self) -> bool:
        return self.pick_index == self.config.total_picks

    @property
    def round_index(self) -> int:
        return min(self.pick_index // self.config.n_teams, self.config.n_rounds - 1)

    @property
    def round_number(self) -> int:
        return self.round_index + 1

    @property
    def current_team(self) -> int | None:
        if self.is_terminal:
            return None
        within_round = self.pick_index % self.config.n_teams
        if self.round_index % 2 == 0:
            return within_round
        return self.config.n_teams - 1 - within_round

    @property
    def player_map(self) -> dict[int, Player]:
        return {player.player_id: player for player in self.players}

    def roster_players(self, team: int) -> tuple[Player, ...]:
        lookup = self.player_map
        return tuple(lookup[player_id] for player_id in self.rosters[team])

    def roster_position_counts(self, team: int) -> dict[Position, int]:
        lookup = self.player_map
        counts = {pos: 0 for pos in POSITIONS}
        for player_id in self.rosters[team]:
            counts[lookup[player_id].position] += 1
        return counts

    def available_players(self) -> tuple[Player, ...]:
        return tuple(player for player in self.players if player.player_id in self.available_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "players": [player.to_dict() for player in self.players],
            "available_ids": sorted(self.available_ids),
            "rosters": [list(roster) for roster in self.rosters],
            "pick_index": self.pick_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DraftState:
        return cls(
            config=LeagueConfig.from_dict(data["config"]),
            players=tuple(Player.from_dict(item) for item in data["players"]),
            available_ids=frozenset(int(value) for value in data["available_ids"]),
            rosters=tuple(tuple(int(value) for value in roster) for roster in data["rosters"]),
            pick_index=int(data["pick_index"]),
        )

class DraftEnvironment:
    """Snake draft enforcing availability and configured physical capacity only."""

    def __init__(self, config: LeagueConfig | None = None) -> None:
        self.config = config or LeagueConfig()

    def initial_state(self, players: Iterable[Player]) -> DraftState:
        ordered = tuple(sorted(players, key=lambda player: player.player_id))
        if len(ordered) < self.config.total_picks:
            raise ValueError("player pool is smaller than the draft")
        return DraftState(
            config=self.config,
            players=ordered,
            available_ids=frozenset(player.player_id for player in ordered),
            rosters=tuple(() for _ in range(self.config.n_teams)),
        )

    def team_at_pick(self, pick_index: int) -> int:
        if not 0 <= pick_index < self.config.total_picks:
            raise IndexError("pick index outside draft")
        round_index, within_round = divmod(pick_index, self.config.n_teams)
        return within_round if round_index % 2 == 0 else self.config.n_teams - 1 - within_round

    def legal_actions(self, state: DraftState) -> tuple[int, ...]:
        if state.is_terminal:
            return ()
        team = state.current_team
        assert team is not None
        roster_ids = state.rosters[team]
        if len(roster_ids) >= self.config.roster.roster_size:
            return ()
        limits = self.config.roster.position_limits
        if limits is None:
            return tuple(sorted(state.available_ids))
        lookup = state.player_map
        counts = {position: 0 for position in limits}
        for rostered_id in roster_ids:
            counts[lookup[rostered_id].position] += 1
        return tuple(
            player_id
            for player_id in sorted(state.available_ids)
            if counts[lookup[player_id].position] < limits[lookup[player_id].position]
        )

    def step(
        self,
        state: DraftState,
        player_id: int,
        *,
        validate: bool = True,
    ) -> DraftState:
        """Apply one legal pick; no randomness is consumed."""

        if validate and player_id not in self.legal_actions(state):
            raise ValueError(f"illegal draft action: {player_id}")
        if not validate and (
            state.is_terminal
            or player_id not in state.available_ids
            or state.current_team is None
            or len(state.rosters[state.current_team]) >= self.config.roster.roster_size
        ):
            raise ValueError(f"physically illegal draft action: {player_id}")
        team = state.current_team
        assert team is not None
        rosters = list(state.rosters)
        rosters[team] = (*rosters[team], player_id)
        return DraftState(
            config=state.config,
            players=state.players,
            available_ids=state.available_ids - {player_id},
            rosters=tuple(rosters),
            pick_index=state.pick_index + 1,
        )

    def picks_until_team(self, state: DraftState, team: int) -> int:
        """Number of intervening picks before ``team`` next acts."""

        if state.is_terminal:
            return 0
        for future_pick in range(state.pick_index, self.config.total_picks):
            if self.team_at_pick(future_pick) == team:
                return future_pick - state.pick_index
        return 0
