"""Fixed-dimensional, information-safe draft-state encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fantasy_draft.draft import DraftEnvironment, DraftState
from fantasy_draft.models import POSITIONS, Player, Position


PLAYER_FEATURES = ("mu", "sigma", "injury_probability")
ENCODER_VERSION = 2


def _triple(player: Player) -> tuple[float, float, float]:
    return player.mu, player.sigma, player.injury_probability


def remaining_requirements(state: DraftState) -> np.ndarray:
    """QB/RB/WR/TE deficits, FLEX deficit, and unconstrained reserve openings."""

    config = state.config.roster
    roster = state.roster_players(state.config.controlled_team)
    counts = {position: 0 for position in POSITIONS}
    for player in roster:
        counts[player.position] += 1
    base = [max(0, config.required[position] - counts[position]) for position in POSITIONS]
    surplus = sum(
        max(0, counts[position] - config.required[position])
        for position in config.flex_eligible
    )
    flex = max(0, config.flex_slots - surplus)
    open_slots = config.roster_size - len(roster)
    reserve = max(0, open_slots - sum(base) - flex)
    return np.asarray([*base, flex, reserve], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class StateEncoder:
    """Canonical state encoding shared across heterogeneous league configurations."""

    board_size_per_position: int = 10
    max_teams: int = 16
    max_roster_size: int = 20
    max_starters_per_position: int = 4
    max_flex_slots: int = 4

    def __post_init__(self) -> None:
        if min(
            self.board_size_per_position,
            self.max_teams,
            self.max_roster_size,
            self.max_starters_per_position,
            self.max_flex_slots,
        ) < 1:
            raise ValueError("encoder maxima must be positive")

    def _validate_supported(self, state: DraftState) -> None:
        roster = state.config.roster
        if state.config.n_teams > self.max_teams:
            raise ValueError(f"encoder supports at most {self.max_teams} teams")
        if roster.roster_size > self.max_roster_size:
            raise ValueError(f"encoder supports at most {self.max_roster_size} roster slots")
        if any(value > self.max_starters_per_position for value in roster.required.values()):
            raise ValueError("starter requirement exceeds encoder maximum")
        if roster.flex_slots > self.max_flex_slots:
            raise ValueError("FLEX requirement exceeds encoder maximum")

    def feature_names(self, state: DraftState | None = None) -> tuple[str, ...]:
        if state is not None:
            self._validate_supported(state)
        names = [
            "config.encoder_version",
            "config.num_teams",
            "config.num_teams_normalized",
        ]
        names.extend(f"config.team_exists.{team}" for team in range(self.max_teams))
        names.append("config.controlled_team_index")
        names.extend(f"config.controlled_team.{team}" for team in range(self.max_teams))
        names.extend(
            [
                "draft.current_round",
                "draft.normalized_progress",
                "draft.picks_until_controlled",
                "draft.total_rounds",
                "draft.rounds_remaining",
                "config.roster_size",
                "draft.remaining_roster_slots",
                "config.bench_slots",
                "config.season_weeks",
                "config.points_per_reception",
                "config.passing_touchdown_points",
            ]
        )
        names.extend(f"config.starters.{position.value}" for position in POSITIONS)
        names.append("config.flex_slots")
        names.extend(f"config.flex_eligible.{position.value}" for position in POSITIONS)
        names.extend(f"config.position_limit.{position.value}" for position in POSITIONS)
        names.extend(f"config.position_limit_mask.{position.value}" for position in POSITIONS)
        for position in POSITIONS:
            for rank in range(self.board_size_per_position):
                names.extend(f"board.{position.value}.{rank}.{feature}" for feature in PLAYER_FEATURES)
        for position in POSITIONS:
            for rank in range(self.board_size_per_position):
                names.append(f"board.{position.value}.{rank}.occupied")
        for slot in range(self.max_roster_size):
            names.extend(f"roster.{slot}.{feature}" for feature in PLAYER_FEATURES)
            names.extend(f"roster.{slot}.position.{position.value}" for position in POSITIONS)
            names.append(f"roster.{slot}.occupied")
        names.extend(
            [
                "need.QB",
                "need.RB",
                "need.WR",
                "need.TE",
                "need.FLEX",
                "open_reserve",
            ]
        )
        return tuple(names)

    def encode_state(self, state: DraftState) -> np.ndarray:
        self._validate_supported(state)
        environment = DraftEnvironment(state.config)
        rounds_remaining = 0 if state.is_terminal else state.config.n_rounds - state.round_index
        controlled_roster = state.roster_players(state.config.controlled_team)
        progress = state.pick_index / state.config.total_picks
        limits = state.config.roster.position_limits
        features: list[float] = [
            float(ENCODER_VERSION),
            float(state.config.n_teams),
            float(state.config.n_teams / self.max_teams),
        ]
        features.extend(
            float(team < state.config.n_teams) for team in range(self.max_teams)
        )
        features.append(float(state.config.controlled_team))
        features.extend(
            float(team == state.config.controlled_team) for team in range(self.max_teams)
        )
        features.extend(
            [
                float(state.round_number if not state.is_terminal else state.config.n_rounds),
                float(progress),
                float(environment.picks_until_team(state, state.config.controlled_team)),
                float(state.config.n_rounds),
                float(rounds_remaining),
                float(state.config.roster.roster_size),
                float(state.config.roster.roster_size - len(controlled_roster)),
                float(state.config.roster.bench_slots),
                float(state.config.season_weeks),
                float(state.config.scoring.points_per_reception),
                float(state.config.scoring.passing_touchdown_points),
            ]
        )
        features.extend(
            float(state.config.roster.required[position]) for position in POSITIONS
        )
        features.append(float(state.config.roster.flex_slots))
        features.extend(
            float(position in state.config.roster.flex_eligible) for position in POSITIONS
        )
        features.extend(
            float(limits[position]) if limits is not None else 0.0 for position in POSITIONS
        )
        features.extend(float(limits is not None) for _ in POSITIONS)

        available = state.available_players()
        boards: list[list[Player]] = []
        for position in POSITIONS:
            board = sorted(
                (player for player in available if player.position == position),
                key=lambda player: (-player.mu, player.player_id),
            )[: self.board_size_per_position]
            boards.append(board)
            for rank in range(self.board_size_per_position):
                features.extend(_triple(board[rank]) if rank < len(board) else (0.0, 0.0, 0.0))
        for board in boards:
            features.extend(
                float(rank < len(board)) for rank in range(self.board_size_per_position)
            )

        ordered_roster = sorted(
            controlled_roster,
            key=lambda player: (POSITIONS.index(player.position), -player.mu, player.player_id),
        )
        for slot in range(self.max_roster_size):
            if slot < len(ordered_roster):
                player = ordered_roster[slot]
                features.extend(_triple(player))
                features.extend(float(player.position == position) for position in POSITIONS)
                features.append(1.0)
            else:
                features.extend((0.0,) * (len(PLAYER_FEATURES) + len(POSITIONS) + 1))

        features.extend(remaining_requirements(state).tolist())
        encoded = np.asarray(features, dtype=np.float32)
        if encoded.size != len(self.feature_names(state)):
            raise AssertionError("feature layout and encoded vector disagree")
        return encoded

    def slices(self, state: DraftState | None = None) -> dict[str, slice]:
        if state is not None:
            self._validate_supported(state)
        names = self.feature_names(state)
        board_start = next(index for index, name in enumerate(names) if name.startswith("board."))
        roster_start = next(index for index, name in enumerate(names) if name.startswith("roster."))
        requirements_start = next(index for index, name in enumerate(names) if name.startswith("need."))
        return {
            "context": slice(0, board_start),
            "board": slice(board_start, roster_start),
            "roster": slice(roster_start, requirements_start),
            "requirements": slice(requirements_start, len(names)),
        }


def encode_state(state: DraftState) -> np.ndarray:
    """Convenience API using the default top-10 board encoder."""

    return StateEncoder().encode_state(state)
