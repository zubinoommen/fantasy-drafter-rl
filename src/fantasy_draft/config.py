"""Validated configuration for leagues, rosters, and synthetic players."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from fantasy_draft.models import POSITIONS, Position


@dataclass(frozen=True, slots=True)
class RosterConfig:
    """Lineup requirements; FLEX is eligibility, not a drafted position."""

    required: dict[Position, int] = field(
        default_factory=lambda: {
            Position.QB: 1,
            Position.RB: 2,
            Position.WR: 2,
            Position.TE: 1,
        }
    )
    flex_slots: int = 1
    flex_eligible: tuple[Position, ...] = (Position.RB, Position.WR, Position.TE)
    bench_slots: int = 3
    position_limits: dict[Position, int] | None = None

    def __post_init__(self) -> None:
        normalized = {position: int(self.required.get(position, 0)) for position in POSITIONS}
        if any(value < 0 for value in normalized.values()):
            raise ValueError("required slot counts must be non-negative")
        if self.flex_slots < 0 or self.bench_slots < 0:
            raise ValueError("flex and bench slot counts must be non-negative")
        if not set(self.flex_eligible).issubset(POSITIONS):
            raise ValueError("unsupported FLEX position")
        limits = None
        if self.position_limits is not None:
            limits = {
                position: int(self.position_limits.get(position, self.roster_size))
                for position in POSITIONS
            }
            if any(limit < normalized[position] for position, limit in limits.items()):
                raise ValueError("position limits cannot be below required starters")
        object.__setattr__(self, "required", normalized)
        object.__setattr__(self, "position_limits", limits)

    @property
    def starter_slots(self) -> int:
        return sum(self.required.values()) + self.flex_slots

    @property
    def roster_size(self) -> int:
        return self.starter_slots + self.bench_slots

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": {position.value: count for position, count in self.required.items()},
            "flex_slots": self.flex_slots,
            "flex_eligible": [position.value for position in self.flex_eligible],
            "bench_slots": self.bench_slots,
            "position_limits": (
                None
                if self.position_limits is None
                else {position.value: count for position, count in self.position_limits.items()}
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RosterConfig:
        return cls(
            required={Position(key): int(value) for key, value in data["required"].items()},
            flex_slots=int(data["flex_slots"]),
            flex_eligible=tuple(Position(value) for value in data["flex_eligible"]),
            bench_slots=int(data["bench_slots"]),
            position_limits=(
                None
                if data.get("position_limits") is None
                else {
                    Position(key): int(value)
                    for key, value in data["position_limits"].items()
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Scoring context under which synthetic ``mu`` values are calibrated."""

    points_per_reception: float = 1.0
    passing_touchdown_points: float = 4.0

    def __post_init__(self) -> None:
        if self.points_per_reception < 0 or self.passing_touchdown_points < 0:
            raise ValueError("scoring values must be non-negative")


@dataclass(frozen=True, slots=True)
class LeagueConfig:
    """Draft and season settings."""

    n_teams: int = 8
    n_rounds: int | None = None
    controlled_team: int = 0
    season_weeks: int = 14
    roster: RosterConfig = field(default_factory=RosterConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    def __post_init__(self) -> None:
        rounds = self.roster.roster_size if self.n_rounds is None else self.n_rounds
        if self.n_teams < 2 or rounds < 1 or self.season_weeks < 1:
            raise ValueError("league dimensions must be positive")
        if not 0 <= self.controlled_team < self.n_teams:
            raise ValueError("controlled_team is out of range")
        if rounds > self.roster.roster_size:
            raise ValueError("n_rounds cannot exceed physical roster capacity")
        object.__setattr__(self, "n_rounds", rounds)

    @property
    def total_picks(self) -> int:
        assert self.n_rounds is not None
        return self.n_teams * self.n_rounds

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["roster"] = self.roster.to_dict()
        data["scoring"] = asdict(self.scoring)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LeagueConfig:
        return cls(
            n_teams=int(data["n_teams"]),
            n_rounds=int(data["n_rounds"]),
            controlled_team=int(data["controlled_team"]),
            season_weeks=int(data["season_weeks"]),
            roster=RosterConfig.from_dict(data["roster"]),
            scoring=ScoringConfig(**data.get("scoring", {})),
        )


@dataclass(frozen=True, slots=True)
class PositionProfile:
    """Parameters for a noisy rank-decay full-PPR talent curve."""

    count: int
    elite_mu: float
    floor_mu: float
    decay_power: float
    mu_noise: float
    sigma_low: float
    sigma_high: float
    injury_alpha: float
    injury_beta: float
    injury_max: float


def default_position_profiles() -> dict[Position, PositionProfile]:
    """2018-2025 nflverse-calibrated weekly full-PPR profiles.

    Values are generated by ``scripts/calibrate_nflfastr.py`` from regular-
    season weekly player stats and roster availability.
    """

    return {
        Position.QB: PositionProfile(
            24, 25.0568, 13.7116, 0.505, 1.0685, 5.7238, 10.2576, 0.2374, 1.9804, 0.5294
        ),
        Position.RB: PositionProfile(
            36, 24.3799, 9.888, 0.43, 0.8267, 5.6183, 10.4249, 0.3915, 2.4962, 0.5294
        ),
        Position.WR: PositionProfile(
            42, 23.6808, 11.1982, 0.395, 0.7779, 5.8632, 10.5997, 0.3246, 2.3489, 0.5294
        ),
        Position.TE: PositionProfile(
            18, 18.2445, 8.1104, 0.43, 0.9215, 4.8407, 9.0177, 0.3092, 1.9312, 0.5294
        ),
    }


@dataclass(frozen=True, slots=True)
class PlayerPoolConfig:
    """Synthetic player-pool configuration."""

    profiles: dict[Position, PositionProfile] = field(default_factory=default_position_profiles)
    nfl_team_count: int = 32

    def __post_init__(self) -> None:
        if set(self.profiles) != set(POSITIONS):
            raise ValueError("profiles must cover QB, RB, WR, and TE")
        if any(profile.count <= 0 for profile in self.profiles.values()):
            raise ValueError("each position needs at least one player")
        if self.nfl_team_count < 1:
            raise ValueError("nfl_team_count must be positive")

    @property
    def total_players(self) -> int:
        return sum(profile.count for profile in self.profiles.values())
