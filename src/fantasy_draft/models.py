"""Core domain types shared by drafting and season simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class Position(str, Enum):
    """Supported fantasy positions."""

    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"


POSITIONS: tuple[Position, ...] = tuple(Position)


@dataclass(frozen=True, slots=True)
class Player:
    """Synthetic player parameters known to the environment."""

    player_id: int
    position: Position
    mu: float
    sigma: float
    injury_probability: float
    team_id: int

    def __post_init__(self) -> None:
        if self.player_id < 0:
            raise ValueError("player_id must be non-negative")
        if self.mu < 0 or self.sigma < 0:
            raise ValueError("mu and sigma must be non-negative")
        if not 0.0 <= self.injury_probability <= 1.0:
            raise ValueError("injury_probability must be in [0, 1]")
        if self.team_id < 0:
            raise ValueError("team_id must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["position"] = self.position.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Player:
        return cls(
            player_id=int(data["player_id"]),
            position=Position(data["position"]),
            mu=float(data["mu"]),
            sigma=float(data["sigma"]),
            injury_probability=float(data["injury_probability"]),
            team_id=int(data["team_id"]),
        )
