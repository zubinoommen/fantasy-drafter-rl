"""Position-specific synthetic full-PPR player generation."""

from __future__ import annotations

import numpy as np

from fantasy_draft.config import PlayerPoolConfig, PositionProfile
from fantasy_draft.models import POSITIONS, Player, Position


def _position_players(
    position: Position,
    profile: PositionProfile,
    first_id: int,
    nfl_team_count: int,
    rng: np.random.Generator,
) -> list[Player]:
    ranks = np.arange(profile.count, dtype=float)
    denominator = max(profile.count - 1, 1)
    fraction = ranks / denominator
    baseline = profile.floor_mu + (profile.elite_mu - profile.floor_mu) * (
        1.0 - fraction**profile.decay_power
    )
    means = np.clip(
        baseline + rng.normal(0.0, profile.mu_noise, profile.count),
        0.75 * profile.floor_mu,
        1.08 * profile.elite_mu,
    )
    # Sort after adding noise: IDs are deterministic rank labels within position.
    means = np.sort(means)[::-1]
    sigmas = rng.uniform(profile.sigma_low, profile.sigma_high, profile.count)
    # Better players are mildly more volatile in absolute point terms.
    sigmas += 0.08 * (means - profile.floor_mu)
    injuries = np.minimum(
        rng.beta(profile.injury_alpha, profile.injury_beta, profile.count),
        profile.injury_max,
    )

    return [
        Player(
            player_id=first_id + rank,
            position=position,
            mu=round(float(means[rank]), 4),
            sigma=round(float(sigmas[rank]), 4),
            injury_probability=round(float(injuries[rank]), 5),
            team_id=(first_id + rank) % nfl_team_count,
        )
        for rank in range(profile.count)
    ]


def generate_players(
    seed: int,
    config: PlayerPoolConfig | None = None,
) -> tuple[Player, ...]:
    """Generate a reproducible pool with distinct positional depth curves."""

    config = config or PlayerPoolConfig()
    seed_sequence = np.random.SeedSequence(seed)
    children = seed_sequence.spawn(len(POSITIONS))
    players: list[Player] = []
    next_id = 0
    for position, child in zip(POSITIONS, children, strict=True):
        position_players = _position_players(
            position,
            config.profiles[position],
            next_id,
            config.nfl_team_count,
            np.random.default_rng(child),
        )
        players.extend(position_players)
        next_id += len(position_players)
    return tuple(players)


def players_by_position(players: tuple[Player, ...]) -> dict[Position, tuple[Player, ...]]:
    """Group players in deterministic descending-mean order."""

    return {
        position: tuple(
            sorted(
                (player for player in players if player.position == position),
                key=lambda player: (-player.mu, player.player_id),
            )
        )
        for position in POSITIONS
    }
