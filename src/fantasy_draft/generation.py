"""Generation of reachable partial draft states, separate from evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from fantasy_draft.config import (
    LeagueConfig,
    PlayerPoolConfig,
    RosterConfig,
    default_position_profiles,
)
from fantasy_draft.draft import DraftEnvironment, DraftState
from fantasy_draft.models import Position
from fantasy_draft.players import generate_players
from fantasy_draft.policies import (
    RosterAwareSoftmaxPolicy,
)
from fantasy_draft.state import ENCODER_VERSION


@dataclass(frozen=True, slots=True)
class GeneratedState:
    state_id: str
    state: DraftState
    phase: str
    generation_seed: int
    policy_name: str
    league_config_id: str
    player_pool_seed: int
    draft_history_seed: int
    opponent_policy_config: dict[str, Any]
    encoder_version: int = ENCODER_VERSION


@dataclass(frozen=True, slots=True)
class LeagueConfigDistribution:
    """Configurable Cartesian family of synthetic leagues.

    By default, 1-QB leagues are weighted 3x compared to 2-QB leagues (75% / 25%),
    and bench slots include 3, 4, or 5 spots (with 3-4 being standard and 5 bench spots
    appearing 20% of the time).
    """

    team_counts: tuple[int, ...] = (8, 10, 12)
    qb_requirements: tuple[int, ...] = (1, 1, 1, 2)
    flex_requirements: tuple[int, ...] = (1, 2)
    bench_slots: tuple[int, ...] = (3, 3, 4, 4, 5)

    def configurations(self) -> tuple[LeagueConfig, ...]:
        configs: list[LeagueConfig] = []
        for teams in self.team_counts:
            for qbs in self.qb_requirements:
                for flex in self.flex_requirements:
                    for bench in self.bench_slots:
                        roster = RosterConfig(
                            required={
                                Position.QB: qbs,
                                Position.RB: 2,
                                Position.WR: 2,
                                Position.TE: 1,
                            },
                            flex_slots=flex,
                            bench_slots=bench,
                        )
                        configs.append(LeagueConfig(n_teams=teams, roster=roster))
        return tuple(configs)


def league_config_id(config: LeagueConfig) -> str:
    required = config.roster.required
    return (
        f"{config.n_teams}t-qb{required[Position.QB]}"
        f"-rb{required[Position.RB]}-wr{required[Position.WR]}"
        f"-te{required[Position.TE]}-flex{config.roster.flex_slots}"
        f"-bench{config.roster.bench_slots}"
    )


def _sample_player_pool_config(
    league: LeagueConfig,
    rng: np.random.Generator,
) -> PlayerPoolConfig:
    """Vary talent/noise while ensuring the pool exceeds the full draft."""

    base = default_position_profiles()
    target_size = league.total_picks + max(40, league.n_teams * 5)
    scale = max(1.0, target_size / sum(profile.count for profile in base.values()))
    profiles = {}
    for position, profile in base.items():
        talent_scale = float(rng.uniform(0.94, 1.06))
        decay_scale = float(rng.uniform(0.85, 1.15))
        variance_scale = float(rng.uniform(0.9, 1.12))
        injury_scale = float(rng.uniform(0.9, 1.1))
        profiles[position] = replace(
            profile,
            count=int(np.ceil(profile.count * scale)),
            elite_mu=profile.elite_mu * talent_scale,
            floor_mu=profile.floor_mu * talent_scale,
            decay_power=profile.decay_power * decay_scale,
            mu_noise=profile.mu_noise * variance_scale,
            sigma_low=profile.sigma_low * variance_scale,
            sigma_high=profile.sigma_high * variance_scale,
            injury_beta=profile.injury_beta / injury_scale,
        )
    return PlayerPoolConfig(profiles=profiles)


def draft_phase(state: DraftState) -> str:
    fraction = state.round_index / state.config.n_rounds
    if fraction < 1 / 3:
        return "early"
    if fraction < 2 / 3:
        return "middle"
    return "late"


def generate_states(
    n_states: int,
    seed: int,
    league_config: LeagueConfig | None = None,
    player_pool_config: PlayerPoolConfig | None = None,
    league_distribution: LeagueConfigDistribution | None = None,
) -> tuple[GeneratedState, ...]:
    """Sample controlled-team decision states from actual mixed-policy drafts."""

    if n_states < 1:
        raise ValueError("n_states must be positive")
    root_rng = np.random.default_rng(seed)
    distribution = league_distribution or LeagueConfigDistribution()
    config_templates = (league_config,) if league_config is not None else distribution.configurations()
    config_order = list(config_templates)
    root_rng.shuffle(config_order)
    phases = ("early", "middle", "late")
    base, extra = divmod(n_states, len(phases))
    targets = {phase: base + (index < extra) for index, phase in enumerate(phases)}
    collected: dict[str, list[GeneratedState]] = {phase: [] for phase in phases}
    draft_number = 0

    while any(len(collected[phase]) < targets[phase] for phase in phases):
        template = config_order[draft_number % len(config_order)]
        controlled_team = int(root_rng.integers(template.n_teams))
        league = replace(template, controlled_team=controlled_team)
        player_seed = int(root_rng.integers(0, np.iinfo(np.int64).max))
        draft_seed = int(root_rng.integers(0, np.iinfo(np.int64).max))
        parameter_rng = np.random.default_rng(player_seed)
        pool = player_pool_config or _sample_player_pool_config(league, parameter_rng)
        if pool.total_players < league.total_picks:
            raise ValueError("player pool is smaller than the configured draft")
        players = generate_players(player_seed, pool)
        environment = DraftEnvironment(league)
        state = environment.initial_state(players)
        draft_rng = np.random.default_rng(draft_seed)
        shared_policy = RosterAwareSoftmaxPolicy(
            temperature=float(draft_rng.uniform(2.0, 5.0)),
            starter_need_bonus=float(draft_rng.uniform(3.0, 8.0)),
            flex_need_bonus=float(draft_rng.uniform(1.0, 4.0)),
            value_above_replacement_weight=float(draft_rng.uniform(0.7, 1.3)),
        )
        policy_metadata = {
            "shared_policy": shared_policy.name,
            "candidate_family": [shared_policy.name],
            "roster_aware_temperature": shared_policy.temperature,
            "starter_need_bonus": shared_policy.starter_need_bonus,
            "flex_need_bonus": shared_policy.flex_need_bonus,
            "value_above_replacement_weight": (
                shared_policy.value_above_replacement_weight
            ),
        }
        captured_phases: set[str] = set()
        while not state.is_terminal:
            if state.current_team == league.controlled_team:
                phase = draft_phase(state)
                if (
                    phase not in captured_phases
                    and len(collected[phase]) < targets[phase]
                ):
                    collected[phase].append(
                        GeneratedState(
                            state_id=f"d{draft_number:06d}-p{state.pick_index:03d}",
                            state=state,
                            phase=phase,
                            generation_seed=draft_seed,
                            policy_name=shared_policy.name,
                            league_config_id=league_config_id(league),
                            player_pool_seed=player_seed,
                            draft_history_seed=draft_seed,
                            opponent_policy_config=policy_metadata,
                        )
                    )
                    captured_phases.add(phase)
            action = shared_policy.select_player(state, environment, draft_rng)
            state = environment.step(state, action, validate=False)
        draft_number += 1

    result = [item for phase in phases for item in collected[phase]]
    root_rng.shuffle(result)
    return tuple(result)
