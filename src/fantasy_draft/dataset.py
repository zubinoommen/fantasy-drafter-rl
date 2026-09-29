"""Versioned, reconstructable state and state-value dataset formats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from fantasy_draft.draft import DraftState
from fantasy_draft.evaluation import EvaluationResult
from fantasy_draft.generation import GeneratedState

FORMAT_VERSION = 2


@dataclass(frozen=True, slots=True)
class ValueObservation:
    state_id: str
    state_vector: np.ndarray
    estimated_value: float
    standard_error: float
    policy_name: str
    evaluation_parameters: dict[str, Any]
    seed: int
    state_metadata: dict[str, Any]

    @classmethod
    def from_result(
        cls,
        generated: GeneratedState,
        state_vector: np.ndarray,
        result: EvaluationResult,
    ) -> ValueObservation:
        return cls(
            state_id=generated.state_id,
            state_vector=np.asarray(state_vector, dtype=np.float32),
            estimated_value=result.mean_value,
            standard_error=result.standard_error,
            policy_name=result.policy_name,
            evaluation_parameters={
                "n_draft_rollouts": result.n_draft_rollouts,
                "n_season_simulations": result.n_season_simulations,
            },
            seed=result.seed_plan.root_seed,
            state_metadata={
                "league_config_id": generated.league_config_id,
                "num_teams": generated.state.config.n_teams,
                "roster_size": generated.state.config.roster.roster_size,
                "starter_requirements": {
                    position.value: count
                    for position, count in generated.state.config.roster.required.items()
                },
                "flex_slots": generated.state.config.roster.flex_slots,
                "flex_eligible": [
                    position.value
                    for position in generated.state.config.roster.flex_eligible
                ],
                "draft_round": generated.state.round_number,
                "controlled_team": generated.state.config.controlled_team,
                "player_pool_seed": generated.player_pool_seed,
                "draft_history_seed": generated.draft_history_seed,
                "opponent_policy_config": generated.opponent_policy_config,
                "encoder_version": generated.encoder_version,
            },
        )


def save_generated_states(path: str | Path, states: Iterable[GeneratedState]) -> None:
    """Write JSONL records containing every field needed to reconstruct a state."""

    with Path(path).open("w", encoding="utf-8") as handle:
        for generated in states:
            record = {
                "format_version": FORMAT_VERSION,
                "state_id": generated.state_id,
                "phase": generated.phase,
                "generation_seed": generated.generation_seed,
                "policy_name": generated.policy_name,
                "league_config_id": generated.league_config_id,
                "player_pool_seed": generated.player_pool_seed,
                "draft_history_seed": generated.draft_history_seed,
                "opponent_policy_config": generated.opponent_policy_config,
                "encoder_version": generated.encoder_version,
                "state": generated.state.to_dict(),
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def load_generated_states(path: str | Path) -> tuple[GeneratedState, ...]:
    result: list[GeneratedState] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["format_version"] != FORMAT_VERSION:
                raise ValueError("unsupported state dataset version")
            result.append(
                GeneratedState(
                    state_id=record["state_id"],
                    state=DraftState.from_dict(record["state"]),
                    phase=record["phase"],
                    generation_seed=int(record["generation_seed"]),
                    policy_name=record["policy_name"],
                    league_config_id=record["league_config_id"],
                    player_pool_seed=int(record["player_pool_seed"]),
                    draft_history_seed=int(record["draft_history_seed"]),
                    opponent_policy_config=record["opponent_policy_config"],
                    encoder_version=int(record["encoder_version"]),
                )
            )
    return tuple(result)


def save_value_observations(
    path: str | Path,
    observations: Iterable[ValueObservation],
) -> None:
    """Save compact numeric arrays plus JSON metadata in an NPZ archive."""

    rows = tuple(observations)
    if not rows:
        raise ValueError("cannot save an empty value dataset")
    dimensions = {row.state_vector.shape for row in rows}
    if len(dimensions) != 1:
        raise ValueError("state vectors must share a shape")
    metadata = [
        json.dumps(
            {
                "format_version": FORMAT_VERSION,
                "state_id": row.state_id,
                "policy_name": row.policy_name,
                "evaluation_parameters": row.evaluation_parameters,
                "seed": row.seed,
                "state_metadata": row.state_metadata,
            },
            separators=(",", ":"),
        )
        for row in rows
    ]
    np.savez_compressed(
        path,
        state_vectors=np.stack([row.state_vector for row in rows]),
        estimated_values=np.asarray([row.estimated_value for row in rows]),
        standard_errors=np.asarray([row.standard_error for row in rows]),
        metadata=np.asarray(metadata),
    )


def load_value_observations(path: str | Path) -> tuple[ValueObservation, ...]:
    with np.load(path, allow_pickle=False) as archive:
        vectors = archive["state_vectors"]
        values = archive["estimated_values"]
        errors = archive["standard_errors"]
        metadata = archive["metadata"]
        rows: list[ValueObservation] = []
        for vector, value, error, raw_metadata in zip(
            vectors, values, errors, metadata, strict=True
        ):
            details = json.loads(str(raw_metadata))
            if details["format_version"] != FORMAT_VERSION:
                raise ValueError("unsupported value dataset version")
            rows.append(
                ValueObservation(
                    state_id=details["state_id"],
                    state_vector=vector.astype(np.float32),
                    estimated_value=float(value),
                    standard_error=float(error),
                    policy_name=details["policy_name"],
                    evaluation_parameters=details["evaluation_parameters"],
                    seed=int(details["seed"]),
                    state_metadata=details["state_metadata"],
                )
            )
    return tuple(rows)
