"""Monte Carlo fitted policy iteration for the controlled draft team.

Opponents remain on ``RosterAwareSoftmaxPolicy``.  The controlled policy is
evaluated from complete on-policy draft trajectories, approximated with an MLP,
and greedified by ranking all legal post-action states.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fantasy_draft.config import LeagueConfig
from fantasy_draft.draft import DraftEnvironment, DraftState
from fantasy_draft.generation import (
    LeagueConfigDistribution,
    _sample_player_pool_config,
    draft_phase,
    league_config_id,
)
from fantasy_draft.players import generate_players
from fantasy_draft.policies import DraftPolicy, RosterAwareSoftmaxPolicy
from fantasy_draft.season import SeasonSimulator, mean_weekly_score
from fantasy_draft.state import ENCODER_VERSION, StateEncoder


class ValueMLP(nn.Module):
    """Checkpoint-compatible state-value network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ControlledPolicy(Protocol):
    """Subset of ``DraftPolicy`` used for controlled-team choices."""

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


class GreedyValuePolicy:
    """Choose the action maximizing the MLP value of the post-pick state."""

    def __init__(
        self,
        checkpoint: str | Path | dict[str, Any],
        *,
        device: str | torch.device = "cpu",
        name: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        if isinstance(checkpoint, (str, Path)):
            data = torch.load(checkpoint, map_location=self.device)
            source_name = Path(checkpoint).stem
        else:
            data = checkpoint
            source_name = f"iter_{int(data.get('iteration', -1)):02d}"
        self.model = ValueMLP(**data["model_config"]).to(self.device)
        self.model.load_state_dict(data["model_state_dict"])
        self.model.eval()
        self.feature_mean = data["feature_mean"].to(self.device).float()
        self.feature_std = data["feature_std"].to(self.device).float()
        self.target_mean = float(data["target_mean"])
        self.target_std = float(data["target_std"])
        self.encoder = StateEncoder()
        self._name = name or f"greedy_value_{source_name}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_deterministic(self) -> bool:
        return True

    def action_values(
        self,
        state: DraftState,
        environment: DraftEnvironment,
    ) -> tuple[tuple[int, ...], np.ndarray]:
        legal = environment.legal_actions(state)
        if not legal:
            raise ValueError("policy called with no legal actions")
        post_states = [
            environment.step(state, action, validate=False) for action in legal
        ]
        vectors = np.stack(
            [self.encoder.encode_state(post_state) for post_state in post_states]
        )
        features = torch.from_numpy(vectors).to(self.device).float()
        normalized = (features - self.feature_mean) / self.feature_std
        with torch.no_grad():
            normalized_values = self.model(normalized).squeeze(1)
        values = (
            normalized_values * self.target_std + self.target_mean
        ).detach().cpu().numpy()
        return legal, values

    def select_player(
        self,
        state: DraftState,
        environment: DraftEnvironment,
        rng: np.random.Generator,
    ) -> int:
        del rng
        legal, values = self.action_values(state, environment)
        return int(legal[int(np.argmax(values))])


@dataclass(frozen=True, slots=True)
class EpsilonPolicy:
    """Uniformly explore legal actions around a controlled base policy."""

    base_policy: ControlledPolicy
    epsilon: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")

    @property
    def name(self) -> str:
        return f"epsilon_{self.epsilon:g}_{self.base_policy.name}"

    @property
    def is_deterministic(self) -> bool:
        return self.epsilon == 0.0 and self.base_policy.is_deterministic

    def select_player(
        self,
        state: DraftState,
        environment: DraftEnvironment,
        rng: np.random.Generator,
    ) -> int:
        legal = environment.legal_actions(state)
        if not legal:
            raise ValueError("policy called with no legal actions")
        if rng.random() < self.epsilon:
            return int(rng.choice(legal))
        return self.base_policy.select_player(state, environment, rng)


@dataclass(frozen=True, slots=True)
class ControlledVsOpponentPolicy:
    """Route controlled picks to one policy and opponents to another."""

    controlled_policy: ControlledPolicy
    opponent_policy: DraftPolicy

    @property
    def name(self) -> str:
        return (
            f"controlled[{self.controlled_policy.name}]"
            f"_vs[{self.opponent_policy.name}]"
        )

    @property
    def is_deterministic(self) -> bool:
        return (
            self.controlled_policy.is_deterministic
            and self.opponent_policy.is_deterministic
        )

    def select_player(
        self,
        state: DraftState,
        environment: DraftEnvironment,
        rng: np.random.Generator,
    ) -> int:
        policy = (
            self.controlled_policy
            if state.current_team == state.config.controlled_team
            else self.opponent_policy
        )
        return policy.select_player(state, environment, rng)


@dataclass(frozen=True, slots=True)
class CollectionResult:
    rows: int
    drafts: int
    output_path: Path
    reward_mean: float
    reward_std: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    checkpoint_path: Path
    history: tuple[dict[str, float], ...]
    test_metrics: dict[str, float]
    split_rows: dict[str, int]


@dataclass(frozen=True, slots=True)
class PolicyIterationConfig:
    iterations: int = 3
    samples_per_iteration: int = 20_000
    season_simulations: int = 25
    epsilon: float = 0.10
    benchmark_drafts: int = 96
    benchmark_seasons: int = 100
    seed: int = 42
    max_epochs: int = 100
    patience: int = 12
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    data_dir: Path = Path("data/policy_iteration")
    model_dir: Path = Path("models/policy_iteration")
    resume: bool = True

    def __post_init__(self) -> None:
        integer_values = (
            self.iterations,
            self.samples_per_iteration,
            self.season_simulations,
            self.benchmark_drafts,
            self.benchmark_seasons,
            self.max_epochs,
            self.patience,
            self.batch_size,
        )
        if any(value < 1 for value in integer_values):
            raise ValueError("iteration and sample counts must be positive")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")


def _season_return(
    terminal: DraftState,
    season_simulations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    simulator = SeasonSimulator(
        roster_config=terminal.config.roster,
        n_weeks=terminal.config.season_weeks,
    )
    seeds = tuple(
        int(value)
        for value in rng.integers(0, 1 << 31, size=season_simulations)
    )
    seasons = simulator.simulate_many(
        terminal.roster_players(terminal.config.controlled_team),
        terminal.available_players(),
        seeds,
        player_universe=terminal.players,
    )
    utilities = np.asarray(
        [mean_weekly_score(season.weekly_scores) for season in seasons],
        dtype=float,
    )
    standard_error = (
        float(utilities.std(ddof=1) / math.sqrt(len(utilities)))
        if len(utilities) > 1
        else 0.0
    )
    return float(utilities.mean()), standard_error


def _write_value_parquet(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    policy_name: str,
    iteration: int,
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            'Parquet output requires pyarrow; install with pip install -e ".[ml]"'
        ) from exc

    vectors = np.stack([row.pop("state_vector") for row in rows]).astype(np.float32)
    columns = {key: [row[key] for row in rows] for key in rows[0]}
    columns["state_vector"] = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.reshape(-1), type=pa.float32()),
        vectors.shape[1],
    )
    table = pa.table(columns)
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"dataset_type": b"on_policy_trajectory_state_value",
            b"reward_semantics": (
                b"terminal_mean_weekly_score_shared_by_controlled_trajectory"
            ),
            b"feature_names": json.dumps(StateEncoder().feature_names()).encode(),
            b"encoder_version": str(ENCODER_VERSION).encode(),
            b"iteration": str(iteration).encode(),
            b"evaluated_policy": policy_name.encode(),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table.replace_schema_metadata(metadata),
        output_path,
        compression="zstd",
    )


def collect_policy_trajectories(
    *,
    n_samples: int,
    controlled_policy: ControlledPolicy,
    output_path: str | Path,
    iteration: int,
    season_simulations: int,
    epsilon: float,
    seed: int,
    league_distribution: LeagueConfigDistribution | None = None,
    verbose: bool = True,
) -> CollectionResult:
    """Collect post-pick states and on-policy Monte Carlo trajectory returns."""

    if n_samples < 1 or season_simulations < 1:
        raise ValueError("sample counts must be positive")
    output = Path(output_path)
    root_rng = np.random.default_rng(seed)
    distribution = league_distribution or LeagueConfigDistribution()
    templates = list(distribution.configurations())
    root_rng.shuffle(templates)
    behavior = EpsilonPolicy(controlled_policy, epsilon)
    opponent = RosterAwareSoftmaxPolicy()
    combined = ControlledVsOpponentPolicy(behavior, opponent)
    encoder = StateEncoder()
    rows: list[dict[str, Any]] = []
    draft_number = 0

    while len(rows) < n_samples:
        template = templates[draft_number % len(templates)]
        controlled_team = int(root_rng.integers(template.n_teams))
        league = replace(template, controlled_team=controlled_team)
        player_seed = int(root_rng.integers(0, np.iinfo(np.int64).max))
        draft_seed = int(root_rng.integers(0, np.iinfo(np.int64).max))
        season_seed = int(root_rng.integers(0, np.iinfo(np.int64).max))
        pool = _sample_player_pool_config(
            league,
            np.random.default_rng(player_seed),
        )
        state = DraftEnvironment(league).initial_state(
            generate_players(player_seed, pool)
        )
        environment = DraftEnvironment(league)
        draft_rng = np.random.default_rng(draft_seed)
        trajectory: list[tuple[DraftState, int, str, int]] = []

        while not state.is_terminal:
            is_controlled = state.current_team == controlled_team
            round_number = state.round_number
            phase = draft_phase(state)
            action = combined.select_player(state, environment, draft_rng)
            state = environment.step(state, action, validate=False)
            if is_controlled:
                trajectory.append((state, action, phase, round_number))

        reward, reward_se = _season_return(
            state,
            season_simulations,
            np.random.default_rng(season_seed),
        )
        draft_id = f"iter-{iteration:02d}-draft-{draft_number:07d}"
        lookup = state.player_map
        for post_state, action, phase, round_number in trajectory:
            if len(rows) >= n_samples:
                break
            player = lookup[action]
            rows.append(
                {
                    "sample_id": f"{draft_id}-pick-{post_state.pick_index:03d}",
                    "draft_id": draft_id,
                    "decision_id": (
                        f"{draft_id}-decision-{post_state.pick_index - 1:03d}"
                    ),
                    "state_vector": encoder.encode_state(post_state),
                    "reward": reward,
                    "reward_standard_error": reward_se,
                    "iteration": iteration,
                    "action_player_id": action,
                    "action_position": player.position.value,
                    "action_mu": player.mu,
                    "round_number": round_number,
                    "phase": phase,
                    "pick_index_after_action": post_state.pick_index,
                    "league_config_id": league_config_id(league),
                    "num_teams": league.n_teams,
                    "controlled_team": controlled_team,
                    "player_pool_seed": player_seed,
                    "draft_history_seed": draft_seed,
                    "season_seed": season_seed,
                    "generation_policy": combined.name,
                    "epsilon": epsilon,
                    "season_simulations": season_simulations,
                    "encoder_version": ENCODER_VERSION,
                }
            )
        draft_number += 1
        if verbose and (draft_number == 1 or draft_number % 250 == 0):
            print(
                f"iteration {iteration}: {len(rows):,}/{n_samples:,} rows "
                f"from {draft_number:,} drafts"
            )

    _write_value_parquet(
        rows,
        output,
        policy_name=combined.name,
        iteration=iteration,
    )
    rewards = np.asarray([row["reward"] for row in rows], dtype=float)
    return CollectionResult(
        rows=len(rows),
        drafts=draft_number,
        output_path=output,
        reward_mean=float(rewards.mean()),
        reward_std=float(rewards.std()),
    )


def _split_drafts(
    frame: pd.DataFrame,
    seed: int,
) -> dict[str, np.ndarray]:
    """Create deterministic 70/15/15 draft-grouped, format-aware splits."""

    rng = np.random.default_rng(seed)
    split_ids: dict[str, set[str]] = {
        "train": set(),
        "val": set(),
        "test": set(),
    }
    drafts = frame[["draft_id", "league_config_id"]].drop_duplicates("draft_id")
    for _, group in drafts.groupby("league_config_id"):
        ids = group["draft_id"].to_numpy(copy=True)
        rng.shuffle(ids)
        count = len(ids)
        if count < 3:
            split_ids["train"].update(ids.tolist())
            continue
        validation = max(1, int(round(count * 0.15)))
        testing = max(1, int(round(count * 0.15)))
        if validation + testing >= count:
            validation = testing = 1
        split_ids["test"].update(ids[-testing:].tolist())
        split_ids["val"].update(ids[-(testing + validation) : -testing].tolist())
        split_ids["train"].update(ids[: -(testing + validation)].tolist())
    if not split_ids["val"] or not split_ids["test"]:
        # Tiny smoke datasets may have fewer than three drafts per format.
        # Preserve draft grouping and fall back to a global deterministic split.
        ids = drafts["draft_id"].to_numpy(copy=True)
        rng.shuffle(ids)
        if len(ids) < 3:
            raise ValueError("dataset needs at least three distinct drafts")
        validation = max(1, int(round(len(ids) * 0.15)))
        testing = max(1, int(round(len(ids) * 0.15)))
        if validation + testing >= len(ids):
            validation = testing = 1
        split_ids = {
            "train": set(ids[: -(testing + validation)].tolist()),
            "val": set(ids[-(testing + validation) : -testing].tolist()),
            "test": set(ids[-testing:].tolist()),
        }
    return {
        name: np.flatnonzero(frame["draft_id"].isin(ids).to_numpy())
        for name, ids in split_ids.items()
    }


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - actual
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r_squared = (
        1.0 - float(np.sum(residual**2)) / denominator
        if denominator > 0
        else float("nan")
    )
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": math.sqrt(mse),
        "R2": r_squared,
    }


def train_value_model(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    *,
    iteration: int,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Train a fresh value network on one iteration's current-policy labels."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            'Training requires pyarrow; install with pip install -e ".[ml]"'
        ) from exc

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device is None:
        selected_device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        selected_device = torch.device(device)

    table = pq.read_table(dataset_path)
    metadata = table.schema.metadata or {}
    feature_names = json.loads(metadata[b"feature_names"].decode())
    frame = table.drop(["state_vector"]).to_pandas()
    features = np.asarray(table["state_vector"].to_pylist(), dtype=np.float32)
    targets = frame["reward"].to_numpy(dtype=np.float32)
    splits = _split_drafts(frame, seed)
    if any(len(splits[name]) == 0 for name in ("train", "val", "test")):
        raise ValueError("dataset needs enough drafts for non-empty grouped splits")

    train_indices = splits["train"]
    feature_mean = features[train_indices].mean(
        axis=0, dtype=np.float64
    ).astype(np.float32)
    feature_std = features[train_indices].std(
        axis=0, dtype=np.float64
    ).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    target_mean = float(targets[train_indices].mean())
    target_std = float(targets[train_indices].std())
    if target_std <= 0:
        raise ValueError("training targets have zero variance")
    scaled_features = (features - feature_mean) / feature_std
    scaled_targets = (targets - target_mean) / target_std

    def dataset(indices: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(scaled_features[indices].astype(np.float32, copy=False)),
            torch.from_numpy(
                scaled_targets[indices, None].astype(np.float32, copy=False)
            ),
        )

    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            dataset(splits["train"]),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=selected_device.type == "cuda",
        ),
        "val": DataLoader(
            dataset(splits["val"]), batch_size=2048, shuffle=False
        ),
        "test": DataLoader(
            dataset(splits["test"]), batch_size=2048, shuffle=False
        ),
    }
    model_config = {
        "input_dim": int(features.shape[1]),
        "hidden_dims": (256, 128, 64),
        "dropout": 0.10,
    }
    model = ValueMLP(**model_config).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )
    loss_function = nn.SmoothL1Loss()
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        epoch_losses: dict[str, float] = {}
        for phase in ("train", "val"):
            model.train(phase == "train")
            total_loss = 0.0
            total_rows = 0
            for batch_features, batch_targets in loaders[phase]:
                batch_features = batch_features.to(selected_device)
                batch_targets = batch_targets.to(selected_device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    predictions = model(batch_features)
                    loss = loss_function(predictions, batch_targets)
                    if phase == "train":
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                total_loss += float(loss.detach()) * len(batch_features)
                total_rows += len(batch_features)
            epoch_losses[phase] = total_loss / total_rows
        scheduler.step(epoch_losses["val"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": epoch_losses["train"],
                "val_loss": epoch_losses["val"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(
                f"iteration {iteration} epoch {epoch:03d}: "
                f"train={epoch_losses['train']:.4f} "
                f"val={epoch_losses['val']:.4f}"
            )
        if epoch_losses["val"] < best_validation - 1e-5:
            best_validation = epoch_losses["val"]
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    test_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch_features, _ in loaders["test"]:
            values = model(batch_features.to(selected_device)).cpu().numpy().ravel()
            test_predictions.append(values)
    predicted = np.concatenate(test_predictions) * target_std + target_mean
    actual = targets[splits["test"]]
    test_metrics = _metrics(actual, predicted)

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "model_config": model_config,
            "feature_names": feature_names,
            "feature_mean": torch.from_numpy(feature_mean),
            "feature_std": torch.from_numpy(feature_std),
            "target_mean": target_mean,
            "target_std": target_std,
            "encoder_version": ENCODER_VERSION,
            "iteration": iteration,
            "training_seed": seed,
            "test_metrics": test_metrics,
            "reward_semantics": (
                "Monte Carlo terminal mean weekly score under the iteration policy"
            ),
            "evaluated_policy": metadata.get(b"evaluated_policy", b"unknown").decode(),
        },
        checkpoint,
    )
    return TrainingResult(
        checkpoint_path=checkpoint,
        history=tuple(history),
        test_metrics=test_metrics,
        split_rows={name: int(len(indices)) for name, indices in splits.items()},
    )


def _complete_draft(
    league: LeagueConfig,
    players: tuple,
    controlled_policy: ControlledPolicy,
    seed: int,
) -> DraftState:
    environment = DraftEnvironment(league)
    state = environment.initial_state(players)
    combined = ControlledVsOpponentPolicy(
        controlled_policy,
        RosterAwareSoftmaxPolicy(),
    )
    rng = np.random.default_rng(seed)
    while not state.is_terminal:
        action = combined.select_player(state, environment, rng)
        state = environment.step(state, action, validate=False)
    return state


def _score_terminal(
    terminal: DraftState,
    season_seeds: tuple[int, ...],
) -> float:
    simulator = SeasonSimulator(
        terminal.config.roster,
        terminal.config.season_weeks,
    )
    seasons = simulator.simulate_many(
        terminal.roster_players(terminal.config.controlled_team),
        terminal.available_players(),
        season_seeds,
        player_universe=terminal.players,
    )
    return float(
        np.mean([mean_weekly_score(season.weekly_scores) for season in seasons])
    )


def benchmark_controlled_policy(
    candidate_policy: ControlledPolicy,
    *,
    n_drafts: int,
    season_simulations: int,
    seed: int,
    league_distribution: LeagueConfigDistribution | None = None,
) -> pd.DataFrame:
    """Paired candidate-versus-baseline evaluation with fixed opponents."""

    if n_drafts < 1 or season_simulations < 1:
        raise ValueError("benchmark sample counts must be positive")
    root_rng = np.random.default_rng(seed)
    distribution = league_distribution or LeagueConfigDistribution()
    templates = list(distribution.configurations())
    baseline = RosterAwareSoftmaxPolicy()
    records: list[dict[str, Any]] = []
    for draft_index in range(n_drafts):
        template = templates[draft_index % len(templates)]
        controlled_team = int(root_rng.integers(template.n_teams))
        league = replace(template, controlled_team=controlled_team)
        player_seed = int(root_rng.integers(0, np.iinfo(np.int64).max))
        draft_seed = int(root_rng.integers(0, np.iinfo(np.int64).max))
        pool = _sample_player_pool_config(
            league,
            np.random.default_rng(player_seed),
        )
        players = generate_players(player_seed, pool)
        candidate_terminal = _complete_draft(
            league,
            players,
            candidate_policy,
            draft_seed,
        )
        baseline_terminal = _complete_draft(
            league,
            players,
            baseline,
            draft_seed,
        )
        season_seeds = tuple(
            int(value)
            for value in root_rng.integers(
                0, 1 << 31, size=season_simulations
            )
        )
        candidate_value = _score_terminal(candidate_terminal, season_seeds)
        baseline_value = _score_terminal(baseline_terminal, season_seeds)
        records.append(
            {
                "draft_index": draft_index,
                "league_config_id": league_config_id(league),
                "controlled_team": controlled_team,
                "player_pool_seed": player_seed,
                "draft_seed": draft_seed,
                "candidate_value": candidate_value,
                "baseline_value": baseline_value,
                "paired_advantage": candidate_value - baseline_value,
                "candidate_won": candidate_value > baseline_value,
            }
        )
    return pd.DataFrame.from_records(records)


def summarize_benchmark(frame: pd.DataFrame) -> dict[str, float]:
    advantages = frame["paired_advantage"].to_numpy(dtype=float)
    standard_error = (
        float(advantages.std(ddof=1) / math.sqrt(len(advantages)))
        if len(advantages) > 1
        else 0.0
    )
    return {
        "candidate_value": float(frame["candidate_value"].mean()),
        "baseline_value": float(frame["baseline_value"].mean()),
        "paired_advantage": float(advantages.mean()),
        "paired_advantage_standard_error": standard_error,
        "paired_advantage_ci_low": float(advantages.mean() - 1.96 * standard_error),
        "paired_advantage_ci_high": float(advantages.mean() + 1.96 * standard_error),
        "paired_win_rate": float(frame["candidate_won"].mean()),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_fitted_policy_iteration(
    config: PolicyIterationConfig,
    *,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run or resume the complete evaluate/train/improve loop."""

    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.data_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": "monte_carlo_fitted_policy_iteration",
        "opponents": "RosterAwareSoftmaxPolicy",
        "config": {
            **asdict(config),
            "data_dir": str(config.data_dir),
            "model_dir": str(config.model_dir),
        },
        "iterations": [],
    }
    if config.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    completed = {
        int(record["iteration"]): record for record in manifest.get("iterations", [])
    }
    summaries: list[dict[str, Any]] = []
    previous_checkpoint: Path | None = None

    for iteration in range(config.iterations):
        dataset_path = config.data_dir / f"iteration_{iteration:02d}.parquet"
        checkpoint_path = config.model_dir / f"value_iter_{iteration:02d}.pt"
        benchmark_path = config.data_dir / f"benchmark_{iteration:02d}.parquet"
        metrics_path = config.data_dir / f"metrics_{iteration:02d}.json"
        if (
            config.resume
            and iteration in completed
            and dataset_path.exists()
            and checkpoint_path.exists()
            and benchmark_path.exists()
            and metrics_path.exists()
        ):
            if verbose:
                print(f"iteration {iteration}: loading completed artifacts")
            summary = json.loads(metrics_path.read_text(encoding="utf-8"))
            summaries.append(summary)
            previous_checkpoint = checkpoint_path
            continue

        if previous_checkpoint is None:
            evaluated_policy: ControlledPolicy = RosterAwareSoftmaxPolicy()
        else:
            evaluated_policy = GreedyValuePolicy(
                previous_checkpoint,
                device=device or "cpu",
            )
        collection = collect_policy_trajectories(
            n_samples=config.samples_per_iteration,
            controlled_policy=evaluated_policy,
            output_path=dataset_path,
            iteration=iteration,
            season_simulations=config.season_simulations,
            epsilon=config.epsilon,
            seed=config.seed + iteration * 10_000,
            verbose=verbose,
        )
        training = train_value_model(
            dataset_path,
            checkpoint_path,
            iteration=iteration,
            seed=config.seed + iteration,
            max_epochs=config.max_epochs,
            patience=config.patience,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            device=device,
            verbose=verbose,
        )
        improved_policy = GreedyValuePolicy(
            checkpoint_path,
            device=device or "cpu",
        )
        benchmark = benchmark_controlled_policy(
            improved_policy,
            n_drafts=config.benchmark_drafts,
            season_simulations=config.benchmark_seasons,
            seed=config.seed + 500_000,
        )
        benchmark.to_parquet(
            benchmark_path,
            index=False,
            compression="zstd",
        )
        summary = {
            "iteration": iteration,
            "evaluated_policy": evaluated_policy.name,
            "improved_policy": improved_policy.name,
            "dataset_path": str(dataset_path),
            "checkpoint_path": str(checkpoint_path),
            "benchmark_path": str(benchmark_path),
            "collection": {
                **asdict(collection),
                "output_path": str(collection.output_path),
            },
            "epochs": len(training.history),
            "split_rows": training.split_rows,
            "test_metrics": training.test_metrics,
            "benchmark": summarize_benchmark(benchmark),
            "history": list(training.history),
        }
        _write_json(metrics_path, summary)
        completed[iteration] = {
            "iteration": iteration,
            "dataset_path": str(dataset_path),
            "checkpoint_path": str(checkpoint_path),
            "benchmark_path": str(benchmark_path),
            "metrics_path": str(metrics_path),
        }
        manifest["iterations"] = [
            completed[index] for index in sorted(completed)
        ]
        _write_json(manifest_path, manifest)
        summaries.append(summary)
        previous_checkpoint = checkpoint_path

    best = max(
        summaries,
        key=lambda item: item["benchmark"]["candidate_value"],
    )
    best_path = Path(best["checkpoint_path"])
    shutil.copy2(best_path, config.model_dir / "best_value.pt")
    manifest["recommended_iteration"] = int(best["iteration"])
    manifest["recommended_checkpoint"] = str(
        config.model_dir / "best_value.pt"
    )
    _write_json(manifest_path, manifest)
    return pd.DataFrame(
        [
            {
                "iteration": item["iteration"],
                "evaluated_policy": item["evaluated_policy"],
                "test_mae": item["test_metrics"]["MAE"],
                "test_rmse": item["test_metrics"]["RMSE"],
                "test_r2": item["test_metrics"]["R2"],
                **item["benchmark"],
            }
            for item in summaries
        ]
    )
