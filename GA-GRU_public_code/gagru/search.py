from __future__ import annotations

import csv
import gc
import importlib.metadata
import json
import os
import platform
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .data import file_sha256
from .errors import SearchValidationError
from .folds import PreparedSplit
from .model import GRUModelConfig
from .pilot import candidate_budget, protocol_training_configs
from .protocol import FrozenProtocol
from .training import TrainingConfig, resolve_device, train_gru


SEARCH_METHODS = ("random", "tpe", "genetic_algorithm")
_METHOD_ALIASES = {"ga": "genetic_algorithm", "genetic": "genetic_algorithm"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(text.encode(encoding))
    os.replace(temporary, path)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class GRUSearchCandidate:
    hidden_size: int
    num_layers: int
    learning_rate: float
    dropout: float
    weight_decay: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @property
    def key(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> GRUSearchCandidate:
        return cls(
            hidden_size=int(values["hidden_size"]),
            num_layers=int(values["num_layers"]),
            learning_rate=float(values["learning_rate"]),
            dropout=float(values["dropout"]),
            weight_decay=float(values["weight_decay"]),
        )


@dataclass(frozen=True)
class GRUSearchSpace:
    hidden_sizes: tuple[int, ...]
    num_layers: tuple[int, ...]
    learning_rate_min: float
    learning_rate_max: float
    dropout_values: tuple[float, ...]
    weight_decay_min: float
    weight_decay_max: float
    force_zero_dropout_for_one_layer: bool

    @classmethod
    def from_protocol(cls, protocol: FrozenProtocol) -> GRUSearchSpace:
        raw = protocol.raw["search"]["space"]
        hidden = raw["hidden_size"]
        return cls(
            hidden_sizes=tuple(
                range(int(hidden["min"]), int(hidden["max"]) + 1, int(hidden["step"]))
            ),
            num_layers=tuple(int(value) for value in raw["num_layers"]["values"]),
            learning_rate_min=float(raw["learning_rate"]["min"]),
            learning_rate_max=float(raw["learning_rate"]["max"]),
            dropout_values=tuple(float(value) for value in raw["dropout"]["values"]),
            weight_decay_min=float(raw["weight_decay"]["min"]),
            weight_decay_max=float(raw["weight_decay"]["max"]),
            force_zero_dropout_for_one_layer=bool(
                raw["dropout"]["force_zero_when_num_layers_is_one"]
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def make_candidate(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        learning_rate: float,
        dropout: float,
        weight_decay: float,
    ) -> GRUSearchCandidate:
        if self.force_zero_dropout_for_one_layer and num_layers == 1:
            dropout = 0.0
        candidate = GRUSearchCandidate(
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            learning_rate=float(learning_rate),
            dropout=float(dropout),
            weight_decay=float(weight_decay),
        )
        self.validate(candidate)
        return candidate

    def validate(self, candidate: GRUSearchCandidate) -> None:
        if candidate.hidden_size not in self.hidden_sizes:
            raise SearchValidationError("hidden_size is outside the frozen search space")
        if candidate.num_layers not in self.num_layers:
            raise SearchValidationError("num_layers is outside the frozen search space")
        if not self.learning_rate_min <= candidate.learning_rate <= self.learning_rate_max:
            raise SearchValidationError("learning_rate is outside the frozen search space")
        if candidate.dropout not in self.dropout_values:
            raise SearchValidationError("dropout is outside the frozen search space")
        if (
            self.force_zero_dropout_for_one_layer
            and candidate.num_layers == 1
            and candidate.dropout != 0.0
        ):
            raise SearchValidationError("A one-layer GRU must have zero recurrent dropout")
        if not self.weight_decay_min <= candidate.weight_decay <= self.weight_decay_max:
            raise SearchValidationError("weight_decay is outside the frozen search space")

    def sample(self, rng: np.random.Generator) -> GRUSearchCandidate:
        num_layers = int(rng.choice(self.num_layers))
        dropout = (
            0.0
            if self.force_zero_dropout_for_one_layer and num_layers == 1
            else float(rng.choice(self.dropout_values))
        )
        return self.make_candidate(
            hidden_size=int(rng.choice(self.hidden_sizes)),
            num_layers=num_layers,
            learning_rate=float(
                np.exp(
                    rng.uniform(
                        np.log(self.learning_rate_min),
                        np.log(self.learning_rate_max),
                    )
                )
            ),
            dropout=dropout,
            weight_decay=float(
                np.exp(
                    rng.uniform(
                        np.log(self.weight_decay_min),
                        np.log(self.weight_decay_max),
                    )
                )
            ),
        )


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_index: int
    candidate_key: str
    candidate: GRUSearchCandidate
    training_seed: int
    selection_score: float
    balanced_accuracy: float
    trainable_parameters: int
    runtime_seconds: float
    peak_gpu_memory_bytes: int | None
    epochs_completed: int
    best_epoch: int
    best_validation_loss: float
    metrics: dict[str, Any]
    history_file: str
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["candidate"] = self.candidate.to_dict()
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CandidateEvaluation:
        return cls(
            candidate_index=int(values["candidate_index"]),
            candidate_key=str(values["candidate_key"]),
            candidate=GRUSearchCandidate.from_dict(values["candidate"]),
            training_seed=int(values["training_seed"]),
            selection_score=float(values["selection_score"]),
            balanced_accuracy=float(values["balanced_accuracy"]),
            trainable_parameters=int(values["trainable_parameters"]),
            runtime_seconds=float(values["runtime_seconds"]),
            peak_gpu_memory_bytes=(
                int(values["peak_gpu_memory_bytes"])
                if values.get("peak_gpu_memory_bytes") is not None
                else None
            ),
            epochs_completed=int(values["epochs_completed"]),
            best_epoch=int(values["best_epoch"]),
            best_validation_loss=float(values["best_validation_loss"]),
            metrics=dict(values["metrics"]),
            history_file=str(values["history_file"]),
            provenance=dict(values["provenance"]),
        )


def evaluation_rank(evaluation: CandidateEvaluation) -> tuple[float, float, int, float]:
    """Frozen tie-breakers, expressed so the lexicographic maximum is best."""

    return (
        evaluation.selection_score,
        evaluation.balanced_accuracy,
        -evaluation.trainable_parameters,
        -evaluation.runtime_seconds,
    )


@dataclass(frozen=True)
class ActiveSearchBudget:
    budget: int
    median_runtime_seconds: float
    report_path: Path
    report_sha256: str
    device_name: str


def _current_pilot_code_hashes() -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    return {
        name: file_sha256(package_dir / name)
        for name in (
            "data.py",
            "folds.py",
            "metrics.py",
            "model.py",
            "preprocessing.py",
            "protocol.py",
            "training.py",
            "windows.py",
            "pilot.py",
        )
    }


def load_active_gpu_budget(
    protocol: FrozenProtocol,
    project_dir: str | Path | None = None,
) -> ActiveSearchBudget:
    project = (
        Path(project_dir).resolve()
        if project_dir is not None
        else Path(__file__).resolve().parents[1]
    )
    report_path = (
        project
        / "outputs"
        / "runtime_pilot"
        / "official_gpu"
        / "runtime_pilot_summary.json"
    )
    digest_path = report_path.with_suffix(".sha256")
    if not report_path.is_file() or not digest_path.is_file():
        raise SearchValidationError("The completed GPU runtime-pilot report is missing")
    actual_digest = file_sha256(report_path)
    expected_digest = digest_path.read_text(encoding="ascii").strip().split()[0].lower()
    if actual_digest != expected_digest:
        raise SearchValidationError("GPU runtime-pilot report SHA-256 verification failed")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETE" or report.get("device") != "cuda":
        raise SearchValidationError("The active runtime pilot is not a completed CUDA run")
    if report.get("performance_metrics_used_to_choose_B") is not False:
        raise SearchValidationError("Runtime budget was not selected independently of performance")
    if report.get("dataset_aggregate_sha256") != protocol.raw["dataset"][
        "aggregate_well_file_sha256"
    ]:
        raise SearchValidationError("Runtime-pilot data fingerprint differs from the protocol")
    median_runtime = float(report["median_runtime_T_seconds"])
    budget = int(report["candidate_budget_B"])
    if budget != candidate_budget(median_runtime):
        raise SearchValidationError("Runtime-pilot budget does not match the frozen threshold rule")
    if report.get("code_sha256") != _current_pilot_code_hashes():
        raise SearchValidationError("Training code changed after the GPU runtime pilot")
    return ActiveSearchBudget(
        budget=budget,
        median_runtime_seconds=median_runtime,
        report_path=report_path,
        report_sha256=actual_digest,
        device_name=str(report["device_name"]),
    )


def _search_code_hashes() -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    return {
        name: file_sha256(package_dir / name)
        for name in (
            "data.py",
            "folds.py",
            "metrics.py",
            "model.py",
            "preprocessing.py",
            "protocol.py",
            "training.py",
            "windows.py",
            "search.py",
        )
    }


class PersistentCandidateEvaluator:
    def __init__(
        self,
        *,
        split: PreparedSplit,
        protocol: FrozenProtocol,
        space: GRUSearchSpace,
        training_config: TrainingConfig,
        training_seed: int,
        device: str,
        output_dir: Path,
    ) -> None:
        locked = protocol.locked_external_wells
        requested = set(split.train_wells).union(split.evaluation_wells)
        if requested.intersection(locked):
            raise SearchValidationError("A locked external well entered a search split")
        self.split = split
        self.protocol = protocol
        self.space = space
        self.training_config = training_config
        self.training_seed = training_seed
        self.device = device
        self.output_dir = output_dir
        self.log_path = output_dir / "candidate_evaluations.jsonl"
        self.history_dir = output_dir / "histories"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._existing = self._load_existing()
        self._by_key = {item.candidate_key: item for item in self._existing}
        self._replay_position = 0

    def _load_existing(self) -> list[CandidateEvaluation]:
        if not self.log_path.is_file():
            return []
        evaluations: list[CandidateEvaluation] = []
        with self.log_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    evaluation = CandidateEvaluation.from_dict(json.loads(line))
                except Exception as exc:
                    raise SearchValidationError(
                        f"Invalid candidate log at line {line_number}"
                    ) from exc
                expected_index = len(evaluations) + 1
                if evaluation.candidate_index != expected_index:
                    raise SearchValidationError("Candidate log indices are not contiguous")
                if evaluation.candidate.key != evaluation.candidate_key:
                    raise SearchValidationError("Candidate log key does not match its parameters")
                if any(item.candidate_key == evaluation.candidate_key for item in evaluations):
                    raise SearchValidationError("Candidate log contains a duplicate candidate")
                self.space.validate(evaluation.candidate)
                if evaluation.training_seed != self.training_seed:
                    raise SearchValidationError("Candidate log uses another training seed")
                if not all(
                    np.isfinite(value)
                    for value in (
                        evaluation.selection_score,
                        evaluation.balanced_accuracy,
                        evaluation.runtime_seconds,
                        evaluation.best_validation_loss,
                    )
                ):
                    raise SearchValidationError("Candidate log contains a non-finite result")
                history_path = (self.output_dir / evaluation.history_file).resolve()
                if self.output_dir not in history_path.parents or not history_path.is_file():
                    raise SearchValidationError("Candidate history file is missing or unsafe")
                evaluations.append(evaluation)
        return evaluations

    def evaluate(
        self,
        candidate: GRUSearchCandidate,
        provenance: dict[str, Any],
    ) -> CandidateEvaluation:
        self.space.validate(candidate)
        if self._replay_position < len(self._existing):
            expected = self._existing[self._replay_position]
            if candidate.key != expected.candidate_key:
                raise SearchValidationError(
                    "Deterministic search replay diverged from the saved candidate log"
                )
            self._replay_position += 1
            print(
                f"Replayed candidate {expected.candidate_index}: "
                f"score={expected.selection_score:.6f}"
            )
            return expected
        if candidate.key in self._by_key:
            raise SearchValidationError("Search algorithm requested a duplicate candidate")

        candidate_index = len(self._existing) + 1
        model_config = GRUModelConfig(
            input_size=len(self.protocol.feature_names),
            hidden_size=candidate.hidden_size,
            num_layers=candidate.num_layers,
            output_size=len(self.protocol.class_names),
            dropout=candidate.dropout,
        )
        candidate_training = replace(
            self.training_config,
            learning_rate=candidate.learning_rate,
            weight_decay=candidate.weight_decay,
        )
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        result = train_gru(
            self.split,
            model_config,
            candidate_training,
            seed=self.training_seed,
            device=self.device,
        )
        peak_gpu_memory = (
            int(torch.cuda.max_memory_allocated()) if result.device == "cuda" else None
        )
        history_path = self.history_dir / f"candidate_{candidate_index:03d}.csv"
        with history_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result.history[0].to_dict()))
            writer.writeheader()
            writer.writerows(record.to_dict() for record in result.history)

        metrics = result.best_metrics.to_dict()
        evaluation = CandidateEvaluation(
            candidate_index=candidate_index,
            candidate_key=candidate.key,
            candidate=candidate,
            training_seed=self.training_seed,
            selection_score=float(result.best_metrics.selection_score),
            balanced_accuracy=float(result.best_metrics.balanced_accuracy),
            trainable_parameters=result.model.trainable_parameters,
            runtime_seconds=float(result.runtime_seconds),
            peak_gpu_memory_bytes=peak_gpu_memory,
            epochs_completed=result.epochs_completed,
            best_epoch=result.best_epoch,
            best_validation_loss=result.best_validation_loss,
            metrics=metrics,
            history_file=str(history_path.relative_to(self.output_dir)),
            provenance=dict(provenance),
        )
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_json(evaluation.to_dict()) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._existing.append(evaluation)
        self._by_key[evaluation.candidate_key] = evaluation
        self._replay_position += 1
        print(
            f"Completed candidate {candidate_index}: "
            f"score={evaluation.selection_score:.6f}, "
            f"epochs={evaluation.epochs_completed}, "
            f"runtime={evaluation.runtime_seconds:.2f} s"
        )
        del result
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        return evaluation

    def assert_replay_complete(self) -> None:
        if self._replay_position != len(self._existing):
            raise SearchValidationError("Saved candidates were not fully replayed")


EvaluationFunction = Callable[
    [GRUSearchCandidate, dict[str, Any]], CandidateEvaluation
]


def _unique_random_candidate(
    space: GRUSearchSpace,
    rng: np.random.Generator,
    seen: set[str],
    *,
    max_attempts: int = 10_000,
) -> GRUSearchCandidate:
    for _ in range(max_attempts):
        candidate = space.sample(rng)
        if candidate.key not in seen:
            return candidate
    raise SearchValidationError("Could not generate another unique random candidate")


def run_random_algorithm(
    space: GRUSearchSpace,
    *,
    budget: int,
    seed: int,
    evaluate: EvaluationFunction,
) -> list[CandidateEvaluation]:
    if budget < 1:
        raise SearchValidationError("Search budget must be positive")
    rng = np.random.default_rng(seed)
    seen: set[str] = set()
    evaluations: list[CandidateEvaluation] = []
    while len(evaluations) < budget:
        candidate = _unique_random_candidate(space, rng, seen)
        seen.add(candidate.key)
        evaluations.append(
            evaluate(candidate, {"method": "random", "draw": len(evaluations) + 1})
        )
    return evaluations


def _tpe_candidate(trial: Any, space: GRUSearchSpace) -> GRUSearchCandidate:
    num_layers = trial.suggest_categorical("num_layers", list(space.num_layers))
    dropout = (
        0.0
        if space.force_zero_dropout_for_one_layer and num_layers == 1
        else trial.suggest_categorical("dropout", list(space.dropout_values))
    )
    return space.make_candidate(
        hidden_size=trial.suggest_int(
            "hidden_size",
            min(space.hidden_sizes),
            max(space.hidden_sizes),
            step=space.hidden_sizes[1] - space.hidden_sizes[0],
        ),
        num_layers=int(num_layers),
        learning_rate=trial.suggest_float(
            "learning_rate",
            space.learning_rate_min,
            space.learning_rate_max,
            log=True,
        ),
        dropout=float(dropout),
        weight_decay=trial.suggest_float(
            "weight_decay",
            space.weight_decay_min,
            space.weight_decay_max,
            log=True,
        ),
    )


def run_tpe_algorithm(
    space: GRUSearchSpace,
    *,
    budget: int,
    seed: int,
    random_warmup_candidates: int,
    evaluate: EvaluationFunction,
) -> list[CandidateEvaluation]:
    if budget < 1:
        raise SearchValidationError("Search budget must be positive")
    try:
        import optuna
    except ImportError as exc:
        raise SearchValidationError("Optuna is required for TPE search") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=random_warmup_candidates,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    seen: set[str] = set()
    evaluations: list[CandidateEvaluation] = []
    duplicate_trials = 0
    while len(evaluations) < budget:
        trial = study.ask()
        candidate = _tpe_candidate(trial, space)
        if candidate.key in seen:
            duplicate_trials += 1
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            if duplicate_trials > 1_000:
                raise SearchValidationError("TPE repeatedly generated duplicate candidates")
            continue
        seen.add(candidate.key)
        evaluation = evaluate(
            candidate,
            {
                "method": "tpe",
                "trial_number": int(trial.number),
                "unique_candidate": len(evaluations) + 1,
            },
        )
        study.tell(trial, evaluation.selection_score)
        evaluations.append(evaluation)
    return evaluations


def _tournament(
    population: list[CandidateEvaluation],
    rng: np.random.Generator,
    tournament_size: int,
) -> GRUSearchCandidate:
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    winner = max((population[int(index)] for index in indices), key=evaluation_rank)
    return winner.candidate


def _ga_child(
    first: GRUSearchCandidate,
    second: GRUSearchCandidate,
    space: GRUSearchSpace,
    rng: np.random.Generator,
    *,
    crossover_probability: float,
    mutation_probability: float,
) -> GRUSearchCandidate:
    names = ("hidden_size", "num_layers", "learning_rate", "dropout", "weight_decay")
    if rng.random() < crossover_probability:
        genes = {
            name: getattr(first if rng.random() < 0.5 else second, name)
            for name in names
        }
    else:
        genes = first.to_dict()

    if rng.random() < mutation_probability:
        genes["hidden_size"] = int(rng.choice(space.hidden_sizes))
    if rng.random() < mutation_probability:
        genes["num_layers"] = int(rng.choice(space.num_layers))
    if rng.random() < mutation_probability:
        genes["learning_rate"] = float(
            np.exp(
                rng.uniform(
                    np.log(space.learning_rate_min),
                    np.log(space.learning_rate_max),
                )
            )
        )
    if rng.random() < mutation_probability:
        genes["dropout"] = float(rng.choice(space.dropout_values))
    if rng.random() < mutation_probability:
        genes["weight_decay"] = float(
            np.exp(
                rng.uniform(
                    np.log(space.weight_decay_min),
                    np.log(space.weight_decay_max),
                )
            )
        )
    return space.make_candidate(**genes)


def run_genetic_algorithm(
    space: GRUSearchSpace,
    *,
    budget: int,
    seed: int,
    population_size: int,
    tournament_size: int,
    crossover_probability: float,
    mutation_probability: float,
    elites: int,
    evaluate: EvaluationFunction,
) -> list[CandidateEvaluation]:
    if budget < population_size:
        raise SearchValidationError("GA budget must cover its initial population")
    if not 1 <= elites < population_size:
        raise SearchValidationError("GA elites must be between 1 and population_size - 1")
    if not 1 <= tournament_size <= population_size:
        raise SearchValidationError("Invalid GA tournament size")
    rng = np.random.default_rng(seed)
    seen: set[str] = set()
    evaluations: list[CandidateEvaluation] = []
    population: list[CandidateEvaluation] = []
    for slot in range(1, population_size + 1):
        candidate = _unique_random_candidate(space, rng, seen)
        seen.add(candidate.key)
        evaluation = evaluate(
            candidate,
            {"method": "genetic_algorithm", "generation": 0, "population_slot": slot},
        )
        evaluations.append(evaluation)
        population.append(evaluation)

    generation = 1
    while len(evaluations) < budget:
        elite_evaluations = sorted(population, key=evaluation_rank, reverse=True)[:elites]
        remaining = min(population_size - elites, budget - len(evaluations))
        children: list[GRUSearchCandidate] = []
        attempts = 0
        while len(children) < remaining:
            attempts += 1
            first = _tournament(population, rng, tournament_size)
            second = _tournament(population, rng, tournament_size)
            child = _ga_child(
                first,
                second,
                space,
                rng,
                crossover_probability=crossover_probability,
                mutation_probability=mutation_probability,
            )
            pending_keys = {candidate.key for candidate in children}
            if child.key in seen or child.key in pending_keys:
                if attempts > 2_000:
                    child = _unique_random_candidate(space, rng, seen.union(pending_keys))
                    attempts = 0
                else:
                    continue
            children.append(child)

        child_evaluations: list[CandidateEvaluation] = []
        for slot, child in enumerate(children, start=elites + 1):
            seen.add(child.key)
            evaluation = evaluate(
                child,
                {
                    "method": "genetic_algorithm",
                    "generation": generation,
                    "population_slot": slot,
                },
            )
            evaluations.append(evaluation)
            child_evaluations.append(evaluation)
        population = elite_evaluations + child_evaluations
        generation += 1
    return evaluations


def _normalise_method(method: str) -> str:
    normalized = _METHOD_ALIASES.get(method.strip().lower(), method.strip().lower())
    if normalized not in SEARCH_METHODS:
        raise SearchValidationError(f"Unknown search method: {method}")
    return normalized


def _build_manifest(
    *,
    protocol: FrozenProtocol,
    split: PreparedSplit,
    method: str,
    budget: int,
    search_seed: int,
    training_seed: int,
    training_config: TrainingConfig,
    space: GRUSearchSpace,
    device: str,
    run_kind: str,
    budget_source: ActiveSearchBudget | None,
) -> dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        raise SearchValidationError("CUDA search requested but CUDA is unavailable")
    ga = protocol.raw["search"]["ga"]
    return {
        "schema_version": "1.0",
        "run_kind": run_kind,
        "method": method,
        "protocol_id": protocol.raw["protocol"]["id"],
        "protocol_version": protocol.raw["protocol"]["version"],
        "dataset_aggregate_sha256": protocol.raw["dataset"][
            "aggregate_well_file_sha256"
        ],
        "split_name": split.split_name,
        "training_wells": list(split.train_wells),
        "validation_wells": list(split.evaluation_wells),
        "training_windows": len(split.train.y),
        "validation_windows": len(split.evaluation.y),
        "features": list(split.train.feature_names),
        "window_length": split.train.window_length,
        "budget_unique_candidates": budget,
        "search_seed": search_seed,
        "candidate_training_seed": training_seed,
        "search_space": space.to_dict(),
        "training_config": training_config.to_dict(),
        "selection_metric": protocol.raw["training"]["selection_metric"],
        "tie_breakers": list(protocol.raw["search"]["tie_breakers"]),
        "tpe_random_warmup_candidates": int(
            protocol.raw["search"]["tpe_random_warmup_candidates"]
        ),
        "ga": {
            "population": int(ga["population"]),
            "tournament_size": int(ga["tournament_size"]),
            "crossover_probability": float(ga["crossover_probability"]),
            "per_gene_mutation_probability": float(
                ga["per_gene_mutation_probability"]
            ),
            "elites": int(ga["elites"]),
            "crossover": "uniform_per_gene",
            "mutation": "independent_random_reset_per_gene",
        },
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "python_executable": os.path.realpath(os.sys.executable),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "optuna_version": _package_version("optuna"),
        "code_sha256": _search_code_hashes(),
        "budget_source": (
            {
                "runtime_pilot_report": str(budget_source.report_path),
                "runtime_pilot_sha256": budget_source.report_sha256,
                "median_runtime_T_seconds": budget_source.median_runtime_seconds,
                "candidate_budget_B": budget_source.budget,
            }
            if budget_source is not None
            else None
        ),
        "locked_external_wells_not_read": sorted(protocol.locked_external_wells),
    }


def _write_or_validate_manifest(output_dir: Path, manifest: dict[str, Any]) -> str:
    manifest_path = output_dir / "search_manifest.json"
    digest_path = manifest_path.with_suffix(".sha256")
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2)
    if manifest_path.is_file():
        if not digest_path.is_file():
            raise SearchValidationError("Existing search manifest digest is missing")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _canonical_json(existing) != _canonical_json(manifest):
            raise SearchValidationError(
                f"Existing search manifest conflicts with this run: {manifest_path}"
            )
        actual_digest = file_sha256(manifest_path)
        expected_digest = digest_path.read_text(encoding="ascii").strip().split()[0]
        if actual_digest != expected_digest:
            raise SearchValidationError("Existing search manifest failed SHA-256 verification")
    else:
        stale_artifacts = (
            output_dir / "candidate_evaluations.jsonl",
            output_dir / "search_summary.json",
            output_dir / "search_summary.sha256",
        )
        if any(path.exists() for path in stale_artifacts):
            raise SearchValidationError("Search artifacts exist without their manifest")
        _atomic_write_text(manifest_path, serialized)
        actual_digest = file_sha256(manifest_path)
        _atomic_write_text(
            digest_path, actual_digest + "\n", encoding="ascii"
        )
    return actual_digest


def _load_completed_summary(
    output_dir: Path,
    *,
    manifest_sha256: str,
    budget: int,
) -> dict[str, Any] | None:
    summary_path = output_dir / "search_summary.json"
    digest_path = summary_path.with_suffix(".sha256")
    if not summary_path.is_file() and not digest_path.is_file():
        return None
    if not summary_path.is_file() or not digest_path.is_file():
        raise SearchValidationError("Search summary or its digest is missing")
    actual_digest = file_sha256(summary_path)
    expected_digest = digest_path.read_text(encoding="ascii").strip().split()[0]
    if actual_digest != expected_digest:
        raise SearchValidationError("Completed search summary failed SHA-256 verification")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "COMPLETE"
        or summary.get("manifest_sha256") != manifest_sha256
        or int(summary.get("unique_candidates_evaluated", -1)) != budget
    ):
        raise SearchValidationError("Completed search summary conflicts with its manifest")
    return summary


def run_gru_search(
    *,
    protocol: FrozenProtocol,
    split: PreparedSplit,
    method: str,
    output_dir: str | Path,
    budget: int,
    search_seed: int,
    training_seed: int,
    device: str = "cuda",
    training_config: TrainingConfig | None = None,
    run_kind: str = "formal_inner_search",
    budget_source: ActiveSearchBudget | None = None,
) -> dict[str, Any]:
    normalized_method = _normalise_method(method)
    resolved_device = resolve_device(device).type
    if budget_source is not None and budget != budget_source.budget:
        raise SearchValidationError("Requested budget differs from the active GPU budget")
    if budget_source is not None and resolved_device != "cuda":
        raise SearchValidationError("The GPU-derived formal budget requires CUDA execution")
    if run_kind == "formal_inner_search" and budget_source is None:
        raise SearchValidationError("Formal search requires a verified GPU budget source")
    if budget < 1:
        raise SearchValidationError("Search budget must be positive")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    space = GRUSearchSpace.from_protocol(protocol)
    if training_config is None:
        _, training_config = protocol_training_configs()
    manifest = _build_manifest(
        protocol=protocol,
        split=split,
        method=normalized_method,
        budget=budget,
        search_seed=search_seed,
        training_seed=training_seed,
        training_config=training_config,
        space=space,
        device=resolved_device,
        run_kind=run_kind,
        budget_source=budget_source,
    )
    manifest_sha256 = _write_or_validate_manifest(output, manifest)
    completed = _load_completed_summary(
        output, manifest_sha256=manifest_sha256, budget=budget
    )
    if completed is not None:
        return completed

    evaluator = PersistentCandidateEvaluator(
        split=split,
        protocol=protocol,
        space=space,
        training_config=training_config,
        training_seed=training_seed,
        device=resolved_device,
        output_dir=output,
    )
    started = time.perf_counter()
    if normalized_method == "random":
        evaluations = run_random_algorithm(
            space, budget=budget, seed=search_seed, evaluate=evaluator.evaluate
        )
    elif normalized_method == "tpe":
        evaluations = run_tpe_algorithm(
            space,
            budget=budget,
            seed=search_seed,
            random_warmup_candidates=int(
                protocol.raw["search"]["tpe_random_warmup_candidates"]
            ),
            evaluate=evaluator.evaluate,
        )
    else:
        ga = protocol.raw["search"]["ga"]
        evaluations = run_genetic_algorithm(
            space,
            budget=budget,
            seed=search_seed,
            population_size=int(ga["population"]),
            tournament_size=int(ga["tournament_size"]),
            crossover_probability=float(ga["crossover_probability"]),
            mutation_probability=float(ga["per_gene_mutation_probability"]),
            elites=int(ga["elites"]),
            evaluate=evaluator.evaluate,
        )
    evaluator.assert_replay_complete()
    if len(evaluations) != budget or len({item.candidate_key for item in evaluations}) != budget:
        raise SearchValidationError("Search did not evaluate the required unique-candidate budget")
    best = max(evaluations, key=evaluation_rank)
    summary = {
        "status": "COMPLETE",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "manifest_sha256": manifest_sha256,
        "method": normalized_method,
        "unique_candidates_evaluated": len(evaluations),
        "search_training_runtime_seconds": float(
            sum(item.runtime_seconds for item in evaluations)
        ),
        "orchestration_runtime_this_invocation_seconds": time.perf_counter() - started,
        "best_candidate_index": best.candidate_index,
        "best_candidate": best.candidate.to_dict(),
        "best_selection_score": best.selection_score,
        "best_balanced_accuracy": best.balanced_accuracy,
        "best_trainable_parameters": best.trainable_parameters,
        "best_runtime_seconds": best.runtime_seconds,
        "candidate_log": evaluator.log_path.name,
        "candidate_results": [item.to_dict() for item in evaluations],
        "locked_external_wells_not_read": sorted(protocol.locked_external_wells),
    }
    summary_path = output / "search_summary.json"
    _atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    _atomic_write_text(
        summary_path.with_suffix(".sha256"),
        file_sha256(summary_path) + "\n",
        encoding="ascii",
    )
    return summary
