"""Run resumable, equal-budget optimizer searches under the full pipeline.

Default execution runs GA, random search, and TPE for all three frozen search
seeds. Use ``--preflight`` to validate data preparation without training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = PROJECT_DIR / "experiment_protocol_v8_optimizer_repeats"
PROTOCOL_PATH = PROTOCOL_DIR / "optimizer_repeat_protocol_v8.json"
SOURCE_PROTOCOL_DIR = PROJECT_DIR / "experiment_protocol_v5_random_center"
OUTPUT_ROOT = PROJECT_DIR / "outputs" / "optimizer_repeats_v1"
SEARCH_ROOT = OUTPUT_ROOT / "searches"
METHODS = ("genetic_algorithm", "random_search", "tpe")

sys.path.insert(0, str(PROJECT_DIR))

from gagru.local_recurrent import fit_local_recurrent_with_validation  # noqa: E402
from gagru.random_center import (  # noqa: E402
    build_random_center_windows,
    impute_from_training_windows,
    remap_labels,
)
from gagru.random_center_ablation import apply_imbalance_strategy  # noqa: E402
from gagru.residual_bigru import prepare_per_well_inputs  # noqa: E402
from gagru.search import GRUSearchCandidate, GRUSearchSpace  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_atomic(path: Path, value: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding=encoding)
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: object) -> None:
    write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def search_space(raw: dict[str, Any]) -> GRUSearchSpace:
    return GRUSearchSpace(
        hidden_sizes=tuple(int(value) for value in raw["hidden_sizes"]),
        num_layers=tuple(int(value) for value in raw["num_layers"]),
        learning_rate_min=float(raw["learning_rate_min"]),
        learning_rate_max=float(raw["learning_rate_max"]),
        dropout_values=tuple(float(value) for value in raw["dropout_values"]),
        weight_decay_min=float(raw["weight_decay_min"]),
        weight_decay_max=float(raw["weight_decay_max"]),
        force_zero_dropout_for_one_layer=bool(
            raw["force_zero_dropout_for_one_layer"]
        ),
    )


def rank(record: dict[str, Any]) -> tuple[float, float, float, int, float]:
    return (
        float(record["mean_per_well_macro_f1"]),
        float(record["mean_per_well_balanced_accuracy"]),
        float(record["mean_per_well_accuracy"]),
        -int(record["total_parameters_across_six_models"]),
        -float(record["charged_training_runtime_seconds"]),
    )


def assignments(well_id: str, split_seed: int) -> pd.DataFrame:
    parts = []
    for split in ("train", "validation"):
        path = (
            SOURCE_PROTOCOL_DIR
            / "center_assignments"
            / f"seed_{split_seed}"
            / well_id
            / f"{well_id}_{split}_centers.csv"
        )
        parts.append(pd.read_csv(path, encoding="utf-8-sig"))
    return pd.concat(parts, ignore_index=True)


def prepare_tasks(
    protocol: dict[str, Any], source_protocol: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    features = tuple(str(value) for value in protocol["features"])
    split_seed = int(protocol["split_seed"])
    window_length = int(protocol["window_length"])
    tasks: dict[str, dict[str, Any]] = {}
    audit_rows = []
    for well_id in protocol["wells"]:
        source_record = source_protocol["well_protocols"][well_id]
        source_path = SOURCE_PROTOCOL_DIR / str(source_record["source_snapshot"])
        relative = str(source_path.relative_to(PROJECT_DIR))
        expected_hash = protocol["source_snapshot_sha256"][relative]
        if sha256_file(source_path) != expected_hash:
            raise RuntimeError(f"Source snapshot hash changed for {well_id}")
        frame = pd.read_csv(source_path, encoding="utf-8-sig")
        development_assignment = assignments(well_id, split_seed)
        windows = build_random_center_windows(
            frame,
            development_assignment,
            features,
            window_length,
            requested_splits=("train", "validation"),
        )
        windows, medians = impute_from_training_windows(windows)
        global_classes = tuple(
            int(value) for value in source_record["included_classes"]
        )
        y_train, mapping = remap_labels(windows["train"], global_classes)
        y_validation, validation_mapping = remap_labels(
            windows["validation"], global_classes
        )
        if mapping != validation_mapping:
            raise RuntimeError(f"Local class mapping differs for {well_id}")
        X_train, X_validation = prepare_per_well_inputs(
            windows["train"],
            windows["validation"],
            representation=str(protocol["representation"]),
        )
        balance_seed = int(protocol["balance_seed_by_well"][well_id])
        balanced = apply_imbalance_strategy(
            X_train,
            y_train,
            str(protocol["imbalance_strategy"]),
            random_state=balance_seed,
        )
        tasks[well_id] = {
            "X_train": balanced.X,
            "y_train": balanced.y,
            "X_validation": X_validation,
            "y_validation": y_validation,
            "training_seed": int(protocol["training_seed_by_well"][well_id]),
            "global_classes": global_classes,
        }
        audit_rows.append(
            {
                "well_id": well_id,
                "source_snapshot_sha256": expected_hash,
                "train_centers_before_balance": int(len(y_train)),
                "train_windows_after_balance": int(len(balanced.y)),
                "validation_centers": int(len(y_validation)),
                "classes": list(global_classes),
                "input_channels": int(X_train.shape[2]),
                "window_length": int(X_train.shape[1]),
                "training_seed": tasks[well_id]["training_seed"],
                "balance_seed": balance_seed,
                "imputation_medians": medians,
                "balance_audit": balanced.audit,
            }
        )
    audit = {
        "status": "PASS",
        "test_assignment_files_opened": False,
        "test_metrics_read": False,
        "representation": protocol["representation"],
        "imbalance_strategy": protocol["imbalance_strategy"],
        "wells": audit_rows,
    }
    return tasks, audit


def unique_random_candidate(
    space: GRUSearchSpace,
    rng: np.random.Generator,
    seen: set[str],
) -> GRUSearchCandidate:
    for _ in range(10_000):
        candidate = space.sample(rng)
        if candidate.key not in seen:
            return candidate
    raise RuntimeError("Could not generate a unique search candidate")


def tournament(
    population: list[dict[str, Any]],
    rng: np.random.Generator,
    size: int,
) -> GRUSearchCandidate:
    indices = rng.choice(len(population), size=size, replace=False)
    winner = max((population[int(index)] for index in indices), key=rank)
    return GRUSearchCandidate.from_dict(winner["candidate"])


def ga_child(
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


def run_ga(
    initial: list[GRUSearchCandidate],
    *,
    space: GRUSearchSpace,
    budget: int,
    seed: int,
    settings: dict[str, Any],
    evaluate: Callable[[GRUSearchCandidate, dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    population_size = int(settings["population_size"])
    if len(initial) != population_size:
        raise RuntimeError("Shared initial candidates must fill the GA population")
    rng = np.random.default_rng(seed)
    seen = {candidate.key for candidate in initial}
    evaluations = []
    population = []
    for slot, candidate in enumerate(initial, start=1):
        result = evaluate(
            candidate,
            {
                "method": "genetic_algorithm",
                "generation": 0,
                "population_slot": slot,
                "shared_initial": True,
            },
        )
        evaluations.append(result)
        population.append(result)
    generation = 1
    while len(evaluations) < budget:
        elites = sorted(population, key=rank, reverse=True)[
            : int(settings["elites"])
        ]
        remaining = min(
            population_size - int(settings["elites"]),
            budget - len(evaluations),
        )
        children: list[GRUSearchCandidate] = []
        attempts = 0
        while len(children) < remaining:
            attempts += 1
            child = ga_child(
                tournament(
                    population, rng, int(settings["tournament_size"])
                ),
                tournament(
                    population, rng, int(settings["tournament_size"])
                ),
                space,
                rng,
                crossover_probability=float(settings["crossover_probability"]),
                mutation_probability=float(
                    settings["per_gene_mutation_probability"]
                ),
            )
            pending = {value.key for value in children}
            if child.key in seen or child.key in pending:
                if attempts > 2_000:
                    child = unique_random_candidate(space, rng, seen.union(pending))
                    attempts = 0
                else:
                    continue
            children.append(child)
        child_results = []
        for slot, child in enumerate(children, start=int(settings["elites"]) + 1):
            seen.add(child.key)
            result = evaluate(
                child,
                {
                    "method": "genetic_algorithm",
                    "generation": generation,
                    "population_slot": slot,
                    "shared_initial": False,
                },
            )
            evaluations.append(result)
            child_results.append(result)
        population = elites + child_results
        generation += 1
    return evaluations


def run_random(
    initial: list[GRUSearchCandidate],
    *,
    space: GRUSearchSpace,
    budget: int,
    seed: int,
    evaluate: Callable[[GRUSearchCandidate, dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    seen = {candidate.key for candidate in initial}
    evaluations = []
    for draw, candidate in enumerate(initial, start=1):
        evaluations.append(
            evaluate(
                candidate,
                {
                    "method": "random_search",
                    "draw": draw,
                    "shared_initial": True,
                },
            )
        )
    while len(evaluations) < budget:
        candidate = unique_random_candidate(space, rng, seen)
        seen.add(candidate.key)
        evaluations.append(
            evaluate(
                candidate,
                {
                    "method": "random_search",
                    "draw": len(evaluations) + 1,
                    "shared_initial": False,
                },
            )
        )
    return evaluations


def tpe_candidate(trial: Any, space: GRUSearchSpace) -> GRUSearchCandidate:
    num_layers = int(trial.suggest_categorical("num_layers", list(space.num_layers)))
    dropout = (
        0.0
        if space.force_zero_dropout_for_one_layer and num_layers == 1
        else float(trial.suggest_categorical("dropout", list(space.dropout_values)))
    )
    return space.make_candidate(
        hidden_size=trial.suggest_int(
            "hidden_size",
            min(space.hidden_sizes),
            max(space.hidden_sizes),
            step=space.hidden_sizes[1] - space.hidden_sizes[0],
        ),
        num_layers=num_layers,
        learning_rate=trial.suggest_float(
            "learning_rate",
            space.learning_rate_min,
            space.learning_rate_max,
            log=True,
        ),
        dropout=dropout,
        weight_decay=trial.suggest_float(
            "weight_decay",
            space.weight_decay_min,
            space.weight_decay_max,
            log=True,
        ),
    )


def run_tpe(
    initial: list[GRUSearchCandidate],
    *,
    space: GRUSearchSpace,
    budget: int,
    seed: int,
    startup_candidates: int,
    evaluate: Callable[[GRUSearchCandidate, dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=startup_candidates,
        ),
    )
    for candidate in initial:
        study.enqueue_trial(candidate.to_dict())
    evaluations = []
    seen: set[str] = set()
    duplicate_trials = 0
    while len(evaluations) < budget:
        trial = study.ask()
        candidate = tpe_candidate(trial, space)
        if candidate.key in seen:
            duplicate_trials += 1
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            if duplicate_trials > 1_000:
                raise RuntimeError("TPE repeatedly generated duplicate candidates")
            continue
        shared_initial = len(evaluations) < len(initial)
        if shared_initial and candidate.key != initial[len(evaluations)].key:
            raise RuntimeError("TPE did not consume the frozen shared initial design")
        seen.add(candidate.key)
        result = evaluate(
            candidate,
            {
                "method": "tpe",
                "trial_number": int(trial.number),
                "unique_candidate": len(evaluations) + 1,
                "shared_initial": shared_initial,
            },
        )
        study.tell(trial, float(result["mean_per_well_macro_f1"]))
        evaluations.append(result)
    return evaluations


def all_existing_logs() -> list[Path]:
    if not SEARCH_ROOT.is_dir():
        return []
    return sorted(SEARCH_ROOT.rglob("candidate_evaluations.jsonl"))


def build_cache() -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for path in all_existing_logs():
        for record in read_jsonl(path):
            key = str(record["candidate_key"])
            if key in cache:
                first = cache[key]
                for metric in (
                    "mean_per_well_macro_f1",
                    "mean_per_well_accuracy",
                    "mean_per_well_balanced_accuracy",
                ):
                    if not np.isclose(
                        float(first[metric]), float(record[metric]), atol=1e-12
                    ):
                        raise RuntimeError(f"Cached metric conflict for {key}")
            elif not bool(record["provenance"].get("evaluation_reused", False)):
                cache[key] = record
    return cache


def search_directory(repeat_id: int, method: str) -> Path:
    return SEARCH_ROOT / f"repeat_{repeat_id}" / method


def run_one_search(
    protocol: dict[str, Any],
    repeat: dict[str, Any],
    method: str,
    tasks: dict[str, dict[str, Any]],
    device: torch.device,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    repeat_id = int(repeat["repeat_id"])
    directory = search_directory(repeat_id, method)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "candidate_evaluations.jsonl"
    manifest_path = directory / "search_manifest.json"
    summary_path = directory / "search_summary.json"
    budget = int(protocol["candidate_budget_per_method_per_repeat"])
    space = search_space(protocol["search_space"])
    initial = [
        GRUSearchCandidate.from_dict(value)
        for value in repeat["shared_initial_candidates"]
    ]
    if len({candidate.key for candidate in initial}) != len(initial):
        raise RuntimeError("Frozen initial candidates are not unique")
    for candidate in initial:
        space.validate(candidate)

    manifest = {
        "status": "FROZEN_BEFORE_SEARCH",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "repeat_id": repeat_id,
        "base_search_seed": int(repeat["base_search_seed"]),
        "method": method,
        "method_search_seed": int(repeat["method_search_seed"]),
        "budget": budget,
        "shared_initial_candidates": [value.to_dict() for value in initial],
        "shared_initial_candidates_count_toward_budget": True,
        "search_space": protocol["search_space"],
        "ga": protocol["ga"] if method == "genetic_algorithm" else None,
        "tpe": protocol["tpe"] if method == "tpe" else None,
        "features": protocol["features"],
        "window_length": protocol["window_length"],
        "representation": protocol["representation"],
        "imbalance_strategy": protocol["imbalance_strategy"],
        "selection_metric": protocol["selection_metric"],
        "wells": protocol["wells"],
        "test_assignment_files_opened": False,
        "test_metrics_read": False,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
    }
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError(f"Existing manifest conflicts: {manifest_path}")
    else:
        write_json_atomic(manifest_path, manifest)

    existing = read_jsonl(log_path)
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "COMPLETE" or len(existing) != budget:
            raise RuntimeError(f"Invalid completed search: {directory}")
        print(
            f"Skipped completed repeat {repeat_id} {method}: "
            f"best macro-F1={summary['best_validation_macro_f1']:.4f}",
            flush=True,
        )
        return summary

    position = 0
    invocation_started = time.perf_counter()

    def evaluate(
        candidate: GRUSearchCandidate, provenance: dict[str, Any]
    ) -> dict[str, Any]:
        nonlocal position
        candidate_index = position + 1
        if position < len(existing):
            saved = existing[position]
            if (
                int(saved["candidate_index"]) != candidate_index
                or str(saved["candidate_key"]) != candidate.key
            ):
                raise RuntimeError(
                    f"Deterministic replay diverged in repeat {repeat_id}/{method}"
                )
            position += 1
            print(
                f"Replayed repeat {repeat_id} {method} "
                f"candidate {candidate_index}/{budget}: "
                f"macro-F1={saved['mean_per_well_macro_f1']:.4f}",
                flush=True,
            )
            return saved

        cached = cache.get(candidate.key)
        if cached is not None:
            record = {
                "candidate_index": candidate_index,
                "candidate_key": candidate.key,
                "candidate": candidate.to_dict(),
                "mean_per_well_macro_f1": float(
                    cached["mean_per_well_macro_f1"]
                ),
                "mean_per_well_accuracy": float(
                    cached["mean_per_well_accuracy"]
                ),
                "pooled_validation_accuracy": float(
                    cached["pooled_validation_accuracy"]
                ),
                "mean_per_well_balanced_accuracy": float(
                    cached["mean_per_well_balanced_accuracy"]
                ),
                "total_parameters_across_six_models": int(
                    cached["total_parameters_across_six_models"]
                ),
                "charged_training_runtime_seconds": float(
                    cached["charged_training_runtime_seconds"]
                ),
                "actual_training_runtime_seconds": 0.0,
                "per_well": copy.deepcopy(cached["per_well"]),
                "provenance": {
                    **provenance,
                    "evaluation_reused": True,
                    "reused_candidate_origin": {
                        "repeat_id": cached["provenance"].get("repeat_id"),
                        "method": cached["provenance"].get("search_method"),
                        "candidate_index": cached.get("candidate_index"),
                    },
                },
            }
            append_jsonl(log_path, record)
            existing.append(record)
            position += 1
            print(
                f"Reused repeat {repeat_id} {method} "
                f"candidate {candidate_index}/{budget}: "
                f"macro-F1={record['mean_per_well_macro_f1']:.4f}",
                flush=True,
            )
            return record

        per_well = []
        total_runtime = 0.0
        total_parameters = 0
        for well_id in protocol["wells"]:
            task = tasks[well_id]
            torch.cuda.empty_cache()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            fit = fit_local_recurrent_with_validation(
                "gru",
                candidate,
                task["X_train"],
                task["y_train"],
                task["X_validation"],
                task["y_validation"],
                device=device,
                seed=int(task["training_seed"]),
                batch_size=int(protocol["batch_size"]),
                max_epochs=int(protocol["maximum_epochs"]),
                patience=int(protocol["early_stopping_patience"]),
            )
            history_path = (
                directory
                / "histories"
                / f"candidate_{candidate_index:03d}"
                / f"{well_id}.csv"
            )
            write_csv_atomic(history_path, pd.DataFrame(fit.history))
            peak_memory = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            )
            per_well.append(
                {
                    "well_id": well_id,
                    "validation_samples": int(len(task["y_validation"])),
                    "macro_f1": fit.validation_macro_f1,
                    "accuracy": fit.validation_accuracy,
                    "balanced_accuracy": fit.validation_balanced_accuracy,
                    "best_epoch": fit.best_epoch,
                    "epochs_completed": fit.epochs_completed,
                    "runtime_seconds": fit.runtime_seconds,
                    "trainable_parameters": fit.trainable_parameters,
                    "peak_gpu_memory_bytes": peak_memory,
                    "history_file": str(history_path.relative_to(PROJECT_DIR)),
                    "history_sha256": sha256_file(history_path),
                }
            )
            total_runtime += fit.runtime_seconds
            total_parameters += fit.trainable_parameters
            del fit
            torch.cuda.empty_cache()
        per_well_frame = pd.DataFrame(per_well)
        record = {
            "candidate_index": candidate_index,
            "candidate_key": candidate.key,
            "candidate": candidate.to_dict(),
            "mean_per_well_macro_f1": float(per_well_frame["macro_f1"].mean()),
            "mean_per_well_accuracy": float(per_well_frame["accuracy"].mean()),
            "pooled_validation_accuracy": float(
                np.average(
                    per_well_frame["accuracy"],
                    weights=per_well_frame["validation_samples"],
                )
            ),
            "mean_per_well_balanced_accuracy": float(
                per_well_frame["balanced_accuracy"].mean()
            ),
            "total_parameters_across_six_models": int(total_parameters),
            "charged_training_runtime_seconds": float(total_runtime),
            "actual_training_runtime_seconds": float(total_runtime),
            "per_well": per_well,
            "provenance": {
                **provenance,
                "evaluation_reused": False,
                "repeat_id": repeat_id,
                "search_method": method,
            },
        }
        append_jsonl(log_path, record)
        existing.append(record)
        cache[candidate.key] = record
        position += 1
        print(
            f"Completed repeat {repeat_id} {method} "
            f"candidate {candidate_index}/{budget}: "
            f"macro-F1={record['mean_per_well_macro_f1']:.4f}, "
            f"accuracy={record['mean_per_well_accuracy']:.4f}, "
            f"runtime={total_runtime:.1f}s",
            flush=True,
        )
        return record

    runners = {
        "genetic_algorithm": lambda: run_ga(
            initial,
            space=space,
            budget=budget,
            seed=int(repeat["method_search_seed"]),
            settings=protocol["ga"],
            evaluate=evaluate,
        ),
        "random_search": lambda: run_random(
            initial,
            space=space,
            budget=budget,
            seed=int(repeat["method_search_seed"]),
            evaluate=evaluate,
        ),
        "tpe": lambda: run_tpe(
            initial,
            space=space,
            budget=budget,
            seed=int(repeat["method_search_seed"]),
            startup_candidates=int(protocol["tpe"]["startup_candidates"]),
            evaluate=evaluate,
        ),
    }
    evaluations = runners[method]()
    if position != budget or len(read_jsonl(log_path)) != budget:
        raise RuntimeError(f"Search did not complete its budget: {directory}")
    if len({record["candidate_key"] for record in evaluations}) != budget:
        raise RuntimeError(f"Search contains duplicate candidates: {directory}")
    winner = max(evaluations, key=rank)
    summary = {
        "status": "COMPLETE",
        "repeat_id": repeat_id,
        "base_search_seed": int(repeat["base_search_seed"]),
        "method": method,
        "budget": budget,
        "unique_candidates": budget,
        "reused_candidate_evaluations": int(
            sum(
                bool(record["provenance"].get("evaluation_reused", False))
                for record in evaluations
            )
        ),
        "best_candidate_index": int(winner["candidate_index"]),
        "best_candidate": winner["candidate"],
        "best_validation_macro_f1": float(
            winner["mean_per_well_macro_f1"]
        ),
        "best_validation_accuracy": float(winner["mean_per_well_accuracy"]),
        "best_validation_balanced_accuracy": float(
            winner["mean_per_well_balanced_accuracy"]
        ),
        "charged_search_training_runtime_seconds": float(
            sum(
                record["charged_training_runtime_seconds"]
                for record in evaluations
            )
        ),
        "actual_training_runtime_seconds": float(
            sum(
                record["actual_training_runtime_seconds"]
                for record in evaluations
            )
        ),
        "orchestration_runtime_this_invocation_seconds": float(
            time.perf_counter() - invocation_started
        ),
        "test_assignment_files_opened": False,
        "test_metrics_read": False,
        "candidate_log": str(log_path.relative_to(PROJECT_DIR)),
        "candidate_log_sha256": sha256_file(log_path),
        "manifest": str(manifest_path.relative_to(PROJECT_DIR)),
        "manifest_sha256": sha256_file(manifest_path),
    }
    write_json_atomic(summary_path, summary)
    print(
        f"Finished repeat {repeat_id} {method}: best macro-F1="
        f"{summary['best_validation_macro_f1']:.4f}, charged runtime="
        f"{summary['charged_search_training_runtime_seconds'] / 60:.1f} min",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument("--repeats", nargs="+", type=int, default=[1, 2, 3])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not PROTOCOL_PATH.is_file():
        raise RuntimeError(
            "Run 113_audit_and_freeze_optimizer_repeats.py before this script"
        )
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_REPEATED_OPTIMIZER_SEARCH":
        raise RuntimeError("Repeated-search protocol is not frozen")
    for relative, expected in protocol["code_sha256"].items():
        path = PROJECT_DIR / relative
        if sha256_file(path) != expected:
            raise RuntimeError(f"Frozen code hash changed: {path}")
    source_protocol = json.loads(
        (PROJECT_DIR / protocol["source_protocol"]).read_text(encoding="utf-8")
    )
    if sha256_file(PROJECT_DIR / protocol["source_protocol"]) != protocol[
        "source_protocol_sha256"
    ]:
        raise RuntimeError("Source protocol hash changed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Repeated formal searches require the project CUDA environment")
    tasks, task_audit = prepare_tasks(protocol, source_protocol)
    write_json_atomic(OUTPUT_ROOT / "task_preparation_audit.json", task_audit)
    print("Repeated-search preflight: PASS")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    for well_id, task in tasks.items():
        print(
            f"{well_id}: train={len(task['y_train'])} after SMOTE-Tomek, "
            f"validation={len(task['y_validation'])}",
            flush=True,
        )
    if args.preflight:
        print("Preflight only; no candidate was trained.")
        return

    repeats = {int(value["repeat_id"]): value for value in protocol["repeats"]}
    unknown = sorted(set(args.repeats).difference(repeats))
    if unknown:
        raise RuntimeError(f"Unknown repeat IDs: {unknown}")
    cache = build_cache()
    summaries = []
    for repeat_id in args.repeats:
        for method in args.methods:
            summaries.append(
                run_one_search(
                    protocol,
                    repeats[repeat_id],
                    method,
                    tasks,
                    device,
                    cache,
                )
            )
    completed = sum(value["status"] == "COMPLETE" for value in summaries)
    print(f"Requested searches complete: {completed}/{len(summaries)}")
    if len(args.repeats) == 3 and set(args.methods) == set(METHODS):
        print("Next: run 115_analyze_repeated_optimizer_searches.py")


if __name__ == "__main__":
    main()
