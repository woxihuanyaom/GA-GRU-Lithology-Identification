"""Run equal-budget GA, random, or TPE search for the paper's plain GRU."""

from __future__ import annotations

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
PROTOCOL_DIR = PROJECT_DIR / "experiment_protocol_v5_random_center"
PROTOCOL_PATH = PROTOCOL_DIR / "random_center_protocol_v5.json"
OUTPUT_ROOT = PROJECT_DIR / "outputs" / "independent_wells_v5" / "plain_gru_searches"
BASELINE_LOG = (
    PROJECT_DIR
    / "outputs"
    / "independent_wells_v5"
    / "recurrent_baseline_screen"
    / "configuration_evaluations.jsonl"
)

METHOD = os.environ.get("GAGRU_PLAIN_GRU_METHOD", "ga").strip().lower()
if METHOD not in {"ga", "random", "tpe"}:
    raise RuntimeError("GAGRU_PLAIN_GRU_METHOD must be ga, random, or tpe")
OUTPUT_DIR = OUTPUT_ROOT / METHOD

BUDGET = 14
POPULATION_SIZE = 6
TOURNAMENT_SIZE = 3
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.2
ELITES = 2
SEARCH_SEEDS = {"ga": 113, "random": 211, "tpe": 307}
TPE_RANDOM_WARMUP = 6
TRAINING_SEED = 1701
BATCH_SIZE = 512
MAX_EPOCHS = 60
PATIENCE = 8

sys.path.insert(0, str(PROJECT_DIR))

from gagru.local_recurrent import fit_local_recurrent_with_validation  # noqa: E402
from gagru.random_center import (  # noqa: E402
    build_random_center_windows,
    impute_from_training_windows,
    remap_labels,
)
from gagru.residual_bigru import prepare_per_well_inputs  # noqa: E402
from gagru.search import GRUSearchCandidate, GRUSearchSpace  # noqa: E402


def fixed_candidate() -> GRUSearchCandidate:
    return GRUSearchCandidate(96, 2, 7e-4, 0.2, 1e-4)


def transferred_candidate() -> GRUSearchCandidate:
    return GRUSearchCandidate(
        128, 3, 0.0029380455401443366, 0.3, 6.532858135218046e-05
    )


def search_space() -> GRUSearchSpace:
    return GRUSearchSpace(
        hidden_sizes=tuple(range(64, 161, 16)),
        num_layers=(1, 2, 3),
        learning_rate_min=3e-4,
        learning_rate_max=4e-3,
        dropout_values=(0.0, 0.1, 0.2, 0.3),
        weight_decay_min=1e-6,
        weight_decay_max=3e-4,
        force_zero_dropout_for_one_layer=True,
    )


def rank(record: dict[str, Any]) -> tuple[float, float, float, int, float]:
    return (
        float(record["mean_per_well_macro_f1"]),
        float(record["mean_per_well_balanced_accuracy"]),
        float(record["mean_per_well_accuracy"]),
        -int(record["total_parameters_across_six_models"]),
        -float(record["charged_runtime_seconds"]),
    )


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
            raise RuntimeError(f"Invalid JSONL line {line_number}: {path}") from exc
    return records


def read_development_assignments(well_id: str, split_seed: int) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_csv(
                PROTOCOL_DIR
                / "center_assignments"
                / f"seed_{split_seed}"
                / well_id
                / f"{well_id}_{split}_centers.csv",
                encoding="utf-8-sig",
            )
            for split in ("train", "validation")
        ],
        ignore_index=True,
    )


def unique_random_candidate(
    space: GRUSearchSpace, rng: np.random.Generator, used: set[str]
) -> GRUSearchCandidate:
    for _ in range(10_000):
        candidate = space.sample(rng)
        if candidate.key not in used:
            return candidate
    raise RuntimeError("Could not generate another unique candidate")


def tournament(
    population: list[dict[str, Any]], rng: np.random.Generator
) -> GRUSearchCandidate:
    indices = rng.choice(len(population), size=TOURNAMENT_SIZE, replace=False)
    winner = max((population[int(index)] for index in indices), key=rank)
    return GRUSearchCandidate.from_dict(winner["candidate"])


def ga_child(
    first: GRUSearchCandidate,
    second: GRUSearchCandidate,
    space: GRUSearchSpace,
    rng: np.random.Generator,
) -> GRUSearchCandidate:
    names = ("hidden_size", "num_layers", "learning_rate", "dropout", "weight_decay")
    if rng.random() < CROSSOVER_PROBABILITY:
        genes = {
            name: getattr(first if rng.random() < 0.5 else second, name)
            for name in names
        }
    else:
        genes = first.to_dict()
    if rng.random() < MUTATION_PROBABILITY:
        genes["hidden_size"] = int(rng.choice(space.hidden_sizes))
    if rng.random() < MUTATION_PROBABILITY:
        genes["num_layers"] = int(rng.choice(space.num_layers))
    if rng.random() < MUTATION_PROBABILITY:
        genes["learning_rate"] = float(
            np.exp(
                rng.uniform(
                    np.log(space.learning_rate_min),
                    np.log(space.learning_rate_max),
                )
            )
        )
    if rng.random() < MUTATION_PROBABILITY:
        genes["dropout"] = float(rng.choice(space.dropout_values))
    if rng.random() < MUTATION_PROBABILITY:
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
    space: GRUSearchSpace,
    evaluate: Callable[[GRUSearchCandidate, dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEARCH_SEEDS["ga"])
    initial = [fixed_candidate(), transferred_candidate()]
    used = {candidate.key for candidate in initial}
    while len(initial) < POPULATION_SIZE:
        candidate = unique_random_candidate(space, rng, used)
        used.add(candidate.key)
        initial.append(candidate)
    evaluations = []
    population = []
    for slot, candidate in enumerate(initial, start=1):
        result = evaluate(
            candidate,
            {
                "method": "warm_started_genetic_algorithm",
                "generation": 0,
                "population_slot": slot,
                "warm_start": slot <= 2,
            },
        )
        evaluations.append(result)
        population.append(result)
    generation = 1
    while len(evaluations) < BUDGET:
        elites = sorted(population, key=rank, reverse=True)[:ELITES]
        remaining = min(POPULATION_SIZE - ELITES, BUDGET - len(evaluations))
        children = []
        attempts = 0
        while len(children) < remaining:
            attempts += 1
            child = ga_child(
                tournament(population, rng),
                tournament(population, rng),
                space,
                rng,
            )
            pending = {candidate.key for candidate in children}
            if child.key in used or child.key in pending:
                if attempts > 2_000:
                    child = unique_random_candidate(space, rng, used.union(pending))
                    attempts = 0
                else:
                    continue
            children.append(child)
        child_results = []
        for slot, child in enumerate(children, start=ELITES + 1):
            used.add(child.key)
            result = evaluate(
                child,
                {
                    "method": "warm_started_genetic_algorithm",
                    "generation": generation,
                    "population_slot": slot,
                    "warm_start": False,
                },
            )
            evaluations.append(result)
            child_results.append(result)
        population = elites + child_results
        generation += 1
    return evaluations


def run_random(
    space: GRUSearchSpace,
    evaluate: Callable[[GRUSearchCandidate, dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEARCH_SEEDS["random"])
    initial = [fixed_candidate(), transferred_candidate()]
    used = set()
    evaluations = []
    for draw, candidate in enumerate(initial, start=1):
        used.add(candidate.key)
        evaluations.append(
            evaluate(
                candidate,
                {
                    "method": "warm_started_random_search",
                    "draw": draw,
                    "warm_start": True,
                },
            )
        )
    while len(evaluations) < BUDGET:
        candidate = unique_random_candidate(space, rng, used)
        used.add(candidate.key)
        evaluations.append(
            evaluate(
                candidate,
                {
                    "method": "warm_started_random_search",
                    "draw": len(evaluations) + 1,
                    "warm_start": False,
                },
            )
        )
    return evaluations


def tpe_candidate(trial: Any, space: GRUSearchSpace) -> GRUSearchCandidate:
    num_layers = int(trial.suggest_categorical("num_layers", list(space.num_layers)))
    dropout = (
        0.0
        if num_layers == 1
        else float(trial.suggest_categorical("dropout", list(space.dropout_values)))
    )
    return space.make_candidate(
        hidden_size=trial.suggest_int(
            "hidden_size", min(space.hidden_sizes), max(space.hidden_sizes), step=16
        ),
        num_layers=num_layers,
        learning_rate=trial.suggest_float(
            "learning_rate", space.learning_rate_min, space.learning_rate_max, log=True
        ),
        dropout=dropout,
        weight_decay=trial.suggest_float(
            "weight_decay", space.weight_decay_min, space.weight_decay_max, log=True
        ),
    )


def run_tpe(
    space: GRUSearchSpace,
    evaluate: Callable[[GRUSearchCandidate, dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=SEARCH_SEEDS["tpe"], n_startup_trials=TPE_RANDOM_WARMUP
        ),
    )
    initial = [fixed_candidate(), transferred_candidate()]
    for candidate in initial:
        study.enqueue_trial(candidate.to_dict())
    evaluations = []
    used = set()
    duplicates = 0
    while len(evaluations) < BUDGET:
        trial = study.ask()
        candidate = tpe_candidate(trial, space)
        if candidate.key in used:
            duplicates += 1
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            if duplicates > 1_000:
                raise RuntimeError("TPE repeatedly generated duplicate candidates")
            continue
        used.add(candidate.key)
        result = evaluate(
            candidate,
            {
                "method": "warm_started_tpe",
                "trial_number": int(trial.number),
                "unique_candidate": len(evaluations) + 1,
                "warm_start": len(evaluations) < 2,
            },
        )
        study.tell(trial, float(result["mean_per_well_macro_f1"]))
        evaluations.append(result)
    return evaluations


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal plain-GRU optimizer search requires CUDA")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    split_seed = int(protocol["split"]["primary_model_selection_split_seed"])
    wells = tuple(str(value) for value in protocol["research_scope"]["wells"])
    features = tuple(str(value) for value in protocol["features"]["extended_seven"])
    window_length = int(protocol["windows"]["length"])
    tasks = {}
    for well_index, well_id in enumerate(wells):
        record = protocol["well_protocols"][well_id]
        classes = tuple(int(value) for value in record["included_classes"])
        frame = pd.read_csv(
            PROTOCOL_DIR / str(record["source_snapshot"]), encoding="utf-8-sig"
        )
        assignment = read_development_assignments(well_id, split_seed)
        windows = build_random_center_windows(
            frame,
            assignment,
            features,
            window_length,
            requested_splits=("train", "validation"),
        )
        windows, _ = impute_from_training_windows(windows)
        y_train, _ = remap_labels(windows["train"], classes)
        y_validation, _ = remap_labels(windows["validation"], classes)
        X_train, X_validation = prepare_per_well_inputs(
            windows["train"], windows["validation"]
        )
        tasks[well_id] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_validation": X_validation,
            "y_validation": y_validation,
            "training_seed": TRAINING_SEED + well_index,
        }

    space = search_space()
    initial = [fixed_candidate(), transferred_candidate()]
    for candidate in initial:
        space.validate(candidate)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    histories = OUTPUT_DIR / "histories"
    histories.mkdir(exist_ok=True)
    log_path = OUTPUT_DIR / "candidate_evaluations.jsonl"
    manifest_path = OUTPUT_DIR / "search_manifest.json"
    summary_path = OUTPUT_DIR / "search_summary.json"
    manifest = {
        "status": "FROZEN_BEFORE_SEARCH",
        "method": METHOD,
        "model": "unidirectional_many_to_one_GRU",
        "search_uses": "training_and_validation_centers_only",
        "test_assignment_files_opened": False,
        "budget": BUDGET,
        "population_size": POPULATION_SIZE if METHOD == "ga" else None,
        "tournament_size": TOURNAMENT_SIZE if METHOD == "ga" else None,
        "crossover_probability": CROSSOVER_PROBABILITY if METHOD == "ga" else None,
        "per_gene_mutation_probability": MUTATION_PROBABILITY if METHOD == "ga" else None,
        "elites": ELITES if METHOD == "ga" else None,
        "tpe_random_warmup": TPE_RANDOM_WARMUP if METHOD == "tpe" else None,
        "search_seed": SEARCH_SEEDS[METHOD],
        "split_seed": split_seed,
        "base_training_seed": TRAINING_SEED,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "selection_metric": "mean of six per-well supported macro-F1 values",
        "secondary_metric": "mean of six per-well balanced accuracies",
        "search_space": space.to_dict(),
        "warm_start_candidates": [candidate.to_dict() for candidate in initial],
        "warm_start_candidates_count_toward_budget": True,
        "cache_rule": "identical deterministic candidates reuse prior plain-GRU evaluations and retain charged runtime",
        "wells": list(wells),
        "samples_shared_between_well_models": False,
        "features": list(features),
        "derived_channels": "three resistivity separations plus first differences",
        "window_length": window_length,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
    }
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Existing plain-GRU search manifest conflicts")
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    baseline_cache = {}
    for record in read_jsonl(BASELINE_LOG):
        if record.get("architecture") == "gru":
            baseline_cache[str(record["candidate_key"])] = record
    for other_method in ("ga", "random", "tpe"):
        for record in read_jsonl(
            OUTPUT_ROOT / other_method / "candidate_evaluations.jsonl"
        ):
            baseline_cache.setdefault(str(record["candidate_key"]), record)
    existing = read_jsonl(log_path)
    position = 0
    wall_started = time.perf_counter()

    def append_record(record: dict[str, Any]) -> None:
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

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
                raise RuntimeError("Deterministic plain-GRU search replay diverged")
            position += 1
            print(
                f"Replayed {METHOD} candidate {candidate_index}/{BUDGET}: "
                f"macro-F1={saved['mean_per_well_macro_f1']:.4f}",
                flush=True,
            )
            return saved

        cached = baseline_cache.get(candidate.key)
        if cached is not None:
            record = {
                "candidate_index": candidate_index,
                "candidate_key": candidate.key,
                "candidate": candidate.to_dict(),
                "mean_per_well_macro_f1": float(
                    cached["mean_per_well_macro_f1"]
                ),
                "mean_per_well_accuracy": float(cached["mean_per_well_accuracy"]),
                "pooled_accuracy": float(cached["pooled_accuracy"]),
                "mean_per_well_balanced_accuracy": float(
                    cached["mean_per_well_balanced_accuracy"]
                ),
                "total_parameters_across_six_models": int(
                    sum(row["trainable_parameters"] for row in cached["per_well"])
                ),
                "charged_runtime_seconds": float(
                    cached.get("runtime_seconds", cached.get("charged_runtime_seconds"))
                ),
                "per_well": cached["per_well"],
                "provenance": {
                    **provenance,
                    "evaluation_reused": True,
                    "reused_from": "prior_plain_gru_validation_cache",
                },
            }
            append_record(record)
            existing.append(record)
            baseline_cache[candidate.key] = record
            position += 1
            print(
                f"Reused {METHOD} candidate {candidate_index}/{BUDGET}: "
                f"macro-F1={record['mean_per_well_macro_f1']:.4f}",
                flush=True,
            )
            return record

        per_well = []
        total_runtime = 0.0
        total_parameters = 0
        for well_id, task in tasks.items():
            torch.cuda.empty_cache()
            fit = fit_local_recurrent_with_validation(
                "gru",
                candidate,
                task["X_train"],
                task["y_train"],
                task["X_validation"],
                task["y_validation"],
                device=device,
                seed=int(task["training_seed"]),
                batch_size=BATCH_SIZE,
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
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
                }
            )
            total_runtime += fit.runtime_seconds
            total_parameters += fit.trainable_parameters
            pd.DataFrame(fit.history).to_csv(
                histories / f"candidate_{candidate_index:03d}_{well_id}.csv",
                index=False,
                encoding="utf-8-sig",
            )
        frame = pd.DataFrame(per_well)
        record = {
            "candidate_index": candidate_index,
            "candidate_key": candidate.key,
            "candidate": candidate.to_dict(),
            "mean_per_well_macro_f1": float(frame["macro_f1"].mean()),
            "mean_per_well_accuracy": float(frame["accuracy"].mean()),
            "pooled_accuracy": float(
                np.average(frame["accuracy"], weights=frame["validation_samples"])
            ),
            "mean_per_well_balanced_accuracy": float(
                frame["balanced_accuracy"].mean()
            ),
            "total_parameters_across_six_models": total_parameters,
            "charged_runtime_seconds": total_runtime,
            "per_well": per_well,
            "provenance": {**provenance, "evaluation_reused": False},
        }
        append_record(record)
        existing.append(record)
        baseline_cache[candidate.key] = record
        position += 1
        print(
            f"Completed {METHOD} candidate {candidate_index}/{BUDGET}: "
            f"macro-F1={record['mean_per_well_macro_f1']:.4f}, "
            f"accuracy={record['mean_per_well_accuracy']:.4f}, "
            f"pooled={record['pooled_accuracy']:.4f}, runtime={total_runtime:.1f}s",
            flush=True,
        )
        return record

    runners = {"ga": run_ga, "random": run_random, "tpe": run_tpe}
    evaluations = runners[METHOD](space, evaluate)
    completed_log = read_jsonl(log_path)
    if position != BUDGET or len(completed_log) != BUDGET:
        raise RuntimeError("Plain-GRU search log does not contain the frozen budget")
    if len({record["candidate_key"] for record in evaluations}) != BUDGET:
        raise RuntimeError("Plain-GRU search contains duplicate candidates")
    winner = max(evaluations, key=rank)
    summary = {
        "status": "COMPLETE",
        "method": METHOD,
        "model": "unidirectional_many_to_one_GRU",
        "test_assignment_files_opened": False,
        "test_metrics_used_for_selection": False,
        "samples_shared_between_well_models": False,
        "budget": BUDGET,
        "unique_candidates_evaluated": len(evaluations),
        "candidate_evaluations_reused": int(
            sum(record["provenance"]["evaluation_reused"] for record in evaluations)
        ),
        "best_candidate_index": int(winner["candidate_index"]),
        "best_candidate": winner["candidate"],
        "best_validation_macro_f1": winner["mean_per_well_macro_f1"],
        "best_validation_mean_accuracy": winner["mean_per_well_accuracy"],
        "best_validation_pooled_accuracy": winner["pooled_accuracy"],
        "best_validation_balanced_accuracy": winner[
            "mean_per_well_balanced_accuracy"
        ],
        "winner_total_parameters_across_six_models": winner[
            "total_parameters_across_six_models"
        ],
        "charged_search_runtime_seconds": float(
            sum(record["charged_runtime_seconds"] for record in evaluations)
        ),
        "actual_wall_runtime_seconds": time.perf_counter() - wall_started,
        "winner_provenance": winner["provenance"],
        "candidate_results": evaluations,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    printable = {key: value for key, value in summary.items() if key != "candidate_results"}
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
    print(f"Search report: {summary_path}")


if __name__ == "__main__":
    main()
