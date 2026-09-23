"""Run a resumable validation-only GA search across six independent well tasks."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = PROJECT_DIR / "experiment_protocol_v5_random_center"
PROTOCOL_PATH = PROTOCOL_DIR / "random_center_protocol_v5.json"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "independent_wells_v5" / "ga_search"

SEARCH_SEED = 113
TRAINING_SEED = 1701
BUDGET = 14
POPULATION_SIZE = 6
TOURNAMENT_SIZE = 3
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.2
ELITES = 2
BATCH_SIZE = 512
MAX_EPOCHS = 60
PATIENCE = 8

sys.path.insert(0, str(PROJECT_DIR))

from gagru.random_center import (  # noqa: E402
    build_random_center_windows,
    impute_from_training_windows,
    remap_labels,
)
from gagru.residual_bigru import fit_with_validation, prepare_per_well_inputs  # noqa: E402
from gagru.search import (  # noqa: E402
    CandidateEvaluation,
    GRUSearchCandidate,
    GRUSearchSpace,
    evaluation_rank,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_candidate() -> GRUSearchCandidate:
    return GRUSearchCandidate(
        hidden_size=96,
        num_layers=2,
        learning_rate=7e-4,
        dropout=0.2,
        weight_decay=1e-4,
    )


def transferred_candidate() -> GRUSearchCandidate:
    return GRUSearchCandidate(
        hidden_size=128,
        num_layers=3,
        learning_rate=0.0029380455401443366,
        dropout=0.3,
        weight_decay=6.532858135218046e-05,
    )


def read_development_assignments(well_id: str, split_seed: int) -> pd.DataFrame:
    parts = []
    for split in ("train", "validation"):
        path = (
            PROTOCOL_DIR
            / "center_assignments"
            / f"seed_{split_seed}"
            / well_id
            / f"{well_id}_{split}_centers.csv"
        )
        parts.append(pd.read_csv(path, encoding="utf-8-sig"))
    return pd.concat(parts, ignore_index=True)


def unique_random_candidate(
    space: GRUSearchSpace, rng: np.random.Generator, used: set[str]
) -> GRUSearchCandidate:
    for _ in range(10_000):
        candidate = space.sample(rng)
        if candidate.key not in used:
            return candidate
    raise RuntimeError("Could not draw a unique GA candidate")


def tournament(
    population: list[CandidateEvaluation],
    rng: np.random.Generator,
    size: int,
) -> GRUSearchCandidate:
    indices = rng.choice(len(population), size=size, replace=False)
    winner = max((population[int(index)] for index in indices), key=evaluation_rank)
    return winner.candidate


def child_candidate(
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


def run_seeded_ga(
    space: GRUSearchSpace,
    *,
    initial_candidates: list[GRUSearchCandidate],
    evaluate: Callable[[GRUSearchCandidate, dict[str, Any]], CandidateEvaluation],
) -> list[CandidateEvaluation]:
    rng = np.random.default_rng(SEARCH_SEED)
    used = {candidate.key for candidate in initial_candidates}
    initial = list(initial_candidates)
    while len(initial) < POPULATION_SIZE:
        candidate = unique_random_candidate(space, rng, used)
        used.add(candidate.key)
        initial.append(candidate)

    evaluations: list[CandidateEvaluation] = []
    population: list[CandidateEvaluation] = []
    for slot, candidate in enumerate(initial, start=1):
        result = evaluate(
            candidate,
            {
                "method": "warm_started_genetic_algorithm",
                "generation": 0,
                "population_slot": slot,
                "warm_start": slot <= len(initial_candidates),
            },
        )
        evaluations.append(result)
        population.append(result)

    generation = 1
    while len(evaluations) < BUDGET:
        elites = sorted(population, key=evaluation_rank, reverse=True)[:ELITES]
        remaining = min(POPULATION_SIZE - ELITES, BUDGET - len(evaluations))
        children: list[GRUSearchCandidate] = []
        attempts = 0
        while len(children) < remaining:
            attempts += 1
            first = tournament(population, rng, TOURNAMENT_SIZE)
            second = tournament(population, rng, TOURNAMENT_SIZE)
            child = child_candidate(first, second, space, rng)
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


def read_existing(path: Path) -> list[CandidateEvaluation]:
    if not path.is_file():
        return []
    results = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            results.append(CandidateEvaluation.from_dict(json.loads(line)))
        except Exception as exc:
            raise RuntimeError(
                f"Invalid candidate log at line {line_number}"
            ) from exc
    return results


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal v5 GA search requires the project CUDA environment")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    split_seed = int(protocol["split"]["primary_model_selection_split_seed"])
    wells = tuple(str(value) for value in protocol["research_scope"]["wells"])
    features = tuple(str(value) for value in protocol["features"]["extended_seven"])
    window_length = int(protocol["windows"]["length"])

    tasks: dict[str, dict[str, Any]] = {}
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
        train = windows["train"]
        validation = windows["validation"]
        y_train, _ = remap_labels(train, classes)
        y_validation, _ = remap_labels(validation, classes)
        X_train, X_validation = prepare_per_well_inputs(train, validation)
        tasks[well_id] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_validation": X_validation,
            "y_validation": y_validation,
            "validation_wells": validation.wells,
            "training_seed": TRAINING_SEED + well_index,
        }

    space = GRUSearchSpace(
        hidden_sizes=tuple(range(64, 161, 16)),
        num_layers=(1, 2, 3),
        learning_rate_min=3e-4,
        learning_rate_max=4e-3,
        dropout_values=(0.0, 0.1, 0.2, 0.3),
        weight_decay_min=1e-6,
        weight_decay_max=3e-4,
        force_zero_dropout_for_one_layer=True,
    )
    initial_candidates = [fixed_candidate(), transferred_candidate()]
    for candidate in initial_candidates:
        space.validate(candidate)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history_dir = OUTPUT_DIR / "histories"
    history_dir.mkdir(exist_ok=True)
    log_path = OUTPUT_DIR / "candidate_evaluations.jsonl"
    manifest_path = OUTPUT_DIR / "search_manifest.json"
    summary_path = OUTPUT_DIR / "search_summary.json"
    manifest = {
        "status": "FROZEN_BEFORE_SEARCH",
        "task": "six independent single-well random-center models",
        "search_uses": "training_and_validation_centers_only",
        "test_assignment_files_opened": False,
        "split_seed": split_seed,
        "search_seed": SEARCH_SEED,
        "base_training_seed": TRAINING_SEED,
        "budget": BUDGET,
        "population_size": POPULATION_SIZE,
        "tournament_size": TOURNAMENT_SIZE,
        "crossover_probability": CROSSOVER_PROBABILITY,
        "per_gene_mutation_probability": MUTATION_PROBABILITY,
        "elites": ELITES,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "selection_metric": "unweighted mean of six per-well supported macro-F1 values",
        "secondary_metric": "unweighted mean of six balanced accuracies",
        "search_space": space.to_dict(),
        "warm_start_candidates": [candidate.to_dict() for candidate in initial_candidates],
        "warm_start_disclosure": (
            "default configuration plus a development-only candidate transferred from the "
            "earlier pooled-task pilot"
        ),
        "wells": list(wells),
        "samples_shared_between_well_models": False,
        "features": list(features),
        "window_length": window_length,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "source_sha256": {
            PROTOCOL_PATH.name: sha256(PROTOCOL_PATH),
            "random_center.py": sha256(PROJECT_DIR / "gagru" / "random_center.py"),
            "residual_bigru.py": sha256(PROJECT_DIR / "gagru" / "residual_bigru.py"),
            Path(__file__).name: sha256(Path(__file__)),
        },
    }
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise RuntimeError("Existing GA manifest conflicts with the frozen search")
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    existing = read_existing(log_path)
    replay_position = 0

    def evaluate(
        candidate: GRUSearchCandidate, provenance: dict[str, Any]
    ) -> CandidateEvaluation:
        nonlocal replay_position
        candidate_index = replay_position + 1
        if replay_position < len(existing):
            saved = existing[replay_position]
            if (
                saved.candidate_index != candidate_index
                or saved.candidate_key != candidate.key
            ):
                raise RuntimeError("Deterministic GA replay diverged from candidate log")
            replay_position += 1
            print(
                f"Replayed candidate {candidate_index}/{BUDGET}: "
                f"mean macro-F1={saved.selection_score:.4f}",
                flush=True,
            )
            return saved

        per_well_rows: list[dict[str, Any]] = []
        total_parameters = 0
        total_runtime = 0.0
        total_epochs = 0
        best_epochs = []
        losses = []
        peak_memory = 0
        for well_id, task in tasks.items():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            fit = fit_with_validation(
                candidate,
                task["X_train"],
                task["y_train"],
                task["X_validation"],
                task["y_validation"],
                task["validation_wells"],
                device=device,
                seed=int(task["training_seed"]),
                batch_size=BATCH_SIZE,
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
            )
            current_peak = int(torch.cuda.max_memory_allocated(device))
            peak_memory = max(peak_memory, current_peak)
            samples = int(len(task["y_validation"]))
            per_well_rows.append(
                {
                    "candidate_index": candidate_index,
                    "well_id": well_id,
                    "validation_samples": samples,
                    "macro_f1": fit.validation_macro_f1,
                    "accuracy": fit.validation_accuracy,
                    "balanced_accuracy": fit.validation_balanced_accuracy,
                    "best_epoch": fit.best_epoch,
                    "epochs_completed": fit.epochs_completed,
                    "runtime_seconds": fit.runtime_seconds,
                    "trainable_parameters": fit.trainable_parameters,
                }
            )
            pd.DataFrame(fit.history).to_csv(
                history_dir / f"candidate_{candidate_index:03d}_{well_id}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            total_parameters += fit.trainable_parameters
            total_runtime += fit.runtime_seconds
            total_epochs += fit.epochs_completed
            best_epochs.append(fit.best_epoch)
            losses.append(fit.best_validation_loss)

        per_well = pd.DataFrame(per_well_rows)
        per_well_path = f"histories/candidate_{candidate_index:03d}_well_metrics.csv"
        per_well.to_csv(OUTPUT_DIR / per_well_path, index=False, encoding="utf-8-sig")
        mean_macro = float(per_well["macro_f1"].mean())
        mean_accuracy = float(per_well["accuracy"].mean())
        mean_balanced = float(per_well["balanced_accuracy"].mean())
        pooled_accuracy = float(
            np.average(per_well["accuracy"], weights=per_well["validation_samples"])
        )
        result = CandidateEvaluation(
            candidate_index=candidate_index,
            candidate_key=candidate.key,
            candidate=candidate,
            training_seed=TRAINING_SEED,
            selection_score=mean_macro,
            balanced_accuracy=mean_balanced,
            trainable_parameters=total_parameters,
            runtime_seconds=total_runtime,
            peak_gpu_memory_bytes=peak_memory,
            epochs_completed=total_epochs,
            best_epoch=int(round(float(np.mean(best_epochs)))),
            best_validation_loss=float(np.mean(losses)),
            metrics={
                "mean_per_well_macro_f1": mean_macro,
                "mean_per_well_accuracy": mean_accuracy,
                "pooled_accuracy": pooled_accuracy,
                "mean_per_well_balanced_accuracy": mean_balanced,
                "per_well": per_well_rows,
            },
            history_file=per_well_path,
            provenance=provenance,
        )
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        existing.append(result)
        replay_position += 1
        print(
            f"Completed candidate {candidate_index}/{BUDGET}: "
            f"mean macro-F1={mean_macro:.4f}, mean accuracy={mean_accuracy:.4f}, "
            f"pooled accuracy={pooled_accuracy:.4f}, runtime={total_runtime:.1f}s",
            flush=True,
        )
        return result

    evaluations = run_seeded_ga(
        space,
        initial_candidates=initial_candidates,
        evaluate=evaluate,
    )
    if replay_position != len(existing):
        raise RuntimeError("Saved candidate log contains unreplayed records")
    if len(evaluations) != BUDGET:
        raise RuntimeError("GA did not consume its frozen candidate budget")
    if len({item.candidate_key for item in evaluations}) != BUDGET:
        raise RuntimeError("GA candidate budget contains duplicates")

    winner = max(evaluations, key=evaluation_rank)
    fixed = next(
        item for item in evaluations if item.candidate_key == fixed_candidate().key
    )
    transferred = next(
        item for item in evaluations if item.candidate_key == transferred_candidate().key
    )
    summary = {
        "status": "COMPLETE",
        "test_assignment_files_opened": False,
        "test_metrics_read_for_selection": False,
        "samples_shared_between_well_models": False,
        "budget": BUDGET,
        "unique_candidates_evaluated": len(evaluations),
        "best_candidate_index": winner.candidate_index,
        "best_candidate": winner.candidate.to_dict(),
        "best_validation_macro_f1": winner.selection_score,
        "best_validation_mean_accuracy": winner.metrics["mean_per_well_accuracy"],
        "best_validation_pooled_accuracy": winner.metrics["pooled_accuracy"],
        "fixed_validation_macro_f1": fixed.selection_score,
        "fixed_validation_mean_accuracy": fixed.metrics["mean_per_well_accuracy"],
        "transferred_validation_macro_f1": transferred.selection_score,
        "transferred_validation_mean_accuracy": transferred.metrics[
            "mean_per_well_accuracy"
        ],
        "winner_provenance": winner.provenance,
        "winner_differs_from_fixed": winner.candidate_key != fixed.candidate_key,
        "candidate_results": [item.to_dict() for item in evaluations],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    printable = {key: value for key, value in summary.items() if key != "candidate_results"}
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
    print(f"Search report: {summary_path}")


if __name__ == "__main__":
    main()
