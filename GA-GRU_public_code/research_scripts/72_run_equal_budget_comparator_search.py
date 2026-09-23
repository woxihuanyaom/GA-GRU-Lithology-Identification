"""Run resumable equal-budget random or TPE search for the v5 well tasks."""

from __future__ import annotations

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
PROTOCOL_DIR = PROJECT_DIR / "experiment_protocol_v5_random_center"
PROTOCOL_PATH = PROTOCOL_DIR / "random_center_protocol_v5.json"
OUTPUT_ROOT = PROJECT_DIR / "outputs" / "independent_wells_v5"
GA_LOG_PATH = OUTPUT_ROOT / "ga_search" / "candidate_evaluations.jsonl"

METHOD = os.environ.get("GAGRU_COMPARATOR_METHOD", "random").strip().lower()
if METHOD not in {"random", "tpe"}:
    raise RuntimeError("GAGRU_COMPARATOR_METHOD must be random or tpe")
OUTPUT_DIR = OUTPUT_ROOT / f"{METHOD}_search"

BUDGET = 14
TRAINING_SEED = 1701
BATCH_SIZE = 512
MAX_EPOCHS = 60
PATIENCE = 8
RANDOM_SEARCH_SEED = 211
TPE_SEARCH_SEED = 307
TPE_RANDOM_WARMUP = 6

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


def read_log(path: Path) -> list[CandidateEvaluation]:
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
            raise RuntimeError(f"Invalid search log line {line_number}: {path}") from exc
    return results


def unique_random_candidate(
    space: GRUSearchSpace, rng: np.random.Generator, used: set[str]
) -> GRUSearchCandidate:
    for _ in range(10_000):
        candidate = space.sample(rng)
        if candidate.key not in used:
            return candidate
    raise RuntimeError("Could not draw a unique random-search candidate")


def run_warm_random(
    space: GRUSearchSpace,
    evaluate: Callable[[GRUSearchCandidate, dict[str, Any]], CandidateEvaluation],
) -> list[CandidateEvaluation]:
    initial = [fixed_candidate(), transferred_candidate()]
    used: set[str] = set()
    results = []
    for slot, candidate in enumerate(initial, start=1):
        used.add(candidate.key)
        results.append(
            evaluate(
                candidate,
                {
                    "method": "warm_started_random_search",
                    "draw": slot,
                    "warm_start": True,
                },
            )
        )
    rng = np.random.default_rng(RANDOM_SEARCH_SEED)
    while len(results) < BUDGET:
        candidate = unique_random_candidate(space, rng, used)
        used.add(candidate.key)
        results.append(
            evaluate(
                candidate,
                {
                    "method": "warm_started_random_search",
                    "draw": len(results) + 1,
                    "warm_start": False,
                },
            )
        )
    return results


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


def run_warm_tpe(
    space: GRUSearchSpace,
    evaluate: Callable[[GRUSearchCandidate, dict[str, Any]], CandidateEvaluation],
) -> list[CandidateEvaluation]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        seed=TPE_SEARCH_SEED,
        n_startup_trials=TPE_RANDOM_WARMUP,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    initial = [fixed_candidate(), transferred_candidate()]
    for candidate in initial:
        study.enqueue_trial(candidate.to_dict())
    used: set[str] = set()
    results = []
    duplicate_trials = 0
    while len(results) < BUDGET:
        trial = study.ask()
        candidate = tpe_candidate(trial, space)
        if candidate.key in used:
            duplicate_trials += 1
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            if duplicate_trials > 1_000:
                raise RuntimeError("TPE repeatedly generated duplicate candidates")
            continue
        used.add(candidate.key)
        result = evaluate(
            candidate,
            {
                "method": "warm_started_tpe",
                "trial_number": int(trial.number),
                "unique_candidate": len(results) + 1,
                "warm_start": len(results) < len(initial),
            },
        )
        study.tell(trial, result.selection_score)
        results.append(result)
    return results


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Equal-budget searches require the project CUDA environment")
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

    space = search_space()
    initial = [fixed_candidate(), transferred_candidate()]
    for candidate in initial:
        space.validate(candidate)
    search_seed = RANDOM_SEARCH_SEED if METHOD == "random" else TPE_SEARCH_SEED
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history_dir = OUTPUT_DIR / "histories"
    history_dir.mkdir(exist_ok=True)
    log_path = OUTPUT_DIR / "candidate_evaluations.jsonl"
    manifest_path = OUTPUT_DIR / "search_manifest.json"
    summary_path = OUTPUT_DIR / "search_summary.json"
    manifest = {
        "status": "FROZEN_BEFORE_SEARCH",
        "method": METHOD,
        "task": "six independent single-well random-center models",
        "search_uses": "training_and_validation_centers_only",
        "test_assignment_files_opened": False,
        "split_seed": split_seed,
        "search_seed": search_seed,
        "base_training_seed": TRAINING_SEED,
        "budget": BUDGET,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "selection_metric": "unweighted mean of six per-well supported macro-F1 values",
        "secondary_metric": "unweighted mean of six balanced accuracies",
        "search_space": space.to_dict(),
        "warm_start_candidates": [candidate.to_dict() for candidate in initial],
        "warm_start_candidates_count_toward_budget": True,
        "tpe_random_warmup": TPE_RANDOM_WARMUP if METHOD == "tpe" else None,
        "candidate_cache_rule": (
            "An identical deterministic candidate may reuse its prior evaluation; the "
            "original training runtime remains charged to this method's compute total."
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
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved_manifest != manifest:
            raise RuntimeError("Existing comparator manifest conflicts with frozen search")
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    existing = read_log(log_path)
    cache: dict[str, tuple[CandidateEvaluation, str]] = {}
    for source_name, source_path in (
        ("ga_search", GA_LOG_PATH),
        ("random_search", OUTPUT_ROOT / "random_search" / "candidate_evaluations.jsonl"),
        ("tpe_search", OUTPUT_ROOT / "tpe_search" / "candidate_evaluations.jsonl"),
    ):
        for item in read_log(source_path):
            cache.setdefault(item.candidate_key, (item, source_name))
    replay_position = 0
    wall_started = time.perf_counter()

    def append_result(result: CandidateEvaluation) -> None:
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

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
                raise RuntimeError("Deterministic search replay diverged from saved log")
            replay_position += 1
            print(
                f"Replayed {METHOD} candidate {candidate_index}/{BUDGET}: "
                f"mean macro-F1={saved.selection_score:.4f}",
                flush=True,
            )
            return saved

        if candidate.key in cache:
            source, source_name = cache[candidate.key]
            metrics = json.loads(json.dumps(source.metrics))
            for row in metrics.get("per_well", []):
                row["candidate_index"] = candidate_index
            result = CandidateEvaluation(
                candidate_index=candidate_index,
                candidate_key=candidate.key,
                candidate=candidate,
                training_seed=source.training_seed,
                selection_score=source.selection_score,
                balanced_accuracy=source.balanced_accuracy,
                trainable_parameters=source.trainable_parameters,
                runtime_seconds=source.runtime_seconds,
                peak_gpu_memory_bytes=source.peak_gpu_memory_bytes,
                epochs_completed=source.epochs_completed,
                best_epoch=source.best_epoch,
                best_validation_loss=source.best_validation_loss,
                metrics=metrics,
                history_file=f"reused_from_{source_name}_candidate_{source.candidate_index}",
                provenance={
                    **provenance,
                    "evaluation_reused": True,
                    "reused_from": source_name,
                    "reused_candidate_index": source.candidate_index,
                },
            )
            append_result(result)
            existing.append(result)
            cache[candidate.key] = (result, f"{METHOD}_search")
            replay_position += 1
            print(
                f"Reused {METHOD} candidate {candidate_index}/{BUDGET}: "
                f"mean macro-F1={result.selection_score:.4f}",
                flush=True,
            )
            return result

        per_well_rows = []
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
            peak_memory = max(
                peak_memory, int(torch.cuda.max_memory_allocated(device))
            )
            per_well_rows.append(
                {
                    "candidate_index": candidate_index,
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
            provenance={**provenance, "evaluation_reused": False},
        )
        append_result(result)
        existing.append(result)
        cache[candidate.key] = (result, f"{METHOD}_search")
        replay_position += 1
        print(
            f"Completed {METHOD} candidate {candidate_index}/{BUDGET}: "
            f"mean macro-F1={mean_macro:.4f}, mean accuracy={mean_accuracy:.4f}, "
            f"pooled accuracy={pooled_accuracy:.4f}, runtime={total_runtime:.1f}s",
            flush=True,
        )
        return result

    if METHOD == "random":
        evaluations = run_warm_random(space, evaluate)
    else:
        evaluations = run_warm_tpe(space, evaluate)
    if replay_position != len(existing):
        raise RuntimeError("Saved comparator log contains unreplayed records")
    if len(evaluations) != BUDGET:
        raise RuntimeError("Comparator did not consume its frozen budget")
    if len({item.candidate_key for item in evaluations}) != BUDGET:
        raise RuntimeError("Comparator candidate budget contains duplicates")

    winner = max(evaluations, key=evaluation_rank)
    summary = {
        "status": "COMPLETE",
        "method": METHOD,
        "test_assignment_files_opened": False,
        "test_metrics_read_for_selection": False,
        "samples_shared_between_well_models": False,
        "budget": BUDGET,
        "unique_candidates_evaluated": len(evaluations),
        "candidate_evaluations_reused": int(
            sum(bool(item.provenance.get("evaluation_reused")) for item in evaluations)
        ),
        "best_candidate_index": winner.candidate_index,
        "best_candidate": winner.candidate.to_dict(),
        "best_validation_macro_f1": winner.selection_score,
        "best_validation_mean_accuracy": winner.metrics["mean_per_well_accuracy"],
        "best_validation_pooled_accuracy": winner.metrics["pooled_accuracy"],
        "charged_candidate_training_runtime_seconds": float(
            sum(item.runtime_seconds for item in evaluations)
        ),
        "actual_wall_runtime_seconds": time.perf_counter() - wall_started,
        "winner_provenance": winner.provenance,
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
