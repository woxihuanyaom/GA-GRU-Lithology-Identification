"""Run validation-only feature, representation, and imbalance ablations."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = PROJECT_DIR / "experiment_protocol_v5_random_center"
PROTOCOL_PATH = PROTOCOL_DIR / "random_center_protocol_v5.json"
OPTIMIZER_FREEZE_PATH = (
    PROJECT_DIR
    / "outputs"
    / "independent_wells_v5"
    / "plain_gru_searches"
    / "audit"
    / "plain_gru_optimizer_winner_freeze.json"
)
OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "independent_wells_v5" / "validation_ablation"
)
RUN_LOG_PATH = OUTPUT_DIR / "well_evaluations.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "ablation_manifest.json"
STAGE1_FREEZE_PATH = OUTPUT_DIR / "stage1_feature_representation_freeze.json"
COMPLETION_PATH = OUTPUT_DIR / "ablation_completion.json"
HISTORY_DIR = OUTPUT_DIR / "histories"

TRAINING_SEED_BASE = 1701
BALANCE_SEED_BASE = 27101
MI_SEED_BASE = 37101
BATCH_SIZE = 512
MAX_EPOCHS = 60
PATIENCE = 8
REPRESENTATIONS = ("physical", "engineered")
STAGE2_STRATEGIES = ("class_weighted", "smote_tomek")

sys.path.insert(0, str(PROJECT_DIR))

from gagru.local_recurrent import fit_local_recurrent_with_validation  # noqa: E402
from gagru.random_center import (  # noqa: E402
    build_random_center_windows,
    impute_from_training_windows,
    remap_labels,
)
from gagru.random_center_ablation import (  # noqa: E402
    apply_imbalance_strategy,
    center_mutual_information,
)
from gagru.residual_bigru import prepare_per_well_inputs  # noqa: E402
from gagru.search import GRUSearchCandidate  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


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


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def feature_sets(protocol: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    return {
        "core_five": tuple(protocol["features"]["core_five"]),
        "manuscript_six": ("MSFL", "LLS", "LLD", "DT", "GR", "NPHI"),
        "extended_seven": tuple(protocol["features"]["extended_seven"]),
    }


def input_channel_count(features: list[str], representation: str) -> int:
    if representation == "physical":
        return len(features)
    if representation == "engineered":
        return 2 * len(features) + 3
    raise ValueError(f"Unknown representation: {representation}")


def stage1_configurations(
    sets: dict[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    return [
        {
            "configuration_id": f"{name}__{representation}__unweighted",
            "stage": "feature_representation",
            "feature_set": name,
            "features": list(features),
            "representation": representation,
            "imbalance_strategy": "unweighted",
            "input_channels": input_channel_count(list(features), representation),
        }
        for name, features in sets.items()
        for representation in REPRESENTATIONS
    ]


def stage2_configurations(winner: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **{
                key: value
                for key, value in winner.items()
                if key
                in {
                    "feature_set",
                    "features",
                    "representation",
                    "input_channels",
                }
            },
            "configuration_id": (
                f"{winner['feature_set']}__{winner['representation']}__{strategy}"
            ),
            "stage": "imbalance",
            "imbalance_strategy": strategy,
        }
        for strategy in STAGE2_STRATEGIES
    ]


def read_development_assignments(
    protocol: dict[str, Any], well_id: str, split_seed: int
) -> pd.DataFrame:
    parts = []
    expected = protocol["well_protocols"][well_id]["split_seeds"][str(split_seed)][
        "assignment_sha256"
    ]
    for split in ("train", "validation"):
        path = (
            PROTOCOL_DIR
            / "center_assignments"
            / f"seed_{split_seed}"
            / well_id
            / f"{well_id}_{split}_centers.csv"
        )
        if sha256_file(path) != expected[split]:
            raise RuntimeError(f"Frozen assignment hash mismatch: {path}")
        parts.append(pd.read_csv(path, encoding="utf-8-sig"))
    return pd.concat(parts, ignore_index=True)


def summary_for(
    configuration: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    if len(frame) == 0:
        raise RuntimeError("Cannot summarize an empty configuration")
    return {
        **configuration,
        "wells": int(len(frame)),
        "validation_samples": int(frame["validation_samples"].sum()),
        "mean_per_well_macro_f1": float(frame["macro_f1"].mean()),
        "mean_per_well_balanced_accuracy": float(
            frame["balanced_accuracy"].mean()
        ),
        "mean_per_well_accuracy": float(frame["accuracy"].mean()),
        "pooled_accuracy": float(
            np.average(frame["accuracy"], weights=frame["validation_samples"])
        ),
        "runtime_seconds": float(frame["runtime_seconds"].sum()),
        "mean_best_epoch": float(frame["best_epoch"].mean()),
    }


def selection_rank(summary: dict[str, Any]) -> tuple[float, float, float, int, str]:
    return (
        float(summary["mean_per_well_macro_f1"]),
        float(summary["mean_per_well_balanced_accuracy"]),
        float(summary["mean_per_well_accuracy"]),
        -int(summary["input_channels"]),
        str(summary["configuration_id"]),
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal validation ablation requires the CUDA environment")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    optimizer_freeze = json.loads(
        OPTIMIZER_FREEZE_PATH.read_text(encoding="utf-8")
    )
    split_seed = int(protocol["split"]["primary_model_selection_split_seed"])
    wells = tuple(str(value) for value in protocol["research_scope"]["wells"])
    window_length = int(protocol["windows"]["length"])
    sets = feature_sets(protocol)
    stage1 = stage1_configurations(sets)
    candidate_dict = optimizer_freeze["methods"]["ga"]["candidate"]
    candidate = GRUSearchCandidate.from_dict(candidate_dict)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "FROZEN_BEFORE_VALIDATION_ABLATION",
        "scope": "training_and_validation_centers_only",
        "test_assignment_files_opened": False,
        "test_metrics_used_for_selection": False,
        "samples_shared_between_well_models": False,
        "model": "unidirectional_many_to_one_GRU",
        "candidate_source": "equal_budget_GA_validation_winner",
        "candidate": candidate_dict,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "optimizer_freeze_sha256": sha256_file(OPTIMIZER_FREEZE_PATH),
        "split_seed": split_seed,
        "training_seed_policy": "1701 plus zero-based well index",
        "balance_seed_policy": "27101 plus zero-based well index",
        "mutual_information_seed_policy": "37101 plus zero-based well index",
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "window_length": window_length,
        "wells": list(wells),
        "stage1_configurations": stage1,
        "stage1_selection_rule": (
            "maximum mean per-well macro-F1, then balanced accuracy, accuracy, "
            "and fewer input channels"
        ),
        "stage2_rule": (
            "compare unweighted, class-weighted cross-entropy, and SMOTE-Tomek "
            "using the frozen stage-1 feature representation"
        ),
        "expected_unique_configurations": 8,
        "expected_well_runs": 48,
        "mutual_information_scope": (
            "training target centers only; seven physical channels; categorical labels"
        ),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
    }
    if MANIFEST_PATH.is_file():
        if json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Existing validation-ablation manifest conflicts")
    else:
        write_json_atomic(MANIFEST_PATH, manifest)

    frame_cache: dict[str, pd.DataFrame] = {}
    assignment_cache: dict[str, pd.DataFrame] = {}
    window_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def windows_for(well_id: str, feature_set: str) -> dict[str, Any]:
        key = (well_id, feature_set)
        if key in window_cache:
            return window_cache[key]
        record = protocol["well_protocols"][well_id]
        if well_id not in frame_cache:
            source_path = PROTOCOL_DIR / str(record["source_snapshot"])
            if sha256_file(source_path) != record["source_snapshot_sha256"]:
                raise RuntimeError(f"Frozen source hash mismatch: {source_path}")
            frame_cache[well_id] = pd.read_csv(source_path, encoding="utf-8-sig")
        if well_id not in assignment_cache:
            assignment_cache[well_id] = read_development_assignments(
                protocol, well_id, split_seed
            )
        windows = build_random_center_windows(
            frame_cache[well_id],
            assignment_cache[well_id],
            sets[feature_set],
            window_length,
            requested_splits=("train", "validation"),
        )
        windows, _ = impute_from_training_windows(windows)
        window_cache[key] = windows
        return windows

    mi_rows = []
    for well_index, well_id in enumerate(wells):
        windows = windows_for(well_id, "extended_seven")
        classes = tuple(
            int(value)
            for value in protocol["well_protocols"][well_id]["included_classes"]
        )
        y_train, _ = remap_labels(windows["train"], classes)
        scores = center_mutual_information(
            windows["train"], y_train, random_state=MI_SEED_BASE + well_index
        )
        total = float(scores.sum())
        ranks = pd.Series(scores).rank(method="average", ascending=False)
        for feature_index, feature in enumerate(sets["extended_seven"]):
            mi_rows.append(
                {
                    "well_id": well_id,
                    "feature": feature,
                    "mutual_information": float(scores[feature_index]),
                    "normalized_mutual_information": (
                        float(scores[feature_index] / total) if total > 0 else 0.0
                    ),
                    "within_well_rank": float(ranks.iloc[feature_index]),
                    "training_centers": int(len(y_train)),
                    "random_state": MI_SEED_BASE + well_index,
                }
            )
    mi_frame = pd.DataFrame(mi_rows)
    mi_summary = (
        mi_frame.groupby("feature", as_index=False)
        .agg(
            mean_mutual_information=("mutual_information", "mean"),
            mean_normalized_mutual_information=(
                "normalized_mutual_information",
                "mean",
            ),
            mean_rank=("within_well_rank", "mean"),
            wells=("well_id", "nunique"),
        )
        .sort_values(
            ["mean_normalized_mutual_information", "feature"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    write_csv_atomic(OUTPUT_DIR / "mutual_information_per_well.csv", mi_frame)
    write_csv_atomic(OUTPUT_DIR / "mutual_information_summary.csv", mi_summary)

    existing_records = read_jsonl(RUN_LOG_PATH)
    existing_by_key: dict[str, dict[str, Any]] = {}
    for record in existing_records:
        run_key = str(record["run_key"])
        if run_key in existing_by_key:
            raise RuntimeError(f"Duplicate saved ablation run: {run_key}")
        existing_by_key[run_key] = record

    def evaluate(configuration: dict[str, Any]) -> list[dict[str, Any]]:
        configuration_records = []
        for well_index, well_id in enumerate(wells):
            run_key = f"{configuration['configuration_id']}|{well_id}"
            if run_key in existing_by_key:
                saved = existing_by_key[run_key]
                for field in (
                    "feature_set",
                    "representation",
                    "imbalance_strategy",
                ):
                    if saved[field] != configuration[field]:
                        raise RuntimeError(f"Saved run conflicts for {run_key}")
                configuration_records.append(saved)
                print(
                    f"Replayed {run_key}: macro-F1={saved['macro_f1']:.4f}",
                    flush=True,
                )
                continue

            windows = windows_for(well_id, str(configuration["feature_set"]))
            classes = tuple(
                int(value)
                for value in protocol["well_protocols"][well_id]["included_classes"]
            )
            y_train, mapping = remap_labels(windows["train"], classes)
            y_validation, _ = remap_labels(windows["validation"], classes)
            X_train, X_validation = prepare_per_well_inputs(
                windows["train"],
                windows["validation"],
                representation=str(configuration["representation"]),
            )
            balanced = apply_imbalance_strategy(
                X_train,
                y_train,
                str(configuration["imbalance_strategy"]),
                random_state=BALANCE_SEED_BASE + well_index,
            )
            torch.cuda.empty_cache()
            fit = fit_local_recurrent_with_validation(
                "gru",
                candidate,
                balanced.X,
                balanced.y,
                X_validation,
                y_validation,
                device=device,
                seed=TRAINING_SEED_BASE + well_index,
                batch_size=BATCH_SIZE,
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
                class_weights=balanced.class_weights,
            )
            history_path = (
                HISTORY_DIR / f"{configuration['configuration_id']}__{well_id}.csv"
            )
            write_csv_atomic(history_path, pd.DataFrame(fit.history))
            record = {
                "run_key": run_key,
                "configuration_id": configuration["configuration_id"],
                "stage": configuration["stage"],
                "feature_set": configuration["feature_set"],
                "features": configuration["features"],
                "representation": configuration["representation"],
                "input_channels": int(configuration["input_channels"]),
                "imbalance_strategy": configuration["imbalance_strategy"],
                "well_id": well_id,
                "global_to_local_class_mapping": {
                    str(key): int(value) for key, value in mapping.items()
                },
                "training_samples_before_balance": int(len(y_train)),
                "training_samples_after_balance": int(len(balanced.y)),
                "validation_samples": int(len(y_validation)),
                "macro_f1": fit.validation_macro_f1,
                "balanced_accuracy": fit.validation_balanced_accuracy,
                "accuracy": fit.validation_accuracy,
                "best_epoch": fit.best_epoch,
                "epochs_completed": fit.epochs_completed,
                "runtime_seconds": fit.runtime_seconds,
                "trainable_parameters": fit.trainable_parameters,
                "training_seed": TRAINING_SEED_BASE + well_index,
                "balance_audit": balanced.audit,
                "history_file": str(history_path.relative_to(PROJECT_DIR)),
                "preprocessing_fit_scope": "training_windows_only",
                "test_assignment_file_opened": False,
            }
            append_jsonl(RUN_LOG_PATH, record)
            existing_by_key[run_key] = record
            configuration_records.append(record)
            print(
                f"Completed {run_key}: macro-F1={record['macro_f1']:.4f}, "
                f"accuracy={record['accuracy']:.4f}",
                flush=True,
            )
        return configuration_records

    started = time.perf_counter()
    stage1_summaries = []
    for index, configuration in enumerate(stage1, start=1):
        print(f"Stage 1 configuration {index}/{len(stage1)}", flush=True)
        stage1_summaries.append(
            summary_for(configuration, evaluate(configuration))
        )
    stage1_winner = max(stage1_summaries, key=selection_rank)
    stage1_freeze = {
        "status": "FROZEN_AFTER_STAGE1_BEFORE_IMBALANCE_COMPARISON",
        "selection_uses": "training_and_validation_centers_only",
        "test_assignment_files_opened": False,
        "selection_rule": manifest["stage1_selection_rule"],
        "winner": stage1_winner,
        "all_stage1_summaries": stage1_summaries,
    }
    if STAGE1_FREEZE_PATH.is_file():
        if json.loads(STAGE1_FREEZE_PATH.read_text(encoding="utf-8")) != stage1_freeze:
            raise RuntimeError("Existing stage-1 freeze conflicts with recomputed results")
    else:
        write_json_atomic(STAGE1_FREEZE_PATH, stage1_freeze)
    print(
        "Stage 1 frozen: "
        f"{stage1_winner['configuration_id']} "
        f"macro-F1={stage1_winner['mean_per_well_macro_f1']:.4f}",
        flush=True,
    )

    stage2 = stage2_configurations(stage1_winner)
    stage2_summaries = [stage1_winner]
    for index, configuration in enumerate(stage2, start=1):
        print(f"Stage 2 new configuration {index}/{len(stage2)}", flush=True)
        stage2_summaries.append(
            summary_for(configuration, evaluate(configuration))
        )
    final_winner = max(stage2_summaries, key=selection_rank)
    all_summaries = stage1_summaries + stage2_summaries[1:]
    all_records = read_jsonl(RUN_LOG_PATH)
    expected_keys = {
        f"{configuration['configuration_id']}|{well_id}"
        for configuration in stage1 + stage2
        for well_id in wells
    }
    actual_keys = {str(record["run_key"]) for record in all_records}
    if actual_keys != expected_keys:
        missing = sorted(expected_keys.difference(actual_keys))
        extra = sorted(actual_keys.difference(expected_keys))
        raise RuntimeError(f"Ablation run set differs; missing={missing}, extra={extra}")
    completion = {
        "status": "COMPLETE_PENDING_INDEPENDENT_AUDIT",
        "test_assignment_files_opened": False,
        "test_metrics_used_for_selection": False,
        "well_runs": len(all_records),
        "unique_configurations": len(all_summaries),
        "stage1_winner": stage1_winner,
        "stage2_winner": final_winner,
        "all_configuration_summaries": all_summaries,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "stage1_freeze_sha256": sha256_file(STAGE1_FREEZE_PATH),
        "run_log_sha256": sha256_file(RUN_LOG_PATH),
        "mutual_information_per_well_sha256": sha256_file(
            OUTPUT_DIR / "mutual_information_per_well.csv"
        ),
        "mutual_information_summary_sha256": sha256_file(
            OUTPUT_DIR / "mutual_information_summary.csv"
        ),
    }
    write_json_atomic(COMPLETION_PATH, completion)
    print("Validation-only ablation: COMPLETE PENDING AUDIT")
    print(pd.DataFrame(all_summaries).to_string(index=False))
    print(f"Provisional winner: {final_winner['configuration_id']}")
    print(f"Completion: {COMPLETION_PATH}")


if __name__ == "__main__":
    main()
