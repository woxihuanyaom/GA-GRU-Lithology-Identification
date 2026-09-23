from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.feature_selection import mutual_info_classif

from .preprocessing import compute_class_weights
from .residual_bigru import physical_features


BALANCE_STRATEGIES = ("unweighted", "class_weighted", "smote_tomek")


@dataclass(frozen=True)
class BalancedTrainingArrays:
    X: np.ndarray
    y: np.ndarray
    class_weights: np.ndarray | None
    audit: dict[str, object]


def class_count_dict(y: np.ndarray) -> dict[str, int]:
    labels = np.asarray(y, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("Labels must be a non-empty one-dimensional array")
    classes, counts = np.unique(labels, return_counts=True)
    expected = np.arange(int(classes.max()) + 1)
    if not np.array_equal(classes, expected):
        raise ValueError("Local labels must be contiguous and start at zero")
    return {str(int(label)): int(count) for label, count in zip(classes, counts)}


def apply_imbalance_strategy(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str,
    *,
    random_state: int,
) -> BalancedTrainingArrays:
    values = np.asarray(X, dtype=np.float32)
    labels = np.asarray(y, dtype=np.int64)
    if values.ndim != 3 or labels.ndim != 1 or len(values) != len(labels):
        raise ValueError("X and y must contain matching sequence windows")
    if not np.isfinite(values).all():
        raise ValueError("Training inputs must be finite before balancing")
    if strategy not in BALANCE_STRATEGIES:
        raise ValueError(f"Unknown imbalance strategy: {strategy}")

    before = class_count_dict(labels)
    class_weights: np.ndarray | None = None
    smote_neighbors: int | None = None
    if strategy == "unweighted":
        transformed_X = values.copy()
        transformed_y = labels.copy()
    elif strategy == "class_weighted":
        transformed_X = values.copy()
        transformed_y = labels.copy()
        class_weights = compute_class_weights(
            labels, num_classes=len(np.unique(labels))
        )
    else:
        minimum_count = min(before.values())
        if minimum_count < 2:
            raise ValueError("SMOTE-Tomek requires at least two samples per class")
        smote_neighbors = min(5, minimum_count - 1)
        sampler = SMOTETomek(
            sampling_strategy="auto",
            random_state=random_state,
            smote=SMOTE(k_neighbors=smote_neighbors, random_state=random_state),
            tomek=TomekLinks(sampling_strategy="all"),
        )
        flat_X, transformed_y = sampler.fit_resample(
            values.reshape(len(values), -1), labels
        )
        transformed_X = flat_X.reshape(
            -1, values.shape[1], values.shape[2]
        ).astype(np.float32)
        transformed_y = np.asarray(transformed_y, dtype=np.int64)

    if not np.isfinite(transformed_X).all():
        raise RuntimeError(f"{strategy} produced non-finite training inputs")
    after = class_count_dict(transformed_y)
    return BalancedTrainingArrays(
        X=np.asarray(transformed_X, dtype=np.float32),
        y=np.asarray(transformed_y, dtype=np.int64),
        class_weights=class_weights,
        audit={
            "strategy": strategy,
            "random_state": int(random_state),
            "training_windows_before": int(len(labels)),
            "training_windows_after": int(len(transformed_y)),
            "class_counts_before": before,
            "class_counts_after": after,
            "class_weights": (
                None
                if class_weights is None
                else [float(value) for value in class_weights]
            ),
            "smote_k_neighbors": smote_neighbors,
            "tomek_sampling_strategy": "all" if strategy == "smote_tomek" else None,
            "resampling_scope": "training_windows_only",
        },
    )


def center_mutual_information(
    windows: object,
    y: np.ndarray,
    *,
    random_state: int,
) -> np.ndarray:
    values = physical_features(np.asarray(windows.X))
    labels = np.asarray(y, dtype=np.int64)
    if len(values) != len(labels):
        raise ValueError("Mutual-information inputs and labels differ in length")
    if not np.isfinite(values).all():
        raise ValueError("Mutual-information inputs must be finite")
    center = values[:, values.shape[1] // 2, :]
    scores = mutual_info_classif(
        center,
        labels,
        discrete_features=False,
        random_state=random_state,
    )
    return np.asarray(scores, dtype=float)
