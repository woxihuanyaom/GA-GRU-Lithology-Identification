"""Training-fold-only feature interpretation helpers."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
from sklearn.feature_selection import mutual_info_classif

from .errors import DataValidationError
from .preprocessing import CurvePreprocessor
from .windows import WindowedDataset


def preprocessor_from_dict(value: dict) -> CurvePreprocessor:
    """Rebuild the frozen log10 preprocessor stored beside an outer model."""

    features = tuple(str(name) for name in value["feature_names"])
    means = np.asarray(value["means"], dtype=np.float64)
    scales = np.asarray(value["scales"], dtype=np.float64)
    indices = tuple(int(index) for index in value["resistivity_indices"])
    fitted_wells = tuple(str(well) for well in value["fitted_wells"])
    if means.shape != (len(features),) or scales.shape != (len(features),):
        raise DataValidationError("Saved preprocessor statistics do not match its feature list")
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise DataValidationError("Saved preprocessor statistics are invalid")
    return CurvePreprocessor(
        feature_names=features,
        resistivity_indices=indices,
        means=means,
        scales=scales,
        fitted_wells=fitted_wells,
    )


def center_mutual_information(
    dataset: WindowedDataset,
    *,
    random_state: int,
    n_neighbors: int = 3,
) -> np.ndarray:
    """Estimate MI from each transformed curve at the center sample."""

    if dataset.X.shape[1] % 2 != 1:
        raise DataValidationError("Mutual-information windows must have an odd length")
    center = dataset.X[:, dataset.window_length // 2, :]
    values = mutual_info_classif(
        center,
        dataset.y.astype(np.int64, copy=False),
        discrete_features=False,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (dataset.X.shape[2],) or not np.isfinite(values).all():
        raise DataValidationError("Mutual-information estimator returned invalid values")
    return values


def permute_feature(
    dataset: WindowedDataset,
    feature_index: int,
    *,
    random_state: int,
) -> WindowedDataset:
    """Permute one feature across windows while preserving each window profile."""

    if feature_index < 0 or feature_index >= dataset.X.shape[2]:
        raise DataValidationError(f"Invalid feature index: {feature_index}")
    rng = np.random.default_rng(random_state)
    transformed = np.array(dataset.X, dtype=np.float32, copy=True)
    order = rng.permutation(len(transformed))
    transformed[:, :, feature_index] = dataset.X[order, :, feature_index]
    return replace(dataset, X=transformed)
