from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gagru.residual_bigru import (
    center_conditioned_features,
    center_contrast_features,
    prepare_per_well_inputs,
)


def batch(values: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        X=values.astype(np.float32),
        wells=np.asarray(["W1"] * len(values)),
    )


def test_physical_and_engineered_representation_shapes() -> None:
    values = np.arange(4 * 9 * 7, dtype=float).reshape(4, 9, 7) + 1.0
    physical = prepare_per_well_inputs(batch(values), representation="physical")[0]
    engineered = prepare_per_well_inputs(batch(values), representation="engineered")[0]
    assert physical.shape == (4, 9, 7)
    assert engineered.shape == (4, 9, 17)
    assert np.isfinite(physical).all()
    assert np.isfinite(engineered).all()


def test_center_conditioned_representation_shapes_and_semantics() -> None:
    values = np.arange(4 * 9 * 7, dtype=float).reshape(4, 9, 7) + 1.0
    repeated = prepare_per_well_inputs(
        batch(values), representation="center_repeated"
    )[0]
    contrast = prepare_per_well_inputs(
        batch(values), representation="center_contrast"
    )[0]
    combined = prepare_per_well_inputs(
        batch(values), representation="center_repeated_contrast"
    )[0]
    assert repeated.shape == (4, 9, 24)
    assert contrast.shape == (4, 9, 24)
    assert combined.shape == (4, 9, 31)
    assert np.isfinite(repeated).all()
    assert np.isfinite(contrast).all()
    assert np.isfinite(combined).all()
    assert np.allclose(repeated[:, 1:, 17:], repeated[:, :-1, 17:])
    raw_contrast = center_contrast_features(values)
    raw_combined = center_conditioned_features(values, include_contrast=True)
    assert np.allclose(raw_contrast[:, 4, 17:], 0.0)
    assert np.allclose(raw_combined[:, 4, 24:], 0.0)


def test_unknown_representation_is_rejected() -> None:
    values = np.ones((2, 9, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="representation"):
        prepare_per_well_inputs(batch(values), representation="unknown")
