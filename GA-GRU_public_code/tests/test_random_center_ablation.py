from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gagru.random_center_ablation import (
    apply_imbalance_strategy,
    center_mutual_information,
)


def example_arrays() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    labels = np.repeat(np.arange(3), [18, 10, 7])
    values = rng.lognormal(size=(len(labels), 5, 5)).astype(np.float32)
    values[:, :, 3:] = rng.normal(size=(len(labels), 5, 2))
    return values, labels


@pytest.mark.parametrize("strategy", ("unweighted", "class_weighted", "smote_tomek"))
def test_imbalance_strategies_preserve_valid_sequences(strategy: str) -> None:
    X, y = example_arrays()
    result = apply_imbalance_strategy(X, y, strategy, random_state=17)
    assert result.X.ndim == 3
    assert len(result.X) == len(result.y)
    assert set(result.y) == {0, 1, 2}
    assert np.isfinite(result.X).all()
    if strategy == "class_weighted":
        assert result.class_weights is not None
        assert result.class_weights.shape == (3,)
    else:
        assert result.class_weights is None


def test_center_mutual_information_uses_one_row_per_target() -> None:
    X, y = example_arrays()
    windows = SimpleNamespace(X=X)
    scores = center_mutual_information(windows, y, random_state=17)
    assert scores.shape == (5,)
    assert np.all(scores >= 0)
