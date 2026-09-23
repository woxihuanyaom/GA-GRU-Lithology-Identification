from __future__ import annotations

import numpy as np
import pytest
import torch

from gagru.local_recurrent import (
    ARCHITECTURES,
    _cross_entropy_criterion,
    _training_classification_loss,
    build_local_recurrent,
    fit_local_recurrent_with_validation,
)
from gagru.search import GRUSearchCandidate


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_local_recurrent_output_shape(architecture: str) -> None:
    candidate = GRUSearchCandidate(
        hidden_size=8,
        num_layers=2,
        learning_rate=1e-3,
        dropout=0.2,
        weight_decay=1e-4,
    )
    X = np.ones((4, 9, 7), dtype=np.float32)
    model = build_local_recurrent(
        architecture,
        candidate,
        X,
        output_size=5,
        device=torch.device("cpu"),
    )
    assert model(torch.from_numpy(X)).shape == (4, 5)
    assert model.trainable_parameters > 0


def test_weighted_cross_entropy_validates_shape() -> None:
    criterion = _cross_entropy_criterion(
        3, np.asarray([1.0, 2.0, 3.0]), torch.device("cpu")
    )
    assert criterion.weight is not None
    assert criterion.weight.tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="shape"):
        _cross_entropy_criterion(3, np.ones(2), torch.device("cpu"))


@pytest.mark.parametrize("architecture", ("center_residual_gru", "local_residual_gru"))
def test_residual_branch_starts_as_plain_gru_logits(architecture: str) -> None:
    candidate = GRUSearchCandidate(8, 2, 1e-3, 0.2, 1e-4)
    X = np.ones((4, 9, 7), dtype=np.float32)
    model = build_local_recurrent(
        architecture,
        candidate,
        X,
        output_size=5,
        device=torch.device("cpu"),
    )
    model.eval()
    values = torch.from_numpy(X)
    output, _ = model.recurrent(values)
    expected = model.classifier(output[:, -1, :])
    assert torch.allclose(model(values), expected)


def test_local_residual_gru_rejects_short_windows() -> None:
    candidate = GRUSearchCandidate(8, 2, 1e-3, 0.2, 1e-4)
    X = np.ones((4, 1, 7), dtype=np.float32)
    model = build_local_recurrent(
        "local_residual_gru",
        candidate,
        X,
        output_size=5,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="at least three"):
        model(torch.from_numpy(X))


def test_focal_loss_reduces_easy_example_contribution() -> None:
    logits = torch.tensor([[8.0, -8.0], [0.1, -0.1]])
    targets = torch.tensor([0, 0])
    criterion = _cross_entropy_criterion(2, None, torch.device("cpu"))
    cross_entropy = _training_classification_loss(
        logits, targets, criterion, focal_gamma=0.0
    )
    focal = _training_classification_loss(logits, targets, criterion, focal_gamma=2.0)
    assert 0 < focal < cross_entropy
    with pytest.raises(ValueError, match="non-negative"):
        _training_classification_loss(logits, targets, criterion, focal_gamma=-1.0)


def test_validation_metric_labels_must_cover_observed_classes() -> None:
    candidate = GRUSearchCandidate(8, 1, 1e-3, 0.0, 1e-4)
    X_train = np.ones((6, 9, 3), dtype=np.float32)
    y_train = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    X_validation = np.ones((2, 9, 3), dtype=np.float32)
    y_validation = np.asarray([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="omit an observed"):
        fit_local_recurrent_with_validation(
            "gru",
            candidate,
            X_train,
            y_train,
            X_validation,
            y_validation,
            device=torch.device("cpu"),
            seed=17,
            batch_size=4,
            max_epochs=1,
            patience=1,
            validation_metric_labels=np.asarray([0]),
        )
