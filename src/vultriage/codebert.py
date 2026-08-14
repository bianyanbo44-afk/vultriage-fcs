"""Frozen CodeBERT utilities for extension-v2 representation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_ID = "microsoft/codebert-base"
MODEL_REVISION = "3b0952feddeffad0063f274080e3c23d75e7eb39"


def masked_mean_pool(
    hidden: Any, attention_mask: Any, special_tokens_mask: Any
) -> Any:
    """Mean-pool final states over attended, non-special tokens."""

    import torch

    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, tokens, hidden_size]")
    if attention_mask.shape != hidden.shape[:2]:
        raise ValueError("attention mask does not align with hidden states")
    if special_tokens_mask.shape != hidden.shape[:2]:
        raise ValueError("special-token mask does not align with hidden states")
    keep = attention_mask.to(dtype=torch.bool) & ~special_tokens_mask.to(
        dtype=torch.bool
    )
    counts = keep.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("at least one sequence contains no non-special token")
    weights = keep.unsqueeze(-1).to(dtype=hidden.dtype)
    return (hidden * weights).sum(dim=1) / counts.unsqueeze(-1).to(hidden.dtype)


def logistic_head(c_value: float, seed: int) -> Pipeline:
    """Create the exact preregistered standardized logistic head."""

    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "head",
                LogisticRegression(
                    C=float(c_value),
                    solver="liblinear",
                    penalty="l2",
                    dual=False,
                    fit_intercept=True,
                    class_weight="balanced",
                    max_iter=2000,
                    tol=1e-6,
                    random_state=int(seed),
                ),
            ),
        ]
    )


@dataclass(frozen=True)
class SelectedHead:
    model: Pipeline
    c_value: float
    validation_pr_auc: float


def fit_logistic_head(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    validation_embeddings: np.ndarray,
    validation_labels: np.ndarray,
    c_grid: Iterable[float],
    seed: int,
) -> SelectedHead:
    """Select C on source validation PR-AUC, then refit on train+validation."""

    train_x = np.asarray(train_embeddings, dtype=np.float32)
    validation_x = np.asarray(validation_embeddings, dtype=np.float32)
    train_y = np.asarray(train_labels, dtype=int)
    validation_y = np.asarray(validation_labels, dtype=int)
    if train_x.ndim != 2 or validation_x.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if train_x.shape[1] != validation_x.shape[1]:
        raise ValueError("train and validation embedding widths differ")
    if set(np.unique(train_y)) != {0, 1} or set(np.unique(validation_y)) != {0, 1}:
        raise ValueError("train and validation must each contain both classes")
    candidates = sorted({float(value) for value in c_grid})
    if not candidates or candidates[0] <= 0:
        raise ValueError("C grid must contain positive values")

    # Fit the representation transform exactly once on source-train data.
    # The frozen protocol requires validation, calibration, and target rows to
    # be transformed by this same scaler; refitting it on train+validation
    # would leak validation distribution into the final head.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    validation_scaled = scaler.transform(validation_x)

    best_c: float | None = None
    best_score = -np.inf
    for c_value in candidates:
        candidate = LogisticRegression(
            C=float(c_value),
            solver="liblinear",
            penalty="l2",
            dual=False,
            fit_intercept=True,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-6,
            random_state=int(seed),
        )
        candidate.fit(train_scaled, train_y)
        probability = candidate.predict_proba(validation_scaled)[:, 1]
        score = float(average_precision_score(validation_y, probability))
        if score > best_score or (
            np.isclose(score, best_score, rtol=0.0, atol=1e-15)
            and (best_c is None or c_value < best_c)
        ):
            best_c = c_value
            best_score = score
    assert best_c is not None
    final_head = LogisticRegression(
        C=float(best_c),
        solver="liblinear",
        penalty="l2",
        dual=False,
        fit_intercept=True,
        class_weight="balanced",
        max_iter=2000,
        tol=1e-6,
        random_state=int(seed),
    )
    final_head.fit(
        np.concatenate([train_scaled, validation_scaled], axis=0),
        np.concatenate([train_y, validation_y], axis=0),
    )
    final = Pipeline([("scale", scaler), ("head", final_head)])
    return SelectedHead(final, best_c, best_score)


def fit_logistic_head_scaled(
    train_scaled: np.ndarray,
    train_labels: np.ndarray,
    validation_scaled: np.ndarray,
    validation_labels: np.ndarray,
    c_grid: Iterable[float],
    seed: int,
    scaler: StandardScaler,
) -> SelectedHead:
    """Fit the same frozen head when source-train scaling is already cached.

    The runner uses this helper to share the source-train-only StandardScaler
    and transformed matrices across technical seeds.  It preserves the
    preregistered liblinear solver and tie-breaking while avoiding repeated
    memmap copies; ``scaler`` is still attached to the returned pipeline so
    downstream calibration and target predictions use the frozen transform.
    """

    train_x = np.asarray(train_scaled)
    validation_x = np.asarray(validation_scaled)
    train_y = np.asarray(train_labels, dtype=int)
    validation_y = np.asarray(validation_labels, dtype=int)
    if train_x.ndim != 2 or validation_x.ndim != 2:
        raise ValueError("scaled embeddings must be two-dimensional")
    if train_x.shape[1] != validation_x.shape[1]:
        raise ValueError("scaled train and validation embedding widths differ")
    if set(np.unique(train_y)) != {0, 1} or set(np.unique(validation_y)) != {0, 1}:
        raise ValueError("train and validation must each contain both classes")
    candidates = sorted({float(value) for value in c_grid})
    if not candidates or candidates[0] <= 0:
        raise ValueError("C grid must contain positive values")

    best_c: float | None = None
    best_score = -np.inf
    for c_value in candidates:
        candidate = LogisticRegression(
            C=float(c_value),
            solver="liblinear",
            penalty="l2",
            dual=False,
            fit_intercept=True,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-6,
            random_state=int(seed),
        )
        candidate.fit(train_x, train_y)
        probability = candidate.predict_proba(validation_x)[:, 1]
        score = float(average_precision_score(validation_y, probability))
        if score > best_score or (
            np.isclose(score, best_score, rtol=0.0, atol=1e-15)
            and (best_c is None or c_value < best_c)
        ):
            best_c = c_value
            best_score = score
    assert best_c is not None
    final_head = LogisticRegression(
        C=float(best_c),
        solver="liblinear",
        penalty="l2",
        dual=False,
        fit_intercept=True,
        class_weight="balanced",
        max_iter=2000,
        tol=1e-6,
        random_state=int(seed),
    )
    final_head.fit(
        np.concatenate([train_x, validation_x], axis=0),
        np.concatenate([train_y, validation_y], axis=0),
    )
    final = Pipeline([("scale", scaler), ("head", final_head)])
    return SelectedHead(final, best_c, best_score)
