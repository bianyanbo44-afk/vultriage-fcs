"""Cross-fitted density-ratio estimation for project-shift experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


def _fold_id(row_id: str, salt: str, folds: int) -> int:
    if folds < 2:
        raise ValueError("Cross-fitting needs at least two folds")
    digest = hashlib.sha256(f"{salt}|{row_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


@dataclass(frozen=True)
class CrossfitDensityRatio:
    row_ids: tuple[str, ...]
    domain_labels: np.ndarray
    fold_ids: np.ndarray
    target_probabilities: np.ndarray
    raw_ratios: np.ndarray
    diagnostics: dict[str, object]


def crossfit_density_ratio(
    features: sparse.spmatrix,
    row_ids: list[str],
    domain_labels: np.ndarray,
    *,
    folds: int,
    alpha: float,
    epochs: int,
    seed: int,
    salt: str,
) -> CrossfitDensityRatio:
    """Estimate target/source ratios with deterministic balanced cross-fitting.

    The domain classifier is trained on equal source/target samples within every
    training complement, so the density ratio is the out-of-fold target odds.
    All held-out examples receive a probability from a model that did not train
    on that example. Sampling is deterministic for a fixed seed and salt.
    """

    labels = np.asarray(domain_labels, dtype=int)
    if features.shape[0] != len(row_ids) or labels.shape != (len(row_ids),):
        raise ValueError("Features, row IDs, and domain labels must align")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both source (0) and target (1) domains are required")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Domain row IDs must be unique")

    fold_ids = np.asarray([_fold_id(row_id, salt, folds) for row_id in row_ids])
    probabilities = np.full(len(row_ids), np.nan, dtype=float)
    fold_records: list[dict[str, object]] = []

    for fold in range(folds):
        train_mask = fold_ids != fold
        held_mask = fold_ids == fold
        if not held_mask.any():
            raise ValueError(f"Cross-fit fold {fold} is empty")
        source_train = np.flatnonzero(train_mask & (labels == 0))
        target_train = np.flatnonzero(train_mask & (labels == 1))
        balanced_n = min(len(source_train), len(target_train))
        if balanced_n == 0:
            raise ValueError(f"Cross-fit fold {fold} has an empty training domain")
        rng = np.random.default_rng(seed + 104729 * fold)
        if len(source_train) > balanced_n:
            source_train = rng.choice(source_train, balanced_n, replace=False)
        if len(target_train) > balanced_n:
            target_train = rng.choice(target_train, balanced_n, replace=False)
        train_indices = np.concatenate([source_train, target_train])
        rng.shuffle(train_indices)

        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=float(alpha),
            random_state=seed + fold,
            max_iter=int(epochs),
            tol=None,
        )
        model.fit(features[train_indices], labels[train_indices])
        held_indices = np.flatnonzero(held_mask)
        probabilities[held_indices] = model.predict_proba(features[held_indices])[:, 1]
        fold_records.append(
            {
                "fold": fold,
                "balanced_training_per_domain": int(balanced_n),
                "held_source": int(np.sum(labels[held_indices] == 0)),
                "held_target": int(np.sum(labels[held_indices] == 1)),
                "effective_target_prior": 0.5,
                "prior_odds_correction": 1.0,
            }
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Cross-fitting failed to produce every probability")
    epsilon = np.finfo(float).eps
    bounded = np.clip(probabilities, epsilon, 1.0 - epsilon)
    raw_ratios = bounded / (1.0 - bounded)
    diagnostics: dict[str, object] = {
        "crossfit_folds": int(folds),
        "out_of_fold": True,
        "domain_auroc": float(roc_auc_score(labels, probabilities)),
        "domain_average_precision": float(
            average_precision_score(labels, probabilities)
        ),
        "source_n": int(np.sum(labels == 0)),
        "target_n": int(np.sum(labels == 1)),
        "folds": fold_records,
    }
    return CrossfitDensityRatio(
        row_ids=tuple(row_ids),
        domain_labels=labels,
        fold_ids=fold_ids,
        target_probabilities=probabilities,
        raw_ratios=raw_ratios,
        diagnostics=diagnostics,
    )


def clipped_weight_summary(raw_ratios: np.ndarray, upper_clip: float) -> dict[str, object]:
    ratios = np.asarray(raw_ratios, dtype=float)
    if ratios.ndim != 1 or not np.isfinite(ratios).all() or (ratios < 0).any():
        raise ValueError("Density ratios must be finite and nonnegative")
    if upper_clip <= 1:
        raise ValueError("Upper clip must exceed one")
    clipped = np.clip(ratios, 1.0 / upper_clip, upper_clip)
    squared = float(np.square(clipped).sum())
    ess = 0.0 if squared == 0 else float(clipped.sum() ** 2 / squared)
    quantiles = np.quantile(clipped, [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])
    return {
        "upper_clip": float(upper_clip),
        "lower_clip": float(1.0 / upper_clip),
        "ess": ess,
        "quantiles": {
            name: float(value)
            for name, value in zip(
                ("min", "p01", "p25", "p50", "p75", "p99", "max"),
                quantiles,
            )
        },
        "clipped_low": int(np.sum(ratios < 1.0 / upper_clip)),
        "clipped_high": int(np.sum(ratios > upper_clip)),
    }
