"""Fit and seal the preregistered PrimeVul-only extension-v2 support gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vultriage.data import sha256


FEATURES = (
    "log1p_total_ess",
    "log1p_safe_ess",
    "log1p_vulnerable_ess",
    "maximum_infinity_mass",
    "p99_infinity_mass",
    "domain_auroc",
    "lower_clipping_fraction",
    "upper_clipping_fraction",
)


def gate_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    penalty="l2",
                    fit_intercept=True,
                    class_weight="balanced",
                    max_iter=2000,
                    tol=1e-6,
                    random_state=20260814,
                ),
            ),
        ]
    )


def labels_for_positions(
    archive_positions: np.ndarray, archive_labels: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    order = np.argsort(archive_positions)
    sorted_positions = archive_positions[order]
    indices = np.searchsorted(sorted_positions, positions)
    if (indices >= len(sorted_positions)).any() or not np.array_equal(
        sorted_positions[indices], positions
    ):
        raise ValueError("calibration position is absent from source label package")
    return archive_labels[order][indices].astype(int, copy=False)


def project_diagnostics(
    prediction_dir: Path, input_dir: Path, group: str
) -> dict[str, float]:
    metadata = json.loads(
        (prediction_dir / "predictions" / group / "seed-13.json").read_text(
            encoding="utf-8"
        )
    )
    sensitivity = metadata["weight_sensitivity"]["20.0"]
    calibration = sensitivity["calibration"]
    source_n = int(metadata["domain_diagnostics"]["source_n"])
    prediction = np.load(prediction_dir / "predictions" / group / "seed-13.npz")
    source = np.load(input_dir / "source_labels" / f"{group}.npz")
    calibration_labels = labels_for_positions(
        source["positions"], source["labels"], prediction["calibration_positions"]
    )
    calibration_weights = np.clip(
        prediction["calibration_raw_ratio"].astype(float), 0.05, 20.0
    )
    target_weights = np.clip(
        prediction["target_raw_ratio"].astype(float), 0.05, 20.0
    )
    per_target = np.zeros(len(target_weights), dtype=float)
    for label in (0, 1):
        mass = float(calibration_weights[calibration_labels == label].sum())
        per_target = np.maximum(per_target, target_weights / (mass + target_weights))
    return {
        "domain_auroc": float(metadata["domain_diagnostics"]["domain_auroc"]),
        "lower_clipping_fraction": float(calibration["clipped_low"] / source_n),
        "upper_clipping_fraction": float(calibration["clipped_high"] / source_n),
        "maximum_infinity_mass": float(per_target.max()),
        "p99_infinity_mass": float(np.quantile(per_target, 0.99)),
    }


def development_frame(
    metrics: Path, prediction_dir: Path, input_dir: Path
) -> pd.DataFrame:
    frame = pd.read_csv(metrics)
    raw = frame[frame["method"] == "estimated_weight_no_support_clip_20"]
    raw = raw.groupby(
        ["target_group", "alpha_vulnerable", "alpha_safe"], as_index=False
    ).agg(
        vulnerable_violation=("vulnerable_violation", "mean"),
        safe_violation=("safe_violation", "mean"),
        raw_singleton_coverage=("singleton_coverage", "mean"),
    )
    support = frame[frame["method"] == "vultriage_clip_20"].copy()
    support["support_json"] = support["support"].map(json.loads)
    support = support.groupby(
        ["target_group", "alpha_vulnerable", "alpha_safe"], as_index=False
    ).first()
    support["total_ess"] = support["support_json"].map(lambda item: item["total_ess"])
    support["safe_ess"] = support["support_json"].map(
        lambda item: item["class_ess"]["0"]
    )
    support["vulnerable_ess"] = support["support_json"].map(
        lambda item: item["class_ess"]["1"]
    )
    merged = raw.merge(
        support[
            [
                "target_group",
                "alpha_vulnerable",
                "alpha_safe",
                "total_ess",
                "safe_ess",
                "vulnerable_ess",
            ]
        ],
        on=["target_group", "alpha_vulnerable", "alpha_safe"],
        validate="one_to_one",
    )
    diagnostics = pd.DataFrame(
        [
            {
                "target_group": group,
                **project_diagnostics(prediction_dir, input_dir, group),
            }
            for group in sorted(merged["target_group"].unique())
        ]
    )
    merged = merged.merge(diagnostics, on="target_group", validate="many_to_one")
    merged["log1p_total_ess"] = np.log1p(merged["total_ess"])
    merged["log1p_safe_ess"] = np.log1p(merged["safe_ess"])
    merged["log1p_vulnerable_ess"] = np.log1p(merged["vulnerable_ess"])
    merged["relative_violation"] = np.maximum(
        merged["vulnerable_violation"] / merged["alpha_vulnerable"],
        merged["safe_violation"] / merged["alpha_safe"],
    )
    merged["severe_violation"] = (merged["relative_violation"] > 0.5).astype(int)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    frame = development_frame(args.metrics, args.predictions, args.inputs)
    x = frame[list(FEATURES)].to_numpy(float)
    y = frame["severe_violation"].to_numpy(int)
    groups = frame["target_group"].to_numpy(str)
    if not np.isfinite(x).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("gate development matrix is nonfinite or single-class")

    probabilities = np.full(len(frame), np.nan)
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        model = gate_model()
        model.fit(x[train], y[train])
        probabilities[test] = model.predict_proba(x[test])[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("leave-one-project-out gate scores are incomplete")
    threshold = 0.5
    frame["crossfit_severe_probability"] = probabilities
    frame["crossfit_gate_pass"] = probabilities < threshold
    frame["crossfit_predicted_severe"] = probabilities >= threshold
    final = gate_model()
    final.fit(x, y)

    development_path = args.output / "primevul_gate_development.csv"
    model_path = args.output / "support_gate.joblib"
    frame.to_csv(development_path, index=False)
    joblib.dump(final, model_path, compress=3)
    scaler = final.named_steps["scale"]
    classifier = final.named_steps["model"]
    result = {
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": config["protocol_version"],
        "config_sha256": sha256(args.config),
        "input_metrics_sha256": sha256(args.metrics),
        "prediction_seal_sha256": sha256(args.predictions / "prediction_seal.json"),
        "source_label_hash_manifest_sha256": sha256(args.inputs / "hashes.json"),
        "projects": int(frame["target_group"].nunique()),
        "development_rows": len(frame),
        "severe_rows": int(y.sum()),
        "nonsevere_rows": int((1 - y).sum()),
        "features": list(FEATURES),
        "severe_relative_violation_threshold": 0.5,
        "probability_threshold": threshold,
        "crossfit_group": "target_group",
        "crossfit_auroc": float(roc_auc_score(y, probabilities)),
        "crossfit_auprc": float(average_precision_score(y, probabilities)),
        "crossfit_accuracy": float(np.mean((probabilities >= threshold) == y)),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficient": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "model_sha256": sha256(model_path),
        "development_csv_sha256": sha256(development_path),
        "detection_outcomes_used": "PrimeVul v1 only",
        "diversevul_model_outputs_used": False,
        "script_sha256": sha256(Path(__file__)),
    }
    (args.output / "gate_seal.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
