"""Controlled known-density diagnostics; no real-project safety guarantee.

Class labels never enter domain fitting. Independent random repetitions expose
the separate roles of density estimation, clipping and conditional-label shift.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def regime_parameters(regime: dict) -> tuple[float | None, bool]:
    specification = regime["target_distribution"]
    if specification == "Normal(0.5,1)":
        return 0.5, False
    if specification == "Normal(1.5,1)":
        return 1.5, False
    if specification == "0.5*Normal(-2,1)+0.5*Normal(2,1)":
        return None, True
    raise ValueError("Unknown predeclared target distribution")


def sample_target(rng: np.random.Generator, n: int, regime: dict) -> np.ndarray:
    mu, mixture = regime_parameters(regime)
    return rng.normal(size=n) + (rng.choice([-2.0, 2.0], n) if mixture else mu)


def exact_ratio(x: np.ndarray, regime: dict) -> np.ndarray:
    mu, mixture = regime_parameters(regime)
    x = np.asarray(x, dtype=float)
    if mixture:
        return 0.5 * np.exp(2 * x - 2) + 0.5 * np.exp(-2 * x - 2)
    return np.exp(mu * x - 0.5 * mu ** 2)


def target_density(x: np.ndarray, regime: dict) -> np.ndarray:
    mu, mixture = regime_parameters(regime)
    return 0.5 * norm.pdf(x, -2, 1) + 0.5 * norm.pdf(x, 2, 1) if mixture else norm.pdf(x, mu, 1)


def class_sets(source_score: np.ndarray, labels: np.ndarray, source_weights: np.ndarray,
               target_score: np.ndarray, target_weights: np.ndarray,
               alpha_s: float, alpha_v: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized test-mass quantiles, independently tested against the v3 routine."""
    source_score, target_score = np.asarray(source_score), np.asarray(target_score)
    labels, source_weights, target_weights = np.asarray(labels), np.asarray(source_weights), np.asarray(target_weights)
    if (source_score.ndim != 1 or target_score.ndim != 1 or labels.shape != source_score.shape
            or source_weights.shape != source_score.shape or target_weights.shape != target_score.shape
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(source_weights).all()
            or not np.isfinite(target_weights).all() or np.any(source_weights <= 0)
            or np.any(target_weights <= 0)):
        raise ValueError("Finite aligned scores/labels/positive weights required")
    thresholds = np.full((len(target_score), 2), np.inf)
    sets = np.zeros_like(thresholds, dtype=bool)
    for y, alpha in ((0, alpha_s), (1, alpha_v)):
        mask = labels == y
        if not mask.any() or not 0 < alpha < 1:
            raise ValueError("Both calibration classes and valid budgets are required")
        scores = source_score[mask] if y == 0 else 1 - source_score[mask]
        order = np.argsort(scores, kind="stable")
        cumulative = np.cumsum(source_weights[mask][order])
        index = np.searchsorted(cumulative, (1 - alpha) * (cumulative[-1] + target_weights), side="left")
        finite = index < len(order)
        thresholds[finite, y] = scores[order][index[finite]]
        score = target_score if y == 0 else 1 - target_score
        sets[:, y] = score <= thresholds[:, y]
    return sets, thresholds


def action_metrics(sets: np.ndarray, thresholds: np.ndarray, labels: np.ndarray) -> dict:
    actions = np.full(len(labels), 2, dtype=np.int8)
    singleton = sets.sum(axis=1) == 1
    actions[singleton] = sets[singleton].argmax(axis=1)
    safe, vulnerable = labels == 0, labels == 1
    auto = actions != 2
    error = auto & (actions != labels)
    return {
        "miscoverage_safe": float(np.mean(~sets[safe, 0])),
        "miscoverage_vulnerable": float(np.mean(~sets[vulnerable, 1])),
        "automatic_miss": float(np.mean(actions[vulnerable] == 0)),
        "automatic_false_alert": float(np.mean(actions[safe] == 1)),
        "automation_fraction": float(np.mean(auto)),
        "automatic_safe_fraction": float(np.mean(actions == 0)),
        "automatic_vulnerable_fraction": float(np.mean(actions == 1)),
        "review_fraction": float(np.mean(~auto)),
        "selective_error": float(np.mean(error[auto])) if auto.any() else None,
        "infinite_safe_fraction": float(np.mean(np.isinf(thresholds[:, 0]))),
        "infinite_vulnerable_fraction": float(np.mean(np.isinf(thresholds[:, 1]))),
    }


def loss_decomposition(grid: np.ndarray, regime: dict, weights: np.ndarray) -> dict:
    source = norm.pdf(grid)
    target = target_density(grid, regime)
    integrate = lambda values: float(np.trapezoid(values, grid))
    normalizer = integrate(source * weights)
    weighted_density = source * weights / normalizer
    eta_source = expit(2 * grid - 2)
    eta_target = expit(2 * grid - 2 + regime["target_conditional_delta"])
    label = expit(1.3 * grid - 1.5) >= 0.5
    loss_source = np.where(label, 1 - eta_source, eta_source)
    loss_target = np.where(label, 1 - eta_target, eta_target)
    risk_target = integrate(target * loss_target)
    risk_source_weighted = integrate(weighted_density * loss_source)
    covariate_tv = 0.5 * integrate(np.abs(weighted_density - target))
    conditional_tv = integrate(target * np.abs(eta_target - eta_source))
    return {"fixed_target_risk": risk_target, "fixed_weighted_source_risk": risk_source_weighted,
            "fixed_risk_absolute_gap": abs(risk_target - risk_source_weighted),
            "covariate_tv": covariate_tv, "conditional_label_tv": conditional_tv,
            "fixed_loss_bound": covariate_tv + conditional_tv,
            "weight_normalizer": normalizer,
            "numerical_source_mass": integrate(source), "numerical_target_mass": integrate(target)}


def domain_features(x: np.ndarray, quadratic: bool) -> np.ndarray:
    x = np.asarray(x)
    return np.column_stack([x, x * x]) if quadratic else x[:, None]


def learned_ratio(model: LogisticRegression, x: np.ndarray, quadratic: bool) -> np.ndarray:
    # Logistic log-odds avoids rounding a probability to exactly zero or one.
    return np.exp(np.clip(model.decision_function(domain_features(x, quadratic)), -40, 40))


def execute(config_path: Path, output: Path) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if output.exists():
        raise ValueError("Use a new output directory; frozen simulated results cannot be overwritten")
    output.mkdir(parents=True)
    atomic_json(output / "design_receipt.json", {
        "status": "DESIGN_FIXED_BEFORE_SIMULATION", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha(config_path), "script_sha256": sha(Path(__file__)),
        "previous_real_outcomes_known": True, "real_target_label_vault_accessed": False,
        "replication_unit": "independent simulated calibration/domain/test draws",
    })
    grid = np.linspace(*cfg["integration_domain"], cfg["integration_grid_n"])
    rows = []
    for regime_index, regime in enumerate(cfg["regimes"]):
        print("Simulating " + regime["name"], flush=True)
        for rep in range(cfg["replications"]):
            rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], regime_index, rep]))
            source_x = rng.normal(size=cfg["source_calibration_n"])
            source_y = (rng.random(len(source_x)) < expit(2 * source_x - 2)).astype(int)
            target_x = sample_target(rng, cfg["target_test_n"], regime)
            target_y = (rng.random(len(target_x)) < expit(2 * target_x - 2 + regime["target_conditional_delta"])).astype(int)
            domain_source = rng.normal(size=cfg["independent_domain_training_per_domain"])
            domain_target = sample_target(rng, cfg["independent_domain_training_per_domain"], regime)
            domain_x = np.concatenate([domain_source, domain_target])
            domain_y = np.concatenate([np.zeros(len(domain_source), dtype=int), np.ones(len(domain_target), dtype=int)])
            methods = [("unweighted", None, np.ones(len(source_x)), np.ones(len(target_x)), np.ones(len(grid))),
                       ("exact_ratio", None, exact_ratio(source_x, regime), exact_ratio(target_x, regime), exact_ratio(grid, regime))]
            for clip in cfg["clips"]:
                methods.append(("exact_ratio_clipped", clip,
                                np.clip(exact_ratio(source_x, regime), 1 / clip, clip),
                                np.clip(exact_ratio(target_x, regime), 1 / clip, clip),
                                np.clip(exact_ratio(grid, regime), 1 / clip, clip)))
            for density in cfg["density_models"]:
                quadratic = density == "quadratic_logistic"
                model = LogisticRegression(C=cfg["density_logistic_C"], max_iter=cfg["density_max_iterations"], solver="lbfgs")
                model.fit(domain_features(domain_x, quadratic), domain_y)
                for clip in cfg["clips"]:
                    methods.append(("learned_quadratic_clipped" if quadratic else "learned_linear_clipped", clip,
                                    np.clip(learned_ratio(model, source_x, quadratic), 1 / clip, clip),
                                    np.clip(learned_ratio(model, target_x, quadratic), 1 / clip, clip),
                                    np.clip(learned_ratio(model, grid, quadratic), 1 / clip, clip)))
            p_source, p_target = expit(1.3 * source_x - 1.5), expit(1.3 * target_x - 1.5)
            for method, clip, w_source, w_target, w_grid in methods:
                sets, thresholds = class_sets(p_source, source_y, w_source, p_target, w_target, cfg["alpha_s"], cfg["alpha_v"])
                metrics = action_metrics(sets, thresholds, target_y)
                decomposition = loss_decomposition(grid, regime, w_grid)
                if decomposition["fixed_risk_absolute_gap"] > decomposition["fixed_loss_bound"] + 1e-7:
                    raise ValueError("Fixed-loss numerical decomposition bound failed")
                if metrics["automatic_miss"] > metrics["miscoverage_vulnerable"] + 1e-12:
                    raise ValueError("Automatic miss exceeded set miscoverage")
                rows.append({"regime": regime["name"], "replicate": rep, "method": method,
                             "clip": clip, "target_positive_count": int(target_y.sum()),
                             "source_positive_count": int(source_y.sum()),
                             "target_upper_clipping_fraction": float(np.mean(w_target == clip)) if clip else 0.0,
                             **metrics, **decomposition})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "replicate_metrics.csv", index=False, float_format="%.12g")
    primary = frame[frame["clip"].isna() | (frame["clip"] == cfg["primary_clip"])].copy()
    measures = ["miscoverage_safe", "miscoverage_vulnerable", "automatic_miss", "automatic_false_alert",
                "automation_fraction", "review_fraction", "selective_error", "covariate_tv", "conditional_label_tv",
                "fixed_risk_absolute_gap", "fixed_loss_bound", "target_upper_clipping_fraction"]
    summaries = []
    for (regime, method), group in primary.groupby(["regime", "method"], sort=False):
        result = {"regime": regime, "method": method, "n_independent_simulations": len(group)}
        for measure in measures:
            values = group[measure].dropna().to_numpy()
            result[measure + "_mean"] = float(np.mean(values))
            result[measure + "_monte_carlo_se"] = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        summaries.append(result)
    pd.DataFrame(summaries).to_csv(output / "primary_summary.csv", index=False, float_format="%.12g")
    manifest = {
        "status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha(config_path), "script_sha256": sha(Path(__file__)),
        "replications_per_regime": cfg["replications"], "regimes": len(cfg["regimes"]),
        "complete_cells": len(frame), "primary_cells": len(primary), "bound_checks": len(frame),
        "new_real_dataset_outcomes_used": False, "domain_fit_uses_class_labels": False,
        "complete_clip_family_retained": cfg["clips"], "independence": "Fresh independent domain-training and calibration/test samples in each simulation",
        "claim_boundary": cfg["claim_boundary"],
        "files_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"status": "PASS", "complete_cells": len(frame), "bound_checks": len(frame)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ratio_diagnostics_v4.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/kbs-validation-v4/ratio_diagnostics")
    args = parser.parse_args()
    execute(args.config.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
