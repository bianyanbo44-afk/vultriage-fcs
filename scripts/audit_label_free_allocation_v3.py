"""Describe sealed allocation and selector identity without target outcomes.

This interpretation audit fixes no new policy or subset.  It distinguishes
structural zero-miss behavior (no safe release) from empirical selection and
records the test-point infinity-mass condition using source labels only.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_neural_validation_v3 import verify_seal
from analyze_neurocomputing_controls import action_codes


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, default=ROOT / "outputs/neural-validation-v3/decisions")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/neural-validation-v3/allocation-structure")
    args = parser.parse_args()
    rows, dependencies = [], {}
    for dataset in ("diversevul", "megavul"):
        path = args.decisions / dataset / "decision_seal.json"
        seal = verify_seal(path)
        if not seal["all_planned_fits_present"]:
            raise ValueError("Interpretation audit requires the complete seven-fit seals")
        dependencies[dataset] = sha(path)
        source = [json.loads(line) for line in Path(seal["source_rows_path"]).open(encoding="utf-8")]
        for entry in seal["decisions"]:
            meta = json.loads((path.parent / entry["metadata"]).read_text(encoding="utf-8"))
            specs = {s["method"]: s for s in meta["decision_specs"]
                     if s["alpha_v"] == .1 and s["alpha_s"] == .2 and s["clip"] == 20}
            with np.load(path.parent / entry["archive"], allow_pickle=False) as data:
                w = action_codes(data[specs["estimated_weight_no_gate"]["key"]])
                a = action_codes(data[specs["class_matched_weighted"]["key"]])
                m = action_codes(data[specs["msp_matched_weighted"]["key"]])
                n = len(w)
                item = dict(dataset=dataset, detector=entry["detector"], seed=entry["seed"], project=entry["project"],
                            n=n, domain_available=meta["domain_status"] == "ok",
                            weighted_auto_safe=int((w == 0).sum()), weighted_auto_vulnerable=int((w == 1).sum()),
                            weighted_review=int((w == 2).sum()), msp_auto_safe=int((m == 0).sum()),
                            msp_auto_vulnerable=int((m == 1).sum()), weighted_safe_fraction=float((w == 0).mean()),
                            weighted_vulnerable_fraction=float((w == 1).mean()), weighted_review_fraction=float((w == 2).mean()),
                            allocation_disagreement_fraction=float((w != a).mean()),
                            exact_allocation_match=bool(np.array_equal(w, a)),
                            no_automatic_safe_release=bool(not (w == 0).any()), full_review=bool((w == 2).all()),
                            vulnerable_infinity_fraction=np.nan, safe_infinity_fraction=np.nan,
                            target_upper_clip_fraction=np.nan, target_lower_clip_fraction=np.nan,
                            source_positive_weight_sum=np.nan, source_safe_weight_sum=np.nan)
                if item["domain_available"]:
                    labels = np.asarray([source[int(i)]["label"] for i in data["calibration_positions"]])
                    cw = np.clip(data["raw_calibration_weights"], 1/20, 20)
                    tw = np.clip(data["raw_target_weights"], 1/20, 20)
                    for label, name, alpha in ((1, "vulnerable", .1), (0, "safe", .2)):
                        total = float(cw[labels == label].sum())
                        infinity = (1-alpha) * (total + tw) > total
                        item[name + "_infinity_fraction"] = float(infinity.mean())
                        item["source_positive_weight_sum" if label else "source_safe_weight_sum"] = total
                    raw_target = data["raw_target_weights"]
                    item["target_upper_clip_fraction"] = float((raw_target >= 20).mean())
                    item["target_lower_clip_fraction"] = float((raw_target <= 1/20).mean())
                rows.append(item)
    frame = pd.DataFrame(rows)
    numeric = [c for c in frame.select_dtypes(include=[np.number]).columns if c != "seed"]
    keys = ["dataset", "detector", "project"]
    averaged = frame.groupby(keys, as_index=False)[numeric].agg(lambda s: s.mean(skipna=False))
    flags = ["domain_available", "exact_allocation_match", "no_automatic_safe_release", "full_review"]
    for flag in flags:
        combined = frame.groupby(keys, as_index=False)[flag].all()
        averaged = averaged.merge(combined, on=keys, validate="one_to_one")
    summary = []
    for (dataset, detector), cell in averaged.groupby(["dataset", "detector"], sort=True):
        supported = cell[cell.domain_available]
        summary.append(dict(dataset=dataset, detector=detector, projects=len(cell), supported_projects=len(supported),
                            no_safe_release_projects=int(cell.no_automatic_safe_release.sum()),
                            supported_no_safe_release_projects=int(supported.no_automatic_safe_release.sum()),
                            full_review_projects=int(cell.full_review.sum()),
                            supported_full_review_projects=int(supported.full_review.sum()),
                            exactly_matched_policy_projects=int(cell.exact_allocation_match.sum()),
                            supported_exactly_matched_policy_projects=int(supported.exact_allocation_match.sum()),
                            median_safe_fraction=float(cell.weighted_safe_fraction.median()),
                            median_vulnerable_fraction=float(cell.weighted_vulnerable_fraction.median()),
                            median_review_fraction=float(cell.weighted_review_fraction.median()),
                            median_allocation_disagreement=float(cell.allocation_disagreement_fraction.median()),
                            median_vulnerable_infinity_fraction=float(supported.vulnerable_infinity_fraction.median()),
                            median_target_upper_clip_fraction=float(supported.target_upper_clip_fraction.median())))
    args.output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output / "project_seed_allocation_structure.csv", index=False)
    averaged.to_csv(args.output / "project_allocation_structure.csv", index=False)
    pd.DataFrame(summary).to_csv(args.output / "allocation_structure_summary.csv", index=False)
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), target_labels_accessed=False,
                    selector_or_hyperparameter_modified=False, source_labels_used="calibration class labels only",
                    scope="Descriptive label-free interpretation at the already-frozen primary budgets/clip; no new subset or policy.",
                    code_sha256=sha(Path(__file__)), decision_seals=dependencies,
                    reports_sha256={p.name: sha(p) for p in args.output.iterdir()},
                    interpretation="No automatic-safe release forces zero automatic misses for any target labels. Exact action equality forces equal automatic errors, automation, review and cost outcomes, but does not imply identical empty/doubleton sets or equal set miscoverage. Infinity fraction checks the stated positive test-point-mass condition, without target labels.")
    (args.output / "allocation_structure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
