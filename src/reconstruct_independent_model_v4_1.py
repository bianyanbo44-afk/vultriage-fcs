"""Independent post-seal reconstruction from decisions, without importing metric helpers.

Only SHA/identity checks and standard NumPy/Pandas operations are used.  All
metrics, seed averages, paired medians and project-bootstrap intervals below are
implemented independently of vultriage.neural_audit and the evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest().upper()


def json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def independently_verify_joint(root: Path, cfg: dict) -> dict[str, dict]:
    """Must finish for both datasets before opening either target-label archive."""
    joint_path = root / "joint_dataset_decision_seal.json"
    if digest(joint_path) != joint_path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Joint seal bytes changed")
    joint = json_object(joint_path)
    if (joint["schema_version"] != "trace-independent-model-v4.1-joint-dataset-seal-1"
            or joint["target_label_vaults_opened_for_this_seal"] is not False
            or set(joint["datasets"]) != {"diversevul", "megavul"}):
        raise ValueError("Both immutable full dataset seals are required")
    verified_files, manifests = {}, {}
    for dataset, record in joint["datasets"].items():
        path = ROOT / record["path"]
        if digest(path) != record["sha256"] or digest(path) != path.with_suffix(".sha256").read_text().strip():
            raise ValueError("Dataset seal bytes changed")
        seal = json_object(path)
        if (seal["schema_version"] != "trace-independent-model-v4.1-decision-seal-1"
                or seal["target_labels_accessed"] is not False or seal["all_planned_attempts_accounted"] is not True
                or seal["projects"] != cfg["target_rosters"][dataset]["projects"]
                or seal["target_rows"] != cfg["target_rosters"][dataset]["rows"]
                or seal["actual_seeds"] != {cfg["detector"]: cfg["fine_tuned_codet5"]["seeds"]}):
            raise ValueError("The complete roster/genuine-seed accounting changed")
        for dependency in seal["dependencies"]:
            target = Path(dependency["path"])
            if "target_label_vault" in target.name.casefold():
                raise ValueError("No target vault may be an input to pre-label seal verification")
            if str(target) not in verified_files:
                verified_files[str(target)] = digest(target)
            if verified_files[str(target)] != dependency["sha256"]:
                raise ValueError(f"Sealed dependency changed: {dependency['role']}")
        seen = set()
        for entry in seal["decisions"]:
            identity = entry["detector"], entry["seed"], entry["project"]
            if identity in seen:
                raise ValueError("Copied/duplicate project-seed decisions")
            seen.add(identity)
            for name in ["archive", "metadata"]:
                target = path.parent / entry[name]
                if digest(target) != entry[f"{name}_sha256"]:
                    raise ValueError("Project decision bytes changed")
        expected = {(cfg["detector"], s, p) for s in cfg["fine_tuned_codet5"]["seeds"] for p in seal["projects"]}
        if seen != expected:
            raise ValueError("The complete project-by-actual-seed grid was not sealed")
        for name, expected_hash in seal["reports"].items():
            if digest(path.parent / name) != expected_hash:
                raise ValueError("A label-free availability/failure report changed")
        manifests[dataset] = seal
    return manifests


def independent_metrics(labels: np.ndarray, sets: np.ndarray, av: float, ass: float, costs: dict, status: str) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    sets = np.asarray(sets, dtype=bool)
    n = len(labels)
    if not n or sets.shape != (n, 2) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid independently reconstructed decisions")
    sizes = sets.sum(axis=1)
    safe_auto = (sizes == 1) & sets[:, 0]
    vulnerable_auto = (sizes == 1) & sets[:, 1]
    review = ~(safe_auto | vulnerable_auto)
    vulnerable, safe = labels == 1, labels == 0
    nv, ns = int(vulnerable.sum()), int(safe.sum())
    fn = int((vulnerable & safe_auto).sum())
    fp = int((safe & vulnerable_auto).sum())
    n_auto = int(safe_auto.sum() + vulnerable_auto.sum())
    cov_v = int((vulnerable & ~review).sum())
    cov_s = int((safe & ~review).sum())
    ratio = lambda numerator, denominator: numerator / denominator if denominator else np.nan
    missing_v = int((vulnerable & ~sets[:, 1]).sum())
    missing_s = int((safe & ~sets[:, 0]).sum())
    result = dict(n=n, n_vulnerable=nv, n_safe=ns, auto_safe=int(safe_auto.sum()),
                  auto_vulnerable=int(vulnerable_auto.sum()), auto_fn=fn, auto_fp=fp,
                  review_count=int(review.sum()), review_vulnerable=int((review & vulnerable).sum()),
                  review_safe=int((review & safe).sum()), empty_count=int((sizes == 0).sum()), singleton_count=n_auto,
                  doubleton_count=int((sizes == 2).sum()), coverage=n_auto / n, review_load=float(review.mean()),
                  auto_fn_rate=ratio(fn, nv), auto_fp_rate=ratio(fp, ns), selective_error=ratio(fn + fp, n_auto),
                  selective_vulnerable_error=ratio(fn, cov_v), selective_safe_error=ratio(fp, cov_s),
                  safe_release_contamination=ratio(fn, int(safe_auto.sum())),
                  vulnerable_alert_precision=ratio(int(vulnerable_auto.sum()) - fp, int(vulnerable_auto.sum())),
                  empty_rate=float((sizes == 0).mean()), doubleton_rate=float((sizes == 2).mean()),
                  vulnerable_miscoverage_count=missing_v, safe_miscoverage_count=missing_s,
                  vulnerable_miscoverage=ratio(missing_v, nv), safe_miscoverage=ratio(missing_s, ns))
    result["maximum_relative_violation"] = max(max(0, result["vulnerable_miscoverage"] - av) / av,
                                                max(0, result["safe_miscoverage"] - ass) / ass) if nv and ns else np.nan
    for cost in costs["miss"]:
        result[f"cost_{int(cost)}"] = (cost * fn + costs["false_alert"] * fp + costs["review"] * int(review.sum())) / n
    if status != "ok":
        for key in ["vulnerable_miscoverage", "safe_miscoverage", "maximum_relative_violation", "vulnerable_miscoverage_count", "safe_miscoverage_count"]:
            result[key] = np.nan
    return result


def independent_seed_averages(frame: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    keys = ["dataset", "detector", "project", "alpha_v", "alpha_s", "clip", "method"]
    numeric = [c for c in frame.select_dtypes(include=[np.number]).columns if c not in keys + ["seed"]]
    rows = []
    for identity, group in frame.groupby(keys, sort=True):
        if len(group) != len(seeds) or sorted(group["seed"].tolist()) != sorted(seeds):
            raise ValueError("Seed duplication or omission in reconstruction")
        row = dict(zip(keys, identity))
        for field in numeric:
            values = group[field].to_numpy(dtype=np.float64)
            row[field] = float(np.mean(values))  # any undefined fit remains undefined
        row.update(actual_seed_count=len(seeds), status="ok" if (group["status"] == "ok").all() else "policy_failure_present")
        rows.append(row)
    return pd.DataFrame(rows)


def independent_discrimination(labels: np.ndarray, probability: np.ndarray, eta: np.ndarray) -> dict:
    labels, probability, eta = np.asarray(labels, dtype=np.int64), np.asarray(probability, dtype=np.float64), np.asarray(eta, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives and negatives:
        order = np.argsort(-probability, kind="stable")
        ordered_p, ordered_y = probability[order], labels[order]
        ends = np.r_[np.flatnonzero(ordered_p[1:] != ordered_p[:-1]), len(labels) - 1]
        tp = np.cumsum(ordered_y)[ends]
        recalls = tp / positives
        ap = float(np.sum(np.diff(np.r_[0.0, recalls]) * tp / (ends + 1)))
        ranks = pd.Series(probability).rank(method="average").to_numpy()
        auc = float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    else:
        ap = auc = np.nan
    def ece(values):
        order = np.argsort(values, kind="stable")
        return float(sum(len(part) / len(labels) * abs(float(labels[part].mean() - values[part].mean()))
                         for part in np.array_split(order, min(15, len(order))) if len(part)))
    return dict(pr_auc=ap, roc_auc=auc, brier=float(np.mean((probability - labels) ** 2)),
                ece_equal_mass_15=ece(probability), source_iso_brier=float(np.mean((eta - labels) ** 2)),
                source_iso_ece_equal_mass_15=ece(eta))


def independent_commit_counts(rows: list[dict], labels: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    counts = sorted(Counter(r["commit_id"] for r, y in zip(rows, labels) if y == 1).values(), reverse=True)
    npositive = int(labels.sum())
    fractions = np.asarray(counts, dtype=float) / npositive if npositive else np.asarray([], dtype=float)
    hhi = float(np.square(fractions).sum()) if npositive else np.nan
    return dict(n=len(rows), n_vulnerable=npositive, n_safe=len(rows) - npositive,
                distinct_commits=len({r["commit_id"] for r in rows}), positive_commits=len(counts),
                largest_positive_commit_functions=counts[0] if counts else 0,
                largest_positive_commit_fraction=float(fractions[0]) if counts else np.nan,
                top3_positive_commit_fraction=float(fractions[:3].sum()) if counts else np.nan,
                positive_commit_hhi=hhi, effective_positive_commit_count=1 / hhi if counts else np.nan)


def compare_frames(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str], fields: list[str]) -> dict:
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError("Duplicate reconstructed/report rows")
    left = actual.set_index(keys).sort_index()
    right = expected.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise ValueError("Reconstructed report roster differs")
    differences = []
    for field in fields:
        a, b = left[field].to_numpy(dtype=float), right[field].to_numpy(dtype=float)
        if not np.array_equal(np.isnan(a), np.isnan(b)) or not np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True):
            raise ValueError(f"Independent metric mismatch: {field}")
        finite = np.isfinite(a) & np.isfinite(b)
        differences.append(float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0)
    return {"rows": len(left), "numeric_fields": len(fields), "maximum_absolute_difference": max(differences, default=0.0), "all_match": True}


def independently_check_pairs(averaged: pd.DataFrame, report: pd.DataFrame, seed: int, supported: list[str]) -> dict:
    checked, maximum_error = 0, 0.0
    resampling_cache, interval_cache = {}, {}
    for row in report.itertuples(index=False):
        subset = averaged[(averaged["alpha_v"] == row.alpha_v) & (averaged["alpha_s"] == row.alpha_s) & (averaged["clip"] == row.clip)]
        if row.cohort == "domain_supported_secondary":
            subset = subset[subset["project"].isin(supported)]
        left = subset[subset["method"] == row.baseline].sort_values("project")
        right = subset[subset["method"] == row.candidate].sort_values("project")
        if left["project"].tolist() != right["project"].tolist() or len(left) != row.projects:
            raise ValueError("A paired summary omitted or changed projects")
        delta = right[row.outcome].to_numpy() - left[row.outcome].to_numpy()
        finite = np.isfinite(delta)
        if int(finite.sum()) != row.defined_projects:
            raise ValueError("The defined-pair count differs")
        if not len(delta) or not finite.all():
            if not np.isnan(row.median_difference) or not np.isnan(row.ci_lower) or not np.isnan(row.ci_upper):
                raise ValueError("Undefined projects must not produce an interval through silent deletion")
            checked += 1
            continue
        median = float(np.median(delta))
        reductions, increases = int((delta < -1e-12).sum()), int((delta > 1e-12).sum())
        if (reductions, increases, len(delta) - reductions - increases) != (row.reductions, row.increases, row.ties):
            raise ValueError("Favorable/unfavorable/tie counts differ")
        if np.all(delta == delta[0]):
            lower = upper = float(delta[0])
        else:
            if len(delta) not in resampling_cache:
                resampling_cache[len(delta)] = np.random.default_rng(seed).integers(0, len(delta), size=(10000, len(delta)))
            key = delta.tobytes()
            if key not in interval_cache:
                interval_cache[key] = np.quantile(np.median(delta[resampling_cache[len(delta)]], axis=1), [0.025, 0.975])
            lower, upper = interval_cache[key]
        for actual, expected in [(median, row.median_difference), (lower, row.ci_lower), (upper, row.ci_upper)]:
            if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
                raise ValueError("Independent project-bootstrap summary differs")
            maximum_error = max(maximum_error, abs(actual - expected))
        checked += 1
    return {"paired_rows_checked": checked, "maximum_absolute_difference": maximum_error, "all_match": True}


def reconstruct(config: Path, output: Path) -> dict:
    cfg = json_object(config)
    root = ROOT / cfg["paths"]["output_root"]
    manifests = independently_verify_joint(root, cfg)
    if output.exists():
        raise FileExistsError("Preserve prior independent reconstruction evidence")
    output.mkdir(parents=True)
    reports = {}
    for dataset, manifest in manifests.items():
        rows = [json.loads(line) for line in Path(manifest["target_rows_path"]).open(encoding="utf-8")]
        prep = json_object(ROOT / cfg["paths"][f"{dataset}_inputs"] / "preparation_manifest.json")
        vault = ROOT / prep["original_target_label_vault_path"]
        if digest(vault) != manifest["prepared_target_label_vault_sha256"]:
            raise ValueError("Post-verification target-label vault bytes differ")
        with np.load(vault, allow_pickle=False) as archive:
            order = np.argsort(archive["positions"], kind="stable")
            positions, ids, labels = archive["positions"][order], archive["row_ids"][order], archive["labels"][order].astype(np.int64)
        if not np.array_equal(positions, np.arange(len(rows))) or not np.array_equal(ids.astype(str), np.asarray([r["row_id"] for r in rows], dtype=str)):
            raise ValueError("The target-vault identities are not exactly aligned")
        policy_rows, discrimination_rows, commit_rows = [], [], []
        for project in manifest["projects"]:
            positions = [i for i, r in enumerate(rows) if r["project_group"] == project]
            commit_rows.append(dict(dataset=dataset, project=project,
                                    **independent_commit_counts([rows[i] for i in positions], labels[positions])))
        for entry in manifest["decisions"]:
            decision_root = root / "decisions" / dataset
            metadata = json_object(decision_root / entry["metadata"])
            with np.load(decision_root / entry["archive"], allow_pickle=False) as archive:
                positions = archive["target_positions"]
                expected = np.asarray([i for i, r in enumerate(rows) if r["project_group"] == entry["project"]])
                if not np.array_equal(positions, expected):
                    raise ValueError("Decision positions do not preserve the full project")
                y = labels[positions]
                discrimination_rows.append(dict(dataset=dataset, detector=entry["detector"], seed=entry["seed"], project=entry["project"],
                                                **independent_discrimination(y, archive["target_probability"], archive["source_eta"])))
                for spec in metadata["decision_specs"]:
                    values = independent_metrics(y, archive[spec["key"]], spec["alpha_v"], spec["alpha_s"], cfg["cost"], spec["status"])
                    policy_rows.append(dict(dataset=dataset, detector=entry["detector"], seed=entry["seed"], project=entry["project"],
                                            **{k: v for k, v in spec.items() if k != "key"}, **values))
        frame = pd.DataFrame(policy_rows)
        averaged = independent_seed_averages(frame, cfg["fine_tuned_codet5"]["seeds"])
        evaluation = root / "evaluation" / dataset
        stored = pd.read_csv(evaluation / "project_seed_metrics.csv")
        stored_average = pd.read_csv(evaluation / "project_seed_averages.csv")
        keys = ["dataset", "detector", "project", "alpha_v", "alpha_s", "clip", "method"]
        numeric = [c for c in frame.select_dtypes(include=[np.number]).columns if c not in keys + ["seed"]]
        report = {"seed_metrics": compare_frames(frame, stored, keys + ["seed"], numeric),
                  "seed_averages": compare_frames(averaged, stored_average, keys, numeric + ["actual_seed_count"]),
                  "paired_project_summaries": independently_check_pairs(averaged, pd.read_csv(evaluation / "paired_project_summaries.csv"),
                                                                        cfg["statistics"]["bootstrap_seed"], manifest["domain_supported_projects"][cfg["detector"]])}
        discrimination = pd.DataFrame(discrimination_rows)
        counts = pd.DataFrame(commit_rows)
        detector_fields = [c for c in discrimination.select_dtypes(include=[np.number]) if c != "seed"]
        count_fields = [c for c in counts.select_dtypes(include=[np.number])]
        report["detector_seed_metrics"] = compare_frames(discrimination, pd.read_csv(evaluation / "detector_project_seed_metrics.csv"),
                                                         ["dataset", "detector", "seed", "project"], detector_fields)
        report["project_commit_denominators"] = compare_frames(counts, pd.read_csv(evaluation / "project_counts_positive_commits.csv"),
                                                                ["dataset", "project"], count_fields)
        if (averaged["status"].tolist() != stored_average.sort_values(keys)["status"].tolist()):
            # compare_frames already compares sorted identity; explicit status uses that same order.
            left = averaged.set_index(keys).sort_index()["status"]
            right = stored_average.set_index(keys).sort_index()["status"]
            if not left.equals(right):
                raise ValueError("Seed-average failure/undefined status differed")
        destination = output / dataset
        destination.mkdir()
        frame.to_csv(destination / "independently_reconstructed_seed_metrics.csv", index=False)
        averaged.to_csv(destination / "independently_reconstructed_seed_averages.csv", index=False)
        reports[dataset] = report
    result = {"schema_version": "trace-independent-model-v4.1-independent-reconstruction-1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "config_sha256": digest(config), "joint_dataset_decision_seal_sha256": digest(root / "joint_dataset_decision_seal.json"),
              "both_complete_seals_verified_before_vault_access": True, "imports_implementation_metric_helpers": False,
              "all_datasets_match": True, "datasets": reports, "script_sha256": digest(Path(__file__)),
              "outputs": {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()}}
    (output / "independent_reconstruction.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def reconstruct_aggregate(export_root: Path, output: Path) -> dict:
    """Portable aggregate-only verification; no original data, scores or vault access."""
    export_root = export_root.resolve()
    manifest_path = export_root / "aggregate_manifest.json"
    manifest = json_object(manifest_path)
    if manifest["schema_version"] != "trace-independent-model-v4.1-public-aggregates-1":
        raise ValueError("Expected the corrected public aggregate manifest")
    if set(manifest["datasets"]) != {"diversevul", "megavul"}:
        raise ValueError("Portable reconstruction requires both unchanged declared datasets")
    for dataset in manifest["datasets"]:
        required = {f"{dataset}/{name}" for name in ("project_seed_metrics.csv", "project_seed_averages.csv", "paired_project_summaries.csv")}
        if not required.issubset(manifest["members"]):
            raise ValueError("A required portable table is absent from the byte manifest")
    verified = []
    for filename, expected in manifest["members"].items():
        path = (export_root / filename).resolve()
        if not path.is_relative_to(export_root) or "target_label_vault" in path.name.casefold():
            raise ValueError("Portable aggregate members must stay inside the aggregate tree")
        if digest(path) != expected:
            raise ValueError(f"Portable aggregate bytes changed: {filename}")
        verified.append(filename)
    if output.exists():
        raise FileExistsError("Preserve prior portable reconstruction evidence")
    output.mkdir(parents=True)
    reports = {}
    for dataset, roster in manifest["datasets"].items():
        root = export_root / dataset
        frame = pd.read_csv(root / "project_seed_metrics.csv")
        if sorted(frame["project"].unique()) != roster["projects"] or sorted(frame["seed"].unique()) != manifest["actual_seeds"]:
            raise ValueError("The portable full project/seed roster differs")
        # Count algebra can be checked without individual labels or actions.
        if not np.allclose(frame["auto_safe"] + frame["auto_vulnerable"] + frame["review_count"], frame["n"]):
            raise ValueError("Aggregate action counts do not sum to all target functions")
        for cost in manifest["analysis"]["miss_costs"]:
            independently_costed = (cost * frame["auto_fn"] + manifest["analysis"]["false_alert_cost"] * frame["auto_fp"]
                                    + manifest["analysis"]["review_cost"] * frame["review_count"]) / frame["n"]
            if not np.allclose(independently_costed, frame[f"cost_{cost}"], rtol=1e-10, atol=1e-12, equal_nan=True):
                raise ValueError("Portable illustrative costs do not follow the fixed count algebra")
        averaged = independent_seed_averages(frame, manifest["actual_seeds"])
        stored = pd.read_csv(root / "project_seed_averages.csv")
        keys = ["dataset", "detector", "project", "alpha_v", "alpha_s", "clip", "method"]
        numeric = [c for c in averaged.select_dtypes(include=[np.number]) if c not in keys]
        reports[dataset] = {
            "seed_averages": compare_frames(averaged, stored, keys, numeric),
            "paired_project_summaries": independently_check_pairs(averaged, pd.read_csv(root / "paired_project_summaries.csv"),
                                                                  manifest["analysis"]["bootstrap_seed"], roster["domain_supported_projects"]),
            "complete_project_count": len(roster["projects"]), "actual_seed_count": len(manifest["actual_seeds"])}
        averaged.to_csv(output / f"{dataset}_reconstructed_project_seed_averages.csv", index=False)
    result = {"schema_version": "trace-independent-model-v4.1-portable-aggregate-reconstruction-1",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "aggregate_manifest_sha256": digest(manifest_path),
              "no_private_or_original_data_accessed": True, "all_datasets_match": True,
              "verified_members": verified, "datasets": reports, "script_sha256": digest(Path(__file__)),
              "scope": "Independent project aggregation/count algebra/pointwise bootstrap; individual prediction/action metrics were separately verified against sealed local decisions"}
    (output / "portable_aggregate_reconstruction.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/independent_model_v4_1.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--aggregate-root", type=Path, help="verify/reconstruct portable public aggregates only; no private input access")
    args = parser.parse_args()
    if args.aggregate_root is not None:
        output = args.output or args.aggregate_root.parent / "portable-aggregate-reconstruction"
        result = reconstruct_aggregate(args.aggregate_root, output)
        print(json.dumps({"all_datasets_match": result["all_datasets_match"], "no_private_data_accessed": True, "datasets": result["datasets"]}, indent=2), flush=True)
        return
    cfg = json_object(args.config)
    output = args.output or ROOT / cfg["paths"]["output_root"] / "independent-reconstruction"
    result = reconstruct(args.config, output)
    print(json.dumps({"all_datasets_match": result["all_datasets_match"], "datasets": result["datasets"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
