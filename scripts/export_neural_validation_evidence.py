"""Release aggregate v3 evidence and generate manuscript tables from verified bytes.

No raw dataset, prediction archive, code function, model weight or label vault
is copied.  The release retains every planned project/fit/budget/clip result,
including technical review fallback and undefined fitted-set endpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/neurocomputing_revision"
DETECTORS = ("finetuned_codebert", "frozen_codebert", "hashing_sgd")
LABELS = {"finetuned_codebert": "Fine-tuned CodeBERT", "frozen_codebert": "Frozen CodeBERT",
          "hashing_sgd": "Hashing-SGD"}
SHORT = {"finetuned_codebert": "Fine-tuned", "frozen_codebert": "Frozen", "hashing_sgd": "Hashing"}
DATASETS = ("diversevul", "megavul")
DS = {"diversevul": "DiverseVul", "megavul": "MegaVul"}
ACTUAL = {"finetuned_codebert": [13, 37, 73], "hashing_sgd": [13, 37, 73], "frozen_codebert": [13]}
WEIGHT_DEPENDENT = {"estimated_weight_no_gate", "msp_matched_weighted", "class_matched_weighted",
                    "source_iso_cost10_matched_weighted", "source_iso_cost50_matched_weighted"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def primary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[np.isclose(frame.alpha_v, .10) & np.isclose(frame.alpha_s, .20) & np.isclose(frame["clip"], 20)]


def fmt(value, places=4) -> str:
    if not np.isfinite(value):
        return "--"
    if abs(value) < .5 * 10 ** (-places):
        value = 0.0
    return f"{value:.{places}f}"


def escape(value: str) -> str:
    return str(value).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def estimate(row: pd.Series) -> str:
    if not np.isfinite(row.median_difference):
        return "--"
    return (r"\begin{tabular}[c]{@{}r@{}}$" + fmt(row.median_difference) + r"$\\"
            + r"\scriptsize $[" + fmt(row.ci_lower) + ", " + fmt(row.ci_upper) + r"]$\end{tabular}")


def paired(cell: pd.DataFrame, detector: str, baseline: str, outcome: str, cohort: str) -> pd.Series:
    found = cell[(cell.detector == detector) & (cell.baseline == baseline)
                 & (cell.outcome == outcome) & (cell.cohort == cohort)]
    if len(found) != 1:
        raise ValueError("Expected one complete paired cell")
    return found.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=ROOT / "outputs/neural-validation-v3/evaluation")
    parser.add_argument("--decisions-root", type=Path, default=ROOT / "outputs/neural-validation-v3/decisions")
    parser.add_argument("--output", type=Path, default=PAPER / "analysis/neural_validation_v3")
    parser.add_argument("--tables", type=Path, default=PAPER / "neural_validation_supplement_tables.tex")
    parser.add_argument("--main-tables", type=Path, default=PAPER / "neural_validation_main_tables.tex")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    structural = ROOT / "outputs/neural-validation-v3/allocation-structure"
    structural_manifest = read(structural / "allocation_structure_manifest.json")
    if structural_manifest["target_labels_accessed"] is not False or structural_manifest["selector_or_hyperparameter_modified"] is not False:
        raise ValueError("Allocation-structure audit must be label-free and must preserve the frozen decisions")
    for name, expected in structural_manifest["reports_sha256"].items():
        if sha(structural / name) != expected or not name.endswith(".csv"):
            raise ValueError("Allocation-structure aggregate changed")
        shutil.copyfile(structural / name, args.output / name)
    shutil.copyfile(structural / "allocation_structure_manifest.json", args.output / "allocation_structure_manifest.json")
    checks, receipts, comparisons, detector_summaries, costs, all_counts, all_actions, all_policy_medians = [], {}, [], [], [], [], [], []

    def require(value: bool, label: str):
        if not bool(value):
            raise ValueError(label)
        checks.append(label)

    for dataset in DATASETS:
        evaluation = args.evaluation_root / dataset
        manifest_path = evaluation / "evaluation_manifest.json"
        manifest = read(manifest_path)
        seal_path = args.decisions_root / dataset / "decision_seal.json"
        seal = read(seal_path)
        require(manifest["decision_seal_sha256"] == sha(seal_path), dataset + " verified decision-seal lineage")
        require(manifest["verified_before_target_label_access"] is True
                and manifest["all_planned_fits_present"] is True and manifest["actual_seeds"] == ACTUAL,
                dataset + " seven actual fitted models before label access")
        release = args.output / dataset
        release.mkdir(parents=True, exist_ok=True)
        for name, expected in manifest["report_hashes"].items():
            path = evaluation / name
            require(path.suffix == ".csv" and sha(path) == expected, dataset + " bound aggregate " + name)
            shutil.copyfile(path, release / name)
        for name in ("domain_fit_availability.csv", "label_free_decision_counts.csv", "failures.json"):
            require(sha(seal_path.parent / name) == seal["reports"][name], dataset + " sealed label-free report " + name)
            shutil.copyfile(seal_path.parent / name, release / name)
        inputs = ROOT / ("outputs/neural-validation-v3/inputs" if dataset == "diversevul"
                         else "outputs/neural-validation-v3/inputs/megavul")
        prep = read(inputs / "preparation_manifest.json")
        for name in ("candidate_projects.csv",):
            require(sha(inputs / name) == prep["output_hashes"][name], dataset + " bound complete candidate ledger")
            shutil.copyfile(inputs / name, release / name)
        receipts[dataset] = {k: v for k, v in manifest.items() if k != "report_hashes"}
        receipts[dataset].update(preparation_manifest_sha256=sha(inputs / "preparation_manifest.json"),
                                 input_hashes={Path(k).name: v for k, v in prep["input_hashes"].items()},
                                 preparation_script_sha256=prep["preparation_script_sha256"],
                                 retained_target_rows_sha256=prep["output_hashes"]["target_rows.jsonl"],
                                 exclusion_counts=prep.get("exclusion_counts", {}),
                                 source_counts=prep["source_counts"],
                                 candidate_count=len(prep["candidate_projects"]),
                                 selected_projects=prep["selected_projects"],
                                 public_release_scope="Aggregate CSVs and receipts only. Raw dependencies remain independently required.")
        write(release / "dataset_receipt.json", receipts[dataset])
        diagnostics = []
        for entry in seal["decisions"]:
            path = seal_path.parent / entry["metadata"]
            require(sha(path) == entry["metadata_sha256"], dataset + " bound project domain diagnostics " + entry["detector"] + f"/{entry['seed']}/{entry['project']}")
            d = read(path)
            diagnostics.append({k: d[k] for k in ("dataset", "detector", "seed", "project", "target_rows",
                                                  "calibration_class_counts", "domain_status", "domain_error",
                                                  "domain_diagnostics", "clip_diagnostics")})
        write(release / "domain_cell_diagnostics.json", diagnostics)
        raw = pd.read_csv(evaluation / "project_seed_metrics.csv")
        avg = pd.read_csv(evaluation / "project_seed_averages.csv")
        count = pd.read_csv(evaluation / "project_counts_positive_commits.csv")
        pairs = pd.read_csv(evaluation / "paired_project_summaries.csv")
        detect = pd.read_csv(evaluation / "detector_project_seed_averages.csv")
        medians = pd.read_csv(evaluation / "project_medians.csv")
        require(len(count) == manifest["projects"] and int(count.n.sum()) == manifest["target_rows"]
                and int(count.n_vulnerable.sum()) == manifest["target_vulnerable"], dataset + " cohort and class denominators")
        require(len(raw) == len(count) * 7 * 9 * 3 * 10 and len(avg) == len(count) * 3 * 9 * 3 * 10,
                dataset + " complete fixed project/actual-fit/budget/clip/policy grid")
        require(not raw.duplicated(["detector", "seed", "project", "alpha_v", "alpha_s", "clip", "method"]).any(),
                dataset + " no duplicated metric identities")
        for field, expression in (("n", raw.auto_safe + raw.auto_vulnerable + raw.review_count),
                                  ("singleton_count", raw.auto_safe + raw.auto_vulnerable),
                                  ("n_vulnerable", raw.auto_fn + raw.review_vulnerable + raw.auto_vulnerable - raw.auto_fp)):
            require(np.array_equal(raw[field].to_numpy(), expression.to_numpy()), dataset + " exact count identity " + field)
        for field, expression in (("auto_fn_rate", raw.auto_fn / raw.n_vulnerable),
                                  ("auto_fp_rate", raw.auto_fp / raw.n_safe),
                                  ("coverage", raw.singleton_count / raw.n),
                                  ("selective_error", (raw.auto_fn + raw.auto_fp) / raw.singleton_count.replace(0, np.nan)),
                                  ("cost_10", (10 * raw.auto_fn + raw.auto_fp + .1 * raw.review_count) / raw.n),
                                  ("cost_50", (50 * raw.auto_fn + raw.auto_fp + .1 * raw.review_count) / raw.n)):
            require(np.allclose(raw[field], expression, atol=1e-12, rtol=1e-11, equal_nan=True),
                    dataset + " independent endpoint arithmetic " + field)
        require(((raw.auto_fn >= 0) & (raw.auto_fn <= raw.n_vulnerable)
                 & (raw.auto_fp >= 0) & (raw.auto_fp <= raw.n_safe)).all(), dataset + " class-error bounds")
        unavailable = raw.status != "ok"
        require(raw.loc[unavailable, "method"].isin(WEIGHT_DEPENDENT).all()
                and raw.loc[unavailable, ["coverage", "auto_fn", "auto_fp"]].eq(0).all().all()
                and raw.loc[unavailable, ["vulnerable_miscoverage", "safe_miscoverage", "maximum_relative_violation"]].isna().all().all(),
                dataset + " unavailable weighted fits retain review and undefined set metrics")
        p = primary(pairs)
        comparisons.append(p)
        all_counts.append(count)
        all_actions.append(primary(avg))
        all_policy_medians.append(primary(medians))
        for detector in DETECTORS:
            d = detect[detect.detector == detector]
            require(len(d) == len(count), dataset + " complete detector-project aggregate " + detector)
            summary = dict(dataset=dataset, detector=detector, projects=len(d), actual_fit_count=len(ACTUAL[detector]),
                           functions=int(count.n.sum()), positives=int(count.n_vulnerable.sum()),
                           domain_supported_projects=len(seal["domain_supported_projects"][detector]))
            for metric in ("pr_auc", "roc_auc", "brier", "ece_equal_mass_15", "source_iso_brier", "source_iso_ece_equal_mass_15"):
                require(d[metric].between(0, 1).all(), dataset + " detector-metric bounds " + detector + "/" + metric)
                summary[metric + "_median"] = float(d[metric].median())
                summary[metric + "_q1"] = float(d[metric].quantile(.25))
                summary[metric + "_q3"] = float(d[metric].quantile(.75))
            detector_summaries.append(summary)
            pm = primary(medians)
            weighted = pm[(pm.detector == detector) & (pm.method == "estimated_weight_no_gate")].iloc[0]
            for ratio in (10, 50):
                for workload in ("matched_weighted", "free_volume"):
                    baseline = f"source_iso_cost{ratio}_{workload}"
                    delta = paired(p, detector, baseline, f"cost_{ratio}", "all_projects_deployment")
                    b = pm[(pm.detector == detector) & (pm.method == baseline)].iloc[0]
                    costs.append(dict(dataset=dataset, detector=detector, miss_cost=ratio, workload=workload,
                                      baseline=baseline, weighted_cost=weighted[f"cost_{ratio}"],
                                      baseline_cost=b[f"cost_{ratio}"], weighted_coverage=weighted.coverage,
                                      baseline_coverage=b.coverage,
                                      **{k: delta[k] for k in ("median_difference", "ci_lower", "ci_upper", "reductions", "increases", "ties", "status")}))
    joint = pd.concat(comparisons, ignore_index=True)
    counts = pd.concat(all_counts, ignore_index=True)
    actions = pd.concat(all_actions, ignore_index=True)
    policy_medians = pd.concat(all_policy_medians, ignore_index=True)
    ds = pd.DataFrame(detector_summaries)
    cost = pd.DataFrame(costs)
    for name, frame in (("primary_budget_paired_summaries.csv", joint), ("detector_summary.csv", ds),
                        ("cost_summary.csv", cost), ("target_project_counts.csv", counts),
                        ("primary_budget_project_actions.csv", actions)):
        frame.to_csv(args.output / name, index=False)
    fit_rows = []
    training_out = args.output / "training"
    training_out.mkdir(exist_ok=True)
    for detector in DETECTORS:
        for seed in ACTUAL[detector]:
            folder = (ROOT / f"outputs/neural-validation-v3/training/seed-{seed}" if detector == "finetuned_codebert"
                      else ROOT / (f"outputs/neural-validation-v3/reference-training/hashing/seed-{seed}"
                                   if detector == "hashing_sgd" else "outputs/neural-validation-v3/reference-training/frozen"))
            receipt = folder / "training_manifest.json"
            r = read(receipt)
            require(r["seed"] == seed and r["target_labels_accessed"] is False, "Source-only training receipt " + detector + f"/{seed}")
            shutil.copyfile(receipt, training_out / f"{detector}_seed-{seed}.json")
            updates = list(r.get("encoder_query_relative_updates", {}).values())
            fit_rows.append(dict(detector=detector, seed=seed, source_validation_ap=r["source_validation_ap"],
                                 selected_epoch=r.get("selected_epoch", np.nan), selected_C=r.get("selected_C", np.nan),
                                 selected_alpha=r.get("selected_alpha", np.nan),
                                 trainable_parameters=r.get("trainable_parameters", np.nan),
                                 changed_encoder_tensors=r.get("changed_encoder_tensor_count", 0),
                                 query_update_min=min(updates) if updates else np.nan,
                                 query_update_max=max(updates) if updates else np.nan,
                                 wall_seconds=r["wall_seconds"], peak_gpu_allocated_bytes=r.get("peak_gpu_allocated_bytes", np.nan),
                                 checkpoint_sha256=r["checkpoint_sha256"], training_manifest_sha256=sha(receipt)))
    fits = pd.DataFrame(fit_rows)
    fits.to_csv(args.output / "source_fit_summary.csv", index=False)
    main_tex, si_tex = [], []
    main_tex.extend([
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Additional detector evidence. AP, AUROC and Brier scores are project medians after averaging actual fits. Every project enters; the source-isotonic map is fitted without target labels. Raw and source-isotonic Brier scores describe calibration separately from ranking.}",
        r"\label{tab:newdetectors}",
        r"\begin{tabular}{@{}llrrrrr@{}}\toprule Collection & Detector & Fits & AP & AUROC & Brier & Iso. Brier\\\midrule"])
    for _, row in ds.iterrows():
        main_tex.append(" & ".join([DS[row.dataset], SHORT[row.detector], str(int(row.actual_fit_count)),
                                    fmt(row.pr_auc_median), fmt(row.roc_auc_median), fmt(row.brier_median), fmt(row.source_iso_brier_median)]) + r"\\")
    main_tex.extend([r"\bottomrule\end{tabular}\end{table}", ""])
    main_tex.extend([
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Additional primary-budget comparisons. Entries are weighted-minus-control paired project medians with pointwise 95\% intervals from 10,000 project-bootstrap draws. The all-project analysis includes sealed review fallback. Supported counts are given separately. Matching both label counts tests selection within the weighted allocation. Tied medians can produce zero-width intervals; these do not establish equivalence.}",
        r"\label{tab:newprimary}",
        r"\begin{tabular}{@{}llcrrr@{}}\toprule Collection & Detector & Supported/all & $\Delta F_v$, MSP & $\Delta F_s$, MSP & $\Delta F_v$, tails\\\midrule"])
    for dataset in DATASETS:
        cell = joint[joint.dataset == dataset]
        for detector in DETECTORS:
            a = paired(cell, detector, "msp_matched_weighted", "auto_fn_rate", "all_projects_deployment")
            b = paired(cell, detector, "msp_matched_weighted", "auto_fp_rate", "all_projects_deployment")
            c = paired(cell, detector, "class_matched_weighted", "auto_fn_rate", "all_projects_deployment")
            main_tex.append(" & ".join([DS[dataset], SHORT[detector], f"{int(c.domain_supported_projects)}/{int(c.all_projects)}",
                                        estimate(a), estimate(b), estimate(c)]) + r"\\[2pt]")
    main_tex.extend([r"\bottomrule\end{tabular}\end{table}", ""])
    main_tex.extend([
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Absolute action levels for the additional weighted policies at the primary budgets and clip 20. Entries are separate project medians after averaging actual fits. Automation, $F_v$ and $F_s$ include every project and sealed review fallback; selective error conditions on automated decisions and uses only the stated defined-project denominator. These marginal medians do not form a pooled confusion matrix.}",
        r"\label{tab:newactions}",
        r"\begin{tabular}{@{}llrrrrc@{}}\toprule Collection & Detector & Automation & $F_v$ & $F_s$ & Sel. error & Defined/all\\\midrule"])
    for dataset in DATASETS:
        for detector in DETECTORS:
            row = policy_medians[(policy_medians.dataset == dataset) & (policy_medians.detector == detector)
                                 & (policy_medians.method == "estimated_weight_no_gate")].iloc[0]
            main_tex.append(" & ".join([DS[dataset], SHORT[detector], fmt(row.coverage), fmt(row.auto_fn_rate),
                                        fmt(row.auto_fp_rate), fmt(row.selective_error),
                                        f"{int(row.selective_error_defined_projects)}/{int(row.projects)}"]) + r"\\")
    main_tex.extend([r"\bottomrule\end{tabular}\end{table}", ""])
    si_tex.extend([
        r"\begin{longtable}{@{}llrrrr@{}}",
        r"\caption{Complete additional target cohort. Positive commits and their largest share describe benchmark label concentration rather than independent vulnerability events. Groups from different collections are not pooled.}\label{si:newcounts}\\",
        r"\toprule Collection & Group & Functions & Positive & Pos. commits & Largest share\\\midrule\endfirsthead",
        r"\toprule Collection & Group & Functions & Positive & Pos. commits & Largest share\\\midrule\endhead",
        r"\bottomrule\endfoot"])
    for _, row in counts.iterrows():
        si_tex.append(" & ".join([DS[row.dataset], escape(row.project), f"{int(row.n):,}", str(int(row.n_vulnerable)),
                                  str(int(row.positive_commits)), fmt(row.largest_positive_commit_fraction, 3)]) + r"\\")
    si_tex.extend([r"\end{longtable}", "",
                   r"\begin{table}[htbp]\centering\small",
                   r"\caption{Actual source-selected fits. Three fine-tuned encoders and three SGD detectors are independent fits; the frozen reference has one head. All are reused across target collections. Epoch is shown for fine-tuning; the other rows give selected $C$ or SGD $\alpha$.}",
                   r"\label{si:newfits}\begin{tabular}{@{}llrrr@{}}\toprule Detector & Seed & Epoch/parameter & Source AP & Changed tensors\\\midrule"])
    for _, row in fits.iterrows():
        parameter = (str(int(row.selected_epoch)) if row.detector == "finetuned_codebert"
                     else (f"$C={row.selected_C:g}$" if row.detector == "frozen_codebert"
                           else "$\\alpha=10^{" + str(int(round(np.log10(row.selected_alpha)))) + "}$"))
        si_tex.append(" & ".join([SHORT[row.detector], str(int(row.seed)), parameter,
                                  fmt(row.source_validation_ap), str(int(row.changed_encoder_tensors))]) + r"\\")
    si_tex.extend([r"\bottomrule\end{tabular}\end{table}", "",
                   r"\begin{table}[htbp]\centering\footnotesize",
                   r"\caption{Domain-supported secondary allocation comparisons. The roster was fixed by label-free availability for every actual fit. Differences are weighted minus allocation-matched tails. R/I/T counts project reductions, increases and ties separately for misses and false alerts. Pointwise intervals are descriptive and do not establish equivalence.}",
                   r"\label{si:newsupported}\begin{tabular}{@{}llcrcr c@{}}\toprule Collection & Detector & $n$ & $\Delta F_v$ [95\% interval] & R/I/T & $\Delta F_s$ [95\% interval] & R/I/T\\\midrule"])
    for dataset in DATASETS:
        cell = joint[joint.dataset == dataset]
        for detector in DETECTORS:
            c = paired(cell, detector, "class_matched_weighted", "auto_fn_rate", "domain_supported_secondary")
            f = paired(cell, detector, "class_matched_weighted", "auto_fp_rate", "domain_supported_secondary")
            si_tex.append(" & ".join([DS[dataset], SHORT[detector], str(int(c.projects)), estimate(c),
                                      f"{int(c.reductions)}/{int(c.increases)}/{int(c.ties)}", estimate(f),
                                      f"{int(f.reductions)}/{int(f.increases)}/{int(f.ties)}"]) + r"\\[2pt]")
    si_tex.extend([r"\bottomrule\end{tabular}\end{table}", "",
                   r"\begin{table}[htbp]\centering\small",
                   r"\caption{Label-free action structure at the primary budgets and clip. Counts require the condition in every actual fit. Exact equality refers to the three action codes and does not imply identical empty/doubleton sets or equal set miscoverage. Supported/all distinguishes fitted-domain cells from review fallback. The disagreement column is the median seed-averaged fraction of different weighted and tail actions.}",
                   r"\label{si:newstructure}\begin{tabular}{@{}llcrcr@{}}\toprule Collection & Detector & Supported/all & Full review & Equal supported/all & Disagreement\\\midrule"])
    structure = pd.read_csv(args.output / "allocation_structure_summary.csv")
    for _, row in structure.iterrows():
        si_tex.append(" & ".join([DS[row.dataset], SHORT[row.detector], f"{int(row.supported_projects)}/{int(row.projects)}",
                                  str(int(row.full_review_projects)),
                                  f"{int(row.supported_exactly_matched_policy_projects)}/{int(row.exactly_matched_policy_projects)}",
                                  fmt(row.median_allocation_disagreement)]) + r"\\")
    si_tex.extend([r"\bottomrule\end{tabular}\end{table}", "",
                   r"\begin{longtable}{@{}llrrrrrr@{}}",
                   r"\caption{Additional costs with source-selected free automation volume. All projects enter. Marginal median costs and automation fractions are listed alongside the paired weighted-minus-free-volume cost difference; a difference of marginal medians is not the paired estimate. Cost units assume correct review, false-alert cost 1 and review cost 0.1.}\label{si:newcosts}\\",
                   r"\toprule Collection & Detector & $c_v$ & $L_W$ & $L_F$ & $A_W$ & $A_F$ & $\Delta L$\\\midrule\endfirsthead",
                   r"\toprule Collection & Detector & $c_v$ & $L_W$ & $L_F$ & $A_W$ & $A_F$ & $\Delta L$\\\midrule\endhead",
                   r"\bottomrule\endfoot"])
    for _, row in cost[cost.workload == "free_volume"].iterrows():
        si_tex.append(" & ".join([DS[row.dataset], SHORT[row.detector], str(int(row.miss_cost)),
                                  fmt(row.weighted_cost), fmt(row.baseline_cost), fmt(row.weighted_coverage),
                                  fmt(row.baseline_coverage), fmt(row.median_difference)]) + r"\\")
    si_tex.extend([r"\end{longtable}", ""])
    args.main_tables.write_text("\n".join(main_tex), encoding="utf-8")
    args.tables.write_text("\n".join(si_tex), encoding="utf-8")
    write(args.output / "aggregate_export_checks.json", dict(status="PASS", check_count=len(checks), checks=checks,
          scope="Verified evaluation/report seals, complete aggregate grids, independent rate/cost arithmetic, fallback semantics and table-source bounds. Bootstrap replay is a separate check."))
    all_files = sorted(p for p in args.output.rglob("*") if p.is_file() and p.name != "public_analysis_manifest.json")
    write(args.output / "public_analysis_manifest.json", dict(schema_version="trace-v3-aggregate-release-1",
          created_at_utc=datetime.now(timezone.utc).isoformat(), scope="Complete aggregate evidence only",
          datasets={d: {k: receipts[d][k] for k in ("projects", "target_rows", "target_vulnerable", "actual_seeds", "decision_seal_sha256", "domain_failure_cells")}
                    for d in DATASETS},
          files_sha256={p.relative_to(args.output).as_posix(): sha(p) for p in all_files},
          table_source_sha256={args.main_tables.name: sha(args.main_tables), args.tables.name: sha(args.tables)},
          no_raw_datasets_predictions_weights_or_vaults=True,
          statistical_unit="Project within dataset, averaging actual fits. Two collections are never pooled."))
    print(json.dumps(dict(status="PASS", checks=len(checks), released_files=len(all_files),
                          target_projects_by_dataset={d: receipts[d]["projects"] for d in DATASETS}), indent=2))


if __name__ == "__main__":
    main()
