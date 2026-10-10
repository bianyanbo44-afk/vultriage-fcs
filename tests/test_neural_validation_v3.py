"""Boundary and allocation tests; every archive here is a synthetic fixture."""
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import evaluate_neural_validation_v3 as workflow
import vultriage.neural_audit as policies
from analyze_neurocomputing_controls import action_codes
from vultriage.data import sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def commit_id(token):
    return hashlib.sha256(str(token).encode()).hexdigest()[:40]


@pytest.fixture
def artifacts(tmp_path):
    """A complete tiny source/target archive, explicitly unrelated to real data."""
    config = json.loads((workflow.WORKSPACE / "configs/neural_validation_v3.json").read_text(encoding="utf-8"))
    cfg_path = tmp_path / "configs/neural_validation_v3.json"
    write_json(cfg_path, config)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source, target = [], []
    for index in range(67):
        role = "train" if index == 0 else ("model_validation" if index <= 6 else "calibration")
        source.append(dict(position=index, row_id=f"src-{index}", project_group="source-project",
                           commit_id=commit_id(f"{role}-commit-{index // 2}"), label=index % 2,
                           dataset="primevul", role=role, text="/* a synthetic label-free code string */"))
    for index in range(60):
        target.append(dict(position=index, row_id=f"target-{index}", project_group="new-project",
                           commit_id=commit_id(f"target-commit-{index // 2}"), dataset="diversevul", text="synthetic"))
    for name, rows in (("source_rows.jsonl", source), ("target_rows.jsonl", target)):
        (inputs / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    labels = np.arange(len(target)) % 2
    vault = inputs / "target_label_vault.npz"
    np.savez(vault, labels=labels, row_ids=np.asarray([r["row_id"] for r in target]), positions=np.arange(len(target)))
    prepare = {"selected_projects": ["new-project"], "output_hashes": {
        name: sha256(inputs / name) for name in ("source_rows.jsonl", "target_rows.jsonl", "target_label_vault.npz")}}
    write_json(inputs / "preparation_manifest.json", prepare)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    checkpoint = evidence / "classifier.bin"
    checkpoint.write_bytes(b"synthetic checkpoint fixture, not an actual detector")
    train = dict(seed=13, config_sha256=sha256(cfg_path), checkpoint_sha256=sha256(checkpoint),
                 target_labels_accessed=False, synthetic_fixture=True)
    write_json(evidence / "training_manifest.json", train)
    for name in ("source_token_manifest", "target_token_manifest"):
        is_source = name == "source_token_manifest"
        write_json(evidence / f"{name}.json", {"synthetic_fixture": True, "target_labels_present": False,
                                               "source_labels_present": is_source,
                                               "input_sha256": sha256(inputs / ("source_rows.jsonl" if is_source else "target_rows.jsonl"))})
    prediction = tmp_path / "predictions/frozen_codebert/seed-13.npz"
    prediction.parent.mkdir(parents=True)
    rng = np.random.default_rng(71)
    source_positions = np.arange(1, len(source), dtype=np.int64)
    arrays = dict(source_positions=source_positions,
                  source_probability=np.linspace(0.02, 0.98, len(source_positions)).astype(np.float32),
                  source_features=rng.normal(size=(len(source_positions), 768)).astype(np.float32),
                  target_positions=np.arange(len(target), dtype=np.int64),
                  target_probability=np.linspace(0, 1, len(target)).astype(np.float32),
                  target_features=rng.normal(size=(len(target), 768)).astype(np.float32))
    np.savez_compressed(prediction, **arrays)
    metadata = dict(detector="frozen_codebert", seed=13, dataset="diversevul",
                    prediction_sha256=sha256(prediction), config_sha256=sha256(cfg_path),
                    dataset_config_sha256=None, source_rows_sha256=sha256(inputs / "source_rows.jsonl"),
                    target_rows_sha256=sha256(inputs / "target_rows.jsonl"), target_labels_accessed=False)
    for name, path in (("training_manifest", evidence / "training_manifest.json"), ("checkpoint", checkpoint),
                       ("source_token_manifest", evidence / "source_token_manifest.json"),
                       ("target_token_manifest", evidence / "target_token_manifest.json")):
        metadata[f"{name}_path"] = str(path.relative_to(tmp_path))
        metadata[f"{name}_sha256"] = sha256(path)
    write_json(prediction.with_suffix(".json"), metadata)
    failure_path = tmp_path / "synthetic_final_failures.json"
    write_json(failure_path, dict(config_sha256=sha256(cfg_path), source_rows_sha256=sha256(inputs / "source_rows.jsonl"),
                                 target_labels_accessed=False, synthetic_fixture=True,
                                 failures=[dict(detector=d, seed=s, status="failed", stage="synthetic_test_fixture",
                                                error="Synthetic missing-model fixture; no real benchmark or fit")
                                           for d in ("finetuned_codebert", "hashing_sgd") for s in (13, 37, 73)]))
    return dict(workspace=tmp_path, config=cfg_path, inputs=inputs, prediction=prediction,
                arrays=arrays, metadata=metadata, prepare=prepare, source=source, target=target,
                labels=labels, vault=vault, output=tmp_path / "sealed", training_failures=failure_path)


def seal_fixture(value, **kwargs):
    kwargs.setdefault("training_failures", value["training_failures"])
    return workflow.seal(config=value["config"], inputs=value["inputs"], predictions=[value["prediction"]],
                         output=value["output"], workspace=value["workspace"], **kwargs)


@pytest.mark.parametrize("field", ["label", "target", "labels", "is_vul"])
def test_target_metadata_cannot_smuggle_outcomes_into_seal(artifacts, field):
    row = {**artifacts["target"][0], field: 1}
    path = artifacts["workspace"] / "contaminated.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains labels"):
        workflow._rows(path, source=False)


def test_seal_cli_has_no_target_label_argument():
    with pytest.raises(SystemExit) as error:
        workflow.build_parser().parse_args([
            "seal", "--config", "cfg", "--inputs", "inputs", "--predictions", "pred",
            "--output", "out", "--target-label-vault", "vault"])
    assert error.value.code == 2


def test_seal_never_opens_or_hashes_the_target_vault(artifacts, monkeypatch):
    # The prepared hash remains fixed, while the actual vault is now unreadable data.
    # Seal must still succeed; only the separate evaluate phase may discover this.
    artifacts["vault"].write_bytes(b"not an NPZ archive")
    real_hash = workflow.sha256

    def guard(path):
        assert Path(path).resolve() != artifacts["vault"].resolve(), "seal accessed target labels"
        return real_hash(path)

    monkeypatch.setattr(workflow, "sha256", guard)
    result = seal_fixture(artifacts)
    assert result["target_labels_accessed"] is False
    assert result["actual_seeds"] == {"frozen_codebert": [13]}
    assert result["all_planned_fits_present"] is False
    monkeypatch.setattr(workflow, "sha256", real_hash)
    with pytest.raises(ValueError, match="vault differs"):
        workflow.evaluate(seal_path=artifacts["output"] / "decision_seal.json",
                          target_label_vault=artifacts["vault"], output=artifacts["workspace"] / "evaluation")


def test_unfinished_planned_fits_block_evaluate_before_target_labels(artifacts, monkeypatch):
    result = seal_fixture(artifacts, training_failures=None)
    assert result["all_planned_attempts_accounted"] is False
    touched = []
    monkeypatch.setattr(workflow, "_load_target_vault", lambda *args: touched.append(True))
    with pytest.raises(ValueError, match="Planned fits remain unresolved"):
        workflow.evaluate(seal_path=artifacts["output"] / "decision_seal.json",
                          target_label_vault=artifacts["vault"], output=artifacts["workspace"] / "evaluation")
    assert not touched


def test_pending_status_cannot_be_reclassified_as_a_final_failure(artifacts):
    value = json.loads(artifacts["training_failures"].read_text())
    value["failures"][0]["status"] = "pending"
    write_json(artifacts["training_failures"], value)
    with pytest.raises(ValueError, match="final failed status"):
        seal_fixture(artifacts)


@pytest.mark.parametrize("tampered", ["archive", "metadata", "source_rows", "manifest"])
def test_tampering_stops_evaluation_before_any_target_label_access(artifacts, monkeypatch, tampered):
    result = seal_fixture(artifacts)
    path = {"archive": artifacts["output"] / result["decisions"][0]["archive"],
            "metadata": artifacts["output"] / result["decisions"][0]["metadata"],
            "source_rows": artifacts["inputs"] / "source_rows.jsonl",
            "manifest": artifacts["output"] / "decision_seal.json"}[tampered]
    with path.open("ab") as stream:
        stream.write(b"\nmodified after sealing\n")
    touched = []
    monkeypatch.setattr(workflow, "_load_target_vault", lambda *args: touched.append(True))
    with pytest.raises(ValueError, match="hash mismatch"):
        workflow.evaluate(seal_path=artifacts["output"] / "decision_seal.json",
                          target_label_vault=artifacts["vault"], output=artifacts["workspace"] / "evaluation")
    assert not touched


def test_vault_identity_alignment_is_checked_after_successful_verification(artifacts):
    np.savez(artifacts["vault"], labels=artifacts["labels"], positions=np.arange(len(artifacts["target"])),
             row_ids=np.asarray([r["row_id"] for r in artifacts["target"]])[::-1])
    artifacts["prepare"]["output_hashes"]["target_label_vault.npz"] = sha256(artifacts["vault"])
    write_json(artifacts["inputs"] / "preparation_manifest.json", artifacts["prepare"])
    seal_fixture(artifacts)
    with pytest.raises(ValueError, match="row IDs"):
        workflow.evaluate(seal_path=artifacts["output"] / "decision_seal.json",
                          target_label_vault=artifacts["vault"], output=artifacts["workspace"] / "evaluation")


def test_frozen_head_seed_copies_are_rejected(artifacts):
    artifacts["metadata"]["seed"] = 37
    write_json(artifacts["prediction"].with_suffix(".json"), artifacts["metadata"])
    with pytest.raises(ValueError, match="only seed 13"):
        seal_fixture(artifacts)


def test_predictor_folder_metadata_and_filename_binding_are_supported(artifacts):
    prediction = artifacts["prediction"].with_name("predictions.npz")
    artifacts["prediction"].replace(prediction)
    artifacts["prediction"].with_suffix(".json").unlink()
    artifacts["metadata"]["prediction_path"] = str(prediction.relative_to(artifacts["workspace"]))
    write_json(prediction.parent / "metadata.json", artifacts["metadata"])
    artifacts["prediction"] = prediction
    assert workflow._prediction_paths([prediction.parent]) == [prediction.resolve()]
    result = seal_fixture(artifacts)
    assert result["actual_seeds"] == {"frozen_codebert": [13]}


def test_duplicate_actual_fit_is_not_counted_twice(artifacts):
    second = artifacts["prediction"].with_name("copy.npz")
    second.write_bytes(artifacts["prediction"].read_bytes())
    write_json(second.with_suffix(".json"), artifacts["metadata"])
    with pytest.raises(ValueError, match="Duplicate actual detector fit"):
        workflow.seal(config=artifacts["config"], inputs=artifacts["inputs"],
                      predictions=[artifacts["prediction"], second], output=artifacts["output"],
                      workspace=artifacts["workspace"])


def test_encoder_update_evidence_is_required_for_a_fine_tuned_detector(artifacts):
    artifacts["metadata"]["detector"] = "finetuned_codebert"
    write_json(artifacts["prediction"].with_suffix(".json"), artifacts["metadata"])
    with pytest.raises(ValueError, match="full encoder updating"):
        seal_fixture(artifacts)


def test_stale_or_label_bearing_target_token_cache_is_rejected(artifacts):
    path = artifacts["workspace"] / artifacts["metadata"]["target_token_manifest_path"]
    value = json.loads(path.read_text())
    value["source_labels_present"] = True
    write_json(path, value)
    artifacts["metadata"]["target_token_manifest_sha256"] = sha256(path)
    write_json(artifacts["prediction"].with_suffix(".json"), artifacts["metadata"])
    with pytest.raises(ValueError, match="Token cache input/label boundary"):
        seal_fixture(artifacts)


def test_grouped_crossfit_never_trains_on_a_held_out_commit(monkeypatch):
    n = 120
    groups = [("source" if i < n // 2 else "target", commit_id(i // 3)) for i in range(n)]
    features = np.column_stack([np.arange(n), np.sin(np.arange(n))]).astype(np.float32)
    labels = np.asarray([0] * (n // 2) + [1] * (n // 2))
    real = policies.SGDClassifier
    observations = []

    class RecordingClassifier:
        def __init__(self, **kwargs):
            self.model = real(**kwargs)

        def fit(self, x, y):
            self.train = x[:, 0].astype(int)
            assert np.sum(y == 0) == np.sum(y == 1)
            observations.append(self.train)
            self.model.fit(x, y)

        def predict_proba(self, x):
            held = x[:, 0].astype(int)
            assert {groups[i] for i in self.train}.isdisjoint({groups[i] for i in held})
            assert {groups[i][1] for i in self.train}.isdisjoint({groups[i][1] for i in held})
            return self.model.predict_proba(x)

    monkeypatch.setattr(policies, "SGDClassifier", RecordingClassifier)
    result = policies.crossfit_grouped_density_ratio(features, [f"id-{i}" for i in range(n)], labels, groups, seed=13)
    assert len(observations) == 5
    assert np.isfinite(result.raw_ratios).all()
    for indices in np.arange(n).reshape(-1, 3):
        assert len(set(result.fold_ids[indices])) == 1
    folds = policies.grouped_fold_ids([("SOURCE", "A" * 40), ("source", "a" * 40)])
    assert folds[0] == folds[1]


def test_single_target_commit_cannot_leak_to_rescue_an_empty_complement():
    groups = [("source", commit_id(i)) for i in range(40)] + [("target", commit_id("one-commit"))] * 20
    with pytest.raises(policies.DomainFitUnavailable, match="empty training domain"):
        policies.crossfit_grouped_density_ratio(
            np.zeros((60, 3), dtype=np.float32), [f"id-{i}" for i in range(60)],
            np.asarray([0] * 40 + [1] * 20), groups, seed=13)


def test_global_commit_sha_rejects_url_and_project_alias_role_leakage():
    sha = "b" * 40
    source = [dict(project_group="php", commit_id=sha.upper() + "?view=patch", role="train"),
              dict(project_group="php-src", commit_id=f"https://git.example/commit/{sha}#diff", role="calibration")]
    target = [dict(project_group="target", commit_id="c" * 40)]
    with pytest.raises(ValueError, match="multiple roles"):
        workflow._verify_role_boundaries(source, target)
    source = source[:1]
    target = [dict(project_group="renamed-target", commit_id=f"https://git.example/commit/{sha}?raw=true")]
    with pytest.raises(ValueError, match="commits overlap"):
        workflow._verify_role_boundaries(source, target)
    # Even within one role, alternate repository names cannot split this event.
    folds = policies.grouped_fold_ids([("fork-a", sha), ("fork-b", sha + "?view=patch")])
    assert folds[0] == folds[1]


def test_actual_seed_average_preserves_undefined_selective_error():
    rows = [dict(dataset="fixture", detector="finetuned_codebert", project="p", alpha_v=0.1,
                 alpha_s=0.2, clip=20, method="weighted", seed=seed, status="ok",
                 auto_fn_rate=rate, selective_error=selective)
            for seed, rate, selective in ((13, 0.1, 0.2), (37, 0.2, np.nan), (73, 0.3, 0.4))]
    averaged = policies.average_actual_seed_metrics(pd.DataFrame(rows), {"finetuned_codebert": [13, 37, 73]})
    assert averaged.iloc[0]["actual_seed_count"] == 3
    assert averaged.iloc[0]["auto_fn_rate"] == pytest.approx(0.2)
    assert np.isnan(averaged.iloc[0]["selective_error"])


def test_complete_decision_grid_preserves_workload_and_each_allocation(artifacts):
    cfg = json.loads(artifacts["config"].read_text())
    probability = np.asarray([0.0, 0.1, 0.5, 0.5, 0.9, 1.0])
    cal_probability = np.linspace(0, 1, 60)
    decisions, specs = policies.construct_decisions(
        cal_probability, np.arange(60) % 2, probability, probability,
        np.linspace(0.02, 80, 60), np.asarray([0.01, 1, 3, 50, 2, 20]), cfg)
    assert len(specs) == 9 * 3 * len(policies.METHODS)
    for av, ass, clip in product(cfg["risk_budgets"]["vulnerable"], cfg["risk_budgets"]["safe"], cfg["weighting"]["clips"]):
        codes = action_codes(decisions[policies.cell_key(av, ass, clip, "estimated_weight_no_gate")])
        tail = action_codes(decisions[policies.cell_key(av, ass, clip, "class_matched_weighted")])
        assert np.sum(tail == 0) == np.sum(codes == 0)
        assert np.sum(tail == 1) == np.sum(codes == 1)
        for name in ("msp_matched_weighted", "source_iso_cost10_matched_weighted", "source_iso_cost50_matched_weighted"):
            assert np.sum(action_codes(decisions[policies.cell_key(av, ass, clip, name)]) != 2) == np.sum(codes != 2)
        assert not (set(np.flatnonzero(tail == 0)) & set(np.flatnonzero(tail == 1)))


def test_unweighted_small_class_keeps_test_point_mass_at_infinity(artifacts):
    cfg = json.loads(artifacts["config"].read_text())
    decisions, specs = policies.construct_decisions(
        np.asarray([0.01, 0.99]), np.asarray([0, 1]), np.asarray([0.0, 0.5, 1.0]),
        np.asarray([0.0, 0.5, 1.0]), np.ones(2), np.ones(3), cfg)
    # With only one calibration score per class, finite calibration mass is 1/2.
    # Every declared 1-alpha level is above 1/2, so the threshold is infinity.
    for spec in specs:
        if spec["method"] in ("unweighted_mondrian", "estimated_weight_no_gate"):
            assert decisions[spec["key"]].all()
            assert np.all(action_codes(decisions[spec["key"]]) == 2)


def test_empty_and_double_sets_review_without_becoming_automatic_misses(artifacts):
    cfg = json.loads(artifacts["config"].read_text())
    labels = np.asarray([1, 1, 0, 0])
    sets = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]])
    result = policies.project_metrics(labels, np.asarray([0.5, 0.1, 0.9, 0.5]), sets, 0.1, 0.2, cfg)
    assert (result["empty_count"], result["doubleton_count"], result["review_count"]) == (1, 1, 2)
    assert result["auto_fn_rate"] == result["auto_fp_rate"] == 0.5
    assert result["vulnerable_miscoverage"] == 1
    assert result["selective_error"] == 1
    assert result["cost_10"] == pytest.approx((10 + 1 + 0.2) / 4)


def test_free_volume_policy_minimizes_cost_and_review_wins_acceptance_ties():
    eta = np.asarray([0, 0.1, 0.5, 1])
    codes = action_codes(policies.free_volume_cost_sets(eta, 1, review_cost=0.1))
    assert codes.tolist() == [0, 2, 2, 1]
    chosen_cost = sum(e if code == 0 else 1 - e if code == 1 else 0.1 for e, code in zip(eta, codes))
    feasible = [sum(e if code == 0 else 1 - e if code == 1 else 0.1 for e, code in zip(eta, actions))
                for actions in product([0, 1, 2], repeat=len(eta))]
    assert chosen_cost == pytest.approx(min(feasible))
    assert np.all(action_codes(policies.free_volume_cost_sets(eta, 1, review_cost=0)) == 2)


def test_failed_or_undefined_project_is_retained_without_a_biased_interval():
    frame = pd.DataFrame([dict(project=p, method=m, status="ok", auto_fn_rate=v)
                          for m, values in (("weighted", [0.1, 0.2, 0.3]), ("tails", [0.2, 0.2, 0.2]))
                          for p, v in zip(["small", "large", "other"], values)])
    result = policies.paired_project_summary(frame, "auto_fn_rate", "tails", "weighted")
    assert result["projects"] == 3
    assert result["median_difference"] == 0
    frame.loc[0, "auto_fn_rate"] = np.nan
    result = policies.paired_project_summary(frame, "auto_fn_rate", "tails", "weighted")
    assert result["projects"] == 3 and result["defined_projects"] == 2
    assert result["status"] == "undefined_endpoint_no_project_dropped"
    assert np.isnan(result["ci_lower"])
    frame.loc[0, ["auto_fn_rate", "status"]] = [0, "policy_failure_present"]
    result = policies.paired_project_summary(frame, "auto_fn_rate", "tails", "weighted")
    assert result["projects"] == 3 and np.isnan(result["median_difference"])
    assert result["status"] == "incomplete_policy_failure_no_project_dropped"


def test_full_cohort_deployment_keeps_fallback_and_reports_supported_counts():
    frame = pd.DataFrame([dict(project=p, method=m, status="policy_failure_present" if p == "fallback" else "ok", auto_fn_rate=v)
                          for m, values in (("weighted", [0, 0.1, 0.2]), ("tails", [0, 0.2, 0.3]))
                          for p, v in zip(["fallback", "supported-a", "supported-b"], values)])
    result = policies.paired_project_summary(
        frame, "auto_fn_rate", "tails", "weighted", include_review_fallback=True,
        all_project_count=3, domain_supported_projects=["supported-a", "supported-b"])
    assert result["median_difference"] == pytest.approx(-0.1)
    assert (result["all_projects"], result["eligible_projects"], result["defined_projects"],
            result["domain_supported_projects"], result["review_fallback_projects"]) == (3, 3, 3, 2, 1)
    assert result["status"] == "deployment_review_fallback_included_not_selection_evidence"


def test_seed_rows_cannot_be_treated_as_independent_projects():
    frame = pd.DataFrame([dict(project="p", method=m, status="ok", auto_fn_rate=0.1)
                          for m in ("weighted", "tails") for _ in range(3)])
    with pytest.raises(ValueError, match="one seed-averaged row"):
        policies.paired_project_summary(frame, "auto_fn_rate", "tails", "weighted")


def test_cpu_seal_evaluate_reports_only_one_actual_frozen_fit(artifacts):
    seal_fixture(artifacts)
    result = workflow.evaluate(seal_path=artifacts["output"] / "decision_seal.json",
                               target_label_vault=artifacts["vault"], output=artifacts["workspace"] / "evaluation")
    assert result["verified_before_target_label_access"] is True
    assert result["actual_seeds"] == {"frozen_codebert": [13]}
    averaged = pd.read_csv(artifacts["workspace"] / "evaluation/project_seed_averages.csv")
    assert averaged["actual_seed_count"].eq(1).all()
    assert len(averaged) == 9 * 3 * len(policies.METHODS)
    counts = pd.read_csv(artifacts["workspace"] / "evaluation/project_counts_positive_commits.csv")
    assert counts.iloc[0]["positive_commits"] == 30
    assert counts.iloc[0]["n_vulnerable"] == 30
    primary = pd.read_csv(artifacts["workspace"] / "evaluation/primary_paired_summary.csv")
    assert len(primary) == 1 and primary.iloc[0]["projects"] == 1


def test_unavailable_raw_sets_do_not_earn_zero_violation_and_roster_is_presealed(artifacts):
    for row in artifacts["target"]:
        row["commit_id"] = commit_id("one target commit")
    target_path = artifacts["inputs"] / "target_rows.jsonl"
    target_path.write_text("".join(json.dumps(row) + "\n" for row in artifacts["target"]), encoding="utf-8")
    artifacts["metadata"]["target_rows_sha256"] = sha256(target_path)
    token_path = artifacts["workspace"] / artifacts["metadata"]["target_token_manifest_path"]
    token = json.loads(token_path.read_text())
    token["input_sha256"] = sha256(target_path)
    write_json(token_path, token)
    artifacts["metadata"]["target_token_manifest_sha256"] = sha256(token_path)
    write_json(artifacts["prediction"].with_suffix(".json"), artifacts["metadata"])
    artifacts["prepare"]["output_hashes"]["target_rows.jsonl"] = sha256(target_path)
    write_json(artifacts["inputs"] / "preparation_manifest.json", artifacts["prepare"])
    sealed = seal_fixture(artifacts)
    assert sealed["domain_supported_projects"] == {"frozen_codebert": []}
    assert sealed["domain_unsupported_projects"] == {"frozen_codebert": ["new-project"]}
    result = workflow.evaluate(seal_path=artifacts["output"] / "decision_seal.json",
                               target_label_vault=artifacts["vault"], output=artifacts["workspace"] / "evaluation")
    assert result["domain_failure_cells"] == 1
    frame = pd.read_csv(artifacts["workspace"] / "evaluation/project_seed_metrics.csv")
    weighted = frame[frame.method == "estimated_weight_no_gate"]
    assert weighted["auto_fn_rate"].eq(0).all() and weighted["coverage"].eq(0).all()
    assert weighted["maximum_relative_violation"].isna().all()
    primary = pd.read_csv(artifacts["workspace"] / "evaluation/primary_paired_summary.csv")
    assert primary.iloc[0]["median_difference"] == 0
    assert primary.iloc[0]["projects"] == primary.iloc[0]["all_projects"] == primary.iloc[0]["defined_projects"] == 1
    assert primary.iloc[0]["domain_supported_projects"] == 0
    secondary = pd.read_csv(artifacts["workspace"] / "evaluation/domain_supported_paired_summaries.csv")
    assert secondary["eligible_projects"].eq(0).all()
    assert secondary["all_projects"].eq(1).all()
    assert secondary["median_difference"].isna().all()


def test_megavul_reuses_source_jsonl_without_a_copy(artifacts):
    shared = artifacts["workspace"] / "shared/source_rows.jsonl"
    shared.parent.mkdir()
    (artifacts["inputs"] / "source_rows.jsonl").replace(shared)
    for row in artifacts["target"]:
        row["dataset"] = "megavul"
    (artifacts["inputs"] / "target_rows.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in artifacts["target"]), encoding="utf-8")
    ds_path = artifacts["workspace"] / "configs/megavul_validation_v3.json"
    write_json(ds_path, {"protocol_version": "trace-second-source-validation-v3", "synthetic_fixture": True})
    artifacts["metadata"].update(dataset="megavul", dataset_config_sha256=sha256(ds_path),
                                  target_rows_sha256=sha256(artifacts["inputs"] / "target_rows.jsonl"))
    token_path = artifacts["workspace"] / artifacts["metadata"]["target_token_manifest_path"]
    token_manifest = json.loads(token_path.read_text())
    token_manifest["input_sha256"] = artifacts["metadata"]["target_rows_sha256"]
    write_json(token_path, token_manifest)
    artifacts["metadata"]["target_token_manifest_sha256"] = sha256(token_path)
    write_json(artifacts["prediction"].with_suffix(".json"), artifacts["metadata"])
    artifacts["prepare"]["output_hashes"].pop("source_rows.jsonl")
    artifacts["prepare"]["output_hashes"]["target_rows.jsonl"] = artifacts["metadata"]["target_rows_sha256"]
    write_json(artifacts["inputs"] / "preparation_manifest.json", artifacts["prepare"])
    result = seal_fixture(artifacts, source_rows=shared, dataset_config=ds_path)
    assert result["dataset"] == "megavul"
    assert Path(result["source_rows_path"]) == shared
    assert not (artifacts["inputs"] / "source_rows.jsonl").exists()


def test_hashing_cache_is_bound_to_full_source_and_target_positions(artifacts):
    artifacts["metadata"]["detector"] = "hashing_sgd"
    arrays = {k: v for k, v in artifacts["arrays"].items() if not k.endswith("features")}
    np.savez_compressed(artifacts["prediction"], **arrays)
    artifacts["metadata"]["prediction_sha256"] = sha256(artifacts["prediction"])
    cache = artifacts["workspace"] / "hashing-cache"
    cache.mkdir()
    rng = np.random.default_rng(13)
    sparse.save_npz(cache / "source_features.npz", sparse.csr_matrix(rng.normal(size=(len(artifacts["source"]), 4))))
    sparse.save_npz(cache / "target_features.npz", sparse.csr_matrix(rng.normal(size=(len(artifacts["target"]), 4))))
    write_json(cache / "manifest.json", {"source_rows_sha256": artifacts["metadata"]["source_rows_sha256"],
                                         "target_rows_sha256": artifacts["metadata"]["target_rows_sha256"],
                                         "hashes": {name: sha256(cache / name) for name in ("source_features.npz", "target_features.npz")}})
    for prefix in ("source_token_manifest", "target_token_manifest"):
        artifacts["metadata"].pop(f"{prefix}_path")
        artifacts["metadata"].pop(f"{prefix}_sha256")
    artifacts["metadata"].update(feature_cache_path=str(cache.relative_to(artifacts["workspace"])),
                                  feature_cache_manifest_sha256=sha256(cache / "manifest.json"))
    write_json(artifacts["prediction"].with_suffix(".json"), artifacts["metadata"])
    result = seal_fixture(artifacts, feature_cache=cache, training_failures=None)
    assert result["actual_seeds"] == {"hashing_sgd": [13]}
