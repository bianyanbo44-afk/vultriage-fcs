from dataclasses import replace

import numpy as np

from vultriage.knowledge_contract import (
    ContractAction,
    ContractEvidence,
    DetectorShiftIdentity,
    TruthValue,
    conjunction,
    execute_contract,
    qualification_metrics,
    stable_project_partition,
    verify_certificate,
)


def identity() -> DetectorShiftIdentity:
    return DetectorShiftIdentity(
        detector="hashing",
        source_domain="PrimeVul",
        target_domain="DiverseVul",
        policy_version="contract-v1",
    )


def evidence(**updates):
    values = {
        "evidence_complete": True,
        "gate_passed": True,
        "detector_shift_match": True,
        "registry_qualified": True,
        "prediction_set": (1,),
        "evidence_payload": {"gate_probability": 0.2},
    }
    values.update(updates)
    return ContractEvidence(**values)


def test_three_valued_conjunction_fails_closed():
    assert conjunction([TruthValue.TRUE, TruthValue.TRUE]) is TruthValue.TRUE
    assert conjunction([TruthValue.TRUE, TruthValue.UNKNOWN]) is TruthValue.UNKNOWN
    assert conjunction([TruthValue.UNKNOWN, TruthValue.FALSE]) is TruthValue.FALSE


def test_stable_partition_is_deterministic_and_disjoint():
    projects = [f"p{index}" for index in range(8)]
    first = stable_project_partition(projects, "salt", 4)
    second = stable_project_partition(projects, "salt", 4)
    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert set(first[0]) | set(first[1]) == set(projects)


def test_qualification_score_combines_ranking_and_triage():
    metric = qualification_metrics(
        severe=np.array([0, 0, 1, 1]),
        gate_probability=np.array([0.1, 0.2, 0.8, 0.9]),
        passed=np.array([True, True, False, False]),
    )
    assert metric.gate_auroc == 1.0
    assert metric.balanced_triage_accuracy == 1.0
    assert metric.qualification_score == 1.0
    assert metric.qualifies(0.7)


def test_contract_issues_verified_automatic_certificate():
    expected_identity = identity()
    expected_evidence = evidence()
    certificate = execute_contract(expected_identity, expected_evidence, 0.1, 0.2)
    assert certificate.action == ContractAction.AUTO_VULNERABLE.value
    assert certificate.entitlement == TruthValue.TRUE.value
    assert verify_certificate(certificate, expected_identity, expected_evidence)


def test_missing_evidence_and_pair_mismatch_cannot_escalate():
    missing_evidence = evidence(evidence_complete=None)
    mismatch_evidence = evidence(detector_shift_match=False)
    missing = execute_contract(
        identity(), missing_evidence, 0.1, 0.2
    )
    mismatch = execute_contract(
        identity(), mismatch_evidence, 0.1, 0.2
    )
    assert missing.action == ContractAction.REVIEW.value
    assert missing.entitlement == TruthValue.UNKNOWN.value
    assert mismatch.action == ContractAction.REVIEW.value
    assert mismatch.entitlement == TruthValue.FALSE.value
    assert verify_certificate(missing, identity(), missing_evidence)
    assert verify_certificate(mismatch, identity(), mismatch_evidence)


def test_certificate_tampering_is_rejected():
    expected_identity = identity()
    expected_evidence = evidence()
    certificate = execute_contract(expected_identity, expected_evidence, 0.1, 0.2)
    downgraded = replace(certificate, action=ContractAction.REVIEW.value)
    relabeled = replace(certificate, action=ContractAction.AUTO_SAFE.value)
    altered_identity = replace(expected_identity, detector="codebert")
    altered_evidence = evidence(evidence_payload={"gate_probability": 0.9})
    assert not verify_certificate(downgraded, expected_identity, expected_evidence)
    assert not verify_certificate(relabeled, expected_identity, expected_evidence)
    assert not verify_certificate(certificate, altered_identity, expected_evidence)
    assert not verify_certificate(certificate, expected_identity, altered_evidence)
