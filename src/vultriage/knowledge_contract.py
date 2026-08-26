"""Executable detector-shift knowledge contracts for selective automation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


class TruthValue(str, Enum):
    """Three-valued evidence state used by fail-closed contract rules."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class ContractAction(str, Enum):
    AUTO_SAFE = "auto-safe"
    AUTO_VULNERABLE = "auto-vulnerable"
    REVIEW = "review"


@dataclass(frozen=True)
class DetectorShiftIdentity:
    detector: str
    source_domain: str
    target_domain: str
    policy_version: str

    @property
    def contract_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class QualificationMetrics:
    projects: int
    severe_projects: int
    nonsevere_projects: int
    gate_auroc: float
    severe_review_rate: float
    nonsevere_pass_rate: float
    balanced_triage_accuracy: float
    qualification_score: float

    def qualifies(self, threshold: float) -> bool:
        return (
            self.severe_projects > 0
            and self.nonsevere_projects > 0
            and np.isfinite(self.qualification_score)
            and self.qualification_score >= threshold
        )


@dataclass(frozen=True)
class ContractEvidence:
    evidence_complete: bool | None
    gate_passed: bool | None
    detector_shift_match: bool
    registry_qualified: bool
    prediction_set: tuple[int, ...]
    evidence_payload: Mapping[str, Any]


@dataclass(frozen=True)
class DecisionCertificate:
    contract_id: str
    detector: str
    source_domain: str
    target_domain: str
    policy_version: str
    alpha_vulnerable: float
    alpha_safe: float
    prediction_set: tuple[int, ...]
    rule_truths: Mapping[str, str]
    entitlement: str
    action: str
    evidence_digest: str


def _truth(value: bool | None) -> TruthValue:
    if value is None:
        return TruthValue.UNKNOWN
    return TruthValue.TRUE if value else TruthValue.FALSE


def conjunction(values: Iterable[TruthValue]) -> TruthValue:
    values = tuple(values)
    if any(value is TruthValue.FALSE for value in values):
        return TruthValue.FALSE
    if any(value is TruthValue.UNKNOWN for value in values):
        return TruthValue.UNKNOWN
    return TruthValue.TRUE


def stable_project_partition(
    projects: Sequence[str], salt: str, registration_size: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    unique = sorted(set(projects))
    if len(unique) != len(projects):
        raise ValueError("project names must be unique")
    if not 0 < registration_size < len(unique):
        raise ValueError("registration size must split the project collection")
    ordered = sorted(
        unique,
        key=lambda project: hashlib.sha256(
            f"{salt}|{project}".encode("utf-8")
        ).hexdigest(),
    )
    return tuple(ordered[:registration_size]), tuple(ordered[registration_size:])


def qualification_metrics(
    severe: Sequence[int] | np.ndarray,
    gate_probability: Sequence[float] | np.ndarray,
    passed: Sequence[bool] | np.ndarray,
) -> QualificationMetrics:
    y = np.asarray(severe, dtype=int)
    score = np.asarray(gate_probability, dtype=float)
    gate_pass = np.asarray(passed, dtype=bool)
    if y.ndim != 1 or score.ndim != 1 or gate_pass.ndim != 1:
        raise ValueError("qualification inputs must be one-dimensional")
    if not (len(y) == len(score) == len(gate_pass)) or len(y) == 0:
        raise ValueError("qualification inputs must be nonempty and aligned")
    if not set(np.unique(y)).issubset({0, 1}) or not np.isfinite(score).all():
        raise ValueError("qualification labels or scores are invalid")
    severe_mask = y == 1
    nonsevere_mask = y == 0
    severe_n = int(severe_mask.sum())
    nonsevere_n = int(nonsevere_mask.sum())
    if severe_n and nonsevere_n:
        auroc = float(roc_auc_score(y, score))
        severe_review = float(np.mean(~gate_pass[severe_mask]))
        nonsevere_pass = float(np.mean(gate_pass[nonsevere_mask]))
        balanced = float((severe_review + nonsevere_pass) / 2.0)
        qualification = float(min(auroc, balanced))
    else:
        auroc = float("nan")
        severe_review = float("nan") if not severe_n else float(
            np.mean(~gate_pass[severe_mask])
        )
        nonsevere_pass = float("nan") if not nonsevere_n else float(
            np.mean(gate_pass[nonsevere_mask])
        )
        balanced = float("nan")
        qualification = float("nan")
    return QualificationMetrics(
        projects=len(y),
        severe_projects=severe_n,
        nonsevere_projects=nonsevere_n,
        gate_auroc=auroc,
        severe_review_rate=severe_review,
        nonsevere_pass_rate=nonsevere_pass,
        balanced_triage_accuracy=balanced,
        qualification_score=qualification,
    )


def execute_contract(
    identity: DetectorShiftIdentity,
    evidence: ContractEvidence,
    alpha_vulnerable: float,
    alpha_safe: float,
) -> DecisionCertificate:
    budget_valid = bool(
        np.isfinite([alpha_vulnerable, alpha_safe]).all()
        and 0.0 < alpha_vulnerable < 1.0
        and 0.0 < alpha_safe < 1.0
    )
    pair_qualified = evidence.detector_shift_match and evidence.registry_qualified
    gate_truth: bool | None
    if not pair_qualified:
        gate_truth = False
    else:
        gate_truth = evidence.gate_passed
    singleton_valid = (
        len(evidence.prediction_set) == 1
        and evidence.prediction_set[0] in (0, 1)
    )
    rule_values = {
        "K1_budget_bound": _truth(budget_valid),
        "K2_evidence_complete": _truth(evidence.evidence_complete),
        "K3_detector_shift_qualified": _truth(gate_truth),
        "K4_singleton_eligible": _truth(singleton_valid),
    }
    entitlement = conjunction(rule_values.values())
    if entitlement is TruthValue.TRUE and evidence.prediction_set == (0,):
        action = ContractAction.AUTO_SAFE
    elif entitlement is TruthValue.TRUE and evidence.prediction_set == (1,):
        action = ContractAction.AUTO_VULNERABLE
    else:
        action = ContractAction.REVIEW
    digest_payload = json.dumps(
        dict(evidence.evidence_payload), sort_keys=True, separators=(",", ":"), default=str
    )
    evidence_digest = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
    return DecisionCertificate(
        contract_id=identity.contract_id,
        detector=identity.detector,
        source_domain=identity.source_domain,
        target_domain=identity.target_domain,
        policy_version=identity.policy_version,
        alpha_vulnerable=float(alpha_vulnerable),
        alpha_safe=float(alpha_safe),
        prediction_set=tuple(evidence.prediction_set),
        rule_truths={key: value.value for key, value in rule_values.items()},
        entitlement=entitlement.value,
        action=action.value,
        evidence_digest=evidence_digest,
    )


def verify_certificate(
    certificate: DecisionCertificate,
    identity: DetectorShiftIdentity,
    evidence: ContractEvidence,
) -> bool:
    try:
        expected = execute_contract(
            identity,
            evidence,
            certificate.alpha_vulnerable,
            certificate.alpha_safe,
        )
        return certificate == expected
    except (TypeError, ValueError):
        return False
