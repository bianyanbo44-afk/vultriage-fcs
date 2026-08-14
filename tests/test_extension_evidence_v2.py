from pathlib import Path

import numpy as np

from build_extension_v2_evidence import bootstrap_median, parse_support


def test_evidence_support_parser_accepts_json_and_python_literal():
    assert parse_support('{"passed": true}') == {"passed": True}
    assert parse_support("{'passed': False}") == {"passed": False}
    assert parse_support(float("nan")) is None


def test_evidence_bootstrap_is_deterministic():
    values = np.asarray([0.1, 0.2, 0.3, 0.4])
    assert bootstrap_median(values, seed=17, replicates=200) == bootstrap_median(
        values, seed=17, replicates=200
    )


def test_evidence_validation_artifact_is_present_and_passed():
    root = Path(__file__).parents[1]
    report_candidates = (
        root / "public_results" / "extension-v2" / "evidence-validation-v2" / "evidence_validation.json",
        root / "outputs" / "extension-v2" / "evidence-validation-v2" / "evidence_validation.json",
    )
    report = next((candidate for candidate in report_candidates if candidate.is_file()), None)
    assert report is not None
    assert '"status": "PASS"' in report.read_text(encoding="utf-8")
