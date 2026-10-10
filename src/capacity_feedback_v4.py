"""Known-outcome v4 hard-capacity and reviewed-label feedback benchmark audit.

The detector and TRACE memberships are read from immutable v3.1 seals.  Pure
policy functions have no target-label arguments.  The delayed oracle is the
only label interface used by sequential calibration.  Outcome metrics are
computed only after new capacity/feedback decisions have been sealed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression

from analyze_neurocomputing_controls import action_codes, probability_vector
from evaluate_neural_validation_v3 import _load_target_vault, verify_seal
from vultriage.data import sha256
from vultriage.neural_audit import canonical_commit_id, cell_key

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "tmp/neurocomputing-strengthening/kbs-v4/capacity_feedback/freeze_receipt.json"
TESTS = ROOT / "scripts/test_capacity_feedback_v4.py"
TARGET_LABEL_KEYS = frozenset({"label", "labels", "target", "target_label", "target_labels", "is_vul", "is_vulnerable", "vulnerable", "y"})
REVIEW, BACKLOG = 2, 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(clean_json(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest().upper()


def number_key(value: float) -> str:
    return format(float(value), ".12g").replace(".", "p")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def capacity_integer(value: Any) -> int:
    require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= 0, "Review cap must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class CostScenario:
    miss: float
    false_alert: float = 1.0
    review: float = 0.1
    review_fn_probability: float = 0.0
    review_fp_probability: float = 0.0

    def __post_init__(self):
        for value in (self.miss, self.false_alert, self.review):
            require(not isinstance(value, bool) and np.isfinite(value) and value >= 0, "Invalid scenario cost")
        require(self.miss > 0 and self.false_alert > 0, "Automatic error costs must be positive")
        for value in (self.review_fn_probability, self.review_fp_probability):
            require(np.isfinite(value) and 0 <= value <= 1, "Invalid review-error scenario probability")

    @property
    def key(self) -> str:
        return f"m{number_key(self.miss)}_r{number_key(self.review)}_fn{number_key(self.review_fn_probability)}_fp{number_key(self.review_fp_probability)}"

    def fields(self) -> dict:
        return {"scenario": self.key, "miss_cost": self.miss, "false_alert_cost": self.false_alert, "review_cost": self.review,
                "review_fn_probability": self.review_fn_probability, "review_fp_probability": self.review_fp_probability}


def expected_action_costs(eta: np.ndarray, scenario: CostScenario) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta = probability_vector(eta)
    safe = scenario.miss * eta
    vulnerable = scenario.false_alert * (1 - eta)
    review = (np.full(len(eta), scenario.review) + scenario.miss * scenario.review_fn_probability * eta
              + scenario.false_alert * scenario.review_fp_probability * (1 - eta))
    return safe, vulnerable, review


def allocate_review(eta: np.ndarray, cap: int, scenario: CostScenario, *, mandatory_codes: np.ndarray | None = None,
                    stable_positions: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Pure source-probability policy; mandatory overflow remains BACKLOG.

    Source cost review uses positive net savings only.  A hard cap never changes
    original mandatory singleton decisions into a forced automatic fallback.
    """
    eta, cap = probability_vector(eta), capacity_integer(cap)
    n = len(eta)
    positions = np.arange(n) if stable_positions is None else np.asarray(stable_positions)
    require(positions.shape == (n,) and len(set(positions.tolist())) == n, "Stable review positions must be unique and aligned")
    safe, vulnerable, review_cost = expected_action_costs(eta, scenario)
    saving = np.minimum(safe, vulnerable) - review_cost
    if mandatory_codes is None:
        codes = (vulnerable <= safe).astype(np.uint8)
        candidates = np.flatnonzero(saving > 0)
    else:
        initial = np.asarray(mandatory_codes)
        require(initial.shape == (n,) and np.isin(initial, [0, 1, REVIEW]).all(), "Invalid mandatory membership codes")
        codes = initial.astype(np.uint8, copy=True)
        candidates = np.flatnonzero(initial == REVIEW)
        codes[candidates] = BACKLOG
    order = np.lexsort((positions[candidates], -saving[candidates]))
    selected = candidates[order[:min(cap, len(candidates))]]
    codes[selected] = REVIEW
    require(int(np.sum(codes == REVIEW)) <= cap, "Review policy exceeded cap")
    return codes, len(candidates)


def msp_review(probability: np.ndarray, cap: int, *, stable_positions: np.ndarray | None = None) -> np.ndarray:
    probability, cap = probability_vector(probability), capacity_integer(cap)
    positions = np.arange(len(probability)) if stable_positions is None else np.asarray(stable_positions)
    require(positions.shape == probability.shape and len(set(positions.tolist())) == len(probability), "MSP stable positions must be unique")
    confidence = np.maximum(probability, 1 - probability)
    order = np.lexsort((positions, confidence))
    codes = (probability > 0.5).astype(np.uint8)
    codes[order[:min(cap, len(probability))]] = REVIEW
    return codes


def capacity_policies(probability: np.ndarray, eta: np.ndarray, memberships: dict[str, np.ndarray], cap: int,
                      scenario: CostScenario, positions: np.ndarray) -> dict[str, tuple[np.ndarray, int]]:
    results = {}
    for method in ("trace_mandatory", "allocation_matched_tails_mandatory", "weighted_volume_msp_mandatory", "all_review_mandatory"):
        membership = (np.full(len(probability), REVIEW, dtype=np.uint8) if method == "all_review_mandatory" else memberships[method])
        results[method] = allocate_review(eta, cap, scenario, mandatory_codes=membership, stable_positions=positions)
    results["source_iso_cap"] = allocate_review(eta, cap, scenario, stable_positions=positions)
    results["msp_cap"] = msp_review(probability, cap, stable_positions=positions), min(cap, len(probability))
    results["forced_argmax"] = (np.asarray(probability) > 0.5).astype(np.uint8), 0
    return results


def action_counts(codes: np.ndarray) -> dict:
    codes = np.asarray(codes)
    require(codes.ndim == 1 and np.isin(codes, [0, 1, REVIEW, BACKLOG]).all(), "Invalid full-roster actions")
    result = {"n": len(codes), "auto_safe": int(np.sum(codes == 0)), "auto_vulnerable": int(np.sum(codes == 1)),
              "reviewed": int(np.sum(codes == REVIEW)), "unresolved": int(np.sum(codes == BACKLOG))}
    result["automatic"] = result["auto_safe"] + result["auto_vulnerable"]
    result["completed"] = result["automatic"] + result["reviewed"]
    result["feasible_complete_roster"] = int(result["unresolved"] == 0)
    require(result["completed"] + result["unresolved"] == result["n"], "Full roster disappeared from action denominator")
    return result


def outcome_metrics(labels: np.ndarray, codes: np.ndarray, scenario: CostScenario, *, arrival_eta: np.ndarray | None = None) -> dict:
    """Post-decision-seal outcome join; unresolved cost is never credited as safety."""
    labels, codes = np.asarray(labels), np.asarray(codes)
    require(labels.shape == codes.shape and np.isin(labels, [0, 1]).all() and len(labels) > 0, "Outcomes must align with complete roster")
    result = action_counts(codes)
    nv, ns = int(np.sum(labels == 1)), int(np.sum(labels == 0))
    fn = int(np.sum((labels == 1) & (codes == 0)))
    fp = int(np.sum((labels == 0) & (codes == 1)))
    rv = int(np.sum((labels == 1) & (codes == REVIEW)))
    rs = int(np.sum((labels == 0) & (codes == REVIEW)))
    uv = int(np.sum((labels == 1) & (codes == BACKLOG)))
    us = int(np.sum((labels == 0) & (codes == BACKLOG)))
    cost = (scenario.miss * fn + scenario.false_alert * fp + scenario.review * result["reviewed"]
            + scenario.miss * scenario.review_fn_probability * rv + scenario.false_alert * scenario.review_fp_probability * rs) / len(labels)
    result.update(n_vulnerable=nv, n_safe=ns, auto_fn=fn, auto_fp=fp, reviewed_vulnerable=rv, reviewed_safe=rs,
                  unresolved_vulnerable=uv, unresolved_safe=us, automatic_miss_rate=fn / nv if nv else np.nan,
                  automatic_false_alert_rate=fp / ns if ns else np.nan,
                  unresolved_vulnerable_rate=uv / nv if nv else np.nan,
                  selective_automatic_error=(fn + fp) / result["automatic"] if result["automatic"] else np.nan,
                  review_fraction=result["reviewed"] / len(labels), unresolved_fraction=result["unresolved"] / len(labels),
                  automatic_fraction=result["automatic"] / len(labels), completed_fraction=result["completed"] / len(labels),
                  cost_omitting_unresolved_lower_bound=cost,
                  cost_complete_per_function=cost if result["unresolved"] == 0 else np.nan,
                  cost_deadline_assumed_per_function=cost + scenario.miss * uv / len(labels),
                  expected_review_fn=scenario.review_fn_probability * rv,
                  expected_review_fp=scenario.review_fp_probability * rs)
    require(rv + rs == result["reviewed"] and uv + us == result["unresolved"] and nv + ns == len(labels), "Outcome denominators changed")
    if arrival_eta is not None:
        arrival_eta = probability_vector(arrival_eta)
        require(arrival_eta.shape == labels.shape, "Arrival calibration probabilities must cover full roster")
        result["arrival_probability_brier"] = float(np.mean(np.square(arrival_eta - labels)))
    return result


@dataclass(frozen=True)
class Batch:
    index: int
    indices: tuple[int, ...]
    commits: tuple[str, ...]
    minimum_date: int | None
    maximum_date: int | None


def make_batches(rows: Sequence[dict], *, ordering: str, nominal_rows: int, salt: str) -> tuple[list[Batch], dict]:
    """Label-free order; a whole commit/date-tie block stays in one batch."""
    require(nominal_rows > 0 and len(rows) > 0, "Batch construction requires rows and a positive nominal size")
    commits: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        require(not any(k.casefold() in TARGET_LABEL_KEYS for k in row), "Batch ordering may not receive target labels")
        commits.setdefault(canonical_commit_id(row["commit_id"]), []).append(i)
    date_map: dict[str, int] = {}
    if ordering == "recorded_commit_author_date":
        for commit, indices in commits.items():
            dates = {rows[i].get("commit_date") for i in indices}
            require(len(dates) == 1 and None not in dates, "Missing or inconsistent commit dates; cannot assert real chronology")
            date = next(iter(dates))
            require(isinstance(date, int) and not isinstance(date, bool), "Commit date must be an integer epoch")
            date_map[commit] = date
        tied: dict[int, list[str]] = {}
        for commit, date in date_map.items():
            tied.setdefault(date, []).append(commit)
        blocks = [(sorted(cs), date) for date, cs in sorted(tied.items())]
    elif ordering == "synthetic_salted_commit_hash":
        blocks = [([commit], None) for commit in sorted(commits, key=lambda c: (hashlib.sha256(f"{salt}|{c}".encode()).hexdigest(), c))]
    else:
        raise ValueError("Unknown temporal ordering rule")
    batches: list[Batch] = []
    pending_indices: list[int] = []
    pending_commits: list[str] = []
    pending_dates: list[int] = []

    def finish() -> None:
        if pending_indices:
            batches.append(Batch(len(batches), tuple(pending_indices), tuple(pending_commits),
                                 min(pending_dates) if pending_dates else None, max(pending_dates) if pending_dates else None))
            pending_indices.clear()
            pending_commits.clear()
            pending_dates.clear()

    for cs, date in blocks:
        indices = sorted(i for c in cs for i in commits[c])
        if pending_indices and len(pending_indices) + len(indices) > nominal_rows:
            finish()
        pending_indices.extend(indices)
        pending_commits.extend(cs)
        if date is not None:
            pending_dates.append(date)
        if len(pending_indices) >= nominal_rows:
            finish()
    finish()
    membership = np.full(len(rows), -1, dtype=int)
    commit_batch: dict[str, int] = {}
    date_batch: dict[int, int] = {}
    for batch in batches:
        for i in batch.indices:
            require(membership[i] == -1, "Duplicate function in temporal batches")
            membership[i] = batch.index
        for commit in batch.commits:
            require(commit not in commit_batch, "A commit was split across batches")
            commit_batch[commit] = batch.index
            if ordering == "recorded_commit_author_date":
                date = date_map[commit]
                require(date not in date_batch or date_batch[date] == batch.index, "Equal commit dates were split across batches")
                date_batch[date] = batch.index
    require(np.all(membership >= 0) and len(commit_batch) == len(commits), "Temporal batching lost part of the roster")
    if ordering == "recorded_commit_author_date":
        require(all(a.maximum_date < b.minimum_date for a, b in zip(batches, batches[1:])), "Chronological batches overlap or reverse")
    return batches, {"ordering": ordering, "rows": len(rows), "commits": len(commits), "atomic_blocks": len(blocks),
                     "batches": len(batches), "nominal_batch_rows": nominal_rows,
                     "largest_batch": max(len(b.indices) for b in batches),
                     "oversized_batches": sum(len(b.indices) > nominal_rows for b in batches),
                     "all_rows_retained": True, "whole_commits": True, "date_ties_same_batch": True,
                     "real_decision_or_feedback_arrival_observed": False, "source_before_target_established": False}


@dataclass(frozen=True)
class ReviewLabel:
    index: int
    label: int
    service_batch: int
    arrival_batch: int


class DelayedReviewOracle:
    """A released benchmark label can be read only after priced past review.

    schedule() never reads a label. release() enforces the fixed delay and
    past-service boundary. Unreviewed items have no release event.
    """
    def __init__(self, benchmark_labels: np.ndarray, delay: int):
        labels = np.asarray(benchmark_labels)
        require(labels.ndim == 1 and np.isin(labels, [0, 1]).all(), "Invalid benchmark-label simulation vault")
        self._labels = labels.copy()
        self.delay = capacity_integer(delay)
        require(self.delay >= 1, "Current-batch feedback is forbidden; delay must be at least one")
        self._pending: dict[int, tuple[int, int]] = {}
        self._reviewed: set[int] = set()
        self._last_release = -1
        self.label_accesses = 0

    def release(self, batch: int) -> list[ReviewLabel]:
        batch = capacity_integer(batch)
        require(batch > self._last_release, "Feedback batch clock must strictly advance")
        self._last_release = batch
        ready = sorted((arrival, service, index) for index, (service, arrival) in self._pending.items() if arrival <= batch)
        events = []
        for arrival, service, index in ready:
            require(index in self._reviewed and service < batch and arrival == service + self.delay, "Future/current or unreviewed feedback leakage")
            events.append(ReviewLabel(index, int(self._labels[index]), service, arrival))
            del self._pending[index]
            self.label_accesses += 1
        return events

    def schedule(self, indices: Iterable[int], *, batch: int, cap: int) -> None:
        indices = list(indices)
        batch, cap = capacity_integer(batch), capacity_integer(cap)
        require(batch == self._last_release, "Reviews must occur at the current released batch clock")
        require(len(indices) <= cap and len(indices) == len(set(indices)), "Review cap overflow or duplicate same-batch review")
        for index in indices:
            require(isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_)) and 0 <= index < len(self._labels), "Review index outside full roster")
            require(int(index) not in self._reviewed, "A function cannot receive repeated free review")
        for index in indices:
            index = int(index)
            self._reviewed.add(index)
            self._pending[index] = batch, batch + self.delay

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def reviewed_count(self) -> int:
        return len(self._reviewed)


class RollingIsotonic:
    """Source initializer plus fixed-window, delayed reviewed-label updates."""
    def __init__(self, source_probability: np.ndarray, source_labels: np.ndarray, knots_x: np.ndarray, knots_y: np.ndarray,
                 config: dict):
        self.source_probability = probability_vector(source_probability)
        self.source_labels = np.asarray(source_labels)
        require(len(self.source_probability) > 0 and self.source_labels.shape == self.source_probability.shape and np.isin(self.source_labels, [0, 1]).all(), "Source calibration labels must align")
        self.x, self.y = np.asarray(knots_x, dtype=float), np.asarray(knots_y, dtype=float)
        require(self.x.ndim == 1 and len(self.x) == len(self.y) and len(self.x) > 0 and np.all(np.diff(self.x) >= 0), "Invalid sealed source isotonic knots")
        self.config = config
        self.history: list[ReviewLabel] = []
        self.fits = 0
        self.used_window: list[ReviewLabel] = []
        self.seen: set[int] = set()

    def predict(self, probability: np.ndarray) -> np.ndarray:
        return np.interp(probability_vector(probability), self.x, self.y)

    def observe(self, events: Sequence[ReviewLabel], *, batch: int, target_probability: np.ndarray) -> dict:
        batch = capacity_integer(batch)
        target_probability = probability_vector(target_probability)
        for event in events:
            require(isinstance(event, ReviewLabel) and 0 <= event.index < len(target_probability) and event.label in (0, 1), "Invalid reviewed feedback event")
            require(0 <= event.service_batch < batch and event.arrival_batch <= batch and event.arrival_batch > event.service_batch, "Rolling fit received future/current-batch labels")
            require(event.index not in self.seen, "Repeated feedback label in calibrator")
        for event in events:
            self.seen.add(event.index)
            self.history.append(event)
        window = self.history[-int(self.config["maximum_arrived_window"]):]
        labels = np.asarray([e.label for e in window], dtype=int)
        counts = {label: int(np.sum(labels == label)) for label in (0, 1)}
        eligible = len(window) >= self.config["minimum_arrived_rows"] and min(counts.values()) >= self.config["minimum_arrived_per_class"]
        status = "no_new_feedback" if not events else "insufficient_reviewed_feedback"
        if events and eligible:
            require(all(e.service_batch < batch and e.arrival_batch <= batch for e in window), "Fit window crossed future-label boundary")
            source_weight = float(self.config["source_total_anchor_weight"]) / len(self.source_probability)
            weights = np.concatenate([np.full(len(self.source_probability), source_weight), np.full(len(window), float(self.config["target_per_row_weight"]))])
            probabilities = np.concatenate([self.source_probability, target_probability[[e.index for e in window]]])
            outcomes = np.concatenate([self.source_labels, labels])
            model = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(probabilities, outcomes, sample_weight=weights)
            self.x, self.y = model.X_thresholds_, model.y_thresholds_
            self.fits += 1
            self.used_window = list(window)
            status = "updated_on_arrived_reviewed_labels"
        used = self.used_window
        return {"calibration_status": status, "rolling_fit_count": self.fits, "arrived_total": len(self.history),
                "available_window_n": len(window), "available_window_safe": counts[0], "available_window_vulnerable": counts[1],
                "used_target_n": len(used), "maximum_used_service_batch": max((e.service_batch for e in used), default=-1),
                "maximum_used_arrival_batch": max((e.arrival_batch for e in used), default=-1),
                "used_window_hash": json_hash([e.__dict__ for e in used]), "knots_sha256": json_hash([self.x, self.y]),
                "source_calibration_n": len(self.source_probability), "no_future_label_assertion": True,
                "fit_window_indices": [e.index for e in used] if status == "updated_on_arrived_reviewed_labels" else []}


def simulate_feedback(probability: np.ndarray, trace_codes: np.ndarray, batches: Sequence[Batch], source_probability: np.ndarray,
                      source_labels: np.ndarray, knots_x: np.ndarray, knots_y: np.ndarray, benchmark_labels: np.ndarray,
                      *, method: str, cap: int, delay: int, scenario: CostScenario, rolling_config: dict,
                      stable_positions: np.ndarray) -> dict:
    """Sequential decisions use no labels except the oracle's arrived review events."""
    probability, cap = probability_vector(probability), capacity_integer(cap)
    n = len(probability)
    trace_codes, positions = np.asarray(trace_codes), np.asarray(stable_positions)
    require(trace_codes.shape == (n,) and np.isin(trace_codes, [0, 1, REVIEW]).all() and positions.shape == (n,), "Invalid sealed temporal input")
    require(method in ("source_iso_static", "source_iso_rolling", "trace_static", "trace_rolling", "msp_static", "forced_argmax_static"), "Unknown feedback policy")
    oracle = DelayedReviewOracle(benchmark_labels, delay)
    calibration = RollingIsotonic(source_probability, source_labels, knots_x, knots_y, rolling_config)
    codes = np.full(n, BACKLOG, dtype=np.uint8)
    arrival_eta = np.full(n, np.nan)
    service_batch = np.full(n, -1, dtype=np.int32)
    prediction_batch = np.full(n, -1, dtype=np.int32)
    pending: set[int] = set()
    ledgers, feedback_events, fit_ledgers = [], [], []
    rolling = method.endswith("rolling")
    for batch in batches:
        require(batch.index == len(ledgers), "Batch indices must be consecutive")
        current = np.asarray(batch.indices, dtype=int)
        events = oracle.release(batch.index)
        feedback_events.extend({"index": e.index, "label": e.label, "service_batch": e.service_batch, "arrival_batch": e.arrival_batch,
                                "reveal_batch": batch.index, "used_for_update_method": int(rolling)} for e in events)
        if rolling:
            receipt = calibration.observe(events, batch=batch.index, target_probability=probability)
        else:
            receipt = {"calibration_status": "static_source_map", "rolling_fit_count": 0, "arrived_total": oracle.label_accesses,
                       "available_window_n": 0, "available_window_safe": 0, "available_window_vulnerable": 0,
                       "used_target_n": 0, "maximum_used_service_batch": -1, "maximum_used_arrival_batch": -1,
                       "used_window_hash": json_hash([]), "knots_sha256": json_hash([calibration.x, calibration.y]),
                       "source_calibration_n": len(source_probability), "no_future_label_assertion": True, "fit_window_indices": []}
        require(receipt["maximum_used_service_batch"] < batch.index and receipt["maximum_used_arrival_batch"] <= batch.index,
                "Temporal receipt includes a future/current label")
        require(np.all(prediction_batch[current] == -1), "A function entered multiple prediction batches")
        prediction_batch[current] = batch.index
        arrival_eta[current] = calibration.predict(probability[current])
        backlog_before = len(pending)
        if method.startswith("trace_"):
            codes[current] = np.where(trace_codes[current] == REVIEW, BACKLOG, trace_codes[current])
            pending.update(int(i) for i in current if trace_codes[i] == REVIEW)
            queue = np.asarray(sorted(pending), dtype=int)
            queue_codes, _ = allocate_review(calibration.predict(probability[queue]), cap, scenario,
                                            mandatory_codes=np.full(len(queue), REVIEW), stable_positions=positions[queue])
            reviewed = queue[queue_codes == REVIEW]
            codes[reviewed] = REVIEW
            pending.difference_update(int(i) for i in reviewed)
            required_current = int(np.sum(trace_codes[current] == REVIEW))
        else:
            if method.startswith("source_iso_"):
                batch_codes, required_current = allocate_review(arrival_eta[current], cap, scenario, stable_positions=positions[current])
            elif method == "msp_static":
                batch_codes = msp_review(probability[current], cap, stable_positions=positions[current])
                required_current = min(cap, len(current))
            else:
                batch_codes = (probability[current] > 0.5).astype(np.uint8)
                required_current = 0
            codes[current] = batch_codes
            reviewed = current[batch_codes == REVIEW]
        require(len(reviewed) <= cap and np.all(service_batch[reviewed] == -1), "Temporal review cap exceeded or review repeated")
        service_batch[reviewed] = batch.index
        oracle.schedule(reviewed.tolist(), batch=batch.index, cap=cap)
        fit_window = receipt.pop("fit_window_indices")
        if fit_window:
            fit_ledgers.append({"batch": batch.index, "target_indices": fit_window, **receipt})
        ledgers.append({"batch": batch.index, "batch_rows": len(current), "batch_commits": len(batch.commits),
                        "minimum_date": batch.minimum_date, "maximum_date": batch.maximum_date, "slots": cap,
                        "required_current_review": required_current, "backlog_before": backlog_before,
                        "reviewed_in_batch": len(reviewed), "backlog_after": len(pending),
                        "current_auto_safe": int(np.sum(codes[current] == 0)), "current_auto_vulnerable": int(np.sum(codes[current] == 1)),
                        "reviewed_cumulative": oracle.reviewed_count, "arrived_labels_cumulative": oracle.label_accesses,
                        "new_arrived_labels": len(events), "pending_reviewed_labels": oracle.pending_count,
                        "cap_assertion": True, **receipt})
    require(np.all(prediction_batch >= 0) and np.isfinite(arrival_eta).all(), "Feedback simulation lost incoming functions")
    require(np.array_equal(service_batch >= 0, codes == REVIEW), "Served review and final actions disagree")
    require(int(np.sum(codes == BACKLOG)) == len(pending), "Unresolved mandatory review disappeared")
    require(all(e["service_batch"] < e["reveal_batch"] and e["arrival_batch"] <= e["reveal_batch"] for e in feedback_events), "Future-label event leaked")
    return {"codes": codes, "arrival_eta": arrival_eta, "service_batch": service_batch, "prediction_batch": prediction_batch,
            "batch_ledger": ledgers, "feedback_events": feedback_events, "fit_ledger": fit_ledgers,
            "summary": {"batches": len(batches), "review_slots_total": cap * len(batches), "rolling_updates": calibration.fits,
                        "arrived_reviewed_labels": oracle.label_accesses, "reviewed_labels_pending_at_end": oracle.pending_count,
                        "no_future_label_assertions": len(batches), "cap_assertions": len(batches),
                        "feedback_from_unreviewed": 0, "free_postrun_drain": 0}}


def source_calibration_view(source_rows: Sequence[dict], source_positions: np.ndarray, source_probability: np.ndarray,
                            calibration_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions, cal = np.asarray(source_positions), np.asarray(calibration_positions)
    probability = probability_vector(source_probability)
    require(positions.shape == probability.shape and len(set(positions.tolist())) == len(positions), "Source prediction identities must be unique")
    expected = np.asarray([i for i, r in enumerate(source_rows) if r["role"] == "calibration"], dtype=int)
    require(np.array_equal(np.sort(cal), expected) and len(cal) == len(set(cal.tolist())), "Calibration positions were replaced by train/validation/target roles")
    require(set(cal.tolist()).issubset(set(positions.tolist())), "Calibration predictions are missing")
    lookup = {int(pos): float(p) for pos, p in zip(positions, probability)}
    labels = np.asarray([source_rows[int(i)]["label"] for i in cal], dtype=int)
    require(np.isin(labels, [0, 1]).all(), "Invalid source calibration labels")
    return np.asarray([lookup[int(i)] for i in cal]), labels


def read_rows(path: Path, *, source: bool) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            require(isinstance(row, dict) and row.get("position") == len(records), "Row metadata must use consecutive sealed positions")
            require(source or not any(k.casefold() in TARGET_LABEL_KEYS for k in row), "Target metadata contains labels")
            records.append({k: v for k, v in row.items() if k != "text"})
    require(len({r["row_id"] for r in records}) == len(records), "Metadata row IDs are not unique")
    return records


def scenario_grid(config: dict) -> list[CostScenario]:
    cost = config["cost"]
    scenarios = [CostScenario(miss, cost["false_alert"], review) for miss in cost["miss"] for review in cost["review_costs"]]
    sensitivity = cost["review_error_sensitivity"]
    scenarios.extend(CostScenario(miss, cost["false_alert"], sensitivity["review_cost"], sensitivity["false_negative_probability"], sensitivity["false_positive_probability"])
                     for miss in cost["miss"])
    return scenarios


def frozen_inputs(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = ROOT / config["protocol_path"]
    clearance_path = ROOT / "tmp/neurocomputing-strengthening/kbs-v4/capacity_feedback_protocol_clearance.json"
    clearance = json.loads(clearance_path.read_text(encoding="utf-8"))
    require(clearance["status"] == "CLEARED", "Root protocol clearance is missing")
    for relative, digest in clearance["inputs_sha256"].items():
        require(sha256(ROOT / relative) == digest, "Config/protocol differs from the pre-outcome clearance")
    require(config["known_outcomes"] and config["statistics"]["no_outcome_based_exclusion"] is True, "Retrospective amendment or full-roster rule missing")
    return {"status": "FROZEN_BEFORE_NEW_COMPARISONS", "created_at_utc": utc_now(),
            "known_outcomes": config["known_outcomes"], "protocol_version": config["protocol_version"],
            "files": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in
                      (config_path, protocol, Path(__file__), TESTS, clearance_path)]}


def freeze(config_path: Path) -> dict:
    receipt = frozen_inputs(config_path)
    write_json(FREEZE, receipt)
    return receipt


def verify_freeze(config_path: Path) -> dict:
    receipt = json.loads(FREEZE.read_text(encoding="utf-8"))
    require(receipt["status"] == "FROZEN_BEFORE_NEW_COMPARISONS", "Analysis was not frozen")
    for record in receipt["files"]:
        require(sha256(Path(record["path"])) == record["sha256"], f"Frozen analysis input changed: {record['path']}")
    require(str(config_path.resolve()) in {r["path"] for r in receipt["files"]}, "Run config not in frozen plan")
    return receipt


def csv_output(path: Path, records: Sequence[dict]) -> None:
    pd.DataFrame(records).to_csv(path, index=False, float_format="%.12g")


def seed_average(frame: pd.DataFrame, keys: list[str], actual_seeds: dict[str, dict[str, list[int]]]) -> pd.DataFrame:
    require(not frame.duplicated(keys + ["seed"]).any(), "Duplicate project/seed results")
    excluded = set(keys + ["seed", "capacity_key", "feedback_key"])
    numeric = [c for c in frame.select_dtypes(include=np.number).columns if c not in excluded]
    rows = []
    for values, group in frame.groupby(keys, sort=True, dropna=False):
        identity = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        expected = actual_seeds[identity["dataset"]][identity["detector"]]
        require(sorted(group.seed.astype(int).tolist()) == expected, "Missing actual fitted seed in project aggregate")
        row = {**identity, "actual_seed_count": len(expected), "actual_seeds": ",".join(map(str, expected)),
               "domain_statuses": ",".join(sorted(set(group.domain_status))),
               "domain_fallback_seed_count": int(np.sum(group.domain_status != "ok"))}
        for column in numeric:
            row[column] = float(group[column].mean())
        row["feasible_seed_fraction"] = float(group.feasible_complete_roster.mean())
        row["feasible_complete_roster"] = int(group.feasible_complete_roster.eq(1).all())
        if group.cost_complete_per_function.isna().any():
            row["cost_complete_per_function"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_outputs(frame: pd.DataFrame, *, settings: list[str], comparisons: list[tuple[str, str]], endpoints: list[str],
                   config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired, summaries = [], []
    bootstrap_n, bootstrap_seed = config["statistics"]["bootstrap_draws"], config["statistics"]["bootstrap_seed"]
    for values, group in frame.groupby(settings, sort=True, dropna=False):
        identity = dict(zip(settings, values if isinstance(values, tuple) else (values,)))
        all_projects = set(group.project)
        for candidate, baseline in comparisons:
            left, right = group[group.method == candidate].set_index("project"), group[group.method == baseline].set_index("project")
            require(set(left.index) == set(right.index) == all_projects, "Paired comparison silently lost a project")
            for project in sorted(all_projects):
                a, b = left.loc[project], right.loc[project]
                require(a["n"] == b["n"] and a["n_vulnerable"] == b["n_vulnerable"] and a["n_safe"] == b["n_safe"], "Paired class denominators changed")
                row = {**identity, "candidate": candidate, "baseline": baseline, "project": project,
                       "n": a["n"], "n_vulnerable": a["n_vulnerable"], "n_safe": a["n_safe"],
                       "candidate_feasible": a["feasible_complete_roster"], "baseline_feasible": b["feasible_complete_roster"]}
                for endpoint in endpoints:
                    row[f"candidate_{endpoint}"] = a[endpoint]
                    row[f"baseline_{endpoint}"] = b[endpoint]
                    row[f"delta_{endpoint}"] = a[endpoint] - b[endpoint]
                paired.append(row)
            for endpoint in endpoints:
                delta = (left[endpoint].sort_index() - right[endpoint].sort_index()).to_numpy(dtype=float)
                finite = np.isfinite(delta)
                row = {**identity, "candidate": candidate, "baseline": baseline, "endpoint": endpoint,
                       "projects_roster": len(delta), "projects_defined": int(finite.sum()), "projects_undefined": int((~finite).sum()),
                       "status": "full_roster_descriptive" if finite.all() else "incomplete_endpoint_not_full_roster",
                       "median_paired_difference": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                       "wins": 0, "losses": 0, "ties": 0, "bootstrap_draws": bootstrap_n}
                if finite.all():
                    seed = (bootstrap_seed + int(json_hash([identity, candidate, baseline, endpoint])[:8], 16)) % (2**32)
                    rng = np.random.default_rng(seed)
                    values_boot = np.median(delta[rng.integers(0, len(delta), size=(bootstrap_n, len(delta)))], axis=1)
                    lo, hi = np.quantile(values_boot, [0.025, 0.975])
                    row.update(median_paired_difference=float(np.median(delta)), ci_lower=float(lo), ci_upper=float(hi),
                               wins=int(np.sum(delta < -1e-12)), losses=int(np.sum(delta > 1e-12)), ties=int(np.sum(np.abs(delta) <= 1e-12)))
                summaries.append(row)
    return pd.DataFrame(paired), pd.DataFrame(summaries)


def curve_summary(frame: pd.DataFrame, settings: list[str], endpoints: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(settings + ["method"], sort=True, dropna=False):
        identity = dict(zip(settings + ["method"], values))
        row = {**identity, "projects_roster": len(group), "projects_feasible": int(group.feasible_complete_roster.sum()),
               "all_projects_feasible": int(group.feasible_complete_roster.eq(1).all())}
        for endpoint in endpoints:
            if endpoint == "cost_complete_per_function" and not group[endpoint].notna().all():
                row[f"median_{endpoint}"] = np.nan
            else:
                row[f"median_{endpoint}"] = float(group[endpoint].median())
            row[f"mean_{endpoint}"] = float(group[endpoint].mean()) if group[endpoint].notna().all() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _write_stream(writer: csv.DictWriter, common: dict, rows: Sequence[dict]) -> None:
    for row in rows:
        writer.writerow({**common, **{k: json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v for k, v in row.items()}})


def run(config_path: Path, output: Path) -> dict:
    started = utc_now()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    receipt = verify_freeze(config_path)
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), "Analysis output must be a new/empty directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "frozen_protocol_receipt.json", receipt)
    lineage_path = ROOT / config["lineage_path"]
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    require(lineage["status"] == "PASS", "Existing prediction lineage check did not pass")
    source_path = ROOT / config["source_rows_path"]
    source = read_rows(source_path, source=True)
    require(sum(r["role"] == "calibration" for r in source) == 4400, "Inherited exact calibration role changed")
    datasets, dependencies, actual_seeds, roster = {}, {}, {}, []
    for dataset, spec in config["datasets"].items():
        seal_path = ROOT / spec["decision_seal_path"]
        seal = verify_seal(seal_path)
        require(seal["dataset"] == dataset and len(seal["projects"]) == spec["expected_projects"] and seal["all_planned_fits_present"], "Inherited project/seed roster changed")
        target_path = Path(seal["target_rows_path"])
        target = read_rows(target_path, source=False)
        require(len(target) == spec["expected_rows"] and set(r["project_group"] for r in target) == set(seal["projects"]), "Prepared full target roster changed")
        input_dir = ROOT / spec["input_dir"]
        preparation_path = input_dir / "preparation_manifest.json"
        preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
        require(preparation["selected_projects"] == seal["projects"], "Decision seal and preparation selection disagree")
        for dep in seal["dependencies"]:
            dependencies[dep["path"]] = dep
        for path, role in [(seal_path, "original_decision_seal"), (preparation_path, "original_preparation_manifest"),
                           (input_dir / "candidate_projects.csv", "original_62_candidate_eligibility"), (lineage_path, "original_prediction_lineage")]:
            dependencies[str(path.resolve())] = {"path": str(path.resolve()), "sha256": sha256(path), "role": role}
        datasets[dataset] = {"seal": seal, "seal_path": seal_path, "target": target, "input_dir": input_dir, "spec": spec}
        actual_seeds[dataset] = seal["actual_seeds"]
        candidates = pd.read_csv(input_dir / "candidate_projects.csv").to_dict("records")
        for candidate in candidates:
            project = candidate.get("project_group", candidate.get("project"))
            roster.append({"dataset": dataset, "project": project, "selected_in_existing_v3": int(project in seal["projects"]),
                           **{f"original_{k}": v for k, v in candidate.items()}})
        print(f"[{utc_now()}] verified {dataset}: {len(target)} rows/{len(seal['projects'])} projects; full roster retained", flush=True)
    for record in receipt["files"]:
        dependencies[record["path"]] = {**record, "role": "new_frozen_analysis_implementation_or_protocol"}
    write_json(output / "input_manifest.json", {"created_at_utc": utc_now(), "protocol_version": config["protocol_version"],
        "known_outcome_amendment": True, "target_labels_opened_for_new_analysis": False, "actual_seeds": actual_seeds,
        "dependencies": sorted(dependencies.values(), key=lambda r: r["path"]),
        "source_calibration_rows": 4400, "source_role_counts": pd.Series([r["role"] for r in source]).value_counts().to_dict(),
        "target_label_vault_expected_hashes": {ds: data["seal"]["prepared_target_label_vault_sha256"] for ds, data in datasets.items()},
        "datasets": {ds: {"projects": data["seal"]["projects"], "target_rows": len(data["target"]),
                           "domain_unsupported_projects": data["seal"]["domain_unsupported_projects"], "ordering": data["spec"]["ordering"]}
                     for ds, data in datasets.items()}})
    csv_output(output / "candidate_roster_lineage.csv", roster)
    # No target-label vault is loaded above or during this capacity/order seal.
    scenarios = scenario_grid(config)
    alpha = config["risk_budget"]
    memberships_map = {"trace_mandatory": "estimated_weight_no_gate", "allocation_matched_tails_mandatory": "class_matched_weighted",
                       "weighted_volume_msp_mandatory": "msp_matched_weighted"}
    cells, capacity_plan, capacity_archives, batch_receipts = [], [], [], []
    prediction_cache: dict[tuple[str, str, int], tuple[dict, dict]] = {}
    detector_dirs = {"finetuned_codebert": "finetuned", "frozen_codebert": "frozen", "hashing_sgd": "hashing"}
    total_cells = sum(len(data["seal"]["decisions"]) for data in datasets.values())
    for dataset, data in datasets.items():
        seal, root, target = data["seal"], data["seal_path"].parent, data["target"]
        for entry in seal["decisions"]:
            detector, seed, project = entry["detector"], entry["seed"], entry["project"]
            identity = {"dataset": dataset, "detector": detector, "seed": seed, "project": project}
            key = dataset, detector, seed
            if key not in prediction_cache:
                metadata_path = ROOT / f"outputs/neural-validation-v3/predictions/{dataset}/{detector_dirs[detector]}/seed-{seed}/metadata.json"
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                require(meta["detector"] == detector and meta["seed"] == seed and meta["dataset"] == dataset and meta["target_labels_accessed"] is False, "Prediction metadata identity/boundary mismatch")
                with np.load(ROOT / meta["prediction_path"], allow_pickle=False) as pred:
                    arrays = {name: pred[name] for name in ("source_positions", "source_probability", "target_positions", "target_probability")}
                require(np.array_equal(arrays["target_positions"], np.arange(len(target))), "Target prediction position order changed")
                prediction_cache[key] = meta, arrays
            _, pred = prediction_cache[key]
            with np.load(root / entry["archive"], allow_pickle=False) as arc:
                positions = arc["target_positions"]
                p, eta = arc["target_probability"], arc["source_eta"]
                calp, caly = source_calibration_view(source, pred["source_positions"], pred["source_probability"], arc["calibration_positions"])
                knots_x, knots_y = arc["isotonic_x"], arc["isotonic_y"]
                memberships = {name: action_codes(arc[cell_key(alpha["alpha_v"], alpha["alpha_s"], alpha["weight_clip"], original)])
                               for name, original in memberships_map.items()}
            require(np.array_equal(p, pred["target_probability"][positions]), "Project probability differs from sealed detector predictions")
            require(np.allclose(eta, np.interp(p, knots_x, knots_y), rtol=0, atol=1e-12), "Source-calibrated target scores differ from sealed initializer")
            initial_check = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(calp, caly)
            require(np.array_equal(initial_check.X_thresholds_, knots_x) and np.array_equal(initial_check.y_thresholds_, knots_y), "Initializer was not fitted on the exact source calibration role")
            meta = json.loads((root / entry["metadata"]).read_text(encoding="utf-8"))
            rows = [target[int(i)] for i in positions]
            require(all(r["project_group"] == project for r in rows), "Project target positions contain another project")
            batches, batch_info = make_batches(rows, ordering=data["spec"]["ordering"], nominal_rows=config["feedback"]["nominal_batch_rows"], salt=config["feedback"]["synthetic_order_salt"])
            archive_dir = output / "capacity_decisions" / dataset / detector / f"seed-{seed}"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / f"{project}.npz"
            arrays = {"target_positions": positions}
            for scenario in scenarios:
                for fraction in config["capacity"]["fractions"]:
                    cap = math.floor(fraction * len(p))
                    policies = capacity_policies(p, eta, memberships, cap, scenario, positions)
                    require(set(policies) == set(config["capacity"]["methods"]), "Capacity method grid changed")
                    for method, (codes, required_count) in policies.items():
                        capacity_key = f"{scenario.key}__r{number_key(fraction)}__{method}"
                        arrays[capacity_key] = codes
                        capacity_plan.append({**identity, **scenario.fields(), "capacity_fraction": fraction, "capacity": cap,
                            "method": method, "domain_status": meta["domain_status"], "required_review": required_count,
                            "capacity_key": capacity_key, "archive": str(archive_path.relative_to(output)), **action_counts(codes)})
            np.savez_compressed(archive_path, **arrays)
            capacity_archives.append({**identity, "path": str(archive_path.relative_to(output)), "sha256": sha256(archive_path)})
            order_path = output / "ordered_batches" / dataset / detector / f"seed-{seed}" / f"{project}.json"
            write_json(order_path, {**identity, **batch_info, "target_labels_accessed": False,
                "batches": [{"batch": b.index, "target_positions": positions[list(b.indices)].tolist(), "commit_ids": b.commits,
                             "minimum_date": b.minimum_date, "maximum_date": b.maximum_date} for b in batches]})
            batch_receipts.append({**identity, **batch_info, "path": str(order_path.relative_to(output)), "sha256": sha256(order_path)})
            cells.append({**identity, "positions": positions, "probability": p, "eta": eta, "source_probability": calp, "source_labels": caly,
                          "knots_x": knots_x, "knots_y": knots_y, "trace_codes": memberships["trace_mandatory"],
                          "batches": batches, "batch_info": batch_info, "domain_status": meta["domain_status"], "rows": rows})
            if len(cells) % 40 == 0 or len(cells) == total_cells:
                print(f"[{utc_now()}] label-free capacity/order cells sealed in progress: {len(cells)}/{total_cells}", flush=True)
    csv_output(output / "capacity_decision_counts.csv", capacity_plan)
    csv_output(output / "ordered_batch_summary.csv", [{k: v for k, v in r.items() if k not in ("path", "sha256")} for r in batch_receipts])
    write_json(output / "capacity_decision_seal.json", {"created_at_utc": utc_now(), "protocol_version": config["protocol_version"],
        "known_outcome_amendment": True, "target_label_vault_opened": False, "input_manifest_sha256": sha256(output / "input_manifest.json"),
        "capacity_decision_counts_sha256": sha256(output / "capacity_decision_counts.csv"), "decision_archives": capacity_archives,
        "ordered_batch_archives": batch_receipts, "full_rosters_retained": True, "overflow_is_backlog": True,
        "project_seed_cells": len(cells), "capacity_policy_cells": len(capacity_plan)})
    print(f"[{utc_now()}] capacity and ordered batches sealed before opening target-label vaults", flush=True)
    labels = {dataset: _load_target_vault(data["input_dir"] / "target_label_vault.npz", data["seal"], data["target"]) for dataset, data in datasets.items()}
    # Sequential adaptation may now consume only reviewed labels through oracle.
    ledger_common = ["dataset", "detector", "seed", "project", "method", "slots_per_batch", "delay_batches", "scenario", "feedback_key"]
    batch_fields = ledger_common + ["batch", "batch_rows", "batch_commits", "minimum_date", "maximum_date", "slots", "required_current_review", "backlog_before", "reviewed_in_batch", "backlog_after", "current_auto_safe", "current_auto_vulnerable", "reviewed_cumulative", "arrived_labels_cumulative", "new_arrived_labels", "pending_reviewed_labels", "cap_assertion", "calibration_status", "rolling_fit_count", "arrived_total", "available_window_n", "available_window_safe", "available_window_vulnerable", "used_target_n", "maximum_used_service_batch", "maximum_used_arrival_batch", "used_window_hash", "knots_sha256", "source_calibration_n", "no_future_label_assertion"]
    event_fields = ledger_common + ["index", "target_position", "row_id", "label", "service_batch", "arrival_batch", "reveal_batch", "used_for_update_method"]
    fit_fields = ledger_common + ["batch", "target_indices", "calibration_status", "rolling_fit_count", "arrived_total", "available_window_n", "available_window_safe", "available_window_vulnerable", "used_target_n", "maximum_used_service_batch", "maximum_used_arrival_batch", "used_window_hash", "knots_sha256", "source_calibration_n", "no_future_label_assertion"]
    feedback_plan, feedback_archives = [], []
    with gzip.open(output / "feedback_batch_ledger.csv.gz", "wt", encoding="utf-8", newline="") as batch_file, \
         gzip.open(output / "feedback_label_events.csv.gz", "wt", encoding="utf-8", newline="") as event_file, \
         gzip.open(output / "feedback_fit_ledger.csv.gz", "wt", encoding="utf-8", newline="") as fit_file:
        batch_writer, event_writer, fit_writer = csv.DictWriter(batch_file, fieldnames=batch_fields), csv.DictWriter(event_file, fieldnames=event_fields), csv.DictWriter(fit_file, fieldnames=fit_fields)
        for writer in (batch_writer, event_writer, fit_writer):
            writer.writeheader()
        for cell_index, cell in enumerate(cells):
            identity = {k: cell[k] for k in ("dataset", "detector", "seed", "project")}
            scenario_labels = labels[cell["dataset"]][cell["positions"]]
            archive_path = output / "feedback_decisions" / cell["dataset"] / cell["detector"] / f"seed-{cell['seed']}" / f"{cell['project']}.npz"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            arrays = {"target_positions": cell["positions"]}
            for miss in config["cost"]["miss"]:
                scenario = CostScenario(miss, config["cost"]["false_alert"], config["cost"]["primary_review_cost"])
                for cap in config["feedback"]["slots_per_batch"]:
                    for delay in config["feedback"]["delays_in_batches"]:
                        for method in config["feedback"]["methods"]:
                            feedback_key = f"{scenario.key}__k{cap}__d{delay}__{method}"
                            common = {**identity, "method": method, "slots_per_batch": cap, "delay_batches": delay,
                                      "scenario": scenario.key, "feedback_key": feedback_key}
                            result = simulate_feedback(cell["probability"], cell["trace_codes"], cell["batches"], cell["source_probability"], cell["source_labels"],
                                cell["knots_x"], cell["knots_y"], scenario_labels, method=method, cap=cap, delay=delay, scenario=scenario,
                                rolling_config=config["feedback"]["rolling_update"], stable_positions=cell["positions"])
                            for name in ("codes", "arrival_eta", "service_batch", "prediction_batch"):
                                arrays[f"{feedback_key}__{name}"] = result[name]
                            _write_stream(batch_writer, common, result["batch_ledger"])
                            events = [{**event, "target_position": int(cell["positions"][event["index"]]), "row_id": cell["rows"][event["index"]]["row_id"]} for event in result["feedback_events"]]
                            _write_stream(event_writer, common, events)
                            _write_stream(fit_writer, common, result["fit_ledger"])
                            feedback_plan.append({**common, **scenario.fields(), "domain_status": cell["domain_status"],
                                "ordering": cell["batch_info"]["ordering"], "archive": str(archive_path.relative_to(output)), **result["summary"], **action_counts(result["codes"])})
            np.savez_compressed(archive_path, **arrays)
            feedback_archives.append({**identity, "path": str(archive_path.relative_to(output)), "sha256": sha256(archive_path)})
            if (cell_index + 1) % 10 == 0 or cell_index + 1 == len(cells):
                print(f"[{utc_now()}] delayed reviewed-label replay: {cell_index + 1}/{len(cells)} full-roster project/seed cells", flush=True)
    csv_output(output / "feedback_decision_counts.csv", feedback_plan)
    write_json(output / "feedback_decision_seal.json", {"created_at_utc": utc_now(), "protocol_version": config["protocol_version"],
        "known_outcome_amendment": True, "feedback_is_simulated_released_benchmark_labels": True,
        "unreviewed_labels_used_for_adaptation": False, "capacity_decision_seal_sha256": sha256(output / "capacity_decision_seal.json"),
        "feedback_archives": feedback_archives, "ledger_sha256": {name: sha256(output / name) for name in
        ("feedback_batch_ledger.csv.gz", "feedback_label_events.csv.gz", "feedback_fit_ledger.csv.gz", "feedback_decision_counts.csv")},
        "full_rosters_retained": True, "future_labels_used": False, "policy_runs": len(feedback_plan),
        "no_future_label_assertions": sum(r["no_future_label_assertions"] for r in feedback_plan), "cap_assertions": sum(r["cap_assertions"] for r in feedback_plan)})
    print(f"[{utc_now()}] feedback decisions/fit histories sealed; computing post-seal paired outcomes", flush=True)
    scenario_lookup = {scenario.key: scenario for scenario in scenarios}
    capacity_metrics = []
    for archive in capacity_archives:
        plan = [r for r in capacity_plan if r["archive"] == archive["path"]]
        with np.load(output / archive["path"], allow_pickle=False) as arc:
            y = labels[archive["dataset"]][arc["target_positions"]]
            for row in plan:
                codes = arc[row["capacity_key"]]
                capacity_metrics.append({**row, **outcome_metrics(y, codes, scenario_lookup[row["scenario"]])})
    csv_output(output / "capacity_project_seed_metrics.csv", capacity_metrics)
    feedback_metrics = []
    for archive in feedback_archives:
        plan = [r for r in feedback_plan if r["archive"] == archive["path"]]
        with np.load(output / archive["path"], allow_pickle=False) as arc:
            y = labels[archive["dataset"]][arc["target_positions"]]
            for row in plan:
                key = row["feedback_key"]
                feedback_metrics.append({**row, **outcome_metrics(y, arc[f"{key}__codes"], scenario_lookup[row["scenario"]], arrival_eta=arc[f"{key}__arrival_eta"])})
    csv_output(output / "feedback_project_seed_metrics.csv", feedback_metrics)
    capacity_setting = ["dataset", "detector", "scenario", "miss_cost", "false_alert_cost", "review_cost", "review_fn_probability", "review_fp_probability", "capacity_fraction"]
    feedback_setting = ["dataset", "detector", "scenario", "miss_cost", "false_alert_cost", "review_cost", "review_fn_probability", "review_fp_probability", "slots_per_batch", "delay_batches", "ordering"]
    capacity_projects = seed_average(pd.DataFrame(capacity_metrics), capacity_setting + ["project", "method"], actual_seeds)
    feedback_projects = seed_average(pd.DataFrame(feedback_metrics), feedback_setting + ["project", "method"], actual_seeds)
    capacity_projects.to_csv(output / "capacity_project_metrics.csv", index=False, float_format="%.12g")
    feedback_projects.to_csv(output / "feedback_project_metrics.csv", index=False, float_format="%.12g")
    endpoints = ["cost_complete_per_function", "cost_deadline_assumed_per_function", "automatic_miss_rate", "automatic_false_alert_rate", "unresolved_fraction", "review_fraction"]
    cap_pairs, cap_summary = paired_outputs(capacity_projects, settings=capacity_setting,
        comparisons=[("source_iso_cap", "msp_cap"), ("trace_mandatory", "allocation_matched_tails_mandatory"), ("trace_mandatory", "source_iso_cap")], endpoints=endpoints, config=config)
    fb_pairs, fb_summary = paired_outputs(feedback_projects, settings=feedback_setting,
        comparisons=[("source_iso_rolling", "source_iso_static"), ("trace_rolling", "trace_static")],
        endpoints=endpoints + ["arrival_probability_brier"], config=config)
    for name, frame in [("capacity_paired_project_differences.csv", cap_pairs), ("capacity_paired_summary.csv", cap_summary),
                        ("feedback_paired_project_differences.csv", fb_pairs), ("feedback_paired_summary.csv", fb_summary)]:
        frame.to_csv(output / name, index=False, float_format="%.12g")
    curve_summary(capacity_projects, capacity_setting, endpoints + ["completed_fraction", "auto_safe", "auto_vulnerable", "capacity"]).to_csv(output / "capacity_curve_plot_data.csv", index=False, float_format="%.12g")
    curve_summary(feedback_projects, feedback_setting, endpoints + ["completed_fraction", "rolling_updates", "arrived_reviewed_labels", "arrival_probability_brier"]).to_csv(output / "feedback_curve_plot_data.csv", index=False, float_format="%.12g")
    write_json(output / "feasibility_and_limitations.json", {
        "study_date": config["study_date"], "known_outcome_amendment": True,
        "source_raw_and_prepared_timestamp_fields": [], "source_before_target_established": False,
        "diversevul_timestamp_fields": [], "diversevul_order": "synthetic salted complete-commit hash; simulated benchmark ordering",
        "megavul_timestamp_fields": ["commit_date", "date"], "megavul_order": "recorded commit-author date, with simulated decisions/review/label arrival",
        "real_analyst_time_cost_or_feedback_arrival_measured": False, "feedback_from_reviewed_items_only": True,
        "feedback_selection_bias": config["feedback"]["selection_bias"], "transductive_trace_limitation": config["feedback"]["transductive_trace_limitation"],
        "mandatory_review_overflow": "unresolved; completed full-roster cost undefined; deadline harm is a separately named assumption",
        "dataset_rosters": {ds: {"projects": len(data["seal"]["projects"]), "functions": len(data["target"]),
                                 "fallback_projects": data["seal"]["domain_unsupported_projects"]} for ds, data in datasets.items()},
        "chronology_and_batch_summary": [{k: v for k, v in r.items() if k not in ("path", "sha256")} for r in batch_receipts]})
    primary_cap = cap_summary[(cap_summary.capacity_fraction == config["capacity"]["primary_fraction"]) & (cap_summary.miss_cost == 10) &
        (cap_summary.review_cost == config["cost"]["primary_review_cost"]) & (cap_summary.review_fn_probability == 0) &
        (cap_summary.candidate == "source_iso_cap") & (cap_summary.baseline == "msp_cap") & (cap_summary.endpoint == "cost_complete_per_function")]
    primary_fb = fb_summary[(fb_summary.slots_per_batch == config["feedback"]["primary_slots_per_batch"]) &
        (fb_summary.delay_batches == config["feedback"]["primary_delay_in_batches"]) & (fb_summary.miss_cost == 10) &
        (fb_summary.candidate == "source_iso_rolling") & (fb_summary.endpoint == "cost_complete_per_function")]
    write_json(output / "concise_results.json", {"protocol_version": config["protocol_version"], "known_outcome_amendment": True,
        "primary_capacity_comparison": primary_cap.to_dict("records"), "primary_feedback_comparison": primary_fb.to_dict("records"),
        "full_project_rosters": {ds: data["seal"]["projects"] for ds, data in datasets.items()},
        "interpretation": "Descriptive released-benchmark simulation. Negative cost delta favors the named candidate. No deployment, real team-cost, target calibration guarantee or prospective temporal-generalization claim.",
        "all_fixed_settings_retained": True, "source_calibration_rows": 4400,
        "capacity_plot_data": "capacity_curve_plot_data.csv", "feedback_plot_data": "feedback_curve_plot_data.csv"})
    integrity = {"status": "PASS", "created_at_utc": utc_now(), "capacity_policy_cells": len(capacity_metrics), "feedback_policy_runs": len(feedback_metrics),
        "project_seed_cells": len(cells), "no_future_label_assertions": sum(r["no_future_label_assertions"] for r in feedback_plan),
        "batch_cap_assertions": sum(r["cap_assertions"] for r in feedback_plan), "all_selected_projects_retained": True,
        "all_actual_seeds_retained": True, "exact_source_calibration_role_verified": True,
        "no_unreviewed_feedback_labels": True, "mandatory_overflow_preserved": True, "incomplete_full_roster_cost_undefined": True}
    write_json(output / "integrity_report.json", integrity)
    lines = ["# Capacity and delayed-feedback benchmark audit", "", f"Dated protocol: {config['study_date']}. All previous benchmark outcomes were already known.", "",
        "Complete rosters: DiverseVul38 projects/26,079 functions; MegaVul6 projects/25,722 functions. Actual detector seeds are averaged within project. Costs are illustrative scenario units; each function review consumes one analyst slot.", "",
        "Mandatory TRACE review above capacity remains unresolved backlog. Complete-roster cost is undefined where any backlog remains. Lower-bound and deadline costs are explicitly separate assumed scenarios.", "",
        "MegaVul uses recorded commit-author order; DiverseVul uses synthetic commit-hash order. All analyst decisions and delayed label arrivals are simulated. Source dates are unavailable. Reviewed feedback is selection-biased, and inherited TRACE domain fits are transductive.", "",
        "Primary cost deltas (candidate minus baseline, project median [pointwise descriptive95% interval]):", ""]
    for label, frame in [("source cap minus MSP, capacity.20", primary_cap), ("rolling minus static source cost, slots16/delay3", primary_fb)]:
        for row in frame.to_dict("records"):
            lines.append(f"- {row['dataset']}/{row['detector']}, {label}: {row['median_paired_difference']:.6g} [{row['ci_lower']:.6g}, {row['ci_upper']:.6g}], n={row['projects_roster']}.")
    lines.extend(["", "All scenarios, per-project pairs and automatic-label allocations are exported. Plot data are in capacity_curve_plot_data.csv and feedback_curve_plot_data.csv. feedback_batch_ledger.csv.gz, feedback_label_events.csv.gz and feedback_fit_ledger.csv.gz retain the label-access audit.", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    # Recheck immutable analysis/source dependencies; outputs below bind every result byte.
    verify_freeze(config_path)
    for record in dependencies.values():
        require(sha256(Path(record["path"])) == record["sha256"], "An inherited input changed during v4 analysis")
    artifacts = [{"path": str(path.relative_to(output)), "sha256": sha256(path), "bytes": path.stat().st_size}
                 for path in sorted(output.rglob("*")) if path.is_file()]
    manifest = {"status": "COMPLETE", "protocol_version": config["protocol_version"], "started_at_utc": started, "completed_at_utc": utc_now(),
        "known_outcome_amendment": True, "config_sha256": sha256(config_path), "protocol_sha256": sha256(ROOT / config["protocol_path"]),
        "script_sha256": sha256(Path(__file__)), "freeze_receipt_sha256": sha256(FREEZE), "artifacts": artifacts,
        "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "device": "CPU"},
        **{k: v for k, v in integrity.items() if k not in ("status", "created_at_utc")}}
    write_json(output / "analysis_manifest.json", manifest)
    (output / "analysis_manifest.sha256").write_text(sha256(output / "analysis_manifest.json") + "\n", encoding="ascii")
    return manifest


def verify_output(output: Path) -> dict:
    manifest_path = output / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["status"] == "COMPLETE" and manifest["known_outcome_amendment"], "Invalid completed analysis manifest")
    require(sha256(manifest_path) == (output / "analysis_manifest.sha256").read_text().strip(), "V4 analysis manifest digest mismatch")
    for record in manifest["artifacts"]:
        path = (output / record["path"]).resolve()
        require(path.is_relative_to(output.resolve()) and sha256(path) == record["sha256"], f"Result artifact changed: {record['path']}")
    checked = 0
    for filename, cap_column in [("capacity_project_seed_metrics.csv", "capacity"), ("feedback_project_seed_metrics.csv", "review_slots_total")]:
        frame = pd.read_csv(output / filename)
        require((frame.reviewed <= frame[cap_column]).all(), "Result review count exceeded capacity")
        require((frame.auto_safe + frame.auto_vulnerable + frame.reviewed + frame.unresolved == frame.n).all(), "Result full roster disappeared")
        require((frame.n_vulnerable + frame.n_safe == frame.n).all(), "Result original class denominator changed")
        require(frame.loc[frame.unresolved > 0, "cost_complete_per_function"].isna().all(), "Incomplete roster received an ordinary full-cost score")
        require(frame.loc[frame.unresolved == 0, "cost_complete_per_function"].notna().all(), "Complete roster is missing its ordinary cost")
        require(np.allclose(frame.automatic_miss_rate, frame.auto_fn / frame.n_vulnerable, atol=1e-10), "Automatic miss denominator changed")
        require(np.allclose(frame.automatic_false_alert_rate, frame.auto_fp / frame.n_safe, atol=1e-10), "Automatic false-alert denominator changed")
        expected = (frame.miss_cost * frame.auto_fn + frame.false_alert_cost * frame.auto_fp + frame.review_cost * frame.reviewed
                    + frame.miss_cost * frame.review_fn_probability * frame.reviewed_vulnerable + frame.false_alert_cost * frame.review_fp_probability * frame.reviewed_safe) / frame.n
        require(np.allclose(expected, frame.cost_omitting_unresolved_lower_bound, atol=1e-9), "Cost scenario calculation changed")
        require(np.allclose(expected + frame.miss_cost * frame.unresolved_vulnerable / frame.n, frame.cost_deadline_assumed_per_function, atol=1e-9), "Backlog received implicit safety credit")
        checked += len(frame)
    batches = pd.read_csv(output / "feedback_batch_ledger.csv.gz")
    require((batches.reviewed_in_batch <= batches.slots).all() and batches.cap_assertion.all(), "Sequential batch review cap exceeded")
    require((batches.maximum_used_service_batch < batches.batch).all() and (batches.maximum_used_arrival_batch <= batches.batch).all() and batches.no_future_label_assertion.all(), "Future-label use in fit ledger")
    events = pd.read_csv(output / "feedback_label_events.csv.gz")
    require((events.service_batch < events.reveal_batch).all() and (events.arrival_batch <= events.reveal_batch).all(), "Future/current-batch feedback in release ledger")
    require((events.arrival_batch == events.service_batch + events.delay_batches).all(), "Feedback delay changed")
    require(not events.duplicated(["dataset", "detector", "seed", "project", "feedback_key", "index"]).any(), "Repeated/free reviewed feedback")
    fits = pd.read_csv(output / "feedback_fit_ledger.csv.gz")
    if len(fits):
        require((fits.available_window_n >= 30).all() and (fits.available_window_safe >= 5).all() and (fits.available_window_vulnerable >= 5).all() and (fits.used_target_n <= 512).all(), "Rolling eligibility/window changed")
    return {"status": "PASS", "artifacts_verified": len(manifest["artifacts"]), "outcome_policy_rows_checked": checked,
            "batch_caps_and_future_label_checks": len(batches), "reviewed_feedback_events_checked": len(events), "eligible_rolling_fits_checked": len(fits),
            "known_outcome_amendment": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "verify"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/capacity_feedback_v4.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/kbs-validation-v4/capacity_feedback")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.config.resolve())
    elif args.command == "run":
        result = run(args.config.resolve(), args.output.resolve())
        result = {k: v for k, v in result.items() if k != "artifacts"}
    else:
        result = verify_output(args.output.resolve())
    print(json.dumps(clean_json(result), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
