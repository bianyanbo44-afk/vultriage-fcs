"""Outcome-independent tests of cap, chronology, source roles and label arrival."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from capacity_feedback_v4 import (BACKLOG, REVIEW, CostScenario, DelayedReviewOracle, ReviewLabel, RollingIsotonic,
    action_counts, allocate_review, capacity_policies, make_batches, msp_review, outcome_metrics,
    simulate_feedback, source_calibration_view)


def row(index, commit, date=None):
    result = {"position": index, "row_id": f"row-{index}", "project_group": "project", "commit_id": f"{commit:040x}"}
    if date is not None:
        result["commit_date"] = date
    return result


ROLLING = {"maximum_arrived_window": 4, "minimum_arrived_rows": 2, "minimum_arrived_per_class": 1,
           "source_total_anchor_weight": 2.0, "target_per_row_weight": 1.0}


class CapacityTests(unittest.TestCase):
    def test_random_hard_caps(self):
        rng = np.random.default_rng(20261011)
        for n in (1, 2, 20, 117):
            p = rng.uniform(size=n)
            membership = rng.integers(0, 3, size=n)
            for cap in (0, 1, n // 2, n, n + 5):
                for mandatory in (None, membership):
                    codes, required = allocate_review(p, cap, CostScenario(10), mandatory_codes=mandatory)
                    self.assertLessEqual(np.sum(codes == REVIEW), cap)
                    self.assertEqual(sum(action_counts(codes)[k] for k in ("auto_safe", "auto_vulnerable", "reviewed", "unresolved")), n)
                    if mandatory is not None:
                        self.assertEqual(np.sum(codes == BACKLOG), max(0, np.sum(membership == REVIEW) - cap))
                        np.testing.assert_array_equal(codes[membership != REVIEW], membership[membership != REVIEW])

    def test_high_cap_does_not_force_noneconomic_review(self):
        codes, required = allocate_review(np.array([0.0, 1.0, 0.25, 0.95]), 100, CostScenario(10, review=0.1))
        self.assertEqual(required, 1)
        np.testing.assert_array_equal(codes, [0, 1, REVIEW, 1])

    def test_review_error_cost_changes_economic_review(self):
        perfect, _ = allocate_review(np.array([0.1, 0.2]), 2, CostScenario(10, review=0.1))
        imperfect, _ = allocate_review(np.array([0.1, 0.2]), 2, CostScenario(10, review=0.1, review_fn_probability=1, review_fp_probability=1))
        self.assertEqual(np.sum(perfect == REVIEW), 2)
        self.assertEqual(np.sum(imperfect == REVIEW), 0)

    def test_mandatory_fallback_retains_infeasible_backlog(self):
        codes, required = allocate_review(np.zeros(7), 2, CostScenario(10), mandatory_codes=np.full(7, REVIEW))
        self.assertEqual(required, 7)
        self.assertEqual(action_counts(codes)["reviewed"], 2)
        self.assertEqual(action_counts(codes)["unresolved"], 5)
        self.assertEqual(action_counts(codes)["feasible_complete_roster"], 0)

    def test_incomplete_cost_is_undefined_and_deadline_penalty_explicit(self):
        result = outcome_metrics(np.array([1, 0, 1, 0]), np.array([BACKLOG, BACKLOG, REVIEW, 1]), CostScenario(10))
        self.assertTrue(np.isnan(result["cost_complete_per_function"]))
        self.assertEqual(result["n_vulnerable"], 2)
        self.assertEqual(result["n_safe"], 2)
        self.assertAlmostEqual(result["cost_omitting_unresolved_lower_bound"], 0.275)
        self.assertAlmostEqual(result["cost_deadline_assumed_per_function"], 2.775)
        self.assertEqual(result["auto_fn"], 0)

    def test_stable_review_rank_and_argmax_tie(self):
        codes = msp_review(np.array([0.5, 0.5, 0.7]), 1, stable_positions=np.array([9, 3, 5]))
        np.testing.assert_array_equal(codes, [0, REVIEW, 1])
        codes, _ = allocate_review(np.full(3, 0.25), 1, CostScenario(10), stable_positions=np.array([9, 3, 5]))
        np.testing.assert_array_equal(codes, [1, REVIEW, 1])

    def test_cap_type_and_memberships_reject_invalid_inputs(self):
        for cap in (-1, 1.2, True):
            with self.assertRaises(ValueError):
                allocate_review(np.array([0.2]), cap, CostScenario(10))
        with self.assertRaises(ValueError):
            allocate_review(np.array([0.2]), 1, CostScenario(10), mandatory_codes=np.array([BACKLOG]))

    def test_original_matched_allocation_is_preserved_under_caps(self):
        memberships = {m: np.array([0, 1, REVIEW, REVIEW]) for m in
                       ("trace_mandatory", "allocation_matched_tails_mandatory", "weighted_volume_msp_mandatory")}
        policies = capacity_policies(np.array([0.1, 0.8, 0.5, 0.2]), np.array([0.1, 0.8, 0.5, 0.2]), memberships, 1, CostScenario(10), np.arange(4))
        for method in memberships:
            counts = action_counts(policies[method][0])
            self.assertEqual((counts["auto_safe"], counts["auto_vulnerable"], counts["reviewed"], counts["unresolved"]), (1, 1, 1, 1))


class ChronologyAndSourceTests(unittest.TestCase):
    def test_commit_and_equal_date_blocks_never_split(self):
        rows = [row(0, 1, 100), row(1, 1, 100), row(2, 2, 100), row(3, 3, 200), row(4, 4, 300)]
        batches, info = make_batches(rows, ordering="recorded_commit_author_date", nominal_rows=2, salt="test")
        self.assertEqual(batches[0].indices, (0, 1, 2))
        self.assertEqual(info["oversized_batches"], 1)
        self.assertTrue(info["date_ties_same_batch"])
        self.assertEqual(sum(len(b.indices) for b in batches), 5)
        self.assertLess(batches[0].maximum_date, batches[1].minimum_date)

    def test_synthetic_order_is_deterministic_and_label_free(self):
        rows = [row(i, i // 2 + 1) for i in range(12)]
        a, _ = make_batches(rows, ordering="synthetic_salted_commit_hash", nominal_rows=3, salt="fixed")
        b, _ = make_batches(rows, ordering="synthetic_salted_commit_hash", nominal_rows=3, salt="fixed")
        self.assertEqual(a, b)
        rows[0]["label"] = 1
        with self.assertRaises(ValueError):
            make_batches(rows, ordering="synthetic_salted_commit_hash", nominal_rows=3, salt="fixed")

    def test_missing_or_conflicting_dates_cannot_be_called_chronological(self):
        for rows in ([row(0, 1)], [row(0, 1, 100), row(1, 1, 101)]):
            with self.assertRaises(ValueError):
                make_batches(rows, ordering="recorded_commit_author_date", nominal_rows=2, salt="test")

    def test_only_exact_source_calibration_positions_are_accepted(self):
        source = [{"role": "train", "label": 1}, {"role": "model_validation", "label": 0},
                  {"role": "calibration", "label": 1}, {"role": "calibration", "label": 0}]
        p, y = source_calibration_view(source, np.array([1, 2, 3]), np.array([0.2, 0.7, 0.3]), np.array([2, 3]))
        np.testing.assert_array_equal(y, [1, 0])
        np.testing.assert_allclose(p, [0.7, 0.3])
        with self.assertRaises(ValueError):
            source_calibration_view(source, np.array([1, 2, 3]), np.array([0.2, 0.7, 0.3]), np.array([1, 2]))


class FeedbackTests(unittest.TestCase):
    def test_oracle_delays_and_never_releases_unreviewed_labels(self):
        oracle = DelayedReviewOracle(np.array([1, 0, 1, 0]), 3)
        self.assertEqual(oracle.release(0), [])
        oracle.schedule([1], batch=0, cap=1)
        self.assertEqual(oracle.label_accesses, 0)
        self.assertEqual(oracle.release(1), [])
        oracle.schedule([], batch=1, cap=1)
        self.assertEqual(oracle.release(2), [])
        oracle.schedule([], batch=2, cap=1)
        events = oracle.release(3)
        self.assertEqual(events, [ReviewLabel(1, 0, 0, 3)])
        self.assertEqual(oracle.label_accesses, 1)
        self.assertEqual(oracle.reviewed_count, 1)

    def test_oracle_rejects_zero_delay_duplicate_review_and_cap_overflow(self):
        with self.assertRaises(ValueError):
            DelayedReviewOracle(np.array([0, 1]), 0)
        oracle = DelayedReviewOracle(np.array([0, 1]), 1)
        oracle.release(0)
        with self.assertRaises(ValueError):
            oracle.schedule([0, 1], batch=0, cap=1)
        oracle.schedule([0], batch=0, cap=1)
        oracle.release(1)
        with self.assertRaises(ValueError):
            oracle.schedule([0], batch=1, cap=1)
        with self.assertRaises(ValueError):
            oracle.release(1)

    def calibrator(self):
        return RollingIsotonic(np.array([0.0, 0.25, 0.75, 1.0]), np.array([0, 0, 1, 1]), np.array([0.0, 0.25, 0.75, 1.0]), np.array([0.0, 0.0, 1.0, 1.0]), ROLLING)

    def test_rolling_rejects_current_and_future_feedback(self):
        for event in (ReviewLabel(0, 1, 2, 3), ReviewLabel(0, 1, 0, 3), ReviewLabel(0, 1, -1, 0)):
            with self.assertRaises(ValueError):
                self.calibrator().observe([event], batch=2, target_probability=np.array([0.5]))

    def test_rolling_eligibility_repeated_labels_and_fixed_window(self):
        model = self.calibrator()
        first = model.observe([ReviewLabel(0, 1, 0, 1)], batch=1, target_probability=np.array([0.2, 0.8, 0.4, 0.6, 0.9]))
        self.assertEqual(first["rolling_fit_count"], 0)
        second = model.observe([ReviewLabel(1, 0, 1, 2)], batch=2, target_probability=np.array([0.2, 0.8, 0.4, 0.6, 0.9]))
        self.assertEqual(second["rolling_fit_count"], 1)
        with self.assertRaises(ValueError):
            model.observe([ReviewLabel(1, 0, 1, 2)], batch=3, target_probability=np.array([0.2, 0.8, 0.4, 0.6, 0.9]))
        third = model.observe([ReviewLabel(2, 1, 2, 3), ReviewLabel(3, 0, 2, 3), ReviewLabel(4, 1, 2, 3)], batch=3,
                              target_probability=np.array([0.2, 0.8, 0.4, 0.6, 0.9]))
        self.assertEqual(third["used_target_n"], 4)
        self.assertEqual(third["maximum_used_service_batch"], 2)

    def test_future_scores_do_not_enter_an_arrived_feedback_fit(self):
        a, b = self.calibrator(), self.calibrator()
        events = [ReviewLabel(0, 1, 0, 1), ReviewLabel(1, 0, 0, 1)]
        x = a.observe(events, batch=1, target_probability=np.array([0.2, 0.8, 0.9, 0.1]))
        y = b.observe(events, batch=1, target_probability=np.array([0.2, 0.8, 0.01, 0.99]))
        self.assertEqual(x["knots_sha256"], y["knots_sha256"])

    def test_temporal_trace_carries_backlog_without_free_drain(self):
        rows = [row(i, i + 1, 100 + i) for i in range(10)]
        batches, _ = make_batches(rows, ordering="recorded_commit_author_date", nominal_rows=2, salt="test")
        result = simulate_feedback(np.full(10, 0.2), np.full(10, REVIEW), batches, np.array([0., 1.]), np.array([0, 1]),
            np.array([0., 1.]), np.array([0., 1.]), np.arange(10) % 2, method="trace_static", cap=1, delay=3,
            scenario=CostScenario(10), rolling_config=ROLLING, stable_positions=np.arange(10))
        self.assertEqual(np.sum(result["codes"] == REVIEW), 5)
        self.assertEqual(np.sum(result["codes"] == BACKLOG), 5)
        self.assertEqual(result["summary"]["reviewed_labels_pending_at_end"], 3)
        self.assertTrue(all(row["reviewed_in_batch"] <= 1 for row in result["batch_ledger"]))
        self.assertEqual(result["summary"]["free_postrun_drain"], 0)

    def test_rolling_decisions_before_future_label_arrival_are_invariant(self):
        rows = [row(i, i + 1, 100 + i) for i in range(12)]
        batches, _ = make_batches(rows, ordering="recorded_commit_author_date", nominal_rows=2, salt="test")
        kwargs = dict(method="source_iso_rolling", cap=1, delay=2, scenario=CostScenario(10), rolling_config=ROLLING, stable_positions=np.arange(12))
        common = (np.full(12, 0.2), np.full(12, REVIEW), batches, np.array([0., 1.]), np.array([0, 1]), np.array([0., 1.]), np.array([0., 1.]))
        labels = np.arange(12) % 2
        changed = labels.copy()
        changed[4:] = 1 - changed[4:]
        a = simulate_feedback(*common, labels, **kwargs)
        b = simulate_feedback(*common, changed, **kwargs)
        # Rows4+ are first reviewed in batch2+, so their labels arrive in batch4+.
        np.testing.assert_array_equal(a["codes"][:8], b["codes"][:8])
        np.testing.assert_array_equal(a["arrival_eta"][:8], b["arrival_eta"][:8])
        self.assertEqual([r["knots_sha256"] for r in a["batch_ledger"][:4]], [r["knots_sha256"] for r in b["batch_ledger"][:4]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
