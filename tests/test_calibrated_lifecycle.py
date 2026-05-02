"""
Test suite for the calibrated detection-existence pipeline.

Covers:
  1. CalibrationTable: bin lookup, P(TP) Bayes posterior, graceful fallback
     for missing bins / NaN data / fallback-source rows.
  2. Phase A: P(TP) birth gate in Fusion — gate on, gate off, gate borderline.
  3. Phase B: IPDA log-odds lifecycle on GlobalTracked / Fusion — spawn-init,
     match-update, miss-decay, confirm/kill thresholds, tracking_score.
  4. Phase A + Phase B composition: gate keeps weak births out of the log-odds
     lifecycle entirely.
  5. Backwards compatibility: legacy lifecycle_mode preserves count-based
     emit/cleanDetections semantics; detectors without `.p_tp` are no-op'd.

These tests deliberately avoid loading any CSV — calibration tables are
constructed in-memory so the test suite stays hermetic. The CSV-loading
path is exercised by a separate integration check.

Run: python -m pytest tests/test_calibrated_lifecycle.py -v -s
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sensor_model_loader import CalibrationBin, CalibrationTable
import sensor_fusion
import sensor


# ============================================================================
# HELPERS
# ============================================================================

def make_det(x, y, p_tp=None, score=0.5, vid=1):
    """Build a minimal DetectedObject for fusion-pipeline tests."""
    d = sensor.DetectedObject(
        vehicle_id=vid, vehicle_type="car",
        detected_bbox=None,
        centroid=[x, y],
        width=2.0, length=4.0, angle=0.0,
        expected_error_gaussian=None,
        velocity_vector=[0.0, 0.0],
    )
    if p_tp is not None:
        d.p_tp = p_tp
    return d


def synthetic_calibration_table(n_tp=100, n_fp=10,
                                 tp_mu=0.75, tp_sigma=0.10,
                                 fp_mu=0.30, fp_sigma=0.10):
    """Single-bin (5-10m, -10°-+10°) calibration table for unit-level
    posterior tests. Defaults: clean TP-FP separation, TP-dominant prior."""
    bins = [CalibrationBin(
        range_lo=5.0, range_hi=10.0,
        angle_lo_deg=-10.0, angle_hi_deg=10.0,
        gt_count=n_tp, n_missed=0, miss_rate=0.0,
        n_fp=n_fp, fp_per_frame=n_fp / 100.0,
        score_mean=tp_mu, score_std=tp_sigma,
        fp_score_mean=fp_mu, fp_score_std=fp_sigma,
        source="measured",
    )]
    return CalibrationTable(bins)


# ============================================================================
# 1. CalibrationTable
# ============================================================================

class TestCalibrationTable(unittest.TestCase):

    def test_lookup_bin_hits_expected_bin(self):
        cal = synthetic_calibration_table()
        b = cal.lookup_bin(distance_m=7.5, angle_rad=math.radians(0.0))
        self.assertIsNotNone(b)
        self.assertEqual(b.range_lo, 5.0)
        self.assertEqual(b.angle_lo_deg, -10.0)

    def test_lookup_bin_out_of_range_returns_none(self):
        cal = synthetic_calibration_table()
        # Distance well outside any bin
        self.assertIsNone(cal.lookup_bin(100.0, math.radians(0.0)))
        # Angle outside the single 20° wedge
        self.assertIsNone(cal.lookup_bin(7.5, math.radians(180.0)))

    def test_p_tp_strong_score_high(self):
        """Score near TP centroid → posterior near 1."""
        cal = synthetic_calibration_table()
        p = cal.tp_probability(7.5, math.radians(0.0), 0.85)
        self.assertGreater(p, 0.95)

    def test_p_tp_weak_score_low(self):
        """Score near FP centroid → posterior near 0."""
        cal = synthetic_calibration_table()
        p = cal.tp_probability(7.5, math.radians(0.0), 0.20)
        self.assertLess(p, 0.30)

    def test_p_tp_no_calibration_returns_one(self):
        """Out-of-range lookup → graceful 1.0 (don't gate)."""
        cal = synthetic_calibration_table()
        self.assertEqual(cal.tp_probability(100.0, math.radians(0.0), 0.5), 1.0)

    def test_p_tp_fallback_bin_uses_global_pool(self):
        """Bin marked source='fallback' with NaN score data falls back to
        the globally-pooled TP/FP score Gaussians; the posterior lands in
        [0, 1] and behaves consistently with the global TP-favoured prior."""
        bins = [
            CalibrationBin(0, 5, -10, 10,
                           gt_count=50, n_missed=0, miss_rate=0,
                           n_fp=5, fp_per_frame=0.05,
                           score_mean=0.8, score_std=0.1,
                           fp_score_mean=0.3, fp_score_std=0.1,
                           source="measured"),
            CalibrationBin(5, 10, -10, 10,
                           gt_count=0, n_missed=0, miss_rate=0,
                           n_fp=0, fp_per_frame=0.0,
                           score_mean=None, score_std=None,
                           fp_score_mean=None, fp_score_std=None,
                           source="fallback"),
        ]
        cal = CalibrationTable(bins)
        # Strong score — global TP centroid at 0.8, FP at 0.3 → near-1
        p_strong = cal.tp_probability(7.5, math.radians(0.0), 0.85)
        # Weak score — should land lower
        p_weak = cal.tp_probability(7.5, math.radians(0.0), 0.20)
        self.assertGreater(p_strong, p_weak)
        self.assertTrue(0.0 <= p_strong <= 1.0)
        self.assertTrue(0.0 <= p_weak <= 1.0)

    def test_p_tp_extreme_priors_dont_blow_up(self):
        """Bin with all TPs and zero FPs (after Laplace) → very high P_TP
        regardless of score; bin with zero data → near 0.5."""
        cal_all_tp = synthetic_calibration_table(n_tp=10000, n_fp=0)
        cal_empty  = CalibrationTable([CalibrationBin(
            5, 10, -10, 10,
            gt_count=0, n_missed=0, miss_rate=0,
            n_fp=0, fp_per_frame=0.0,
            score_mean=None, score_std=None,
            fp_score_mean=None, fp_score_std=None,
            source="measured")])
        p_high = cal_all_tp.tp_probability(7.5, math.radians(0.0), 0.5)
        p_mid  = cal_empty.tp_probability(7.5, math.radians(0.0), 0.5)
        self.assertGreater(p_high, 0.99)
        # Empty bin: Laplace-smoothed (1 TP, 1 FP) + score 0.5 between global
        # means → posterior should be near 0.5.
        self.assertTrue(0.2 < p_mid < 0.8)

    def test_p_tp_angle_wraps_correctly(self):
        """Lookups at +185° and -175° hit the same wrapped bin."""
        bins = [CalibrationBin(0, 5, -180, -170,
                               gt_count=10, n_missed=0, miss_rate=0,
                               n_fp=1, fp_per_frame=0.01,
                               score_mean=0.7, score_std=0.1,
                               fp_score_mean=0.3, fp_score_std=0.1,
                               source="measured")]
        cal = CalibrationTable(bins)
        b1 = cal.lookup_bin(2.5, math.radians(185.0))
        b2 = cal.lookup_bin(2.5, math.radians(-175.0))
        self.assertIsNotNone(b1)
        self.assertEqual(b1.angle_lo_deg, b2.angle_lo_deg)


# ============================================================================
# 2. Phase A — birth gate
# ============================================================================

class TestPhaseABirthGate(unittest.TestCase):

    def test_no_gate_spawns_everything(self):
        f = sensor_fusion.Fusion(0, p_tp_birth_gate=None)
        dets = [make_det(x*5, 0, p_tp=0.1, vid=i) for i, x in enumerate(range(5))]
        f.processDetectionFrame(0.0, dets, 0.5)
        self.assertEqual(len(f.tracked_list), 5)

    def test_gate_blocks_low_p_tp(self):
        f = sensor_fusion.Fusion(0, p_tp_birth_gate=0.5)
        dets = [
            make_det( 0, 0, p_tp=0.9, vid=1),
            make_det(10, 0, p_tp=0.3, vid=2),
            make_det(20, 0, p_tp=0.6, vid=3),
        ]
        f.processDetectionFrame(0.0, dets, 0.5)
        self.assertEqual(len(f.tracked_list), 2)

    def test_gate_borderline_inclusive_above(self):
        """A detection with p_tp exactly equal to the gate spawns (>=, not >)."""
        f = sensor_fusion.Fusion(0, p_tp_birth_gate=0.5)
        f.processDetectionFrame(0.0, [make_det(0, 0, p_tp=0.5)], 0.5)
        self.assertEqual(len(f.tracked_list), 1)

    def test_gate_no_p_tp_attribute_passes(self):
        """Detections without `.p_tp` are treated as 1.0 — no gate effect.
        Important for backwards compatibility with the simulation pipeline,
        which doesn't set p_tp on its DetectedObjects yet."""
        f = sensor_fusion.Fusion(0, p_tp_birth_gate=0.99)
        det = make_det(0, 0, p_tp=None)   # no .p_tp attribute
        f.processDetectionFrame(0.0, [det], 0.5)
        self.assertEqual(len(f.tracked_list), 1)


# ============================================================================
# 3. Phase B — log-odds lifecycle
# ============================================================================

class TestPhaseBLogOddsLifecycle(unittest.TestCase):

    def test_spawn_initializes_S_t_from_p_tp(self):
        f = sensor_fusion.Fusion(0, lifecycle_mode="log_odds")
        f.processDetectionFrame(0.0, [make_det(0, 0, p_tp=0.95)], 0.5)
        t = f.tracked_list[0]
        # logit(0.95) ≈ +2.944
        self.assertAlmostEqual(t.S_t, math.log(0.95 / 0.05), places=3)

    def test_spawn_default_S_t_is_zero_when_no_p_tp(self):
        f = sensor_fusion.Fusion(0, lifecycle_mode="log_odds")
        f.processDetectionFrame(0.0, [make_det(0, 0, p_tp=None)], 0.5)
        t = f.tracked_list[0]
        self.assertAlmostEqual(t.S_t, 0.0, places=6)

    def test_match_grows_S_t(self):
        f = sensor_fusion.Fusion(0, lifecycle_mode="log_odds")
        det = make_det(0, 0, p_tp=0.8)
        f.processDetectionFrame(0.0, [det], 0.5)
        t = f.tracked_list[0]
        s0 = t.S_t
        # second frame: same detection (slight nudge so it matches)
        det2 = make_det(0.05, 0, p_tp=0.8, vid=1)
        f.processDetectionFrame(0.1, [det2], 0.5)
        f.fuseDetectionFrame(0.1)
        self.assertGreater(t.S_t, s0)

    def test_miss_decays_S_t(self):
        f = sensor_fusion.Fusion(0, lifecycle_mode="log_odds")
        f.processDetectionFrame(0.0, [make_det(0, 0, p_tp=0.95)], 0.5)
        f.fuseDetectionFrame(0.0)
        t = f.tracked_list[0]
        s0 = t.S_t
        # Run a frame with no detections — miss decay should fire
        f.processDetectionFrame(0.1, [], 50.0)  # huge cleanup_time keeps stale-delete out
        f.fuseDetectionFrame(0.1)
        self.assertAlmostEqual(t.S_t - s0, f.log_lr_miss, places=6)

    def test_kill_threshold_removes_track(self):
        f = sensor_fusion.Fusion(0, lifecycle_mode="log_odds")
        f.processDetectionFrame(0.0, [make_det(0, 0, p_tp=0.95)], 0.5)
        f.fuseDetectionFrame(0.0)
        # Pile on misses until S_t < kill threshold
        for k in range(50):
            f.processDetectionFrame(0.1 * (k + 1), [], 50.0)
            f.fuseDetectionFrame(0.1 * (k + 1))
        self.assertEqual(len(f.tracked_list), 0)

    def test_confirm_threshold_emits(self):
        """Strong p_tp on spawn alone exceeds confirm; track emitted on
        first fuseDetectionFrame."""
        f = sensor_fusion.Fusion(0, lifecycle_mode="log_odds")
        f.processDetectionFrame(0.0, [make_det(0, 0, p_tp=0.99)], 0.5)
        result, dets, _, _ = f.fuseDetectionFrame(0.0)
        # logit(0.99) > log(20) ≈ 2.996 → emit
        self.assertEqual(len(result), 1)

    def test_legacy_mode_preserves_count_based_emit(self):
        """In legacy mode, a single detection does NOT emit (needs >4 fusion
        steps). Backwards compatibility for the simulation pipeline."""
        f = sensor_fusion.Fusion(0, lifecycle_mode="legacy")
        f.processDetectionFrame(0.0, [make_det(0, 0, p_tp=0.99)], 0.5)
        result, _, _, _ = f.fuseDetectionFrame(0.0)
        self.assertEqual(len(result), 0)

    def test_tracking_score_is_sigmoid_of_S_t(self):
        f = sensor_fusion.Fusion(0, lifecycle_mode="log_odds")
        f.processDetectionFrame(0.0, [make_det(0, 0, p_tp=0.7)], 0.5)
        t = f.tracked_list[0]
        # sigmoid(logit(0.7)) = 0.7
        self.assertAlmostEqual(t.tracking_score, 0.7, places=6)

    def test_log_lr_clamp_prevents_runaway(self):
        """An adversarial p_tp=1-eps (extreme logit) is clamped per update."""
        f = sensor_fusion.Fusion(0, lifecycle_mode="log_odds")
        d = make_det(0, 0, p_tp=0.999999)
        f.processDetectionFrame(0.0, [d], 0.5)
        t = f.tracked_list[0]
        # Spawn S_t comes from _logit at GlobalTracked construction (no clamp
        # there because it's a one-shot init). But subsequent updates clamp.
        s0 = t.S_t
        for k in range(100):
            d2 = make_det(0.01 * (k + 1), 0, p_tp=0.999999, vid=1)
            f.processDetectionFrame(0.1 * (k + 1), [d2], 0.5)
            f.fuseDetectionFrame(0.1 * (k + 1))
        # 100 matches × LOG_LR_CLAMP=6.0 ⇒ unbounded would be ~1380, clamp
        # caps each step at 6.0 so we should land near s0 + 100*6 = 614 max.
        # But also miss-decay isn't triggered here (every frame has a match).
        self.assertLessEqual(t.S_t, s0 + 100 * sensor_fusion.LOG_LR_CLAMP + 1.0)


# ============================================================================
# 4. Phase A + Phase B composition
# ============================================================================

class TestComposition(unittest.TestCase):

    def test_gate_blocks_birth_into_log_odds_pipeline(self):
        f = sensor_fusion.Fusion(0, p_tp_birth_gate=0.5, lifecycle_mode="log_odds")
        # Strong detection passes gate AND seeds a high S_t
        # Weak detection blocked by gate — doesn't enter the lifecycle at all
        f.processDetectionFrame(0.0, [
            make_det( 0, 0, p_tp=0.9, vid=1),
            make_det(20, 0, p_tp=0.2, vid=2),
        ], 0.5)
        self.assertEqual(len(f.tracked_list), 1)
        self.assertAlmostEqual(f.tracked_list[0].S_t, math.log(0.9 / 0.1), places=3)


# ============================================================================
# 5. Backwards compatibility — simulation pipeline
# ============================================================================

class TestBackwardsCompatibility(unittest.TestCase):

    def test_default_construction_unchanged(self):
        """Fusion() with no Phase A/B kwargs reproduces legacy behaviour:
        no birth gate, count-based emit, no log-odds decay."""
        f = sensor_fusion.Fusion(0)
        self.assertIsNone(f.p_tp_birth_gate)
        self.assertEqual(f.lifecycle_mode, "legacy")

    def test_legacy_track_has_S_t_field_but_unused(self):
        """GlobalTracked still gets S_t for type consistency, but legacy mode
        ignores it for emit/kill decisions."""
        f = sensor_fusion.Fusion(0)
        f.processDetectionFrame(0.0, [make_det(0, 0, p_tp=None)], 0.5)
        t = f.tracked_list[0]
        self.assertTrue(hasattr(t, "S_t"))   # always present
        # Default S_t = logit(0.5) = 0; tracking_score property still callable
        self.assertAlmostEqual(t.tracking_score, 0.5, places=6)


# ============================================================================
# 6. Pluggable Hungarian cost (mahal_weight / iou_weight / mahal_gate)
# ============================================================================

class TestPluggableHungarian(unittest.TestCase):
    """The matching cost in Fusion.matchDetections is a hybrid of
    Mahalanobis + IoU. mahal_weight and iou_weight are now config knobs
    so we can swap to pure-IoU (AB3DMOT-style) or pure-Mahalanobis."""

    def test_default_weights_are_hybrid(self):
        f = sensor_fusion.Fusion(0)
        self.assertAlmostEqual(f.mahal_weight, 0.6)
        self.assertAlmostEqual(f.iou_weight,   0.4)
        self.assertAlmostEqual(f.mahal_gate,  13.82)

    def test_pure_iou_config(self):
        """Pure IoU = mahal_weight=0, iou_weight=1, large gate."""
        f = sensor_fusion.Fusion(0, mahal_weight=0.0, iou_weight=1.0,
                                 mahal_gate=1e9)
        self.assertEqual(f.mahal_weight, 0.0)
        self.assertEqual(f.iou_weight,   1.0)
        # With pure-IoU config a fresh detection that overlaps the existing
        # track matches; no overlap → unmatched.
        f.processDetectionFrame(0.0, [make_det(0, 0)], 0.5)
        self.assertEqual(len(f.tracked_list), 1)
        # Overlapping detection at near-identical pose: should match (no
        # spawn of duplicate track).
        f.processDetectionFrame(0.1, [make_det(0.05, 0)], 0.5)
        self.assertEqual(len(f.tracked_list), 1)

    def test_pure_mahalanobis_config(self):
        """Pure Mahalanobis = mahal_weight=1, iou_weight=0."""
        f = sensor_fusion.Fusion(0, mahal_weight=1.0, iou_weight=0.0)
        self.assertEqual(f.mahal_weight, 1.0)
        self.assertEqual(f.iou_weight,   0.0)
        # Tracks should still spawn / match correctly under this regime.
        f.processDetectionFrame(0.0, [make_det(0, 0)], 0.5)
        self.assertEqual(len(f.tracked_list), 1)

    def test_far_detection_does_not_match_under_strict_gate(self):
        """A detection 100 m from the existing track should never match,
        provided we use the appropriate gate for the weighting:
          - hybrid / pure-Mahal: mahal_gate (chi² 13.82)
          - pure-IoU:           iou_gate (e.g. 0.01 like AB3DMOT)
        """
        configs = [
            dict(mahal_weight=0.6, iou_weight=0.4, mahal_gate=13.82, iou_gate=0.0),
            dict(mahal_weight=1.0, iou_weight=0.0, mahal_gate=13.82, iou_gate=0.0),
            dict(mahal_weight=0.0, iou_weight=1.0, mahal_gate=1e9,   iou_gate=0.01),
        ]
        for cfg in configs:
            f = sensor_fusion.Fusion(0, **cfg)
            f.processDetectionFrame(0.0, [make_det(0, 0)], 0.5)
            f.processDetectionFrame(0.1, [make_det(100.0, 0)], 0.5)
            self.assertEqual(len(f.tracked_list), 2,
                             f"cfg {cfg} should not match far det")

    def test_iou_gate_drops_zero_overlap(self):
        """With an IoU gate set, a detection that has zero overlap with
        any track (after the matching gate's bbox-inflation step) gets
        rejected even with a loose Mahalanobis gate."""
        f = sensor_fusion.Fusion(0, mahal_weight=0.0, iou_weight=1.0,
                                 mahal_gate=1e9, iou_gate=0.01)
        f.processDetectionFrame(0.0, [make_det(0, 0)], 0.5)
        # Detection 30 m away — well outside any covariance-inflated bbox
        f.processDetectionFrame(0.1, [make_det(30.0, 0)], 0.5)
        self.assertEqual(len(f.tracked_list), 2,
                         "iou_gate should reject non-overlapping match")


# ============================================================================
# 7. Phase B.5 — inactive-track preservation
# ============================================================================

class TestPhaseB5InactivePreservation(unittest.TestCase):

    def test_collapse_with_preservation_off_deletes(self):
        """Default (preservation off): S_t collapse → track deleted."""
        f = sensor_fusion.Fusion(0, lifecycle_mode="log_odds",
                                 enable_inactive_preservation=False)
        f.processDetectionFrame(0.0, [make_det(0, 0, p_tp=0.95)], 0.5)
        f.fuseDetectionFrame(0.0)
        for k in range(50):
            f.processDetectionFrame(0.1 * (k + 1), [], 50.0)
            f.fuseDetectionFrame(0.1 * (k + 1))
        self.assertEqual(len(f.tracked_list), 0)

    def test_collapse_with_preservation_marks_inactive(self):
        """B.5 ON: S_t collapse → track parked inactive, stays in list."""
        f = sensor_fusion.Fusion(0, lifecycle_mode="log_odds",
                                 enable_inactive_preservation=True,
                                 inactive_grace_s=2.0)
        f.processDetectionFrame(0.0, [make_det(0, 0, p_tp=0.95)], 0.5)
        f.fuseDetectionFrame(0.0)
        # Pile on misses: drives S_t < kill but within grace_s
        for k in range(10):
            f.processDetectionFrame(0.1 * (k + 1), [], 50.0)
            f.fuseDetectionFrame(0.1 * (k + 1))
        self.assertEqual(len(f.tracked_list), 1)
        self.assertTrue(f.tracked_list[0].is_inactive)
        # Inactive tracks don't emit
        result, dets, _, _ = f.fuseDetectionFrame(1.5)
        self.assertEqual(len(result), 0)

    def test_inactive_track_revived_on_match_keeps_id(self):
        """B.5 ON: detection matching an inactive track restores same ID."""
        f = sensor_fusion.Fusion(0, lifecycle_mode="log_odds",
                                 enable_inactive_preservation=True,
                                 inactive_grace_s=5.0)
        f.processDetectionFrame(0.0, [make_det(0, 0, p_tp=0.95)], 0.5)
        f.fuseDetectionFrame(0.0)
        original_id = f.tracked_list[0].id
        # Crash S_t with a long miss streak
        for k in range(15):
            f.processDetectionFrame(0.1 * (k + 1), [], 50.0)
            f.fuseDetectionFrame(0.1 * (k + 1))
        self.assertTrue(f.tracked_list[0].is_inactive)
        # Now a fresh detection lands near the inactive track's last position
        f.processDetectionFrame(2.0, [make_det(0.1, 0, p_tp=0.9)], 50.0)
        f.fuseDetectionFrame(2.0)
        # Same ID restored, no new track spawned, revive_count incremented
        self.assertEqual(len(f.tracked_list), 1)
        self.assertEqual(f.tracked_list[0].id, original_id)
        self.assertFalse(f.tracked_list[0].is_inactive)
        self.assertEqual(f.tracked_list[0].revive_count, 1)
        self.assertGreater(f.tracked_list[0].S_t, f.confirm_log_odds)

    def test_inactive_track_deleted_after_grace_period(self):
        """B.5 ON: inactive track deleted once grace period expires."""
        f = sensor_fusion.Fusion(0, lifecycle_mode="log_odds",
                                 enable_inactive_preservation=True,
                                 inactive_grace_s=2.0)  # generous grace
        f.processDetectionFrame(0.0, [make_det(0, 0, p_tp=0.95)], 0.5)
        f.fuseDetectionFrame(0.0)
        # Crash S_t with enough misses to go inactive (within grace)
        for k in range(8):
            f.processDetectionFrame(0.1 * (k + 1), [], 50.0)
            f.fuseDetectionFrame(0.1 * (k + 1))
        # Track should be inactive but still alive at t=0.8 (grace 2.0)
        self.assertEqual(len(f.tracked_list), 1)
        self.assertTrue(f.tracked_list[0].is_inactive)
        # Now push past inactive_since + grace_s to trigger deletion.
        # inactive_since ≈ 0.4 (when S_t crossed kill); push to t=3.0 (well past).
        for k in range(25):
            f.processDetectionFrame(1.0 + 0.1 * k, [], 50.0)
            f.fuseDetectionFrame(1.0 + 0.1 * k)
        self.assertEqual(len(f.tracked_list), 0)


if __name__ == "__main__":
    unittest.main()
