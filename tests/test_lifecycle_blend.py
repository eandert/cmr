"""Unit tests for the per-source LIFECYCLE BLEND in the cooperative tracker.

For mixed-detector configs (astuff=PP, tesla=CPft) the track-level lifecycle
params route per the DETECTION'S source vehicle instead of the legacy
first-match-wins astuff-global scalar. Three blended params:

  1. p_tp_birth_gate   — per-source birth gate (Fusion._birth_gate_for).
  2. confirm_log_odds   — inherited from the birthing detection's source
                          (GlobalTracked.confirm_log_odds), used by emit.
  3. log_lr_miss        — source-count-weighted average over the track's
                          source mix (GlobalTracked.blended_log_lr_miss).

kill_log_odds stays a single global floor (not blended).

The detection's source id is recovered as `vehicle_id // max_id`, where
Fusion.processDetectionFrame encodes
    vehicle_id = source_participant_id * max_id + idx
so passing source_participant_id=1 (astuff) / 2 (tesla) routes a detection to
that source's params. Single-detector / sim runs use source_participant_id=0
and None per-source maps, which degenerate to the scalar path exactly.

These tests are hermetic: no CSV / real model is loaded.

Run: PYTHONPATH=src python -m pytest tests/test_lifecycle_blend.py -v
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import sensor_fusion
import sensor

ASTUFF = 1   # VEHICLE_PARTICIPANT_IDS["astuff"]
TESLA = 2    # VEHICLE_PARTICIPANT_IDS["tesla"]


def make_det(x, y, p_tp=None, score=0.5):
    """Minimal DetectedObject. vehicle_id is set by processDetectionFrame."""
    d = sensor.DetectedObject(
        vehicle_id=None, vehicle_type="car",
        detected_bbox=None,
        centroid=[x, y],
        width=2.0, length=4.0, angle=0.0,
        expected_error_gaussian=None,
        velocity_vector=[0.0, 0.0],
    )
    if p_tp is not None:
        d.p_tp = p_tp
    d.det_score = score
    return d


def submit(f, t, dets, source_participant_id, cleanup=50.0):
    """Submit a per-source detection frame (encodes the source id)."""
    f.processDetectionFrame(t, dets, cleanup, source_participant_id=source_participant_id)


# ============================================================================
# 1. Single-detector degeneracy — per-source maps with identical values
#    reproduce the scalar path EXACTLY.
# ============================================================================

class TestSingleDetectorDegeneracy(unittest.TestCase):
    """Identical per-source values must give bit-identical birth / confirm /
    kill / miss-decay decisions to the scalar-only Fusion."""

    def _run(self, **fusion_kwargs):
        """Drive a fixed scripted scenario, return (S_t trace, emit trace,
        n_tracks trace)."""
        f = sensor_fusion.Fusion(0, lifecycle_mode="log_odds", **fusion_kwargs)
        s_trace, emit_trace, n_trace = [], [], []
        # Frame 0: spawn from source 1.
        submit(f, 0.0, [make_det(0, 0, p_tp=0.97)], ASTUFF)
        res, _, _, _ = f.fuseDetectionFrame(0.0)
        s_trace.append(round(f.tracked_list[0].S_t, 9) if f.tracked_list else None)
        emit_trace.append(len(res))
        n_trace.append(len(f.tracked_list))
        # Frames 1-3: matched updates from source 1.
        for k in range(1, 4):
            submit(f, 0.1 * k, [make_det(0.02 * k, 0, p_tp=0.8)], ASTUFF)
            res, _, _, _ = f.fuseDetectionFrame(0.1 * k)
            s_trace.append(round(f.tracked_list[0].S_t, 9) if f.tracked_list else None)
            emit_trace.append(len(res))
            n_trace.append(len(f.tracked_list))
        # Frames 4-30: pure misses (drive S_t toward kill).
        for k in range(4, 31):
            submit(f, 0.1 * k, [], ASTUFF)
            res, _, _, _ = f.fuseDetectionFrame(0.1 * k)
            s_trace.append(round(f.tracked_list[0].S_t, 9) if f.tracked_list else None)
            emit_trace.append(len(res))
            n_trace.append(len(f.tracked_list))
        return s_trace, emit_trace, n_trace

    def test_identical_maps_match_scalar_path(self):
        gate = 0.5
        confirm = 2.0
        miss = math.log(0.30 / 0.70)
        scalar = self._run(
            p_tp_birth_gate=gate,
            confirm_log_odds=confirm,
            log_lr_miss=miss,
        )
        # Same scalars, but ALSO supply per-source maps with identical values
        # for every source the scenario touches (source 1) plus source 2.
        blended = self._run(
            p_tp_birth_gate=gate,
            confirm_log_odds=confirm,
            log_lr_miss=miss,
            p_tp_birth_gate_by_source={ASTUFF: gate, TESLA: gate},
            confirm_log_odds_by_source={ASTUFF: confirm, TESLA: confirm},
            log_lr_miss_by_source={ASTUFF: miss, TESLA: miss},
        )
        self.assertEqual(scalar, blended,
                         "identical per-source maps must reproduce scalar path exactly")

    def test_none_maps_match_scalar_path(self):
        """None per-source maps (single-detector run) == scalar path."""
        gate, confirm, miss = 0.5, 2.0, math.log(0.30 / 0.70)
        a = self._run(p_tp_birth_gate=gate, confirm_log_odds=confirm, log_lr_miss=miss)
        b = self._run(p_tp_birth_gate=gate, confirm_log_odds=confirm, log_lr_miss=miss,
                      p_tp_birth_gate_by_source=None,
                      confirm_log_odds_by_source=None,
                      log_lr_miss_by_source=None)
        self.assertEqual(a, b)


# ============================================================================
# 2. Per-source birth gate
# ============================================================================

class TestPerSourceBirthGate(unittest.TestCase):

    def test_high_gate_source_rejected_low_gate_source_born(self):
        """A det from the high-gate source below its gate is rejected while an
        equal-score det from the low-gate source is born."""
        f = sensor_fusion.Fusion(
            0, lifecycle_mode="log_odds",
            p_tp_birth_gate=0.5,  # scalar fallback
            p_tp_birth_gate_by_source={ASTUFF: 0.8, TESLA: 0.2},
        )
        # Equal-score (p_tp=0.5) detections, well separated so neither matches.
        submit(f, 0.0, [make_det(0, 0, p_tp=0.5)], ASTUFF)    # 0.5 < 0.8 → reject
        submit(f, 0.0, [make_det(50, 0, p_tp=0.5)], TESLA)    # 0.5 >= 0.2 → born
        born_sources = sorted(t.birth_source_id for t in f.tracked_list)
        self.assertEqual(born_sources, [TESLA])

    def test_scalar_fallback_for_unmapped_source(self):
        """A source absent from the map falls back to the scalar gate."""
        f = sensor_fusion.Fusion(
            0, lifecycle_mode="log_odds",
            p_tp_birth_gate=0.9,
            p_tp_birth_gate_by_source={ASTUFF: 0.1},  # TESLA not present
        )
        submit(f, 0.0, [make_det(0, 0, p_tp=0.5)], ASTUFF)   # 0.5 >= 0.1 → born
        submit(f, 0.0, [make_det(50, 0, p_tp=0.5)], TESLA)   # 0.5 < 0.9 scalar → reject
        born_sources = sorted(t.birth_source_id for t in f.tracked_list)
        self.assertEqual(born_sources, [ASTUFF])


# ============================================================================
# 3. Confirm inheritance — two tracks born from different sources confirm at
#    their own thresholds.
# ============================================================================

class TestConfirmInheritance(unittest.TestCase):

    def test_tracks_confirm_at_own_threshold(self):
        # logit(0.95) ≈ +2.944. Source 1 confirm 1.0 (track emits), source 2
        # confirm 5.0 (track does NOT emit from spawn alone).
        f = sensor_fusion.Fusion(
            0, lifecycle_mode="log_odds",
            confirm_log_odds=2.996,  # scalar fallback (canonical)
            confirm_log_odds_by_source={ASTUFF: 1.0, TESLA: 5.0},
        )
        submit(f, 0.0, [make_det(0, 0, p_tp=0.95)], ASTUFF)
        submit(f, 0.0, [make_det(60, 0, p_tp=0.95)], TESLA)
        # Both tracks exist with S_t = logit(0.95) ≈ 2.944.
        self.assertEqual(len(f.tracked_list), 2)
        by_src = {t.birth_source_id: t for t in f.tracked_list}
        self.assertAlmostEqual(by_src[ASTUFF].confirm_log_odds, 1.0)
        self.assertAlmostEqual(by_src[TESLA].confirm_log_odds, 5.0)
        res, _, _, _ = f.fuseDetectionFrame(0.0)
        # Only the astuff track (confirm 1.0 < 2.944) emits.
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0], by_src[ASTUFF].id)

    def test_confirm_falls_back_to_scalar_when_unmapped(self):
        f = sensor_fusion.Fusion(
            0, lifecycle_mode="log_odds",
            confirm_log_odds=1.0,
            confirm_log_odds_by_source={ASTUFF: 5.0},  # TESLA unmapped → scalar 1.0
        )
        submit(f, 0.0, [make_det(0, 0, p_tp=0.95)], TESLA)
        t = f.tracked_list[0]
        self.assertAlmostEqual(t.confirm_log_odds, 1.0)


# ============================================================================
# 4. Miss-decay blend — a 3:1 astuff:tesla source mix decays by the
#    0.75/0.25-weighted log_lr_miss.
# ============================================================================

class TestMissDecayBlend(unittest.TestCase):

    def test_three_to_one_source_mix_weighted_decay(self):
        miss_astuff = -0.4
        miss_tesla = -1.2
        f = sensor_fusion.Fusion(
            0, lifecycle_mode="log_odds",
            log_lr_miss=-0.847,  # scalar fallback (unused once mix known)
            log_lr_miss_by_source={ASTUFF: miss_astuff, TESLA: miss_tesla},
            # generous confirm so the track stays around / no revive interference
            confirm_log_odds=10.0,
            kill_log_odds=-100.0,
        )
        # Spawn from astuff (hit #1 astuff).
        submit(f, 0.0, [make_det(0, 0, p_tp=0.9)], ASTUFF)
        f.fuseDetectionFrame(0.0)
        t = f.tracked_list[0]
        # Two more astuff matched updates (hits #2, #3 astuff), tiny nudges.
        submit(f, 0.1, [make_det(0.02, 0, p_tp=0.9)], ASTUFF)
        f.fuseDetectionFrame(0.1)
        submit(f, 0.2, [make_det(0.04, 0, p_tp=0.9)], ASTUFF)
        f.fuseDetectionFrame(0.2)
        # One tesla matched update (hit #1 tesla) — same track.
        submit(f, 0.3, [make_det(0.06, 0, p_tp=0.9)], TESLA)
        f.fuseDetectionFrame(0.3)
        self.assertEqual(len(f.tracked_list), 1)
        self.assertEqual(t.source_hits[ASTUFF], 3)
        self.assertEqual(t.source_hits[TESLA], 1)
        expected_blend = 0.75 * miss_astuff + 0.25 * miss_tesla
        self.assertAlmostEqual(t.blended_log_lr_miss(f.log_lr_miss),
                               expected_blend, places=9)
        # And the decay actually applied on a miss frame equals the blend.
        s_before = t.S_t
        submit(f, 0.4, [], ASTUFF)   # pure miss
        f.fuseDetectionFrame(0.4)
        self.assertAlmostEqual(t.S_t - s_before, expected_blend, places=9)

    def test_single_source_blend_equals_that_source(self):
        miss_astuff = -0.3
        f = sensor_fusion.Fusion(
            0, lifecycle_mode="log_odds",
            log_lr_miss=-0.847,
            log_lr_miss_by_source={ASTUFF: miss_astuff, TESLA: -1.5},
            kill_log_odds=-100.0,
        )
        submit(f, 0.0, [make_det(0, 0, p_tp=0.9)], ASTUFF)
        f.fuseDetectionFrame(0.0)
        t = f.tracked_list[0]
        self.assertAlmostEqual(t.blended_log_lr_miss(f.log_lr_miss),
                               miss_astuff, places=9)

    def test_no_map_uses_global_scalar(self):
        f = sensor_fusion.Fusion(0, lifecycle_mode="log_odds", log_lr_miss=-0.7)
        submit(f, 0.0, [make_det(0, 0, p_tp=0.9)], ASTUFF)
        f.fuseDetectionFrame(0.0)
        t = f.tracked_list[0]
        self.assertAlmostEqual(t.blended_log_lr_miss(f.log_lr_miss), -0.7, places=9)


# ============================================================================
# 5. Source-id recovery helper
# ============================================================================

class TestSourceIdHelper(unittest.TestCase):

    def test_source_id_from_encoded_vehicle_id(self):
        d = make_det(0, 0)
        d.vehicle_id = TESLA * sensor_fusion.max_id + 7
        self.assertEqual(sensor_fusion._source_id_of(d), TESLA)

    def test_source_id_none_vehicle_id_is_zero(self):
        d = make_det(0, 0)
        d.vehicle_id = None
        self.assertEqual(sensor_fusion._source_id_of(d), 0)


if __name__ == "__main__":
    unittest.main()
