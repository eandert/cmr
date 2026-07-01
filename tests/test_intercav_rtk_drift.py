"""Unit tests for scripts/estimate_intercav_rtk_drift.py.

Covers the load-time ass_id co-observation semantics, the ego-cross-observation
translation estimator (incl. common-mode detector-bias cancellation), the OU
RTS smoother (variance reduction + gap fill + mean preservation), and the yaw
diagnostic. These pin the math; the end-to-end numbers are verified by running
the script on the real export.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import estimate_intercav_rtk_drift as drift  # noqa: E402


# --- ego translation: both-direction mean cancels common-mode bias ----------
def test_frame_translation_recovers_drift_and_cancels_common_mode_bias():
    t_true = np.array([0.5, 1.0])
    b = np.array([0.3, -0.2])  # COMMON-mode detector bias on both directions
    # pair1 (astuff sees tesla): T-A = t - b ; pair2 (tesla sees astuff): T-A = t + b
    F = drift.Frame(
        ego_A=[(0.0, 0.0), (5.0, 5.0)],
        ego_T=[tuple((0.0, 0.0) + (t_true - b)), tuple((5.0, 5.0) + (t_true + b))],
    )
    tx, ty, var, n_ego, n_tp = drift.frame_translation(F)
    assert n_ego == 2 and n_tp == 0
    assert np.allclose([tx, ty], t_true, atol=1e-9)  # bias cancels exactly
    assert var > 0


def test_frame_translation_residual_bias_when_asymmetric():
    t_true = np.array([0.5, 1.0])
    bA = np.array([0.4, 0.0]); bT = np.array([0.0, 0.6])
    F = drift.Frame(
        ego_A=[(0.0, 0.0), (5.0, 5.0)],
        ego_T=[tuple((0.0, 0.0) + (t_true - bA)), tuple((5.0, 5.0) + (t_true + bT))],
    )
    tx, ty, *_ = drift.frame_translation(F)
    assert np.allclose([tx, ty], t_true + (bT - bA) / 2.0, atol=1e-9)


def test_frame_translation_single_ego_inflates_variance():
    F = drift.Frame(ego_A=[(0.0, 0.0)], ego_T=[(0.5, 1.0)])
    tx, ty, var, n_ego, _ = drift.frame_translation(F)
    assert n_ego == 1
    assert np.allclose([tx, ty], [0.5, 1.0])
    assert var == pytest.approx(drift.OBS_VAR_FLOOR_M2 * drift.ONE_EGO_VAR_MULT)


def test_frame_translation_thirdparty_fallback():
    F = drift.Frame(tp_A=[(0.0, 0.0), (2.0, 0.0)], tp_T=[(0.3, 0.1), (2.3, 0.1)])
    tx, ty, var, n_ego, n_tp = drift.frame_translation(F)
    assert n_ego == 0 and n_tp == 2
    assert np.allclose([tx, ty], [0.3, 0.1])
    assert var == pytest.approx(drift.OBS_VAR_FLOOR_M2 * drift.TP_FALLBACK_VAR_MULT)


# --- OU RTS smoother --------------------------------------------------------
def test_ou_smoother_reduces_variance_and_preserves_mean():
    rng = np.random.default_rng(0)
    n = 200
    truth = 2.0 + 0.3 * np.sin(np.linspace(0, 6, n))  # slow drift
    noise = rng.normal(0, 0.5, n)
    y = truth + noise
    r = np.full(n, 0.25)
    s = drift.ou_rts_smooth(y, r)
    assert np.var(s - truth) < np.var(y - truth)          # tracks better than raw
    assert abs(np.mean(s) - np.mean(truth)) < 0.1          # mean preserved
    assert np.mean(np.abs(s - truth)) < np.mean(np.abs(y - truth))


def test_ou_smoother_fills_gaps():
    n = 50
    y = np.full(n, 1.0)
    y[20:30] = np.nan                                       # a measurement gap
    r = np.where(np.isfinite(y), 0.1, np.inf)
    s = drift.ou_rts_smooth(y, r)
    assert np.all(np.isfinite(s))
    assert np.allclose(s, 1.0, atol=0.05)                  # mean-reverts across gap


# --- yaw diagnostic ---------------------------------------------------------
def test_kabsch_theta_recovers_known_rotation():
    rng = np.random.default_rng(1)
    A = rng.normal(0, 10, (30, 2))
    th = 0.05
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    T = A @ R.T + np.array([3.0, -2.0])                    # rotation + translation
    assert drift.kabsch_theta(A, T) == pytest.approx(th, abs=1e-6)


# --- build_frames: ass_id co-observation + ego gating -----------------------
def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def test_build_frames_associates_assid_and_gates_ego(tmp_path):
    sd = tmp_path / "test__scen"
    sd.mkdir()
    _write(sd / "ego_state.csv",
           ["timestamp", "vehicle_id", "frame_id", "x", "y", "z", "yaw",
            "vx", "vy", "length", "width", "height"],
           [[0.0, "astuff", 0, 0.0, 0.0, 0, 0, 0, 0, 5, 2, 1.7],
            [0.0, "tesla", 0, 10.0, 0.0, 0, 0, 0, 0, 5, 2, 1.5]])
    _write(sd / "detections.csv",
           ["timestamp", "vehicle_id", "frame_id", "det_idx", "x", "y", "z",
            "yaw", "vx", "vy", "length", "width", "height", "score", "label"],
           [  # astuff detects tesla near (10,0); also a far decoy
            [0.0, "astuff", 0, 0, 10.5, 0.4, 0, 0, 0, 0, 5, 2, 1.5, 0.9, "car"],
            [0.0, "astuff", 0, 1, 50.0, 50.0, 0, 0, 0, 0, 5, 2, 1.5, 0.9, "car"],
            # tesla detects astuff near (0,0)
            [0.0, "tesla", 0, 0, -0.3, 0.2, 0, 0, 0, 0, 5, 2, 1.7, 0.9, "car"]])
    _write(sd / "gt_objects.csv",
           ["timestamp", "vehicle_id", "frame_id", "ass_id", "obj_id",
            "x", "y", "z", "yaw", "length", "width", "height", "label"],
           [[0.0, "tesla", 0, -1, 1, 20.0, 5.0, 0, 0, 4, 2, 1.6, "car"],
            [0.0, "astuff", 0, 1, 7, 20.4, 5.3, 0, 0, 4, 2, 1.6, "car"],
            [0.0, "astuff", 0, -1, 9, 99.0, 99.0, 0, 0, 4, 2, 1.6, "car"]])
    frames = drift.build_frames(sd)
    assert set(frames) == {0}
    F = frames[0]
    # ego: two directions matched within gate
    assert len(F.ego_A) == 2 and len(F.ego_T) == 2
    assert (10.5, 0.4) in F.ego_A and (10.0, 0.0) in F.ego_T          # astuff->tesla
    assert (0.0, 0.0) in F.ego_A and (-0.3, 0.2) in F.ego_T           # tesla->astuff
    # third-party: ass_id=1 on astuff -> tesla obj_id=1; the ass_id=-1 row excluded
    assert F.tp_A == [(20.4, 5.3)] and F.tp_T == [(20.0, 5.0)]


def test_build_frames_drops_ego_match_outside_gate(tmp_path):
    sd = tmp_path / "test__far"
    sd.mkdir()
    _write(sd / "ego_state.csv",
           ["timestamp", "vehicle_id", "frame_id", "x", "y", "z", "yaw",
            "vx", "vy", "length", "width", "height"],
           [[0.0, "astuff", 0, 0.0, 0.0, 0, 0, 0, 0, 5, 2, 1.7],
            [0.0, "tesla", 0, 10.0, 0.0, 0, 0, 0, 0, 5, 2, 1.5]])
    _write(sd / "detections.csv",
           ["timestamp", "vehicle_id", "frame_id", "det_idx", "x", "y", "z",
            "yaw", "vx", "vy", "length", "width", "height", "score", "label"],
           [[0.0, "astuff", 0, 0, 100.0, 0.0, 0, 0, 0, 0, 5, 2, 1.5, 0.9, "car"]])
    _write(sd / "gt_objects.csv",
           ["timestamp", "vehicle_id", "frame_id", "ass_id", "obj_id",
            "x", "y", "z", "yaw", "length", "width", "height", "label"],
           [[0.0, "tesla", 0, -1, 1, 20.0, 5.0, 0, 0, 4, 2, 1.6, "car"]])
    # only a far astuff detection (>6m from tesla); no tesla det; no tp pair
    frames = drift.build_frames(sd)
    assert frames == {}            # nothing within gate, no tp pair -> no frame
