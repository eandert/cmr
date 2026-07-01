"""Adversarial pin-tests for the evalpy <-> ours equivalence.

These tests guard the two recently-added safeguards in DMSTrack's evaluate.py
that make our scorer in src/metrics/v2v4real_metrics.py bit-identical
(worst |AMOTA diff| <= 1.11e-16 on 5 real tracker outputs):

  1. `--protocol {v2v,ours}` flag    — v2v keeps the paper's min_height FP-ignore;
                                       ours explicitly counts unmatched dets as FP.
  2. `state_mode {clean,paper_compat}` — clean snapshots/restores tracker and GT
                                       attributes between recall-sweep passes so
                                       the sweep is stateless; paper_compat leaves
                                       the published mutation behaviour in place
                                       so the literal paper numbers reproduce
                                       (CoBEVT S1 EKF AMOTA = 0.3716).

Each test below ATTACKS one of these guarantees: regress the AMOTA bit-identity,
the paper-number reproduction, the FP protocol semantics, the IDS counter
correctness, the recall-threshold mechanics, and the pooled-tape integrity.

Run with:
    pytest tests/test_evaluator_equivalence.py -v
"""
from __future__ import annotations

import os
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest


# ─── sys.path setup (mirrors verify_eval_stack.py + existing tests) ──────────
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

AB3DMOT_ROOT = REPO / "third_party" / "AB3DMOT"
if AB3DMOT_ROOT.exists():
    sys.path.insert(0, str(AB3DMOT_ROOT))
    sys.path.insert(0, str(AB3DMOT_ROOT / "Xinshuo_PyToolbox"))

from eval_config import (  # noqa: E402
    CLEAN_MODE_TOL_AMOTA,
    IOU_3D_GATE,
    SEQ_ID_OFFSET,
    SEQMAP_VAL,
    V2V4REAL_RESULTS_ROOT,
)
from gt_integrity import parse_seqmap  # noqa: E402
from metrics.v2v4real_metrics import (  # noqa: E402
    _count_id_switches_traj,
    _get_thresholds,
    match_frame_3d_iou,
    compute_ab3dmot_metrics_iou,
)


# ─── Per-sequence frame counts (from the canonical seqmap) ────────────────────
# verify_eval_stack only uses parse_seqmap, but several adversarial tests below
# need the per-sequence length to bound frame_idx − seq_offset.
def _seq_to_num_frames() -> Dict[str, int]:
    seqmap = parse_seqmap(SEQMAP_VAL)
    return {seq: (end - start + 1) for seq, (start, end) in seqmap.items()}


SEQ_TO_NUM_FRAMES = _seq_to_num_frames() if SEQMAP_VAL.exists() else {}


# ─── Result SHAs used by real-data tests ──────────────────────────────────────
COBEVT_S1_SHA = "cobevt_Car_val_CoBEVT_S1_EKF_H1"
PP_DMSTRACK_S1_SHA = "pp_dmstrack_per_cav_Car_val_DMS_S1_EKF_H1"


def _have_sha(sha: str) -> bool:
    return (V2V4REAL_RESULTS_ROOT / sha / "data_0").exists()


def _skip_if_missing(sha: str) -> None:
    if not _have_sha(sha):
        pytest.skip(f"missing result_sha on disk: {sha}")


# ─── Lazy import of the harness (only when real-data tests run) ───────────────
def _harness():
    """Return the verify_eval_stack module (lazy, after sys.path is set up)."""
    import verify_eval_stack  # noqa: WPS433  (only available with REPO/scripts on path)
    return verify_eval_stack


def _call_evalpy(sha: str, protocol: str, state_mode: str) -> Dict[str, float]:
    """Run evaluate.py's evaluate() in-process; mirror verify_eval_stack.score_evalpy.

    Caller is responsible for any module-state isolation (importlib.reload),
    which we do INSIDE this helper so two consecutive calls produce independent
    trackingEvaluation instances (the published mutating side effect is on
    instance attributes, not module state, but reload also clears any global
    caches inside evaluate.py).
    """
    cwd = os.getcwd()
    os.chdir(str(AB3DMOT_ROOT))
    try:
        from scripts.KITTI import evaluate as ev
        importlib.reload(ev)
        mail = ev.mailpy.Mail("")
        out = ev.evaluate(
            sha, mail, "1", True, False, IOU_3D_GATE,
            evaluate_v2v4real=True, seq_eval_mode="all",
            v2v4real_split="val", protocol=protocol,
            gt_label_subdir=None, state_mode=state_mode,
        )
    finally:
        os.chdir(cwd)
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  A.  Clean-mode bit-identity
# ═════════════════════════════════════════════════════════════════════════════

# We pin much tighter than CLEAN_MODE_TOL_AMOTA (the harness default 1e-5). Once
# evaluate.py's state restoration is in place the two paths take the exact same
# numeric route and the residual lives at the ulp scale (~1e-16). Locking 1e-12
# is still 4 orders of magnitude looser than what we measured — but loud enough
# to fail if anyone re-introduces a non-trivial side-effect contamination.
BIT_IDENTICAL_TOL = 1e-12


def test_clean_mode_bit_identical_cobevt():
    """A.1 — CoBEVT S1 + v2v + clean: AMOTA must match evalpy to <= 1e-12."""
    _skip_if_missing(COBEVT_S1_SHA)
    h = _harness()
    seqmap = parse_seqmap(SEQMAP_VAL)
    r = h.compare(
        COBEVT_S1_SHA, "v2v", "v2v4real_val_label",
        seqmap, convex_hull=True, state_mode="clean",
    )
    diff = abs(r["amota_diff"])
    assert diff <= BIT_IDENTICAL_TOL, (
        f"CoBEVT S1 clean-mode AMOTA diverged: ours={r['amota_ours']:.18f} "
        f"evalpy={r['amota_evalpy']:.18f} |diff|={diff:.3e} > {BIT_IDENTICAL_TOL:g}"
    )


def test_clean_mode_bit_identical_pp_dmstrack():
    """A.2 — pp_dmstrack S1 EKF + v2v + clean: AMOTA must match evalpy to <= 1e-12."""
    _skip_if_missing(PP_DMSTRACK_S1_SHA)
    h = _harness()
    seqmap = parse_seqmap(SEQMAP_VAL)
    r = h.compare(
        PP_DMSTRACK_S1_SHA, "v2v", "v2v4real_val_label",
        seqmap, convex_hull=True, state_mode="clean",
    )
    diff = abs(r["amota_diff"])
    assert diff <= BIT_IDENTICAL_TOL, (
        f"pp_dmstrack S1 clean-mode AMOTA diverged: ours={r['amota_ours']:.18f} "
        f"evalpy={r['amota_evalpy']:.18f} |diff|={diff:.3e} > {BIT_IDENTICAL_TOL:g}"
    )


def test_clean_mode_idempotent():
    """A.3 — two consecutive evalpy(... state_mode='clean') calls give identical AMOTA.

    The state-restoration must be complete: any leftover mutation on tracker
    .score/.valid/.corners_3d_cam or GT .tracker/.id_switch/.fragmentation/
    .ignored/.valid/.corners_3d_cam would alter the second pass's metrics.
    """
    _skip_if_missing(COBEVT_S1_SHA)
    out1 = _call_evalpy(COBEVT_S1_SHA, "v2v", "clean")
    out2 = _call_evalpy(COBEVT_S1_SHA, "v2v", "clean")
    a1, a2 = float(out1["AMOTA"]), float(out2["AMOTA"])
    diff = abs(a1 - a2)
    assert diff <= BIT_IDENTICAL_TOL, (
        f"clean-mode evalpy() not idempotent: run1 AMOTA={a1:.18f} "
        f"run2 AMOTA={a2:.18f} |diff|={diff:.3e}"
    )


# ═════════════════════════════════════════════════════════════════════════════
#  B.  Paper-compat preservation
# ═════════════════════════════════════════════════════════════════════════════

PAPER_AMOTA_COBEVT_S1 = 0.3716     # CoBEVT S1 EKF, DMSTrack paper, v2v + paper_compat
PAPER_NUM_TOL = 1e-4


def test_paper_compat_reproduces_0_3716():
    """B.4 — paper_compat must reproduce the published CoBEVT S1 AMOTA = 0.3716."""
    _skip_if_missing(COBEVT_S1_SHA)
    out = _call_evalpy(COBEVT_S1_SHA, "v2v", "paper_compat")
    amota = float(out["AMOTA"])
    diff = abs(amota - PAPER_AMOTA_COBEVT_S1)
    assert diff <= PAPER_NUM_TOL, (
        f"paper_compat broken: AMOTA={amota:.6f} expected={PAPER_AMOTA_COBEVT_S1:.6f} "
        f"|diff|={diff:.3e} > {PAPER_NUM_TOL:g} "
        f"(the published mutation behaviour must remain intact under paper_compat)"
    )


def test_paper_compat_differs_from_clean_on_pp_dmstrack():
    """B.5 — paper_compat AMOTA != clean AMOTA on pp_dmstrack S1 EKF.

    The documented contamination magnitude is ~0.0037 (sufficient to flip
    which tracks pass per-threshold filtering). If paper_compat ever stops
    being a different code path, the regression will surface here.
    """
    _skip_if_missing(PP_DMSTRACK_S1_SHA)
    out_clean = _call_evalpy(PP_DMSTRACK_S1_SHA, "v2v", "clean")
    out_paper = _call_evalpy(PP_DMSTRACK_S1_SHA, "v2v", "paper_compat")
    a_clean = float(out_clean["AMOTA"])
    a_paper = float(out_paper["AMOTA"])
    delta = abs(a_clean - a_paper)
    # Documented gap is ~0.0037. We require it to be at least 1e-4 (so a
    # silent "fix" that collapses paper_compat onto clean trips this hard),
    # and at most 0.02 (a wider gap would mean something else regressed).
    assert delta >= 1e-4, (
        f"paper_compat collapsed onto clean: clean AMOTA={a_clean:.6f} "
        f"paper_compat AMOTA={a_paper:.6f} delta={delta:.6f} "
        f"(expected ~0.0037 contamination magnitude)"
    )
    assert delta <= 0.02, (
        f"paper_compat/clean delta unexpectedly large: clean={a_clean:.6f} "
        f"paper={a_paper:.6f} delta={delta:.6f} > 0.02 (something else regressed)"
    )


# ═════════════════════════════════════════════════════════════════════════════
#  C.  Protocol semantics
# ═════════════════════════════════════════════════════════════════════════════

def _track_tuple(tid, cx, cy, w, l, yaw, score, h, z_center):
    """Tape tuple for match_frame_3d_iou: h@7, z_center@8."""
    return (tid, cx, cy, w, l, yaw, score, h, z_center)


def _gt_tuple(gid, cx, cy, w, l, yaw, h, z_center):
    """GT tape tuple for match_frame_3d_iou: h@6, z_center@7."""
    return (gid, cx, cy, w, l, yaw, h, z_center)


def test_v2v_ignores_fps_ours_counts():
    """C.6 — synthetic single-frame tape: 5 tracks (1 matched, 4 unmatched).

    Under v2v (ignore_unmatched_fps=True) FP must be 0; under ours, FP must
    be 4. TP must be identical (1) under both protocols.
    """
    # One GT, well separated; 5 tracks; only track #1 overlaps the GT.
    gt = _gt_tuple(gid=1, cx=0.0, cy=0.0, w=2.0, l=4.0, yaw=0.0, h=1.5, z_center=0.0)
    matched_track = _track_tuple(
        tid=100, cx=0.0, cy=0.0, w=2.0, l=4.0, yaw=0.0, score=0.9, h=1.5, z_center=0.0,
    )
    # Place the 4 unmatched tracks far away so IoU=0 (well outside 0.25 gate).
    unmatched = [
        _track_tuple(tid=200 + k, cx=50.0 + 10.0 * k, cy=50.0 + 10.0 * k,
                     w=2.0, l=4.0, yaw=0.0, score=0.8, h=1.5, z_center=0.0)
        for k in range(4)
    ]
    tape = [{
        "frame_idx": 0,
        "tracks": [matched_track] + unmatched,
        "gts": [gt],
    }]

    res_v2v = compute_ab3dmot_metrics_iou(
        tape, iou_threshold=IOU_3D_GATE, match_3d=True, convex_hull=True,
        ignore_unmatched_fps=True, use_track_avg_score=True,
        ab3dmot_rematching=True,
    )
    res_ours = compute_ab3dmot_metrics_iou(
        tape, iou_threshold=IOU_3D_GATE, match_3d=True, convex_hull=True,
        ignore_unmatched_fps=False, use_track_avg_score=True,
        ab3dmot_rematching=True,
    )
    assert int(res_v2v["tp_total"]) == 1, f"v2v TP expected 1, got {res_v2v['tp_total']}"
    assert int(res_ours["tp_total"]) == 1, f"ours TP expected 1, got {res_ours['tp_total']}"
    assert int(res_v2v["fp_total"]) == 0, (
        f"v2v protocol must ignore unmatched FPs: FP got {res_v2v['fp_total']}, expected 0"
    )
    assert int(res_ours["fp_total"]) == 4, (
        f"ours protocol must count all 4 unmatched dets as FP, got {res_ours['fp_total']}"
    )


def test_zero_2d_box_assertion_fires_on_nonzero(tmp_path, monkeypatch):
    """C.7 — evalpy must raise ValueError mentioning 'nonzero 2D box' for any
    tracker row whose 2D bbox is not all-zero (V2V4Real is LiDAR-only)."""
    _skip_if_missing(COBEVT_S1_SHA)  # need the seqmap-named seq for evalpy to load

    # Write a fake result tree alongside the real ones so evaluate.py can find it
    # via its hardcoded ./results/v2v4real prefix. Use a clearly-fake suffix and
    # clean up afterwards.
    fake_sha = f"_pytest_nonzero_2dbox_{os.getpid()}"
    fake_root = V2V4REAL_RESULTS_ROOT / fake_sha
    fake_data = fake_root / "data_0"
    fake_data.mkdir(parents=True, exist_ok=False)

    # Need files for every sequence the seqmap enumerates; only seq 0000 carries
    # the offending row, the rest are empty (valid KITTI MOT — no detections).
    try:
        for seq in sorted(parse_seqmap(SEQMAP_VAL).keys()):
            f = fake_data / f"{seq}.txt"
            if seq == "0000":
                # Single row: frame=0 id=1 Car, with y2=10 (nonzero 2D box).
                # KITTI MOT format: frame id type trunc occ alpha x1 y1 x2 y2
                #                   h w l X Y Z ry [score]
                f.write_text(
                    "0 1 Car 0 0 0 0 0 0 10 1.5 2.0 4.0 0.0 0.0 5.0 0.0 0.9\n"
                )
            else:
                f.write_text("")  # empty results for other seqs

        with pytest.raises(ValueError) as exc_info:
            _call_evalpy(fake_sha, "v2v", "clean")
        msg = str(exc_info.value)
        assert "nonzero 2D box" in msg, (
            f"expected ValueError to mention 'nonzero 2D box'; got: {msg!r}"
        )
    finally:
        # Cleanup the entire fake result tree, even on failure.
        import shutil
        shutil.rmtree(fake_root, ignore_errors=True)


# ═════════════════════════════════════════════════════════════════════════════
#  D.  IDS counter correctness (regression for the 211 -> 22 fix)
# ═════════════════════════════════════════════════════════════════════════════

def test_count_ids_no_switches_when_gap_between_appearances():
    """D.8 — GT matched at f0=A, present-but-unmatched at f1 (-1), matched f2=B.

    evalpy's guard `g[f-1] != -1` makes this 0 switches (the unmatched gap
    breaks the chain). Our _count_id_switches_traj must agree.
    """
    traj = {1: [1001, -1, 1002]}  # A=1001, gap=-1, B=1002
    assert _count_id_switches_traj(traj) == 0, (
        f"unmatched gap must break the IDS chain (g[f-1] == -1 guard), "
        f"but got {_count_id_switches_traj(traj)} switches"
    )


def test_count_ids_switches_only_consecutive_matched():
    """D.9 — Three matched-frame patterns:

       A,A,B   -> 1 switch (B != A at last position, prev was A != -1)
       A,B,A   -> 2 switches (A->B at f1, B->A at f2)
       A,A,A   -> 0 switches
    """
    assert _count_id_switches_traj({1: [10, 10, 20]}) == 1
    assert _count_id_switches_traj({1: [10, 20, 10]}) == 2
    assert _count_id_switches_traj({1: [10, 10, 10]}) == 0


def test_count_ids_long_run_then_relabel_counts_one_switch():
    """D.10 — GT matched at frames 0..9 as tid=A, gap, then 20..29 as tid=B.

    NOTE: this test's earlier draft expected 0 switches (reasoning from the
    FRAME-LEVEL `last_ids` mechanism at evaluate.py:836-853). That mechanism
    only updates a per-frame *local* counter that is never written back to
    `self.id_switches`. The AUTHORITATIVE per-trajectory loop at lines
    902-915 operates on `seq_trajectories[gid]` — exactly the list of
    present-frame tids we pass to `_count_id_switches_traj` — and returns
    1 switch at the A→B boundary. Our function matches that authoritative
    count. Empirical confirmation: the multi-config differential gives
    1.11e-16 worst |AMOTA diff| across 5 real tracker outputs, so the
    two counters agree on real data.
    """
    A, B = 1001, 1002
    traj_long_gap = {1: [A] * 10 + [B] * 10}
    ids = _count_id_switches_traj(traj_long_gap)
    assert ids == 1, (
        f"present-frames trajectory [A]*10 + [B]*10 must count exactly one "
        f"A→B switch under the trajectory-loop semantics evalpy reports as "
        f"self.id_switches, but got {ids}."
    )


# ═════════════════════════════════════════════════════════════════════════════
#  E.  Threshold/recall mechanics
# ═════════════════════════════════════════════════════════════════════════════

def _evalpy_get_thresholds():
    """Return (callable, num_sample_pts) for evalpy's getThresholds.

    Builds a trackingEvaluation object in 'v2v' + 'clean' mode just for access
    to its getThresholds method (which is stateless / does not touch self).
    """
    cwd = os.getcwd()
    os.chdir(str(AB3DMOT_ROOT))
    try:
        from scripts.KITTI import evaluate as ev
        importlib.reload(ev)
        mail = ev.mailpy.Mail("")
        e = ev.trackingEvaluation(
            t_sha="__test_stub__", mail=mail, cls="car",
            eval_3diou=True, eval_2diou=False, num_hypo=1, thres=IOU_3D_GATE,
            evaluate_v2v4real=True, seq_eval_mode="all", v2v4real_split="val",
            protocol="v2v", state_mode="clean",
        )
        return e.getThresholds, ev.num_sample_pts
    finally:
        os.chdir(cwd)


def test_get_thresholds_matches_evalpy_bit_for_bit():
    """E.11 — Same scores + num_gt → both implementations return identical lists."""
    rng = np.random.default_rng(0xCAFE)
    tp_scores = rng.uniform(0.0, 1.0, size=200).tolist()
    num_gt = 250

    ours_thresh, ours_recalls = _get_thresholds(tp_scores, num_gt, num_sample_pts=41)

    ev_get, n_sample = _evalpy_get_thresholds()
    ev_thresh, ev_recalls = ev_get(list(tp_scores), num_gt, num_sample_pts=41)

    assert len(ours_thresh) == len(ev_thresh), (
        f"threshold list length mismatch: ours={len(ours_thresh)} evalpy={len(ev_thresh)}"
    )
    if len(ours_thresh) > 0:
        max_t_diff = max(abs(float(a) - float(b)) for a, b in zip(ours_thresh, ev_thresh))
        max_r_diff = max(abs(float(a) - float(b)) for a, b in zip(ours_recalls, ev_recalls))
        assert max_t_diff == 0.0, (
            f"getThresholds threshold list diverges: max|diff|={max_t_diff:.3e}"
        )
        assert max_r_diff == 0.0, (
            f"getThresholds recall list diverges: max|diff|={max_r_diff:.3e}"
        )


def test_get_thresholds_handles_score_ties():
    """E.12 — Tied scores [0.9]*100 + [0.5]*100, num_gt=200: identical output."""
    tp_scores = [0.9] * 100 + [0.5] * 100
    num_gt = 200

    ours_thresh, ours_recalls = _get_thresholds(tp_scores, num_gt, num_sample_pts=41)
    ev_get, _ = _evalpy_get_thresholds()
    ev_thresh, ev_recalls = ev_get(list(tp_scores), num_gt, num_sample_pts=41)

    assert len(ours_thresh) == len(ev_thresh), (
        f"tied-scores threshold length mismatch: ours={len(ours_thresh)} "
        f"evalpy={len(ev_thresh)}"
    )
    for i, (a, b) in enumerate(zip(ours_thresh, ev_thresh)):
        assert float(a) == float(b), (
            f"tied-scores threshold[{i}] diverges: ours={a} evalpy={b}"
        )
    for i, (a, b) in enumerate(zip(ours_recalls, ev_recalls)):
        assert float(a) == float(b), (
            f"tied-scores recall[{i}] diverges: ours={a} evalpy={b}"
        )


def test_get_thresholds_handles_more_gt_than_tp():
    """E.13 — Sparse TP set (5 entries) against large num_gt (200): identical output.

    Recall caps at 5/200; both implementations must agree on the (likely short)
    returned threshold list.
    """
    tp_scores = [0.95, 0.80, 0.60, 0.40, 0.20]
    num_gt = 200

    ours_thresh, ours_recalls = _get_thresholds(tp_scores, num_gt, num_sample_pts=41)
    ev_get, _ = _evalpy_get_thresholds()
    ev_thresh, ev_recalls = ev_get(list(tp_scores), num_gt, num_sample_pts=41)

    assert len(ours_thresh) == len(ev_thresh), (
        f"sparse-TP threshold length mismatch: ours={len(ours_thresh)} "
        f"evalpy={len(ev_thresh)}"
    )
    if len(ours_thresh) > 0:
        max_r = max(float(r) for r in ev_recalls)
        # 5 TPs / 200 GT = 0.025 recall ceiling (one bin's worth at most).
        assert max_r <= (5.0 / 200.0) + 1e-12, (
            f"evalpy recall list overruns the {5/200:.4f} ceiling: max={max_r}"
        )
        for i, (a, b) in enumerate(zip(ours_thresh, ev_thresh)):
            assert float(a) == float(b), (
                f"sparse-TP threshold[{i}] diverges: ours={a} evalpy={b}"
            )


# ═════════════════════════════════════════════════════════════════════════════
#  F.  Pooled-tape integrity (the SEQ_ID_OFFSET convention)
# ═════════════════════════════════════════════════════════════════════════════

def test_pooled_tape_no_cross_seq_id_collision():
    """F.14 — After per-sequence offsetting, no two sequences share a track_id
    or gt_id in the pooled tape."""
    _skip_if_missing(COBEVT_S1_SHA)
    h = _harness()
    seqmap = parse_seqmap(SEQMAP_VAL)
    tape = h.build_pooled_tape(COBEVT_S1_SHA, "v2v4real_val_label", seqmap)

    # Reconstruct each entry's seq bucket from its frame_idx (frame_idx is
    # also offset by si * SEQ_ID_OFFSET, so // SEQ_ID_OFFSET yields seq index).
    by_seq_track: Dict[int, set] = {}
    by_seq_gt: Dict[int, set] = {}
    for f in tape:
        si = int(f["frame_idx"]) // SEQ_ID_OFFSET
        track_ids = by_seq_track.setdefault(si, set())
        gt_ids = by_seq_gt.setdefault(si, set())
        for t in f["tracks"]:
            track_ids.add(int(t[0]))
        for g in f["gts"]:
            gt_ids.add(int(g[0]))

    # All-pairs: no two sequences may share any id (track or gt).
    seqs = sorted(by_seq_track.keys() | by_seq_gt.keys())
    for i, a in enumerate(seqs):
        for b in seqs[i + 1:]:
            t_collisions = by_seq_track.get(a, set()) & by_seq_track.get(b, set())
            g_collisions = by_seq_gt.get(a, set()) & by_seq_gt.get(b, set())
            assert not t_collisions, (
                f"track_id collision between seq buckets {a} and {b}: "
                f"{sorted(t_collisions)[:5]}... ({len(t_collisions)} total)"
            )
            assert not g_collisions, (
                f"gt_id collision between seq buckets {a} and {b}: "
                f"{sorted(g_collisions)[:5]}... ({len(g_collisions)} total)"
            )


def test_pooled_tape_frame_indices_strictly_within_seq_offset():
    """F.15 — For each pooled frame, (frame_idx − seq_start_offset) must be
    < the seq's local frame count (no frame_idx leaks into another seq's range)."""
    _skip_if_missing(COBEVT_S1_SHA)
    if not SEQ_TO_NUM_FRAMES:
        pytest.skip("SEQMAP_VAL not parseable — cannot bound per-seq frame counts")

    h = _harness()
    seqmap = parse_seqmap(SEQMAP_VAL)
    tape = h.build_pooled_tape(COBEVT_S1_SHA, "v2v4real_val_label", seqmap)

    # The offset for seq index si is (si + 1) * SEQ_ID_OFFSET (see
    # build_pooled_tape). The list of sorted seq names gives us si.
    seq_list = sorted(seqmap.keys())
    for f in tape:
        fidx = int(f["frame_idx"])
        # Recover seq index from the offset (frame_idx = off + local_fid,
        # off = (si+1)*SEQ_ID_OFFSET, so si = fidx // SEQ_ID_OFFSET - 1).
        si = (fidx // SEQ_ID_OFFSET) - 1
        assert 0 <= si < len(seq_list), (
            f"frame_idx {fidx} resolves to out-of-range seq index {si}"
        )
        seq = seq_list[si]
        n_frames = SEQ_TO_NUM_FRAMES[seq]
        off = (si + 1) * SEQ_ID_OFFSET
        local_fid = fidx - off
        assert 0 <= local_fid < n_frames, (
            f"frame_idx {fidx} (seq {seq}, off {off}) has local_fid {local_fid} "
            f"not in [0, {n_frames})"
        )
