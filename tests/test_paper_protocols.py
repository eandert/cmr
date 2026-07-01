"""Invariants for the paper's 3-stage metric decomposition + HOTA correctness.

These tests pin the SEMANTICS of the V2V / OG+FP / Ours stages and the
HOTA computation. The differential harness already proves bit-identity vs
DMSTrack's evaluate.py for the shared metrics; this file covers what's
specific to OUR paper (the protocol decomposition + HOTA, which evaluate.py
doesn't compute).

Run with:
  PYTHONPATH=src:scripts:configs:third_party/AB3DMOT:third_party/AB3DMOT/Xinshuo_PyToolbox \
  pytest tests/test_paper_protocols.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "configs"))
AB3DMOT_ROOT = REPO / "third_party" / "AB3DMOT"
if AB3DMOT_ROOT.exists():
    sys.path.insert(0, str(AB3DMOT_ROOT))
    sys.path.insert(0, str(AB3DMOT_ROOT / "Xinshuo_PyToolbox"))

from eval_config import (  # noqa: E402
    BASE_GT_LABEL_DIR,
    WITH_CAV_GT_LABEL_DIR,
    SEQMAP_VAL,
    IOU_3D_GATE,
    TESLA_GT_ID,
    ASTUFF_GT_ID,
)
from gt_integrity import parse_seqmap  # noqa: E402
from metrics.v2v4real_metrics import (  # noqa: E402
    compute_ab3dmot_metrics_iou,
    compute_hota_iou,
)
from verify_eval_stack import build_pooled_tape  # noqa: E402
from paper_metrics import compute_stage  # noqa: E402


# Pin one config that's known to be on disk; tests skip if it's gone.
CANON_CONFIG = "cobevt_Car_val_CoBEVT_S1_EKF_H1"
RESULTS_DIR = AB3DMOT_ROOT / "results" / "v2v4real" / CANON_CONFIG / "data_0"


def _have_canon_config() -> bool:
    return RESULTS_DIR.is_dir() and any(RESULTS_DIR.glob("*.txt"))


pytestmark = pytest.mark.skipif(
    not _have_canon_config(),
    reason=f"canonical config {CANON_CONFIG} not present under {AB3DMOT_ROOT}/results/v2v4real/",
)


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers — synthetic minimal tapes for HOTA invariants
# ═════════════════════════════════════════════════════════════════════════════

def _box(cx, cy, w=2.0, l=4.0, yaw=0.0, h=1.5, z=-1.0, score=0.9):
    """Build one tracker/GT box tuple. Track tuple shape (id, x, y, w, l, yaw, score, h, z)."""
    return (cx, cy, w, l, yaw, score, h, z)


def _synthetic_tape(n_frames: int = 10, n_objs: int = 3,
                    track_ids=None, gt_ids=None, drop_ratio: float = 0.0,
                    seed: int = 0):
    """Tape of n_objs static objects across n_frames; tracker copies GT.

    track_ids/gt_ids: lists of length n_objs; defaults to 1..n_objs for both.
    drop_ratio: per-frame, randomly drop this fraction of tracker rows.
    """
    rng = np.random.default_rng(seed)
    if track_ids is None: track_ids = list(range(1, n_objs + 1))
    if gt_ids is None:    gt_ids = list(range(1, n_objs + 1))
    tape = []
    for f in range(n_frames):
        gts = []
        tracks = []
        for i, (tid, gid) in enumerate(zip(track_ids, gt_ids)):
            cx, cy = 10.0 * (i + 1), 5.0 * (i + 1)
            gts.append((gid, cx, cy, 2.0, 4.0, 0.0, 1.5, -1.0))           # (id,x,y,w,l,yaw,h,z)
            if rng.random() >= drop_ratio:
                tracks.append((tid, cx, cy, 2.0, 4.0, 0.0, 0.9, 1.5, -1.0))  # (id,x,y,w,l,yaw,score,h,z)
        tape.append({"frame_idx": f, "tracks": tracks, "gts": gts})
    return tape


# ═════════════════════════════════════════════════════════════════════════════
#  A. Protocol decomposition invariants (real config)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def seqmap():
    return parse_seqmap(SEQMAP_VAL)


@pytest.fixture(scope="module")
def stages_real(seqmap):
    """Compute all 3 stages once on the canonical config; reused across tests."""
    return {
        "v2v_base":     compute_stage(CANON_CONFIG, "v2v",  BASE_GT_LABEL_DIR.name,     seqmap),
        "ours_base":    compute_stage(CANON_CONFIG, "ours", BASE_GT_LABEL_DIR.name,     seqmap),
        "ours_withcav": compute_stage(CANON_CONFIG, "ours", WITH_CAV_GT_LABEL_DIR.name, seqmap),
    }


def test_v2v_amota_ge_og_fp_on_same_gt(stages_real):
    """V2V ignores FPs; OG+FP counts them. With same GT and same matches,
    V2V's MOTA cannot be lower than OG+FP's (FP only ever hurts MOTA), and
    AMOTA inherits that monotonicity per recall threshold."""
    v2v = stages_real["v2v_base"]["amota"]
    ogfp = stages_real["ours_base"]["amota"]
    assert v2v >= ogfp - 1e-12, (
        f"V2V {v2v:.6f} should be >= OG+FP {ogfp:.6f} on the same base GT "
        f"(FP-ignore can only help or be neutral). Diff = {v2v - ogfp:+.6f}.")


def test_ours_gt_total_exceeds_base_by_ego_count(stages_real, seqmap):
    """The with_cav variant adds exactly 2 ego rows per frame.
    GT_total should rise by 2 × sum(n_frames per seq)."""
    base = stages_real["ours_base"]["gt_total"]
    full = stages_real["ours_withcav"]["gt_total"]
    expected_extra = 2 * sum(end - start + 1 for start, end in seqmap.values())
    assert full - base == expected_extra, (
        f"with_cav added {full - base} GT rows; expected exactly "
        f"{expected_extra} (= 2 × total frames across all seqs).")


def test_ours_amota_finite_and_in_range(stages_real):
    """Headline sanity rail — AMOTA always in [0, 1]."""
    for stage_name, m in stages_real.items():
        amota = m["amota"]
        assert 0.0 <= amota <= 1.0 and math.isfinite(amota), (
            f"{stage_name} AMOTA out of [0,1] or non-finite: {amota}")


def test_hota_in_unit_interval(stages_real):
    """HOTA must be in [0, 1] for every stage; same for DetA, AssA."""
    for stage_name, m in stages_real.items():
        for k in ("hota", "deta", "assa"):
            v = m[k]
            assert 0.0 <= v <= 1.0 and math.isfinite(v), (
                f"{stage_name}.{k} out of [0,1] or non-finite: {v}")


def test_hota_equals_geomean_of_deta_assa(stages_real):
    """HOTA definition: HOTA = sqrt(DetA × AssA). Pin to within 1e-10."""
    for stage_name, m in stages_real.items():
        expected = math.sqrt(m["deta"] * m["assa"])
        assert abs(m["hota"] - expected) < 1e-10, (
            f"{stage_name}: HOTA={m['hota']} but sqrt(DetA·AssA)={expected}; "
            f"|diff|={abs(m['hota'] - expected)}")


# ═════════════════════════════════════════════════════════════════════════════
#  B. HOTA invariants on synthetic tapes
# ═════════════════════════════════════════════════════════════════════════════

def test_hota_perfect_tracker_is_one():
    """Tape where tracker == GT (same ids, same boxes) ⇒ HOTA = 1.0."""
    tape = _synthetic_tape(n_frames=20, n_objs=3)
    h = compute_hota_iou(tape, iou_threshold=IOU_3D_GATE, match_3d=True,
                          ignore_unmatched_fps=False)
    assert h["hota"] == pytest.approx(1.0, abs=1e-9), h
    assert h["deta"] == pytest.approx(1.0, abs=1e-9)
    assert h["assa"] == pytest.approx(1.0, abs=1e-9)


def test_hota_empty_tracker_is_zero():
    """Tape with GT but no tracker rows ⇒ HOTA = 0."""
    base = _synthetic_tape(n_frames=10, n_objs=3)
    empty = [{"frame_idx": f["frame_idx"], "tracks": [], "gts": f["gts"]} for f in base]
    h = compute_hota_iou(empty, iou_threshold=IOU_3D_GATE, match_3d=True,
                         ignore_unmatched_fps=False)
    assert h["hota"] == pytest.approx(0.0, abs=1e-12), h


def test_hota_id_randomization_drops_assa_not_deta():
    """Tracker boxes identical to GT but with EVERY frame relabeling track ids
    to fresh integers ⇒ DetA stays high; AssA collapses; HOTA drops."""
    n_frames, n_objs = 20, 3
    rng = np.random.default_rng(seed=42)
    perfect_tape = _synthetic_tape(n_frames=n_frames, n_objs=n_objs)
    # Build a "id-randomized" tape: same boxes, new track ids per frame.
    rand_tape = []
    next_id = 1000
    for f in perfect_tape:
        relabel = {}
        for t in f["tracks"]:
            relabel[t[0]] = next_id
            next_id += 1
        new_tracks = [(relabel[t[0]],) + t[1:] for t in f["tracks"]]
        rand_tape.append({"frame_idx": f["frame_idx"], "tracks": new_tracks, "gts": f["gts"]})

    h_good = compute_hota_iou(perfect_tape, iou_threshold=IOU_3D_GATE,
                              match_3d=True, ignore_unmatched_fps=False)
    h_rand = compute_hota_iou(rand_tape,    iou_threshold=IOU_3D_GATE,
                              match_3d=True, ignore_unmatched_fps=False)
    # DetA depends only on per-frame matches; unchanged under id relabel.
    assert h_rand["deta"] == pytest.approx(h_good["deta"], abs=1e-9), (
        f"DetA must not depend on track ids; good={h_good['deta']} rand={h_rand['deta']}")
    # AssA should drop dramatically.
    assert h_rand["assa"] < 0.5 * h_good["assa"], (
        f"AssA should drop under id-randomization; good={h_good['assa']} rand={h_rand['assa']}")
    assert h_rand["hota"] < h_good["hota"], "HOTA must decrease when AssA decreases"


def test_hota_partial_recall_lowers_deta():
    """Dropping 50% of tracker rows ⇒ DetA roughly halves; AssA stays high
    (the surviving track-GT associations are still consistent)."""
    perfect = _synthetic_tape(n_frames=20, n_objs=3, drop_ratio=0.0, seed=0)
    half    = _synthetic_tape(n_frames=20, n_objs=3, drop_ratio=0.5, seed=1)
    h_full = compute_hota_iou(perfect, iou_threshold=IOU_3D_GATE, match_3d=True,
                              ignore_unmatched_fps=False)
    h_half = compute_hota_iou(half,    iou_threshold=IOU_3D_GATE, match_3d=True,
                              ignore_unmatched_fps=False)
    assert h_half["deta"] < h_full["deta"], "dropping rows must lower DetA"
    # AssA may dip slightly because the trajectory is broken, but not by a
    # huge margin — pin loosely.
    assert h_half["assa"] >= 0.5 * h_full["assa"]


def test_fp_ignore_does_not_inflate_hota_when_no_fps_present():
    """If the tracker has no unmatched dets, ignore_unmatched_fps must be
    a no-op on HOTA (the two protocols should agree to floating point)."""
    tape = _synthetic_tape(n_frames=20, n_objs=3)
    h_v2v  = compute_hota_iou(tape, iou_threshold=IOU_3D_GATE, match_3d=True,
                              ignore_unmatched_fps=True)
    h_ours = compute_hota_iou(tape, iou_threshold=IOU_3D_GATE, match_3d=True,
                              ignore_unmatched_fps=False)
    assert h_v2v["hota"] == pytest.approx(h_ours["hota"], abs=1e-12)


def test_fp_ignore_lifts_hota_when_fps_present():
    """Add far-from-GT tracker rows (unmatched FPs). v2v ignores them; ours
    counts them and HOTA drops. v2v should equal the no-FP HOTA."""
    base = _synthetic_tape(n_frames=20, n_objs=3)
    # Add 4 far-away FPs per frame
    with_fps = []
    next_id = 5000
    for f in base:
        extra = []
        for k in range(4):
            extra.append((next_id, 500.0 + k, 500.0 + k, 2.0, 4.0, 0.0, 0.5, 1.5, -1.0))
            next_id += 1
        with_fps.append({
            "frame_idx": f["frame_idx"],
            "tracks": list(f["tracks"]) + extra,
            "gts":    f["gts"],
        })
    h_clean   = compute_hota_iou(base,     match_3d=True, iou_threshold=IOU_3D_GATE,
                                  ignore_unmatched_fps=False)
    h_v2v     = compute_hota_iou(with_fps, match_3d=True, iou_threshold=IOU_3D_GATE,
                                  ignore_unmatched_fps=True)
    h_ours    = compute_hota_iou(with_fps, match_3d=True, iou_threshold=IOU_3D_GATE,
                                  ignore_unmatched_fps=False)
    assert h_v2v["hota"] == pytest.approx(h_clean["hota"], abs=1e-9), (
        f"v2v with unmatched FPs should equal clean tape: {h_v2v['hota']} vs {h_clean['hota']}")
    assert h_ours["hota"] < h_clean["hota"], (
        f"ours with unmatched FPs must penalize HOTA: ours={h_ours['hota']} clean={h_clean['hota']}")


def test_hota_invariant_to_translation():
    """Shift all boxes by (Δx, Δy) and HOTA must be unchanged (within fp)."""
    tape = _synthetic_tape(n_frames=15, n_objs=3)
    DX, DY = 25.0, -12.5
    shifted = []
    for f in tape:
        s_tracks = [(tid, cx + DX, cy + DY, w, l, yaw, score, h, z)
                    for (tid, cx, cy, w, l, yaw, score, h, z) in f["tracks"]]
        s_gts = [(gid, cx + DX, cy + DY, w, l, yaw, h, z)
                 for (gid, cx, cy, w, l, yaw, h, z) in f["gts"]]
        shifted.append({"frame_idx": f["frame_idx"], "tracks": s_tracks, "gts": s_gts})
    h0 = compute_hota_iou(tape,    iou_threshold=IOU_3D_GATE, match_3d=True,
                          ignore_unmatched_fps=False)
    h1 = compute_hota_iou(shifted, iou_threshold=IOU_3D_GATE, match_3d=True,
                          ignore_unmatched_fps=False)
    assert h1["hota"] == pytest.approx(h0["hota"], abs=1e-9)
    assert h1["deta"] == pytest.approx(h0["deta"], abs=1e-9)
    assert h1["assa"] == pytest.approx(h0["assa"], abs=1e-9)


def test_amota_mota_recovered_under_perfect_tracker():
    """Perfect tracker on a 3-object, 20-frame tape ⇒ AMOTA == MOTA == 1.0
    under both v2v and ours protocols (no FPs to differ on)."""
    tape = _synthetic_tape(n_frames=20, n_objs=3)
    for proto, ignore in (("v2v", True), ("ours", False)):
        out = compute_ab3dmot_metrics_iou(
            tape, iou_threshold=IOU_3D_GATE, match_3d=True,
            ignore_unmatched_fps=ignore, use_track_avg_score=True,
            ab3dmot_rematching=True,
        )
        assert out["amota"] == pytest.approx(1.0, abs=1e-6), (proto, out["amota"])
        assert out["mota"]  == pytest.approx(1.0, abs=1e-6), (proto, out["mota"])
        assert out["ids_total"] == 0


# ═════════════════════════════════════════════════════════════════════════════
#  C. Ego-augmented GT specifically (sanity checks vs the manifest)
# ═════════════════════════════════════════════════════════════════════════════

def test_with_cav_extra_track_ids_round_trip(seqmap):
    """Build the pooled tape against the with_cav GT and confirm the 90001 /
    90002 ego track ids show up in the GT set (offset by SEQ_ID_OFFSET per seq)."""
    tape = build_pooled_tape(CANON_CONFIG, WITH_CAV_GT_LABEL_DIR.name, seqmap)
    gt_ids = set()
    for f in tape:
        for g in f["gts"]:
            gt_ids.add(g[0])
    # Find ids that, modulo SEQ_ID_OFFSET (1_000_000), equal TESLA_GT_ID/ASTUFF_GT_ID.
    from eval_config import SEQ_ID_OFFSET
    tesla_hits = [gid for gid in gt_ids if gid % SEQ_ID_OFFSET == TESLA_GT_ID]
    astuff_hits = [gid for gid in gt_ids if gid % SEQ_ID_OFFSET == ASTUFF_GT_ID]
    # Expect one tesla + one astuff per sequence.
    assert len(tesla_hits) == len(seqmap), (
        f"expected 1 tesla ego id per seq; got {len(tesla_hits)} of {len(seqmap)}")
    assert len(astuff_hits) == len(seqmap), (
        f"expected 1 astuff ego id per seq; got {len(astuff_hits)} of {len(seqmap)}")
