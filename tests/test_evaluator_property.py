"""Property-style equivalence tests: our scorer === DMSTrack evaluate.py (clean).

We just proved bit-identical agreement on 5 real tracker outputs (worst
|AMOTA diff| = 1.11e-16). This file extends that coverage to ADVERSARIAL
random tapes: for each generated tape, both
``cmr/src/metrics/v2v4real_metrics.compute_ab3dmot_metrics_iou(...)`` and
DMSTrack's ``scripts/KITTI/evaluate.py::evaluate(..., state_mode='clean')``
must produce the SAME AMOTA to within ``CLEAN_MODE_TOL``.

Why this exists
---------------
Recall sweeps are touchy: a single tie-breaking change, a single FP-ignore
toggle, or a single ``num_sample_pts`` mismatch shifts the answer by 1e-3
or more. The two implementations are completely independent codebases —
they should agree only when their semantics line up across every recall
threshold, every Hungarian assignment, every track-average rebinding,
and every FP-ignore branch. The property tests below exercise tape styles
chosen to surface each of those branches.

Tape styles
-----------
* ``echo``              – tracker output is the GT verbatim, score=1.0 → all matched.
* ``perturbed_echo``    – GT with small XZ+yaw jitter → IoU stays above the gate.
* ``partial_recall``    – drop ~30 % of GT detections → real FN load.
* ``id_swaps``          – relabel ~every-10-frame track row → forces IDS counter.
* ``fp_added``          – GT + K extra unmatched fakes → exercises FP-ignore (v2v).
* ``track_avg_ties``    – every track gets the same score → ties in getThresholds.
* ``score_at_boundary`` – scores clustered on recall discretisation boundaries.
* ``single_long_track`` – one synthetic track for all frames → MT/ML/IDS=0 path.
* ``mostly_missed``     – sparse tracker (~1 row per N frames) → near-empty pass.
* ``score_ramp``        – scores uniform in [0,1] + 50 % match → general stress.

Counter-examples
----------------
If a property test FAILS, the offending tape is preserved at
``cmr/tests/fixtures/property_failures/<style>_<seed>/`` so we can replay
it deterministically. This file does NOT fix bugs it discovers; it
reports.

Speed
-----
Per-test wall time is bounded by evaluate.py's ConvexHull-based 3D-IoU
matching over the tape (~10 s with the default ``TAPE_GT_KEEP_FRAC``;
ours adds ~4 s on top). Total suite wall time on the V2V4Real val GT is
~2 min 40 s for K=15 tapes. No pytest-timeout plugin is installed in this
env, so deadlines are enforced by GT subsampling — each tape carries
~10³ tracker rows, not the full 31 419-row GT.
"""
from __future__ import annotations

import contextlib
import io
import shutil
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

# ── Path setup (test convention from other v2v4real tests in this dir) ────────
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "configs"))

# Break a circular import between sensor.py / sensor_fusion.py the same way
# test_v2v4real_dedup.py does.
import sensor_fusion  # noqa: F401,E402

from eval_config import (  # noqa: E402
    BASE_GT_LABEL_DIR,
    V2V4REAL_RESULTS_ROOT,
    SEQMAP_VAL,
)
from gt_integrity import parse_seqmap  # noqa: E402
from score_kitti_tracks_through_run_suite import (  # noqa: E402
    parse_v2v4real_label_file,
)
from verify_eval_stack import score_ours, score_evalpy  # noqa: E402

# ── Test-wide constants ───────────────────────────────────────────────────────
# Bit-identity target. The historical real-data residual is 1.11e-16; we add a
# little headroom so platform-specific BLAS rounding does not flake. Anything
# above this is a real divergence, not float noise.
CLEAN_MODE_TOL = 1e-12

# Sub-millimetre positional floor applied to every synthetic tracker row.
# evaluate.py's iou_3d goes through ``scipy.spatial.ConvexHull`` on the
# Sutherland–Hodgman clip polygon; when tracker and GT corners are exactly
# coincident the clip returns colinear/degenerate points and ConvexHull
# raises ``ValueError: Points cannot contain NaN``. A jitter on the order
# of 1e-4 m breaks the degeneracy without measurably affecting the
# returned IoU (well below any of the discretisation/sweep effects we
# care about). This is a scipy/Qhull-platform property, not a scorer
# bug — both scorers see the same jittered tape, so the equivalence test
# is unaffected.
COINCIDENCE_FLOOR_M = 1e-4

# Where to preserve failing tapes for regression replay.
FAILURE_FIXTURE_ROOT = REPO / "tests" / "fixtures" / "property_failures"


# ── GT preload (shared across tests; the GT files are pinned by SHA) ──────────
@pytest.fixture(scope="module")
def seqmap() -> Dict[str, Tuple[int, int]]:
    return parse_seqmap(SEQMAP_VAL)


@pytest.fixture(scope="module")
def gt_per_seq() -> Dict[str, Dict[int, List[Tuple]]]:
    """{seq4: {frame: [(tid, X, Y, Z, h, w, l, ry)]}} parsed once for all tests.

    NOTE: This is the FULL real V2V4Real val GT (31 419 rows across 9 sequences).
    evalpy reads the same GT files in its own process; this fixture feeds the
    tape generators. Tape generators that emit one tracker row per GT row are
    expensive (each row → a per-frame ConvexHull IoU call × 41 recall passes
    → minutes per test). Such generators thin to ``TAPE_GT_KEEP_FRAC`` of GT
    so per-test wall time stays in the few-second range.
    """
    return {
        seq_file.stem: parse_v2v4real_label_file(seq_file)
        for seq_file in sorted(BASE_GT_LABEL_DIR.glob("*.txt"))
    }


# Per-tape tracker-row budget knob. Anything above ~10 % blows past the
# "few seconds per test" target on this corpus (full GT = 31 419 rows;
# ConvexHull-based IoU is the bottleneck). 5 % gives ~1.5k tracker rows
# and ~15 s end-to-end per tape.
TAPE_GT_KEEP_FRAC = 0.05


@pytest.fixture
def throwaway_result_sha(request):
    """Yield a unique result_sha and tear down its data_0 dir on completion.

    The dir lives under ``AB3DMOT/results/v2v4real/`` because evaluate.py
    resolves tracker outputs relative to ``AB3DMOT_ROOT`` (chdir at call time).
    """
    sha = f"_proptest_{request.node.name}_{uuid.uuid4().hex[:8]}"
    out_dir = V2V4REAL_RESULTS_ROOT / sha
    (out_dir / "data_0").mkdir(parents=True, exist_ok=True)
    yield sha, out_dir
    # Tear-down only if the test passed; failing tests preserve the tape via
    # ``_capture_fixture`` and we still want the working dir cleaned up so the
    # AB3DMOT results tree stays tidy.
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fmt_tracker_row(frame: int, tid: int, X: float, Y: float, Z: float,
                     h: float, w: float, l: float, ry: float, score: float) -> str:
    """17-col AB3DMOT KITTI-MOT tracker output. 2D bbox forced to 0 (v2v4real)."""
    return (f"{int(frame)} {int(tid)} Car 0 0 0 0 0 0 0 "
            f"{h:.6f} {w:.6f} {l:.6f} {X:.6f} {Y:.6f} {Z:.6f} {ry:.6f} {score:.6f}")


def _write_seq_files(out_dir: Path,
                     rows_by_seq: Dict[str, List[str]]) -> None:
    """Write per-sequence tracker output files (always all 9 — empty if no rows).

    evaluate.py reads every sequence listed in the seqmap, so the file MUST
    exist even when empty. If every file is empty, evaluate.py errors at
    plot-time (IndexError on recall_list); the styles below avoid that by
    always seeding at least a handful of rows.
    """
    data_dir = out_dir / "data_0"
    for seq in [f"{i:04d}" for i in range(9)]:
        lines = rows_by_seq.get(seq, [])
        (data_dir / f"{seq}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def _gt_iter(gt_per_seq: Dict[str, Dict[int, List[Tuple]]]):
    """Yield (seq, frame, (tid, X, Y, Z, h, w, l, ry)) for every GT row."""
    for seq, frames in gt_per_seq.items():
        for fid in sorted(frames):
            for row in frames[fid]:
                yield seq, fid, row


def _subsample_gt(gt_per_seq: Dict[str, Dict[int, List[Tuple]]], rng,
                  keep_frac: float = TAPE_GT_KEEP_FRAC,
                  ) -> Dict[str, Dict[int, List[Tuple]]]:
    """Pre-thin GT to ``keep_frac`` so per-test tape size stays bounded.

    Sub-sampling acts on (seq, frame, row) tuples uniformly. We guarantee at
    least one row per sequence (otherwise evalpy's plot() crashes with an
    IndexError on an empty recall_list) by force-keeping a fallback row.
    """
    thinned: Dict[str, Dict[int, List[Tuple]]] = {}
    for seq, frames in gt_per_seq.items():
        kept_frames: Dict[int, List[Tuple]] = {}
        for fid in sorted(frames):
            keep_rows = [row for row in frames[fid] if rng.random() <= keep_frac]
            if keep_rows:
                kept_frames[fid] = keep_rows
        if not kept_frames:
            # safety: keep the earliest frame's first row
            first_fid = min(frames)
            if frames[first_fid]:
                kept_frames[first_fid] = [frames[first_fid][0]]
        thinned[seq] = kept_frames
    return thinned


def _score_both(sha: str, seqmap, *, protocol: str = "v2v",
                gt_subdir: str = "v2v4real_val_label"):
    """Run ours + evalpy on the throwaway sha. Returns (a_ours, a_ev, ours_full).

    Both scorers run with the ConvexHull-based 3D IoU (``convex_hull=True``)
    because that is the path evaluate.py uses internally. The 5-real-tape
    bit-identity result (worst |diff|=1.11e-16) was obtained with this
    setting; the trapezoid-2D fallback drifts by a few percent and would
    be a different audit.

    evalpy is chatty; we swallow its stdout so pytest -v output stays readable.
    """
    ours = score_ours(sha, protocol, gt_subdir, seqmap, True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev = score_evalpy(sha, protocol, gt_subdir, state_mode="clean")
    if not isinstance(ev, dict):
        pytest.fail(f"evalpy returned non-dict ({type(ev).__name__}); tape too sparse?\n"
                    f"evalpy stdout tail:\n{buf.getvalue()[-1500:]}")
    return float(ours["amota"]), float(ev["AMOTA"]), ours


def _capture_fixture(out_dir: Path, style: str, seed: int, diff: float,
                     a_ours: float, a_ev: float) -> Path:
    """Copy the failing tape into the fixtures dir for regression replay."""
    dst = FAILURE_FIXTURE_ROOT / f"{style}_seed{seed}"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    # Copy the entire data_0 directory plus a small README describing it.
    shutil.copytree(out_dir / "data_0", dst / "data_0")
    (dst / "README.md").write_text(
        f"# Property-test counter-example\n\n"
        f"- style: `{style}`\n- seed: `{seed}`\n"
        f"- ours AMOTA: `{a_ours!r}`\n- evalpy AMOTA: `{a_ev!r}`\n"
        f"- diff:        `{diff!r}`\n- tolerance:   `{CLEAN_MODE_TOL!r}`\n\n"
        f"To replay, copy `data_0/` into `AB3DMOT/results/v2v4real/<sha>/` and\n"
        f"compare via `scripts/verify_eval_stack.py --configs <sha> --state-mode clean`.\n"
    )
    return dst


def _check_equiv(style: str, seed: int, sha: str, out_dir: Path, seqmap, *,
                 protocol: str = "v2v"):
    """Score both ways; on mismatch, capture the tape and fail loudly."""
    a_ours, a_ev, ours_full = _score_both(sha, seqmap, protocol=protocol)
    diff = a_ours - a_ev
    if abs(diff) > CLEAN_MODE_TOL:
        fix = _capture_fixture(out_dir, style, seed, diff, a_ours, a_ev)
        pytest.fail(
            f"[{style} seed={seed}] AMOTA divergence {diff:+.3e} exceeds "
            f"tol {CLEAN_MODE_TOL:g}\n"
            f"  ours   = {a_ours!r}\n"
            f"  evalpy = {a_ev!r}\n"
            f"  FP={ours_full['fp_total']} IDS={ours_full['ids_total']} "
            f"GT={ours_full['gt_total']} TP={ours_full['tp_total']}\n"
            f"  captured fixture: {fix}"
        )


# ── Tape generators ───────────────────────────────────────────────────────────
def _floor_jitter(rng) -> Tuple[float, float, float]:
    """Sub-mm anti-degeneracy jitter for (X, Z, ry) — see COINCIDENCE_FLOOR_M."""
    return (
        float(rng.uniform(-COINCIDENCE_FLOOR_M, COINCIDENCE_FLOOR_M)),
        float(rng.uniform(-COINCIDENCE_FLOOR_M, COINCIDENCE_FLOOR_M)),
        float(rng.uniform(-COINCIDENCE_FLOOR_M, COINCIDENCE_FLOOR_M)),
    )


def _gen_echo(gt_per_seq, rng, score: float = 1.0) -> Dict[str, List[str]]:
    """Perfect tracker: tid = gt_tid, pose ≈ gt_pose (sub-mm jitter), score=const."""
    rows: Dict[str, List[str]] = {}
    for seq, fid, (tid, X, Y, Z, h, w, l, ry) in _gt_iter(gt_per_seq):
        dX, dZ, dRy = _floor_jitter(rng)
        rows.setdefault(seq, []).append(
            _fmt_tracker_row(fid, tid, X + dX, Y, Z + dZ, h, w, l, ry + dRy, score))
    return rows


def _gen_perturbed_echo(gt_per_seq, rng, xy_sigma=0.5, yaw_sigma=0.05) -> Dict[str, List[str]]:
    rows: Dict[str, List[str]] = {}
    for seq, fid, (tid, X, Y, Z, h, w, l, ry) in _gt_iter(gt_per_seq):
        dX = float(rng.normal(0.0, xy_sigma))
        dZ = float(rng.normal(0.0, xy_sigma))
        dRy = float(rng.normal(0.0, yaw_sigma))
        rows.setdefault(seq, []).append(
            _fmt_tracker_row(fid, tid, X + dX, Y, Z + dZ, h, w, l, ry + dRy, 0.9))
    return rows


def _gen_partial_recall(gt_per_seq, rng, keep_p: float = 0.7) -> Dict[str, List[str]]:
    rows: Dict[str, List[str]] = {}
    for seq, fid, (tid, X, Y, Z, h, w, l, ry) in _gt_iter(gt_per_seq):
        if rng.random() > keep_p:
            continue
        dX, dZ, dRy = _floor_jitter(rng)
        rows.setdefault(seq, []).append(
            _fmt_tracker_row(fid, tid, X + dX, Y, Z + dZ, h, w, l, ry + dRy, 0.8))
    return rows


def _gen_id_swaps(gt_per_seq, rng, swap_every: int = 10) -> Dict[str, List[str]]:
    """Every ~swap_every frames, relabel a track to a fresh, non-colliding id."""
    rows: Dict[str, List[str]] = {}
    next_synth = 50_000
    for seq, frames in gt_per_seq.items():
        # Build a per-seq remap that flips at every swap_every-th frame
        # encountered in sorted order.
        relabel: Dict[int, int] = {}
        for k, fid in enumerate(sorted(frames)):
            for (tid, X, Y, Z, h, w, l, ry) in frames[fid]:
                emit_tid = tid
                if k > 0 and k % swap_every == 0 and rng.random() < 0.5:
                    # Force an ID switch for this track on this frame.
                    relabel[tid] = next_synth
                    next_synth += 1
                if tid in relabel:
                    emit_tid = relabel[tid]
                dX, dZ, dRy = _floor_jitter(rng)
                rows.setdefault(seq, []).append(
                    _fmt_tracker_row(fid, emit_tid, X + dX, Y, Z + dZ, h, w, l, ry + dRy, 0.8))
    return rows


def _gen_fp_added(gt_per_seq, rng, n_fps_per_frame: int = 5) -> Dict[str, List[str]]:
    """Echo + K far-away fake tracks per frame so they all stay unmatched."""
    rows = _gen_echo(gt_per_seq, rng, score=0.95)
    fake_tid_base = 80_000
    for seq, frames in gt_per_seq.items():
        for fid in sorted(frames):
            for k in range(n_fps_per_frame):
                # Park the FP at (200 + small, 200 + small) in cam-X/cam-Z so
                # 3D-IoU with any GT is zero. Y kept in a reasonable range.
                fx = 200.0 + float(rng.uniform(0, 50))
                fz = 200.0 + float(rng.uniform(0, 50))
                rows.setdefault(seq, []).append(_fmt_tracker_row(
                    fid, fake_tid_base + k, fx, -1.0, fz,
                    1.6, 2.0, 4.5, 0.0, float(rng.uniform(0.1, 0.9))))
    return rows


def _gen_track_avg_ties(gt_per_seq, rng) -> Dict[str, List[str]]:
    """All tracks share score=0.5. AB3DMOT then assigns the same track-avg
    to every track → getThresholds ties are exercised."""
    rows: Dict[str, List[str]] = {}
    for seq, fid, (tid, X, Y, Z, h, w, l, ry) in _gt_iter(gt_per_seq):
        dX, dZ, dRy = _floor_jitter(rng)
        rows.setdefault(seq, []).append(
            _fmt_tracker_row(fid, tid, X + dX, Y, Z + dZ, h, w, l, ry + dRy, 0.5))
    return rows


def _gen_score_at_boundary(gt_per_seq, rng) -> Dict[str, List[str]]:
    """Scores cluster on the 41-pt recall discretisation boundaries (0.025*k).

    Per-track scores (track avg) sit on boundaries so the sweep's argmax
    transition is exactly where we expect — tiny rebinding errors surface
    as MOTA jumps.
    """
    boundaries = np.arange(0.025, 1.0001, 0.025)
    rows: Dict[str, List[str]] = {}
    for seq, frames in gt_per_seq.items():
        for fid in sorted(frames):
            for (tid, X, Y, Z, h, w, l, ry) in frames[fid]:
                # one boundary per (seq, tid) so the track-avg lands exactly there
                idx = (hash((seq, tid)) % len(boundaries))
                s = float(boundaries[idx])
                dX, dZ, dRy = _floor_jitter(rng)
                rows.setdefault(seq, []).append(
                    _fmt_tracker_row(fid, tid, X + dX, Y, Z + dZ, h, w, l, ry + dRy, s))
    return rows


def _gen_single_long_track(gt_per_seq, rng) -> Dict[str, List[str]]:
    """One synthetic track copying GT id=1 (or the first GT id) per seq.

    Yields exactly one TP track per sequence — MT/ML accounting and IDS=0
    are exercised cleanly.
    """
    rows: Dict[str, List[str]] = {}
    for seq, frames in gt_per_seq.items():
        # pick the GT track id present in the earliest frame
        first_fid = min(frames)
        if not frames[first_fid]:
            continue
        target_tid = frames[first_fid][0][0]
        for fid in sorted(frames):
            for (tid, X, Y, Z, h, w, l, ry) in frames[fid]:
                if tid != target_tid:
                    continue
                dX, dZ, dRy = _floor_jitter(rng)
                rows.setdefault(seq, []).append(
                    _fmt_tracker_row(fid, tid, X + dX, Y, Z + dZ, h, w, l, ry + dRy, 0.9))
    return rows


def _gen_mostly_missed(gt_per_seq, rng, keep_p: float = 0.02) -> Dict[str, List[str]]:
    """Sparse tracker — keep ~2 % of GT to dodge evaluate.py's empty-tape crash.

    The plain "all_missed" tape style requested in the brief crashes
    evaluate.py at plot-time (IndexError on an empty recall_list). We
    approximate it: keep a tiny fraction of GT so the sweep populates
    recall_list with at least one point, and verify the two scorers still
    agree on the near-zero AMOTA.
    """
    rows: Dict[str, List[str]] = {}
    for seq, fid, (tid, X, Y, Z, h, w, l, ry) in _gt_iter(gt_per_seq):
        if rng.random() > keep_p:
            continue
        dX, dZ, dRy = _floor_jitter(rng)
        rows.setdefault(seq, []).append(
            _fmt_tracker_row(fid, tid, X + dX, Y, Z + dZ, h, w, l, ry + dRy, 0.3))
    # Make sure every sequence has ≥ 1 row — otherwise evalpy still chokes.
    for seq, frames in gt_per_seq.items():
        if rows.get(seq):
            continue
        first_fid = min(frames)
        if not frames[first_fid]:
            continue
        (tid, X, Y, Z, h, w, l, ry) = frames[first_fid][0]
        dX, dZ, dRy = _floor_jitter(rng)
        rows.setdefault(seq, []).append(
            _fmt_tracker_row(first_fid, tid, X + dX, Y, Z + dZ, h, w, l, ry + dRy, 0.05))
    return rows


def _gen_score_ramp(gt_per_seq, rng, match_p: float = 0.5) -> Dict[str, List[str]]:
    """Random uniform scores; 50 % of GT echoed (matched), 50 % dropped (FN)."""
    rows: Dict[str, List[str]] = {}
    for seq, fid, (tid, X, Y, Z, h, w, l, ry) in _gt_iter(gt_per_seq):
        if rng.random() > match_p:
            continue
        s = float(rng.uniform(0.0, 1.0))
        dX, dZ, dRy = _floor_jitter(rng)
        rows.setdefault(seq, []).append(
            _fmt_tracker_row(fid, tid, X + dX, Y, Z + dZ, h, w, l, ry + dRy, s))
    return rows


# Tape generators in the order they appear in the brief. Each entry is
# (style_name, fn(gt_per_seq, rng) -> rows_by_seq).
TAPE_STYLES = [
    ("echo",              _gen_echo),
    ("perturbed_echo",    _gen_perturbed_echo),
    ("partial_recall",    _gen_partial_recall),
    ("id_swaps",          _gen_id_swaps),
    ("fp_added",          _gen_fp_added),
    ("track_avg_ties",    _gen_track_avg_ties),
    ("score_at_boundary", _gen_score_at_boundary),
    ("single_long_track", _gen_single_long_track),
    ("mostly_missed",     _gen_mostly_missed),
    ("score_ramp",        _gen_score_ramp),
]


# ── The property test ────────────────────────────────────────────────────────
@pytest.mark.parametrize("style,gen_fn,seed", [
    # K = 15 (one per style + 5 extra ramp/perturbed runs with different seeds)
    *[(name, fn, i) for i, (name, fn) in enumerate(TAPE_STYLES)],
    ("perturbed_echo", _gen_perturbed_echo, 100),
    ("score_ramp",     _gen_score_ramp,     101),
    ("partial_recall", _gen_partial_recall, 102),
    ("fp_added",       _gen_fp_added,       103),
    ("id_swaps",       _gen_id_swaps,       104),
])
def test_property_evaluator_equiv(
    style: str, gen_fn, seed: int,
    gt_per_seq, seqmap, throwaway_result_sha,
):
    """Generate a tape, write it, score it both ways, assert bit-identity.

    Per-test wall time: expected ~15 s (evalpy's ConvexHull-IoU pass over
    a few thousand tracker rows × 41 recall sweeps dominates). No
    pytest-timeout plugin is installed in this env; we cap runtime by
    pre-thinning GT to ``TAPE_GT_KEEP_FRAC`` BEFORE feeding the tape
    generator so each tape carries on the order of 10³ tracker rows.
    """
    sha, out_dir = throwaway_result_sha
    rng = np.random.default_rng(seed=seed)
    gt_small = _subsample_gt(gt_per_seq, rng)
    rows_by_seq = gen_fn(gt_small, rng)
    _write_seq_files(out_dir, rows_by_seq)
    _check_equiv(style, seed, sha, out_dir, seqmap, protocol="v2v")
