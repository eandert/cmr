#!/usr/bin/env python3
"""Differential oracle: our AMOTA scorer vs DMSTrack's canonical evaluate.py.

The two evaluators must agree to within --tol on the SAME tracker outputs and
the SAME ground truth. evaluate.py is the gold standard (it reproduces the
DMSTrack paper numbers); our scorer in cmr/src/metrics/v2v4real_metrics.py is
the second, independent implementation. Any disagreement is a bug in one of
them — this harness is the forcing function that drives them bit-identical.

Key alignment points baked in here (learned the hard way):
  * evaluate.py pools ALL sequences into ONE global recall sweep — so we build
    ONE pooled tape with per-sequence offsets on frame/track/gt ids (so IDS and
    per-trajectory bookkeeping stay sequence-local while scores pool globally).
  * evaluate.py replaces each track's score with its track-average score
    (id_average_score) → use_track_avg_score=True.
  * evaluate.py for V2V4Real ignores unmatched FPs (protocol v2v); --protocol
    ours counts them.
  * 3D IoU @ 0.25, matched via Hungarian — match_3d=True, convex_hull as flagged.

Usage:
  python scripts/verify_eval_stack.py --configs <result_sha> [...] \
      [--protocol v2v|ours] [--tol 1e-6] [--convex-hull] [--report PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from eval_config import (  # noqa: E402
    AB3DMOT_ROOT,
    KITTI_SCRIPTS_DIR as KITTI_DIR,
    V2V4REAL_RESULTS_ROOT as RESULTS_ROOT,
    SEQ_ID_OFFSET as SEQ_OFFSET,
    SEQMAP_VAL,
    IOU_3D_GATE,
    CLEAN_MODE_TOL_AMOTA,
    BASE_GT_LABEL_DIR,
    WITH_CAV_GT_LABEL_DIR,
    WITH_CAV_MERGED_GT_LABEL_DIR,
)


def _resolve_gt_dir(gt_subdir) -> Path:
    """Map a GT specifier to its on-disk directory.

    Accepts a Path, an absolute path string, or a leaf subdir name. Leaf names
    are first matched against the canonical eval_config aliases (so a callsite
    that says ``"kitti_labels"`` resolves to ``BASE_GT_LABEL_DIR`` even after
    the bytes moved into the canonical input tree); failing that they fall
    back to ``KITTI_DIR / <subdir>`` for backward compat.
    """
    p = Path(gt_subdir) if not isinstance(gt_subdir, Path) else gt_subdir
    if p.is_absolute():
        return p
    leaf = p.name
    if leaf == BASE_GT_LABEL_DIR.name:
        return BASE_GT_LABEL_DIR
    if leaf == WITH_CAV_GT_LABEL_DIR.name:
        return WITH_CAV_GT_LABEL_DIR
    if leaf == WITH_CAV_MERGED_GT_LABEL_DIR.name:
        return WITH_CAV_MERGED_GT_LABEL_DIR
    return KITTI_DIR / leaf
from gt_integrity import parse_seqmap  # noqa: E402
from metrics.v2v4real_metrics import compute_ab3dmot_metrics_iou  # noqa: E402
from score_kitti_tracks_through_run_suite import (  # noqa: E402
    parse_kitti_track_file,
    parse_v2v4real_label_file,
)


def build_pooled_tape(result_sha: str, gt_subdir: str,
                      seqmap: Dict[str, Tuple[int, int]]) -> List[Dict]:
    """One global tape across all sequences, cam-frame, with per-seq id offsets.

    ``gt_subdir`` can be either:
      - an absolute path (used as-is)
      - a relative subdir name resolved under ``KITTI_DIR`` (legacy convention)
      - the leaf name of a canonical GT bucket (resolved via eval_config aliases)

    Track tuple : (id, X_cam, Z_cam, w, l, ry, score, h, Y_cam)
    GT tuple    : (id, X_cam, Z_cam, w, l, ry, h, Y_cam)
    (matches match_frame_3d_iou's expected layout: h at -2, z_center at -1.)
    """
    data_dir = RESULTS_ROOT / result_sha / "data_0"
    label_dir = _resolve_gt_dir(gt_subdir)
    tape: List[Dict] = []
    for si, seq in enumerate(sorted(seqmap)):
        off = (si + 1) * SEQ_OFFSET
        tk = parse_kitti_track_file(data_dir / f"{seq}.txt")
        gt = parse_v2v4real_label_file(label_dir / f"{seq}.txt")
        frames = set(tk) | set(gt)
        for fid in sorted(frames):
            tracks = [
                (tid + off, X, Z, w, l, ry, score, h, Y)
                for (tid, X, Y, Z, h, w, l, ry, score) in tk.get(fid, [])
            ]
            gts = [
                (gid + off, X, Z, w, l, ry, h, Y)
                for (gid, X, Y, Z, h, w, l, ry) in gt.get(fid, [])
            ]
            tape.append({"frame_idx": off + fid, "tracks": tracks, "gts": gts})
    return tape


def score_ours(result_sha: str, protocol: str, gt_subdir: str,
               seqmap: Dict[str, Tuple[int, int]], convex_hull: bool) -> Dict:
    tape = build_pooled_tape(result_sha, gt_subdir, seqmap)
    res = compute_ab3dmot_metrics_iou(
        tape,
        iou_threshold=IOU_3D_GATE,
        match_3d=True,
        convex_hull=convex_hull,
        ignore_unmatched_fps=(protocol == "v2v"),
        use_track_avg_score=True,
        ab3dmot_rematching=True,
    )
    return res


def score_evalpy(result_sha: str, protocol: str, gt_subdir: str,
                 state_mode: str = "clean") -> Dict:
    """Call DMSTrack evaluate.py's evaluate() in-process; return its metrics dict.

    state_mode 'clean' (default) restores raw scores between passes so the
    AMOTA sweep is stateless; 'paper_compat' reproduces the published numbers
    with the original score-mutation side effect.
    """
    sys.path.insert(0, str(AB3DMOT_ROOT))
    sys.path.insert(0, str(AB3DMOT_ROOT / "Xinshuo_PyToolbox"))
    import os
    cwd = os.getcwd()
    os.chdir(str(AB3DMOT_ROOT))  # evaluate.py resolves ./results/v2v4real relatively
    try:
        from scripts.KITTI import evaluate as ev
        import importlib
        importlib.reload(ev)
        mail = ev.mailpy.Mail("")
        gt_label_subdir = None if gt_subdir == f"v2v4real_val_label" else gt_subdir
        out = ev.evaluate(result_sha, mail, "1", True, False, IOU_3D_GATE,
                          evaluate_v2v4real=True, seq_eval_mode="all",
                          v2v4real_split="val", protocol=protocol,
                          gt_label_subdir=gt_label_subdir, state_mode=state_mode)
    finally:
        os.chdir(cwd)
    return out  # {'sAMOTA':..., 'AMOTA':..., 'AMOTP':...}


def compare(result_sha: str, protocol: str, gt_subdir: str,
            seqmap: Dict, convex_hull: bool, state_mode: str = "clean") -> Dict:
    ours = score_ours(result_sha, protocol, gt_subdir, seqmap, convex_hull)
    ev = score_evalpy(result_sha, protocol, gt_subdir, state_mode=state_mode)
    a_ours = float(ours["amota"])
    a_ev = float(ev["AMOTA"])
    return {
        "config": result_sha,
        "protocol": protocol,
        "amota_ours": a_ours,
        "amota_evalpy": a_ev,
        "amota_diff": a_ours - a_ev,
        "samota_ours": float(ours["samota"]),
        "samota_evalpy": float(ev["sAMOTA"]),
        "amotp_ours": float(ours["amotp"]),
        "amotp_evalpy": float(ev["AMOTP"]),
        "gt_total": int(ours["gt_total"]),
        "tp_total": int(ours["tp_total"]),
        "fp_total": int(ours["fp_total"]),
        "ids_total": int(ours["ids_total"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True,
                    help="result_sha dirs under results/v2v4real/")
    ap.add_argument("--protocol", choices=["v2v", "ours"], default="v2v")
    ap.add_argument("--gt-subdir", default="v2v4real_val_label")
    ap.add_argument("--tol", type=float, default=CLEAN_MODE_TOL_AMOTA,
                    help="AMOTA tolerance for PASS (default from eval_config).")
    ap.add_argument("--convex-hull", action="store_true",
                    help="use scipy ConvexHull intersection (matches evaluate.py exactly)")
    ap.add_argument("--state-mode", choices=["clean", "paper_compat"], default="clean",
                    help="evaluate.py score-state handling: clean (stateless, default) "
                         "or paper_compat (reproduce published numbers)")
    ap.add_argument("--report", type=str, default=None)
    args = ap.parse_args()

    seqmap = parse_seqmap(SEQMAP_VAL)

    rows: List[Dict] = []
    worst = 0.0
    for sha in args.configs:
        r = compare(sha, args.protocol, args.gt_subdir, seqmap, args.convex_hull,
                    state_mode=args.state_mode)
        rows.append(r)
        worst = max(worst, abs(r["amota_diff"]))
        status = "PASS" if abs(r["amota_diff"]) <= args.tol else "FAIL"
        print(f"[{status}] {sha} ({args.protocol}): "
              f"ours={r['amota_ours']:.6f} evalpy={r['amota_evalpy']:.6f} "
              f"diff={r['amota_diff']:+.6f}  (FP={r['fp_total']} IDS={r['ids_total']} GT={r['gt_total']})")

    print(f"\nworst |AMOTA diff| = {worst:.6e}  (tol {args.tol:g})")

    if args.report:
        _write_report(Path(args.report), rows, args.protocol, args.tol, worst)
        print(f"report → {args.report}")

    return 0 if worst <= args.tol else 1


def _write_report(path: Path, rows: List[Dict], protocol: str, tol: float, worst: float):
    lines = [
        "# Eval-stack audit",
        "",
        f"Protocol: **{protocol}** · tolerance: {tol:g} · worst |AMOTA diff|: {worst:.3e}",
        "",
        "| config | ours AMOTA | evalpy AMOTA | diff | FP | IDS | GT |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['amota_ours']:.6f} | {r['amota_evalpy']:.6f} "
            f"| {r['amota_diff']:+.6f} | {r['fp_total']} | {r['ids_total']} | {r['gt_total']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
