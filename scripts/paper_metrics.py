#!/usr/bin/env python3
"""Paper-ready metric table: AMOTA / MOTA / HOTA for our V2V4Real tracker outputs.

The verifier (`verify_eval_stack.py`) PROVED our scorer == DMSTrack's clean
`evaluate.py` to ~1e-16 on real outputs. This script uses our scorer alone
(since equivalence is locked) to emit the 3-stage paper headline:

  V2V    : FP-ignore   + base GT          — DMSTrack-paper-comparable column
  OG+FP  : FP-counted  + base GT          — isolates the FP-counting penalty
  Ours   : FP-counted  + ego-augmented GT — our paper's headline

HOTA is computed via `compute_hota_iou` (Luiten et al. IJCV 2021) at single
3D-IoU threshold 0.25, matching the V2V4Real convention. evaluate.py does not
report HOTA; ours is the only source.

PARALLELISM. The (config × stage) work items are pure functions over the same
``seqmap``: no global state, no chdir, no module reload. ``--workers N``
spreads them across a ``multiprocessing.Pool``; default is
``min(4, cpu_count - 2)``. Pass ``--workers 1`` for serial execution
(useful for debugging or when running other CPU-heavy jobs alongside).

Usage:
  python scripts/paper_metrics.py --configs SHA1 SHA2 ... [--out PATH] [--workers N]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "configs"))

from eval_config import (   # noqa: E402
    BASE_GT_LABEL_DIR,
    WITH_CAV_GT_LABEL_DIR,
    WITH_CAV_MERGED_GT_LABEL_DIR,
    SEQMAP_VAL,
    IOU_3D_GATE,
)
from gt_integrity import parse_seqmap  # noqa: E402
from metrics.v2v4real_metrics import compute_ab3dmot_metrics_iou, compute_hota_iou  # noqa: E402
from verify_eval_stack import build_pooled_tape  # noqa: E402


# Four stages, each a (protocol, gt_subdir, label) tuple.
STAGES: List[Tuple[str, str, str]] = [
    ("v2v",  BASE_GT_LABEL_DIR.name,            "V2V (FP-ignore, base GT)"),
    ("ours", BASE_GT_LABEL_DIR.name,            "OG+FP (FP-counted, base GT)"),
    ("ours", WITH_CAV_GT_LABEL_DIR.name,        "Ours (FP-counted, +ego GT)"),
    ("ours", WITH_CAV_MERGED_GT_LABEL_DIR.name, "Ours-merged (FP-counted, +ego GT, dup-merged)"),
]


def compute_stage(result_sha: str, protocol: str, gt_subdir: str,
                  seqmap: Dict) -> Dict:
    """Run one stage's metric pass on the pooled tape."""
    tape = build_pooled_tape(result_sha, gt_subdir, seqmap)
    amotas = compute_ab3dmot_metrics_iou(
        tape,
        iou_threshold=IOU_3D_GATE,
        match_3d=True,
        ignore_unmatched_fps=(protocol == "v2v"),
        use_track_avg_score=True,
        ab3dmot_rematching=True,
    )
    hota = compute_hota_iou(
        tape,
        iou_threshold=IOU_3D_GATE,
        match_3d=True,
        ignore_unmatched_fps=(protocol == "v2v"),
    )
    return {
        "amota":    float(amotas["amota"]),
        "amotp":    float(amotas["amotp"]),
        "samota":   float(amotas["samota"]),
        "mota":     float(amotas["mota"]),
        "mt":       float(amotas["mt"]),
        "ml":       float(amotas["ml"]),
        "hota":     float(hota["hota"]),
        "deta":     float(hota["deta"]),
        "assa":     float(hota["assa"]),
        "loca":     float(hota.get("loca", 0.0)),
        "gt_total": int(amotas["gt_total"]),
        "fp_total": int(amotas["fp_total"]),
        "ids":      int(amotas["ids_total"]),
    }


# ── multiprocessing-Pool plumbing ─────────────────────────────────────────────

def _worker_init() -> None:
    """Worker setup: cap each child's BLAS/OpenMP thread fan-out to 1 so
    N workers × M threads doesn't oversubscribe the cores. Only set if the
    user hasn't pinned them already (respect their env)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def _worker_run(task: Tuple[int, int, str, str, str, str, Dict]
                ) -> Tuple[int, int, str, str, Dict]:
    """Pool worker. task = (cfg_idx, stage_idx, sha, protocol, gt_subdir, label, seqmap).
    Returns (cfg_idx, stage_idx, sha, label, metrics) so we can re-order results
    deterministically after the pool returns them out of order.
    """
    cfg_idx, stage_idx, sha, protocol, gt_subdir, label, seqmap = task
    m = compute_stage(sha, protocol, gt_subdir, seqmap)
    return cfg_idx, stage_idx, sha, label, m


def emit_markdown_table(rows: List[Tuple[str, str, Dict]]) -> str:
    out = [
        "# V2V4Real paper metrics — 3-stage decomposition",
        "",
        "All three stages use the same fixed evaluator (`compute_ab3dmot_metrics_iou`,",
        "proven bit-identical to DMSTrack's clean `evaluate.py` at ~1e-16). Only the",
        "protocol and the GT label set differ between stages.",
        "",
        "| config | stage | AMOTA | MOTA | sAMOTA | HOTA | DetA | AssA | GT_total | FP | IDS |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for config, label, m in rows:
        out.append(
            f"| `{config}` | {label} | "
            f"{m['amota']*100:.2f} | {m['mota']*100:.2f} | {m['samota']*100:.2f} | "
            f"{m['hota']*100:.2f} | {m['deta']*100:.2f} | {m['assa']*100:.2f} | "
            f"{m['gt_total']} | {m['fp_total']} | {m['ids']} |"
        )
    return "\n".join(out) + "\n"


def _default_workers() -> int:
    """Default worker count: min(8, cpu_count - 2), with a floor of 1.
    Tuned for the 32-core dev box; bump if you have more cores and memory
    headroom (each worker uses ~500 MB for the pooled tape)."""
    return max(1, min(8, (os.cpu_count() or 8) - 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True,
                    help="result_sha directories under results/v2v4real/")
    ap.add_argument("--out", type=str, default=None,
                    help="markdown output path (also prints to stdout)")
    ap.add_argument("--workers", type=int, default=_default_workers(),
                    help=f"parallel workers (default {_default_workers()}); use 1 for serial")
    args = ap.parse_args()

    seqmap = parse_seqmap(SEQMAP_VAL)

    # Build the full task list: every (config, stage) cross-product.
    tasks: List[Tuple] = []
    for cfg_idx, sha in enumerate(args.configs):
        for stage_idx, (protocol, gt_subdir, label) in enumerate(STAGES):
            tasks.append((cfg_idx, stage_idx, sha, protocol, gt_subdir, label, seqmap))

    n_total = len(tasks)
    n_workers = max(1, min(args.workers, n_total))
    print(f"[paper_metrics] {n_total} (config × stage) work items, {n_workers} worker(s)",
          flush=True)

    # Collect results indexed by (cfg_idx, stage_idx) so we can re-order
    # the markdown table back into the user-requested config order regardless
    # of completion order.
    results: Dict[Tuple[int, int], Tuple[str, str, Dict]] = {}
    t0 = time.time()

    if n_workers == 1:
        for task in tasks:
            ci, si, sha, label = task[0], task[1], task[2], task[5]
            print(f"[{sha}] {label} …", flush=True)
            _, _, _, _, m = _worker_run(task)
            results[(ci, si)] = (sha, label, m)
            print(f"  AMOTA={m['amota']*100:6.2f}  MOTA={m['mota']*100:6.2f}  "
                  f"HOTA={m['hota']*100:6.2f}  GT={m['gt_total']:6d}  "
                  f"FP={m['fp_total']:6d}  IDS={m['ids']:4d}", flush=True)
    else:
        # spawn-context avoids forking after numpy/scipy have already loaded
        # threadpools; safer cross-platform and matches Python 3.12 default.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
            done = 0
            for ci, si, sha, label, m in pool.imap_unordered(_worker_run, tasks):
                results[(ci, si)] = (sha, label, m)
                done += 1
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-9)
                eta_s = (n_total - done) / max(rate, 1e-9)
                print(f"[{done:2d}/{n_total} {elapsed:6.0f}s ETA {eta_s:4.0f}s] "
                      f"{sha} :: {label} :: "
                      f"AMOTA={m['amota']*100:6.2f}  MOTA={m['mota']*100:6.2f}  "
                      f"HOTA={m['hota']*100:6.2f}  GT={m['gt_total']:6d}  "
                      f"FP={m['fp_total']:6d}  IDS={m['ids']:4d}", flush=True)

    # Re-order back into user-requested (config, stage) order for the table.
    rows: List[Tuple[str, str, Dict]] = []
    for cfg_idx, sha in enumerate(args.configs):
        for stage_idx, (_, _, label) in enumerate(STAGES):
            if (cfg_idx, stage_idx) in results:
                rows.append(results[(cfg_idx, stage_idx)])

    table = emit_markdown_table(rows)
    print()
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table)
        print(f"wrote {args.out}")

    total_s = time.time() - t0
    print(f"[paper_metrics] done in {total_s:.0f}s "
          f"({total_s / max(n_total, 1):.0f}s per work item)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
