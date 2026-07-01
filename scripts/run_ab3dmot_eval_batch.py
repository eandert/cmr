#!/usr/bin/env python3
"""Batch-run AB3DMOT's evaluate.py over many dumped streams.

For each stream subdir under <root>/<stream>/data_0/, this:
  1. Symlinks it into AB3DMOT/results/v2v4real/<prefix>_<stream>_H1/data_0
  2. Invokes scripts/KITTI/evaluate.py with the given dim/thres
  3. Parses the resulting summary_car_average_eval{2D|3D}.txt
  4. Writes a consolidated CSV to <root>/ab3dmot_eval_<dim>_<thres>.csv

Usage:
    python scripts/run_ab3dmot_eval_batch.py \\
        --root /tmp/pp_lf_kitti_multi \\
        --prefix late_fusion_pp_replica \\
        --dim 3D --thres 0.25
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

AB3DMOT_ROOT = REPO / "third_party" / "AB3DMOT"
EVAL_PY = AB3DMOT_ROOT / "scripts" / "KITTI" / "evaluate.py"


def parse_summary(summary_path: Path) -> dict:
    """Parse AB3DMOT's summary_car_average_eval{2D|3D}.txt for AMOTA/sAMOTA/MOTA."""
    txt = summary_path.read_text()
    # Final line of file has aggregate AMOTA etc. Format from evaluate.py:
    #   sAMOTA    AMOTA   AMOTP    MT     ML     IDS  FRAG    F1   Prec  Recall  FAR     TP    FP    FN
    #   <values>
    # Sometimes printed as "MOTA = 0.xxx" lines too. Scan for "AMOTA"-keyed line.
    m = re.search(r"AMOTA\s*=\s*([0-9.]+)", txt)
    samota = re.search(r"sAMOTA\s*=\s*([0-9.]+)", txt)
    amotp = re.search(r"AMOTP\s*=\s*([0-9.]+)", txt)
    mota = re.search(r"^\s*MOTA\s*=\s*([0-9.]+)", txt, re.MULTILINE)
    out = {}
    if m: out["AMOTA"] = float(m.group(1))
    if samota: out["sAMOTA"] = float(samota.group(1))
    if amotp: out["AMOTP"] = float(amotp.group(1))
    if mota: out["MOTA"] = float(mota.group(1))
    return out


def link_and_eval(stream_root: Path, sha: str, dim: str, thres: float) -> dict:
    dest = AB3DMOT_ROOT / "results" / "v2v4real" / f"{sha}_H1"
    dest.mkdir(parents=True, exist_ok=True)
    link = dest / "data_0"
    if link.is_symlink() or link.exists():
        try: link.unlink()
        except IsADirectoryError: subprocess.run(["rm", "-rf", str(link)])
    link.symlink_to(stream_root.resolve())

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{AB3DMOT_ROOT}:{AB3DMOT_ROOT}/Xinshuo_PyToolbox:" + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, str(EVAL_PY), sha, "1", dim, str(thres), "all", "val"],
        cwd=str(AB3DMOT_ROOT), env=env,
        capture_output=True, text=True, timeout=600,
    )
    summary_path = dest / f"summary_car_average_eval{dim}.txt"
    metrics = {}
    if summary_path.exists():
        metrics = parse_summary(summary_path)
    if not metrics:
        # capture last 30 lines of stdout for debug
        tail = "\n".join(r.stdout.splitlines()[-30:])
        metrics = {"_error": f"no summary parsed; stdout tail:\n{tail}"}
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="Dir containing <stream>/data_0 subdirs from dump_tracks --streams")
    ap.add_argument("--prefix", type=str, required=True,
                    help="Prefix for result_sha — MUST contain {cobevt,late_fusion,multi_sensor} "
                         "for evaluate.py auto-detect of v2v4real mode.")
    ap.add_argument("--dim", choices=["2D", "3D"], default="3D")
    ap.add_argument("--thres", type=float, default=0.25)
    args = ap.parse_args()

    if not re.search(r"(cobevt|late_fusion|multi_sensor)", args.prefix):
        sys.exit(f"--prefix '{args.prefix}' must contain cobevt|late_fusion|multi_sensor")

    streams = sorted([
        d.name for d in args.root.iterdir()
        if d.is_dir() and (d / "data_0").is_dir()
    ])
    if not streams:
        sys.exit(f"No <stream>/data_0/ subdirs found under {args.root}")

    print(f"Found {len(streams)} streams under {args.root}: {streams}")
    print(f"Running AB3DMOT evaluator ({args.dim} @ {args.thres}) for each…\n")

    out_csv = args.root / f"ab3dmot_eval_{args.dim}_{args.thres}.csv"
    rows = []
    for s in streams:
        sha = f"{args.prefix}_{s}"
        stream_data_0 = args.root / s / "data_0"
        print(f"  [{s}] → sha={sha}", flush=True)
        m = link_and_eval(stream_data_0, sha, args.dim, args.thres)
        row = {"stream": s, "sha": sha, **m}
        rows.append(row)
        amota = m.get("AMOTA", float('nan'))
        samota = m.get("sAMOTA", float('nan'))
        mota = m.get("MOTA", float('nan'))
        err = m.get("_error", "")
        print(f"    AMOTA={amota:.4f}  sAMOTA={samota:.4f}  MOTA={mota:.4f}  {err}")

    # Write CSV
    all_keys = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stream", "sha"] +
                           [k for k in all_keys if k not in ("stream", "sha")])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
