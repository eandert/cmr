#!/usr/bin/env python3
"""1-D hyperparameter sweep on the V2V4Real TRAINING split.

Sweeps four axes one at a time, holding the others at their current
defaults. Outputs go to results/V2V4Real_SWEEP_<axis>_<value>_<ts>/.
After every config a one-line summary is printed; a final consolidated
table is written to /tmp/sweep_log_odds_training.summary.txt.

Axes (current default in **bold**):
    birth_gate       0.2 / **0.3** / 0.4 / 0.5
    confirm_log_odds log(10)=2.303 / **log(20)=2.996** / log(50)=3.912
    kill_log_odds    log(0.1)=-2.303 / **log(0.05)=-2.996** / log(0.02)=-3.912
    log_lr_miss      log(0.20/0.80)=-1.386 / **log(0.30/0.70)=-0.847** / log(0.40/0.60)=-0.405

Total: 4 + 3 + 3 + 3 - 3 (default counted once per axis but different from
sweeping default elsewhere) = 10 distinct configs. Default (gate=0.3,
confirm=log(20), kill=log(0.05), miss=log(0.30/0.70)) runs once and the
result is reused as the centre of every axis.

Usage:
    python scripts/sweep_log_odds_training.py [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time as _time
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
EXPORT = Path("/home/rave/test/mmdetection3d/work_dirs/cmr_export")
DETECTOR = "pointpillar_v2v4real_score02_{vehicle}"
LOCALIZER = "rtk_v2v4real"
SCORE_THRESH = 0.30
MAX_RANGE = 50.0
SPLIT = "train"

DEFAULTS = dict(
    p_tp_birth_gate = 0.3,
    confirm_log_odds = math.log(20.0),
    kill_log_odds    = math.log(1.0/20.0),
    log_lr_miss      = math.log(0.30/0.70),
)

# (axis_name, list of values to try). Default value is INCLUDED so we get
# a clean centre for the comparison curve.
SWEEP_AXES: List[Tuple[str, List[float]]] = [
    ("p_tp_birth_gate",  [0.2, 0.3, 0.4, 0.5]),
    ("confirm_log_odds", [math.log(10.0), math.log(20.0), math.log(50.0)]),
    ("kill_log_odds",    [math.log(1.0/10.0), math.log(1.0/20.0), math.log(1.0/50.0)]),
    ("log_lr_miss",      [math.log(0.20/0.80), math.log(0.30/0.70), math.log(0.40/0.60)]),
]


def run_one(label: str, overrides: Dict[str, float], workers: int) -> Tuple[str, str]:
    """Fire one suite invocation; returns (label, output_dir_path) on success."""
    ts = _time.strftime("%Y-%m-%d_%H%M%S")
    out_dir = REPO / "results" / f"V2V4Real_SWEEP_{label}_{ts}"
    cfg = {**DEFAULTS, **overrides}
    cmd = [
        "python", str(REPO / "scripts" / "run_v2v4real_evaluation.py"),
        "--export-dir",       str(EXPORT),
        "--output-dir",       str(out_dir),
        "--detector-name",    DETECTOR,
        "--localizer-name",   LOCALIZER,
        "--score-threshold",  str(SCORE_THRESH),
        "--max-range",        str(MAX_RANGE),
        "--splits",           SPLIT,
        "--lifecycle-mode",   "log_odds",
        "--p-tp-birth-gate",  str(cfg["p_tp_birth_gate"]),
        "--confirm-log-odds", str(cfg["confirm_log_odds"]),
        "--kill-log-odds",    str(cfg["kill_log_odds"]),
        "--log-lr-miss",      str(cfg["log_lr_miss"]),
        "-j",                 str(workers),
    ]
    print(f"[{_time.strftime('%H:%M:%S')}] running {label}...", flush=True)
    t0 = _time.time()
    rc = subprocess.run(cmd, cwd=str(REPO))
    elapsed = (_time.time() - t0) / 60
    status = "OK" if rc.returncode == 0 else f"FAIL ({rc.returncode})"
    print(f"[{_time.strftime('%H:%M:%S')}] {label} done in {elapsed:.1f} min — {status}", flush=True)
    return label, str(out_dir)


def extract_amota(out_dir: str) -> Dict[str, float]:
    """Pull the merged-AMOTA values for headline streams from summary.json."""
    summary = Path(out_dir) / SPLIT / "summary.json"
    if not summary.exists():
        return {}
    data = json.load(open(summary))
    out = {}
    for key in ("ci_gpem_quadratic", "bici_gpem_quadratic", "sabre_gpem_quadratic",
                "gpem_quadratic", "baseline"):
        # Each stream is keyed `<stream>_av_100.0pct` in summary.json.
        full = f"{key}_av_100.0pct"
        if full in data.get("results", {}):
            v = data["results"][full].get("amota_merged_paper_mean", None)
            if v is not None:
                out[key] = v * 100.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j", "--workers", type=int, default=16)
    args = ap.parse_args()

    print(f"=== V2V4Real log_odds hyperparameter sweep — {SPLIT} split ===")
    print(f"Defaults: gate={DEFAULTS['p_tp_birth_gate']}  "
          f"confirm={DEFAULTS['confirm_log_odds']:.3f}  "
          f"kill={DEFAULTS['kill_log_odds']:.3f}  "
          f"miss={DEFAULTS['log_lr_miss']:.3f}")
    total = sum(len(v) for _, v in SWEEP_AXES)
    print(f"Total configs: {total}  (axes: {[a for a, _ in SWEEP_AXES]})")
    print()

    summary: List[Dict] = []
    for axis_name, values in SWEEP_AXES:
        for value in values:
            label = f"{axis_name}={value:.3f}"
            override = {axis_name: value}
            _, out_dir = run_one(label, override, args.workers)
            amota = extract_amota(out_dir)
            summary.append({"axis": axis_name, "value": value,
                            "out_dir": out_dir, **amota})
            # Ongoing summary line
            ci = amota.get("ci_gpem_quadratic", float("nan"))
            print(f"  ↳ {label:<32s}  ci_gpem_quad={ci:>6.2f}", flush=True)

    # Final consolidated table
    out_path = Path("/tmp/sweep_log_odds_training.summary.txt")
    with open(out_path, "w") as f:
        f.write("axis,value,baseline,gpem_quadratic,ci_gpem_quadratic,bici_gpem_quadratic,sabre_gpem_quadratic,out_dir\n")
        for s in summary:
            f.write(f"{s['axis']},{s['value']:.4f},"
                    f"{s.get('baseline', float('nan')):.2f},"
                    f"{s.get('gpem_quadratic', float('nan')):.2f},"
                    f"{s.get('ci_gpem_quadratic', float('nan')):.2f},"
                    f"{s.get('bici_gpem_quadratic', float('nan')):.2f},"
                    f"{s.get('sabre_gpem_quadratic', float('nan')):.2f},"
                    f"{s['out_dir']}\n")
    print()
    print(f"Sweep complete. Summary: {out_path}")


if __name__ == "__main__":
    main()
