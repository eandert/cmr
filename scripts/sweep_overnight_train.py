#!/usr/bin/env python3
"""Overnight hyperparameter sweep on V2V4Real train split.

Optimizes for the STANDARD AMOTA evaluator (FPs counted, methodologically
honest). V2V protocol numbers are reported alongside but not used for
selection (avoids overfitting to the FP-ignore artifact).

Sweeps four axes 1D, holding others at the current first-pass settings:

    First-pass anchor (preserved as the centre point of every axis):
      detector_name     = centerpoint_100m_on_v2v4real     (CP zero-shot, where we lead)
      score_threshold   = 0.30
      birth_gate        = 0.30
      confirm_log_odds  = log(20)  ≈ 2.996
      kill_log_odds     = log(1/20)≈ -2.996
      log_lr_miss       = log(0.30/0.70) ≈ -0.847
      lifecycle_mode    = log_odds

    Sweep axes:
      birth_gate        : 0.20  0.25  0.30*  0.35  0.40
      confirm_log_odds  : log(10)  log(20)*  log(50)
      kill_log_odds     : log(1/10)  log(1/20)*  log(1/50)
      log_lr_miss       : log(0.20/0.80)  log(0.30/0.70)*  log(0.40/0.60)
      score_threshold   : 0.20  0.30*  0.40

    The * marks the first-pass default. Default is INCLUDED in every axis so
    we get a clean baseline measurement on the train split (allows estimating
    train-vs-test generalisation gap).

Total configs: 5 + 3 + 3 + 3 + 3 - 4 (default reused per axis) = 13.
Wall time estimate: 30-scenario train @ -j16 ≈ 50 min/config × 13 = ~11 h.

Output: results/V2V4Real_OVERNIGHT_<axis>_<value>_<ts>/ per config.
        /tmp/sweep_overnight.summary.csv  consolidated summary.

Usage:
    python scripts/sweep_overnight_train.py [--dry-run]
"""
from __future__ import annotations

import argparse, json, math, subprocess, time as _time
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
EXPORT = Path("/home/rave/test/mmdetection3d/work_dirs/cmr_export")
DETECTOR = "centerpoint_100m_on_v2v4real"
LOCALIZER = "rtk_v2v4real"
SPLIT = "train"
WORKERS = 16

# ---- First-pass anchor (the headline config, preserved as the reference) ----
FIRST_PASS = dict(
    score_threshold = 0.30,
    p_tp_birth_gate = 0.30,
    confirm_log_odds = math.log(20.0),
    kill_log_odds    = math.log(1.0/20.0),
    log_lr_miss      = math.log(0.30/0.70),
)

# ---- Sweep axes (default value is INCLUDED in each list) ----
AXES: List[Tuple[str, List[float]]] = [
    ("p_tp_birth_gate",  [0.20, 0.25, 0.30, 0.35, 0.40]),
    ("confirm_log_odds", [math.log(10.0), math.log(20.0), math.log(50.0)]),
    ("kill_log_odds",    [math.log(1.0/10.0), math.log(1.0/20.0), math.log(1.0/50.0)]),
    ("log_lr_miss",      [math.log(0.20/0.80), math.log(0.30/0.70), math.log(0.40/0.60)]),
    ("score_threshold",  [0.20, 0.30, 0.40]),
]


def run_one(label: str, override: Dict[str, float], dry_run: bool) -> Tuple[str, str]:
    ts = _time.strftime("%Y-%m-%d_%H%M%S")
    out_dir = REPO / "results" / f"V2V4Real_OVERNIGHT_{label}_{ts}"
    cfg = {**FIRST_PASS, **override}
    cmd = [
        "python", str(REPO / "scripts" / "run_v2v4real_evaluation.py"),
        "--export-dir",      str(EXPORT),
        "--output-dir",      str(out_dir),
        "--detector-name",   DETECTOR,
        "--localizer-name",  LOCALIZER,
        "--score-threshold", str(cfg["score_threshold"]),
        "--max-range",       "50.0",
        "--splits",          SPLIT,
        "--lifecycle-mode",  "log_odds",
        "--p-tp-birth-gate", str(cfg["p_tp_birth_gate"]),
        "--confirm-log-odds",str(cfg["confirm_log_odds"]),
        "--kill-log-odds",   str(cfg["kill_log_odds"]),
        "--log-lr-miss",     str(cfg["log_lr_miss"]),
        "-j",                str(WORKERS),
    ]
    if dry_run:
        print("[DRY] " + " ".join(cmd))
        return label, str(out_dir)
    print(f"[{_time.strftime('%H:%M:%S')}] {label} ...", flush=True)
    t0 = _time.time()
    rc = subprocess.run(cmd, cwd=str(REPO))
    elapsed = (_time.time() - t0) / 60
    status = "OK" if rc.returncode == 0 else f"FAIL ({rc.returncode})"
    print(f"[{_time.strftime('%H:%M:%S')}] {label} done in {elapsed:.1f}m — {status}", flush=True)
    return label, str(out_dir)


def extract_metrics(out_dir: str) -> Dict:
    summary = Path(out_dir) / SPLIT / "summary.json"
    if not summary.exists(): return {}
    data = json.load(open(summary))
    out = {}
    for k in ("baseline", "gpem_quadratic", "akf_gpem_quadratic",
              "ci_gpem_quadratic", "bici_gpem_quadratic", "sabre_gpem_quadratic"):
        full = f"{k}_av_100.0pct"
        r = data.get("results", {}).get(full, {})
        out[f"{k}_std"] = r.get("paper_amota_mean", float("nan"))
        out[f"{k}_v2v"] = r.get("v2v4real_amota_mean", float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"=== V2V4Real overnight sweep ({SPLIT} split) ===")
    print(f"Detector: {DETECTOR}")
    print(f"First-pass anchor preserved: {FIRST_PASS}")
    print()
    total = sum(len(v) for _, v in AXES)
    print(f"Total configs: {total} (~{total*50} min wall total at -j{WORKERS})")
    print()

    summary_rows: List[Dict] = []
    for axis, values in AXES:
        for value in values:
            label = f"{axis}={value:.4f}"
            override = {axis: value}
            _, out_dir = run_one(label, override, args.dry_run)
            metrics = extract_metrics(out_dir) if not args.dry_run else {}
            row = {"axis": axis, "value": value, "out_dir": out_dir, **metrics}
            summary_rows.append(row)
            ci = metrics.get("ci_gpem_quadratic_std", float("nan"))
            ci_v = metrics.get("ci_gpem_quadratic_v2v", float("nan"))
            print(f"  ↳ ci_gpem_quad std={ci:.2f}  V2V={ci_v:.2f}", flush=True)

    out_path = Path("/tmp/sweep_overnight.summary.csv")
    if not args.dry_run and summary_rows:
        keys = sorted({k for r in summary_rows for k in r})
        with open(out_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in summary_rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        print(f"\nSummary: {out_path}")


if __name__ == "__main__":
    main()
