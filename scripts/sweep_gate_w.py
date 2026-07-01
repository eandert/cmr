#!/usr/bin/env python3
"""Dual-protocol --w autotune sweep for the log_odds birth gate + lifecycle.

For each detector bucket {pp_score0, cp_zeroshot_100m, cp_finetune_100m} and
each FP-cost weight w in {0.5, 1.0, 2.0, 3.0, 5.0}:

  1. Run scripts/solve_amota_bayes_gate.py --w <w> (legacy native-bin mode —
     the production invocation) into an isolated SOLVER_OUT_DIR so the
     production results/birth_gate_calibration/*.csv (w=1.0 defaults) stay
     untouched.
  2. Score the candidate gate+lifecycle by RUNNING the log_odds tracker on the
     TRAIN-split merged views (data/v2v4real_merged_views/<bucket>_train,
     official 32-scene train split minus the scenario name duplicated between
     V2V4Real's train/ and test/ dirs), single stream sabre_gpem_polar.
  3. Record BOTH protocols per cell:
        v2v_amota  = v2v4real_amota_mean   (V2V4Real protocol, per-ego GT)
        ours_amota = paper_amota_mean      (AB3DMOT protocol, merged GT)
     into results/birth_gate_calibration/w_sweep/<bucket>_w_sweep.csv.

Finally (--make-variants), pick per-bucket w_v2v (max V2V) and w_ours
(max Ours) and re-run the solver at the chosen w into
results/birth_gate_calibration/variant_{v2v,ours}/ — each accumulating a
merge-safe birth_gate_curves.csv + recommended_lifecycle.csv across buckets.

The BLIND test split is never read; tuning decisions use train-split runs only.

Usage:
    python scripts/sweep_gate_w.py                        # full 3x5 sweep
    python scripts/sweep_gate_w.py --buckets cp_zeroshot_100m --ws 1.0 \
        --max-scenarios 2                                 # one-cell validation
    python scripts/sweep_gate_w.py --make-variants        # after the sweep
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# Load the SUITE runner from src/ explicitly by file path (same name-collision
# guard as scripts/run_v2v4real_benchmark.py — scripts/v2v4real_runner.py is a
# different module with no run_suite()). Registered in sys.modules BEFORE exec
# so ProcessPoolExecutor children can re-import it.
import importlib.util as _ilu
_runner_spec = _ilu.spec_from_file_location(
    "v2v4real_runner", REPO / "src" / "v2v4real_runner.py")
v2v4real_runner = _ilu.module_from_spec(_runner_spec)
sys.modules["v2v4real_runner"] = v2v4real_runner
_runner_spec.loader.exec_module(v2v4real_runner)

SOLVER = REPO / "scripts" / "solve_amota_bayes_gate.py"
MERGED_VIEWS = REPO / "data" / "v2v4real_merged_views"
SWEEP_ROOT = REPO / "results" / "birth_gate_calibration" / "w_sweep"
VARIANT_ROOT = REPO / "results" / "birth_gate_calibration"
STREAM = "sabre_gpem_polar"
STREAM_SUMMARY_KEY = "sabre_gpem_polar_av_100.0pct"

MMDET3D = REPO.parent / "mmdetection3d" / "work_dirs"
CPFT_EA = (REPO / "data" / "v2v4real_inputs" / "ours" / "detectors"
           / "cp_finetune_100m" / "gpem_calibration" / "_error_analysis")

# Per-bucket sweep spec. `solver_detectors` are the canonical detector names the
# production log_odds tracker looks up (and the solver writes); for cpft those
# names are not in the solver's DETECTORS map, so --error-analysis-dir overrides
# point at the bucket's persistent characterization dirs.
BUCKETS: Dict[str, Dict] = {
    "pp_score0": {
        "export_dir": MERGED_VIEWS / "pp_score0_train",
        "detector_name": "pointpillar_v2v4real_{vehicle}",
        "solver_detectors": ["pointpillar_v2v4real_astuff",
                             "pointpillar_v2v4real_tesla"],
        "error_analysis_overrides": {},   # in DETECTORS map (reexport_pp_*)
    },
    "cp_zeroshot_100m": {
        "export_dir": MERGED_VIEWS / "cp_zeroshot_100m_train",
        "detector_name": "centerpoint_100m_on_v2v4real",
        "solver_detectors": ["centerpoint_100m_on_v2v4real"],
        "error_analysis_overrides": {},   # in DETECTORS map (preds_cpzs_100m_tesla)
    },
    "cp_finetune_100m": {
        "export_dir": MERGED_VIEWS / "cp_finetune_100m_train",
        "detector_name": "centerpoint_100m_v2v4real_finetune_{vehicle}",
        "solver_detectors": ["centerpoint_100m_v2v4real_finetune_astuff",
                             "centerpoint_100m_v2v4real_finetune_tesla"],
        "error_analysis_overrides": {
            "centerpoint_100m_v2v4real_finetune_astuff":
                CPFT_EA / "centerpoint_100m_v2v4real_finetune_astuff",
            "centerpoint_100m_v2v4real_finetune_tesla":
                CPFT_EA / "centerpoint_100m_v2v4real_finetune_tesla",
        },
    },
}

DEFAULT_WS = [0.5, 1.0, 2.0, 3.0, 5.0]


def _wtag(w: float) -> str:
    return f"w{w:g}"


def run_solver(bucket: str, w: float, out_dir: Path, force: bool = False) -> Path:
    """Run solve_amota_bayes_gate.py for one bucket at one w into out_dir.

    Returns out_dir. The solver's parent merge-safe outputs land at
    out_dir/birth_gate_curves.csv + out_dir/recommended_lifecycle.csv
    (SOLVER_OUT_DIR redirection), leaving the production CSVs untouched.
    """
    spec = BUCKETS[bucket]
    gate_csv = out_dir / "birth_gate_curves.csv"
    life_csv = out_dir / "recommended_lifecycle.csv"
    if gate_csv.is_file() and life_csv.is_file() and not force:
        print(f"  [solver] reusing {out_dir}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Mirror the production invocation exactly (build_v2v4real_input_bucket
    # step_gate): NO pooling flags — legacy native 5m x 10 deg bins. The
    # production parent CSVs' polar tables are all ring_used=0, confirming the
    # legacy path; --adaptive-pooling would yield different fits.
    cmd = [sys.executable, str(SOLVER),
           "--w", str(w),
           "--detectors", *spec["solver_detectors"]]
    for name, path in spec["error_analysis_overrides"].items():
        cmd += ["--error-analysis-dir", f"{name}={path}"]
    env = dict(os.environ, SOLVER_OUT_DIR=str(out_dir))
    print(f"  [solver] {bucket} w={w} -> {out_dir}")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (out_dir / "solver_stdout.log").write_text(res.stdout + "\n" + res.stderr)
    if res.returncode != 0:
        raise RuntimeError(
            f"solver failed for {bucket} w={w} (see {out_dir}/solver_stdout.log):\n"
            f"{res.stderr[-2000:]}")
    if not gate_csv.is_file() or not life_csv.is_file():
        raise RuntimeError(f"solver produced no outputs in {out_dir}")
    return out_dir


def run_tracker(bucket: str, gate_csv: Path, life_csv: Path, run_dir: Path,
                workers: int, max_scenarios: Optional[int]) -> Dict[str, float]:
    """Run the log_odds tracker (single sabre_gpem_polar stream) on the
    bucket's TRAIN merged view with the candidate gate+lifecycle CSVs.

    Returns {"v2v_amota": ..., "ours_amota": ...} from the train summary.
    """
    spec = BUCKETS[bucket]
    summary_path = run_dir / "train" / "summary.json"
    if not summary_path.is_file():
        v2v4real_runner.run_suite(
            export_dir=spec["export_dir"],
            output_dir=run_dir,
            splits=["train"],
            workers=workers,
            max_scenarios=max_scenarios,
            # Production log_odds params for these buckets (see
            # run_v2v4real_benchmark.py *_gpem_logodds entries + _OUR_DEFAULTS).
            detector_name=spec["detector_name"],
            localizer_name="rtk_v2v4real",
            detector_max_range=100.0,
            score_threshold=0.0,
            lifecycle_mode="log_odds",
            data_driven_lifecycle=True,
            data_driven_lifecycle_csv=str(life_csv),
            data_driven_gate_alpha=99.0,
            data_driven_gate_fit="quadratic",
            data_driven_gate_csv=str(gate_csv),
            self_report_egos=True,
            streams_keep=[STREAM],
        )
    if not summary_path.is_file():
        raise RuntimeError(f"tracker run produced no summary at {summary_path}")
    with open(summary_path) as f:
        summary = json.load(f)
    r = summary["results"][STREAM_SUMMARY_KEY]
    return {"v2v_amota": r["v2v4real_amota_mean"],
            "ours_amota": r["paper_amota_mean"]}


def _read_sweep_csv(path: Path) -> Dict[float, Dict[str, float]]:
    rows: Dict[float, Dict[str, float]] = {}
    if path.is_file():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[float(r["w"])] = {"v2v_amota": float(r["v2v_amota"]),
                                       "ours_amota": float(r["ours_amota"])}
    return rows


def _write_sweep_csv(path: Path, bucket: str,
                     rows: Dict[float, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["bucket", "w", "v2v_amota", "ours_amota"])
        for w in sorted(rows):
            wr.writerow([bucket, w,
                         f"{rows[w]['v2v_amota']:.4f}",
                         f"{rows[w]['ours_amota']:.4f}"])


def sweep(buckets: List[str], ws: List[float], workers: int,
          max_scenarios: Optional[int], force_solver: bool) -> None:
    for bucket in buckets:
        csv_path = SWEEP_ROOT / f"{bucket}_w_sweep.csv"
        rows = _read_sweep_csv(csv_path)
        for w in ws:
            tag = _wtag(w)
            if w in rows and max_scenarios is None:
                print(f"[skip] {bucket} {tag} — already in {csv_path.name}")
                continue
            t0 = time.time()
            solver_dir = SWEEP_ROOT / "solver" / f"{bucket}_{tag}"
            run_solver(bucket, w, solver_dir, force=force_solver)
            run_dir = SWEEP_ROOT / "runs" / f"{bucket}_{tag}"
            metrics = run_tracker(
                bucket,
                gate_csv=solver_dir / "birth_gate_curves.csv",
                life_csv=solver_dir / "recommended_lifecycle.csv",
                run_dir=run_dir, workers=workers, max_scenarios=max_scenarios)
            print(f"[cell] {bucket} {tag}: V2V={metrics['v2v_amota']:.2f} "
                  f"Ours={metrics['ours_amota']:.2f} "
                  f"({(time.time()-t0)/60:.1f} min)")
            if max_scenarios is None:
                rows[w] = metrics
                _write_sweep_csv(csv_path, bucket, rows)
            else:
                print("  [validation mode] not recording to the sweep CSV "
                      f"(max_scenarios={max_scenarios})")


def make_variants(buckets: List[str]) -> None:
    """Pick per-bucket w_v2v / w_ours from the sweep CSVs and write the two
    named variant dirs by re-running the solver at the chosen w."""
    picks: Dict[str, Dict[str, float]] = {}
    for bucket in buckets:
        csv_path = SWEEP_ROOT / f"{bucket}_w_sweep.csv"
        rows = _read_sweep_csv(csv_path)
        if not rows:
            print(f"[make-variants] no sweep rows for {bucket} — skipping")
            continue
        w_v2v = max(rows, key=lambda w: rows[w]["v2v_amota"])
        w_ours = max(rows, key=lambda w: rows[w]["ours_amota"])
        picks[bucket] = {"v2v": w_v2v, "ours": w_ours}
        print(f"[pick] {bucket}: w_v2v={w_v2v} "
              f"(V2V={rows[w_v2v]['v2v_amota']:.2f})  w_ours={w_ours} "
              f"(Ours={rows[w_ours]['ours_amota']:.2f})")

    for metric in ("v2v", "ours"):
        out_dir = VARIANT_ROOT / f"variant_{metric}"
        for bucket, p in picks.items():
            # Force a fresh solver run so the variant dir's merge-safe parent
            # CSVs accumulate every bucket at its chosen w.
            run_solver(bucket, p[metric], out_dir, force=True)
        # Provenance: which w each bucket used in this variant.
        prov = out_dir / "chosen_w.csv"
        with open(prov, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["bucket", "metric", "w"])
            for bucket, p in picks.items():
                wr.writerow([bucket, metric, p[metric]])
        print(f"[variant_{metric}] -> {out_dir} "
              f"(birth_gate_curves.csv + recommended_lifecycle.csv)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buckets", nargs="*", default=list(BUCKETS.keys()),
                    choices=list(BUCKETS.keys()))
    ap.add_argument("--ws", nargs="*", type=float, default=DEFAULT_WS)
    ap.add_argument("-j", "--workers", type=int, default=8)
    ap.add_argument("--max-scenarios", type=int, default=None,
                    help="Cap train scenarios (validation only — results are "
                         "NOT recorded to the sweep CSV).")
    ap.add_argument("--force-solver", action="store_true",
                    help="Re-run the solver even if its outputs exist.")
    ap.add_argument("--make-variants", action="store_true",
                    help="Skip the sweep; pick per-bucket w from the sweep "
                         "CSVs and write variant_{v2v,ours}/.")
    args = ap.parse_args()

    if args.make_variants:
        make_variants(args.buckets)
        return
    sweep(args.buckets, args.ws, args.workers, args.max_scenarios,
          args.force_solver)


if __name__ == "__main__":
    main()
