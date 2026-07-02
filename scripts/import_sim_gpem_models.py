#!/usr/bin/env python3
"""Import the simulator's GPEM error models from mmdetection3d — the FIT path.

The SUMO simulator uses three detector error models (DETR3D, BEVFusion,
CenterPoint). Each is *fitted* in mmdetection3d (`tools/evaluate_errors.py`) and
converted to the CMR schema by `tools/regression_to_cmr_csv.py`, which emits the
smooth `polar_std_*` coefficients that `mode='polar'` reads at runtime. GPEM is a
*fit*, not a bin lookup, so those coefficients are mandatory. (This supersedes the
retired `scripts/old/import_mmdet_error_models.py`, which wrote bins but no
`polar_std_*` — a second, incomplete copy of the conversion that could drift.)

This script is the single, reproducible, manifest-driven entry point for the
simulator detectors. For each entry it runs the two-step chain:

  1. `regression_to_cmr_csv.py`  → {name}{,_distributions,_polar_distributions}.csv
     (all regression coeffs incl. polar_std, + distance/polar bins)
  2. `synthesize_overall_bins.py` → append the count-weighted overall-bin rows the
     static loader needs (regression_to_cmr_csv does not emit them)

then normalises line endings and verifies all five covariance modes load.

V2V4Real detectors go through `build_v2v4real_input_bucket.py` (which already runs
regression_to_cmr_csv); DAIR polar is not yet generated. See
`docs/IMPORTING_GPEM_MODELS.md` for the full cross-domain manifest.

Usage:
    python3 scripts/import_sim_gpem_models.py             # import all sim detectors
    python3 scripts/import_sim_gpem_models.py --dry-run   # print commands only
    python3 scripts/import_sim_gpem_models.py --only detr3d
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MMDET3D = REPO.parent / "mmdetection3d"
CONVERTER = MMDET3D / "tools" / "regression_to_cmr_csv.py"
DEST = REPO / "src" / "data" / "sensor_models"

# Manifest: CMR detector name -> mmdetection3d error_analysis dir (relative to
# work_dirs). The `*_100m_full` characterisations are the canonical full-range
# fits. mmdetection3d emits the polar fit; this script verifies it loads (below).
SIM_MANIFEST = {
    "detr3d":      "old/detr3d_100m_full/error_analysis",
    "bev_fusion":  "old/bevfusion_100m_full/error_analysis",
    "centerpoint": "old/centerpoint_100m_full/error_analysis",
}

MODES = ("static", "linear", "quadratic", "polar")


def _run(cmd, dry):
    print("  $ " + " ".join(str(c) for c in cmd))
    if not dry:
        subprocess.run(cmd, check=True)


def _normalise_lf(name):
    for suf in ("", "_distributions", "_polar_distributions"):
        f = DEST / f"{name}{suf}.csv"
        if f.exists():
            f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n"))


def _verify(name):
    sys.path.insert(0, str(REPO / "src"))
    import math
    from error_model import ErrorModel
    for mode in MODES:
        em = ErrorModel(name, mode=mode, max_range=100.0)
        s = (em.get_perpendicular_std(40.0, 30.0) if mode == "polar"
             else em.get_perpendicular_std(40.0))
        assert s >= 0, f"{name}/{mode} produced negative std {s}"
    print(f"  ✓ {name}: all modes load ({', '.join(MODES)})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(SIM_MANIFEST),
                    help="import only these detectors (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="print commands, do nothing")
    args = ap.parse_args()

    if not CONVERTER.exists():
        print(f"ERROR: converter not found: {CONVERTER}\n"
              f"       mmdetection3d must be a sibling of this repo.", file=sys.stderr)
        return 2

    names = args.only or list(SIM_MANIFEST)
    for name in names:
        ea = MMDET3D / "work_dirs" / SIM_MANIFEST[name]
        print(f"\n=== {name}  <-  {ea} ===")
        if not (ea / "regression_models.txt").exists():
            print(f"  SKIP: {ea}/regression_models.txt not found", file=sys.stderr)
            continue
        _run([sys.executable, str(CONVERTER), "--error-analysis", str(ea),
              "--out-dir", str(DEST), "--name", name], args.dry_run)
        _run([sys.executable, str(REPO / "scripts" / "synthesize_overall_bins.py"), name],
             args.dry_run)
        if not args.dry_run:
            _normalise_lf(name)
            _verify(name)

    if not args.dry_run:
        print("\nDone. Re-run a fast_city smoke to validate end-to-end:")
        print("  python3 src/experiment_runner.py --suite gpem_distribution_sweep "
              "--runs 1 --map fast_city --warmup 40 --record 80 -j 8 --output results/SMOKE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
