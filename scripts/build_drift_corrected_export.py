#!/usr/bin/env python3
"""Write a drift-corrected copy of a per-CAV V2V4Real export.

Export-preprocessing correction (no tracker-code change — same pattern as the
DAIR L1-online self-calibration): for the NON-DEFAULT vehicle (astuff), add the
per-frame OU-smoothed inter-CAV RTK drift t=(tx,ty) (from
``estimate_intercav_rtk_drift.py``) to BOTH its detections and its GT objects,
bringing astuff's frame into alignment with the ego (tesla) frame. tesla rows,
ego_state, sensor_models and meta are copied unchanged.

Why correct ALL astuff rows (not only co-observed): the RTK drift rigidly shifts
astuff's whole frame, so the correction is global. For astuff SINGLETON objects
the detection AND its GT move together by the same t → their relative geometry is
unchanged → ZERO score effect. So a global correction is exactly score-equivalent
to "correct only the co-observed case", but simpler and oracle-free. The only
objects whose score changes are the co-observed ones — which is the point: GPEM
precision-weighted fusion blends the two CAVs' detections, and the merged GT
(replay's ``average`` strategy) averages the two GT boxes; aligning them first
should tighten both.

t is a relative translation (δ_tesla−δ_astuff), so it is center_world-invariant
and detector-independent — the drift measured on one export applies to any export
of the same scenarios (matched by folder name).

Run the benchmark on the original vs the corrected export (same config) and
compare, especially the merged-GT ``average`` strategy metrics:
    python scripts/run_v2v4real_benchmark.py --configs pp_score0_gpem_ab3dmot \\
        --export-dir <out> --output-dir results/DRIFT_AB_corrected
"""
from __future__ import annotations

import argparse
import csv
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import build_merged_gt as bmg  # noqa: E402  (reuse load_drift)

DEFAULT_SRC = REPO / "data" / "v2v4real_merged_views" / "pp_score0"
DEFAULT_DRIFT = REPO / "results" / "GT_MERGE_RTK" / "drift"
NON_DEFAULT = "astuff"   # the vehicle whose frame is corrected toward the ego (tesla)


def _offset_csv(src: Path, dst: Path, drift: dict, vehicle: str):
    """Copy a CSV, adding (tx,ty) to x,y of rows whose vehicle_id == `vehicle`."""
    n_off = 0
    with open(src) as fi:
        r = csv.DictReader(fi)
        fields = r.fieldnames
        rows = list(r)
    for row in rows:
        if row["vehicle_id"] == vehicle:
            tx, ty = drift[int(row["frame_id"])]
            row["x"] = f"{float(row['x']) + tx:.6f}"
            row["y"] = f"{float(row['y']) + ty:.6f}"
            n_off += 1
    with open(dst, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return n_off


def static_drift(scen: str, drift_dir: Path):
    """Per-scenario CONSTANT offset = robust (median) of the raw per-frame ego
    drift. The inter-CAV RTK offset is predominantly a static bias (the
    frame-to-frame wiggle is measurement noise), so a single median offset is
    more robust than the per-frame OU estimate, which over-fits noise in
    low-ego-coverage scenarios. Returned as a frame->(tx,ty) lookup that always
    yields the same constant.
    """
    path = drift_dir / f"{scen}.csv"
    if not path.exists():
        raise SystemExit(f"missing drift series {path}; run estimate_intercav_rtk_drift.py")
    xs, ys = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["tx_raw"]:
                xs.append(float(r["tx_raw"])); ys.append(float(r["ty_raw"]))
    tx = statistics.median(xs) if xs else 0.0
    ty = statistics.median(ys) if ys else 0.0
    return defaultdict(lambda: (tx, ty))


def _drift_for_mode(scen: str, drift_dir: Path, mode: str):
    """mode: 'none' (no offset), 'static' (per-scenario median), 'ou' (per-frame OU)."""
    if mode == "static":
        return static_drift(scen, drift_dir)
    if mode == "ou":
        return bmg.load_drift(scen, drift_dir, use_drift=True)
    return None


def correct_scenario(src_dir: Path, out_dir: Path, scen: str, drift_dir: Path,
                     det_mode: str = "static", gt_mode: str = "static"):
    """Independently drift-correct detections and GT. Modes: none / static / ou.
    Examples: T1 det=none,gt=static (clean-midpoint GT, raw det); T2 det=static,
    gt=static; 3c ("perfect loc") det=ou,gt=static (per-frame det correction)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    import shutil as _sh
    det_drift = _drift_for_mode(scen, drift_dir, det_mode)
    gt_drift = _drift_for_mode(scen, drift_dir, gt_mode)
    if det_drift is not None:
        nd = _offset_csv(src_dir / "detections.csv", out_dir / "detections.csv", det_drift, NON_DEFAULT)
    else:
        _sh.copy2(src_dir / "detections.csv", out_dir / "detections.csv"); nd = 0
    if gt_drift is not None:
        ng = _offset_csv(src_dir / "gt_objects.csv", out_dir / "gt_objects.csv", gt_drift, NON_DEFAULT)
    else:
        _sh.copy2(src_dir / "gt_objects.csv", out_dir / "gt_objects.csv"); ng = 0
    # everything else copied verbatim (ego_state, meta, overview png, etc.)
    for f in src_dir.iterdir():
        if f.name in ("detections.csv", "gt_objects.csv"):
            continue
        (shutil.copytree if f.is_dir() else shutil.copy2)(f, out_dir / f.name)
    return nd, ng


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-export", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--drift-dir", type=Path, default=DEFAULT_DRIFT)
    ap.add_argument("--out-export", type=Path, required=True)
    ap.add_argument("--det-correct", choices=["none", "static", "ou"], default="static",
                    help="how to drift-correct DETECTIONS (none/static-median/per-frame-OU)")
    ap.add_argument("--gt-correct", choices=["none", "static", "ou"], default="static",
                    help="how to drift-correct GT objects (none/static-median/per-frame-OU)")
    args = ap.parse_args()

    scen_dirs = sorted(d for d in args.src_export.iterdir()
                       if d.is_dir() and d.name.startswith("test__"))
    if not scen_dirs:
        raise SystemExit(f"no test__ scenarios under {args.src_export}")
    # shared (non-scenario) dirs like sensor_models/ copied once
    args.out_export.mkdir(parents=True, exist_ok=True)
    for d in args.src_export.iterdir():
        if d.is_dir() and not d.name.startswith("test__"):
            dst = args.out_export / d.name
            if not dst.exists():
                shutil.copytree(d, dst)
        elif d.is_file():
            shutil.copy2(d, args.out_export / d.name)

    tot_d = tot_g = 0
    for sd in scen_dirs:
        nd, ng = correct_scenario(sd, args.out_export / sd.name, sd.name, args.drift_dir,
                                  det_mode=args.det_correct, gt_mode=args.gt_correct)
        tot_d += nd; tot_g += ng
    print(f"corrected {len(scen_dirs)} scenarios (det={args.det_correct}, gt={args.gt_correct}): "
          f"offset {tot_d} astuff detections + {tot_g} astuff GT rows -> {args.out_export}")


if __name__ == "__main__":
    main()
