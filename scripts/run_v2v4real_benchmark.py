#!/usr/bin/env python3
"""Single-command V2V4Real benchmark runner.

Runs all canonical detector configs in sequence and prints a unified leaderboard
with three metric blocks per row:
  - V2V4Real protocol  (per-ego + FP-ignore, apples-to-apples with DMSTrack paper)
  - Per-ego strict     (per-ego, FPs fully counted, our stricter version)
  - Merged-GT          (spatially deduped GT, our primary cooperative-fusion eval)

Usage:
    python scripts/run_v2v4real_benchmark.py \\
        --export-dir /home/rave/test/mmdetection3d/work_dirs/cmr_export \\
        [--output-dir results/V2V4Real_BENCHMARK_custom_name] \\
        [--configs cp_zeroshot,cpft_pervehicle] \\
        [--resume] [-j 8] [--splits test]

    # Resume a partial run (skip configs with existing summary JSON):
    python scripts/run_v2v4real_benchmark.py --export-dir ... \\
        --output-dir results/V2V4Real_BENCHMARK_<existing> --resume

    # Smoke-test one config:
    python scripts/run_v2v4real_benchmark.py --export-dir ... \\
        --configs cp_zeroshot --output-dir /tmp/bm_smoke
"""

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import v2v4real_runner
from metrics.v2v4real_metrics import compute_hota_iou
from reaggregate_v2v4real_summaries import reaggregate

# ---------------------------------------------------------------------------
# Per-detector export directories
# ---------------------------------------------------------------------------

_EXPORTS = Path("/home/rave/test/mmdetection3d/work_dirs")
_EXPORT_CPT    = _EXPORTS / "cmr_export"                     # CPft test detections
_EXPORT_PP     = _EXPORTS / "cmr_export_pointpillar_score02" # PP score02 test detections
_EXPORT_COBEVT     = _EXPORTS / "cmr_export_cobevt"              # CoBEVT cooperative dets (per-vehicle dup)
_EXPORT_COBEVT_SE  = _EXPORTS / "cmr_export_cobevt_se"           # CoBEVT cooperative dets (single-ego, apples to AB3DMOT)
# TODO: add _EXPORT_CP_ZS once a CP NuScenes 100m test export is generated.
# For now cp_zeroshot reuses _EXPORT_CPT (CPft boxes, CP zero-shot covariance model).

_COBEVT_TRACKING_DIR = Path(
    "/home/rave/test/DMSTrack/AB3DMOT/results/v2v4real/cobevt_Car_val_H1/data_0")
_DMSTRACK_GT_DIR = Path(
    "/home/rave/test/DMSTrack/AB3DMOT/scripts/KITTI/v2v4real_val_label")


# ---------------------------------------------------------------------------
# Canonical benchmark configurations — the single source of truth
# ---------------------------------------------------------------------------

BENCHMARK_CONFIGS: List[Dict] = [
    {
        "name": "cp_zeroshot",
        "label": "CenterPoint zero-shot",
        "kind": "our",
        "export_dir": _EXPORT_CPT,  # TODO: replace with CP NuScenes 100m export
        "params": dict(
            detector_name="centerpoint_100m_on_v2v4real",
            lifecycle_mode="log_odds",
            detector_max_range=50.0,
            p_tp_birth_gate=0.3,
            score_threshold=0.3,
        ),
    },
    {
        "name": "pp_score02_pervehicle",
        "label": "PointPillar score02 (per-vehicle)",
        "kind": "our",
        "export_dir": _EXPORT_PP,
        "params": dict(
            detector_name="pointpillar_v2v4real_score02_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=50.0,
            p_tp_birth_gate=0.3,
            score_threshold=0.3,
        ),
    },
    {
        "name": "pp_lf_replica",
        "label": "Late Fusion replica (PP + AB3DMOT)",
        "kind": "our",
        "export_dir": _EXPORT_PP,
        "params": dict(
            detector_name="pointpillar_v2v4real_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=50.0,
            score_threshold=0.2,
        ),
    },
    {
        "name": "cpft_pervehicle",
        "label": "CPft per-vehicle (astuff + tesla)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_54m_v2v4real_finetune_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.3,
        ),
    },
    {
        "name": "cpft_lf_protocol",
        "label": "CPft per-vehicle (LF protocol: AB3DMOT + score 0.2)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_54m_v2v4real_finetune_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=50.0,
            score_threshold=0.2,
        ),
    },
    {
        "name": "cobevt_dets_cmr_se",
        "label": "CoBEVT dets + GPEM tracker (single-ego)",
        "kind": "our",
        "export_dir": _EXPORT_COBEVT_SE,
        "params": dict(
            detector_name="cobevt_tracker_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            p_tp_birth_gate=0.25,
            score_threshold=0.0,
            self_report_egos=False,
            ab3dmot_min_hits=3,
            ab3dmot_max_age=1,        # H1 to match AB3DMOT
        ),
    },
    {
        "name": "cobevt_dets_cmr",
        "label": "CoBEVT dets + GPEM tracker (per-vehicle dup)",
        "kind": "our",
        "export_dir": _EXPORT_COBEVT,
        "show_in_table": False,
        "params": dict(
            detector_name="cobevt_tracker_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            p_tp_birth_gate=0.25,
            score_threshold=0.2,
            self_report_egos=False,
        ),
    },
    {
        "name": "lf_dmstrack",
        "label": "Late Fusion (DMSTrack-bundled)",
        "kind": "dmstrack_lf",
        "show_in_table": False,
        "params": {},
    },
    {
        "name": "cobevt_dmstrack",
        "label": "CoBEVT (DMSTrack-bundled)",
        "kind": "dmstrack_cobevt",
        "show_in_table": False,
        "params": {},
    },
    {
        "name": "cobevt_precomputed",
        "label": "CoBEVT+AB3DMOT (official)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _COBEVT_TRACKING_DIR,
            "gt_dir": _DMSTRACK_GT_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": False,
        },
    },
]

# Shared params applied to all "our" configs.
_OUR_DEFAULTS = dict(
    localizer_name="rtk_v2v4real",
    ab3dmot_min_hits=3,
    ab3dmot_max_age=2,
    self_report_egos=True,
)

# Stream-name patterns produced by run_suite() / reaggregate for each CMR filter.
_FILTER_GROUPS: Dict[str, Dict[str, str]] = {
    "EKF": {
        "baseline": "baseline_av_100.0pct",
        "lin":      "gpem_linear_av_100.0pct",
        "quad":     "gpem_quadratic_av_100.0pct",
        "polar":    "gpem_polar_av_100.0pct",
    },
    "CI": {
        "baseline": "ci_baseline_av_100.0pct",
        "lin":      "ci_gpem_linear_av_100.0pct",
        "quad":     "ci_gpem_quadratic_av_100.0pct",
        "polar":    "ci_gpem_polar_av_100.0pct",
    },
    "AKF": {
        "baseline": "akf_baseline_av_100.0pct",
        "lin":      "akf_gpem_linear_av_100.0pct",
        "quad":     "akf_gpem_quadratic_av_100.0pct",
        "polar":    "akf_gpem_polar_av_100.0pct",
    },
    "PF": {
        "baseline": "pf_baseline_av_100.0pct",
        "lin":      "pf_gpem_linear_av_100.0pct",
        "quad":     "pf_gpem_quadratic_av_100.0pct",
        "polar":    "pf_gpem_polar_av_100.0pct",
    },
    "BICI": {
        "baseline": "bici_baseline_av_100.0pct",
        "lin":      "bici_gpem_linear_av_100.0pct",
        "quad":     "bici_gpem_quadratic_av_100.0pct",
        "polar":    "bici_gpem_polar_av_100.0pct",
    },
    "SABRE": {
        "baseline": "sabre_baseline_av_100.0pct",
        "lin":      "sabre_gpem_linear_av_100.0pct",
        "quad":     "sabre_gpem_quadratic_av_100.0pct",
        "polar":    "sabre_gpem_polar_av_100.0pct",
    },
}

# Fixed stream key used in DMSTrack summary JSONs.
_DMSTRACK_KEY = "dmstrack_av_100.0pct"


# ---------------------------------------------------------------------------
# DMSTrack adapter
# ---------------------------------------------------------------------------

def _load_dmstrack_mod(kind: str):
    fname = ("dmstrack_lf_through_our_eval.py" if kind == "dmstrack_lf"
             else "dmstrack_cobevt_through_our_eval.py")
    spec = importlib.util.spec_from_file_location("_dmstrack", REPO / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _wmean(values: List[float], weights: List[float]) -> float:
    tot = sum(weights)
    if tot <= 0:
        return mean(values) if values else 0.0
    return sum(v * w for v, w in zip(values, weights)) / tot


def _run_dmstrack(cfg: Dict, subdir: Path) -> None:
    """Run all 9 DMSTrack scenarios and write summary_v2v4real_protocol.json."""
    mod = _load_dmstrack_mod(cfg["kind"])
    ns = SimpleNamespace(score_threshold=0.20, nms_iou=0.15,
                         match_iou_gate=0.01, min_hits=3, max_age=2)

    # Metric keys returned by compute_ab3dmot_metrics / _iou are [0,1] fractions.
    # Multiply by 100 to match the percentage units stored by run_suite().
    RATE_PAIRS = [
        ("paper_per_ego_amota",  "paper_per_ego_gt_total"),
        ("paper_per_ego_amotp",  "paper_per_ego_gt_total"),
        ("paper_per_ego_samota", "paper_per_ego_gt_total"),
        ("paper_per_ego_mota",   "paper_per_ego_gt_total"),
        ("paper_per_ego_mt",     "paper_per_ego_gt_total"),
        ("paper_per_ego_ml",     "paper_per_ego_gt_total"),
        ("v2v4real_amota",  "v2v4real_gt_total"),
        ("v2v4real_amotp",  "v2v4real_gt_total"),
        ("v2v4real_samota", "v2v4real_gt_total"),
        ("v2v4real_mota",   "v2v4real_gt_total"),
        ("v2v4real_mt",     "v2v4real_gt_total"),
        ("v2v4real_ml",     "v2v4real_gt_total"),
    ]
    HOTA_PAIRS = [("hota", None), ("deta", None), ("assa", None), ("loca", None)]
    COUNT_KEYS = ["paper_per_ego_gt_total", "paper_per_ego_ids_total", "v2v4real_gt_total"]

    runs: List[Dict] = []
    for sid in [f"{i:04d}" for i in range(9)]:
        std_m, v2v_m, gt_n = mod.run_one(sid, ns)

        # Rebuild tape to compute HOTA (V2V protocol: FPs ignored in GT-free regions).
        dets   = mod.parse_kitti_det(mod.DET_DIR / f"{sid}.txt")
        gts    = mod.parse_kitti_gt(mod.GT_DIR  / f"{sid}.txt")
        tracks = mod.track_scenario(dets, score_threshold=ns.score_threshold,
                                    nms_iou=ns.nms_iou, match_iou_gate=ns.match_iou_gate,
                                    min_hits=ns.min_hits, max_age=ns.max_age)
        tape   = mod.build_per_frame_tape(tracks, gts)
        v2v_h  = compute_hota_iou(tape, iou_threshold=0.25, ignore_unmatched_fps=False)

        runs.append({
            "paper_per_ego_amota":   100.0 * std_m["amota"],
            "paper_per_ego_amotp":   100.0 * std_m["amotp"],
            "paper_per_ego_samota":  100.0 * std_m["samota"],
            "paper_per_ego_mota":    100.0 * std_m["mota"],
            "paper_per_ego_mt":      100.0 * std_m.get("mt", 0.0),
            "paper_per_ego_ml":      100.0 * std_m.get("ml", 0.0),
            "paper_per_ego_gt_total":   gt_n,
            "paper_per_ego_ids_total":  int(std_m.get("ids_total", 0)),
            "v2v4real_amota":   100.0 * v2v_m["amota"],
            "v2v4real_amotp":   100.0 * v2v_m["amotp"],
            "v2v4real_samota":  100.0 * v2v_m["samota"],
            "v2v4real_mota":    100.0 * v2v_m["mota"],
            "v2v4real_mt":      100.0 * v2v_m.get("mt", 0.0),
            "v2v4real_ml":      100.0 * v2v_m.get("ml", 0.0),
            "v2v4real_gt_total": gt_n,
            "hota": 100.0 * v2v_h["hota"],
            "deta": 100.0 * v2v_h["deta"],
            "assa": 100.0 * v2v_h["assa"],
            "loca": 100.0 * v2v_h["loca"],
        })

    n = len(runs)
    agg: Dict = {"num_runs": n}
    for field, wf in RATE_PAIRS:
        vals = [r.get(field, 0.0) for r in runs]
        wts  = [r.get(wf, 0) for r in runs]
        agg[f"{field}_mean"] = _wmean(vals, wts)
        agg[f"{field}_std"]  = stdev(vals) if n > 1 else 0.0
    for hf, _ in HOTA_PAIRS:
        vals = [r.get(hf, 0.0) for r in runs]
        agg[f"{hf}_mean"] = mean(vals) if vals else 0.0
        agg[f"{hf}_std"]  = stdev(vals) if n > 1 else 0.0
    for cf in COUNT_KEYS:
        agg[cf] = sum(int(r.get(cf, 0)) for r in runs)

    split_dir = subdir / "test"
    split_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "suite_name":   cfg["label"],
        "split":        "test",
        "total_runs":   n,
        "show_in_table": cfg.get("show_in_table", True),
        "results":      {_DMSTRACK_KEY: agg},
    }
    out = split_dir / "summary_v2v4real_protocol.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Precomputed-tracks adapter
# ---------------------------------------------------------------------------

def _run_precomputed_tracks(cfg: Dict, subdir: Path) -> None:
    """Evaluate pre-computed KITTI MOT tracking files and write summary JSON."""
    spec = importlib.util.spec_from_file_location(
        "_eval_precomputed", REPO / "scripts" / "evaluate_precomputed_tracks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    p = cfg["params"]
    tracking_dir = Path(p["tracking_dir"])
    gt_dir       = Path(p.get("gt_dir", str(_DMSTRACK_GT_DIR)))
    camera_frame = bool(p.get("camera_frame", False))
    iou_thr      = float(p.get("iou_threshold", 0.25))
    match_3d     = bool(p.get("match_3d", False))

    RATE_PAIRS = [
        ("paper_per_ego_amota",  "paper_per_ego_gt_total"),
        ("paper_per_ego_amotp",  "paper_per_ego_gt_total"),
        ("paper_per_ego_samota", "paper_per_ego_gt_total"),
        ("paper_per_ego_mota",   "paper_per_ego_gt_total"),
        ("paper_per_ego_mt",     "paper_per_ego_gt_total"),
        ("paper_per_ego_ml",     "paper_per_ego_gt_total"),
        ("v2v4real_amota",  "v2v4real_gt_total"),
        ("v2v4real_amotp",  "v2v4real_gt_total"),
        ("v2v4real_samota", "v2v4real_gt_total"),
        ("v2v4real_mota",   "v2v4real_gt_total"),
        ("v2v4real_mt",     "v2v4real_gt_total"),
        ("v2v4real_ml",     "v2v4real_gt_total"),
    ]
    HOTA_PAIRS = [("hota", None), ("deta", None), ("assa", None), ("loca", None)]
    COUNT_KEYS = ["paper_per_ego_gt_total", "paper_per_ego_ids_total", "v2v4real_gt_total"]

    runs: List[Dict] = []
    for sid in [f"{i:04d}" for i in range(9)]:
        std_m, v2v_m, std_h, _vh, gt_n = mod.run_one(
            sid, tracking_dir, gt_dir, camera_frame, iou_thr, match_3d=match_3d)
        runs.append({
            "paper_per_ego_amota":   100.0 * std_m["amota"],
            "paper_per_ego_amotp":   100.0 * std_m["amotp"],
            "paper_per_ego_samota":  100.0 * std_m["samota"],
            "paper_per_ego_mota":    100.0 * std_m["mota"],
            "paper_per_ego_mt":      100.0 * std_m.get("mt", 0.0),
            "paper_per_ego_ml":      100.0 * std_m.get("ml", 0.0),
            "paper_per_ego_gt_total":  gt_n,
            "paper_per_ego_ids_total": int(std_m.get("ids_total", 0)),
            "v2v4real_amota":   100.0 * v2v_m["amota"],
            "v2v4real_amotp":   100.0 * v2v_m["amotp"],
            "v2v4real_samota":  100.0 * v2v_m["samota"],
            "v2v4real_mota":    100.0 * v2v_m["mota"],
            "v2v4real_mt":      100.0 * v2v_m.get("mt", 0.0),
            "v2v4real_ml":      100.0 * v2v_m.get("ml", 0.0),
            "v2v4real_gt_total": gt_n,
            "hota": 100.0 * std_h["hota"],
            "deta": 100.0 * std_h["deta"],
            "assa": 100.0 * std_h["assa"],
            "loca": 100.0 * std_h["loca"],
        })

    n = len(runs)
    agg: Dict = {"num_runs": n}
    for field, wf in RATE_PAIRS:
        vals = [r.get(field, 0.0) for r in runs]
        wts  = [r.get(wf, 0) for r in runs]
        agg[f"{field}_mean"] = _wmean(vals, wts)
        agg[f"{field}_std"]  = stdev(vals) if n > 1 else 0.0
    for hf, _ in HOTA_PAIRS:
        vals = [r.get(hf, 0.0) for r in runs]
        agg[f"{hf}_mean"] = mean(vals) if vals else 0.0
        agg[f"{hf}_std"]  = stdev(vals) if n > 1 else 0.0
    for cf in COUNT_KEYS:
        agg[cf] = sum(int(r.get(cf, 0)) for r in runs)

    split_dir = subdir / "test"
    split_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "suite_name":   cfg["label"],
        "split":        "test",
        "total_runs":   n,
        "show_in_table": cfg.get("show_in_table", True),
        "results":      {_DMSTRACK_KEY: agg},
    }
    out = split_dir / "summary_v2v4real_protocol.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Leaderboard (CSV)
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "method", "detector", "filter", "mode",
    # V2V4Real protocol (per-ego + FP-ignore — matches DMSTrack paper)
    "v2v_amota", "v2v_amotp", "v2v_samota", "v2v_mota", "v2v_mt", "v2v_ml",
    # Per-ego strict (FPs fully counted)
    "pe_amota", "pe_amotp", "pe_samota", "pe_mota", "pe_mt", "pe_ml",
    # Merged-GT (our primary cooperative eval)
    "mg_amota", "mg_amotp", "mg_samota", "mg_mota", "mg_mt", "mg_ml",
    # HOTA (standard: FPs fully counted, consistent across all configs)
    "hota", "deta", "assa", "loca",
    # Diagnostics
    "ids", "gt",
]


def _row_dict(method: str, detector: str, filter_name: str, mode: str, r: Dict) -> Dict:
    def g6(prefix: str, short: str) -> Dict:
        keys = ["amota", "amotp", "samota", "mota", "mt", "ml"]
        return {f"{short}_{k}": round(r.get(f"{prefix}{k}_mean", 0.0), 4) for k in keys}

    d = {"method": method, "detector": detector, "filter": filter_name, "mode": mode}
    d.update(g6("v2v4real_", "v2v"))
    d.update(g6("paper_per_ego_", "pe"))
    d.update(g6("paper_", "mg"))
    for hf in ("hota", "deta", "assa", "loca"):
        d[hf] = round(r.get(f"{hf}_mean", 0.0), 4)
    d["ids"] = int(r.get("paper_per_ego_ids_total", 0))
    d["gt"]  = int(r.get("v2v4real_gt_total", r.get("paper_per_ego_gt_total", 0)))
    return d


def print_leaderboard(output_dir: Path, configs: List[Dict], split: str = "test") -> List[Dict]:
    """Write leaderboard.csv and print a compact best-per-detector summary."""
    rows: List[Dict] = []

    for cfg in configs:
        summary_path = output_dir / cfg["name"] / split / "summary_v2v4real_protocol.json"
        if not summary_path.exists():
            print(f"  # missing: {summary_path}")
            continue
        with open(summary_path) as f:
            results = json.load(f).get("results", {})

        if cfg["kind"].startswith("dmstrack_") or cfg["kind"] == "precomputed_tracks":
            r = results.get(_DMSTRACK_KEY)
            if r:
                rows.append(_row_dict(cfg["label"], cfg["label"], "—", "—", r))
        else:
            for fg_name, fg_keys in _FILTER_GROUPS.items():
                base_r = results.get(fg_keys["baseline"])
                if base_r:
                    rows.append(_row_dict(
                        f"{cfg['label']} + {fg_name} (base)",
                        cfg["label"], fg_name, "base", base_r))

                best_r, best_mode = None, None
                for mode in ("lin", "quad", "polar"):
                    r = results.get(fg_keys[mode])
                    if r is None:
                        continue
                    if (best_r is None or
                            r.get("v2v4real_amota_mean", 0.0) >
                            best_r.get("v2v4real_amota_mean", 0.0)):
                        best_r, best_mode = r, mode
                if best_r is not None:
                    rows.append(_row_dict(
                        f"{cfg['label']} + {fg_name} + GPEM-{best_mode}",
                        cfg["label"], fg_name, f"gpem_{best_mode}", best_r))

    # Write CSV
    csv_path = output_dir / "leaderboard.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    # Print compact best-per-detector summary to stdout
    print(f"\n{'Method':<52}  {'V2V AMOTA':>9}  {'PE AMOTA':>8}  {'MG AMOTA':>8}"
          f"  {'HOTA':>6}  {'DetA':>6}  {'AssA':>6}  {'LocA':>6}  {'IDS':>5}  {'GT':>6}")
    print("-" * 118)
    seen: set = set()
    for row in rows:
        det = row["detector"]
        if det not in seen:
            # Print the best row for this detector (already argmax'd per filter above)
            seen.add(det)
        loca_str = f"{row['loca']:>6.2f}" if row['loca'] else "    —"
        print(f"  {row['method']:<50}  {row['v2v_amota']:>9.2f}  {row['pe_amota']:>8.2f}"
              f"  {row['mg_amota']:>8.2f}"
              f"  {row['hota']:>6.2f}  {row['deta']:>6.2f}  {row['assa']:>6.2f}"
              f"  {loca_str}  {row['ids']:>5}  {row['gt']:>6}")

    print(f"\n  Saved to {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _write_manifest(output_dir: Path, args: argparse.Namespace) -> None:
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        git_hash = "unknown"
    def _jsonify(obj):
        if isinstance(obj, Path): return str(obj)
        if isinstance(obj, dict): return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_jsonify(v) for v in obj]
        return obj

    manifest = {
        "timestamp":  datetime.now().isoformat(),
        "git_hash":   git_hash,
        "export_dir": str(args.export_dir),
        "splits":     args.splits,
        "configs":    _jsonify(BENCHMARK_CONFIGS),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", type=Path, default=None,
                    help="Fallback export root for configs without a built-in export_dir.")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Root output dir (default: results/V2V4Real_BENCHMARK_<ts>).")
    ap.add_argument("--configs", default=None,
                    help="Comma-separated config names to run (default: all).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip configs that already have summary_v2v4real_protocol.json.")
    ap.add_argument("-j", "--workers", type=int, default=8,
                    help="Parallel workers per run_suite() call (default: 8).")
    ap.add_argument("--splits", nargs="+", default=["test"],
                    help="Data splits to evaluate (default: test).")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        args.output_dir = REPO / "results" / f"V2V4Real_BENCHMARK_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _write_manifest(args.output_dir, args)

    configs = BENCHMARK_CONFIGS
    if args.configs:
        wanted = set(args.configs.split(","))
        configs = [c for c in BENCHMARK_CONFIGS if c["name"] in wanted]

    for cfg in configs:
        subdir = args.output_dir / cfg["name"]
        summary_path = subdir / "test" / "summary_v2v4real_protocol.json"

        if args.resume and summary_path.exists():
            print(f"\n  [skip] {cfg['name']} — summary already exists")
            continue

        print(f"\n{'=' * 70}")
        print(f"  {cfg['label']}")
        print(f"{'=' * 70}")
        subdir.mkdir(exist_ok=True)

        if cfg["kind"] == "our":
            export_dir = cfg.get("export_dir") or args.export_dir
            if export_dir is None:
                print(f"  [skip] {cfg['name']} — no export_dir in config or --export-dir")
                continue
            v2v4real_runner.run_suite(
                export_dir=export_dir,
                output_dir=subdir,
                splits=args.splits,
                workers=args.workers,
                **{**_OUR_DEFAULTS, **cfg["params"]},
            )
            reaggregate(subdir)
        elif cfg["kind"].startswith("dmstrack_"):
            _run_dmstrack(cfg, subdir)
        elif cfg["kind"] == "precomputed_tracks":
            _run_precomputed_tracks(cfg, subdir)

    print(f"\n{'=' * 70}")
    print("  LEADERBOARD")
    print(f"{'=' * 70}\n")
    print_leaderboard(args.output_dir, configs)


if __name__ == "__main__":
    main()
