#!/usr/bin/env python3
"""Master CSV dump of every V2V4Real test-split result we've produced.

One row per (config, stream, evaluator) combo. Long-format so any
charting / pivoting tool eats it. Columns:

    detector       — pointpillar_v2v4real_score02 / centerpoint_100m_on_v2v4real /
                     centerpoint_finetuned_v2v4real / cobevt(DMSTrack-bundled) /
                     late_fusion(DMSTrack-bundled)
    config         — runner config tag (eg "stacked_box_score040", "spam_lf",
                     "stacked_box_score0", ...)
    stream         — fusion stream name (eg "ci_gpem_quadratic") for our pipeline,
                     "ab3dmot" for adapter-driven AB3DMOT replicas.
    eval_mode      — "std" (2m center-distance, FPs counted) or
                     "v2v_proto" (IoU 0.25, FPs ignored).
    amota_pct, amotp_pct, samota_pct, mota_pct
    mt_pct, ml_pct
    tp_total, fp_total, fn_total, ids_total, gt_total
    notes          — freeform context.

Output: /tmp/v2v4real_master_dump.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# ----------------------------------------------------------------------
# CMR/GPEM pipeline summary.json sources (each provides 30 streams ×
# both evaluators — std (paper_*) and V2V proto (v2v4real_*)).
# ----------------------------------------------------------------------
PIPELINE_RUNS: List[Tuple[str, str, str, str]] = [
    # (detector_label, config_label, summary_json_path, notes)
    ("centerpoint_zs", "stacked_box_score040",
     "V2V4Real_TEST_stacked_boxfilter_2026-04-28_104534",
     "score=0.40 + birth=0.40 + confirm=log10 + kill=log10 + miss=log0.40/0.60 + box[1.2-3.5,2.5-12]"),
    ("centerpoint_zs", "score040_only",
     "V2V4Real_TEST_score040_only_2026-04-28_100544",
     "score=0.40, all other knobs at first-pass anchor"),
    ("centerpoint_zs", "allbest_stacked_no_box",
     "V2V4Real_TEST_allbest_stacked_2026-04-28_100910",
     "all-best stacked, no box-size filter"),
    ("centerpoint_zs", "stacked_box_score0",
     "V2V4Real_TEST_stacked_boxfilter_score0_2026-04-28_130548",
     "score=0 (calibration-only test)"),
    ("pointpillar_v2v4real_score02", "stacked_box_score040",
     "V2V4Real_TEST_PP_stacked_boxfilter_LATEST",
     "score=0.40 + birth=0.40 + confirm=log10 + kill=log10 + miss=log0.40/0.60 + box[1.2-3.5,2.5-12]"),
]

CMR_STREAMS = [
    "baseline",
    "static_cov",
    "gpem_linear", "gpem_quadratic", "gpem_polar",
    "akf_baseline", "akf_static", "akf_gpem_linear", "akf_gpem_quadratic", "akf_gpem_polar",
    "ci_baseline", "ci_static", "ci_gpem_linear", "ci_gpem_quadratic", "ci_gpem_polar",
    "bici_baseline", "bici_static", "bici_gpem_linear", "bici_gpem_quadratic", "bici_gpem_polar",
    "sabre_baseline", "sabre_static", "sabre_gpem_linear", "sabre_gpem_quadratic", "sabre_gpem_polar",
    "pf_baseline", "pf_static", "pf_gpem_linear", "pf_gpem_quadratic", "pf_gpem_polar",
]


def latest_dir(prefix: str) -> Path | None:
    matches = sorted(RESULTS.glob(prefix + "*"))
    return matches[-1] if matches else None


def emit_pipeline_rows(detector: str, config: str, summary_dir_or_glob: str,
                        notes: str) -> List[Dict]:
    """Read the test split summary.json and emit one row per stream × evaluator."""
    if summary_dir_or_glob.endswith("LATEST"):
        d = latest_dir(summary_dir_or_glob[:-6])
    else:
        d = RESULTS / summary_dir_or_glob
    if d is None or not d.exists():
        print(f"  ✗ skip: {summary_dir_or_glob} (no dir)")
        return []
    summary = d / "test" / "summary.json"
    if not summary.exists():
        print(f"  ✗ skip: {d.name} (no test/summary.json)")
        return []

    data = json.load(open(summary))
    out: List[Dict] = []
    results = data.get("results", {})
    for stream in CMR_STREAMS:
        key = f"{stream}_av_100.0pct"
        r = results.get(key)
        if r is None:
            continue
        for eval_mode, prefix in (("std", "paper"), ("v2v_proto", "v2v4real")):
            row = {
                "detector": detector,
                "config":   config,
                "stream":   stream,
                "eval_mode": eval_mode,
                "amota_pct":  r.get(f"{prefix}_amota_mean", float("nan")) * 100,
                "amotp_pct":  r.get(f"{prefix}_amotp_mean", float("nan")) * 100,
                "samota_pct": r.get(f"{prefix}_samota_mean", float("nan")) * 100,
                "mota_pct":   r.get(f"{prefix}_mota_mean", float("nan")) * 100,
                "mt_pct":     r.get(f"{prefix}_mt_mean", float("nan")) * 100,
                "ml_pct":     r.get(f"{prefix}_ml_mean", float("nan")) * 100,
                "tp_total":   r.get(f"{prefix}_tp_total", 0),
                "fp_total":   r.get(f"{prefix}_fp_total", 0),
                "fn_total":   r.get(f"{prefix}_fn_total", 0),
                "ids_total":  r.get(f"{prefix}_ids_total", 0),
                "gt_total":   r.get(f"{prefix}_gt_total", 0),
                "n_scenarios": data.get("total_runs", 9),
                "notes":      notes,
            }
            out.append(row)
    print(f"  + {detector}/{config} ({d.name}): {len(out)} rows")
    return out


# ----------------------------------------------------------------------
# Adapter-driven (AB3DMOT replicas) — each run produces a single "stream"
# (ab3dmot) with std + v2v_proto.
# ----------------------------------------------------------------------

def emit_lfreplica_rows(detector_label: str, config_label: str, notes: str,
                         export_dir: str, score_threshold: float, nms_iou: float,
                         match_iou_gate: float) -> List[Dict]:
    """Run v2v4real_ab3dmot_replica programmatically on cmr_export-style dirs."""
    from v2v4real_ab3dmot_replica import replay_scenario
    import os

    LIDAR_RANGE = [-70.4, -40, -5, 70.4, 40, 3]
    GT_RANGE    = [-100, -40, -5, 100, 40, 3]

    scenarios = sorted([d for d in Path(export_dir).iterdir()
                         if d.is_dir() and d.name.startswith("test__")])

    sums = {("std", k): 0.0 for k in ("amota","amotp","samota","mota","mt","ml")}
    sums.update({("v2v_proto", k): 0.0 for k in ("amota","amotp","samota","mota","mt","ml")})
    counts = {("std", k): 0 for k in ("tp_total","fp_total","fn_total","ids_total","gt_total")}
    counts.update({("v2v_proto", k): 0 for k in ("tp_total","fp_total","fn_total","ids_total","gt_total")})
    n = 0
    for scen in scenarios:
        for eval_mode, kwargs in (("std", dict(eval_mode="center_dist", ignore_unmatched_fps=False)),
                                   ("v2v_proto", dict(eval_mode="iou", eval_iou_threshold=0.25,
                                                      ignore_unmatched_fps=True))):
            m = replay_scenario(scen,
                                 score_threshold=score_threshold,
                                 nms_iou=nms_iou,
                                 match_iou_gate=match_iou_gate,
                                 min_hits=3, max_age=2,
                                 lidar_range=LIDAR_RANGE,
                                 gt_range=GT_RANGE,
                                 car_only=True,
                                 **kwargs)
            for k in ("amota","amotp","samota","mota","mt","ml"):
                sums[(eval_mode, k)] += m.get(k, 0.0)
            for k in ("tp_total","fp_total","fn_total","ids_total","gt_total"):
                counts[(eval_mode, k)] += m.get(k, 0)
        n += 1

    out: List[Dict] = []
    for eval_mode in ("std", "v2v_proto"):
        out.append({
            "detector": detector_label,
            "config":   config_label,
            "stream":   "ab3dmot",
            "eval_mode": eval_mode,
            "amota_pct":  sums[(eval_mode, "amota")] / n * 100,
            "amotp_pct":  sums[(eval_mode, "amotp")] / n * 100,
            "samota_pct": sums[(eval_mode, "samota")] / n * 100,
            "mota_pct":   sums[(eval_mode, "mota")]  / n * 100,
            "mt_pct":     sums[(eval_mode, "mt")]    / n * 100,
            "ml_pct":     sums[(eval_mode, "ml")]    / n * 100,
            "tp_total":   counts[(eval_mode, "tp_total")],
            "fp_total":   counts[(eval_mode, "fp_total")],
            "fn_total":   counts[(eval_mode, "fn_total")],
            "ids_total":  counts[(eval_mode, "ids_total")],
            "gt_total":   counts[(eval_mode, "gt_total")],
            "n_scenarios": n,
            "notes":       notes,
        })
    print(f"  + {detector_label}/{config_label}: 2 rows (std + v2v_proto)")
    return out


def emit_dmstrack_rows(detector_label: str, config_label: str, notes: str,
                        det_dir: str, score_threshold: float, nms_iou: float,
                        match_iou_gate: float) -> List[Dict]:
    """Run dmstrack_cobevt_through_our_eval programmatically on KITTI-format inputs."""
    from dmstrack_cobevt_through_our_eval import (
        parse_kitti_det, parse_kitti_gt, track_scenario, build_per_frame_tape,
        GT_DIR,
    )
    from metrics.v2v4real_metrics import compute_ab3dmot_metrics, compute_ab3dmot_metrics_iou

    det_dir = Path(det_dir)
    sums = {("std", k): 0.0 for k in ("amota","amotp","samota","mota","mt","ml")}
    sums.update({("v2v_proto", k): 0.0 for k in ("amota","amotp","samota","mota","mt","ml")})
    counts = {("std", k): 0 for k in ("tp_total","fp_total","fn_total","ids_total","gt_total")}
    counts.update({("v2v_proto", k): 0 for k in ("tp_total","fp_total","fn_total","ids_total","gt_total")})

    scen_ids = [f"{i:04d}" for i in range(9)]
    n = 0
    for sid in scen_ids:
        det_path = det_dir / f"{sid}.txt"
        gt_path  = GT_DIR  / f"{sid}.txt"
        if not det_path.exists() or not gt_path.exists():
            continue
        dets = parse_kitti_det(det_path)
        gts  = parse_kitti_gt(gt_path)
        tracks = track_scenario(dets, score_threshold=score_threshold,
                                  nms_iou=nms_iou, match_iou_gate=match_iou_gate,
                                  min_hits=3, max_age=2)
        tape = build_per_frame_tape(tracks, gts)
        s = compute_ab3dmot_metrics(tape)
        v = compute_ab3dmot_metrics_iou(tape, iou_threshold=0.25,
                                         ignore_unmatched_fps=True)
        for k in ("amota","amotp","samota","mota","mt","ml"):
            sums[("std", k)]       += s.get(k, 0.0)
            sums[("v2v_proto", k)] += v.get(k, 0.0)
        for k in ("tp_total","fp_total","fn_total","ids_total","gt_total"):
            counts[("std", k)]       += s.get(k, 0)
            counts[("v2v_proto", k)] += v.get(k, 0)
        n += 1

    out: List[Dict] = []
    for eval_mode in ("std", "v2v_proto"):
        out.append({
            "detector": detector_label,
            "config":   config_label,
            "stream":   "ab3dmot",
            "eval_mode": eval_mode,
            "amota_pct":  sums[(eval_mode, "amota")] / n * 100,
            "amotp_pct":  sums[(eval_mode, "amotp")] / n * 100,
            "samota_pct": sums[(eval_mode, "samota")] / n * 100,
            "mota_pct":   sums[(eval_mode, "mota")]  / n * 100,
            "mt_pct":     sums[(eval_mode, "mt")]    / n * 100,
            "ml_pct":     sums[(eval_mode, "ml")]    / n * 100,
            "tp_total":   counts[(eval_mode, "tp_total")],
            "fp_total":   counts[(eval_mode, "fp_total")],
            "fn_total":   counts[(eval_mode, "fn_total")],
            "ids_total":  counts[(eval_mode, "ids_total")],
            "gt_total":   counts[(eval_mode, "gt_total")],
            "n_scenarios": n,
            "notes":       notes,
        })
    print(f"  + {detector_label}/{config_label}: 2 rows (std + v2v_proto)")
    return out


def main():
    rows: List[Dict] = []

    # -------- our pipeline summary.json runs ---------------------------
    print("== CMR/GPEM pipeline runs ==")
    for det, cfg, sumdir, notes in PIPELINE_RUNS:
        rows.extend(emit_pipeline_rows(det, cfg, sumdir, notes))

    # -------- AB3DMOT replicas on cmr_export ---------------------------
    print("== AB3DMOT replicas on cmr_export-style dirs ==")
    rows.extend(emit_lfreplica_rows(
        "centerpoint_zs", "lfreplica_dmstrack_params",
        "score=0 NMS=off IoU-gate=0.0001 (DMSTrack v2v4real.yml recipe)",
        str(_LOCAL_PATHS.get("mmdet3d_root") / "work_dirs/cmr_export"),
        score_threshold=0.0, nms_iou=1.0, match_iou_gate=0.0001))
    rows.extend(emit_lfreplica_rows(
        "centerpoint_zs", "lfreplica_our_params",
        "score=0.20 NMS=0.15 IoU-gate=0.01 (our defaults)",
        str(_LOCAL_PATHS.get("mmdet3d_root") / "work_dirs/cmr_export"),
        score_threshold=0.20, nms_iou=0.15, match_iou_gate=0.01))
    rows.extend(emit_lfreplica_rows(
        "pointpillar_v2v4real_score02", "lfreplica_dmstrack_params",
        "score=0 NMS=off IoU-gate=0.0001",
        str(_LOCAL_PATHS.get("mmdet3d_root") / "work_dirs/cmr_export_pointpillar_score02"),
        score_threshold=0.0, nms_iou=1.0, match_iou_gate=0.0001))

    # -------- DMSTrack-bundled detection adapters ---------------------
    print("== AB3DMOT replicas on DMSTrack-bundled detection inputs ==")
    rows.extend(emit_dmstrack_rows(
        "cobevt_dmstrack_bundled", "ab3dmot_dmstrack_params",
        "DMSTrack v2v4real.yml recipe — closest reproduction of their CoBEVT(*) 37.16 row",
        "third_party/AB3DMOT/data/v2v4real/detection/cobevt_Car_val",
        score_threshold=0.0, nms_iou=1.0, match_iou_gate=0.0001))
    rows.extend(emit_dmstrack_rows(
        "late_fusion_dmstrack_bundled", "ab3dmot_dmstrack_params",
        "DMSTrack v2v4real.yml recipe — closest reproduction of V2V4Real LF 29.28 row",
        "third_party/AB3DMOT/data/v2v4real/detection/late_fusion_Car_val",
        score_threshold=0.0, nms_iou=1.0, match_iou_gate=0.0001))

    # -------- write CSV -----------------------------------------------
    out_csv = Path("/tmp/v2v4real_master_dump.csv")
    cols = ["detector","config","stream","eval_mode",
            "amota_pct","amotp_pct","samota_pct","mota_pct","mt_pct","ml_pct",
            "tp_total","fp_total","fn_total","ids_total","gt_total",
            "n_scenarios","notes"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nMASTER DUMP: {out_csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
