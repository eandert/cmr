#!/usr/bin/env python3
"""
Run V2V4Real evaluation suite.

Replays each V2V4Real scenario through the 30-stream fusion pipeline (5 cov
modes × 6 filter classes) using real CenterPoint detections and the GPEM
covariance models you've already characterised. Outputs per-split summary.json
files matching the simulation pipeline's format.

Example:
  python scripts/run_v2v4real_evaluation.py \\
    --export-dir data/v2v4real_cmr_export \\
    --output-dir results/V2V4Real_centerpoint_oneshot_perfectloc_$(date +%Y-%m-%d_%H%M%S) \\
    -j 8 --splits test
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

# Make src/ importable when running from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import v2v4real_runner  # noqa: E402


def _default_output_dir(detector_name: str, localizer_name: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"results/V2V4Real_{detector_name}_{localizer_name}_{ts}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--export-dir", type=Path,
        default=_REPO_ROOT / "data" / "v2v4real_cmr_export",
        help="Root of V2V4Real CMR export (contains <split>__<scenario>/ dirs)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write per-split summaries. "
             "Default: results/V2V4Real_<detector>_<localizer>_<timestamp>/",
    )
    parser.add_argument(
        "--splits", default=None,
        help="Comma-separated list of splits to run (e.g. 'test' or "
             "'train,val,test'). Default: all four (train, val, test, unknown)."
    )
    parser.add_argument("-j", "--workers", type=int, default=8,
                        help="Parallel workers (one per scenario). Default 8.")
    parser.add_argument("--detector-name", default="centerpoint_100m_on_v2v4real",
                        help="Detector model name (looked up in src/data/sensor_models/). "
                             "Default: centerpoint_100m_on_v2v4real.")
    parser.add_argument("--localizer-name", default="rtk_v2v4real",
                        help="Localizer model name. Default: rtk_v2v4real.")
    parser.add_argument("--max-range", type=float, default=100.0,
                        help="Detector max range (m) — used only for GPEM "
                             "covariance model clamping and birth-gate p_tp=0 "
                             "cutoff (NOT spatial crop; that is lidar_range). "
                             "Default 100 safely contains the entire V2V4Real "
                             "rectangular lidar window (box diagonal ~81 m) so "
                             "no detections inside the spatial crop are blocked.")
    parser.add_argument("--variance-floor", type=float, default=None,
                        help="Optional minimum diagonal R variance (m^2).")
    parser.add_argument("--score-threshold", type=float, default=0.3,
                        help="Drop detections with CenterPoint score < this. "
                             "Default 0.3 (standard nuScenes inference threshold).")
    parser.add_argument("--vehicle-classes", default="car,truck,bus,construction_vehicle",
                        help="Comma-separated nuScenes class names to keep. "
                             "Default car,truck,bus,construction_vehicle (the four "
                             "vehicle classes V2V4Real annotates as GT). "
                             "Pass 'all' to disable class filtering.")
    parser.add_argument("--class-thresholds", default=None,
                        help="Optional per-class score thresholds, comma-sep "
                             "k=v pairs (e.g. 'car=0.3,truck=0.4,bus=0.4'). "
                             "Overrides --score-threshold for listed classes.")
    parser.add_argument("--no-static-matching", action="store_true",
                        help="Disable static matching covariance (uses each "
                             "stream's own R for matching gating instead).")
    parser.add_argument("--no-self-report", action="store_true",
                        help="Disable V2X self-report (each ego broadcasts "
                             "its own RTK pose to the fusion as a tight-cov "
                             "detection of itself). On by default.")
    parser.add_argument("--p-tp-birth-gate", type=float, default=None,
                        help="Phase A: minimum calibrated P(TP|d,θ,score) "
                             "to spawn a new tentative track. e.g. 0.5 = "
                             "Bayesian-coin-flip threshold. Detectors without "
                             "a *_polar_calibration.csv companion file are "
                             "treated as P_TP=1 (gate is a no-op for them). "
                             "Currently applies the polar-calibration lookup "
                             "uniformly across all 30 fusion streams; per-cov-"
                             "mode P(TP) variants are a follow-up.")
    parser.add_argument("--lifecycle-mode", default="legacy",
                        choices=["legacy", "log_odds", "ab3dmot"],
                        help="Phase B: track lifecycle. 'legacy' (default) — "
                             "count-based confirmation (fusion_steps > 4). "
                             "'log_odds' — IPDA-style: each track holds an "
                             "S_t log-odds existence score, updated +log(p/(1-p)) "
                             "on match and +log(p_miss/(1-p_miss)) on miss; "
                             "confirm at S_t > log(20), kill at S_t < log(1/20). "
                             "'ab3dmot' — V2V4Real-style hard counts: confirm "
                             "after --ab3dmot-min-hits matched frames, kill "
                             "after --ab3dmot-max-age consecutive misses. "
                             "Combines well with our P(TP) birth gate.")
    parser.add_argument("--ab3dmot-min-hits", type=int, default=3,
                        help="ab3dmot lifecycle: confirm after this many "
                             "matched frames (incl. spawn). Default 3 = "
                             "V2V4Real / AB3DMOT default.")
    parser.add_argument("--ab3dmot-max-age", type=int, default=2,
                        help="ab3dmot lifecycle: kill after this many "
                             "consecutive missed frames. Default 2 = "
                             "V2V4Real / AB3DMOT default.")
    parser.add_argument("--enable-inactive-preservation", action="store_true",
                        help="Phase B.5: when a track's S_t collapses below "
                             "kill_log_odds, mark it inactive and keep it in "
                             "the matching pool for --inactive-grace-s "
                             "seconds. A fresh detection that lands within "
                             "its Mahalanobis gate revives the same ID. "
                             "Default off for clean A/B. Only takes effect "
                             "in --lifecycle-mode log_odds.")
    parser.add_argument("--inactive-grace-s", type=float, default=1.0,
                        help="Phase B.5 grace period in seconds before an "
                             "inactive track is hard-deleted. Default 1.0 "
                             "(~10 frames at V2V4Real's 10Hz). Only used "
                             "when --enable-inactive-preservation is set.")
    parser.add_argument("--per-ego-nms-iou", type=float, default=1.0,
                        help="BEV NMS IoU threshold applied PER-CAV. Default "
                             "1.0 = disabled. Hurts cross-vehicle fusion "
                             "streams. Opt-in for ablation only.")
    parser.add_argument("--combined-nms-iou", type=float, default=1.0,
                        help="BEV NMS IoU threshold applied to the UNION of "
                             "all CAVs' detections per frame (cross-CAV "
                             "concat-then-NMS, V2V4Real-style). Default 1.0 "
                             "= disabled. Hurts cross-vehicle fusion streams "
                             "(empirical: -3 AMOTA on CI/BICI/SABRE). Opt-in "
                             "for ablation only.")
    parser.add_argument("--confirm-log-odds", type=float, default=None,
                        help="log_odds lifecycle: emit threshold (default "
                             "log(20) ≈ 2.996). Lowering speeds confirmation "
                             "but admits more weak tracks.")
    parser.add_argument("--kill-log-odds", type=float, default=None,
                        help="log_odds lifecycle: kill threshold (default "
                             "log(1/20) ≈ -2.996). Raising kills sooner.")
    parser.add_argument("--log-lr-miss", type=float, default=None,
                        help="log_odds lifecycle: per-frame log-LR miss "
                             "decay (default log(0.30/0.70) ≈ -0.847, "
                             "tuned for ~70%% per-frame detection rate). "
                             "More-negative → faster decay.")
    parser.add_argument("--mahal-weight", type=float, default=0.6,
                        help="Hungarian-cost weight on Mahalanobis distance "
                             "(default 0.6). Set 0 for pure-IoU matching "
                             "(AB3DMOT-style).")
    parser.add_argument("--iou-weight", type=float, default=0.4,
                        help="Hungarian-cost weight on IoU (default 0.4). "
                             "Set 1 with --mahal-weight 0 + large "
                             "--mahal-gate for pure-IoU matching.")
    parser.add_argument("--mahal-gate", type=float, default=13.82,
                        help="Mahalanobis gate (chi² threshold; default "
                             "13.82 = 99.9%% / 2 DOF). Set very large (e.g. "
                             "1e9) to disable when running pure-IoU.")
    args = parser.parse_args()

    # Parse vehicle-classes flag.
    if args.vehicle_classes.strip().lower() == "all":
        vehicle_classes = None
    else:
        vehicle_classes = [c.strip() for c in args.vehicle_classes.split(",") if c.strip()]

    # Parse per-class thresholds flag.
    class_thresholds = None
    if args.class_thresholds:
        class_thresholds = {}
        for pair in args.class_thresholds.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                class_thresholds[k.strip()] = float(v.strip())

    if args.output_dir is None:
        args.output_dir = _REPO_ROOT / _default_output_dir(
            args.detector_name, args.localizer_name
        )

    splits = None
    if args.splits:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    v2v4real_runner.run_suite(
        export_dir=args.export_dir,
        output_dir=args.output_dir,
        splits=splits,
        workers=args.workers,
        detector_name=args.detector_name,
        localizer_name=args.localizer_name,
        detector_max_range=args.max_range,
        variance_floor=args.variance_floor,
        score_threshold=args.score_threshold,
        vehicle_classes=vehicle_classes,
        class_thresholds=class_thresholds,
        use_static_matching=not args.no_static_matching,
        self_report_egos=not args.no_self_report,
        p_tp_birth_gate=args.p_tp_birth_gate,
        lifecycle_mode=args.lifecycle_mode,
        ab3dmot_min_hits=args.ab3dmot_min_hits,
        ab3dmot_max_age=args.ab3dmot_max_age,
        per_ego_nms_iou=args.per_ego_nms_iou,
        combined_nms_iou=args.combined_nms_iou,
        enable_inactive_preservation=args.enable_inactive_preservation,
        inactive_grace_s=args.inactive_grace_s,
        confirm_log_odds=args.confirm_log_odds,
        kill_log_odds=args.kill_log_odds,
        log_lr_miss=args.log_lr_miss,
        mahal_weight=args.mahal_weight,
        iou_weight=args.iou_weight,
        mahal_gate=args.mahal_gate,
    )


if __name__ == "__main__":
    main()
