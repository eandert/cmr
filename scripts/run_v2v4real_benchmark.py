#!/usr/bin/env python3
"""Single-command V2V4Real benchmark runner.

Runs all canonical detector configs in sequence and prints a unified leaderboard
with three metric blocks per row:
  - V2V4Real protocol  (per-ego + FP-ignore, apples-to-apples with DMSTrack paper)
  - Per-ego strict     (per-ego, FPs fully counted, our stricter version)
  - Merged-GT          (spatially deduped GT, our primary cooperative-fusion eval)

Usage:
    python scripts/run_v2v4real_benchmark.py \\
        --export-dir <your-mmdet3d>/work_dirs/cmr_export \\
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

# Load the SUITE runner from src/ explicitly by file path. A same-named
# experiment-by-name runner lives at scripts/v2v4real_runner.py; because Python
# prepends the script's own dir (scripts/) to sys.path[0] when this benchmark is
# run as `python scripts/run_v2v4real_benchmark.py`, a bare `import
# v2v4real_runner` resolves to that scripts/ module (which has no run_suite()).
# Pin to the file path to beat the name collision, and register it under the
# canonical module name in sys.modules BEFORE exec so the run_suite()
# ProcessPoolExecutor workers (which pickle module-qualified function refs like
# _replay_one) can re-import it in child processes.
import importlib.util as _ilu
_runner_spec = _ilu.spec_from_file_location(
    "v2v4real_runner", REPO / "src" / "v2v4real_runner.py")
v2v4real_runner = _ilu.module_from_spec(_runner_spec)
sys.modules["v2v4real_runner"] = v2v4real_runner
_runner_spec.loader.exec_module(v2v4real_runner)
from metrics.v2v4real_metrics import compute_hota_iou
from reaggregate_v2v4real_summaries import reaggregate

# ---------------------------------------------------------------------------
# Per-detector export directories
# ---------------------------------------------------------------------------

_EXPORTS = (REPO / "data" / "v2v4real_inputs" / "ours")
_EXPORT_CPT    = _EXPORTS / "cmr_export"                     # CPft test detections
_EXPORT_PP     = _EXPORTS / "cmr_export_pointpillar_score02" # PP score02 test detections
_EXPORT_COBEVT     = _EXPORTS / "cmr_export_cobevt"              # CoBEVT cooperative dets (per-vehicle dup)
_EXPORT_COBEVT_SE  = _EXPORTS / "cmr_export_cobevt_se"           # CoBEVT cooperative dets (single-ego, apples to AB3DMOT)
# Cross-detector mix exports (heterogeneous fleet experiments)
_EXPORT_MIX_PP_CPFT = _EXPORTS / "cmr_export_mix_pp_cpft"        # astuff=PP, tesla=CPft
_EXPORT_MIX_CPFT_PP = _EXPORTS / "cmr_export_mix_cpft_pp"        # astuff=CPft, tesla=PP
_EXPORT_MIX_PP_CPZS = _EXPORTS / "cmr_export_mix_pp_cpzs"        # astuff=PP, tesla=CPft-boxes (CPzs cov via symlink)
_EXPORT_MIX_CPZS_PP = _EXPORTS / "cmr_export_mix_cpzs_pp"        # astuff=CPft-boxes (CPzs cov), tesla=PP
# Cov-only mixes (cases 5,6) reuse _EXPORT_CPT — same boxes, different cov model per vehicle via symlink
# TODO: add _EXPORT_CP_ZS once a CP NuScenes 100m test export is generated.
# For now cp_zeroshot reuses _EXPORT_CPT (CPft boxes, CP zero-shot covariance model).

# Freshly-fixed homogeneous-pair buckets, REHYDRATED to a persistent merged view.
# The bucket cmr_export/ is split per-vehicle (astuff/, tesla/) with PER-TREE
# center_world centroids; feeding it raw puts the two CAVs ~80 m apart (the S4
# coordinate-drift bug). These dirs hold the test__<scenario>/ output of
# scripts/input_bucket/materialize_merged_view.py (which copies the verified
# merged_view_from_bucket rehydrator's output verbatim), so discover_scenarios /
# run_suite read a single coherent shared-frame scenario. Rebuild with:
#   python scripts/input_bucket/materialize_merged_view.py \
#       --bucket data/v2v4real_inputs/ours/detectors/<bucket> \
#       --out    data/v2v4real_merged_views/<bucket> --force
_MERGED_VIEWS   = (REPO / "data" / "v2v4real_merged_views")
_EXPORT_PP_S0   = _MERGED_VIEWS / "pp_score0"          # PointPillar score-0, both CAVs (rehydrated)
_EXPORT_CPZS    = _MERGED_VIEWS / "cp_zeroshot_100m"   # CenterPoint zero-shot 100m, both CAVs (rehydrated)
_EXPORT_CPFT    = _MERGED_VIEWS / "cp_finetune_100m"   # CenterPoint finetuned 100m, both CAVs (rehydrated)
_EXPORT_MIX_PP_CPZS_V   = _MERGED_VIEWS / "mix_pp_cpzs"     # astuff=PP,   tesla=CPzs
_EXPORT_MIX_CPZS_PP_V   = _MERGED_VIEWS / "mix_cpzs_pp"     # astuff=CPzs, tesla=PP
_EXPORT_MIX_PP_CPFT_V   = _MERGED_VIEWS / "mix_pp_cpft"     # astuff=PP,   tesla=CPft
_EXPORT_MIX_CPFT_PP_V   = _MERGED_VIEWS / "mix_cpft_pp"     # astuff=CPft, tesla=PP
_EXPORT_MIX_CPZS_CPFT_V = _MERGED_VIEWS / "mix_cpzs_cpft"   # astuff=CPzs, tesla=CPft
_EXPORT_MIX_CPFT_CPZS_V = _MERGED_VIEWS / "mix_cpft_cpzs"   # astuff=CPft, tesla=CPzs

_COBEVT_TRACKING_DIR = Path(
    str(REPO / "third_party" / "AB3DMOT" / "results" / "v2v4real" / "cobevt_Car_val_H1" / "data_0"))
_DMSTRACK_GT_DIR = Path(
    "data/v2v4real_inputs/baselines/paper_gt/kitti_labels/v2v4real_val_label")
# Frozen official V2V4Real val GT (kitti_labels symlinks DMSTrack's
# v2v4real_val_label — same bytes the anchor replications were scored on).
_PAPER_GT_LABELS_DIR = REPO / "data" / "v2v4real_inputs" / "baselines" / "paper_gt" / "kitti_labels"
# Our S3 tracker dumps (KITTI MOT) live in the AB3DMOT results tree.
_S3_TRACKS_BASE = REPO / "third_party" / "AB3DMOT" / "results" / "v2v4real"
# DMSTrack's published val tracking output (their learned diff. KF on CoBEVT dets).
# This is what produces the 43.52 AMOTA in their paper.
_DMSTRACK_TRACKING_DIR = Path(
    str((REPO / "third_party" / "AB3DMOT").resolve().parent / "DMSTrack" / "results" / "v2v4real") + "/"
    "reproducing_official_result/"
    "evaluation_multi_sensor_differentiable_kalman_filter_Car_val_all_H1_epoch_0/data_0")
# DMSTrack's tracker run with our GPEM-derived R injected via
# dmstrack_gpem_injector.py. Same architecture, lifecycle, KF, gate as
# DMSTrack — only the per-detection R source swapped from learned net to
# our closed-form GPEM regression. Reproduces V2V AMOTA = 41.26 under
# DMSTrack's evaluator; under our strict eval pulls out Paper AMOTA + HOTA
# for direct comparison.
_DMSTRACK_GPEM_TRACKING_DIR = Path(
    str((REPO / "third_party" / "AB3DMOT").resolve().parent / "DMSTrack" / "results" / "v2v4real") + "/"
    "gpem_r_ablation/"
    "evaluation_multi_sensor_differentiable_kalman_filter_Car_val_all_H1_epoch_0/data_0")
# Our auto-tuned cobevt tracker dumped to KITTI MOT format via
# dump_tracks_to_kitti_mot.py (Tier 1.5 auto-lifecycle, sabre_static stream).
# Lets us run our tracker through the same precomputed_tracks pipeline that
# gives us apples-to-apples Paper AMOTA vs DMSTrack and CoBEVT+AB3DMOT.
_OUR_COBEVT_TRACKING_DIR = Path(
    "/home/rave/test/cmr/results/V2V4Real_OUR_COBEVT_KITTI_DUMP_2026-05-15_200158/data_0")

# Mixed-detector auto-tune KITTI dumps from 2026-05-15_233657 (overnight).
# Each is a per-config tracker output ready to run through the precomputed_tracks
# evaluator for apples-to-apples Paper AMOTA + HOTA vs DMSTrack baseline.
_OUR_DUMP_BASE = Path("/home/rave/test/cmr/results")
_OUR_DUMPS = {
    'cp_zeroshot':        _OUR_DUMP_BASE / 'V2V4Real_KITTI_DUMP_cp_zeroshot_dd_lc_static_2026-05-15_233657'        / 'data_0',
    'cpft_pervehicle':    _OUR_DUMP_BASE / 'V2V4Real_KITTI_DUMP_cpft_pervehicle_dd_ab_static_2026-05-15_233657'    / 'data_0',
    'u_cp_cp_lf':         _OUR_DUMP_BASE / 'V2V4Real_KITTI_DUMP_u_cp_cp_lf_dd_lc_static_2026-05-15_233657'         / 'data_0',
    'u_cpft_cpft_lf':     _OUR_DUMP_BASE / 'V2V4Real_KITTI_DUMP_u_cpft_cpft_lf_dd_ab_static_2026-05-15_233657'     / 'data_0',
    'u_cp_cpft_lf':       _OUR_DUMP_BASE / 'V2V4Real_KITTI_DUMP_u_cp_cpft_lf_dd_lc_static_2026-05-15_233657'       / 'data_0',
    'u_pp_cp_lf':         _OUR_DUMP_BASE / 'V2V4Real_KITTI_DUMP_u_pp_cp_lf_dd_ab_ab3dmot_static_2026-05-15_233657' / 'data_0',
}


# ---------------------------------------------------------------------------
# Canonical benchmark configurations — the single source of truth.
#
# See docs/CONFIG_REFERENCE.md for a 10-second-scan of every entry below
# (which detector / lifecycle / params / why) and the audit trail.
#
# Lifecycle rule of thumb (validated experimentally):
#   - clean cooperative input (CoBEVT) / homogeneous fleet → `log_odds`
#   - heterogeneous mixed-detector fleet                   → `ab3dmot` ma=2
#   - V2V4Real Late-Fusion replica baselines               → `ab3dmot` mh=3 ma=2 st=0.2
# `p_tp_birth_gate` only matters in log_odds; `min_hits/max_age` only in ab3dmot.
# ---------------------------------------------------------------------------

BENCHMARK_CONFIGS: List[Dict] = [
    # ============================================================
    # Single-detector homogeneous — log_odds family
    # ============================================================
    {
        "name": "cp_zeroshot",
        "label": "CenterPoint zero-shot",
        # log_odds lifecycle, bg=0.3 — single-detector homogeneous fleet.
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
        # log_odds lifecycle, bg=0.3, st=0.3 to match PP's training distribution.
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
    # ============================================================
    # Single-detector homogeneous — ab3dmot/LF family (V2V4Real replicas)
    # ============================================================
    {
        "name": "pp_lf_replica",
        "label": "Late Fusion replica (PP + AB3DMOT)",
        # AB3DMOT lifecycle (mh=3, ma=2 from _OUR_DEFAULTS) — V2V4Real Table 4
        # reproduction. Compare paper LF numbers against this.
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
        # AB3DMOT lifecycle (no log_odds variant tuned for CPft yet — see
        # docs/CONFIG_REFERENCE.md §E "known gaps"). max_range=100m matches
        # CPft's fine-tune range; st=0.3 is the calibrated CPft floor.
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
        # AB3DMOT/LF replica baseline for CPft — same boxes as cpft_pervehicle
        # but with V2V4Real LF-table params (mr=50, st=0.2). Use this against
        # published CPft+LF numbers; cpft_pervehicle is for the calibrated-cov
        # comparison instead.
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_54m_v2v4real_finetune_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=50.0,
            score_threshold=0.2,
        ),
    },
    # ============================================================
    # CoBEVT cooperative (single-ego, headline paper config)
    # - Untuned uses ab3dmot lifecycle (matches V2V4Real H1, max_age=1).
    # - Tuned uses log_odds (train sweep showed +2.5 V2V on test).
    # - See docs/CONFIG_REFERENCE.md §C for the apples-to-apples comparison
    #   vs CoBEVT+AB3DMOT published 38.77.
    # ============================================================
    {
        "name": "cobevt_dets_cmr_se",
        "label": "CoBEVT dets + GPEM tracker (single-ego, untuned)",
        # Untuned snapshot baseline: ab3dmot lifecycle, max_age=1 (H1) — matches
        # V2V4Real's AB3DMOT setup. Reference snapshot at
        # results/V2V4Real_COBEVT_SE_UNTUNED_BASELINE/ (MG-V2V 38.89).
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
        "name": "cobevt_dets_cmr_se_tuned",
        "label": "CoBEVT dets + GPEM tracker (single-ego, train-tuned)",
        # TUNED HEADLINE: log_odds lifecycle, st=0.0 wins by +2.5 V2V over 0.3.
        # Train sweep (18 combos × 30 train scenarios) showed bg/md have ZERO
        # effect on CoBEVT input (its internal score floor of 0.2 makes lifecycle
        # gates trivially satisfied). Reference snapshot at
        # results/V2V4Real_COBEVT_SE_TUNED_FINAL/ — MG-V2V 41.19 (sabre_static),
        # beats published CoBEVT+AB3DMOT 38.77 by +2.4.
        "kind": "our",
        "export_dir": _EXPORT_COBEVT_SE,
        "params": dict(
            detector_name="cobevt_tracker_{vehicle}",
            lifecycle_mode="log_odds",   # tuned: log_odds (sweep showed bg/md don't matter; this is canonical mode)
            detector_max_range=100.0,
            p_tp_birth_gate=0.25,        # tuned: any 0.1-0.4 yields identical results
            score_threshold=0.0,         # tuned: 0.0 wins by 2.5 V2V over 0.3
            self_report_egos=False,
        ),
    },
    # ============================================================
    # GPEM data-driven birth-gate variants (sigmoid + per-cov lifecycle).
    # Loads per-detector quadratic curve from
    # results/birth_gate_calibration/birth_gate_curves.csv:
    #     gate(d) = a·d² + b·d + c   (TP-preserving formula, α from config)
    # At det creation: p_tp = σ(k · (score − gate(d))). Score=gate → p_tp=0.5
    # (neutral). Above-gate scores → smooth positive log-odds; below-gate →
    # negative. Birth-gate threshold p_tp_birth_gate=0.5 means "score ≥ gate".
    # Per-cov lifecycle: baseline cov uses ab3dmot (count-based, no p_tp
    # dependency since flat cov can't exploit per-detection signal); GPEM
    # cov modes use log_odds with the calibrated gate. α=1: ~84% TP
    # retention; α=2: ~97.7%. See docs/GPEM_BIRTH_GATE.md.
    # ============================================================
    {
        "name": "cobevt_dets_cmr_se_dd_a1",
        "label": "CoBEVT GPEM tracker + data-driven gate α=1",
        "kind": "our",
        "export_dir": _EXPORT_COBEVT_SE,
        "params": dict(
            detector_name="cobevt_tracker_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,         # sigmoid-centered: 0.5 = "score ≥ gate"
            score_threshold=0.0,
            data_driven_gate_alpha=1.0,
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},  # native lifecycle for flat cov
            self_report_egos=False,
        ),
    },
    {
        "name": "cobevt_dets_cmr_se_dd_a2",
        "label": "CoBEVT GPEM tracker + data-driven gate α=2",
        "kind": "our",
        "export_dir": _EXPORT_COBEVT_SE,
        "params": dict(
            detector_name="cobevt_tracker_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=2.0,
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
            self_report_egos=False,
        ),
    },
    {
        "name": "cp_zeroshot_dd_a1",
        "label": "CP zero-shot + data-driven gate α=1",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_100m_on_v2v4real",
            lifecycle_mode="log_odds",
            detector_max_range=50.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=1.0,
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "cp_zeroshot_dd_a2",
        "label": "CP zero-shot + data-driven gate α=2",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_100m_on_v2v4real",
            lifecycle_mode="log_odds",
            detector_max_range=50.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=2.0,
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "pp_score02_dd_a1",
        "label": "PP score02 per-vehicle + data-driven gate α=1",
        "kind": "our",
        "export_dir": _EXPORT_PP,
        "params": dict(
            detector_name="pointpillar_v2v4real_score02_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=50.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=1.0,
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "pp_score02_dd_a2",
        "label": "PP score02 per-vehicle + data-driven gate α=2",
        "kind": "our",
        "export_dir": _EXPORT_PP,
        "params": dict(
            detector_name="pointpillar_v2v4real_score02_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=50.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=2.0,
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "cobevt_dets_cmr",
        "label": "CoBEVT dets + GPEM tracker (per-vehicle dup)",
        # Per-vehicle dup of cooperative dets. show_in_table=False because doubled
        # GT eval is misleading (same boxes counted twice in per_ego mode).
        # Use cobevt_dets_cmr_se / _se_tuned for headline single-ego comparisons,
        # or cobevt_dets_cmr_2veh_tuned (below) for the 6-variant GT-merge grid.
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
        # Same as cobevt_dets_cmr_se_tuned but on 2-vehicle export so the 6-variant
        # GT-merge grid (per_ego/world × sort/cav0/average) actually differs across
        # cells. Used for the master 3-setup head-to-head comparison vs
        # AB3DMOT/DMSTrack precomputed tracks.
        "name": "cobevt_dets_cmr_2veh_tuned",
        "label": "CoBEVT dets + GPEM tracker (2-vehicle GT, train-tuned)",
        "kind": "our",
        "export_dir": _EXPORT_COBEVT,    # per-vehicle-dup (BOTH vehicles' GT present)
        "params": dict(
            detector_name="cobevt_tracker_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.25,
            score_threshold=0.0,
            self_report_egos=False,
        ),
    },
    # ============================================================
    # Cross-detector mixes (heterogeneous fleet experiments)
    # - Each pair has 2 configs: log_odds (paper-style) + _lf (AB3DMOT replica).
    # - Validated finding: ab3dmot/LF wins on heterogeneous input by avg
    #   +2.98 V2V across the 6 pairs (mismatched score distributions accumulate
    #   bad tracks under permissive log_odds; ma=2 prunes them aggressively).
    # - bg=0.25 in log_odds variants is a generic default (NOT calibrated
    #   per-detector); LF variants inherit _OUR_DEFAULTS mh=3 ma=2 which IS the
    #   V2V4Real LF setup.
    # - Symlinks in src/data/sensor_models/mix_*_{astuff,tesla}.csv point at
    #   the real per-vehicle sensor model for each combo.
    # ============================================================
    {
        "name": "mix_pp_cpft",
        "label": "Mix: PP-astuff + CPft-tesla (log_odds)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPFT,
        "params": dict(
            detector_name="mix_pp_cpft_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=50.0,
            p_tp_birth_gate=0.25,
            score_threshold=0.2,
        ),
    },
    {
        "name": "mix_pp_cpft_lf",
        "label": "Mix: PP-astuff + CPft-tesla (LF/AB3DMOT)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPFT,
        "params": dict(
            detector_name="mix_pp_cpft_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=50.0,
            score_threshold=0.2,
        ),
    },
    {
        "name": "mix_cpft_pp",
        "label": "Mix: CPft-astuff + PP-tesla (log_odds)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_CPFT_PP,
        "params": dict(
            detector_name="mix_cpft_pp_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=50.0,
            p_tp_birth_gate=0.25,
            score_threshold=0.2,
        ),
    },
    {
        "name": "mix_cpft_pp_lf",
        "label": "Mix: CPft-astuff + PP-tesla (LF/AB3DMOT)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_CPFT_PP,
        "params": dict(
            detector_name="mix_cpft_pp_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=50.0,
            score_threshold=0.2,
        ),
    },
    {
        "name": "mix_pp_cpzs",
        "label": "Mix: PP-astuff + CPzs-tesla (log_odds)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPZS,
        "params": dict(
            detector_name="mix_pp_cpzs_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=50.0,
            p_tp_birth_gate=0.25,
            score_threshold=0.2,
        ),
    },
    {
        "name": "mix_pp_cpzs_lf",
        "label": "Mix: PP-astuff + CPzs-tesla (LF/AB3DMOT)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPZS,
        "params": dict(
            detector_name="mix_pp_cpzs_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=50.0,
            score_threshold=0.2,
        ),
    },
    {
        "name": "mix_cpzs_pp",
        "label": "Mix: CPzs-astuff + PP-tesla (log_odds)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_CPZS_PP,
        "params": dict(
            detector_name="mix_cpzs_pp_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=50.0,
            p_tp_birth_gate=0.25,
            score_threshold=0.2,
        ),
    },
    {
        "name": "mix_cpzs_pp_lf",
        "label": "Mix: CPzs-astuff + PP-tesla (LF/AB3DMOT)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_CPZS_PP,
        "params": dict(
            detector_name="mix_cpzs_pp_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=50.0,
            score_threshold=0.2,
        ),
    },
    {
        "name": "mix_cpft_cpzs",
        "label": "Mix: CPft-astuff + CPzs-tesla (log_odds)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,  # cov-only mix: same CPft boxes, different cov via symlink
        "params": dict(
            detector_name="mix_cpft_cpzs_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=50.0,
            p_tp_birth_gate=0.25,
            score_threshold=0.2,
        ),
    },
    {
        "name": "mix_cpft_cpzs_lf",
        "label": "Mix: CPft-astuff + CPzs-tesla (LF/AB3DMOT)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="mix_cpft_cpzs_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=50.0,
            score_threshold=0.2,
        ),
    },
    {
        "name": "mix_cpzs_cpft",
        "label": "Mix: CPzs-astuff + CPft-tesla (log_odds)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="mix_cpzs_cpft_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=50.0,
            p_tp_birth_gate=0.25,
            score_threshold=0.2,
        ),
    },
    {
        "name": "mix_cpzs_cpft_lf",
        "label": "Mix: CPzs-astuff + CPft-tesla (LF/AB3DMOT)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="mix_cpzs_cpft_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=50.0,
            score_threshold=0.2,
        ),
    },
    # ----- end cross-detector mix configs -----
    # ============================================================
    # Unified-params experiment (max_r=100m, log_odds bg=0.3 st=0.3,
    # ab3dmot LF replica with default min_hits=3 max_age=2 st=0.2).
    # 9 detector pairings × 2 lifecycles = 18 configs.
    # Goal: head-to-head LF-replica vs GPEM (log_odds) per pairing, with one
    # set of params across the matrix so only the lifecycle differs (controlled
    # ablation). Differs from the mix_* block above which uses st=0.2 / mr=50.
    # ============================================================
    # ---- Homogeneous (2x same detector) ----
    {
        "name": "u_pp_pp_log_odds",
        "label": "Unif: 2xPP (log_odds + GPEM)",
        "kind": "our",
        "export_dir": _EXPORT_PP,
        "params": dict(
            detector_name="u_pp_pp_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.3,
            score_threshold=0.3,
        ),
    },
    {
        "name": "u_pp_pp_lf",
        "label": "Unif: 2xPP (LF/AB3DMOT)",
        "kind": "our",
        "export_dir": _EXPORT_PP,
        "params": dict(
            detector_name="u_pp_pp_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.2,
        ),
    },
    {
        "name": "u_cp_cp_log_odds",
        "label": "Unif: 2xCP-zs (log_odds + GPEM)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,    # CPft boxes; CPzs cov via symlink
        "params": dict(
            detector_name="u_cp_cp_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.3,
            score_threshold=0.3,
        ),
    },
    {
        "name": "u_cp_cp_lf",
        "label": "Unif: 2xCP-zs (LF/AB3DMOT)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="u_cp_cp_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.2,
        ),
    },
    {
        "name": "u_cpft_cpft_log_odds",
        "label": "Unif: 2xCPft (log_odds + GPEM)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="u_cpft_cpft_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.3,
            score_threshold=0.3,
        ),
    },
    {
        "name": "u_cpft_cpft_lf",
        "label": "Unif: 2xCPft (LF/AB3DMOT)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="u_cpft_cpft_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.2,
        ),
    },
    # ---- Heterogeneous mixes (1x + 1x) ----
    {
        "name": "u_pp_cp_log_odds",
        "label": "Unif: PP+CP-zs (log_odds + GPEM)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPZS,
        "params": dict(
            detector_name="mix_pp_cpzs_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.3,
            score_threshold=0.3,
        ),
    },
    {
        "name": "u_pp_cp_lf",
        "label": "Unif: PP+CP-zs (LF/AB3DMOT)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPZS,
        "params": dict(
            detector_name="mix_pp_cpzs_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.2,
        ),
    },
    {
        "name": "u_pp_cp_lf_dd_a1",
        "label": "Unif: PP+CP-zs (LF/AB3DMOT) + data-driven gate α=1",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPZS,
        "params": dict(
            detector_name="mix_pp_cpzs_{vehicle}",
            lifecycle_mode="log_odds",            # gate-friendly lifecycle
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,                  # sigmoid: 0.5 = "score ≥ gate"
            score_threshold=0.0,
            data_driven_gate_alpha=1.0,
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_pp_cp_lf_dd_a2",
        "label": "Unif: PP+CP-zs (LF/AB3DMOT) + data-driven gate α=2",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPZS,
        "params": dict(
            detector_name="mix_pp_cpzs_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=2.0,
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    # ============================================================
    # AMOTA-Bayes data-driven gate variants (closed-form, no α sweep).
    # Per-bin λ = fp_lifetime_mean / tp_lifetime_mean from the new 69-col
    # characterization. Quadratic-form-only at runtime; alpha sentinels:
    #   alpha=97.0 → static (constant gate = scalar median across bins)
    #   alpha=98.0 → linear (gate(r) = a*r + b)
    #   alpha=99.0 → quadratic (gate(r) = a*r² + b*r + c)
    # See docs/CHARACTERIZATION_FINDINGS.md for the math.
    # ============================================================
    {
        "name": "cobevt_dets_cmr_se_dd_ab_static",
        "label": "CoBEVT GPEM tracker + AMOTA-Bayes (static)",
        "kind": "our",
        "export_dir": _EXPORT_COBEVT_SE,
        "params": dict(
            detector_name="cobevt_tracker_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,         # static (constant gate)
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
            self_report_egos=False,
        ),
    },
    {
        "name": "cobevt_dets_cmr_se_dd_ab_linear",
        "label": "CoBEVT GPEM tracker + AMOTA-Bayes (linear)",
        "kind": "our",
        "export_dir": _EXPORT_COBEVT_SE,
        "params": dict(
            detector_name="cobevt_tracker_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=98.0,         # linear gate
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
            self_report_egos=False,
        ),
    },
    {
        "name": "cobevt_dets_cmr_se_dd_ab_quad",
        "label": "CoBEVT GPEM tracker + AMOTA-Bayes (quadratic)",
        "kind": "our",
        "export_dir": _EXPORT_COBEVT_SE,
        "params": dict(
            detector_name="cobevt_tracker_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=99.0,         # quadratic gate
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
            self_report_egos=False,
        ),
    },
    {
        "name": "cp_zeroshot_dd_ab_static",
        "label": "CP zero-shot + AMOTA-Bayes (static)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_100m_on_v2v4real",
            lifecycle_mode="log_odds",
            detector_max_range=50.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "cp_zeroshot_dd_ab_linear",
        "label": "CP zero-shot + AMOTA-Bayes (linear)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_100m_on_v2v4real",
            lifecycle_mode="log_odds",
            detector_max_range=50.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=98.0,
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "cp_zeroshot_dd_ab_quad",
        "label": "CP zero-shot + AMOTA-Bayes (quadratic)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_100m_on_v2v4real",
            lifecycle_mode="log_odds",
            detector_max_range=50.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=99.0,
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    # Gate 1 expansion: AMOTA-Bayes static gate on CPft + cooperative
    # heterogeneous mixes — the configs we expect the gate to shine on
    # (per-vehicle detectors with normal noise stats, not cooperative-fused).
    {
        "name": "cpft_pervehicle_dd_ab_static",
        "label": "CPft per-vehicle + AMOTA-Bayes (static)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_54m_v2v4real_finetune_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_cp_cp_lf_dd_ab_static",
        "label": "Unif: 2xCP-zs + AMOTA-Bayes (static)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="u_cp_cp_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_cpft_cpft_lf_dd_ab_static",
        "label": "Unif: 2xCPft + AMOTA-Bayes (static)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="u_cpft_cpft_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_cp_cpft_lf_dd_ab_static",
        "label": "Unif: CP-zs+CPft + AMOTA-Bayes (static, heterogeneous mix)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="mix_cpzs_cpft_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    # Linear + quadratic for the 4 new detectors (matching the static set)
    {
        "name": "cpft_pervehicle_dd_ab_linear",
        "label": "CPft per-vehicle + AMOTA-Bayes (linear)",
        "kind": "our", "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_54m_v2v4real_finetune_{vehicle}",
            lifecycle_mode="log_odds", detector_max_range=100.0,
            p_tp_birth_gate=0.5, score_threshold=0.0,
            data_driven_gate_alpha=98.0, data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "cpft_pervehicle_dd_ab_quad",
        "label": "CPft per-vehicle + AMOTA-Bayes (quadratic)",
        "kind": "our", "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_54m_v2v4real_finetune_{vehicle}",
            lifecycle_mode="log_odds", detector_max_range=100.0,
            p_tp_birth_gate=0.5, score_threshold=0.0,
            data_driven_gate_alpha=99.0, data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_cp_cp_lf_dd_ab_linear",
        "label": "Unif: 2xCP-zs + AMOTA-Bayes (linear)",
        "kind": "our", "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="u_cp_cp_{vehicle}",
            lifecycle_mode="log_odds", detector_max_range=100.0,
            p_tp_birth_gate=0.5, score_threshold=0.0,
            data_driven_gate_alpha=98.0, data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_cp_cp_lf_dd_ab_quad",
        "label": "Unif: 2xCP-zs + AMOTA-Bayes (quadratic)",
        "kind": "our", "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="u_cp_cp_{vehicle}",
            lifecycle_mode="log_odds", detector_max_range=100.0,
            p_tp_birth_gate=0.5, score_threshold=0.0,
            data_driven_gate_alpha=99.0, data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_cpft_cpft_lf_dd_ab_linear",
        "label": "Unif: 2xCPft + AMOTA-Bayes (linear)",
        "kind": "our", "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="u_cpft_cpft_{vehicle}",
            lifecycle_mode="log_odds", detector_max_range=100.0,
            p_tp_birth_gate=0.5, score_threshold=0.0,
            data_driven_gate_alpha=98.0, data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_cpft_cpft_lf_dd_ab_quad",
        "label": "Unif: 2xCPft + AMOTA-Bayes (quadratic)",
        "kind": "our", "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="u_cpft_cpft_{vehicle}",
            lifecycle_mode="log_odds", detector_max_range=100.0,
            p_tp_birth_gate=0.5, score_threshold=0.0,
            data_driven_gate_alpha=99.0, data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_cp_cpft_lf_dd_ab_linear",
        "label": "Unif: CP-zs+CPft + AMOTA-Bayes (linear, heterogeneous mix)",
        "kind": "our", "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="mix_cpzs_cpft_{vehicle}",
            lifecycle_mode="log_odds", detector_max_range=100.0,
            p_tp_birth_gate=0.5, score_threshold=0.0,
            data_driven_gate_alpha=98.0, data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_cp_cpft_lf_dd_ab_quad",
        "label": "Unif: CP-zs+CPft + AMOTA-Bayes (quadratic, heterogeneous mix)",
        "kind": "our", "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="mix_cpzs_cpft_{vehicle}",
            lifecycle_mode="log_odds", detector_max_range=100.0,
            p_tp_birth_gate=0.5, score_threshold=0.0,
            data_driven_gate_alpha=99.0, data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    # Polar mode (alpha=96.0): per-(range, angle) bin lookup at runtime, no
    # smoothing — mirrors GPEM cov polar mode. Highest fidelity, no fit-error.
    {
        "name": "cobevt_dets_cmr_se_dd_ab_polar",
        "label": "CoBEVT GPEM tracker + AMOTA-Bayes (polar)",
        "kind": "our",
        "export_dir": _EXPORT_COBEVT_SE,
        "params": dict(
            detector_name="cobevt_tracker_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=96.0,
            data_driven_gate_k=10.0,
            data_driven_gate_fit="polar",
            per_cov_lifecycle={"baseline": "ab3dmot"},
            self_report_egos=False,
        ),
    },
    {
        "name": "cp_zeroshot_dd_ab_polar",
        "label": "CP zero-shot + AMOTA-Bayes (polar)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_100m_on_v2v4real",
            lifecycle_mode="log_odds",
            detector_max_range=50.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=96.0,
            data_driven_gate_k=10.0,
            data_driven_gate_fit="polar",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "cpft_pervehicle_dd_ab_polar",
        "label": "CPft per-vehicle + AMOTA-Bayes (polar)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_54m_v2v4real_finetune_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=96.0,
            data_driven_gate_k=10.0,
            data_driven_gate_fit="polar",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_cp_cp_lf_dd_ab_polar",
        "label": "Unif: 2xCP-zs + AMOTA-Bayes (polar)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="u_cp_cp_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=96.0,
            data_driven_gate_k=10.0,
            data_driven_gate_fit="polar",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_cpft_cpft_lf_dd_ab_polar",
        "label": "Unif: 2xCPft + AMOTA-Bayes (polar)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="u_cpft_cpft_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=96.0,
            data_driven_gate_k=10.0,
            data_driven_gate_fit="polar",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_cp_cpft_lf_dd_ab_polar",
        "label": "Unif: CP-zs+CPft + AMOTA-Bayes (polar, heterogeneous mix)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="mix_cpzs_cpft_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=96.0,
            data_driven_gate_k=10.0,
            data_driven_gate_fit="polar",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    # ── PP + CP-zs heterogeneous (master.csv top performer: 39.35 V2V AMOTA
    #    in u_pp_cp_lf with ab3dmot lifecycle). These dd_ab variants test
    #    whether log_odds + auto-derived gate can match that result.
    {
        "name": "u_pp_cp_lf_dd_ab_static",
        "label": "Unif: PP+CP-zs + AMOTA-Bayes (static, heterogeneous mix)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPZS,
        "params": dict(
            detector_name="mix_pp_cpzs_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_pp_cp_lf_dd_ab_linear",
        "label": "Unif: PP+CP-zs + AMOTA-Bayes (linear, heterogeneous mix)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPZS,
        "params": dict(
            detector_name="mix_pp_cpzs_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=98.0,
            data_driven_gate_k=10.0,
            data_driven_gate_fit="linear",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_pp_cp_lf_dd_ab_quad",
        "label": "Unif: PP+CP-zs + AMOTA-Bayes (quadratic, heterogeneous mix)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPZS,
        "params": dict(
            detector_name="mix_pp_cpzs_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=99.0,
            data_driven_gate_k=10.0,
            data_driven_gate_fit="quadratic",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_pp_cp_lf_dd_ab_polar",
        "label": "Unif: PP+CP-zs + AMOTA-Bayes (polar, heterogeneous mix)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPZS,
        "params": dict(
            detector_name="mix_pp_cpzs_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=96.0,
            data_driven_gate_k=10.0,
            data_driven_gate_fit="polar",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    # ── AMOTA-Bayes gate + AB3DMOT lifecycle variants ──
    # Mirrors the dd_ab_static configs but with `lifecycle_mode="ab3dmot"`.
    # Test hypothesis: variant B in LIFECYCLE_TEST showed ab3dmot beats log_odds
    # by +1.5 V2V at apples-to-apples on cp_zeroshot. Does our auto-gate also
    # benefit from ab3dmot? Note: ab3dmot's min_hits=3 confirmation stacks on
    # top of our birth gate — may double-filter for case-A detectors (CP-zs
    # gate=0.29) but is essentially a no-op for case-B (cobevt/CPft gate=0).
    {
        "name": "cobevt_dets_cmr_se_dd_ab_ab3dmot_static",
        "label": "CoBEVT GPEM tracker + AMOTA-Bayes gate + AB3DMOT lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_COBEVT_SE,
        "params": dict(
            detector_name="cobevt_tracker_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
        ),
    },
    {
        "name": "cp_zeroshot_dd_ab_ab3dmot_static",
        "label": "CP-zs + AMOTA-Bayes gate + AB3DMOT lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_100m_on_v2v4real",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
        ),
    },
    {
        "name": "cpft_pervehicle_dd_ab_ab3dmot_static",
        "label": "CPft per-vehicle + AMOTA-Bayes gate + AB3DMOT lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_54m_v2v4real_finetune_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
        ),
    },
    {
        "name": "u_cp_cp_lf_dd_ab_ab3dmot_static",
        "label": "Unif: 2xCP-zs + AMOTA-Bayes gate + AB3DMOT lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="u_cp_cp_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
        ),
    },
    {
        "name": "u_cpft_cpft_lf_dd_ab_ab3dmot_static",
        "label": "Unif: 2xCPft + AMOTA-Bayes gate + AB3DMOT lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="u_cpft_cpft_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
        ),
    },
    {
        "name": "u_cp_cpft_lf_dd_ab_ab3dmot_static",
        "label": "Unif: CP-zs+CPft + AMOTA-Bayes gate + AB3DMOT lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="mix_cpzs_cpft_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
        ),
    },
    {
        "name": "u_pp_cp_lf_dd_ab_ab3dmot_static",
        "label": "Unif: PP+CP-zs + AMOTA-Bayes gate + AB3DMOT lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPZS,
        "params": dict(
            detector_name="mix_pp_cpzs_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
        ),
    },
    {
        "name": "u_cp_pp_log_odds",
        "label": "Unif: CP-zs+PP (log_odds + GPEM)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_CPZS_PP,
        "params": dict(
            detector_name="mix_cpzs_pp_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.3,
            score_threshold=0.3,
        ),
    },
    # ── Tier 1 data-driven LIFECYCLE configs ────────────────────────────
    # Auto-gate (AMOTA-Bayes static, alpha=97) + auto-lifecycle (per-detector
    # log_lr_miss derived from natural TP miss rate & lifetime). Compare
    # against the matching `*_dd_ab_static` configs to isolate the lifecycle
    # effect from the gate.
    {
        "name": "cobevt_dets_cmr_se_dd_lc_static",
        "label": "CoBEVT GPEM tracker + AMOTA-Bayes gate + auto-lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_COBEVT_SE,
        "params": dict(
            detector_name="cobevt_tracker_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
            data_driven_lifecycle=True,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "cp_zeroshot_dd_lc_static",
        "label": "CP-zs + AMOTA-Bayes gate + auto-lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_100m_on_v2v4real",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
            data_driven_lifecycle=True,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    # α-slope sweep on cp_zeroshot. Tests whether a gentle range-dependent slope
    # around the scalar P25 (0.29) beats the flat baseline. Paired with Tier 1.5
    # lifecycle (the now-working auto-confirm + auto-tape). Encoded as synthetic
    # alpha values in birth_gate_curves.csv:
    #   α=89.0 → flat gate=0.29 (baseline, identical to α=97 static)
    #   α=89.1 → slope=-0.005 (gate: 0.54 near → 0.04 far)
    #   α=89.2 → slope=-0.010 (gate: 0.79 near → 0   far, matches linear-fit slope)
    #   α=89.3 → slope=+0.005 (inverted: 0.04 near → 0.54 far — sanity check)
    {
        "name": "cp_zeroshot_alpha_sweep_flat",
        "label": "CP-zs α-sweep: flat 0.29 (baseline) + auto-lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_100m_on_v2v4real",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=89.0,
            data_driven_gate_fit="linear",
            data_driven_gate_k=10.0,
            data_driven_lifecycle=True,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "cp_zeroshot_alpha_sweep_neg005",
        "label": "CP-zs α-sweep: gentle -0.005 + auto-lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_100m_on_v2v4real",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=89.1,
            data_driven_gate_fit="linear",
            data_driven_gate_k=10.0,
            data_driven_lifecycle=True,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "cp_zeroshot_alpha_sweep_neg010",
        "label": "CP-zs α-sweep: steep -0.010 + auto-lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_100m_on_v2v4real",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=89.2,
            data_driven_gate_fit="linear",
            data_driven_gate_k=10.0,
            data_driven_lifecycle=True,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "cp_zeroshot_alpha_sweep_pos005",
        "label": "CP-zs α-sweep: inverted +0.005 (sanity) + auto-lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_100m_on_v2v4real",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=89.3,
            data_driven_gate_fit="linear",
            data_driven_gate_k=10.0,
            data_driven_lifecycle=True,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "cpft_pervehicle_dd_lc_static",
        "label": "CPft per-vehicle + AMOTA-Bayes gate + auto-lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_54m_v2v4real_finetune_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
            data_driven_lifecycle=True,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_cp_cp_lf_dd_lc_static",
        "label": "Unif: 2xCP-zs + AMOTA-Bayes gate + auto-lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="u_cp_cp_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
            data_driven_lifecycle=True,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_cpft_cpft_lf_dd_lc_static",
        "label": "Unif: 2xCPft + AMOTA-Bayes gate + auto-lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="u_cpft_cpft_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
            data_driven_lifecycle=True,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_cp_cpft_lf_dd_lc_static",
        "label": "Unif: CP-zs+CPft + AMOTA-Bayes gate + auto-lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="mix_cpzs_cpft_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
            data_driven_lifecycle=True,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_pp_cp_lf_dd_lc_static",
        "label": "Unif: PP+CP-zs + AMOTA-Bayes gate + auto-lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPZS,
        "params": dict(
            detector_name="mix_pp_cpzs_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
            data_driven_lifecycle=True,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    # ── Homogeneous PP+PP auto-tuned variants (DMSTrack head-to-head set) ──
    # DMSTrack's published 43.52 V2V AMOTA uses per-vehicle PointPillar with
    # their learned-R differentiable KF. These configs match the same
    # detector backbone (PP on both vehicles) and let us compare to their
    # number under matched conditions. Auto-gate=0 for case-B-dominant PP;
    # the polar mode is the more interesting variant since per-cell s* can
    # be nonzero where local case-A bins exist.
    {
        "name": "u_pp_pp_lf_dd_ab_static",
        "label": "Unif: 2xPP + AMOTA-Bayes (static)",
        "kind": "our",
        "export_dir": _EXPORT_PP,
        "params": dict(
            detector_name="u_pp_pp_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_pp_pp_lf_dd_ab_linear",
        "label": "Unif: 2xPP + AMOTA-Bayes (linear)",
        "kind": "our",
        "export_dir": _EXPORT_PP,
        "params": dict(
            detector_name="u_pp_pp_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=98.0,
            data_driven_gate_k=10.0,
            data_driven_gate_fit="linear",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_pp_pp_lf_dd_ab_quad",
        "label": "Unif: 2xPP + AMOTA-Bayes (quadratic)",
        "kind": "our",
        "export_dir": _EXPORT_PP,
        "params": dict(
            detector_name="u_pp_pp_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=99.0,
            data_driven_gate_k=10.0,
            data_driven_gate_fit="quadratic",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_pp_pp_lf_dd_ab_polar",
        "label": "Unif: 2xPP + AMOTA-Bayes (polar)",
        "kind": "our",
        "export_dir": _EXPORT_PP,
        "params": dict(
            detector_name="u_pp_pp_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=96.0,
            data_driven_gate_k=10.0,
            data_driven_gate_fit="polar",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_pp_pp_lf_dd_ab_ab3dmot_static",
        "label": "Unif: 2xPP + AMOTA-Bayes gate + AB3DMOT lifecycle",
        "kind": "our",
        "export_dir": _EXPORT_PP,
        "params": dict(
            detector_name="u_pp_pp_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
        ),
    },
    {
        "name": "u_pp_pp_lf_dd_lc_static",
        "label": "Unif: 2xPP + AMOTA-Bayes gate + auto-lifecycle (Tier 1.5)",
        "kind": "our",
        "export_dir": _EXPORT_PP,
        "params": dict(
            detector_name="u_pp_pp_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,
            score_threshold=0.0,
            data_driven_gate_alpha=97.0,
            data_driven_gate_k=10.0,
            data_driven_lifecycle=True,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "u_cp_pp_lf",
        "label": "Unif: CP-zs+PP (LF/AB3DMOT)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_CPZS_PP,
        "params": dict(
            detector_name="mix_cpzs_pp_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.2,
        ),
    },
    {
        "name": "u_pp_cpft_log_odds",
        "label": "Unif: PP+CPft (log_odds + GPEM)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPFT,
        "params": dict(
            detector_name="mix_pp_cpft_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.3,
            score_threshold=0.3,
        ),
    },
    {
        "name": "u_pp_cpft_lf",
        "label": "Unif: PP+CPft (LF/AB3DMOT)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPFT,
        "params": dict(
            detector_name="mix_pp_cpft_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.2,
        ),
    },
    {
        "name": "u_cpft_pp_log_odds",
        "label": "Unif: CPft+PP (log_odds + GPEM)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_CPFT_PP,
        "params": dict(
            detector_name="mix_cpft_pp_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.3,
            score_threshold=0.3,
        ),
    },
    {
        "name": "u_cpft_pp_lf",
        "label": "Unif: CPft+PP (LF/AB3DMOT)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_CPFT_PP,
        "params": dict(
            detector_name="mix_cpft_pp_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.2,
        ),
    },
    {
        "name": "u_cp_cpft_log_odds",
        "label": "Unif: CP-zs+CPft (log_odds + GPEM)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,    # same boxes both vehicles, only cov differs
        "params": dict(
            detector_name="mix_cpzs_cpft_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.3,
            score_threshold=0.3,
        ),
    },
    {
        "name": "u_cp_cpft_lf",
        "label": "Unif: CP-zs+CPft (LF/AB3DMOT)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="mix_cpzs_cpft_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.2,
        ),
    },
    {
        "name": "u_cpft_cp_log_odds",
        "label": "Unif: CPft+CP-zs (log_odds + GPEM)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="mix_cpft_cpzs_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.3,
            score_threshold=0.3,
        ),
    },
    {
        "name": "u_cpft_cp_lf",
        "label": "Unif: CPft+CP-zs (LF/AB3DMOT)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="mix_cpft_cpzs_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.2,
        ),
    },
    # ----- end unified-params experiment -----
    # ============================================================
    # External baselines (precomputed eval paths)
    # - These configs do not run our tracker — they evaluate pre-existing
    #   KITTI MOT output through the same paper / v2v_mg / HOTA grid.
    # - The (2D | 3D | DMSTrack-exact) triplet is parallel for AB3DMOT and
    #   DMSTrack so apples-to-apples comparisons are possible.
    # - dmstrack_exact = 3D-IoU + convex_hull + AB3DMOT rematching + global
    #   eval (all 9 sequences combined). Reproduces published 37.16 (AB3DMOT)
    #   and 43.52 (DMSTrack) within ~0.1.
    # ============================================================
    {
        "name": "lf_dmstrack",
        "label": "Late Fusion (DMSTrack-bundled)",
        # DMSTrack's bundled LF baseline path (uses their eval, not precomputed).
        # Hidden — superseded by pp_lf_replica which we control end-to-end.
        "kind": "dmstrack_lf",
        "show_in_table": False,
        "params": {},
    },
    {
        "name": "cobevt_dmstrack",
        "label": "CoBEVT (DMSTrack-bundled)",
        # DMSTrack's bundled CoBEVT path (uses their eval, not precomputed).
        # Hidden — superseded by cobevt_precomputed* configs.
        "kind": "dmstrack_cobevt",
        "show_in_table": False,
        "params": {},
    },
    {
        "name": "cobevt_precomputed",
        "label": "CoBEVT+AB3DMOT (official, 2D-IoU per-seq)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _COBEVT_TRACKING_DIR,
            "gt_dir": _DMSTRACK_GT_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": False,
        },
    },
    {
        "name": "cobevt_precomputed_3d",
        "label": "CoBEVT+AB3DMOT (3D-IoU per-seq)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _COBEVT_TRACKING_DIR,
            "gt_dir": _DMSTRACK_GT_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": True,
            "convex_hull": True,
        },
    },
    {
        # DMSTrack-exact replica: global eval (all 9 sequences combined into
        # one tape) + 3D-IoU + convex_hull + AB3DMOT rematching. Should
        # reproduce the published 37.16 V2V AMOTA within ~0.1.
        "name": "cobevt_precomputed_dmstrack_exact",
        "label": "CoBEVT+AB3DMOT (DMSTrack-exact: 3D + global + rematching)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _COBEVT_TRACKING_DIR,
            "gt_dir": _DMSTRACK_GT_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": True,
            "convex_hull": True,
            "ab3dmot_rematching": True,
            "global_eval": True,
        },
    },
    # DMSTrack's published val tracking output, evaluated through our pipeline.
    # Three protocol variants (2D per-seq, 3D per-seq, DMSTrack-exact) for
    # parallel comparison with the cobevt_precomputed* configs above.
    {
        "name": "dmstrack_precomputed",
        "label": "CoBEVT+DMSTrack (DMSTrack-bundled tracks, 2D-IoU per-seq)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _DMSTRACK_TRACKING_DIR,
            "gt_dir": _DMSTRACK_GT_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": False,
        },
    },
    {
        "name": "dmstrack_precomputed_3d",
        "label": "CoBEVT+DMSTrack (DMSTrack-bundled tracks, 3D-IoU per-seq)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _DMSTRACK_TRACKING_DIR,
            "gt_dir": _DMSTRACK_GT_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": True,
            "convex_hull": True,
        },
    },
    {
        # Should reproduce the published 43.52 V2V AMOTA within ~0.1.
        "name": "dmstrack_precomputed_dmstrack_exact",
        "label": "CoBEVT+DMSTrack (DMSTrack-exact: 3D + global + rematching)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _DMSTRACK_TRACKING_DIR,
            "gt_dir": _DMSTRACK_GT_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": True,
            "convex_hull": True,
            "ab3dmot_rematching": True,
            "global_eval": True,
        },
    },
    # DMSTrack-with-GPEM-R ablation: same architecture/lifecycle as DMSTrack,
    # only the learned R is replaced by our GPEM regression. Three protocol
    # variants for parallel comparison with cobevt_/dmstrack_precomputed.
    {
        "name": "dmstrack_gpem_precomputed",
        "label": "CoBEVT+DMSTrack-GPEM-R (2D-IoU per-seq)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _DMSTRACK_GPEM_TRACKING_DIR,
            "gt_dir": _DMSTRACK_GT_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": False,
        },
    },
    {
        "name": "dmstrack_gpem_precomputed_3d",
        "label": "CoBEVT+DMSTrack-GPEM-R (3D-IoU per-seq)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _DMSTRACK_GPEM_TRACKING_DIR,
            "gt_dir": _DMSTRACK_GT_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": True,
            "convex_hull": True,
        },
    },
    {
        # Should reproduce the GPEM-R AMOTA = 41.26 within ~0.1.
        "name": "dmstrack_gpem_precomputed_dmstrack_exact",
        "label": "CoBEVT+DMSTrack-GPEM-R (DMSTrack-exact: 3D + global + rematching)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _DMSTRACK_GPEM_TRACKING_DIR,
            "gt_dir": _DMSTRACK_GT_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": True,
            "convex_hull": True,
            "ab3dmot_rematching": True,
            "global_eval": True,
        },
    },
    # Our auto-tuned cobevt tracker through the same precomputed_tracks
    # pipeline. Apples-to-apples comparison vs DMSTrack and CoBEVT+AB3DMOT.
    {
        "name": "our_cobevt_autotune_precomputed",
        "label": "CoBEVT+Ours-AutoTune (2D-IoU per-seq)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _OUR_COBEVT_TRACKING_DIR,
            "gt_dir": _DMSTRACK_GT_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": False,
        },
    },
    {
        "name": "our_cobevt_autotune_precomputed_3d",
        "label": "CoBEVT+Ours-AutoTune (3D-IoU per-seq)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _OUR_COBEVT_TRACKING_DIR,
            "gt_dir": _DMSTRACK_GT_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": True,
            "convex_hull": True,
        },
    },
    {
        "name": "our_cobevt_autotune_precomputed_dmstrack_exact",
        "label": "CoBEVT+Ours-AutoTune (DMSTrack-exact: 3D + global + rematching)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _OUR_COBEVT_TRACKING_DIR,
            "gt_dir": _DMSTRACK_GT_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": True,
            "convex_hull": True,
            "ab3dmot_rematching": True,
            "global_eval": True,
        },
    },

    # ============================================================
    # Ours S3 — GPEM filters running directly on DMSTrack's per-CAV
    # PointPillar detections (no self-inject), dumped to KITTI MOT under
    # third_party/AB3DMOT/results/v2v4real/ and re-scored through the same
    # precomputed_tracks pipeline as the anchors above (DMSTrack-exact
    # protocol: 3D-IoU + global tape + AB3DMOT rematching). The V2V AMOTA
    # reproduces results/paper_metrics_V2V_noinject.md (EKF 46.78,
    # CI 46.70, SABRE 44.33) and the autotuned
    # DMS_S3NoNorm_sabre_quadratic 41.97. Run with:
    #   python scripts/run_v2v4real_benchmark.py \
    #       --configs s3_dms_percav_ekf_noinject,s3_dms_percav_ci_noinject,\
    #                 s3_dms_percav_sabre_noinject,s3_dms_percav_sabre_quad_autotuned \
    #       --output-dir results/V2V4Real_S3_RESCORE
    # ============================================================
    {
        "name": "s3_dms_percav_ekf_noinject",
        "label": "Ours S3 — GPEM filters on DMSTrack per-CAV PP dets, EKF (no self-inject)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _S3_TRACKS_BASE / "pp_dmstrack_per_cav_Car_val_DMSPerCavSweepEkf_H1" / "data_0",
            "gt_dir": _PAPER_GT_LABELS_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": True,
            "convex_hull": True,
            "ab3dmot_rematching": True,
            "global_eval": True,
        },
    },
    {
        "name": "s3_dms_percav_ci_noinject",
        "label": "Ours S3 — GPEM filters on DMSTrack per-CAV PP dets, CI (no self-inject)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _S3_TRACKS_BASE / "pp_dmstrack_per_cav_Car_val_DMSPerCavSweepCi_H1" / "data_0",
            "gt_dir": _PAPER_GT_LABELS_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": True,
            "convex_hull": True,
            "ab3dmot_rematching": True,
            "global_eval": True,
        },
    },
    {
        "name": "mh1ma5_cobevt_ekf",
        "label": "CoBEVT + EKF, immediate-emit (mh=1/ma=5)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _S3_TRACKS_BASE / "cobevt_Car_val_MH1MA5_CoBEVT_ekf_H1" / "data_0",
            "gt_dir": _PAPER_GT_LABELS_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": True,
            "convex_hull": True,
            "ab3dmot_rematching": True,
            "global_eval": True,
        },
    },
    {
        "name": "mh1ma5_cobevt_sabre",
        "label": "CoBEVT + SABRE + GPEM-quadratic, immediate-emit (mh=1/ma=5)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _S3_TRACKS_BASE / "cobevt_Car_val_MH1MA5_CoBEVT_sabre_quadratic_H1" / "data_0",
            "gt_dir": _PAPER_GT_LABELS_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": True,
            "convex_hull": True,
            "ab3dmot_rematching": True,
            "global_eval": True,
        },
    },
    {
        "name": "mh1ma5_dms_sabre",
        "label": "Ours S3 — DMS per-CAV PP dets, SABRE + GPEM-quadratic, immediate-emit (mh=1/ma=5)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _S3_TRACKS_BASE / "pp_dmstrack_per_cav_Car_val_MH1MA5_DMS_sabre_quadratic_H1" / "data_0",
            "gt_dir": _PAPER_GT_LABELS_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": True,
            "convex_hull": True,
            "ab3dmot_rematching": True,
            "global_eval": True,
        },
    },
    {
        "name": "s3_dms_percav_sabre_noinject",
        "label": "Ours S3 — GPEM filters on DMSTrack per-CAV PP dets, SABRE (no self-inject)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _S3_TRACKS_BASE / "pp_dmstrack_per_cav_Car_val_DMSPerCavSweepSabre_H1" / "data_0",
            "gt_dir": _PAPER_GT_LABELS_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": True,
            "convex_hull": True,
            "ab3dmot_rematching": True,
            "global_eval": True,
        },
    },
    {
        "name": "s3_dms_percav_sabre_quad_autotuned",
        "label": "Ours S3 — GPEM filters on DMSTrack per-CAV PP dets, SABRE + GPEM-quadratic (autotuned)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _S3_TRACKS_BASE / "pp_dmstrack_per_cav_Car_val_DMS_S3NoNorm_sabre_quadratic_H1" / "data_0",
            "gt_dir": _PAPER_GT_LABELS_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": True,
            "convex_hull": True,
            "ab3dmot_rematching": True,
            "global_eval": True,
        },
    },

    # ============================================================
    # GPEM ablation sweeps — homogeneous cooperative pairs on the
    # freshly-fixed (rehydrated) buckets. Each scores ALL 30 streams
    # (5 cov modes x 6 filters) in two lifecycle variants:
    #   *_logodds : data-driven log_odds lifecycle + birth gate (autotuned
    #               from results/birth_gate_calibration/*.csv, keyed by the
    #               per-vehicle detector names), score_threshold=0.0.
    #   *_ab3dmot : count-based AB3DMOT lifecycle (mh=3, ma=2), score 0.2.
    # export_dir points at the persistent REHYDRATED merged view (NOT the raw
    # per-vehicle bucket — see _MERGED_VIEWS note above). The log_odds variant's
    # data_driven_* params resolve their curves from the default autotuned CSVs.
    # ============================================================
    {
        "name": "pp_score0_gpem_logodds",
        "label": "PP score-0 pair (GPEM ablation, data-driven log_odds)",
        "kind": "our",
        "export_dir": _EXPORT_PP_S0,
        "params": dict(
            detector_name="pointpillar_v2v4real_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            score_threshold=0.0,
            data_driven_lifecycle=True,
            data_driven_gate_alpha=99.0,   # AMOTA-Bayes quadratic gate (α=2.0 has no curve row → inactive)
            data_driven_gate_fit="quadratic",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "pp_score0_gpem_ab3dmot",
        "label": "PP score-0 pair (GPEM ablation, AB3DMOT lifecycle)",
        "kind": "our",
        "export_dir": _EXPORT_PP_S0,
        "params": dict(
            detector_name="pointpillar_v2v4real_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.2,
            ab3dmot_min_hits=3,
            ab3dmot_max_age=2,
        ),
    },
    {
        "name": "pp_thresh02_gpem_ab3dmot",
        "label": "PP @0.2 (DMSTrack threshold), AB3DMOT mh3/ma2",
        "kind": "our",
        "export_dir": _MERGED_VIEWS / "pp_score0",
        "params": dict(
            detector_name="pointpillar_v2v4real_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.2,
            ab3dmot_min_hits=3,
            ab3dmot_max_age=2,
            streams_keep=["sabre_gpem_polar"],
        ),
    },
    {
        "name": "pp_thresh02_gpem_ab3dmot_mh1ma5",
        "label": "PP @0.2 (DMSTrack threshold), AB3DMOT mh1/ma5",
        "kind": "our",
        "export_dir": _MERGED_VIEWS / "pp_score0",
        "params": dict(
            detector_name="pointpillar_v2v4real_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.2,
            ab3dmot_min_hits=1,
            ab3dmot_max_age=5,
            streams_keep=["sabre_gpem_polar"],
        ),
    },
    {
        "name": "cp_zeroshot_100m_gpem_logodds",
        "label": "CPzs 100m pair (GPEM ablation, data-driven log_odds)",
        "kind": "our",
        "export_dir": _EXPORT_CPZS,
        "params": dict(
            detector_name="centerpoint_100m_on_v2v4real",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            score_threshold=0.0,
            data_driven_lifecycle=True,
            data_driven_gate_alpha=99.0,   # AMOTA-Bayes quadratic gate (α=2.0 has no curve row → inactive)
            data_driven_gate_fit="quadratic",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "cp_zeroshot_100m_gpem_ab3dmot",
        "label": "CPzs 100m pair (GPEM ablation, AB3DMOT lifecycle)",
        "kind": "our",
        "export_dir": _EXPORT_CPZS,
        "params": dict(
            detector_name="centerpoint_100m_on_v2v4real",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.3,
            ab3dmot_min_hits=3,
            ab3dmot_max_age=2,
        ),
    },
    # CPft 100m — PER-VEHICLE calibration: detector_name uses the {vehicle}
    # placeholder so the per-CAV lookup resolves
    # centerpoint_100m_v2v4real_finetune_{astuff,tesla} (two separate fits, like
    # cp_finetune_54m; unlike the single fleet-wide cp_zeroshot fit above).
    {
        "name": "cpft_100m_gpem_logodds",
        "label": "CPft 100m pair (GPEM ablation, data-driven log_odds)",
        "kind": "our",
        "export_dir": _EXPORT_CPFT,
        "params": dict(
            detector_name="centerpoint_100m_v2v4real_finetune_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            score_threshold=0.0,
            data_driven_lifecycle=True,
            data_driven_gate_alpha=99.0,   # AMOTA-Bayes quadratic gate (α=2.0 has no curve row → inactive)
            data_driven_gate_fit="quadratic",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "cpft_100m_gpem_ab3dmot",
        "label": "CPft 100m pair (GPEM ablation, AB3DMOT lifecycle)",
        "kind": "our",
        "export_dir": _EXPORT_CPFT,
        "params": dict(
            detector_name="centerpoint_100m_v2v4real_finetune_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.3,
            ab3dmot_min_hits=3,
            ab3dmot_max_age=2,
        ),
    },

    # ============================================================
    # MH1MA5 immediate-emit lifecycle check (2026-06-12) — clones of
    # the three *_gpem_ab3dmot configs above with min_hits=1 / max_age=5
    # (the permissive lifecycle FORENSIC_mh1ma5_cmrekf proved is what
    # inflates the legacy DMSPerCavSweep* V2V numbers). Restricted to the
    # single sabre_gpem_polar stream for speed — that's the stream whose
    # standard-lifecycle numbers we quote (pp 35.30/31.55, cpzs
    # 38.83/33.92, cpft 39.60/37.95 V2V/Ours-merged). NB: run_suite()
    # has no dump_stream kwarg — the summaries' v2v4real_amota_mean /
    # paper_amota_mean carry both protocols, so no KITTI dump is needed.
    # ============================================================
    {
        "name": "pp_score0_gpem_ab3dmot_mh1ma5",
        "label": "PP score-0 pair (AB3DMOT lifecycle, mh=1 ma=5)",
        "kind": "our",
        "export_dir": _EXPORT_PP_S0,
        "params": dict(
            detector_name="pointpillar_v2v4real_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.3,
            ab3dmot_min_hits=1,
            ab3dmot_max_age=5,
            streams_keep=["sabre_gpem_polar"],
        ),
    },
    {
        "name": "cp_zeroshot_100m_gpem_ab3dmot_mh1ma5",
        "label": "CPzs 100m pair (AB3DMOT lifecycle, mh=1 ma=5)",
        "kind": "our",
        "export_dir": _EXPORT_CPZS,
        "params": dict(
            detector_name="centerpoint_100m_on_v2v4real",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.3,
            ab3dmot_min_hits=1,
            ab3dmot_max_age=5,
            streams_keep=["sabre_gpem_polar"],
        ),
    },
    {
        "name": "cpft_100m_gpem_ab3dmot_mh1ma5",
        "label": "CPft 100m pair (AB3DMOT lifecycle, mh=1 ma=5)",
        "kind": "our",
        "export_dir": _EXPORT_CPFT,
        "params": dict(
            detector_name="centerpoint_100m_v2v4real_finetune_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.3,
            ab3dmot_min_hits=1,
            ab3dmot_max_age=5,
            streams_keep=["sabre_gpem_polar"],
        ),
    },
    {
        "name": "mix_pp_cpzs_gpem_logodds",
        "label": "Mix astuff=PP / tesla=CPzs (GPEM, data-driven log_odds)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPZS_V,
        "params": dict(
            detector_name="mix_pp_cpzs_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            score_threshold=0.0,
            data_driven_lifecycle=True,
            data_driven_gate_alpha=99.0,   # AMOTA-Bayes quadratic gate
            data_driven_gate_fit="quadratic",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "mix_pp_cpzs_gpem_ab3dmot",
        "label": "Mix astuff=PP / tesla=CPzs (GPEM, AB3DMOT lifecycle)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPZS_V,
        "params": dict(
            detector_name="mix_pp_cpzs_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.3,
            score_threshold_by_vehicle={"astuff": 0.2, "tesla": 0.3},
            ab3dmot_min_hits=3,
            ab3dmot_max_age=2,
        ),
    },
    {
        "name": "mix_cpzs_pp_gpem_logodds",
        "label": "Mix astuff=CPzs / tesla=PP (GPEM, data-driven log_odds)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_CPZS_PP_V,
        "params": dict(
            detector_name="mix_cpzs_pp_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            score_threshold=0.0,
            data_driven_lifecycle=True,
            data_driven_gate_alpha=99.0,   # AMOTA-Bayes quadratic gate
            data_driven_gate_fit="quadratic",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "mix_cpzs_pp_gpem_ab3dmot",
        "label": "Mix astuff=CPzs / tesla=PP (GPEM, AB3DMOT lifecycle)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_CPZS_PP_V,
        "params": dict(
            detector_name="mix_cpzs_pp_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.3,
            score_threshold_by_vehicle={"astuff": 0.3, "tesla": 0.2},
            ab3dmot_min_hits=3,
            ab3dmot_max_age=2,
        ),
    },
    {
        "name": "mix_pp_cpft_gpem_logodds",
        "label": "Mix astuff=PP / tesla=CPft (GPEM, data-driven log_odds)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPFT_V,
        "params": dict(
            detector_name="mix_pp_cpft_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            score_threshold=0.0,
            data_driven_lifecycle=True,
            data_driven_gate_alpha=99.0,   # AMOTA-Bayes quadratic gate
            data_driven_gate_fit="quadratic",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "mix_pp_cpft_gpem_ab3dmot",
        "label": "Mix astuff=PP / tesla=CPft (GPEM, AB3DMOT lifecycle)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_PP_CPFT_V,
        "params": dict(
            detector_name="mix_pp_cpft_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.3,
            score_threshold_by_vehicle={"astuff": 0.2, "tesla": 0.3},
            ab3dmot_min_hits=3,
            ab3dmot_max_age=2,
        ),
    },
    {
        "name": "mix_cpft_pp_gpem_logodds",
        "label": "Mix astuff=CPft / tesla=PP (GPEM, data-driven log_odds)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_CPFT_PP_V,
        "params": dict(
            detector_name="mix_cpft_pp_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            score_threshold=0.0,
            data_driven_lifecycle=True,
            data_driven_gate_alpha=99.0,   # AMOTA-Bayes quadratic gate
            data_driven_gate_fit="quadratic",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "mix_cpft_pp_gpem_ab3dmot",
        "label": "Mix astuff=CPft / tesla=PP (GPEM, AB3DMOT lifecycle)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_CPFT_PP_V,
        "params": dict(
            detector_name="mix_cpft_pp_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.3,
            score_threshold_by_vehicle={"astuff": 0.3, "tesla": 0.2},
            ab3dmot_min_hits=3,
            ab3dmot_max_age=2,
        ),
    },
    {
        "name": "mix_cpzs_cpft_gpem_logodds",
        "label": "Mix astuff=CPzs / tesla=CPft (GPEM, data-driven log_odds)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_CPZS_CPFT_V,
        "params": dict(
            detector_name="mix_cpzs_cpft_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            score_threshold=0.0,
            data_driven_lifecycle=True,
            data_driven_gate_alpha=99.0,   # AMOTA-Bayes quadratic gate
            data_driven_gate_fit="quadratic",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "mix_cpzs_cpft_gpem_ab3dmot",
        "label": "Mix astuff=CPzs / tesla=CPft (GPEM, AB3DMOT lifecycle)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_CPZS_CPFT_V,
        "params": dict(
            detector_name="mix_cpzs_cpft_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.3,
            score_threshold_by_vehicle={"astuff": 0.3, "tesla": 0.3},
            ab3dmot_min_hits=3,
            ab3dmot_max_age=2,
        ),
    },
    {
        "name": "mix_cpft_cpzs_gpem_logodds",
        "label": "Mix astuff=CPft / tesla=CPzs (GPEM, data-driven log_odds)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_CPFT_CPZS_V,
        "params": dict(
            detector_name="mix_cpft_cpzs_{vehicle}",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            score_threshold=0.0,
            data_driven_lifecycle=True,
            data_driven_gate_alpha=99.0,   # AMOTA-Bayes quadratic gate
            data_driven_gate_fit="quadratic",
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    },
    {
        "name": "mix_cpft_cpzs_gpem_ab3dmot",
        "label": "Mix astuff=CPft / tesla=CPzs (GPEM, AB3DMOT lifecycle)",
        "kind": "our",
        "export_dir": _EXPORT_MIX_CPFT_CPZS_V,
        "params": dict(
            detector_name="mix_cpft_cpzs_{vehicle}",
            lifecycle_mode="ab3dmot",
            detector_max_range=100.0,
            score_threshold=0.3,
            score_threshold_by_vehicle={"astuff": 0.3, "tesla": 0.3},
            ab3dmot_min_hits=3,
            ab3dmot_max_age=2,
        ),
    },
]

# AB3DMOT + LateFusion + default-R baseline (so we have an apples-to-apples
# cell for comparing GPEM lift on LF specifically).
_LF_DEFAULT_BASE = Path("third_party/AB3DMOT/results/v2v4real/late_fusion_Car_val_default_R_H1/data_0")
for _proto, _proto_label, _proto_params in [
    ('2d',     '2D-IoU per-seq',  {'match_3d': False}),
    ('3d',     '3D-IoU per-seq',  {'match_3d': True, 'convex_hull': True}),
    ('global', 'AB3DMOT-exact',   {'match_3d': True, 'convex_hull': True,
                                     'ab3dmot_rematching': True, 'global_eval': True}),
]:
    BENCHMARK_CONFIGS.append({
        "name": f"ab3dmot_late_fusion_default_precomputed_{_proto}",
        "label": f"AB3DMOT + LateFusion + default R ({_proto_label})",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _LF_DEFAULT_BASE,
            "gt_dir": _DMSTRACK_GT_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            **_proto_params,
        },
    })

# AB3DMOT-GPEM-R ingest configs. AB3DMOT's tracker output natively contains
# valid h and z_center fields so 3D-IoU evaluation works (unlike our
# v2v4real_replay dumps). One config per (detector, fit_mode, protocol).
_AB3DMOT_GPEM_BASE = Path("third_party/AB3DMOT/results/v2v4real")
for _det in ('cobevt', 'late_fusion'):
    for _fit in ('static', 'linear', 'quadratic', 'polar'):
        _track_dir = _AB3DMOT_GPEM_BASE / f'{_det}_Car_val_gpem_{_fit}_H1' / 'data_0'
        for _proto, _proto_label, _proto_params in [
            ('2d',     '2D-IoU per-seq',            {'match_3d': False}),
            ('3d',     '3D-IoU per-seq',            {'match_3d': True, 'convex_hull': True}),
            ('global', 'AB3DMOT-exact (3D+global+rematch)', {'match_3d': True, 'convex_hull': True,
                                                              'ab3dmot_rematching': True, 'global_eval': True}),
        ]:
            BENCHMARK_CONFIGS.append({
                "name": f"ab3dmot_{_det}_gpem_{_fit}_precomputed_{_proto}",
                "label": f"AB3DMOT+{_det} + GPEM-{_fit} R ({_proto_label})",
                "kind": "precomputed_tracks",
                "params": {
                    "tracking_dir": _track_dir,
                    "gt_dir": _DMSTRACK_GT_DIR,
                    "camera_frame": False,
                    "iou_threshold": 0.25,
                    **_proto_params,
                },
            })

# Programmatically register a precomputed_tracks ingest config per mixed-detector
# auto-tune dump. Apples-to-apples comparison vs DMSTrack & CoBEVT+AB3DMOT
# under the same metrics pipeline.
for _det, _track_dir in _OUR_DUMPS.items():
    BENCHMARK_CONFIGS.append({
        "name": f"our_{_det}_autotune_precomputed",
        "label": f"Ours-AutoTune {_det} (2D-IoU per-seq)",
        "kind": "precomputed_tracks",
        "params": {
            "tracking_dir": _track_dir,
            "gt_dir": _DMSTRACK_GT_DIR,
            "camera_frame": False,
            "iou_threshold": 0.25,
            "match_3d": False,
        },
    })

# P50 vs P25 metric-aware ablation. Same 7 trusted configs as our headline
# set, but each loads from results/birth_gate_calibration_p50/{...}.csv
# instead of the default P25 location. Tests whether switching the
# aggregation percentile from 0.25 to 0.50 shifts methodology from
# recall-biased (V2V/AMOTA-optimal) to symmetric (HOTA-optimal) and recovers
# Paper AMOTA against strict eval.
_P50_GATE_CSV = REPO / "results" / "birth_gate_calibration_p50" / "birth_gate_curves.csv"
_P50_LIFECYCLE_CSV = REPO / "results" / "birth_gate_calibration_p50" / "recommended_lifecycle.csv"

_P50_CONFIGS = [
    # (name, base_config_name_to_clone, optional_overrides)
    ("cp_zeroshot_dd_lc_static_p50",    "cp_zeroshot_dd_lc_static",    {}),
    ("cpft_pervehicle_dd_ab_static_p50", "cpft_pervehicle_dd_ab_static", {}),
    ("u_cp_cp_lf_dd_lc_static_p50",     "u_cp_cp_lf_dd_lc_static",     {}),
    ("u_cpft_cpft_lf_dd_ab_static_p50", "u_cpft_cpft_lf_dd_ab_static", {}),
    ("u_cp_cpft_lf_dd_lc_static_p50",   "u_cp_cpft_lf_dd_lc_static",   {}),
    ("u_pp_cp_lf_dd_ab_ab3dmot_static_p50", "u_pp_cp_lf_dd_ab_ab3dmot_static", {}),
    ("cobevt_dets_cmr_se_dd_lc_static_p50", "cobevt_dets_cmr_se_dd_lc_static", {}),
]
for _p50_name, _base_name, _overrides in _P50_CONFIGS:
    _base = next((c for c in BENCHMARK_CONFIGS if c["name"] == _base_name), None)
    if _base is None:
        raise RuntimeError(f"P50 clone: base config '{_base_name}' not in BENCHMARK_CONFIGS yet")
    _new = dict(_base)
    _new["name"] = _p50_name
    _new["label"] = _base["label"] + " [P50 aggregation]"
    _new["params"] = dict(_base["params"])
    _new["params"]["data_driven_gate_csv"] = str(_P50_GATE_CSV)
    _new["params"]["data_driven_lifecycle_csv"] = str(_P50_LIFECYCLE_CSV)
    _new["params"].update(_overrides)
    BENCHMARK_CONFIGS.append(_new)

# Fine gate sweep on cp_zeroshot — generated programmatically to avoid 31 hand-written entries.
# Tests V2V AMOTA(gate) curve on the now-working Tier 1.5 lifecycle.
# Encoded in birth_gate_curves.csv as alpha = 200 + gate*100 → α ∈ {220.0, 221.0, ..., 250.0}.
for _i in range(31):
    _gate = round(0.20 + _i * 0.01, 2)
    _alpha = round(200.0 + _gate * 100.0, 2)
    BENCHMARK_CONFIGS.append({
        "name": f"cp_zeroshot_fine_gate_{_gate:.2f}".replace(".", "p"),
        "label": f"CP-zs fine sweep: gate={_gate:.2f} + auto-lifecycle (T1.5)",
        "kind": "our",
        "export_dir": _EXPORT_CPT,
        "params": dict(
            detector_name="centerpoint_100m_on_v2v4real",
            lifecycle_mode="log_odds",
            detector_max_range=100.0,
            p_tp_birth_gate=0.5,            # overridden by auto-gate
            score_threshold=0.0,
            data_driven_gate_alpha=_alpha,
            data_driven_gate_fit="static",
            data_driven_gate_k=10.0,
            data_driven_lifecycle=True,
            per_cov_lifecycle={"baseline": "ab3dmot"},
        ),
    })

# ============================================================
# Instant-tune configs (P2 consumption) — read a closed-form
# instant_config.json directly. The JSON is a COMPLETE tracker config
# derived from calibration alone (scripts/derive_instant_config.py, P1):
# per-source gate curves + polar tables, per-source log_odds lifecycle
# (p_birth / confirm / kill / log_lr_miss / tape decay), global score
# floor F, sigmoid k, mahal gate G, detector_max_range Rng, protocol
# weight w. _instant_params() loads one JSON and translates it into the
# run_suite kwargs, reusing the existing data_driven_* override plumbing:
#   gate curves      -> data_driven_gate_csv (generated polar curves CSV)
#                       + data_driven_gate_polar_dir (the JSON dir, where
#                         the per-source <det>_polar.csv tables already live)
#   per-source life  -> data_driven_lifecycle_csv (generated recommended_
#                       lifecycle.csv built from solver_lifecycle_per_source)
#   score floor F    -> score_threshold
#   sigmoid k        -> data_driven_gate_k
#   mahal gate G     -> mahal_gate
#   Rng              -> detector_max_range
#   confirm/kill/    -> carried by the lifecycle CSV (per-source blend),
#     log_lr_miss       with the fleet scalars also passed for the
#                       single-source degenerate path
# NOTE: protocol weight w (metric_weight_w) is a DERIVATION-time quantity
# only — it already shaped confirm_log_odds / log_lr_miss when the JSON was
# solved; the tracker runtime has no w knob, so it is intentionally not
# consumed here (documented residual).
_INSTANT_TUNE_DIR = REPO / "results" / "instant_tune"
_INSTANT_EXPORT = {
    "pp_score0":        _EXPORT_PP_S0,
    "cp_zeroshot_100m": _EXPORT_CPZS,
    "cp_finetune_100m": _EXPORT_CPFT,
}


def _instant_params(detector: str, protocol: str) -> Dict:
    """Load results/instant_tune/<detector>/<protocol>/instant_config.json and
    translate it into run_suite kwargs. Generates the two override CSVs the
    data_driven_* plumbing expects (gate curves + lifecycle) into a
    `_generated/` subdir; the per-source polar bin tables are consumed in
    place from the JSON directory. Returns the params dict."""
    import csv as _csv
    cfg_dir = _INSTANT_TUNE_DIR / detector / protocol
    with open(cfg_dir / "instant_config.json") as _f:
        ic = json.load(_f)
    tp = ic["tracker_params"]
    gen = cfg_dir / "_generated"
    gen.mkdir(parents=True, exist_ok=True)

    alpha = float(tp.get("data_driven_gate_alpha", 99.0))
    # Curves CSV: one fit_type=polar row per source detector name. The polar
    # bin table is loaded lazily by load_birth_gate_curve from
    # <polar_dir>/<detector_name>_polar.csv — those files already sit next to
    # the JSON, named exactly as the per-source detector keys.
    gate_csv = gen / "birth_gate_curves.csv"
    with open(gate_csv, "w", newline="") as _f:
        w = _csv.writer(_f)
        w.writerow(["detector", "alpha", "fit_type", "a_quad", "b_quad",
                    "c_quad", "a_lin", "b_lin", "n_range_bins"])
        for src_name, bg in ic["birth_gate"].items():
            fits = bg.get("fits", {})
            w.writerow([src_name, f"{alpha}", "amota_bayes_polar",
                        fits.get("quad_a", 0.0), fits.get("quad_b", 0.0),
                        fits.get("quad_c", 0.0), fits.get("lin_a", 0.0),
                        fits.get("lin_b", 0.0), int(fits.get("n_range_bins", 0))])

    # Lifecycle CSV in recommended_lifecycle.csv schema (one row per source
    # detector name), built from solver_lifecycle_per_source so the per-source
    # BLEND path in v2v4real_replay picks each CAV's own derived lifecycle.
    life_csv = gen / "recommended_lifecycle.csv"
    fields = ["detector", "p_birth", "log_lr_miss", "kill_log_odds",
              "confirm_log_odds", "tape_score_miss_decay",
              "p_miss_tp_observed", "l_tp_observed", "n_tracks_observed"]
    with open(life_csv, "w", newline="") as _f:
        w = _csv.DictWriter(_f, fieldnames=fields)
        w.writeheader()
        for src_name, lc in ic["solver_lifecycle_per_source"].items():
            w.writerow({
                "detector": src_name,
                "p_birth": lc["p_birth"],
                "log_lr_miss": lc["log_lr_miss"],
                "kill_log_odds": lc["kill_log_odds"],
                "confirm_log_odds": lc["confirm_log_odds"],
                "tape_score_miss_decay": lc["tape_score_miss_decay"],
                "p_miss_tp_observed": lc.get("p_miss_tp_observed", ""),
                "l_tp_observed": lc.get("l_tp_observed", ""),
                "n_tracks_observed": lc.get("n_tracks_observed", 0),
            })

    return dict(
        detector_name=tp["detector_name"],
        lifecycle_mode=tp.get("lifecycle_mode", "log_odds"),
        detector_max_range=float(tp["detector_max_range"]),       # Rng
        score_threshold=float(tp["score_threshold"]),             # F
        p_tp_birth_gate=float(tp["p_tp_birth_gate"]),
        confirm_log_odds=float(tp["confirm_log_odds"]),           # fleet scalar fallback
        kill_log_odds=float(tp["kill_log_odds"]),
        log_lr_miss=float(tp["log_lr_miss"]),
        tape_score_miss_decay=float(tp["tape_score_miss_decay"]),
        mahal_gate=float(tp["mahal_gate"]),                       # G
        data_driven_gate_alpha=alpha,
        data_driven_gate_fit="polar",
        data_driven_gate_k=float(tp["data_driven_gate_k"]),       # k
        data_driven_gate_csv=str(gate_csv),
        data_driven_gate_polar_dir=str(cfg_dir),
        data_driven_lifecycle=True,
        data_driven_lifecycle_csv=str(life_csv),
        per_cov_lifecycle={"baseline": "ab3dmot"},
        streams_keep=["sabre_gpem_polar"],
    )


for _idet in ("pp_score0", "cp_zeroshot_100m", "cp_finetune_100m"):
    for _iproto in ("v2v", "ours"):
        if not (_INSTANT_TUNE_DIR / _idet / _iproto / "instant_config.json").exists():
            continue
        BENCHMARK_CONFIGS.append({
            "name": f"instant_{_idet}_{_iproto}",
            "label": f"Instant-tune {_idet} ({_iproto} protocol) — closed-form config",
            "kind": "our",
            "export_dir": _INSTANT_EXPORT[_idet],
            "params": _instant_params(_idet, _iproto),
        })

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
        "static":   "static_cov_av_100.0pct",
        "lin":      "gpem_linear_av_100.0pct",
        "quad":     "gpem_quadratic_av_100.0pct",
        "polar":    "gpem_polar_av_100.0pct",
    },
    "CI": {
        "baseline": "ci_baseline_av_100.0pct",
        "static":   "ci_static_av_100.0pct",
        "lin":      "ci_gpem_linear_av_100.0pct",
        "quad":     "ci_gpem_quadratic_av_100.0pct",
        "polar":    "ci_gpem_polar_av_100.0pct",
    },
    "AKF": {
        "baseline": "akf_baseline_av_100.0pct",
        "static":   "akf_static_av_100.0pct",
        "lin":      "akf_gpem_linear_av_100.0pct",
        "quad":     "akf_gpem_quadratic_av_100.0pct",
        "polar":    "akf_gpem_polar_av_100.0pct",
    },
    "PF": {
        "baseline": "pf_baseline_av_100.0pct",
        "static":   "pf_static_av_100.0pct",
        "lin":      "pf_gpem_linear_av_100.0pct",
        "quad":     "pf_gpem_quadratic_av_100.0pct",
        "polar":    "pf_gpem_polar_av_100.0pct",
    },
    "BICI": {
        "baseline": "bici_baseline_av_100.0pct",
        "static":   "bici_static_av_100.0pct",
        "lin":      "bici_gpem_linear_av_100.0pct",
        "quad":     "bici_gpem_quadratic_av_100.0pct",
        "polar":    "bici_gpem_polar_av_100.0pct",
    },
    "SABRE": {
        "baseline": "sabre_baseline_av_100.0pct",
        "static":   "sabre_static_av_100.0pct",
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
    convex_hull  = bool(p.get("convex_hull", False))
    ab3dmot_rematching   = bool(p.get("ab3dmot_rematching", False))
    use_track_avg_score  = bool(p.get("use_track_avg_score", False))
    global_eval  = bool(p.get("global_eval", False))

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

    # ALWAYS compute world-merged-GT (global tape) metrics in addition to the
    # per-sequence rows. The global view is what gives us the honest
    # "paper_amota_mean" (FPs counted, world-frame) — the strict metric that
    # exposes the FP-ignore inflation. The V2V variant of the same global
    # tape gives us "v2v4real_mg_amota_mean" (still FP-ignore, but merged-GT
    # so directly comparable to our replay pipeline's v2v4real_mg_*).
    # When global_eval=True we ALSO override v2v4real_* with global values
    # (DMSTrack-exact protocol, reproduces 37.16 / 43.52 within 0.1).
    std_global, v2v_global, total_gt = mod.run_global(
        [f"{i:04d}" for i in range(9)],
        tracking_dir, gt_dir, camera_frame, iou_thr,
        match_3d=match_3d, convex_hull=convex_hull,
        ab3dmot_rematching=ab3dmot_rematching,
    )

    runs: List[Dict] = []
    for sid in [f"{i:04d}" for i in range(9)]:
        std_m, v2v_m, std_h, _vh, gt_n = mod.run_one(
            sid, tracking_dir, gt_dir, camera_frame, iou_thr,
            match_3d=match_3d,
            convex_hull=convex_hull,
            ab3dmot_rematching=ab3dmot_rematching,
            use_track_avg_score=use_track_avg_score,
        )
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

    # Always emit the global (world-merged-GT) metrics. These are the
    # apples-to-apples comparison columns for our replay pipeline's
    # paper_amota_mean (FPs counted) and v2v4real_mg_amota_mean (FP-ignore
    # but merged-GT). Same combined-tape, two FP-handling modes.
    agg["paper_amota_mean"]        = 100.0 * std_global["amota"]
    agg["paper_amotp_mean"]        = 100.0 * std_global["amotp"]
    agg["paper_samota_mean"]       = 100.0 * std_global["samota"]
    agg["paper_mota_mean"]         = 100.0 * std_global["mota"]
    agg["paper_mt_mean"]           = 100.0 * std_global.get("mt", 0.0)
    agg["paper_ml_mean"]           = 100.0 * std_global.get("ml", 0.0)
    agg["paper_ids_total"]         = int(std_global.get("ids_total", 0))
    agg["paper_gt_total"]          = int(total_gt)
    agg["v2v4real_mg_amota_mean"]  = 100.0 * v2v_global["amota"]
    agg["v2v4real_mg_amotp_mean"]  = 100.0 * v2v_global["amotp"]
    agg["v2v4real_mg_samota_mean"] = 100.0 * v2v_global["samota"]
    agg["v2v4real_mg_mota_mean"]   = 100.0 * v2v_global["mota"]
    agg["v2v4real_mg_mt_mean"]     = 100.0 * v2v_global.get("mt", 0.0)
    agg["v2v4real_mg_ml_mean"]     = 100.0 * v2v_global.get("ml", 0.0)
    agg["v2v4real_mg_gt_total"]    = int(total_gt)

    # World-merged-GT HOTA (strict + V2V variants). Apples-to-apples with
    # paper_amota_mean / v2v4real_mg_amota_mean above. The per-sequence
    # hota_mean above stays as a separate (per-ego strict) field; the global
    # version is what DMSTrack-vs-our comparison should cite.
    agg["paper_hota_mean"]         = 100.0 * std_global.get("global_hota", 0.0)
    agg["paper_deta_mean"]         = 100.0 * std_global.get("global_deta", 0.0)
    agg["paper_assa_mean"]         = 100.0 * std_global.get("global_assa", 0.0)
    agg["paper_loca_mean"]         = 100.0 * std_global.get("global_loca", 0.0)
    agg["v2v4real_mg_hota_mean"]   = 100.0 * v2v_global.get("global_hota", 0.0)
    agg["v2v4real_mg_deta_mean"]   = 100.0 * v2v_global.get("global_deta", 0.0)
    agg["v2v4real_mg_assa_mean"]   = 100.0 * v2v_global.get("global_assa", 0.0)
    agg["v2v4real_mg_loca_mean"]   = 100.0 * v2v_global.get("global_loca", 0.0)

    # If global_eval=True, ALSO override per-sequence v2v4real_* fields with
    # all-sequences-combined values (DMSTrack-exact protocol). With
    # match_3d=True + ab3dmot_rematching=True this reproduces 37.16 / 43.52
    # within 0.1.
    if global_eval:
        agg["v2v4real_amota_mean"]   = 100.0 * v2v_global["amota"]
        agg["v2v4real_amotp_mean"]   = 100.0 * v2v_global["amotp"]
        agg["v2v4real_samota_mean"]  = 100.0 * v2v_global["samota"]
        agg["v2v4real_mota_mean"]    = 100.0 * v2v_global["mota"]
        agg["v2v4real_mt_mean"]      = 100.0 * v2v_global.get("mt", 0.0)
        agg["v2v4real_ml_mean"]      = 100.0 * v2v_global.get("ml", 0.0)
        agg["v2v4real_amota_std"]    = 0.0
        agg["v2v4real_gt_total"]     = int(total_gt)
        agg["global_eval"]           = True
        agg["match_3d"]              = bool(match_3d)
        agg["convex_hull"]           = bool(convex_hull)
        agg["ab3dmot_rematching"]    = bool(ab3dmot_rematching)

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
                # Emit baseline (uncalibrated)
                base_r = results.get(fg_keys["baseline"])
                if base_r:
                    rows.append(_row_dict(
                        f"{cfg['label']} + {fg_name} (base)",
                        cfg["label"], fg_name, "base", base_r))

                # Emit static (calibrated cov, no GPEM regression)
                static_r = results.get(fg_keys.get("static"))
                if static_r:
                    rows.append(_row_dict(
                        f"{cfg['label']} + {fg_name} + static",
                        cfg["label"], fg_name, "static", static_r))

                # Emit best GPEM mode (lin/quad/polar)
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
    ap.add_argument("--max-scenarios", type=int, default=None,
                    help="Cap scenarios at first N (default: all). For fast smoke tests.")
    ap.add_argument("--dump-match-tapes", action="store_true",
                    help="Persist per-scenario match_tapes.json.gz (per-ego tape "
                         "for every stream) for offline AMOTP re-scoring "
                         "(only affects kind='our' configs).")
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
                max_scenarios=args.max_scenarios,
                dump_match_tapes=args.dump_match_tapes,
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
