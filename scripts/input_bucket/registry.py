"""Bucket metadata registry.

Each entry pins what a bucket is and where its upstream artifact came from.
The build script reads this to populate MANIFEST.json's ``upstream`` section
and to render the bucket's README.md.

Adding a new bucket = adding one entry here + running
``scripts/build_v2v4real_input_bucket.py --bucket <name> --rebuild manifest readme --apply``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class GpemSource:
    """One upstream artifact that gets refit into GPEM regression CSVs.

    For detector buckets, ``mmdet3d_config_relpath`` + ``preds_relpath`` are
    paths under the developer's ``mmdet3d_root`` (set in paths.local.yaml).
    For localizer buckets, ``preds_relpath`` points at the upstream
    ``binned_errors.csv`` and ``mmdet3d_config_relpath`` may be empty.
    ``cmr_basename`` is the GPEM CSV stem written into the bucket.

    ``ann_file_relpath`` (optional, relative to ``mmdet3d_root``) names an infos
    pkl that is aligned 1:1 with ``preds_relpath``. evaluate_errors.py pairs
    ``preds[i]`` to ``dataset.get_ann_info(i)`` POSITIONALLY, so if a detector's
    preds order/length differs from the config's default ann_file (e.g.
    OpenCOOD-produced PointPillar), this override must be supplied — the refit
    appends it as ``--cfg-options test_dataloader.dataset.ann_file=<abs path>``.
    Leave empty ("") for mmdet3d-native, order-preserving detectors that align
    with the config's default ann_file (e.g. CenterPoint).
    """
    vehicle: str                       # "astuff", "tesla", or "" for non-per-vehicle
    mmdet3d_config_relpath: str        # configs/.../<...>.py under mmdet3d_root
    preds_relpath: str                 # work_dirs/.../preds_trainval.pkl under mmdet3d_root
    cmr_basename: str                  # e.g. "pointpillar_v2v4real_astuff"
    ann_file_relpath: str = ""         # optional aligned infos pkl (see docstring)


@dataclass(frozen=True)
class BucketSpec:
    label: str                    # canonical path under v2v4real_inputs/, e.g. "baselines/dmstrack_pp"
    kind: str                     # "detector", "localizer", or "gt"
    tier: str                     # "tier1" (lives in cmr) or "tier2" (fetched at setup)
    short_description: str        # one sentence — used in README header
    long_description: str         # full paragraph — used in README body
    source_kind: str              # "dmstrack_release", "v2v4real_release", "pointpillar_v0.4", etc.
    source_repo_url: str          # external URL (or empty for Tier 1 buckets without one)
    source_commit_sha_or_tag: str # pinned version
    source_artifact_description: str
    consumed_by_configs: List[str]  # ExperimentConfig names that use this bucket
    notes: List[str]              # extra paragraphs for the README "Notes" section
    gpem_sources: List[GpemSource] = field(default_factory=list)


# Order in this dict is the order buckets are walked by the build script.
BUCKETS: Dict[str, BucketSpec] = {
    "baselines/dmstrack_pp": BucketSpec(
        label="baselines/dmstrack_pp",
        kind="detector",
        tier="tier2",
        short_description="DMSTrack-paper PointPillar per-CAV detections.",
        long_description=(
            "Per-vehicle (tesla, astuff) PointPillar detection outputs as released "
            "with the DMSTrack publication. These are the canonical comparison "
            "baseline for DMSTrack-paper numbers. Detections are pre-thresholded "
            "and pre-filtered by the upstream pipeline; we do NOT regenerate them."
        ),
        source_kind="dmstrack_release",
        source_repo_url="https://github.com/eddyhkchiu/DMSTrack",
        source_commit_sha_or_tag="TBD-after-pinning",
        source_artifact_description=(
            "data/v2v4real/detection/pp_dmstrack_per_cav_Car_val/ — per-sequence KITTI MOT "
            "format with per-CAV filename suffix (0000_astuff.txt, 0000_tesla.txt)."
        ),
        consumed_by_configs=[
            "DMSTrack_noinj_anchor", "DMSPerCavSweepEkf", "DMSPerCavSweepAkf",
            "DMSPerCavSweepCi", "DMSPerCavSweepBici", "DMSPerCavSweepSabre",
            "DMS_S3NoNorm_sabre_quadratic_headline",
        ],
        notes=[
            "AB3DMOT reads these by setting up a forward symlink at runtime; the "
            "bytes themselves live in the upstream DMSTrack repo (Tier 2).",
        ],
    ),
    "baselines/cobevt": BucketSpec(
        label="baselines/cobevt",
        kind="detector",
        tier="tier2",
        short_description="V2V4Real-paper CoBEVT cooperative detection outputs.",
        long_description=(
            "BEV-transformer cooperative-fusion detections as released with the "
            "V2V4Real paper. These are the comparison baseline for any "
            "CoBEVT-flavored numbers."
        ),
        source_kind="v2v4real_release",
        source_repo_url="https://github.com/ucla-mobility/V2V4Real",
        source_commit_sha_or_tag="TBD-after-pinning",
        source_artifact_description="data/v2v4real/detection/cobevt_Car_val/ — per-sequence KITTI MOT.",
        consumed_by_configs=["CoBEVT_V2V4Real_anchor"],
        notes=[],
    ),
    "baselines/paper_gt": BucketSpec(
        label="baselines/paper_gt",
        kind="gt",
        tier="tier2",
        short_description="Original V2V4Real ground-truth labels (golden).",
        long_description=(
            "The official frozen V2V4Real validation-split GT, per-sequence KITTI "
            "MOT format. Treated as immutable; any drift fails the GT-integrity "
            "test in tests/test_gt_integrity.py."
        ),
        source_kind="v2v4real_release",
        source_repo_url="https://github.com/ucla-mobility/V2V4Real",
        source_commit_sha_or_tag="TBD-after-pinning",
        source_artifact_description=(
            "scripts/KITTI/v2v4real_val_label/ — 9 sequences, 31,419 GT rows, 2D-bbox cols "
            "all-zero (which interacts with the FP-ignore protocol — see docs/DATA_PIPELINE.md)."
        ),
        consumed_by_configs=[],  # implicit — every config uses this GT for the V2V/OGFP stages
        notes=[
            "This is THE reference GT — do not modify. Augmentations land in augmented_gt/.",
        ],
    ),
    "ours/detectors/pp_score0": BucketSpec(
        label="ours/detectors/pp_score0",
        kind="detector",
        tier="tier1",
        short_description="Our regenerated PointPillar with no score threshold (~0.03 floor).",
        long_description=(
            "PointPillar detector re-exported without the historical 0.20 score "
            "threshold. Detections include very low-confidence proposals (score "
            "down to ~0.03) which gives the tracker more material to fuse and is "
            "the appropriate input for GPEM-aware association. The per-CAV split "
            "(astuff vs tesla) corresponds to per-vehicle PointPillar inference."
        ),
        source_kind="pointpillar_v2v4real",
        source_repo_url="https://github.com/eandert/mmdetection3d (pending publication)",
        source_commit_sha_or_tag="TBD",
        source_artifact_description=(
            "work_dirs/reexport_pp_{astuff,tesla}/preds_trainval.pkl converted via "
            "tools/export_v2v4real_to_cmr.py at score_threshold=0.0."
        ),
        consumed_by_configs=[],  # to populate once Phase F adds detector_bucket field
        notes=[
            "Row count is 4-7x larger than the 0.20-score-floor variant. Tracker "
            "lifecycle (p_tp_birth_gate) may need re-tuning to handle the low-score noise.",
            "DATA GAP: the May 2026 cmr_export_pp_{astuff,tesla} dirs only cover "
            "the train split (1 of 9 val scenarios). Re-run "
            "mmdetection3d/tools/export_v2v4real_to_cmr.py over the val/test "
            "scenarios from the existing preds_trainval.pkl before --rebuild "
            "import will succeed.",
        ],
        gpem_sources=[
            # mmdet3d_config_relpath: evaluate_errors.py only reads the config's
            # DATASET definition (point_cloud_range / classes / ann_file), NOT the
            # model — so a centerpoint 100m config is fine for characterizing PP error.
            GpemSource(
                vehicle="astuff",
                mmdet3d_config_relpath="configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_finetuned_v2v4real-54m_trainval_astuff.py",
                preds_relpath="work_dirs/reexport_pp_astuff/preds_train.pkl",
                cmr_basename="pointpillar_v2v4real_astuff",
                # OpenCOOD-produced preds — order differs from the config default
                # ann_file; supply the 1:1-aligned infos to satisfy the positional pairing.
                ann_file_relpath="work_dirs/reexport_pp_astuff/preds_train_infos_aligned.pkl",
            ),
            GpemSource(
                vehicle="tesla",
                mmdet3d_config_relpath="configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_finetuned_v2v4real-54m_trainval_tesla.py",
                preds_relpath="work_dirs/reexport_pp_tesla/preds_train.pkl",
                cmr_basename="pointpillar_v2v4real_tesla",
                ann_file_relpath="work_dirs/reexport_pp_tesla/preds_train_infos_aligned.pkl",
            ),
        ],
    ),
    "ours/detectors/cp_zeroshot_100m": BucketSpec(
        label="ours/detectors/cp_zeroshot_100m",
        kind="detector",
        tier="tier1",
        short_description="CenterPoint zero-shot inference at 100m range.",
        long_description=(
            "CenterPoint pretrained on a larger autonomous-driving corpus and "
            "evaluated zero-shot on V2V4Real at 100m detection range. No "
            "finetuning — measures generalization of the detection prior."
        ),
        source_kind="centerpoint_zeroshot_100m",
        source_repo_url="https://github.com/eandert/mmdetection3d (pending publication)",
        source_commit_sha_or_tag="TBD",
        source_artifact_description=(
            "work_dirs/preds_cpzs_100m_tesla/preds_trainval.pkl (range-fixed, "
            "8000 frames, 93.9% det→GT recall, aligned with the config's default "
            "ann_file)."
        ),
        consumed_by_configs=[],
        notes=[
            "CPZS is a zero-shot model — the SAME weights run on both CAVs — so a "
            "single fleet-wide calibration is fit from the tesla preds and reused "
            "for both vehicles. astuff is intentionally NOT separately fit (it would "
            "produce the same model). This is not a data gap.",
            "DATA GAP: cmr_export_cpzs_100m_tesla covers the train split only (1 of "
            "9 val scenarios). Re-run mmdetection3d/tools/export_v2v4real_to_cmr.py "
            "with val scenarios included before --rebuild import will succeed.",
        ],
        gpem_sources=[
            GpemSource(
                vehicle="tesla",
                mmdet3d_config_relpath="configs/centerpoint/centerpoint_voxel0075_zeroshot_v2v4real-100m_trainval_tesla_nohistory.py",
                preds_relpath="work_dirs/preds_cpzs_100m_tesla/preds_trainval.pkl",
                cmr_basename="centerpoint_100m_on_v2v4real",
                # CenterPoint is mmdet3d-native and order-preserving — it aligns
                # with the config's default ann_file, so no ann_file_relpath needed.
            ),
            # Single fleet-wide fit from tesla (zero-shot model is identical on both
            # CAVs); astuff is intentionally not separately fit.
        ],
    ),
    "ours/detectors/cp_finetune_54m": BucketSpec(
        label="ours/detectors/cp_finetune_54m",
        kind="detector",
        tier="tier1",
        short_description="CenterPoint finetuned on V2V4Real at 54m range.",
        long_description=(
            "CenterPoint with finetuning on V2V4Real's train split, evaluated at "
            "54m detection range. This is the highest-precision detection lineage "
            "we have but caps at 54m, so far-range objects are missed."
        ),
        source_kind="centerpoint_finetune_54m",
        source_repo_url="https://github.com/eandert/mmdetection3d (pending publication)",
        source_commit_sha_or_tag="TBD",
        source_artifact_description=(
            "work_dirs/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_finetuned_"
            "v2v4real-54m_trainval_{astuff,tesla}/preds_trainval.pkl."
        ),
        consumed_by_configs=[],
        notes=[
            "Detector caps at 54m. If eval gt_range=100m, this bucket scores worse "
            "than a 100m-trained detector on far-range objects by design. See plan "
            "open question #3 (cpft_54m range).",
            "DATA GAP: cmr_export_cpft_54m_{astuff,tesla} cover the train split only "
            "(1 of 9 val scenarios). Re-run mmdetection3d/tools/export_v2v4real_to_cmr.py "
            "with val scenarios included before --rebuild import will succeed.",
        ],
        gpem_sources=[
            GpemSource(
                vehicle="astuff",
                mmdet3d_config_relpath="configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_finetuned_v2v4real-54m_trainval_astuff.py",
                preds_relpath="work_dirs/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_finetuned_v2v4real-54m_trainval_astuff/preds_trainval.pkl",
                cmr_basename="centerpoint_54m_v2v4real_finetune_astuff",
            ),
            GpemSource(
                vehicle="tesla",
                mmdet3d_config_relpath="configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_finetuned_v2v4real-54m_trainval_tesla.py",
                preds_relpath="work_dirs/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_finetuned_v2v4real-54m_trainval_tesla/preds_trainval.pkl",
                cmr_basename="centerpoint_54m_v2v4real_finetune_tesla",
            ),
        ],
    ),
    "ours/detectors/cp_finetune_100m": BucketSpec(
        label="ours/detectors/cp_finetune_100m",
        kind="detector",
        tier="tier1",
        short_description="CenterPoint finetuned on V2V4Real at 100m range.",
        long_description=(
            "CenterPoint with finetuning on V2V4Real's train split, evaluated at "
            "100m detection range. Mirrors cp_finetune_54m but extends the "
            "detection range to 100m so far-range objects are retained. This is "
            "the highest-precision detection lineage at full range."
        ),
        source_kind="centerpoint_finetune_100m",
        source_repo_url="https://github.com/eandert/mmdetection3d (pending publication)",
        source_commit_sha_or_tag="TBD",
        source_artifact_description=(
            "work_dirs/preds_cpft_100m_{astuff,tesla}/preds_trainval.pkl (8000 "
            "frames each, ~75-129m range) for calibration; "
            "work_dirs/cmr_export_blind_test/cpft_100m_{astuff,tesla}/ for the "
            "test export (98.1% astuff / 95.8% tesla det→GT recall, "
            "rehydration-safe)."
        ),
        # Consumers live in run_v2v4real_benchmark.py BENCHMARK_CONFIGS
        # (cpft_100m_gpem_{logodds,ab3dmot}), not the experiments registry this
        # field indexes — same situation as pp_score0 above.
        consumed_by_configs=[],
        notes=[
            "CPft is PER-VEHICLE: astuff and tesla are finetuned separately, so "
            "two distinct calibrations are fit (like cp_finetune_54m; unlike the "
            "single fleet-wide cp_zeroshot fit).",
            "ann_file_relpath points each per-vehicle preds at its matching "
            "trainval infos pkl so evaluate_errors.py's positional preds[i]↔"
            "ann_info[i] pairing is aligned (the train preds are per-vehicle over "
            "the trainval split, not the config's default ann_file order).",
        ],
        gpem_sources=[
            GpemSource(
                vehicle="astuff",
                mmdet3d_config_relpath="configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_finetuned_v2v4real-100m_test.py",
                preds_relpath="work_dirs/preds_cpft_100m_astuff/preds_trainval.pkl",
                cmr_basename="centerpoint_100m_v2v4real_finetune_astuff",
                ann_file_relpath="data/v2v4real/v2v4real_infos_finetune_trainval_astuff.pkl",
            ),
            GpemSource(
                vehicle="tesla",
                mmdet3d_config_relpath="configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_finetuned_v2v4real-100m_test.py",
                preds_relpath="work_dirs/preds_cpft_100m_tesla/preds_trainval.pkl",
                cmr_basename="centerpoint_100m_v2v4real_finetune_tesla",
                ann_file_relpath="data/v2v4real/v2v4real_infos_finetune_trainval_tesla.pkl",
            ),
        ],
    ),
    "augmented_gt": BucketSpec(
        label="augmented_gt",
        kind="gt",
        tier="tier1",
        short_description="Our ego-augmented + dup-merged ground-truth variants.",
        long_description=(
            "Two derived-from-paper-GT variants used by the 'Ours' and "
            "'Ours-merged' evaluation stages:\n\n"
            "  - **with_cav/**: paper GT + each CAV's self-report as a "
            "synthetic GT object (tesla → track_id 90001, astuff → 90002). "
            "Adds 2 rows per frame.\n"
            "  - **with_cav_merged/**: with_cav with overlapping GT rows merged "
            "at 3D-IoU>0.5. ~4.78% of rows merged. Reduces phantom-FP penalties "
            "from spatially-coincident duplicate annotations."
        ),
        source_kind="augmentation",
        source_repo_url="",  # generated in-repo
        source_commit_sha_or_tag="",
        source_artifact_description=(
            "Generated by cmr/scripts/augment_v2v4real_with_cav_self_reports.py "
            "(with_cav) and cmr/scripts/merge_overlapping_gts.py (with_cav_merged)."
        ),
        consumed_by_configs=[],
        notes=[
            "This is Tier 1 — bytes belong in cmr proper. The augmentation is "
            "derived deterministically from baselines/paper_gt/kitti_labels + raw "
            "ego_state.csv data (see DATA_PIPELINE.md).",
        ],
    ),
    "ours/localizers/kiss_icp": BucketSpec(
        label="ours/localizers/kiss_icp",
        kind="localizer",
        tier="tier1",
        short_description="KISS-ICP localization error model (clean run).",
        long_description=(
            "Velocity-binned position/yaw error distribution of the KISS-ICP "
            "localizer, fit from the clean (no synthetic noise injection) "
            "evaluation pass on V2V4Real. Used by the SUMO replay harness to "
            "sample realistic localization errors when scoring trust-aware "
            "fusion variants."
        ),
        source_kind="kiss_icp_v1",
        source_repo_url="",
        source_commit_sha_or_tag="",
        source_artifact_description=(
            "binned_errors.csv + regression_models.txt from "
            "localization_project/evaluation_results/kiss_icp/clean/."
        ),
        consumed_by_configs=[],
        notes=[
            "Refit: see refit_localizer_gpem.py. Upstream evaluation pipeline "
            "lives in the localization_project repo (set the path in "
            "paths.local.yaml under a future 'localization_root' key).",
        ],
    ),
    "ours/localizers/kiss_icp_noisy": BucketSpec(
        label="ours/localizers/kiss_icp_noisy",
        kind="localizer",
        tier="tier1",
        short_description="KISS-ICP localization error model with synthetic noise injection.",
        long_description=(
            "Same KISS-ICP localizer as ours/localizers/kiss_icp, but evaluated "
            "with synthetic noise injected at the IMU + LiDAR feed to characterize "
            "degraded-localization conditions. The error distribution shifts "
            "noticeably wider at speed — this bucket is the ablation comparison "
            "for the clean variant."
        ),
        source_kind="kiss_icp_noisy_v1",
        source_repo_url="",
        source_commit_sha_or_tag="",
        source_artifact_description=(
            "binned_errors.csv + regression_models.txt from "
            "localization_project/evaluation_results/kiss_icp/noisy/."
        ),
        consumed_by_configs=[],
        notes=[
            "Companion to ours/localizers/kiss_icp — same refit pipeline, "
            "different upstream eval pass.",
        ],
    ),
}
