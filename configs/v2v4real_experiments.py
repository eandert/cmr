"""V2V4Real tracker-experiment registry — single source of truth.

Every paper-table cell, every ablation, every reproduction attempt maps to a
named entry here. The registry replaces the bash run scripts' implicit flag
combinations.

Usage:
    from v2v4real_experiments import EXPERIMENTS, get, list_all
    cfg = get("V2V4Real_LF_paper_anchor")
    argv = cfg.to_argv()           # [--det_name late_fusion, ...]
    suffix = cfg.result_suffix()   # canonical _SuffixName for AB3DMOT --result_suffix

Inspect from the shell:
    python -m configs.v2v4real_experiments --list
    python -m configs.v2v4real_experiments --describe DMS_S3_SABRE_quadratic_headline

Adding a new experiment: define a new ExperimentConfig at the bottom and add
to EXPERIMENTS. Don't reach inline into run scripts — they read from here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from typing import Dict, List, Literal, Optional, Tuple

PaperRole = Literal[
    "v2v4real_paper_anchor",  # reproduces V2V4Real-paper baseline
    "dmstrack_paper_anchor",   # reproduces DMSTrack-paper baseline
    "ablation",                # diagnostic / ablation
    "headline",                # our paper's headline configurations
    "side_quest",              # investigation runs (e.g. LF=29.28 hunt)
]

Pipeline = Literal[
    # Vanilla AB3DMOT (count-based lifecycle) — what scripts/v2v4real_runner.py
    # invokes. Expects pre-thresholded detection input (~0.20 score floor).
    "ab3dmot",
    # Our log_odds-lifecycle tracker (src/v2v4real_runner.py + v2v4real_replay
    # + dump_tracks_to_kitti_mot.py + evaluate_precomputed_tracks.py).
    # Score-aware birth gating via p_tp_birth_gate / data-driven gate curves
    # in results/birth_gate_calibration/. Handles score-0 floor input.
    "log_odds",
]


@dataclass(frozen=True)
class LogOddsParams:
    """Params for the ``log_odds`` tracker pipeline.

    Only consumed when ``ExperimentConfig.pipeline == "log_odds"``. Field
    semantics match ``scripts/run_v2v4real_benchmark.py``'s BENCHMARK_CONFIGS
    ``params`` dicts. Fields left ``None`` defer to the runner's defaults.
    """

    # ─── detector + score filtering ────────────────────────────────────────
    detector_name: str = ""
    """Detector identifier; may contain a ``{vehicle}`` placeholder for per-ego
    variants, e.g. ``pointpillar_v2v4real_{vehicle}``."""

    detector_max_range: Optional[float] = None
    """Range cutoff in meters. None → runner default (~70m)."""

    score_threshold: Optional[float] = None
    """Detection score floor applied before fusion. 0.0 for score-0 input."""

    # ─── lifecycle ─────────────────────────────────────────────────────────
    lifecycle_mode: Optional[str] = None  # "log_odds" | "ab3dmot" | "legacy"
    p_tp_birth_gate: Optional[float] = None
    confirm_log_odds: Optional[float] = None
    kill_log_odds: Optional[float] = None
    log_lr_miss: Optional[float] = None
    ab3dmot_min_hits: Optional[int] = None
    ab3dmot_max_age: Optional[int] = None
    per_cov_lifecycle: Optional[Dict[str, str]] = None
    """Per-cov-mode lifecycle override, e.g. ``{"baseline": "ab3dmot"}``."""

    data_driven_lifecycle: bool = False
    """When True, load per-detector lifecycle params (p_birth, log_lr_miss,
    kill) from ``results/birth_gate_calibration/recommended_lifecycle.csv``."""

    data_driven_lifecycle_csv: Optional[str] = None  # override path

    # ─── data-driven birth gate ────────────────────────────────────────────
    data_driven_gate_alpha: Optional[float] = None
    """Selects the birth-gate curve from birth_gate_curves.csv.
    1.0/2.0 = TP-preserving α; 96.0 = polar; 97.0 = static; 98.0 = linear;
    99.0 = quadratic; None = no data-driven gate."""

    data_driven_gate_k: Optional[float] = None  # sigmoid steepness
    data_driven_gate_fit: Optional[str] = None  # "static" | "linear" | "quadratic" | "polar"
    data_driven_gate_csv: Optional[str] = None  # override path to curves CSV

    # ─── matching / NMS ────────────────────────────────────────────────────
    mahal_weight: Optional[float] = None
    iou_weight: Optional[float] = None
    mahal_gate: Optional[float] = None
    per_ego_nms_iou: Optional[float] = None
    combined_nms_iou: Optional[float] = None
    use_static_matching: Optional[bool] = None

    # ─── inactive-track handling ───────────────────────────────────────────
    enable_inactive_preservation: Optional[bool] = None
    inactive_grace_s: Optional[float] = None
    tape_score_miss_decay: Optional[float] = None
    variance_floor: Optional[float] = None

    # ─── self-reports + classes ────────────────────────────────────────────
    self_report_egos: Optional[bool] = None
    vehicle_classes: Optional[Tuple[str, ...]] = None
    class_thresholds: Optional[Dict[str, float]] = None

    # ─── localizer + dump stream ───────────────────────────────────────────
    localizer_name: Optional[str] = None
    dump_stream: Optional[str] = None
    """Stream name to capture for KITTI MOT dump (e.g. ``sabre_static``).
    Required when this config is run through ``dump_tracks_to_kitti_mot.py``."""

    # ─── stream subset (fast iteration) ────────────────────────────────────
    streams_keep: Optional[Tuple[str, ...]] = None
    """Restrict the 30-stream sweep to a subset of STREAM_KEYS. None = all 30.
    The runner auto-includes ``dump_stream`` so the KITTI dump always works.
    Used for dev iteration where one filter+cov combo gives a fast signal."""

    def to_params_dict(self) -> Dict:
        """Convert to the dict shape ``run_v2v4real_benchmark.py`` expects.
        Drops Nones so the runner falls back to its defaults."""
        from dataclasses import asdict
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class ExperimentConfig:
    """One V2V4Real tracker experiment — every flag explicit, no hidden defaults.

    Maps to one ``--result_suffix`` and one ``results/v2v4real/<det>_Car_val_<suffix>_H1``
    directory. Two ExperimentConfigs with the same hash of (detector, *flags) MUST
    produce bit-identical tracker output.
    """

    # ─── identity ────────────────────────────────────────────────────────────
    name: str
    """Short identifier (used as result_suffix and for lookup)."""

    description: str
    """One-liner explaining what this experiment is for / which question it answers."""

    paper_role: PaperRole
    """Which kind of result is this — anchor, ablation, headline?"""

    # ─── core ────────────────────────────────────────────────────────────────
    detector: str
    """det_name for AB3DMOT main.py (late_fusion, late_fusion_with_self, cobevt,
    pp_dmstrack_per_cav, etc). When ``pipeline == "log_odds"`` this is
    informational only; the actual detector name used by our tracker comes
    from ``log_odds_params.detector_name``."""

    pipeline: Pipeline = "ab3dmot"
    """Tracker pipeline this experiment runs through:
      - ``"ab3dmot"``: vanilla AB3DMOT (count-based lifecycle). What
        ``scripts/v2v4real_runner.py`` invokes. Expects ~0.20 score-floor input.
      - ``"log_odds"``: our score-aware tracker (``src/v2v4real_runner.py``
        + ``v2v4real_replay``). Reads ``log_odds_params``; handles score-0 input
        via the data-driven birth gate."""

    log_odds_params: Optional[LogOddsParams] = None
    """Params for the log_odds pipeline. Required when ``pipeline == "log_odds"``,
    ignored otherwise."""

    bucket_label: Optional[str] = None
    """Label of a canonical input bucket under ``cmr/data/v2v4real_inputs/``
    whose ``ab3dmot_detections/`` backs this experiment. Example:
    ``"ours/detectors/pp_score0"``. When set, the runner ensures a forward
    symlink at ``<AB3DMOT>/data/v2v4real/detection/<detector>_Car_val/``
    points into the bucket before launching AB3DMOT.

    Leave as ``None`` for legacy configs whose detection inputs already live
    under ``<AB3DMOT>/data/v2v4real/detection/`` directly."""

    # ─── cooperative / per-source flags ──────────────────────────────────────
    per_source_dets: bool = False
    per_source_cavs: Optional[str] = None  # e.g. "astuff,tesla"
    inject_self_reports: bool = False
    rtk_source_flag: bool = False

    # ─── CMR filter swap ─────────────────────────────────────────────────────
    cmr_filter: Optional[str] = None  # ekf | akf | ci | bici | sabre
    cmr_ci_optimize: bool = False
    cmr_sigma_a: Optional[float] = None

    # ─── GPEM measurement-noise injection ────────────────────────────────────
    use_gpem_r: bool = False
    gpem_fit: Optional[str] = None  # static | linear | quadratic | polar
    gpem_normalize_axiswise: bool = False  # NB: empirically suboptimal; see GPEM-R memory
    gpem_sensor: Optional[str] = None
    """CMR sensor-model name for GPEM (e.g. ``pointpillar_v2v4real_tesla``).
    None → AB3DMOT's default (``cobevt_tracker_tesla``)."""

    gpem_source_range: Optional[str] = None
    """GPEM range basis (audit M1): ``"on"`` = each detection's range for the
    GPEM lookup is measured from the SOURCE CAV's per-frame pose; ``"off"`` =
    legacy tesla-frame range. ``None`` → flag omitted from argv; main.py's
    own default is ``"on"`` (2026-06-12+). NB: every entry registered before
    2026-06-12 actually RAN with the legacy tesla-frame behavior (the flag
    did not exist); set ``"off"`` explicitly to reproduce those bit-exactly."""

    # AB3DMOT-pipeline Mahalanobis association gate (ab3dmot_gpem_injector).
    # Replaces giou_3d matching with a per-pair Mahalanobis distance gate built
    # from S = HPHᵀ + R_det (R from GPEM). Requires use_gpem_r=True (enforced by
    # AB3DMOT main.py). Emitted by to_argv as --mahal_gate / --mahal_gate_thres.
    # Distinct from LogOddsParams.mahal_gate (that one tunes the replay-pipeline
    # Fusion matcher, not this AB3DMOT path).
    ab3dmot_mahal_gate: bool = False
    ab3dmot_mahal_gate_thres: Optional[float] = None  # affinity = -m_dist; default -5.0

    # Hybrid Mahalanobis + 3D-GIoU gate (cost = 0.6·mahal_norm + 0.4·iou_cost).
    # Keeps IoU's spatial dedup (which pure --mahal_gate discards, breaking
    # cross-source dedup) while adding R-awareness. Emitted as
    # --hybrid_mahal_iou_gate / --hybrid_gate_thres. Requires use_gpem_r=True.
    ab3dmot_hybrid_gate: bool = False
    ab3dmot_hybrid_gate_thres: Optional[float] = None  # default -2.0 (soft: no hard reject)

    # Physically-sane process noise for the stock filterpy KF (white-noise-accel,
    # sigma_a≈10 m/s² ≈ 1g — a physical constant, not a data-fit knob). AB3DMOT's
    # default identity Q is ~400× too loose, which makes R irrelevant (the gain is
    # process-noise-bound) and masks any GPEM-R effect. Emitted as --tight_q.
    ab3dmot_tight_q: bool = False

    # Position-only Q clamp: set the (x,y,z) process variance directly, leaving
    # velocity Q at the 0.01 CV default (unlike ab3dmot_tight_q, whose WNA coupling
    # loosens velocity to 1.0 and fragments tracks). Matches Q to the GPEM R scale
    # (Q_pos ≈ 0.1·R_pos) — the decoupled analog of the DKF's --gpem_q_pos.
    # Emitted as --position_q.
    ab3dmot_q_pos: Optional[float] = None

    # CMR-filter (sabre/ci/...) decoupled baseline Q — the CORRECT Q lever for the
    # --cmr_filter path (ab3dmot_q_pos/--position_q only patches the vanilla
    # filterpy kf, NOT the CMR adapter). Filterpy default cmr_q_pos=1.0 (~20-100x
    # the GPEM R scale → baseline gain-saturation before SABRE's NIS adapts).
    # Emitted as --cmr_q_pos/--cmr_q_vel. Leave cmr_q_vel at 0.01 (CV default).
    cmr_q_pos: Optional[float] = None
    cmr_q_vel: Optional[float] = None

    # ─── lifecycle overrides ─────────────────────────────────────────────────
    # None = use AB3DMOT default (which for V2V4Real LF is min_hits=3, max_age=2
    # per AB3DMOT_libs/model.py:get_param).
    min_hits_override: Optional[int] = None
    max_age_override: Optional[int] = None

    # ─── paper-reference metadata (for anchor experiments) ───────────────────
    # When this experiment is a paper anchor, encode the published reference
    # number here so the paper-table generator can render an "us vs published"
    # column. Only set on paper_role in {v2v4real_paper_anchor, dmstrack_paper_anchor}.
    reference_amota: Optional[float] = None
    reference_source: Optional[str] = None  # e.g., "V2V4Real Table 4", "DMSTrack README"

    # ─── derived / convenience ───────────────────────────────────────────────
    def result_suffix(self) -> str:
        """The string passed to ``--result_suffix``. Derived from name so the
        result dir maps back to the registry entry deterministically."""
        return f"_{self.name}"

    def result_dir_name(self) -> str:
        """The leaf dir name under ``results/v2v4real/`` (no path)."""
        return f"{self.detector}_Car_val_{self.name}_H1"

    def to_argv(self) -> List[str]:
        """Resolve to the AB3DMOT main.py argument list (excluding --dataset,
        --split which are fixed). The leading ``--det_name`` is included.

        The argv is the SOLE source of truth for what AB3DMOT actually runs;
        registry entries are immutable, and this method has no hidden defaults.
        """
        argv: List[str] = ["--det_name", self.detector]

        if self.per_source_dets:
            argv.append("--per_source_dets")
        if self.per_source_cavs is not None:
            argv += ["--per_source_cavs", self.per_source_cavs]
        if self.inject_self_reports:
            argv.append("--inject_self_reports")
        if self.rtk_source_flag:
            argv.append("--rtk_source_flag")

        if self.cmr_filter is not None:
            argv += ["--cmr_filter", self.cmr_filter]
        if self.cmr_ci_optimize:
            argv.append("--cmr_ci_optimize")
        if self.cmr_sigma_a is not None:
            argv += ["--cmr_sigma_a", str(self.cmr_sigma_a)]

        if self.use_gpem_r:
            argv.append("--use_gpem_r")
        if self.gpem_fit is not None:
            argv += ["--gpem_fit", self.gpem_fit]
        if self.gpem_normalize_axiswise:
            argv.append("--gpem_normalize_axiswise")
        if self.gpem_sensor is not None:
            argv += ["--gpem_sensor", self.gpem_sensor]
        if self.gpem_source_range is not None:
            argv += ["--gpem_source_range", self.gpem_source_range]

        if self.ab3dmot_mahal_gate:
            argv.append("--mahal_gate")
        if self.ab3dmot_mahal_gate_thres is not None:
            argv += ["--mahal_gate_thres", str(self.ab3dmot_mahal_gate_thres)]
        if self.ab3dmot_hybrid_gate:
            argv.append("--hybrid_mahal_iou_gate")
        if self.ab3dmot_hybrid_gate_thres is not None:
            argv += ["--hybrid_gate_thres", str(self.ab3dmot_hybrid_gate_thres)]
        if self.ab3dmot_tight_q:
            argv.append("--tight_q")
        if self.ab3dmot_q_pos is not None:
            argv += ["--position_q", str(self.ab3dmot_q_pos)]
        if self.cmr_q_pos is not None:
            argv += ["--cmr_q_pos", str(self.cmr_q_pos)]
        if self.cmr_q_vel is not None:
            argv += ["--cmr_q_vel", str(self.cmr_q_vel)]

        if self.min_hits_override is not None:
            argv += ["--min_hits_override", str(self.min_hits_override)]
        if self.max_age_override is not None:
            argv += ["--max_age_override", str(self.max_age_override)]

        argv += ["--result_suffix", self.result_suffix()]
        return argv

    def with_(self, **overrides) -> "ExperimentConfig":
        """Return a new ExperimentConfig with selected fields overridden.
        Useful for composing sweeps from base entries.
        """
        return replace(self, **overrides)

    def describe(self) -> str:
        """Pretty-print this experiment for human review."""
        lines = [
            f"name:         {self.name}",
            f"description:  {self.description}",
            f"paper_role:   {self.paper_role}",
            f"detector:     {self.detector}",
            f"result_dir:   results/v2v4real/{self.result_dir_name()}/",
            "",
            "argv (to AB3DMOT main.py):",
            f"  --dataset v2v4real --split val \\",
        ]
        argv = self.to_argv()
        for i in range(0, len(argv), 2):
            chunk = " ".join(argv[i:i + 2])
            lines.append(f"  {chunk} \\" if i + 2 < len(argv) else f"  {chunk}")
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  Registry
# ═════════════════════════════════════════════════════════════════════════════

EXPERIMENTS: Dict[str, ExperimentConfig] = {}


def _register(cfg: ExperimentConfig) -> ExperimentConfig:
    """Add an entry to the registry (and detect duplicate names)."""
    if cfg.name in EXPERIMENTS:
        raise ValueError(
            f"duplicate experiment name {cfg.name!r}: existing "
            f"({EXPERIMENTS[cfg.name].description}) vs new ({cfg.description})")
    EXPERIMENTS[cfg.name] = cfg
    return cfg


# ─── Phase 1: anchor baselines (S1 EKF, no GPEM, no CMR filter swap) ─────────
# These reproduce (or attempt to reproduce) published V2V4Real-paper /
# DMSTrack-paper baselines. They use AB3DMOT defaults for V2V4Real:
# metric=giou_3d, thres=-0.2, min_hits=3, max_age=2 (per AB3DMOT_libs/model.py).

LF_V2V4Real_anchor = _register(ExperimentConfig(
    name="LF_V2V4Real_anchor",
    description="V2V4Real-paper LF baseline. Vanilla AB3DMOT, no GPEM, no CMR "
                "filter swap, no self-injection. Detector-vintage gap explained "
                "in Phase 1.A.",
    paper_role="v2v4real_paper_anchor",
    detector="late_fusion",
    reference_amota=29.28,
    reference_source="V2V4Real Table 4 (LF + AB3DMOT)",
))

CoBEVT_V2V4Real_anchor = _register(ExperimentConfig(
    name="CoBEVT_V2V4Real_anchor",
    description="V2V4Real-paper CoBEVT baseline. Vanilla AB3DMOT, no GPEM, no CMR "
                "filter swap.",
    paper_role="v2v4real_paper_anchor",
    detector="cobevt",
    reference_amota=32.12,
    reference_source="V2V4Real Table 4 (CoBEVT + AB3DMOT)",
))

DMSTrack_noinj_anchor = _register(ExperimentConfig(
    name="DMSTrack_noinj_anchor",
    description="AB3DMOT-multi-sensor PROXY on DMSTrack's released per-CAV "
                "input — NOT DMSTrack's tracker (audit M4). Uses DMSTrack's "
                "detections but OUR per-source sequential loop, pre-filter, "
                "intra-source NMS and hits-per-CAV accounting (model.py:653-827), "
                "not DMSTrack's differentiable-covariance KF; the astuff stream "
                "is a post-hoc transform (~16.5% native match). Per-source "
                "matching ON (tesla+astuff), no self-injection, no GPEM, filterpy "
                "default Kalman. A true published-baseline row needs an actual "
                "DMSTrack-repo run (<dmstrack_root>/DMSTrack).",
    paper_role="dmstrack_paper_anchor",
    detector="pp_dmstrack_per_cav",
    per_source_dets=True,
    per_source_cavs="astuff,tesla",
    reference_amota=43.0,
    reference_source="DMSTrack README (best reproduced)",
))

# ─── DMSPerCavSweep* legacy sweep cells (audit M5, 2026-06-12) ───────────────
# Five per-filter pp_dmstrack cells from the May-21 sweep. The 2026-06-12
# forensics (see FORENSIC_mh1ma5_* below and the registry comment there)
# recovered the TRUE flags from the surviving AB3DMOT run logs: every run used
# --cmr_filter <filter> (CMR adapter, sigma_a default, R=I) plus the
# permissive --min_hits_override 1 --max_age_override 5 lifecycle. Re-running
# with these flags now reproduces the on-disk output BIT-IDENTICALLY (proven
# for Ekf via FORENSIC_mh1ma5_cmrekf). The mh1/ma5 immediate-emit lifecycle
# inflates the FP-ignore V2V protocol (+4.7 AMOTA) while costing the
# FP-counted Ours protocol — these are protocol-exploit baselines, NOT
# paper-comparable DMSTrack anchors; paper_role demoted accordingly.
#
# The name MUST match the existing dir suffix exactly so paper_metrics finds
# the on-disk dir.

for _filt, _ref in [("Ekf", 46.78), ("Akf", 41.82), ("Ci", 46.70),
                    ("Bici", 46.70), ("Sabre", 44.33)]:
    _register(ExperimentConfig(
        name=f"DMSPerCavSweep{_filt}",
        description=f"Legacy per-CAV {_filt.upper()} sweep cell (no self-inject), "
                    f"V2V AMOTA {_ref:.2f}. Flags below are the RECOVERED truth "
                    f"(2026-06-12 forensics): cmr_filter={_filt.lower()} + "
                    f"min_hits=1/max_age=5 immediate-emit lifecycle — re-running "
                    f"now reproduces the on-disk output bit-identically (proven "
                    f"for Ekf). Protocol-exploit baseline (FP-ignore V2V "
                    f"+4.7), not a DMSTrack-paper anchor.",
        paper_role="side_quest",
        detector="pp_dmstrack_per_cav",
        per_source_dets=True,
        per_source_cavs="astuff,tesla",
        cmr_filter=_filt.lower(),               # recovered true flag set (M5)
        min_hits_override=1,
        max_age_override=5,
        reference_amota=_ref,
        reference_source=f"our older sweep (DMSPerCavSweep{_filt}_H1, no self-inject)",
    ))


# ─── Phase 1.A side-quest entries (LF=29.28 hunt) ────────────────────────────
# Each entry tests one hypothesis from PROJECT_PLAN.md's Phase 1.A suspect list.
# Result dirs land alongside the anchor so they're easy to diff.

LF_sidequest_helpers_off = _register(ExperimentConfig(
    name="LF_sidequest_helpers_off",
    description="Side quest #1: LF anchor with helper flags explicitly off. "
                "Confirms no hidden tuning is inflating LF. Empirically IDENTICAL "
                "to LF_V2V4Real_anchor at S1 EKF (39.54/33.91/26.87 verbatim).",
    paper_role="side_quest",
    detector="late_fusion",
))


# ─── Phase 2/3 ablation: S2 (EKF + GPEM-R) ───────────────────────────────────
# Same EKF, three GPEM modes. NoNorm flag mirrors the GPEM-R memory's
# recommendation (axiswise normalization hurts adaptive filters).

def _s2_ekf_gpem(detector: str, gpem_fit: str, *,
                 inject_self_reports: bool = False,
                 per_source: bool = False,
                 normalize_axiswise: bool = False,
                 with_helpers: bool = False) -> ExperimentConfig:
    """Build an S2 (EKF + GPEM-R) config. ``with_helpers`` toggles the CMR
    tuning bundle (sigma_a=10, ci_optimize, rtk flag, lifecycle overrides)."""
    suffix = "S2_EKF_" + gpem_fit
    if not normalize_axiswise:
        suffix = "S2NoNorm_EKF_" + gpem_fit
    if inject_self_reports:
        suffix = "inj_" + suffix
    name_prefix = {
        "late_fusion": "LF",
        "cobevt": "CoBEVT",
        "pp_dmstrack_per_cav": "DMS",
    }[detector]
    name = f"{name_prefix}_{suffix}"
    if per_source and not inject_self_reports:
        name = f"{name_prefix}noinj_{suffix}"
    return ExperimentConfig(
        name=name,
        description=f"S2 {detector} + GPEM-R ({gpem_fit}){' + axiswise norm' if normalize_axiswise else ''}"
                    f"{', helpers on' if with_helpers else ', helpers off'}.",
        paper_role="ablation",
        detector=detector,
        per_source_dets=per_source,
        per_source_cavs="astuff,tesla" if per_source else None,
        inject_self_reports=inject_self_reports,
        rtk_source_flag=with_helpers,
        cmr_sigma_a=10.0 if with_helpers else None,
        cmr_ci_optimize=with_helpers,
        use_gpem_r=True,
        gpem_fit=gpem_fit,
        gpem_normalize_axiswise=normalize_axiswise,
        min_hits_override=3 if with_helpers else None,
        max_age_override=2 if with_helpers else None,
    )


# ─── Phase 3 S3: CMR filter swap + GPEM-R ────────────────────────────────────

def _s3_cmr_gpem(detector: str, cmr_filter: str, gpem_fit: str, *,
                 inject_self_reports: bool = False,
                 per_source: bool = False,
                 normalize_axiswise: bool = False,
                 with_helpers: bool = True) -> ExperimentConfig:
    """Build an S3 (CMR filter + GPEM-R) config. with_helpers defaults to True
    here because S3 is the regime where CMR helpers actually do something
    (S1/S2 had them as no-ops since no --cmr_filter)."""
    suffix = "S3_" + cmr_filter + "_" + gpem_fit
    if not normalize_axiswise:
        suffix = "S3NoNorm_" + cmr_filter + "_" + gpem_fit
    name_prefix = {
        "late_fusion": "LF",
        "cobevt": "CoBEVT",
        "pp_dmstrack_per_cav": "DMS",
    }[detector]
    name = f"{name_prefix}_{suffix}"
    if per_source and not inject_self_reports:
        name = f"{name_prefix}noinj_{suffix}"
    return ExperimentConfig(
        name=name,
        description=f"S3 {detector} + CMR {cmr_filter} + GPEM-R ({gpem_fit})"
                    f"{' + axiswise norm' if normalize_axiswise else ''}, "
                    f"helpers={'on' if with_helpers else 'off'}.",
        paper_role="ablation",
        detector=detector,
        per_source_dets=per_source,
        per_source_cavs="astuff,tesla" if per_source else None,
        inject_self_reports=inject_self_reports,
        rtk_source_flag=with_helpers,
        cmr_filter=cmr_filter,
        cmr_sigma_a=10.0 if with_helpers else None,
        cmr_ci_optimize=with_helpers,
        use_gpem_r=True,
        gpem_fit=gpem_fit,
        gpem_normalize_axiswise=normalize_axiswise,
        min_hits_override=3 if with_helpers else None,
        max_age_override=2 if with_helpers else None,
    )


# ─── Headline: S3 SABRE quadratic, our cooperative-tracker paper number ──────
DMS_S3_SABRE_quad_headline = _register(ExperimentConfig(
    name="DMS_S3NoNorm_sabre_quadratic_headline",
    description="HEADLINE: our cooperative tracker — SABRE filter + GPEM-R "
                "quadratic, axiswise-norm OFF, self-injection ON, all CMR helpers ON. "
                "Scores 29.06 Ours AMOTA / 64.93 HOTA / 42.22 MOTA.",
    paper_role="headline",
    detector="pp_dmstrack_per_cav",
    per_source_dets=True,
    per_source_cavs="astuff,tesla",
    inject_self_reports=True,
    rtk_source_flag=True,
    cmr_filter="sabre",
    cmr_sigma_a=10.0,
    cmr_ci_optimize=True,
    use_gpem_r=True,
    gpem_fit="quadratic",
    gpem_normalize_axiswise=False,
    min_hits_override=3,
    max_age_override=2,
))


# ─── Programmatic registration of the S2/S3 sweep ────────────────────────────

GPEM_MODES = ("static", "linear", "quadratic")
CMR_FILTERS = ("ekf", "akf", "ci", "bici", "sabre")
DETECTORS_S2_S3 = (
    ("late_fusion",          False, False),  # vanilla LF, no per-source, no self-inject
    ("cobevt",               False, False),  # CoBEVT — no per-source, no self-inject
    ("pp_dmstrack_per_cav",  True,  True),   # DMSTrack — per-source + self-inject
)
# Also DMSnoinj for paper-comparable S1-S3 (per-source, NO self-inject)
DETECTORS_S2_S3_NOINJ = (
    ("pp_dmstrack_per_cav",  True,  False),
)

for det, per_src, inj in DETECTORS_S2_S3 + DETECTORS_S2_S3_NOINJ:
    for mode in GPEM_MODES:
        cfg = _s2_ekf_gpem(det, mode,
                           inject_self_reports=inj, per_source=per_src,
                           normalize_axiswise=False, with_helpers=inj)
        if cfg.name not in EXPERIMENTS:
            _register(cfg)
    for cmr in CMR_FILTERS:
        for mode in GPEM_MODES:
            cfg = _s3_cmr_gpem(det, cmr, mode,
                               inject_self_reports=inj, per_source=per_src,
                               normalize_axiswise=False, with_helpers=inj)
            if cfg.name not in EXPERIMENTS:
                _register(cfg)


# Q/R-ratio autotune test (memory project_qr_ratio_autotune_lead). The base
# LF_S2NoNorm_EKF_quadratic runs vanilla AB3DMOT Q (Q_pos=1.0) into a calibrated
# GPEM R (~0.05) → Q/R≈20, over-trusting each detection (FP/ID blowup vs the
# const-R LF anchor). This variant clamps position Q to the ratio-DERIVED scale
# (Q_pos ≈ 0.1·R_pos ≈ 0.005), leaving velocity Q at the 0.01 CV default — a
# single derived number, no tuning — to test whether the rule that fixed the
# DMSTrack DKF also "just works" on the LF vanilla EKF.
LF_S2_GPEM_quad_qpos005 = _register(
    _s2_ekf_gpem("late_fusion", "quadratic").with_(
        name="LF_S2_GPEM_quad_qpos005",
        description="LF S2 GPEM-R (quad) + position-only Q clamp q_pos=0.005 "
                    "(Q/R-ratio autotune test; velocity Q left at 0.01).",
        ab3dmot_q_pos=0.005,
    ))

# Severe-mismatch case: the `late_fusion_with_self` detection set (ego
# self-reports included) is where raw GPEM blew up worst (OURS ~27.82, IDS ~89 in
# an earlier run). A/B to test whether the same ratio-derived q_pos rescues it.
LFself_S2_GPEM_quad_raw = _register(
    _s2_ekf_gpem("late_fusion", "quadratic").with_(
        name="LFself_S2_GPEM_quad_raw", detector="late_fusion_with_self",
        description="LF-with-self S2 GPEM-R (quad), vanilla Q (raw blowup case)."))
LFself_S2_GPEM_quad_qpos005 = _register(
    _s2_ekf_gpem("late_fusion", "quadratic").with_(
        name="LFself_S2_GPEM_quad_qpos005", detector="late_fusion_with_self",
        ab3dmot_q_pos=0.005,
        description="LF-with-self S2 GPEM-R (quad) + position-only Q clamp 0.005."))


# CoBEVT S2 (EKF + GPEM-R quadratic) WITH the AB3DMOT Mahalanobis association
# gate turned on. The gate-off baseline is CoBEVT_S2NoNorm_EKF_quadratic (the
# "GPEM R Drop-In" CoBEVT row in Table 1). A/B pair to answer: does R-aware
# association gating help the single fused CoBEVT stream, or is it a wash (in
# which case Table 1 keeps the "== S2 / N/A" note for CoBEVT's S3 cell)?
CoBEVT_S2NoNorm_EKF_quadratic_mahalgate = _register(
    EXPERIMENTS["CoBEVT_S2NoNorm_EKF_quadratic"].with_(
        name="CoBEVT_S2NoNorm_EKF_quadratic_mahalgate",
        description="A/B: CoBEVT S2 (EKF + GPEM-R quadratic) + AB3DMOT Mahalanobis "
                    "association gate ON (replaces giou_3d with the R-aware gate). "
                    "Gate-off baseline = CoBEVT_S2NoNorm_EKF_quadratic.",
        ab3dmot_mahal_gate=True,
    ))


# ─── First config wired to the canonical-tree pp_score0 bucket ───────────────
# Validates that --rebuild import produces tracker-readable detections and that
# the runner's ensure_ab3dmot_dets_symlink() helper hooks AB3DMOT correctly.
# Use as a smoke test: it's a vanilla S1 EKF per-CAV run, comparable to the
# DMSPerCavSweepEkf anchor but on our regenerated PP @ score 0.
PP_Score0_S1_Ekf_bucket = _register(ExperimentConfig(
    name="PP_Score0_S1_Ekf_bucket",
    description="Smoke-test: vanilla S1 EKF per-CAV against the new "
                "ours/detectors/pp_score0 bucket (PP score-0 floor, val-blind "
                "data). NB: DMSTrack-style configs expect a 0.2 pre-filtered "
                "score floor; the score-0 floor needs the GPEM post-filter path. "
                "Use PP_Score0_S3_sabre_quadratic for the real test.",
    paper_role="ablation",
    detector="pp_score0",
    bucket_label="ours/detectors/pp_score0",
    per_source_dets=True,
    per_source_cavs="astuff,tesla",
))


# GPEM-aware config matched to pp_score0's input convention. Mirrors the existing
# DMS_S3NoNorm_sabre_quadratic_headline but:
#   - detector=pp_score0 (bucket-backed) instead of pp_dmstrack_per_cav
#   - inject_self_reports=False (self-reports are ALREADY baked into the .txt
#     files by --rebuild import; runtime injection would double them)
#   - gpem_sensor=pointpillar_v2v4real_tesla so the GPEM R model matches the
#     detector lineage (not CoBEVT's default)
# Our GPEM-autotuned config on pp_score0. Uses the log_odds tracker
# pipeline — the right harness for score-0 floor data (DMSTrack-style
# AB3DMOT configs above expect ~0.20 pre-filtered scores; the log_odds
# pipeline does score-aware post-filtering via the data-driven birth gate).
#
# Calibration: birth_gate curves + recommended_lifecycle params come from
# results/birth_gate_calibration/, which currently reflect the old 55m
# detector range. Before this config is meaningful, the user must:
#   1. Run `--rebuild gpem` for ours/detectors/pp_score0 (requires
#      paths.local.yaml::mmdet3d_root).
#   2. Re-run scripts/derive_optimal_birth_gate.py against the new
#      polar_calibration CSVs to refresh recommended_lifecycle.csv at 80m.
# Until step 2 lands, the gate values below use stale 55m calibration —
# numbers will be off.
PP_Score0_log_odds_autotuned = _register(ExperimentConfig(
    name="PP_Score0_log_odds_autotuned",
    description="GPEM-autotuned log_odds pipeline on pp_score0 (the bucket's "
                "val-blind data). Score-0 floor handled by data-driven birth "
                "gate (alpha=quadratic) + recommended per-detector lifecycle. "
                "Requires refreshed calibration at 80m range; see config "
                "comment for the gating refit steps.",
    paper_role="headline",
    detector="pp_score0",   # informational for the log_odds pipeline
    pipeline="log_odds",
    bucket_label="ours/detectors/pp_score0",
    log_odds_params=LogOddsParams(
        # Uses the freshly-refit GPEM calibration for pointpillar_v2v4real_{astuff,tesla}
        # (no _score02 suffix), derived from the actual score-0 preds at 80m range
        # via `--rebuild gpem`. Birth-gate curves were refreshed via
        # derive_optimal_birth_gate.py against the new sensor models.
        # Other params follow the documented pp_score02_pervehicle pattern
        # (CONFIG_REFERENCE.md §B.1) but at score_threshold=0.0 for the
        # score-0-floor data and detector_max_range=80 for the new range.
        detector_name="pointpillar_v2v4real_{vehicle}",
        # The polar fit equation (a + b·r) · (1 + c·cos θ + d·cos 2θ) extrapolates
        # smoothly beyond the 0-50m characterization range, so we set max to the
        # actual detector envelope (PP emits up to ~83m). Per-axis b coefficients
        # are ~1e-3, so extrapolating from 50m to 100m only inflates std by ~0.05m.
        # If/when polar-mode bin-coverage is needed at 80m+, re-refit with a 100m
        # mmdet3d config.
        detector_max_range=100.0,
        score_threshold=0.0,
        lifecycle_mode="log_odds",
        # Autotune: data-driven birth gate (quadratic curve from
        # birth_gate_curves.csv at α=2.0, TP-preserving) + data-driven
        # lifecycle (recommended_lifecycle.csv). The lifecycle CSV is missing
        # pointpillar_v2v4real_{vehicle} rows as of 2026-06-08 — the loader
        # warns and falls back to the manual p_tp_birth_gate below. Refit
        # via scripts/solve_amota_bayes_gate.py for full autotune.
        data_driven_lifecycle=True,
        data_driven_gate_alpha=2.0,
        data_driven_gate_fit="quadratic",
        p_tp_birth_gate=0.30,
        dump_stream="sabre_static",
    ),
))


# Fast-iteration sibling: same params but only the dump_stream is built.
# Runs ~30× fewer fusion streams per frame for quick signal during dev.
PP_Score0_log_odds_autotuned_fast = _register(ExperimentConfig(
    name="PP_Score0_log_odds_autotuned_fast",
    description="Same as PP_Score0_log_odds_autotuned, but restricted to the "
                "single ci_gpem_quadratic stream — CI filter + GPEM-R "
                "quadratic noise mode, usually the strongest signal for "
                "this input. Fast iteration: skips the other 29 streams.",
    paper_role="dev",
    detector="pp_score0",
    pipeline="log_odds",
    bucket_label="ours/detectors/pp_score0",
    log_odds_params=LogOddsParams(
        detector_name="pointpillar_v2v4real_{vehicle}",
        detector_max_range=100.0,
        score_threshold=0.0,
        lifecycle_mode="log_odds",
        # Autotune (AMOTA-Bayes). The gate + lifecycle come from
        # scripts/solve_amota_bayes_gate.py, which encodes its fit forms at
        # alpha 96=polar / 97=static / 98=linear / 99=quadratic (NOT the 1.0/2.0
        # tp-preserving alphas from derive_optimal_birth_gate.py). alpha=99.0 +
        # fit="quadratic" selects the AMOTA-Bayes quadratic gate for
        # pointpillar_v2v4real_{vehicle}; data_driven_lifecycle reads the
        # matching recommended_lifecycle.csv row (now refit on the aligned
        # calibration).
        data_driven_lifecycle=True,
        data_driven_gate_alpha=99.0,
        data_driven_gate_fit="quadratic",
        p_tp_birth_gate=0.30,
        dump_stream="ci_gpem_quadratic",
        streams_keep=("ci_gpem_quadratic",),
    ),
))


# Diagnostic: same as ..._fast but score_threshold=0.20 to isolate whether the
# log_odds path's score filter actually reduces over-emission. Expectation:
# if the filter works, total tracks should drop ~10x vs the score=0.0 run.
PP_Score0_log_odds_autotuned_fast_score020 = _register(ExperimentConfig(
    name="PP_Score0_log_odds_autotuned_fast_score020",
    description="Diagnostic: PP_Score0_log_odds_autotuned_fast with "
                "score_threshold=0.20 (was 0.0). Tests whether the log_odds "
                "score filter at v2v4real_replay.py:963 actually engages.",
    paper_role="dev",
    detector="pp_score0",
    pipeline="log_odds",
    bucket_label="ours/detectors/pp_score0",
    log_odds_params=LogOddsParams(
        detector_name="pointpillar_v2v4real_{vehicle}",
        detector_max_range=100.0,
        score_threshold=0.20,   # <-- the only difference vs ..._fast
        lifecycle_mode="log_odds",
        data_driven_lifecycle=True,
        data_driven_gate_alpha=2.0,
        data_driven_gate_fit="quadratic",
        p_tp_birth_gate=0.30,
        dump_stream="ci_gpem_quadratic",
        streams_keep=("ci_gpem_quadratic",),
    ),
))


# Verification config: CenterPoint zero-shot 100m through the log_odds S4 path,
# single ci_gpem_quadratic stream — mirrors PP_Score0_log_odds_autotuned_fast_score020
# for an apples-to-apples CI+GPEM-quadratic comparison after the blind-test
# export fix. CPZS has no birth-gate curve (data_driven_gate is inactive for it),
# so it relies on the score floor + the GPEM model's detection_probability for
# p_tp. detector_name has no {vehicle} placeholder — one zero-shot model for both
# CAVs (centerpoint_100m_on_v2v4real). score_threshold=0.20 matches the PP floor.
CPZS_log_odds_fast_score020 = _register(ExperimentConfig(
    name="CPZS_log_odds_fast_score020",
    description="CenterPoint zero-shot 100m, log_odds S4 path, single "
                "ci_gpem_quadratic stream, score>=0.20. Verifies the re-exported "
                "cp_zeroshot_100m bucket works end-to-end (CI + GPEM-quad).",
    paper_role="dev",
    detector="cp_zeroshot_100m",
    pipeline="log_odds",
    bucket_label="ours/detectors/cp_zeroshot_100m",
    log_odds_params=LogOddsParams(
        detector_name="centerpoint_100m_on_v2v4real",
        detector_max_range=100.0,
        score_threshold=0.20,
        lifecycle_mode="log_odds",
        # AMOTA-Bayes autotune: alpha=99.0 = the solver's quadratic gate
        # (see PP_Score0_log_odds_autotuned_fast for the alpha-encoding note).
        # data_driven_lifecycle reads centerpoint_100m_on_v2v4real's refit row.
        data_driven_lifecycle=True,
        data_driven_gate_alpha=99.0,
        data_driven_gate_fit="quadratic",
        p_tp_birth_gate=0.30,
        dump_stream="ci_gpem_quadratic",
        streams_keep=("ci_gpem_quadratic",),
    ),
))


PP_Score0_S3_sabre_quadratic = _register(ExperimentConfig(
    name="PP_Score0_S3_sabre_quadratic",
    description="GPEM-aware S3: SABRE filter + GPEM-R quadratic on the new "
                "pp_score0 bucket. Real test for the score-0 input (uses the "
                "GPEM post-filter path that the DMSTrack S1 EKF lacks).",
    paper_role="ablation",
    detector="pp_score0",
    bucket_label="ours/detectors/pp_score0",
    per_source_dets=True,
    per_source_cavs="astuff,tesla",
    inject_self_reports=False,    # ← already baked into the bucket's .txt files
    rtk_source_flag=True,
    cmr_filter="sabre",
    cmr_sigma_a=10.0,
    cmr_ci_optimize=True,
    use_gpem_r=True,
    gpem_fit="quadratic",
    gpem_normalize_axiswise=False,
    gpem_sensor="pointpillar_v2v4real_tesla",
    min_hits_override=3,
    max_age_override=2,
))


# ─── S3 GPEM-calibration-upgrade experiments (2026-06-11) ────────────────────
# Question: does swapping the GPEM-R calibration under the S3 cooperative
# tracker (pp_dmstrack_per_cav input) move the headline numbers?
#
# Finding while wiring this up: the ab3dmot pipeline loads ONE GPEM model via
# --gpem_sensor (default cobevt_tracker_tesla) — the existing headline and
# legacy sweeps never used per-CAV calibration. New "{cav}"-placeholder
# sensors (e.g. pointpillar_v2v4real_{cav}) load one model per CAV and route
# updates on kf._source_id (see ab3dmot_gpem_injector.inject_gpem_into_ab3dmot).
#
# Calibration candidates (same underlying PP detector as DMSTrack's release):
#   ppcal   = pointpillar_v2v4real_{cav}          (Jun 9-11 aligned fit, 72% TP recall)
#   pp02cal = pointpillar_v2v4real_score02_{cav}  (Apr 26 fit on score>=0.2 PP, 75.9%)
#   dmscal  = dmstrack_{cav}                      (May 1 fit on DMSTrack release dets)

_S3_HEADLINE = EXPERIMENTS["DMS_S3NoNorm_sabre_quadratic_headline"]
# Frozen UNREGISTERED snapshot of the pre-M5 "best-guess" DMSPerCavSweepEkf
# flag set (vanilla filterpy KF, default mh3/ma2 lifecycle). The derived
# entries below (DMS_S1_ekf_vanillaR_repro, FORENSIC_mh1ma5_*,
# DMS_S3_ekf_quad_*) were defined — and their on-disk results produced —
# against THESE flags. M5 corrected the registered DMSPerCavSweepEkf entry to
# the recovered truth (cmr_filter=ekf + mh1/ma5), so deriving from the
# registered entry would silently change these clones' argv.
_LEGACY_EKF = ExperimentConfig(
    name="_DMSPerCavSweepEkf_bestguess_base",  # never registered
    description="(unregistered) pre-M5 best-guess DMSPerCavSweepEkf flag set",
    paper_role="side_quest",
    detector="pp_dmstrack_per_cav",
    per_source_dets=True,
    per_source_cavs="astuff,tesla",
    reference_amota=46.78,
    reference_source="our older sweep (DMSPerCavSweepEkf_H1, no self-inject)",
)

DMS_S3_sabre_quad_defaultcal_check = _register(_S3_HEADLINE.with_(
    name="DMS_S3_sabre_quad_defaultcal_check",
    description="Wiring check: exact clone of DMS_S3NoNorm_sabre_quadratic_headline "
                "with gpem_sensor EXPLICITLY set to the pipeline default "
                "(cobevt_tracker_tesla). Must be bit-identical to the headline "
                "(41.97 V2V) — validates the override path end-to-end.",
    paper_role="ablation",
    gpem_sensor="cobevt_tracker_tesla",
))

DMS_S3_sabre_quad_dmscal = _register(_S3_HEADLINE.with_(
    name="DMS_S3_sabre_quad_dmscal",
    description="Headline clone with per-CAV dmstrack_{cav} GPEM calibration "
                "(May 1 fit on DMSTrack release dets; 19.3%/55.9% TP recall). "
                "Tests whether the input-matched-but-low-recall calibration "
                "differs from the cobevt_tracker_tesla default the headline used.",
    paper_role="ablation",
    gpem_sensor="dmstrack_{cav}",
))

DMS_S3_sabre_quad_ppcal = _register(_S3_HEADLINE.with_(
    name="DMS_S3_sabre_quad_ppcal",
    description="Headline clone with per-CAV pointpillar_v2v4real_{cav} GPEM "
                "calibration (Jun 9-11 aligned fit, 72% TP recall, score-0 raw "
                "stream).",
    paper_role="ablation",
    gpem_sensor="pointpillar_v2v4real_{cav}",
))

DMS_S3_sabre_quad_pp02cal = _register(_S3_HEADLINE.with_(
    name="DMS_S3_sabre_quad_pp02cal",
    description="Headline clone with per-CAV pointpillar_v2v4real_score02_{cav} "
                "GPEM calibration (Apr 26 fit on score>=0.2-floored PP — the "
                "distribution closest to DMSTrack's pre-filtered release dets).",
    paper_role="ablation",
    gpem_sensor="pointpillar_v2v4real_score02_{cav}",
))

DMS_S1_ekf_vanillaR_repro = _register(_LEGACY_EKF.with_(
    name="DMS_S1_ekf_vanillaR_repro",
    description="Provenance check for the 46.78 legacy EKF anchor: re-run of "
                "DMSPerCavSweepEkf's best-guess flags (per-source, no self-inject, "
                "vanilla filterpy KF + constant R, no GPEM) under a NEW suffix. "
                "If this reproduces 46.78 V2V, the legacy anchor used vanilla R. "
                "OUTCOME (2026-06-12 forensic): does NOT reproduce (42.02) — the "
                "legacy anchor's missing flags were the LIFECYCLE, not R. See "
                "FORENSIC_mh1ma5_cmrekf.",
    paper_role="ablation",
))

# ─── 2026-06-12 forensic resolution of the DMSPerCavSweep* provenance ────────
# The May-21 sweep's exact flags were recovered from the surviving AB3DMOT run
# logs (results/v2v4real/log/log_20260521_11h49m35s..12h07m53s): every run used
# `min hits is 1.000000 / max age is 5.000000`, i.e.
# --min_hits_override 1 --max_age_override 5, plus --cmr_filter <filter>
# (CMR EKFAdapter with default sigma_a=10 and R=I — NOT vanilla filterpy).
# FORENSIC_mh1ma5_cmrekf below reproduces DMSPerCavSweepEkf_H1 BIT-IDENTICALLY
# (diff -r of data_0 clean, 65,677 rows). The permissive lifecycle emits every
# birth for >=5 frames (no min_hits gate + 4 coast frames), which inflates the
# FP-ignore V2V protocol (+4.7 AMOTA) while costing the FP-counted Ours
# protocol (-1.7). The DMSPerCavSweep* anchors are therefore protocol-exploit
# baselines, not paper-comparable DMSTrack baselines.

DMS_forensic_mh1ma5_cmrekf = _register(_LEGACY_EKF.with_(
    name="FORENSIC_mh1ma5_cmrekf",
    description="BIT-IDENTICAL reproduction of DMSPerCavSweepEkf_H1 (V2V 46.78): "
                "per-source, no self-inject, --cmr_filter ekf (adapter, sigma_a=10, "
                "R=I), min_hits=1, max_age=5. Proves the legacy sweep's missing "
                "flags were the permissive lifecycle overrides.",
    paper_role="side_quest",
    cmr_filter="ekf",
    min_hits_override=1,
    max_age_override=5,
))

DMS_forensic_mh1ma5_vanilla = _register(_LEGACY_EKF.with_(
    name="FORENSIC_mh1ma5_vanilla",
    description="Lifecycle-isolation control: vanilla filterpy KF (no cmr_filter) "
                "with the legacy sweep's min_hits=1/max_age=5. Near-identical "
                "structure to the legacy output (66,216 vs 65,677 rows) — shows "
                "the lifecycle, not the filter, drives the 46.78 V2V number.",
    paper_role="side_quest",
    min_hits_override=1,
    max_age_override=5,
))

DMS_S3_ekf_quad_ppcal = _register(_LEGACY_EKF.with_(
    name="DMS_S3_ekf_quad_ppcal",
    description="Legacy-EKF-anchor clone (per-source, no self-inject, filterpy "
                "KF) + GPEM-R quadratic with per-CAV pointpillar_v2v4real_{cav} "
                "calibration. Chases the 46.78 EKF anchor with upgraded R.",
    paper_role="ablation",
    use_gpem_r=True,
    gpem_fit="quadratic",
    gpem_normalize_axiswise=False,
    gpem_sensor="pointpillar_v2v4real_{cav}",
))

DMS_S3_ekf_quad_pp02cal = _register(_LEGACY_EKF.with_(
    name="DMS_S3_ekf_quad_pp02cal",
    description="Legacy-EKF-anchor clone + GPEM-R quadratic with per-CAV "
                "pointpillar_v2v4real_score02_{cav} calibration.",
    paper_role="ablation",
    use_gpem_r=True,
    gpem_fit="quadratic",
    gpem_normalize_axiswise=False,
    gpem_sensor="pointpillar_v2v4real_score02_{cav}",
))


# ─── 2026-06-12 MH1MA5 generality matrix ─────────────────────────────────────
# FORENSIC_mh1ma5_cmrekf proved the legacy DMSPerCavSweepEkf 46.78 V2V is just
# the immediate-emit lifecycle (min_hits=1, max_age=5) on the standard CMR EKF
# (std mh=3/ma=2 scores 42.02). Question: is mh1ma5 a GENERAL V2V-protocol
# lever (FP-ignore makes immediate emission nearly free recall) or specific to
# the per-CAV PP input? These three clone existing standard-lifecycle runs
# EXACTLY except min_hits=1 / max_age=5. Counterparts on disk:
#   DMS_S3NoNorm_sabre_quadratic_headline  (41.97 V2V / 32.53 Ours-merged)
#   CoBEVT_V2V4Real_anchor                 (38.77 V2V repl)
#   CoBEVT_S3NoNorm_sabre_quadratic        (helpers-off S3 sweep cell)

MH1MA5_DMS_sabre_quadratic = _register(
    EXPERIMENTS["DMS_S3NoNorm_sabre_quadratic_headline"].with_(
        name="MH1MA5_DMS_sabre_quadratic",
        description="Immediate-emit lifecycle check: exact clone of "
                    "DMS_S3NoNorm_sabre_quadratic_headline (SABRE + GPEM-R "
                    "quadratic, self-inject, helpers on) with min_hits=1 / "
                    "max_age=5 instead of 3/2.",
        paper_role="side_quest",
        min_hits_override=1,
        max_age_override=5,
    ))

MH1MA5_CoBEVT_ekf = _register(
    EXPERIMENTS["CoBEVT_V2V4Real_anchor"].with_(
        name="MH1MA5_CoBEVT_ekf",
        description="Immediate-emit lifecycle check: CoBEVT anchor (vanilla "
                    "AB3DMOT EKF, no GPEM) with min_hits=1 / max_age=5 "
                    "instead of the V2V4Real default 3/2.",
        paper_role="side_quest",
        min_hits_override=1,
        max_age_override=5,
        reference_amota=None,
        reference_source=None,
    ))

MH1MA5_CoBEVT_sabre_quadratic = _register(
    EXPERIMENTS["CoBEVT_S3NoNorm_sabre_quadratic"].with_(
        name="MH1MA5_CoBEVT_sabre_quadratic",
        description="Immediate-emit lifecycle check: exact clone of "
                    "CoBEVT_S3NoNorm_sabre_quadratic (SABRE + GPEM-R "
                    "quadratic, helpers off) with min_hits=1 / max_age=5.",
        paper_role="side_quest",
        min_hits_override=1,
        max_age_override=5,
    ))


# ─── 2026-06-12 S2/S3 INTENT grid (audit M1-M3 + M6 corrected design) ────────
# The corrected experiment grid implementing the GPEM design intent end-to-end
# on the DMS per-CAV input:
#   S2 = SAME input as S1 (per-source 2-vehicle, NO self-injection), same
#        filterpy KF — ONLY R changes: GPEM-R quadratic with input-matched
#        per-CAV calibration + source-local range (M1).
#   S3 = same per-source input, our CMR filter set (ekf/ci/sabre), GPEM-R
#        quadratic per-CAV, source-local range, self-injection per the current
#        headline convention.
# M6 discipline: argv carries ONLY live flags. No --rtk_source_flag (dead in
# per-source mode — AB3DMOT.update is never called there), no redundant
# min_hits=3/max_age=2 (equal to the v2v4real defaults), and cmr_ci_optimize
# only on the CI cell (it is a no-op for ekf/sabre adapters).
#
# R lineage: pointpillar_v2v4real_score02_{cav} — per-CAV PP calibration fit
# on the score>=0.2-floored stream, the distribution closest to DMSTrack's
# pre-filtered release dets (see the ppcal/pp02cal comment above). The plain
# pointpillar_v2v4real_{cav} (score-0 raw) fits are under a cloud until the
# upstream cmr_export NMS fix is re-exported (2026-06-11 finding), so the
# INTENT grid pins the score02 lineage.
#
# Source-range mechanism (M1): main.py computes each CAV's planar offset in
# the common tesla-KITTI frame from the per-frame ego poses and the injector
# measures GPEM range from the SOURCE CAV (--gpem_source_range on). Validated
# against the pp_score0 bucket's _ranges.csv sidecars: the pose math
# reconstructs the sidecar's 3D source-local range to <1e-4 m; the runtime
# planar form differs from the 3D sidecar only by vertical foreshortening
# (mean ~0.1 m). The *_srcoff twins below A/B the legacy tesla-frame range.

DMS_S2_INTENT_ekf_quad = _register(ExperimentConfig(
    name="DMS_S2_INTENT_ekf_quad",
    description="S2 INTENT (audit M1+M3+M6): DMS per-CAV input identical to "
                "S1 (per-source, no self-inject), vanilla filterpy KF — only R "
                "changes: GPEM-R quadratic, per-CAV "
                "pointpillar_v2v4real_score02_{cav} calibration, source-local "
                "range ON, default mh3/ma2 lifecycle, no dead flags.",
    paper_role="ablation",
    detector="pp_dmstrack_per_cav",
    per_source_dets=True,
    per_source_cavs="astuff,tesla",
    use_gpem_r=True,
    gpem_fit="quadratic",
    gpem_sensor="pointpillar_v2v4real_score02_{cav}",
    gpem_source_range="on",
))

DMS_S2_INTENT_ekf_quad_srcoff = _register(DMS_S2_INTENT_ekf_quad.with_(
    name="DMS_S2_INTENT_ekf_quad_srcoff",
    description="A/B twin of DMS_S2_INTENT_ekf_quad with the LEGACY "
                "tesla-frame GPEM range (--gpem_source_range off). Isolates "
                "the M1 source-range effect at fixed R lineage.",
    gpem_source_range="off",
))

DMS_S2_INTENT_ekf_quad_mahalgate = _register(DMS_S2_INTENT_ekf_quad.with_(
    name="DMS_S2_INTENT_ekf_quad_mahalgate",
    description="A/B twin of DMS_S2_INTENT_ekf_quad with the AB3DMOT "
                "Mahalanobis association gate ON (replaces giou_3d with the "
                "R-aware per-pair gate). Isolates the gate at fixed per-CAV R "
                "lineage. RESULT (2026-06-17): catastrophic regression (IDS "
                "65->1093) — pure Mahalanobis discards giou's cross-source "
                "dedup, duplicating co-located cross-CAV tracks. See "
                "_hybridgate twin for the dedup-preserving fix.",
    ab3dmot_mahal_gate=True,
))

DMS_S2_INTENT_ekf_quad_hybridgate = _register(DMS_S2_INTENT_ekf_quad.with_(
    name="DMS_S2_INTENT_ekf_quad_hybridgate",
    description="A/B twin of DMS_S2_INTENT_ekf_quad with the HYBRID "
                "Mahalanobis+GIoU gate (0.6 mahal + 0.4 iou). Keeps IoU's "
                "spatial cross-source dedup while adding R-awareness — the "
                "fair test of whether R-aware matching helps once dedup is "
                "preserved (vs the pure-gate _mahalgate twin that broke it).",
    ab3dmot_hybrid_gate=True,
))

# Tight-Q A/B: AB3DMOT's default identity Q is ~400x too loose, making R
# irrelevant (gain is process-noise-bound) — so GPEM-R drop-in shows no lift on
# the stock filter. These two isolate GPEM-R's effect under a physically-sane Q
# (sigma_a~10 m/s2): same per-source vanilla EKF + tight Q, only R differs.
DMS_S1_tightq_constR = _register(DMS_S1_ekf_vanillaR_repro.with_(
    name="DMS_S1_tightq_constR",
    description="S1 baseline + physically-sane Q (--tight_q, sigma_a~10): "
                "per-source vanilla EKF, CONSTANT R, tight Q. Pairs with "
                "DMS_S2_tightq_gpem to isolate GPEM-R under a sane Q.",
    ab3dmot_tight_q=True,
))

DMS_S2_tightq_gpem = _register(DMS_S2_INTENT_ekf_quad.with_(
    name="DMS_S2_tightq_gpem",
    description="S2 + physically-sane Q (--tight_q, sigma_a~10): per-source "
                "vanilla EKF, per-CAV GPEM-R (score02), source-range on, tight "
                "Q. A/B vs DMS_S1_tightq_constR — the real test of whether GPEM "
                "R helps once Q no longer swamps R.",
    ab3dmot_tight_q=True,
))

# Static-normalized GPEM R (active/static -> mean ~1.0): rescales GPEM R to the
# regime the stock filter responds to (default R=1.0), keeping per-detection +
# per-CAV modulation around the detector's own calibrated baseline. Raw GPEM R
# (~0.05) is ~20x under default R, so it just over-snaps; this lands at ~1.0
# scale with discrimination. Stock (loose) Q, vanilla EKF — A/B vs S1 (default
# R=1.0) and S2 raw GPEM.
DMS_S2_INTENT_ekf_quad_norm = _register(DMS_S2_INTENT_ekf_quad.with_(
    name="DMS_S2_INTENT_ekf_quad_norm",
    description="S2 INTENT with axiswise static-normalized GPEM R "
                "(--gpem_normalize_axiswise): R = GPEM_active/GPEM_static, mean "
                "~1.0, so it sits at the stock filter's responsive scale while "
                "modulating per-detection/per-CAV around the detector baseline.",
    gpem_normalize_axiswise=True,
))

_S3_INTENT_COMMON = dict(
    paper_role="ablation",
    detector="pp_dmstrack_per_cav",
    per_source_dets=True,
    per_source_cavs="astuff,tesla",
    inject_self_reports=True,        # headline convention
    cmr_sigma_a=10.0,                # live for every CMR adapter
    use_gpem_r=True,
    gpem_fit="quadratic",
    gpem_sensor="pointpillar_v2v4real_score02_{cav}",
    gpem_source_range="on",
)

DMS_S3_INTENT_ekf_quad = _register(ExperimentConfig(
    name="DMS_S3_INTENT_ekf_quad",
    description="S3 INTENT: CMR EKF adapter + GPEM-R quadratic per-CAV "
                "(score02 lineage) + source-local range + self-inject. "
                "M6-clean argv (no dead flags).",
    cmr_filter="ekf",
    **_S3_INTENT_COMMON,
))

DMS_S3_INTENT_ci_quad = _register(ExperimentConfig(
    name="DMS_S3_INTENT_ci_quad",
    description="S3 INTENT: CMR CI adapter (omega-optimized fusion ON) + "
                "GPEM-R quadratic per-CAV (score02 lineage) + source-local "
                "range + self-inject. M6-clean argv.",
    cmr_filter="ci",
    cmr_ci_optimize=True,            # live only for CI/BICI
    **_S3_INTENT_COMMON,
))

DMS_S3_INTENT_sabre_quad = _register(ExperimentConfig(
    name="DMS_S3_INTENT_sabre_quad",
    description="S3 INTENT: SABRE adapter + GPEM-R quadratic per-CAV (score02 "
                "lineage) + source-local range + self-inject. The corrected "
                "counterpart of DMS_S3NoNorm_sabre_quadratic_headline (which "
                "ran a global cobevt_tracker_tesla R and tesla-frame range).",
    cmr_filter="sabre",
    **_S3_INTENT_COMMON,
))

DMS_S3_INTENT_sabre_quad_srcoff = _register(DMS_S3_INTENT_sabre_quad.with_(
    name="DMS_S3_INTENT_sabre_quad_srcoff",
    description="A/B twin of DMS_S3_INTENT_sabre_quad with the LEGACY "
                "tesla-frame GPEM range. Isolates the M1 source-range effect "
                "under the corrected per-CAV R lineage.",
    gpem_source_range="off",
))

# ─── 2026-06-17 GPEM-fit-quality A/B (the "does GPEM help DMSTrack input?" test) ─
# Diagnosis (results/match_debug/dms_s2_*_probe_summary.json + this session's
# audit): the score02 per-CAV fits (May-18) had their quadratic std regression
# corrupted by ~9 degenerate far-range bins (≤1 sample → std=0 past ~74 m),
# forcing quad_a negative on the distal axis. The fit therefore UNDER-reads R by
# 30-40% past 60 m and collapses to ~0.05 at 90 m — so the *far* CAV's view of a
# shared object was over-trusted, cancelling GPEM's "trust the closer CAV" lever.
# The June-8 refit (`pointpillar_v2v4real_{cav}`, coarse 10 m bins, no degenerate
# tails) is monotonic and physical (planar var rises to ~0.16 at 90 m vs the
# stale 0.015 — 10×). Same detector; the only characterization difference is the
# score floor (0.20 score02 vs 0.0 generic), which is ~identical at operating
# ranges. These three rows isolate calibration QUALITY and PRESENCE on SABRE:
#   _ppfix : corrected (June) GPEM   — the fix
#   (stale): DMS_S3_INTENT_sabre_quad — the May-18 broken fit (reference)
#   _flatR : no GPEM (CMR adapter R=I) — the uncalibrated control
DMS_S3_sabre_ppfix = _register(DMS_S3_INTENT_sabre_quad.with_(
    name="DMS_S3_sabre_ppfix",
    description="GPEM-fit-quality fix: DMS_S3_INTENT_sabre_quad but with the "
                "CORRECTED June refit GPEM (pointpillar_v2v4real_{cav}, clean "
                "coarse-bin fit, no far-range collapse) instead of the stale "
                "May-18 score02 fit. Same detector, score-floor 0 vs 0.2 "
                "(~identical at operating ranges). Tests whether fixing the "
                "far-range R under-read unlocks a GPEM lift on the full "
                "SABRE architecture.",
    gpem_sensor="pointpillar_v2v4real_{cav}",
))

DMS_S3_sabre_flatR = _register(DMS_S3_INTENT_sabre_quad.with_(
    name="DMS_S3_sabre_flatR",
    description="Uncalibrated control for the GPEM-fit-quality A/B: SABRE "
                "adapter with NO GPEM (use_gpem_r off → CMR adapter constant "
                "R=I), everything else identical to DMS_S3_sabre_ppfix. "
                "Isolates whether calibrated R helps at all on SABRE for "
                "V2V4Real DMSTrack input. NB: R=I is ~10-20x the GPEM planar "
                "scale, so this confounds calibration with scale (SABRE's NIS "
                "Q-adaptation assumes a calibrated R) — read with that caveat.",
    use_gpem_r=False,
    gpem_fit=None,
    gpem_sensor=None,
    gpem_source_range=None,
))

# Scale-vs-range-structure decomposition for the +2.5 AMOTA GPEM lift. The
# _ppfix↔_flatR gap confounds two things: calibration SCALE (GPEM planar R is
# ~10-20x under R=I) and range STRUCTURE (R rises with distance). This row holds
# the June refit but uses the per-detector mean (gpem_fit="static") -> the right
# scale, but FLAT across range. Reading the three SABRE points:
#   _flatR  (R=I, ~10-20x scale, flat)        : uncalibrated control
#   _static (June mean, right scale, flat)    : SCALE only
#   _ppfix  (June quad, right scale, by range): SCALE + range structure
# static≈ppfix  -> the win is scale (calibration magnitude); range-structure is
# secondary (the DAIR pattern: static≈quad>>baseline). static«ppfix -> range
# structure carries the lift. Resolves the R=I scale-confound caveat on Table 1.
DMS_S3_sabre_static = _register(DMS_S3_sabre_ppfix.with_(
    name="DMS_S3_sabre_static",
    description="Scale-only GPEM control for the +2.5 AMOTA decomposition: "
                "identical to DMS_S3_sabre_ppfix (corrected June refit, full "
                "SABRE) but gpem_fit='static' -> per-detector count-weighted "
                "mean R (right calibration scale, FLAT across range). Splits "
                "the GPEM lift into scale (static-vs-flatR) and range structure "
                "(ppfix-vs-static).",
    gpem_fit="static",
))


# ─── 2026-06-18 self-injection isolation (the 34.18-vs-31.31 SABRE gap) ──────
# Two "SABRE on GPEM" numbers coexist: the clean matched triple's
# DMSnoinj_S3NoNorm_sabre_quadratic (34.18 OURS — self-inject OFF, helpers OFF)
# and the Table-2 DMS_S3_sabre_ppfix (31.31 — self-inject ON + cmr_sigma_a=10
# via _S3_INTENT_COMMON). They differ on ~4 axes (inject, sigma_a helper, R
# lineage, source-range), so the ~2.9 gap can't be attributed by inspection.
# This row toggles ONLY self-injection on top of the clean 34.18 config —
# helpers still off, default R/range unchanged — to test the hypothesis (from
# the with-self LF blowup: raw 27.86, association-driven) that self-report
# injection is the dominant lever.
DMSnoinj_S3NoNorm_sabre_quadratic_injonly = _register(
    EXPERIMENTS["DMSnoinj_S3NoNorm_sabre_quadratic"].with_(
        name="DMSnoinj_S3NoNorm_sabre_quadratic_injonly",
        description="Self-injection isolation: the clean noinj SABRE (34.18 "
                    "OURS) with ONLY inject_self_reports=True flipped on — "
                    "helpers still off, same default R/range. Isolates the "
                    "self-injection penalty in the 34.18-vs-31.31 (ppfix) gap.",
        inject_self_reports=True,
    ))


# ─── 2026-06-18 per-CAV GPEM sensor FIX (the clean triple's bug) ─────────────
# BUG: the clean DMSnoinj_S{2,3}NoNorm_* triple (EKF 32.92 / CI 34.06 / SABRE
# 34.18) was run with the injector DEFAULT --gpem_sensor=cobevt_tracker_tesla —
# a SINGLE GLOBAL R (both per-CAV sources get the IDENTICAL R, so GPEM's
# cross-source reliability weighting is neutralized) fit on COOPERATIVE-tracker
# output, NOT the per-CAV PointPillar detection error (see memory
# project_cooperative_vs_per_sensor_characterization). Same bug class fixed for
# the DKF but never for the AB3DMOT triple. These _ppcav variants are the FAIR
# test: identical to the clean triple (noinj, no helpers, source-local range on,
# quad) but with the CORRECT per-CAV June-refit sensor pointpillar_v2v4real_{cav}.
for _ppcav_base in ("DMSnoinj_S2NoNorm_EKF_quadratic",
                    "DMSnoinj_S3NoNorm_ci_quadratic",
                    "DMSnoinj_S3NoNorm_sabre_quadratic"):
    _register(EXPERIMENTS[_ppcav_base].with_(
        name=_ppcav_base + "_ppcav",
        description=EXPERIMENTS[_ppcav_base].description.rstrip(".")
                    + " [per-CAV fix: gpem_sensor=pointpillar_v2v4real_{cav}, "
                      "the correct per-CAV PP error model instead of the "
                      "cobevt_tracker_tesla default].",
        gpem_sensor="pointpillar_v2v4real_{cav}",
    ))


# ─── 2026-06-19 Q-tune sweep on the corrected per-CAV SABRE/CI ───────────────
# "Make damn sure" there's no residual gain-saturation hiding GPEM headroom on
# PP. The CMR adapter's baseline Q defaults to cmr_q_pos=1.0 (~20-100x the GPEM
# R scale); SABRE then adapts via NIS, but the baseline seeds it. Sweep cmr_q_pos
# toward the Q/R-ratio target (≈0.1·R_pos ≈ 0.002-0.007; project_qr_ratio_autotune_lead),
# cmr_q_vel pinned 0.01 (CV default, swept-optimal). On the FAIR per-CAV configs.
for _qp in (0.01, 0.005, 0.002, 0.001):
    _register(EXPERIMENTS["DMSnoinj_S3NoNorm_sabre_quadratic_ppcav"].with_(
        name=f"DMSnoinj_S3NoNorm_sabre_quadratic_ppcav_q{str(_qp).replace('0.','')}",
        description="Corrected per-CAV SABRE + cmr_q_pos Q-tune (cmr_q_vel=0.01).",
        cmr_q_pos=_qp, cmr_q_vel=0.01))
for _qp in (0.005, 0.002):
    _register(EXPERIMENTS["DMSnoinj_S3NoNorm_ci_quadratic_ppcav"].with_(
        name=f"DMSnoinj_S3NoNorm_ci_quadratic_ppcav_q{str(_qp).replace('0.','')}",
        description="Corrected per-CAV CI + cmr_q_pos Q-tune (cmr_q_vel=0.01).",
        cmr_q_pos=_qp, cmr_q_vel=0.01))

# 2026-06-19 (cont.) — LOOSER-than-default Q sweep (user: "tune q higher?"). The
# tighter sweep above peaked at the untuned default (34.11); test whether going
# looser keeps rising or declines toward the gain-saturated EKF (32.77). Both
# levers: higher decoupled cmr_q_pos (qhi*) and a looser sigma_a (sahi*).
for _qp in (1.0, 2.0, 5.0):
    _register(EXPERIMENTS["DMSnoinj_S3NoNorm_sabre_quadratic_ppcav"].with_(
        name=f"DMSnoinj_S3NoNorm_sabre_quadratic_ppcav_qhi{str(_qp).replace('.0','').replace('.','')}",
        description="Corrected per-CAV SABRE + LOOSER cmr_q_pos (cmr_q_vel=0.01).",
        cmr_q_pos=_qp, cmr_q_vel=0.01))
for _sa in (400.0, 1000.0):
    _register(EXPERIMENTS["DMSnoinj_S3NoNorm_sabre_quadratic_ppcav"].with_(
        name=f"DMSnoinj_S3NoNorm_sabre_quadratic_ppcav_sahi{int(_sa)}",
        description="Corrected per-CAV SABRE + looser sigma_a than default (~200).",
        cmr_sigma_a=_sa))


# ═════════════════════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════════════════════

def get(name: str) -> ExperimentConfig:
    """Look up an experiment by name. Raises if not found."""
    if name not in EXPERIMENTS:
        candidates = [n for n in EXPERIMENTS if name.lower() in n.lower()]
        hint = f"  did you mean: {candidates[:5]!r}?" if candidates else ""
        raise KeyError(f"experiment {name!r} not registered.{hint}")
    return EXPERIMENTS[name]


def list_all(*, role: Optional[PaperRole] = None) -> List[ExperimentConfig]:
    """Return all registered experiments, optionally filtered by paper_role."""
    out = list(EXPERIMENTS.values())
    if role is not None:
        out = [c for c in out if c.paper_role == role]
    return out


def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="V2V4Real experiment registry")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="list all registered experiments")
    g.add_argument("--describe", metavar="NAME", help="describe one experiment")
    g.add_argument("--role", metavar="ROLE",
                   choices=["v2v4real_paper_anchor", "dmstrack_paper_anchor",
                            "ablation", "headline", "side_quest"],
                   help="list experiments with this paper_role")
    args = ap.parse_args()

    if args.list:
        by_role: Dict[str, List[ExperimentConfig]] = {}
        for cfg in EXPERIMENTS.values():
            by_role.setdefault(cfg.paper_role, []).append(cfg)
        for role in ("v2v4real_paper_anchor", "dmstrack_paper_anchor",
                     "headline", "ablation", "side_quest"):
            entries = by_role.get(role, [])
            if not entries:
                continue
            print(f"\n[{role}] ({len(entries)})")
            for c in entries:
                print(f"  {c.name:55s}  {c.description[:80]}")
        print(f"\ntotal: {len(EXPERIMENTS)} experiments")
        return 0

    if args.describe:
        print(get(args.describe).describe())
        return 0

    if args.role:
        for c in list_all(role=args.role):
            print(c.name)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
