#!/usr/bin/env python3
"""
Run the CMR GPEM+SABRE tracker over DAIR-V2X-Seq (SPD) exported scenarios.

Phase B of the DAIR-V2X-Seq integration (see results/PROJECT_PLAN.md and the
`dair_integration_1` handoff). Consumes the SAME detections that feed DAIR's
late-fusion baseline — already laid out as CMR scenario dirs by the off-repo
exporter `~/test/dair_v2x/inference/export_spd_to_cmr.py` — runs them through
`v2v4real_replay.run_scenario`, and dumps the per-frame confirmed tracks (WORLD
frame) for each requested fusion stream. The off-repo B3 converter then maps
those world tracks back to the vehicle LiDAR / camera frame and scores them with
DAIR's own `eval_tracking`, so GPEM+SABRE is compared to the LF baseline (MOTA
0.5486 with-oracle) on identical detections through an identical evaluator.

This script does NOT score anything itself — DAIR's evaluator owns the paper
number. The per-stream JSON it writes is the B2→B3 contract.

DAIR-specific configuration (baked in as defaults; all are correctness-critical):
  * detector_name="dair_pp_{vehicle}" — resolves per source to the B0-fitted
    `dair_pp_veh` / `dair_pp_inf` sensor models, which drive BOTH the GPEM
    covariance and the calibration tables.
  * self_report_egos=False — the vehicle ego and the roadside unit are NOT
    annotated GT objects in DAIR; broadcasting them would inject pure false
    positives.
  * WIDE-Z lidar_range / gt_range — DAIR's novatel world frame has z ~= 20 m,
    but `v2v4real_replay._in_ego_range` checks z ABSOLUTELY against the range
    box. The V2V4Real-tuned z-gate ([-5, 3]) would silently drop every object
    detection, leaving only phantom self-report tracks. We widen z and size the
    x/y box to DAIR's 0-100 m forward evaluation range. CMR is 2D; z/h are
    re-attached downstream in B3, so widening z here changes nothing else.

Track source: `result["match_tape_per_stream"][stream]` — one entry per vehicle
frame `{frame_idx, tracks: [(id, x, y, w, l, yaw, score), ...], gts}` in WORLD
frame. (NOT `match_tape_per_ego`: its per-ego frame marker hardcodes tesla=1 /
else=2, so DAIR's `veh` and `inf` egos would collide on marker 2.)

NOTE: GPEM `polar` mode is unavailable until B0 is regenerated to emit the smooth
`polar_std_a..d` columns in `dair_pp_*.csv` (the per-bin `_polar_distributions.csv`
B0 wrote is not what `mode='polar'` reads at runtime). Polar streams are excluded
from the default stream set and will raise `ErrorModelCoverageError` if requested.

Example:
  python scripts/run_dair_evaluation.py \\
    --export-root /media/rave/eddie_drive1/dair_v2x_seq/cmr_export_spd \\
    --out-root    /media/rave/eddie_drive1/dair_v2x_seq/cmr_tracks_out \\
    --streams ab3dmot_baseline,sabre_static,sabre_gpem_linear,sabre_gpem_quadratic \\
    -j 8
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

# Make src/ importable when running from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Streams that don't need GPEM polar (which is blocked until B0 regen). This is
# the LF-analog control (ab3dmot_baseline) plus the GPEM progression on SABRE.
DEFAULT_STREAMS = [
    "ab3dmot_baseline",
    "sabre_static",
    "sabre_gpem_linear",
    "sabre_gpem_quadratic",
]

# DAIR world-frame spatial gate. x is forward along the ego heading (DAIR scores
# 0-100 m forward), y lateral (DAIR ~+-39.68 m), z effectively unbounded — see
# the module docstring on the absolute-z gate. DAIR's `eval_tracking` applies the
# authoritative veh-frame range filter, so this box is a permissive superset.
_WIDE_Z = 1.0e4
DAIR_LIDAR_RANGE = [-100.0, -40.0, -_WIDE_Z, 100.0, 40.0, _WIDE_Z]


def discover_scenarios(export_root: Path, splits: List[str] | None = None) -> List[Path]:
    """Return sorted scenario dirs (``<split>__<scenario>/``) holding a meta.yaml.

    If ``splits`` is given, keep only dirs whose name starts with ``<split>__``.
    """
    out = []
    for d in sorted(export_root.iterdir()):
        if not d.is_dir() or "__" not in d.name or not (d / "meta.yaml").exists():
            continue
        if splits is not None and d.name.split("__", 1)[0] not in splits:
            continue
        out.append(d)
    return out


def tape_to_track_frames(tape: List[Dict]) -> List[Dict]:
    """Convert a `match_tape_per_stream` tape to the serializable B2→B3 form.

    Each input entry is ``{"frame_idx": fid, "tracks": [(id, x, y, w, l, yaw,
    score, height, z), ...], "gts": [...]}``; ``frame_idx`` is the vehicle frame
    id. Output is ``[{"frame_id": int, "tracks": [[id, x, y, w, l, yaw, score,
    height, z], ...]}, ...]`` sorted by frame, with numerics cast to native
    Python types (the tracker emits numpy floats that don't serialize to JSON).
    ``height``/``z`` are the tracker's per-track vertical EMA — ``None`` until a
    track has seen a detection carrying z/height (B3 falls back to re-attach).
    """
    def _f(v):
        return None if v is None else float(v)

    frames = []
    for entry in sorted(tape, key=lambda e: e["frame_idx"]):
        tracks = [
            [int(t[0]), float(t[1]), float(t[2]), float(t[3]),
             float(t[4]), float(t[5]), float(t[6]),
             _f(t[7]) if len(t) > 7 else None,
             _f(t[8]) if len(t) > 8 else None]
            for t in entry["tracks"]
        ]
        frames.append({"frame_id": int(entry["frame_idx"]), "tracks": tracks})
    return frames


def _mean_axis_variance(model_name: str, range_cap: float) -> float:
    """Representative per-axis position variance R for a GPEM model (detector or localizer).

    Count-unweighted mean of sigma^2 over the planar position-error bins within
    ``range_cap`` m, then averaged across the two planar axes. Detector models name the
    axes distal/perpendicular; localizer models longitudinal/lateral — both are handled.
    This is the same ``param2`` (std) the static covariance path consumes.
    """
    from sensor_model_loader import load_distribution_bins_csv, DEFAULT_DATA_DIR
    bins = load_distribution_bins_csv(
        str(Path(DEFAULT_DATA_DIR) / f"{model_name}_distributions.csv"))
    for ax_a, ax_b in (("distal", "perpendicular"), ("longitudinal", "lateral")):
        if ax_a in bins and ax_b in bins:
            axis_var = {}
            for ax in (ax_a, ax_b):
                s = [b.params["sigma"] ** 2 for b in bins[ax]
                     if b.dist_max <= range_cap and b.params.get("sigma")]
                if s:
                    axis_var[ax] = sum(s) / len(s)
            if len(axis_var) == 2:
                return 0.5 * (axis_var[ax_a] + axis_var[ax_b])
    raise ValueError(f"no planar (distal/perp or long/lat) bins <= {range_cap} m for {model_name}")


def _filter_config_module():
    """Load src/filters/filter_config.py standalone, bypassing filters/__init__.py.

    The package __init__ has a COLD circular import (filters -> kalman_ctrv -> utils ->
    sensor_package -> sensor_fusion -> filters.kalman_ctrv), so importing
    ``filters.filter_config`` from the parent process before the fusion stack is set up
    crashes. filter_config.py itself only imports ``os``, so a path-based load is clean.
    Used only to read the auto-Q law/constant here in the parent; the worker installs the
    override via the real package module (after ``import v2v4real_replay`` resolves the cycle).
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cmr_filter_config_standalone", _REPO_ROOT / "src" / "filters" / "filter_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def auto_q_overrides(localizer_name: str, include_loc: bool, range_cap: float, k=None,
                     filters=("sabre", "ci")) -> Dict[str, float]:
    """Auto-Q: sigma_a = k*sqrt(mean R), returned as {filter_type: sigma_a}.

    R = mean per-axis position variance over the DAIR detector pair (dair_pp_veh/inf),
    plus (if ``include_loc``) the configured localizer's R (Eq. 18 R_total = R_det + R_loc).
    ``k`` defaults to filter_config.SIGMA_A_K (12.6). Calibrated for the SABRE/CI adaptive
    family (both peak at sigma_a~=2 at the detector R~=0.025 on DAIR — the qsweep finding);
    the other filter families have a different optimum at the same R, so they keep their
    hand-tuned defaults (per-family k is future work). Off-DAIR detector templates would
    need the per-source names generalized.
    """
    fc = _filter_config_module()
    if k is None:
        k = fc.SIGMA_A_K
    sigma_a_from_R = fc.sigma_a_from_R
    R_det = sum(_mean_axis_variance(f"dair_pp_{v}", range_cap) for v in ("veh", "inf")) / 2.0
    R_total, parts = R_det, f"R_det={R_det:.4f}"
    if include_loc:
        if "{vehicle}" in localizer_name:
            R_loc = sum(_mean_axis_variance(localizer_name.format(vehicle=v), range_cap)
                        for v in ("veh", "inf")) / 2.0
        else:
            R_loc = _mean_axis_variance(localizer_name, range_cap)
        R_total = R_det + R_loc
        parts += f" + R_loc={R_loc:.4f} = {R_total:.4f}"
    sa = sigma_a_from_R(R_total, k)
    print(f"[auto-Q] {parts} -> sigma_a = {k:g}*sqrt(R) = {sa:.3f}  (filters: {', '.join(filters)})")
    return {ft: sa for ft in filters}


def build_config(streams: List[str], cfg_overrides: Dict) -> Dict:
    """Assemble the DAIR run_scenario config for the given streams.

    Bakes in the DAIR-specific defaults documented in the module docstring;
    ``cfg_overrides`` (CLI knobs) are layered on top.
    """
    config = {
        "detector_name": "dair_pp_{vehicle}",
        "localizer_name": "rtk_v2v4real",
        "detector_max_range": 110.0,
        "score_threshold": 0.0,
        "lifecycle_mode": "ab3dmot",
        "self_report_egos": False,
        "lidar_range": list(DAIR_LIDAR_RANGE),
        "gt_range": list(DAIR_LIDAR_RANGE),
        "streams_keep": list(streams),
        "record_tape_for": list(streams),
    }
    config.update(cfg_overrides)
    # streams_keep / record_tape_for always track the requested streams.
    config["streams_keep"] = list(streams)
    config["record_tape_for"] = list(streams)
    return config


def materialize_instant_tune(protocol: str, enable_gate: bool) -> Dict:
    """Build the consumed data-driven CSVs from the DAIR instant-tune configs.

    Reads the per-source instant-tune JSONs
    ``results/instant_tune/dair_pp_{veh,inf}/<protocol>/instant_config.json``,
    writes a combined ``recommended_lifecycle.csv`` + ``birth_gate_curves.csv``
    into ``results/instant_tune/dair_pp_combined/<protocol>/_generated/``, and
    returns the ``cfg_overrides`` that point ``v2v4real_replay`` at them.

    The per-source LIFECYCLE values follow the validated DAIR recipe (baton
    dair_integration_1, the 0.7174 win): the m*=1 immediate-emit lifecycle —
    ``confirm_log_odds`` = the L_confirm value (``tracker_params``), kill = -3.0,
    flat birth gate p_birth = 0.0 (the proper birth gate is the SEPARATE
    per-range ``data_driven_gate``, not this flat p_tp gate) — but with
    ``log_lr_miss`` taken PER SOURCE from ``solver_lifecycle_per_source``
    (veh -0.2323, inf -0.1261). Because veh/inf differ on log_lr_miss but agree
    on confirm/p_birth, only the per-source miss-decay BLEND activates; confirm
    and the flat gate stay global scalars — exactly the intended isolation.

    The per-source ``<detector>_polar.csv`` bin tables already sit next to the
    JSONs (written by derive_instant_config.py); the gate curve CSV references
    them by detector name and ``--data-driven-gate-polar-dir`` points the loader
    at a dir containing both.
    """
    import csv as _csv
    import shutil

    sources = {"veh": "dair_pp_veh", "inf": "dair_pp_inf"}
    out_dir = _REPO_ROOT / "results" / "instant_tune" / "dair_pp_combined" / protocol / "_generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs: Dict[str, Dict] = {}
    for veh, det in sources.items():
        jpath = _REPO_ROOT / "results" / "instant_tune" / det / protocol / "instant_config.json"
        if not jpath.exists():
            sys.exit(f"instant-tune config missing: {jpath}")
        with open(jpath) as f:
            configs[veh] = json.load(f)
        # Copy the per-source polar bin table alongside the generated CSVs so a
        # single --data-driven-gate-polar-dir resolves every source.
        polar_src = jpath.parent / f"{det}_polar.csv"
        if polar_src.exists():
            shutil.copy2(polar_src, out_dir / f"{det}_polar.csv")

    alpha = float(configs["veh"]["tracker_params"].get("data_driven_gate_alpha", 99.0))
    gate_k = float(configs["veh"]["tracker_params"].get("data_driven_gate_k", 10.0))

    # --- recommended_lifecycle.csv (per-source rows) ---
    life_csv = out_dir / "recommended_lifecycle.csv"
    fields = ["detector", "p_birth", "log_lr_miss", "kill_log_odds",
              "confirm_log_odds", "tape_score_miss_decay",
              "p_miss_tp_observed", "l_tp_observed", "n_tracks_observed"]
    # confirm + tape decay come from the global m*=1 lifecycle (veh's tracker_params
    # is the fleet scalar fallback; identical confirm across sources keeps the
    # confirm blend inactive). p_birth forced to 0.0 = flat gate off.
    confirm = float(configs["veh"]["tracker_params"]["confirm_log_odds"])   # 0.8473
    kill = float(configs["veh"]["tracker_params"]["kill_log_odds"])         # -3.0
    tape_decay = float(configs["veh"]["tracker_params"]["tape_score_miss_decay"])
    with open(life_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for veh, det in sources.items():
            lc = configs[veh]["solver_lifecycle_per_source"][det]
            w.writerow({
                "detector": det,
                "p_birth": 0.0,                       # flat gate off (per-range gate is separate)
                "log_lr_miss": lc["log_lr_miss"],      # PER SOURCE (the heterogeneity to exploit)
                "kill_log_odds": kill,
                "confirm_log_odds": confirm,           # m*=1 immediate-emit (global scalar)
                "tape_score_miss_decay": tape_decay,
                "p_miss_tp_observed": lc.get("p_miss_tp_observed", ""),
                "l_tp_observed": lc.get("l_tp_observed", ""),
                "n_tracks_observed": lc.get("n_tracks_observed", 0),
            })

    # --- birth_gate_curves.csv (one polar row per source detector) ---
    gate_csv = out_dir / "birth_gate_curves.csv"
    with open(gate_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["detector", "alpha", "fit_type", "a_quad", "b_quad",
                    "c_quad", "a_lin", "b_lin", "n_range_bins"])
        for veh, det in sources.items():
            fits = configs[veh]["birth_gate"][det].get("fits", {})
            w.writerow([det, f"{alpha}", "amota_bayes_polar",
                        fits.get("quad_a", 0.0), fits.get("quad_b", 0.0),
                        fits.get("quad_c", 0.0), fits.get("lin_a", 0.0),
                        fits.get("lin_b", 0.0), int(fits.get("n_range_bins", 0))])

    overrides: Dict = {
        "lifecycle_mode": "log_odds",
        "data_driven_lifecycle": True,
        "data_driven_lifecycle_csv": str(life_csv),
        # Global scalar fallbacks for the single-source degenerate path (these
        # are also what get used for confirm/kill/p_birth/tape_decay since the
        # CSV rows agree on them). log_lr_miss scalar is irrelevant once the
        # per-source blend is active, but pass veh's for completeness.
        "confirm_log_odds": confirm,
        "kill_log_odds": kill,
        "p_tp_birth_gate": 0.0,
        "tape_score_miss_decay": tape_decay,
    }
    if enable_gate:
        overrides.update({
            "data_driven_gate_alpha": alpha,
            "data_driven_gate_fit": "polar",
            "data_driven_gate_k": gate_k,
            "data_driven_gate_csv": str(gate_csv),
            "data_driven_gate_polar_dir": str(out_dir),
            # The per-range gate sets obj.p_tp = sigmoid(k·(score−gate(d))); to
            # let it actually reject sub-gate detections at birth the flat
            # birth-gate threshold must sit at the sigmoid neutral point 0.5
            # (score=gate → p_tp=0.5). Below that → rejected; at/above → born.
            "p_tp_birth_gate": 0.5,
        })
    return overrides


def run_one(scenario_dir: Path, out_root: Path, streams: List[str],
            cfg_overrides: Dict) -> Dict:
    """Run one scenario and write `<out_root>/<stream>/<scenario>.json` per stream.

    Returns a small per-scenario summary (frames + per-stream track-row counts).
    """
    import v2v4real_replay  # imported here so ProcessPool workers pick it up

    config = build_config(streams, cfg_overrides)
    # Auto-Q bridge: the filters call get_sigma_a() without a config (they are built
    # deep in the fusion stack), so install the run's sigma_a override process-locally
    # here in the worker, before any filter is constructed. No-op unless --auto-q set.
    if config.get("sigma_a_override"):
        from filters.filter_config import set_sigma_a_overrides
        set_sigma_a_overrides(config["sigma_a_override"])
    result = v2v4real_replay.run_scenario(str(scenario_dir), config)
    meta = v2v4real_replay.load_scenario(str(scenario_dir)).meta
    split = meta.get("split", scenario_dir.name.split("__", 1)[0])
    scenario = str(meta.get("scenario", scenario_dir.name.split("__", 1)[-1]))

    tapes = result.get("match_tape_per_stream", {})
    per_stream_counts = {}
    for stream in streams:
        frames = tape_to_track_frames(tapes.get(stream, []))
        per_stream_counts[stream] = sum(len(f["tracks"]) for f in frames)
        stream_dir = out_root / stream
        stream_dir.mkdir(parents=True, exist_ok=True)
        with open(stream_dir / f"{scenario}.json", "w") as f:
            json.dump({
                "scenario": scenario,
                "split": split,
                "stream": stream,
                "config": config,
                "frames": frames,
            }, f)
    return {
        "scenario": scenario,
        "split": split,
        "frames": result.get("iterations", 0),
        "track_rows": per_stream_counts,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--export-root", type=Path,
        default=Path("/media/rave/eddie_drive1/dair_v2x_seq/cmr_export_spd"),
        help="Root of the DAIR CMR export (contains <split>__<scenario>/ dirs).",
    )
    parser.add_argument(
        "--out-root", type=Path,
        default=Path("/media/rave/eddie_drive1/dair_v2x_seq/cmr_tracks_out"),
        help="Where to write per-stream world-frame track JSONs (off repo root).",
    )
    parser.add_argument(
        "--streams", default=",".join(DEFAULT_STREAMS),
        help="Comma-separated fusion stream keys, or 'all' for every built "
             "stream. Default: the LF-analog control + SABRE GPEM progression "
             "(polar excluded — needs B0 regen).",
    )
    parser.add_argument("--splits", default=None,
                        help="Comma-separated splits to run (e.g. 'val'). "
                             "Default: every split found under --export-root.")
    parser.add_argument("--limit-seqs", type=int, default=None,
                        help="Only run the first N scenarios (smoke testing).")
    parser.add_argument("-j", "--workers", type=int, default=8,
                        help="Parallel workers (one per scenario). Default 8.")
    parser.add_argument("--lifecycle-mode", default="ab3dmot",
                        choices=["legacy", "log_odds", "ab3dmot"],
                        help="Track lifecycle. Default ab3dmot (heterogeneous "
                             "mixed-detector input: the veh and inf detectors "
                             "are separately trained).")
    parser.add_argument("--ab3dmot-min-hits", type=int, default=3,
                        help="ab3dmot lifecycle: confirm after this many hits.")
    parser.add_argument("--ab3dmot-max-age", type=int, default=2,
                        help="ab3dmot lifecycle: kill after this many misses.")
    parser.add_argument("--score-threshold", type=float, default=0.0,
                        help="Drop detections below this score. Default 0.0 — "
                             "DAIR's detector already gates at its inference op "
                             "point, so the LF baseline saw all of these.")
    parser.add_argument("--detector-max-range", type=float, default=110.0,
                        help="GPEM covariance clamp / range cutoff (m). Default "
                             "110 contains DAIR's 100 m forward eval corners.")
    parser.add_argument("--localizer-name", default="rtk_v2v4real",
                        help="Localizer model providing loc_cov (added to perc "
                             "covariance, Eq. 18). Default rtk_v2v4real = the ~0 "
                             "RTK floor (the no-loc baseline). Pass "
                             "'dair_loc_{vehicle}' for the per-source DAIR "
                             "localizer (dair_loc_veh RTK / dair_loc_inf the "
                             "OU-characterized infrastructure error) — the "
                             "no-oracle loc_cov contribution (S2/S3).")
    # log_odds lifecycle params (global scalars). Used only when
    # --lifecycle-mode log_odds. These let an externally-derived (instant-tune)
    # lifecycle be passed straight in without the per-source data_driven_lifecycle
    # path (which is V2V4Real-specific). None = fall back to the Fusion defaults.
    parser.add_argument("--p-tp-birth-gate", type=float, default=None,
                        help="log_odds: min detection P(TP) to birth a track.")
    parser.add_argument("--confirm-log-odds", type=float, default=None,
                        help="log_odds: existence-score threshold to confirm a track.")
    parser.add_argument("--kill-log-odds", type=float, default=None,
                        help="log_odds: existence-score floor below which a track dies.")
    parser.add_argument("--log-lr-miss", type=float, default=None,
                        help="log_odds: per-miss log-likelihood-ratio decrement (<0).")
    parser.add_argument("--tape-score-miss-decay", type=float, default=None,
                        help="log_odds: coast-track score decay per missed frame.")
    # --- Per-source data-driven lifecycle + per-range birth gate -------------
    # These wire the per-source log_odds lifecycle BLEND and the GPEM-style
    # per-range birth gate (both generalized from V2V4Real to iterate
    # scenario.meta["vehicles"]) for the DAIR veh/inf pair. Params come from the
    # instant-tune solver (results/instant_tune/dair_pp_{veh,inf}/<proto>/), so
    # --instant-tune is the one-flag way to enable both; the lower-level
    # --data-driven-* flags are kept for manual A/B against hand-built CSVs.
    parser.add_argument("--instant-tune", default=None,
                        help="Enable the DAIR per-source instant-tune config: "
                             "loads results/instant_tune/dair_pp_{veh,inf}/<PROTO>/"
                             "instant_config.json, materializes the consumed "
                             "recommended_lifecycle.csv + birth_gate_curves.csv "
                             "into a _generated/ dir, and turns on "
                             "data_driven_lifecycle (per-source log_lr_miss blend). "
                             "Pass the protocol token, e.g. 'ours' or 'v2v'.")
    parser.add_argument("--instant-tune-gate", action="store_true",
                        help="With --instant-tune: ALSO enable the per-range "
                             "birth gate (data_driven_gate). Off → the proper "
                             "per-range gate is disabled (flat birth gate stays "
                             "0.0), isolating the per-source lifecycle effect.")
    parser.add_argument("--data-driven-lifecycle", action="store_true",
                        help="Enable per-source log_odds lifecycle from a CSV "
                             "(needs --data-driven-lifecycle-csv). Implied by "
                             "--instant-tune.")
    parser.add_argument("--data-driven-lifecycle-csv", type=Path, default=None,
                        help="recommended_lifecycle.csv with per-source rows "
                             "(detector names dair_pp_veh / dair_pp_inf).")
    parser.add_argument("--data-driven-gate", action="store_true",
                        help="Enable the per-range birth gate (needs "
                             "--data-driven-gate-alpha + curves). Implied by "
                             "--instant-tune-gate.")
    parser.add_argument("--data-driven-gate-alpha", type=float, default=None,
                        help="Birth-gate alpha key (matches the curve CSV rows; "
                             "instant-tune uses 99.0).")
    parser.add_argument("--data-driven-gate-csv", type=Path, default=None,
                        help="birth_gate_curves.csv (one row per source detector).")
    parser.add_argument("--data-driven-gate-polar-dir", type=Path, default=None,
                        help="Dir holding the per-source <detector>_polar.csv "
                             "bin tables (polar gate fit).")
    parser.add_argument("--data-driven-gate-fit", default="quadratic",
                        choices=["static", "linear", "quadratic", "polar"],
                        help="Birth-gate curve fit form. instant-tune uses polar.")
    # --- Auto-Q: derive process noise sigma_a from the calibrated R --------------
    parser.add_argument("--auto-q", action="store_true",
                        help="Derive process noise sigma_a closed-form from the GPEM mean R "
                             "(sigma_a = k*sqrt(R), k=12.6) instead of the hand-tuned default, "
                             "for the SABRE/CI family. Default OFF — production unaffected.")
    parser.add_argument("--auto-q-include-loc", action="store_true",
                        help="--auto-q: add the localizer R to the mean R (Eq. 18 "
                             "R_total = R_det + R_loc) so an inflated loc cov auto-raises sigma_a.")
    parser.add_argument("--auto-q-k", type=float, default=None,
                        help="--auto-q: override the law constant k (default "
                             "filter_config.SIGMA_A_K = 12.6).")
    parser.add_argument("--auto-q-range-cap", type=float, default=80.0,
                        help="--auto-q: max range (m) of the R bins averaged. Default 80.")
    args = parser.parse_args()

    if args.streams.strip().lower() == "all":
        streams = None  # resolved per-scenario from the built stream set
    else:
        streams = [s.strip() for s in args.streams.split(",") if s.strip()]
    if streams is None:
        # 'all' would require enumerating STREAM_KEYS; keep the explicit default
        # set so a stray 'all' can't silently request the blocked polar streams.
        import v2v4real_replay
        streams = [k for k in v2v4real_replay.STREAM_KEYS if "polar" not in k]
        print(f"--streams all → {len(streams)} non-polar streams")

    splits = [s.strip() for s in args.splits.split(",")] if args.splits else None
    scenarios = discover_scenarios(args.export_root, splits)
    if args.limit_seqs is not None:
        scenarios = scenarios[: args.limit_seqs]
    if not scenarios:
        sys.exit(f"No scenarios found under {args.export_root}")

    cfg_overrides = {
        "lifecycle_mode": args.lifecycle_mode,
        "ab3dmot_min_hits": args.ab3dmot_min_hits,
        "ab3dmot_max_age": args.ab3dmot_max_age,
        "score_threshold": args.score_threshold,
        "detector_max_range": args.detector_max_range,
        "localizer_name": args.localizer_name,
    }
    # --instant-tune: load the DAIR per-source instant-tune config and let it
    # set lifecycle_mode + the data-driven CSVs. Applied FIRST so explicit CLI
    # scalars below can still override individual knobs for A/B sweeps.
    if args.instant_tune is not None:
        cfg_overrides.update(
            materialize_instant_tune(args.instant_tune, args.instant_tune_gate)
        )

    # Manual per-source data-driven toggles (for hand-built CSVs / A/B). These
    # layer on top of (or instead of) --instant-tune.
    if args.data_driven_lifecycle:
        cfg_overrides["data_driven_lifecycle"] = True
        cfg_overrides["lifecycle_mode"] = "log_odds"
    if args.data_driven_lifecycle_csv is not None:
        cfg_overrides["data_driven_lifecycle_csv"] = str(args.data_driven_lifecycle_csv)
    if args.data_driven_gate and args.data_driven_gate_alpha is None:
        sys.exit("--data-driven-gate requires --data-driven-gate-alpha "
                 "(alpha is what v2v4real_replay keys the gate on).")
    if args.data_driven_gate_alpha is not None:
        cfg_overrides["data_driven_gate_alpha"] = args.data_driven_gate_alpha
    if args.data_driven_gate_csv is not None:
        cfg_overrides["data_driven_gate_csv"] = str(args.data_driven_gate_csv)
    if args.data_driven_gate_polar_dir is not None:
        cfg_overrides["data_driven_gate_polar_dir"] = str(args.data_driven_gate_polar_dir)
    cfg_overrides["data_driven_gate_fit"] = args.data_driven_gate_fit

    # Pass through any explicitly-set log_odds lifecycle scalars. None = leave
    # whatever --instant-tune set (or the Fusion default).
    for _key, _val in (
        ("p_tp_birth_gate", args.p_tp_birth_gate),
        ("confirm_log_odds", args.confirm_log_odds),
        ("kill_log_odds", args.kill_log_odds),
        ("log_lr_miss", args.log_lr_miss),
        ("tape_score_miss_decay", args.tape_score_miss_decay),
    ):
        if _val is not None:
            cfg_overrides[_key] = _val

    # --auto-q: derive sigma_a from the calibrated R and carry it in the config so the
    # worker bridge (run_one) installs it. Default OFF => no sigma_a_override key => the
    # filters keep their hand-tuned SIGMA_A defaults (reproducibility preserved).
    if args.auto_q:
        cfg_overrides["sigma_a_override"] = auto_q_overrides(
            args.localizer_name, args.auto_q_include_loc, args.auto_q_range_cap, args.auto_q_k)

    print(f"Running {len(scenarios)} scenarios × {len(streams)} streams "
          f"→ {args.out_root}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    if args.workers <= 1:
        for sc in scenarios:
            summaries.append(run_one(sc, args.out_root, streams, cfg_overrides))
            print(f"  {summaries[-1]['split']}__{summaries[-1]['scenario']}: "
                  f"{summaries[-1]['frames']} frames")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, sc, args.out_root, streams, cfg_overrides): sc
                    for sc in scenarios}
            for fut in as_completed(futs):
                s = fut.result()
                summaries.append(s)
                print(f"  {s['split']}__{s['scenario']}: {s['frames']} frames")

    total_frames = sum(s["frames"] for s in summaries)
    totals = {st: sum(s["track_rows"].get(st, 0) for s in summaries) for st in streams}
    print(f"\nDone: {len(summaries)} scenarios, {total_frames} frames.")
    for st in streams:
        print(f"  {st:24s} {totals[st]:7d} track-rows "
              f"(~{totals[st]/max(total_frames,1):.1f}/frame)")
    print(f"\nTracks written under {args.out_root}/<stream>/<scenario>.json")
    print("Next: B3 — convert world tracks → veh-lidar/camera KITTI, "
          "run DAIR eval_tracking, compare to LF 0.5486.")


if __name__ == "__main__":
    main()
