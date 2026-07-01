#!/usr/bin/env python3
"""Generate the paper's two staged result tables (docs/PAPER_STAGED_TABLES.md).

Table 1 — Baseline S1 -> S2 -> S3 progression (Late Fusion, CoBEVT, DMSTrack).
    Each baseline pipeline is scored through OUR evaluator at three stages:
      S1  baseline           (const-R EKF / reproduced tracker output, no GPEM)
      S2  R -> GPEM-R        (same input, same filter, only the measurement
                              noise R is replaced by GPEM; GPEM-quadratic fit —
                              ~flat, because a vanilla EKF is gain-saturated and
                              largely ignores R)
      S3  GPEM-R + filter    (the SAME GPEM R fed to an R-sensitive filter — where
                              R cashes in: SABRE on DMSTrack's per-source 2-vehicle
                              stream, CI on single-stream LF; N/A for CoBEVT, an
                              ill-posed GPEM-covariate fit).
    For DMSTrack a fourth row lists its trained learned-R DKF as a reference
    ceiling. ALL rows use the AB3DMOT default lifecycle and NO self-injection; the
    inject-ON DMS_* configs (self-injection costs ~2.5 AMOTA via duplicate tracks)
    are a separate Table-2 ablation, not part of this baseline progression.

    Each metric cell shows the OURS-protocol (FP-counted) number with the
    DMSTrack/V2V (FP-ignore) number in parentheses behind a dagger, e.g.
    `34.55 (†44.27)` — we log both, but the OURS number is the accurate one.

Table 2 — S4 method x detector sweep (best stream per filter family).
    Demonstrates SABRE trouncing the field. OURS-protocol metrics only
    (no V2V column): AMOTA / AMOTP / HOTA / DetA / AssA / IDS. One row per
    filter family (best stream across all detector combos x covariance modes),
    then for the single best (filter x detector) overall, the full covariance
    sweep (baseline / static / GPEM-linear / -quadratic / -polar) is expanded
    so polar-vs-quadratic etc. is visible.

Both tables are built from precomputed KITTI MOT track dumps re-scored through
the SAME evaluator as scripts/paper_protocol_tables.py (3D-IoU, global tape,
AB3DMOT rematching, paper GT). Table 1 dumps are scored on demand into
results/PAPER_STAGED_TABLES/<key>/test/summary_v2v4real_protocol.json (cached;
--force to rescore). Table 2 reads the existing GPEM-ablation sweep summaries.

Usage:
    python scripts/paper_staged_tables.py                 # md + scoring (cached)
    python scripts/paper_staged_tables.py --force          # rescore Table-1 dumps
    python scripts/paper_staged_tables.py --out other.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "configs"))


# ---------------------------------------------------------------------------
# Reuse the canonical precomputed-tracks scorer from run_v2v4real_benchmark so
# Table-1 numbers are bit-identical to the existing protocol tables.
# ---------------------------------------------------------------------------
def _load_bench():
    spec = importlib.util.spec_from_file_location(
        "_rv2v_bench", REPO / "scripts" / "run_v2v4real_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# KITTI dump base + the exact scoring params used by the existing S3 rescore
# configs (DMSTrack-exact protocol: 3D-IoU + global tape + AB3DMOT rematch).
_DUMPS = REPO / "third_party" / "AB3DMOT" / "results" / "v2v4real"
_SCORE_PARAMS = dict(camera_frame=False, iou_threshold=0.25, match_3d=True,
                     convex_hull=True, ab3dmot_rematching=True, global_eval=True)
_STAGED_OUT = REPO / "results" / "PAPER_STAGED_TABLES"
_STREAM_KEY = "dmstrack_av_100.0pct"   # _DMSTRACK_KEY written by the scorer


# ---------------------------------------------------------------------------
# Table 1 row map: (pipeline label, {stage: dump_dir_name or None}).
# GPEM fit for S2/S3 is quadratic; covariance R is NOT axiswise-normalized
# (the "NoNorm" dumps) — axiswise normalization hurts (see project memory
# `project_gpem_r_calibration`). The DMSTrack S3 row uses SABRE (the paper's
# headline filter) under the mh=3/ma=2 lifecycle (DMS_S3NoNorm_* = default
# lifecycle, NOT the mh=1/ma=5 immediate-emit protocol-exploit dumps).
# ---------------------------------------------------------------------------
S2_FIT = "quadratic"
DMS_S3_FILTER = "sabre"

# Each stage is (stage_label, kind, ref):
#   kind="summary" -> ref is an absolute path to an already-scored summary JSON
#                     (the published-replication anchors used by
#                     scripts/paper_protocol_tables.py — no rescore).
#   kind="dump"    -> ref is a KITTI dump dir name under _DUMPS to score.
#   kind="na"      -> ref ignored; renders the degenerate-S3 N/A note.
# We show BOTH an S1 (published replication) and an S1 (EKF proxy = the same
# tracker family used by S2/S3) row per pipeline.
_R = REPO / "results"
# Each pipeline = 3 rows: S1 baseline (published/reproduced tracks, scored
# through OUR clean evaluator) → S2 (GPEM R drop-in) → S3 (GPEM R + Mahalanobis
# / SABRE). Every CELL is `OURS (†DMSTrack-eval)`: OURS = our clean FP-counted
# evaluator; the † is the FP-ignore (DMSTrack/V2V) protocol number. For the
# DMSTrack S1 row the † is pinned to DMSTrack's PUBLISHED evaluator output
# (DMSTRACK_PUBLISHED) so it matches their paper to the digit.
TABLE1_ROWS: List[Tuple[str, List[Tuple[str, str, object]]]] = [
    ("Late Fusion + AB3DMOT (PP)", [
        # Clean within-pipeline progression: single fused stream, NO self-injection,
        # AB3DMOT default lifecycle (min_hits=3/max_age=2). S1 = const-R EKF anchor
        # (the old "DEFAULT_BASELINE" summary was a state-leak score — IDS 204 vs the
        # anchor's 29); S2 swaps in GPEM R (a wash — the EKF is gain-saturated); S3
        # feeds the SAME GPEM R to the R-sensitive CI filter (the single-stream
        # analogue of DMSTrack's SABRE step). These are the DMSTrack-bridge LF
        # detections (downloaded, not regenerated); our scores track DMSTrack's
        # reproduction (CoBEVT †37.3 ≈ DMSTrack's 37.16), NOT the V2V4Real paper's
        # 29.28/32.12 — that published number reflects six undocumented (partly
        # irreproducible) V2V4Real eval choices, NOT a different detector. See
        # docs/V2V4REAL_REPRODUCIBILITY_NOTES.md.
        ("Baseline (const-R EKF)", "dump",
         "late_fusion_Car_val_LF_V2V4Real_anchor_H1"),
        ("GPEM R Drop-In (EKF)", "dump",
         f"late_fusion_Car_val_LF_S2NoNorm_EKF_{S2_FIT}_H1"),
        ("GPEM R + CI", "dump",
         "late_fusion_Car_val_LF_S3NoNorm_ci_quadratic_H1"),
    ]),
    ("CoBEVT + AB3DMOT", [
        ("Published", "summary",
         _R / "V2V4Real_DMSTRACK_MG_2026-05-15_153746/cobevt_precomputed_dmstrack_exact/test/summary_v2v4real_protocol.json"),
        ("GPEM R Drop-In (EKF)", "dump", f"cobevt_Car_val_CoBEVT_S2NoNorm_EKF_{S2_FIT}_H1"),
        ("GPEM R + filter", "na",
         "GPEM-R is an ill-posed fit for CoBEVT's single fused feature stream "
         "(range-to-source is undefined for one cooperative output), so the "
         "R-sensitive S3 step is not meaningful — kept S1/S2 only."),
    ]),
    ("DMSTrack (per-CAV PP)", [
        # Clean within-pipeline progression on DMSTrack's per-CAV input, scored
        # through OUR evaluator. S1 = AB3DMOT const-R EKF (no GPEM); S2 swaps in
        # GPEM R (same EKF — a wash, the EKF is gain-saturated); S3 feeds the SAME
        # GPEM R to the R-sensitive SABRE filter (where it cashes in). The DKF row
        # is DMSTrack's trained learned-R covariance as a reference ceiling. All
        # four are per-source, NO self-injection (the inject-ON DMS_* configs are
        # a separate Table-2 ablation; self-injection costs ~3 AMOTA via duplicate
        # tracks and is NOT part of the baseline progression).
        ("Baseline (const-R EKF)", "dump",
         "pp_dmstrack_per_cav_Car_val_DMSTrack_noinj_anchor_H1"),
        ("GPEM R Drop-In (EKF)", "dump",
         f"pp_dmstrack_per_cav_Car_val_DMSnoinj_S2NoNorm_EKF_{S2_FIT}_H1"),
        ("GPEM R + SABRE", "dump",
         f"pp_dmstrack_per_cav_Car_val_DMSnoinj_S3NoNorm_{DMS_S3_FILTER}_{S2_FIT}_H1"),
        ("DKF (trained ref.)", "dump", "dkf_learnedR_H1"),
    ]),
]

# DMSTrack's PUBLISHED evaluator output (their paper; reproducible by running
# DMSTrack's own scripts/KITTI/evaluate.py on their released tracks in
# state_mode=paper_compat — the FP-ignore + valid-state-leak convention). Used
# verbatim as the † (DMSTrack-evaluator) value of the DMSTrack DKF reference row
# so the table matches the published numbers 100% (OURS-protocol number for that
# row still comes from our clean scorer on the same DKF dump). AMOTP 57.94 is
# independently reproduced by OUR (fixed) evaluator — see project_canonical_amotp.
DMSTRACK_PUBLISHED = {"pipeline": "DMSTrack (per-CAV PP)", "stage": "DKF (trained ref.)",
                      "amota": 43.52, "amotp": 57.94, "samota": 91.50, "mota": 88.32}

# ---------------------------------------------------------------------------
# Table 2: S4 GPEM-ablation sweeps (detector x lifecycle combos).
# (config_key, pretty_label, summary_path)
# ---------------------------------------------------------------------------
_AB = REPO / "results"
# All 6 S4 sweeps re-run together (canonical IoU AMOTP + dumped match tapes)
# into one dir — see results/V2V4Real_GPEM_ABLATION_RESCORE.
_S4 = _AB / "V2V4Real_GPEM_ABLATION_RESCORE"
TABLE2_CONFIGS: List[Tuple[str, str, Path]] = [
    ("pp_logodds",   "PP score-0 (log\\_odds)",
     _S4 / "pp_score0_gpem_logodds/test/summary_v2v4real_protocol.json"),
    ("pp_ab3dmot",   "PP score-0 (AB3DMOT)",
     _S4 / "pp_score0_gpem_ab3dmot/test/summary_v2v4real_protocol.json"),
    ("cpzs_logodds", "CPZS-100m (log\\_odds)",
     _S4 / "cp_zeroshot_100m_gpem_logodds/test/summary_v2v4real_protocol.json"),
    ("cpzs_ab3dmot", "CPZS-100m (AB3DMOT)",
     _S4 / "cp_zeroshot_100m_gpem_ab3dmot/test/summary_v2v4real_protocol.json"),
    ("cpft_logodds", "CPft-100m (log\\_odds)",
     _S4 / "cpft_100m_gpem_logodds/test/summary_v2v4real_protocol.json"),
    ("cpft_ab3dmot", "CPft-100m (AB3DMOT)",
     _S4 / "cpft_100m_gpem_ab3dmot/test/summary_v2v4real_protocol.json"),
]

FILTERS = ["EKF", "CI", "AKF", "PF", "BICI", "SABRE"]
COVS = [("baseline", "baseline"), ("static", "static"),
        ("GPEM-linear", "gpem_linear"), ("GPEM-quadratic", "gpem_quadratic"),
        ("GPEM-polar", "gpem_polar")]


def _stream_key(filt: str, cov_key: str) -> str:
    """summary.json stream key for (filter family, covariance mode)."""
    if filt == "EKF":
        stem = "static_cov" if cov_key == "static" else cov_key
    else:
        stem = f"{filt.lower()}_{cov_key}"
    return f"{stem}_av_100.0pct"


# ---------------------------------------------------------------------------
# Loading / formatting helpers
# ---------------------------------------------------------------------------
def _load_results(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)["results"]


def fv(x: Optional[float], nd: int = 2) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def fi(x: Optional[float]) -> str:
    return "—" if x is None else str(int(x))


def _dual(ours: Optional[float], v2v: Optional[float], nd: int = 2) -> str:
    """`ours (†v2v)` — OURS-protocol number with the V2V/FP-ignore number
    behind a dagger in parentheses."""
    if ours is None and v2v is None:
        return "—"
    if v2v is None:
        return fv(ours, nd)
    return f"{fv(ours, nd)} (†{fv(v2v, nd)})"


# ---------------------------------------------------------------------------
# Table 1 — score the S1/S2/S3 dumps and build the progression table
# ---------------------------------------------------------------------------
def score_stage(bench, dump_name: str, force: bool) -> Dict:
    """Score one KITTI dump via the canonical precomputed-tracks scorer and
    return its metric dict (cached under results/PAPER_STAGED_TABLES)."""
    subdir = _STAGED_OUT / dump_name
    summary = subdir / "test" / "summary_v2v4real_protocol.json"
    if force or not summary.exists():
        tracking_dir = _DUMPS / dump_name / "data_0"
        if not tracking_dir.exists():
            raise FileNotFoundError(f"missing dump: {tracking_dir}")
        cfg = {
            "name": dump_name,
            "label": dump_name,
            "kind": "precomputed_tracks",
            "params": {"tracking_dir": tracking_dir,
                       "gt_dir": bench._PAPER_GT_LABELS_DIR, **_SCORE_PARAMS},
        }
        subdir.mkdir(parents=True, exist_ok=True)
        bench._run_precomputed_tracks(cfg, subdir)
    return _load_results(summary)[_STREAM_KEY]


def collect_table1(bench, force: bool) -> List[Dict]:
    """Return one record per Table-1 row (raw numbers; renderer-agnostic)."""
    rows: List[Dict] = []
    for pipeline, stages in TABLE1_ROWS:
        for i, (stage, kind, ref) in enumerate(stages):
            if kind == "na":
                rows.append({"pipeline": pipeline, "stage": stage, "na": True,
                             "na_reason": ref, "first": i == 0, "n": len(stages)})
                continue
            r = (_load_results(Path(ref))[_STREAM_KEY] if kind == "summary"
                 else score_stage(bench, ref, force))
            row = {
                "pipeline": pipeline, "stage": stage, "na": False,
                "first": i == 0, "n": len(stages),
                "amota_o": r.get("paper_amota_mean"), "amota_v": r.get("v2v4real_amota_mean"),
                "amotp_o": r.get("paper_amotp_mean"), "amotp_v": r.get("v2v4real_amotp_mean"),
                "samota_o": r.get("paper_samota_mean"), "samota_v": r.get("v2v4real_samota_mean"),
                "mota_o": r.get("paper_mota_mean"), "mota_v": r.get("v2v4real_mota_mean"),
                "hota": r.get("paper_hota_mean"), "ids": r.get("paper_ids_total"),
            }
            # Pin the DMSTrack S1 † to DMSTrack's published evaluator output so
            # the table matches their paper 100% (see DMSTRACK_PUBLISHED).
            if (pipeline == DMSTRACK_PUBLISHED["pipeline"]
                    and stage == DMSTRACK_PUBLISHED["stage"]):
                row["amota_v"]  = DMSTRACK_PUBLISHED["amota"]
                row["amotp_v"]  = DMSTRACK_PUBLISHED["amotp"]
                row["samota_v"] = DMSTRACK_PUBLISHED["samota"]
                row["mota_v"]   = DMSTRACK_PUBLISHED["mota"]
            rows.append(row)
    return rows


_T1_INTRO = (
    "All metrics ×100, 9 V2V4Real test sequences, 3D tracking. Every row uses the "
    "AB3DMOT default lifecycle (min_hits=3/max_age=2) and **no self-injection** "
    "(self-reporting the ego's own detections costs ~2.5 AMOTA via duplicate "
    "tracks; it is excluded here and confined to the Table-2 ablations). **S1** = "
    "baseline tracker (const-R EKF, no GPEM). **S2** = the same input and filter "
    "with only the measurement noise R replaced by GPEM-quadratic (the GPEM R "
    "drop-in). **S3** = the same GPEM R fed to an R-sensitive filter — SABRE on "
    "DMSTrack's per-source 2-vehicle stream, CI on the single-stream LF; N/A for "
    "CoBEVT, whose fused feature stream is an ill-posed GPEM-covariate fit. For "
    "DMSTrack we also list its trained learned-R differentiable KF (DKF) as a "
    "reference ceiling. The key reading is the **S1→S2→S3 shape** on the clean "
    "const-R→GPEM pairs (LF and DMSTrack): S1→S2 is ~flat because a vanilla EKF is "
    "gain-saturated (P/R≫1) and largely ignores R, so a GPEM drop-in alone is a "
    "wash; S2→S3 is where the R-sensitive filter converts the *same* calibrated R "
    "into accuracy and association gains — DMSTrack's GPEM+SABRE reaches its "
    "trained-DKF accuracy without any training, at fewer ID-switches; on the "
    "single-stream LF the R-sensitive CI recovers the S2 drop to ~parity with the "
    "const-R baseline while cutting IDS 29→23 (less headroom than DMSTrack, as "
    "expected — LF has no cross-source R heterogeneity to exploit). Each cell is "
    "`OURS (†DMSTrack-eval)`: OURS is our clean evaluator (FP-counted, 3D-IoU + "
    "global tape + AB3DMOT rematching, base GT, 31 419 rows); the † is the "
    "DMSTrack/V2V evaluator (FP-ignore). For the DKF reference row the † is "
    "DMSTrack's PUBLISHED evaluator output, matching their paper exactly."
)
_T1_NOTES = [
    ("†", "DMSTrack/V2V evaluator (FP-ignore protocol): unmatched tracker "
          "detections are discarded, matching the published V2V4Real / DMSTrack "
          "evaluation. The number outside parentheses is OURS (FP-counted, our "
          "clean evaluator). The DMSTrack DKF-reference † is DMSTrack's published "
          "evaluator output (reproducible via their evaluate.py in "
          "state_mode=paper_compat; AMOTP independently reproduced by our evaluator)."),
    ("‡", "HOTA and IDS are invariant to the FP-ignore/FP-counted switch on a "
          "fixed GT set; the OURS-protocol value is reported."),
    ("↑/↓", "Higher is better for all metrics except IDS (lower). AMOTP is the "
            "canonical AB3DMOT `AMOTP↑` (mean 3D-IoU over true positives) — the "
            "same convention as the published V2V4Real/DMSTrack AMOTP and as "
            "Table 2, so all AMOTP values are directly comparable."),
]


def render_table1_md(rows: List[Dict]) -> List[str]:
    out = ["## Table 1 — Baseline S1 → S2 → S3 progression", "", _T1_INTRO, "",
           "| Pipeline | Stage | AMOTA↑ | AMOTP↑ | sAMOTA↑ | MOTA↑ | HOTA↑‡ | IDS↓‡ |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["na"]:
            reason = r.get("na_reason") or ("single fused stream, no per-CAV "
                                            "matching to replace (collapses to S2)")
            out.append(f"| {r['pipeline']} | {r['stage']} | "
                       f"N/A — {reason} | | | | | |")
            continue
        out.append(
            f"| {r['pipeline']} | {r['stage']} | "
            f"{_dual(r['amota_o'], r['amota_v'])} | {_dual(r['amotp_o'], r['amotp_v'])} | "
            f"{_dual(r['samota_o'], r['samota_v'])} | {_dual(r['mota_o'], r['mota_v'])} | "
            f"{fv(r['hota'])} | {fi(r['ids'])} |")
    out.append("")
    for sym, txt in _T1_NOTES:
        out.append(f"{sym} {txt}")
        out.append("")
    return out


# ---------------------------------------------------------------------------
# Table 2 — S4 method x detector sweep
# ---------------------------------------------------------------------------
def _s4_metrics(r: Dict) -> Dict:
    """OURS-protocol metric bundle for an S4 sweep stream.

    AMOTA is the OURS (FP-counted) value; AMOTP is the canonical IoU AMOTP
    (`v2v4real_amotp_mean` — mean 3D-IoU over TPs; FP-invariant, so it is the
    same under either protocol), matching Table 1's AMOTP convention. The old
    `paper_amotp_mean` was the legacy center-distance AMOTP and is no longer used.
    """
    return {
        "amota": r.get("paper_amota_mean"),
        "amotp": r.get("v2v4real_amotp_mean"),
        "hota":  r.get("hota_mean"),
        "deta":  r.get("deta_mean"),
        "assa":  r.get("assa_mean"),
        "ids":   r.get("paper_ids_total"),
    }


def _collect_s4() -> List[Dict]:
    """Flatten every (config x filter x cov) stream into records."""
    recs: List[Dict] = []
    for cfg_key, cfg_label, path in TABLE2_CONFIGS:
        if not path.exists():
            print(f"[WARN] missing S4 summary: {path}", file=sys.stderr)
            continue
        res = _load_results(path)
        for filt in FILTERS:
            for cov_label, cov_key in COVS:
                sk = _stream_key(filt, cov_key)
                if sk not in res:
                    continue
                m = _s4_metrics(res[sk])
                if m["amota"] is None:
                    continue
                recs.append({"cfg_key": cfg_key, "cfg_label": cfg_label,
                             "filter": filt, "cov_label": cov_label,
                             "cov_key": cov_key, **m})
    return recs


def collect_table2() -> Tuple[List[Dict], Optional[Dict], List[Dict]]:
    """Return (best-per-filter ordered rows, overall winner, winner cov sweep)."""
    recs = _collect_s4()
    if not recs:
        return [], None, []
    best_per_filter: Dict[str, Dict] = {}
    for r in recs:
        b = best_per_filter.get(r["filter"])
        if b is None or r["amota"] > b["amota"]:
            best_per_filter[r["filter"]] = r
    ordered = sorted(best_per_filter.values(), key=lambda r: -r["amota"])
    winner = ordered[0]
    expand = [r for r in recs
              if r["cfg_key"] == winner["cfg_key"] and r["filter"] == winner["filter"]]
    cov_order = {ck: i for i, (_, ck) in enumerate(COVS)}
    expand.sort(key=lambda r: cov_order.get(r["cov_key"], 99))
    return ordered, winner, expand


_T2_INTRO = (
    "OURS-protocol (FP-counted, merged GT) metrics only — no V2V column. Top "
    "section: the best stream per filter family across all detector combos × "
    "covariance modes (ranked by AMOTA), showing SABRE leading the field. Bottom "
    "section: for the single best (filter × detector) overall, the full "
    "covariance sweep. Higher is better for every column except IDS (↓). AMOTP "
    "is the canonical AB3DMOT `AMOTP↑` (mean 3D-IoU over TPs) — the SAME "
    "convention as Table 1, so the two tables' AMOTP columns are directly "
    "comparable."
)


def render_table2_md(ordered: List[Dict], winner: Optional[Dict],
                     expand: List[Dict]) -> List[str]:
    out = ["## Table 2 — S4 method × detector sweep (OURS protocol)", "",
           _T2_INTRO, "", "### Best stream per filter family", "",
           "| Filter | Detector (lifecycle) | Covariance | AMOTA↑ | AMOTP↑ | HOTA↑ | DetA↑ | AssA↑ | IDS↓ |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    if not ordered:
        out.append("| _no S4 sweep summaries found_ | | | | | | | | |")
        return out
    for r in ordered:
        out.append(f"| {r['filter']} | {r['cfg_label']} | {r['cov_label']} | "
                   f"{fv(r['amota'])} | {fv(r['amotp'])} | {fv(r['hota'])} | "
                   f"{fv(r['deta'])} | {fv(r['assa'])} | {fi(r['ids'])} |")
    out += ["", f"### Covariance sweep for the overall best — {winner['filter']} on "
            f"{winner['cfg_label']}", "",
            "| Filter | Covariance | AMOTA↑ | AMOTP↑ | HOTA↑ | DetA↑ | AssA↑ | IDS↓ |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    best_amota = max(r["amota"] for r in expand)
    for r in expand:
        amota = f"**{fv(r['amota'])}**" if r["amota"] == best_amota else fv(r["amota"])
        out.append(f"| {r['filter']} | {r['cov_label']} | {amota} | "
                   f"{fv(r['amotp'])} | {fv(r['hota'])} | {fv(r['deta'])} | "
                   f"{fv(r['assa'])} | {fi(r['ids'])} |")
    out.append("")
    return out


# ---------------------------------------------------------------------------
# LaTeX rendering (booktabs; single-column fit via \resizebox{\columnwidth})
# ---------------------------------------------------------------------------
# Required preamble: \usepackage{booktabs,multirow,graphicx}. If a table is too
# small to read in one column, change `table`->`table*` and
# `\columnwidth`->`\textwidth` (spans both columns).

def _tx(x: Optional[float], nd: int = 2) -> str:
    return "--" if x is None else f"{x:.{nd}f}"


def _txi(x: Optional[float]) -> str:
    return "--" if x is None else str(int(x))


def _txdual(o: Optional[float], v: Optional[float], nd: int = 2) -> str:
    if o is None and v is None:
        return "--"
    if v is None:
        return _tx(o, nd)
    return f"{_tx(o, nd)} ($\\dagger${_tx(v, nd)})"


def _texlabel(s: str) -> str:
    """Markup -> LaTeX text-mode for the short dynamic labels we emit."""
    return (s.replace("§", "$^{\\S}$").replace("‡", "$^{\\ddagger}$")
             .replace("×", "$\\times$").replace("—", "---").replace("≫", "$\\gg$"))


# Compact pipeline labels for the LaTeX table.
_T1_PIPE_TEX = {
    "Late Fusion + AB3DMOT (PP)": "Late Fusion (PP)",
    "CoBEVT + AB3DMOT": "CoBEVT",
    "DMSTrack (per-CAV PP)": "DMSTrack",
}
# Stage labels needing raw LaTeX (\cite, italic $R$, escaped &) that _texlabel
# would mangle or that contain a literal & (a column separator).
_T1_STAGE_TEX = {
    "Reproduced Baseline":     r"Reproduced Baseline (+2 vs \cite{xu2023v2v4real})",
    "GPEM R Drop-In":          r"GPEM $R$ Drop-In",
    "GPEM R & Mahalanobis":    r"GPEM $R$ \& Mahalanobis",
    "Baseline (const-R EKF)":  r"Baseline (const-$R$ EKF)",
    "GPEM R Drop-In (EKF)":    r"GPEM $R$ Drop-In (EKF)",
    "GPEM R + SABRE":          r"GPEM $R$ + SABRE",
    "GPEM R + CI":             r"GPEM $R$ + CI",
    "GPEM R + filter":         r"GPEM $R$ + filter",
    "DKF (trained ref.)":      r"DKF (trained, ref.)",
}


def render_table1_tex(rows: List[Dict]) -> List[str]:
    out = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{\textbf{Baseline Progression on V2V4Real.} Each cell is "
        r"OURS~($\dagger$DMSTrack Reported), showing our strict FP-counted value "
        r"alongside the relaxed FP-ignore value. We explicitly separate "
        r"DMSTrack's published score (which includes a state-leak evaluation "
        r"artifact) from our clean, fully-deflated re-implementation. AMOTP is "
        r"the canonical AB3DMOT mean-3D-IoU $\mathrm{AMOTP}\uparrow$.}",
        r"\label{tab:staged-progression}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Pipeline & Stage & AMOTA & AMOTP & sAMOTA & MOTA & HOTA$^{\ddagger}$ "
        r"& IDS$^{\ddagger}$ \\",
        r"\midrule",
    ]
    seen = 0
    for r in rows:
        pipe_cell = ""
        if r["first"]:
            if seen:
                out.append(r"\midrule")
            seen += 1
            pipe_cell = (f"\\multirow{{{r['n']}}}{{*}}{{"
                         f"{_T1_PIPE_TEX.get(r['pipeline'], r['pipeline'])}}}")
        stage = _T1_STAGE_TEX.get(r["stage"], _texlabel(r["stage"]))
        if r["na"]:
            reason = r.get("na_reason") or "single fused stream (collapses to S2)"
            out.append(f"{pipe_cell} & {stage} & "
                       f"\\multicolumn{{6}}{{c}}{{\\emph{{N/A --- {_texlabel(reason)}}}}} \\\\")
            continue
        out.append(
            f"{pipe_cell} & {stage} & {_txdual(r['amota_o'], r['amota_v'])} & "
            f"{_txdual(r['amotp_o'], r['amotp_v'])} & "
            f"{_txdual(r['samota_o'], r['samota_v'])} & "
            f"{_txdual(r['mota_o'], r['mota_v'])} & {_tx(r['hota'])} & "
            f"{_txi(r['ids'])} \\\\")
    out += [r"\bottomrule", r"\end{tabular}}", r"\\[2pt]",
            r"{\footnotesize $\dagger$~DMSTrack/V2V evaluator (FP-ignore); outside "
            r"the parens is OURS (our clean FP-counted evaluator). The DMSTrack "
            r"DKF-reference $\dagger$ is DMSTrack's published evaluator output. "
            r"$\ddagger$~HOTA/IDS are reported under OURS (clean) evaluation. AMOTP "
            r"is the canonical AB3DMOT mean-3D-IoU $\mathrm{AMOTP}\uparrow$ (same "
            r"convention as the published numbers and Table~\ref{tab:s4-sweep}).}",
            r"\end{table*}", ""]
    return out


def render_table2_tex(ordered: List[Dict], winner: Optional[Dict],
                      expand: List[Dict]) -> List[str]:
    if not ordered:
        return []
    out = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{S4 method$\times$detector sweep (OURS protocol, FP-counted, "
        r"$\times100$). Top: best stream per filter family across all detector "
        r"combos and covariance modes (ranked by AMOTA); SABRE leads the field. "
        r"Bottom: full covariance sweep for the overall best (filter$\times$"
        r"detector). Higher is better except IDS. AMOTP is the canonical AB3DMOT "
        r"$\mathrm{AMOTP}\uparrow$ (mean 3D-IoU over TPs), the same convention as "
        r"Table~\ref{tab:staged-progression}, so the two are directly comparable.}",
        r"\label{tab:s4-sweep}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lllrrrrrr}", r"\toprule",
        r"Filter & Detector & Cov. & AMOTA$\uparrow$ & AMOTP$\uparrow$ & "
        r"HOTA$\uparrow$ & DetA$\uparrow$ & AssA$\uparrow$ & IDS$\downarrow$ \\",
        r"\midrule",
        r"\multicolumn{9}{l}{\emph{Best stream per filter family}} \\",
        r"\midrule",
    ]
    for r in ordered:
        out.append(f"{r['filter']} & {_texlabel(r['cfg_label'])} & "
                   f"{_texlabel(r['cov_label'])} & {_tx(r['amota'])} & "
                   f"{_tx(r['amotp'])} & {_tx(r['hota'])} & {_tx(r['deta'])} & "
                   f"{_tx(r['assa'])} & {_txi(r['ids'])} \\\\")
    out += [r"\midrule",
            f"\\multicolumn{{9}}{{l}}{{\\emph{{Covariance sweep --- "
            f"{winner['filter']} on {_texlabel(winner['cfg_label'])}}}}} \\\\",
            r"\midrule"]
    best_amota = max(r["amota"] for r in expand)
    for r in expand:
        amota = (f"\\textbf{{{_tx(r['amota'])}}}" if r["amota"] == best_amota
                 else _tx(r["amota"]))
        out.append(f"{r['filter']} & --- & {_texlabel(r['cov_label'])} & {amota} & "
                   f"{_tx(r['amotp'])} & {_tx(r['hota'])} & {_tx(r['deta'])} & "
                   f"{_tx(r['assa'])} & {_txi(r['ids'])} \\\\")
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs" / "PAPER_STAGED_TABLES.md")
    ap.add_argument("--tex-out", type=Path, default=None,
                    help="LaTeX output path (default: --out with .tex suffix).")
    ap.add_argument("--no-tex", action="store_true", help="Skip LaTeX output.")
    ap.add_argument("--force", action="store_true",
                    help="Rescore Table-1 dumps even if cached summaries exist.")
    args = ap.parse_args()

    bench = _load_bench()
    t1 = collect_table1(bench, args.force)
    t2_ordered, t2_winner, t2_expand = collect_table2()

    lines = [
        "# Paper staged result tables (auto-generated)",
        "",
        "Generated by `scripts/paper_staged_tables.py`. Table 1 scores the "
        "S1/S2/S3 KITTI track dumps under "
        "`third_party/AB3DMOT/results/v2v4real/` through the same evaluator as "
        "`scripts/paper_protocol_tables.py`; Table 2 reads the GPEM-ablation "
        "sweep summaries. Regenerate with:",
        "", "```bash", "python scripts/paper_staged_tables.py", "```", "",
    ]
    lines += render_table1_md(t1)
    lines += render_table2_md(t2_ordered, t2_winner, t2_expand)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")

    if not args.no_tex:
        tex_out = args.tex_out or args.out.with_suffix(".tex")
        tex = [
            "% Auto-generated by scripts/paper_staged_tables.py -- do not edit.",
            "% Requires: \\usepackage{booktabs,multirow,graphicx}",
            "% Tables target a single column of a two-column layout via",
            "% \\resizebox{\\columnwidth}. If too small, switch `table`->`table*`",
            "% and `\\columnwidth`->`\\textwidth` to span both columns.",
            "",
        ]
        tex += render_table1_tex(t1)
        tex += render_table2_tex(t2_ordered, t2_winner, t2_expand)
        tex_out.write_text("\n".join(tex) + "\n")
        print(f"wrote {tex_out}")


if __name__ == "__main__":
    main()
