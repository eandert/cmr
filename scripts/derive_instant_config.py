#!/usr/bin/env python3
"""Instant-tune: derive the FULL tracker configuration from GPEM characterization.

Implements Phase P1 of docs/INSTANT_TUNE_SPEC.md (§2.2 F, L, C, K, G, Rng, W)
on top of the shipped AMOTA-Bayes solver (§2.1, scripts/solve_amota_bayes_gate.py,
imported read-only). For each detector × protocol it emits a single COMPLETE

    results/instant_tune/<detector>/<protocol>/instant_config.json

containing every tunable the tracker needs (gate curves, lifecycle, floor,
coast, sigmoid steepness, association gate, max range, protocol weight w)
plus provenance (source CSV paths, formula version, derivation wall-clock).

Inputs are the per-detector `polar_binned_errors.csv` characterization files
(5 m × 10° bins; Block A–G trajectory/score/density statistics) and the
cartesian `binned_errors.csv` (for the TP-bearing range span). NO tracker
runs are needed — every parameter below is a closed-form function of the
calibration statistics; the whole 3-detector × 2-protocol derivation takes
well under a second of CPU.

Protocols
=========
- "v2v"  — DMSTrack/V2V4Real protocol: FP-IGNORE (false positives are not
           counted by the metric; their only cost is association pollution).
- "ours" — FP-COUNTED (+ego GT): false positives and false negatives are
           priced equally by the metric.

The protocol is the ONLY human input: every parameter is re-derived per
protocol from the same calibration statistics.

Derivation summary (full formulas in each function's docstring)
===============================================================
(W)   w_v2v  = post-gate FP-mass-weighted GT bin co-occupancy (association-
              pollution probability; fixed point with the gate).
      w_ours = (L_fp + T_coast) / L_fp  — coast-tail amplification of the
              data-derived λ = L_fp/L_tp.
(F)   floor  = min over TP-bearing range bands of the per-range gate curve,
              clipped from above at the FP score-mixture P99.
(L)   m*     = argmin_m  (m−1)·N_tracks·f̄ + 1[FP-counted]·N_clu(g)·q^{m−1}
              ·D_sweep·(L_fp + T_coast); confirm_log_odds equivalent emitted.
(C)   K_v2v  = gap-length quantile (P(gap ≤ K) ≥ 0.95), K_ours = marginal
              bridge-benefit vs sweep-discounted coast-FP cost.
(K)   k      = 2 / (σ_TP + σ_FP) count-weighted over the gate's case-A bins.
(G)   mahal_gate = χ²₀.₉₉(df=2) = −2·ln(0.01) ≈ 9.2103 (closed form, df=2
              because association innovations are 2-D x/y).
(Rng) detector_max_range = largest cartesian range-bin edge with ≥5 TPs.

Usage:
    python scripts/derive_instant_config.py [--detectors pp_score0 ...]
                                            [--out results/instant_tune]
"""
import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import solve_amota_bayes_gate as sg  # read-only reuse (CSV load / solve / lifecycle)

FORMULA_VERSION = "instant_tune_v1 (2026-06-12)"

# ---------------------------------------------------------------------------
# Detector registry: bucket label → {cmr_basename: error_analysis dir}.
# pp / cpzs characterizations live in the sibling mmdetection3d clone
# (same dirs the solver's DETECTORS map points at); cpft lives in the
# bucket's persistent gpem_calibration/_error_analysis.
# ---------------------------------------------------------------------------
MMDET3D = REPO.parent / "mmdetection3d" / "work_dirs"
CPFT_EA = (REPO / "data" / "v2v4real_inputs" / "ours" / "detectors"
           / "cp_finetune_100m" / "gpem_calibration" / "_error_analysis")
# DAIR-V2X-Seq SPD PointPillars error-analysis (Block A-G trajectory/density
# stats emitted off-repo by dair_v2x/inference/characterize_spd_detectors.py
# --emit-instant-tune). One detector per sensor side; inf is POOLED over all
# 6 CIS nodes. Lives off-root on the data drive (the train detections are
# there too) rather than under the repo.
DAIR_EA = Path("/media/rave/eddie_drive1/dair_v2x_seq/dair_error_analysis")

DETECTOR_SOURCES: Dict[str, Dict[str, Path]] = {
    "pp_score0": {
        "pointpillar_v2v4real_astuff": MMDET3D / "reexport_pp_astuff" / "error_analysis",
        "pointpillar_v2v4real_tesla":  MMDET3D / "reexport_pp_tesla" / "error_analysis",
    },
    "cp_zeroshot_100m": {
        "centerpoint_100m_on_v2v4real": MMDET3D / "preds_cpzs_100m_tesla" / "error_analysis",
    },
    "cp_finetune_100m": {
        "centerpoint_100m_v2v4real_finetune_astuff":
            CPFT_EA / "centerpoint_100m_v2v4real_finetune_astuff",
        "centerpoint_100m_v2v4real_finetune_tesla":
            CPFT_EA / "centerpoint_100m_v2v4real_finetune_tesla",
    },
    # ── DAIR-V2X-Seq SPD detectors (vehicle-side + pooled infrastructure-side) ──
    "dair_pp_veh": {"dair_pp_veh": DAIR_EA / "veh"},
    "dair_pp_inf": {"dair_pp_inf": DAIR_EA / "inf"},
}

# detector_name the log_odds config should use (``{vehicle}`` expands per CAV)
DETECTOR_CONFIG_NAMES = {
    "pp_score0":        "pointpillar_v2v4real_{vehicle}",
    "cp_zeroshot_100m": "centerpoint_100m_on_v2v4real",
    "cp_finetune_100m": "centerpoint_100m_v2v4real_finetune_{vehicle}",
    # DAIR detectors are per-side (no {vehicle} expansion).
    "dair_pp_veh":      "dair_pp_veh",
    "dair_pp_inf":      "dair_pp_inf",
}

# GPEM R calibration CSVs (already-shipped §2.1 quantity; referenced for
# completeness so the JSON is a self-contained config source).
GPEM_CALIBRATION = {
    "pp_score0": [
        "data/v2v4real_inputs/ours/detectors/pp_score0/gpem_calibration/pointpillar_v2v4real_astuff_polar_calibration.csv",
        "data/v2v4real_inputs/ours/detectors/pp_score0/gpem_calibration/pointpillar_v2v4real_tesla_polar_calibration.csv",
    ],
    "cp_zeroshot_100m": [
        "data/v2v4real_inputs/ours/detectors/cp_zeroshot_100m/gpem_calibration/centerpoint_100m_on_v2v4real_polar_calibration.csv",
    ],
    "cp_finetune_100m": [
        "data/v2v4real_inputs/ours/detectors/cp_finetune_100m/gpem_calibration/centerpoint_100m_v2v4real_finetune_astuff_polar_calibration.csv",
        "data/v2v4real_inputs/ours/detectors/cp_finetune_100m/gpem_calibration/centerpoint_100m_v2v4real_finetune_tesla_polar_calibration.csv",
    ],
    # DAIR-V2X-Seq SPD per-side GPEM R calibration (already shipped, locked).
    "dair_pp_veh": ["src/data/sensor_models/dair_pp_veh_polar_calibration.csv"],
    "dair_pp_inf": ["src/data/sensor_models/dair_pp_inf_polar_calibration.csv"],
}

PROTOCOLS = ("v2v", "ours")
DEFAULT_OUT = REPO / "results" / "instant_tune"

# ── derivation constants (each documented at its point of use) ──────────────
COAST_CAP = 10          # max coast frames (1 s @ 10 Hz) — beyond this the CV
                        # prediction drifts past the 2 m match radius (no GT
                        # dynamics stats exported yet; see spec §2.2-Q).
CONFIRM_MAX = 8         # search range for the confirm delay m*
GAP_COVERAGE_V2V = 0.95  # (C) v2v: required P(real-track gap bridged ≤ K)
FP_CLIP_Q = 0.99        # (F) clip percentile of the FP score mixture
MAHAL_DF = 2            # (G) association innovations are 2-D (x, y)
MAHAL_ALPHA = 0.99      # (G) spec: mahal_gate = chi2.ppf(0.99, df)
SIGMOID_WIDTH_MULT = 2.0  # (K) sigmoid 12→88 % transition spans the overlap
P_MISS_FLOOR = 0.05     # same floor the solver uses (avoid degenerate logs)
RNG_MIN_TP = 5          # (Rng) min TPs for a range bin to count as covered


# ---------------------------------------------------------------------------
# Loading & pooled statistics
# ---------------------------------------------------------------------------

def _f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    v = (row.get(key, "") or "").strip()
    if v == "" or v.lower() == "nan":
        return default
    try:
        return float(v)
    except ValueError:
        return default


# Superset of the solver's loader: keeps the solver's field names (so solver
# functions accept these rows) and adds the Block-G density columns the new
# derivations need.
_RICH_KEYS = (
    "range_lo", "range_hi", "angle_lo", "angle_hi",
    "gt_count", "missed", "n_fp",
    "score_mean", "score_std", "fp_score_mean", "fp_score_std",
    "tp_lifetime_mean", "tp_lifetime_std", "fp_lifetime_mean", "fp_lifetime_std",
    "n_gt_tracks", "n_fp_clusters",
    "gt_density_per_scen_mean", "fp_density_per_scen_mean", "fp_per_frame",
)
_RENAMES = {  # solver-loader names for the lifecycle columns
    "tp_p_match": "tp_p_match_per_track_mean",
    "fp_p_match": "fp_p_match_per_cluster_mean",
    "tp_fragments": "tp_fragments_per_track_mean",
    "fp_fragments": "fp_fragments_per_cluster_mean",
}


def load_bins(path: Path) -> List[Dict[str, float]]:
    """Load polar_binned_errors.csv into solver-compatible rich bin dicts."""
    rows: List[Dict[str, float]] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            d = {k: _f(r, k) for k in _RICH_KEYS}
            for short, col in _RENAMES.items():
                d[short] = _f(r, col)
            rows.append(d)
    return rows


def aggregate_stats(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Pool per-bin stats into detector-level aggregates.

    Counts are summed; per-track / per-cluster / per-detection means are
    weighted by their natural sample counts (n_gt_tracks, n_fp_clusters,
    n_tp, n_fp respectively) — same weighting discipline as the solver's
    lifecycle autotune.
    """
    def wmean(key: str, wkey: str) -> float:
        num = den = 0.0
        for r in rows:
            w, v = r.get(wkey, 0.0), r.get(key, 0.0)
            if w > 0 and v > 0:
                num += v * w
                den += w
        return num / den if den > 0 else float("nan")

    n_tp = sum(max(0.0, r["gt_count"] - r["missed"]) for r in rows)
    out = {
        "n_tp": n_tp,
        "n_fp": sum(r["n_fp"] for r in rows),
        "n_gt_tracks": sum(r["n_gt_tracks"] for r in rows),
        "n_fp_clusters": sum(r["n_fp_clusters"] for r in rows),
        "L_tp": wmean("tp_lifetime_mean", "n_gt_tracks"),
        "L_fp": wmean("fp_lifetime_mean", "n_fp_clusters"),
        "p_match": wmean("tp_p_match", "n_gt_tracks"),
        "frags": wmean("tp_fragments", "n_gt_tracks"),
        "mu_tp": wmean("score_mean", "gt_count"),
        "sig_tp": wmean("score_std", "gt_count"),
    }
    out["p_miss"] = max(P_MISS_FLOOR, 1.0 - out["p_match"]) if out["p_match"] == out["p_match"] else float("nan")
    return out


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def fp_postgate_stats(rows: List[Dict[str, float]], gate: float,
                      mu_tp: float, sig_tp: float) -> Dict[str, float]:
    """Post-gate FP-stream statistics from the per-bin FP score mixture.

    Each bin's FP scores are modeled N(fp_score_mean, fp_score_std); the
    survival of a score gate g is 1 − Φ((g − μ_b)/σ_b) per bin (a Gaussian
    MIXTURE over bins — robust where a single pooled Gaussian badly
    misrepresents flood detectors like PP whose FP mass hugs 0).

    Returns:
      dets       — expected surviving FP detections
      clusters   — expected surviving FP clusters (cluster peak score ≈ its
                   detections' score distribution; first-order approximation)
      occupancy  — surviving-FP-mass-weighted P(≥1 GT in the FP's (range,
                   angle) bin in the same frame) — the association-pollution
                   probability proxy used by (W) v2v
      d_sweep    — surviving-FP-mass-weighted Φ((E[s|s≥g] − μ_TP)/σ_TP):
                   the AMOTA recall-sweep discount. An emitted FP track only
                   counts as FP at sweep operating points whose confidence
                   threshold lies below the FP's score; the thresholds are
                   set by the TP confidence distribution, so the discount is
                   the TP-score CDF evaluated at the surviving FPs' mean.
    """
    dets = clu = occ_n = dsw_n = 0.0
    sig_tp = max(sig_tp, 1e-3)
    for r in rows:
        nfp = r["n_fp"]
        if nfp <= 0:
            continue
        mu, sig = r["fp_score_mean"], max(r["fp_score_std"], 1e-3)
        a = (gate - mu) / sig
        surv = 1.0 - _phi(a)
        if surv <= 1e-12:
            continue
        es = mu + sig * _pdf(a) / surv   # truncated-normal mean of survivors
        dets += nfp * surv
        clu += r["n_fp_clusters"] * surv
        occ_n += nfp * surv * r["gt_density_per_scen_mean"]
        dsw_n += nfp * surv * _phi((es - mu_tp) / sig_tp)
    return {
        "dets": dets,
        "clusters": clu,
        "occupancy": occ_n / dets if dets > 0 else 0.0,
        "d_sweep": dsw_n / dets if dets > 0 else 0.0,
    }


def fp_score_percentile(rows: List[Dict[str, float]], q: float) -> float:
    """q-th percentile of the FP score Gaussian mixture (bisection)."""
    n_all = sum(r["n_fp"] for r in rows)
    if n_all <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        surv = sum(r["n_fp"] * (1.0 - _phi((mid - r["fp_score_mean"])
                                           / max(r["fp_score_std"], 1e-3)))
                   for r in rows if r["n_fp"] > 0)
        if surv / n_all > (1.0 - q):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# (W) protocol weight
# ---------------------------------------------------------------------------

def derive_w_ours(stats: Dict[str, float]) -> Dict[str, float]:
    """(W) FP-counted protocol weight: coast-tail amplification.

    The solver's data-derived λ = L_fp/L_tp prices an accepted FP cluster at
    its DETECTED lifetime. Under an FP-counted metric the tracker emits the
    cluster for its detected lifetime PLUS the terminal coast tail before
    the lifecycle kills it. The TP-preserving lifecycle (solver Tier 1) is
    tuned so a track survives its natural miss budget, i.e. the kill tail is

        T_coast = L_tp · p_miss_TP      (capped at COAST_CAP frames)

    so the effective FP price is amplified by

        w_ours = (L_fp + T_coast) / L_fp.

    FN and FP frames are otherwise priced equally by the FP-counted metric,
    so no further protocol factor applies.
    """
    t_coast = min(stats["L_tp"] * stats["p_miss"], COAST_CAP)
    w = (stats["L_fp"] + t_coast) / max(stats["L_fp"], 1e-6)
    return {"w": w, "t_coast": t_coast, "L_fp": stats["L_fp"],
            "L_tp": stats["L_tp"], "p_miss": stats["p_miss"]}


def derive_w_v2v(rows: List[Dict[str, float]], stats: Dict[str, float],
                 gate_for_w: Callable[[float], float],
                 iterations: int = 3) -> Dict[str, float]:
    """(W) FP-ignore protocol weight: association-pollution probability.

    Under the V2V protocol the metric prices emitted FPs at ZERO; the only
    residual cost of accepting an FP is association pollution — the FP can
    steal a real track's match (FN / IDS) only when it co-locates with a GT
    object. Proxy: the probability that the FP's (range, angle) calibration
    bin contains ≥1 GT in the same frame (Block-G `gt_density_per_scen_mean`
    occupancy), weighted by the FP mass that actually SURVIVES the gate.

    Because the gate itself depends on w, this is a small fixed point:
        w₀ = occupancy(gate=0);  wᵢ₊₁ = occupancy(gate(wᵢ))
    which converges in 1–3 iterations (the occupancy surface is smooth).

    `gate_for_w(w) -> static gate` is injected so unit tests can supply a
    synthetic solver; production passes the AMOTA-Bayes static-gate solve.
    """
    pg0 = fp_postgate_stats(rows, 0.0, stats["mu_tp"], stats["sig_tp"])
    w = max(pg0["occupancy"], 1e-3)
    history = [{"w": w, "gate": 0.0}]
    for _ in range(iterations):
        g = gate_for_w(w)
        pg = fp_postgate_stats(rows, g, stats["mu_tp"], stats["sig_tp"])
        w_new = max(pg["occupancy"], 1e-3)
        history.append({"w": w_new, "gate": g})
        if abs(w_new - w) < 1e-4:
            w = w_new
            break
        w = w_new
    return {"w": w, "fixed_point": history}


# ---------------------------------------------------------------------------
# (F) score floor
# ---------------------------------------------------------------------------

def derive_floor(per_range_by_source: Dict[str, Dict[str, float]],
                 rows: List[Dict[str, float]],
                 fp_clip_q: float = FP_CLIP_Q) -> Dict[str, float]:
    """(F) global score floor — the pre-filter UNDER the gate.

    Spec §2.2-F: the floor is the score below which the expected AMOTA
    contribution is negative even at the MOST FAVORABLE range:

        floor = min_d gate(d)        (per-range AMOTA-Bayes gate curve),

    clipped from above at a high percentile of the FP score mixture
    (P99 here: a floor above effectively-all FP scores would be pure TP
    loss — it can never pay for itself).

    Per-CAV sources are combined with a fleet MIN (the floor is a single
    global config value; min preserves the weaker CAV's TPs).

    Well-separated detectors degenerate to 0 automatically: their per-bin
    Bayes solve is case-B ("TP dominates, no gate") over whole range bands,
    so min_d gate(d) = 0.

    NOTE vs spec parenthetical "(e.g. P50_fp)": P50 of the raw FP mixture is
    dominated by the sub-floor FP mass for flood detectors (PP's FP median
    is ≈0.04) and would defeat the floor's purpose; P99 implements the
    intended "never floor above the FP distribution" clip.

    NOTE (2026-06-14): a w-weighted GLOBAL Bayes floor was tried to fix the
    P2 PP-Ours gap (F=0 there) — it over-floored and REGRESSED every Ours
    cell (cpft-ours 39.03->28.53) without fixing PP, and was reverted. The
    PP-Ours gap is a lifecycle-MODE problem (ab3dmot beats log_odds for a
    weak detector under FP-counting), not a score-floor problem. See
    docs/INSTANT_TUNE_SPEC.md §6 and [[instant-tune-p2-results]].
    """
    mins = []
    for _src, per_range in per_range_by_source.items():
        vals = [v for v in per_range.values() if v is not None]
        if vals:
            mins.append(min(vals))
    raw_min = min(mins) if mins else 0.0
    clip = fp_score_percentile(rows, fp_clip_q)
    return {"floor": max(0.0, min(raw_min, clip)),
            "min_gate_over_range": raw_min,
            "fp_clip": clip, "fp_clip_q": fp_clip_q}


# ---------------------------------------------------------------------------
# (L) lifecycle permissiveness — confirm delay m* + confirm_log_odds
# ---------------------------------------------------------------------------

def derive_confirm(stats: Dict[str, float], protocol: str,
                   postgate_clusters: float, d_sweep: float,
                   m_max: int = CONFIRM_MAX) -> Dict[str, float]:
    """(L) emission economics: optimal confirm delay m*.

    A track confirmed after m frames loses its first (m−1) frames of recall;
    with f̄ fragments per GT track (each fragment re-confirms after a kill)
    the expected recall loss per confirm step is N_tracks·f̄ frames.

    An FP cluster (geometric lifetime, mean L_fp, survival q = 1 − 1/L_fp)
    survives an m-frame confirm with probability q^{m−1}; once emitted it
    costs its remaining detected lifetime plus the terminal coast tail,
    discounted by the AMOTA recall-sweep factor D_sweep (FP tracks score far
    below the TP confidences that set the sweep thresholds, so they pollute
    only the low-threshold operating points — see fp_postgate_stats).

        J(m) = (m−1)·N_tracks·f̄
             + 1[FP-counted] · N_clu(gate) · q^{m−1} · D_sweep · (L_fp + T_coast)

        m* = argmin_{1 ≤ m ≤ m_max} J(m)

    Protocol weighting: under the FP-IGNORE protocol the indicator is 0 —
    emitted FPs carry no metric price, so delaying confirmation has zero
    metric benefit and J(m) is strictly increasing ⇒ m* = 1 (immediate
    emit). This IS the FP-ignore limit, not an assumption: the only
    remaining FP cost channel (association pollution) is already priced
    into the gate via (W) and is independent of the confirm delay to first
    order (a confirm-pending track still associates).

    Returns m*, the J table, and the log-odds confirm threshold equivalent.
    """
    q = max(0.0, 1.0 - 1.0 / max(stats["L_fp"], 1.01))
    t_coast = min(stats["L_tp"] * stats["p_miss"], COAST_CAP)
    step = stats["n_gt_tracks"] * max(stats["frags"], 1.0)
    fp_price = postgate_clusters * d_sweep * (stats["L_fp"] + t_coast)
    w_fp = 0.0 if protocol == "v2v" else 1.0

    j_table = {}
    best_m, best_j = 1, None
    for m in range(1, m_max + 1):
        j = (m - 1) * step + w_fp * fp_price * (q ** (m - 1))
        j_table[m] = j
        if best_j is None or j < best_j - 1e-9:
            best_m, best_j = m, j
    return {"m_star": best_m, "q": q, "step_cost": step,
            "fp_price": fp_price, "d_sweep": d_sweep, "j_table": j_table}


def confirm_log_odds_equivalent(m_star: int, p_birth: float,
                                p_match: float) -> float:
    """Log-odds confirm threshold equivalent of an m*-frame confirm rule.

    A track is born at S₀ = logit(p_birth); each subsequent matched frame
    adds the hit likelihood ratio ≈ logit(p_match). Confirming after m*
    detections therefore corresponds to

        confirm_log_odds = logit(p_birth) + (m* − 1) · max(logit(p_match), 0.2)

    m* = 1 ⇒ confirm_log_odds = logit(p_birth): the track is confirmed by
    its birth detection alone (immediate emit — at-or-below the single-
    detection log-odds increment). The 0.2 floor keeps the threshold
    meaningful for detectors with p_match ≈ 0.5.
    """
    p_b = min(max(p_birth, 1e-6), 1.0 - 1e-6)
    p_m = min(max(p_match, 1e-6), 1.0 - 1e-6)
    hit = max(math.log(p_m / (1.0 - p_m)), 0.2)
    return math.log(p_b / (1.0 - p_b)) + (m_star - 1) * hit


# ---------------------------------------------------------------------------
# (C) coast budget (max_age analog)
# ---------------------------------------------------------------------------

def derive_coast(stats: Dict[str, float], protocol: str,
                 postgate_clusters: float, d_sweep: float,
                 m_star: int, cap: int = COAST_CAP) -> Dict[str, float]:
    """(C) coast budget K from the TP gap statistics.

    Real-track detection gaps: with per-frame match probability p =
    tp_p_match and f̄ fragments per track, the mean gap length is

        ḡ = L_tp · (1 − p) / max(f̄ − 1, 0.25)      (clamped to [1.05, 20])

    and gap lengths are modeled geometric: P(gap ≥ k) = (1 − 1/ḡ)^{k−1}.

    v2v (FP-ignore): coasting is free in the metric (a coasted FP frame is
    ignored; a coasted TP frame that still matches GT is pure recall), so K
    is purely coverage-driven — the smallest K with P(gap ≤ K) ≥ 0.95:

        K = ceil( ln(1 − 0.95) / ln(1 − 1/ḡ) ),  capped at COAST_CAP.

    ours (FP-counted): every emitted track that dies coasts K extra frames
    that count as FPs (sweep-discounted by D_sweep, since coasted tape
    scores sit below the TP confidence band). The K-th coast frame is kept
    while its expected bridged-gap benefit exceeds that cost:

        N_gaps · (1 − 1/ḡ)^{K−1}  ≥  D_sweep · (N_tracks + N_emitted_clusters)

    where N_gaps = N_tracks·max(f̄−1, 0.25) and N_emitted_clusters =
    N_clu(gate)·q^{m*−1} (clusters surviving gate + confirm).
    """
    G = max(stats["frags"] - 1.0, 0.25)
    gbar = min(max(stats["L_tp"] * stats["p_miss"] / G, 1.05), 20.0)
    keep = 1.0 - 1.0 / gbar
    if protocol == "v2v":
        k = math.ceil(math.log(1.0 - GAP_COVERAGE_V2V) / math.log(keep))
        k = max(1, min(cap, k))
        return {"K": k, "gbar": gbar, "rule": "gap_coverage_0.95"}
    q = max(0.0, 1.0 - 1.0 / max(stats["L_fp"], 1.01))
    n_gaps = stats["n_gt_tracks"] * G
    n_emit = postgate_clusters * (q ** (m_star - 1))
    rhs = d_sweep * (stats["n_gt_tracks"] + n_emit)
    k = 1
    for cand in range(1, cap + 1):
        if n_gaps * (keep ** (cand - 1)) >= rhs:
            k = cand
        else:
            break
    return {"K": k, "gbar": gbar, "rule": "marginal_bridge_vs_coast_fp",
            "n_gaps": n_gaps, "n_emitted": n_emit, "rhs": rhs}


# ---------------------------------------------------------------------------
# (K) sigmoid steepness
# ---------------------------------------------------------------------------

def derive_sigmoid_k(per_bin: List[Dict[str, float]],
                     width_mult: float = SIGMOID_WIDTH_MULT) -> Dict[str, float]:
    """(K) soft-gate sigmoid steepness from the TP/FP score overlap width.

    The soft gate accepts a detection with p = sigmoid(k·(s − gate)). Its
    12 %→88 % transition spans Δs = 4/k; matching that span to twice the
    local score-overlap width (σ_TP + σ_FP) gives

        k = 2 / (σ_TP + σ_FP)

    with the σs count-weighted (n_TP + n_FP) over the gate's operating band
    — the case-A bins where the gate actually has a crossing to soften.
    Recovers the hand-set default k = 10 at a typical overlap of 0.2.
    """
    num = den = 0.0
    for b in per_bin:
        if b.get("case") == "A":
            w = b.get("n_tp", 0.0) + b.get("n_fp", 0.0)
            num += w * (b.get("tp_sigma", 0.0) + b.get("fp_sigma", 0.0))
            den += w
    if den <= 0:   # no case-A bins (no gate anywhere) → all bins
        for b in per_bin:
            w = b.get("n_tp", 0.0) + b.get("n_fp", 0.0)
            num += w * (b.get("tp_sigma", 0.0) + b.get("fp_sigma", 0.0))
            den += w
    sig_sum = num / den if den > 0 else 0.2
    return {"k": width_mult / max(sig_sum, 1e-2), "sigma_sum": sig_sum}


# ---------------------------------------------------------------------------
# (G) association gate
# ---------------------------------------------------------------------------

def derive_mahal_gate(alpha: float = MAHAL_ALPHA, df: int = MAHAL_DF) -> float:
    """(G) Mahalanobis association gate — a pure χ² quantile.

    With GPEM-calibrated R the innovation d² = νᵀS⁻¹ν is χ²(df)-distributed
    for true matches, so the gate is the quantile chi2.ppf(alpha, df) with
    no data needed. Association innovations are 2-D (x, y) in
    SensorFusion's batch matcher ⇒ df = 2, for which the quantile is closed
    form: chi2.ppf(α, 2) = −2·ln(1 − α) → 9.2103 at α = 0.99.

    Convention note: SensorFusion compares its mahal DISTANCE against
    `mahal_gate` whose shipped default 13.82 equals chi2.ppf(0.999, 2) —
    the gate value is used in the codebase's χ²-value units, which we
    follow. IoU/mahal *weights* stay fixed (spec §2.2-G, v1 scope).
    """
    if df == 2:
        return -2.0 * math.log(1.0 - alpha)
    # General df via Wilson–Hilferty approximation (not used for df=2).
    from statistics import NormalDist
    z = NormalDist().inv_cdf(alpha)
    return df * (1.0 - 2.0 / (9.0 * df) + z * math.sqrt(2.0 / (9.0 * df))) ** 3


# ---------------------------------------------------------------------------
# (Rng) detector max range
# ---------------------------------------------------------------------------

def derive_max_range_from_rows(binned_rows: List[Dict[str, float]],
                               min_tp: float = RNG_MIN_TP) -> float:
    """(Rng) TP-bearing range span from cartesian binned_errors.csv rows.

    detector_max_range = the largest range-bin upper edge whose bin holds at
    least `min_tp` true positives (count − missed ≥ 5 — the same support
    filter the gate solver uses). Replaces the hand-set 100.0.
    """
    best = 0.0
    for r in binned_rows:
        if r.get("count", 0.0) - r.get("missed", 0.0) >= min_tp:
            best = max(best, r.get("bin_hi", 0.0))
    return best


def load_cartesian_bins(path: Path) -> List[Dict[str, float]]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({"bin_lo": _f(r, "bin_lo"), "bin_hi": _f(r, "bin_hi"),
                         "count": _f(r, "count"), "missed": _f(r, "missed")})
    return rows


# ---------------------------------------------------------------------------
# w-sweep plateau check (validation of (W), not a tuning step)
# ---------------------------------------------------------------------------

def check_w_plateau(detector: str, protocol: str, w_derived: float,
                    sweep_dir: Path = REPO / "results" / "birth_gate_calibration" / "w_sweep",
                    tol: float = 0.5) -> Dict:
    """Compare the derived w against the empirical w-sweep plateau.

    The plateau is the set of swept w whose AMOTA is within `tol` of the
    sweep best. Verdicts: "inside_plateau", "outside_plateau",
    "below_swept_range" / "above_swept_range" (with a trend note when the
    best sits on the corresponding sweep boundary), or "no_sweep_data".
    """
    path = sweep_dir / f"{detector}_w_sweep.csv"
    if not path.exists():
        return {"verdict": "no_sweep_data", "sweep_csv": None}
    col = "v2v_amota" if protocol == "v2v" else "ours_amota"
    pts = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                pts.append((float(r["w"]), float(r[col])))
            except (KeyError, ValueError):
                continue
    if not pts:
        return {"verdict": "no_sweep_data", "sweep_csv": str(path)}
    pts.sort()
    best_w, best_a = max(pts, key=lambda t: t[1])
    plateau = [w for w, a in pts if a >= best_a - tol]
    lo, hi = min(w for w, _ in pts), max(w for w, _ in pts)
    out = {"sweep_csv": str(path), "best_w": best_w, "best_amota": best_a,
           "plateau_w": plateau, "swept_range": [lo, hi],
           "w_derived": w_derived}
    if w_derived < lo:
        out["verdict"] = "below_swept_range" + (
            "_trend_consistent" if best_w == lo else "")
    elif w_derived > hi:
        out["verdict"] = "above_swept_range" + (
            "_trend_consistent" if best_w == hi else "")
    elif min(plateau) <= w_derived <= max(plateau):
        out["verdict"] = "inside_plateau"
    else:
        out["verdict"] = "outside_plateau"
    return out


# ---------------------------------------------------------------------------
# Full per-detector × per-protocol derivation
# ---------------------------------------------------------------------------

def derive_for_detector(detector: str,
                        sources: Optional[Dict[str, Path]] = None
                        ) -> Dict[str, Dict]:
    """Run the complete §2.1 + §2.2 derivation for one detector.

    Returns {protocol: config_dict}; config_dict is the instant_config.json
    payload (minus wall-clock, filled by the caller).
    """
    sources = sources or DETECTOR_SOURCES[detector]
    t0 = time.perf_counter()

    rows_by_source = {bn: load_bins(src / "polar_binned_errors.csv")
                      for bn, src in sources.items()}
    fleet_rows = [r for rows in rows_by_source.values() for r in rows]
    stats = aggregate_stats(fleet_rows)

    # (Rng) — protocol-independent
    max_range = 0.0
    for bn, src in sources.items():
        cart = src / "binned_errors.csv"
        if cart.exists():
            max_range = max(max_range,
                            derive_max_range_from_rows(load_cartesian_bins(cart)))

    # (W)
    w_ours = derive_w_ours(stats)

    def _static_gate_at(w: float) -> float:
        gates = []
        for bn, src in sources.items():
            out = sg.analyze_detector(bn, src, w)
            gates.append(out["scalar"] if out and out["scalar"] is not None else 0.0)
        return min(gates) if gates else 0.0

    w_v2v = derive_w_v2v(fleet_rows, stats, _static_gate_at)
    t_shared = time.perf_counter() - t0

    configs: Dict[str, Dict] = {}
    for protocol in PROTOCOLS:
        tp0 = time.perf_counter()
        wp = w_v2v["w"] if protocol == "v2v" else w_ours["w"]

        # §2.1 solver pass at the protocol weight (gate curves + lifecycle)
        per_source = {}
        for bn, src in sources.items():
            out = sg.analyze_detector(bn, src, wp)
            if out is None:
                raise RuntimeError(f"{detector}/{bn}: solver produced no bins")
            per_source[bn] = out

        scalars = [o["scalar"] for o in per_source.values()
                   if o["scalar"] is not None]
        fleet_scalar = min(scalars) if scalars else 0.0

        # (F)
        floor = derive_floor({bn: o["per_range"] for bn, o in per_source.items()},
                             fleet_rows)
        g_eff = max(fleet_scalar, floor["floor"])
        pg = fp_postgate_stats(fleet_rows, g_eff, stats["mu_tp"], stats["sig_tp"])

        # (L)
        conf = derive_confirm(stats, protocol, pg["clusters"], pg["d_sweep"])
        # p_birth follows the solver convention (gate scalar, 0.7 fallback)
        p_birth = fleet_scalar if fleet_scalar > 1e-6 else 0.7
        p_birth = min(max(p_birth, 1e-6), 1.0 - 1e-6)
        confirm_lo = confirm_log_odds_equivalent(conf["m_star"], p_birth,
                                                 stats["p_match"])

        # (C)
        coast = derive_coast(stats, protocol, pg["clusters"], pg["d_sweep"],
                             conf["m_star"])

        # (K)
        all_bins = [b for o in per_source.values() for b in o["per_bin"]]
        sigk = derive_sigmoid_k(all_bins)

        # (G)
        mahal_gate = derive_mahal_gate()

        # §2.1 fleet lifecycle (solver Tier 1/1.5 on the pooled rows)
        solver_lifecycle = sg.solve_lifecycle_params(
            fleet_rows, p_birth_gate=fleet_scalar if fleet_scalar > 1e-6 else None)

        # plateau validation for (W)
        plateau = check_w_plateau(detector, protocol, wp)

        config = {
            "detector": detector,
            "protocol": protocol,
            "formula_version": FORMULA_VERSION,
            "provenance": {
                "sources": {bn: str(src) for bn, src in sources.items()},
                "calibration_csvs": {
                    bn: str(src / "polar_binned_errors.csv")
                    for bn, src in sources.items()},
                "cartesian_csvs": {
                    bn: str(src / "binned_errors.csv")
                    for bn, src in sources.items()},
                "gpem_r_calibration": GPEM_CALIBRATION.get(detector, []),
                "solver": "scripts/solve_amota_bayes_gate.py (read-only import)",
                "generated_unix": time.time(),
            },
            # ── directly consumable LogOddsParams-shaped block ──────────────
            "tracker_params": {
                "detector_name": DETECTOR_CONFIG_NAMES[detector],
                "detector_max_range": max_range,
                "score_threshold": round(floor["floor"], 4),
                "lifecycle_mode": "log_odds",
                "p_tp_birth_gate": round(p_birth, 4),
                "confirm_log_odds": round(confirm_lo, 4),
                "kill_log_odds": solver_lifecycle["kill_log_odds"],
                "log_lr_miss": round(solver_lifecycle["log_lr_miss"], 4),
                "ab3dmot_min_hits": conf["m_star"],
                "ab3dmot_max_age": coast["K"],
                "tape_score_miss_decay": solver_lifecycle["tape_score_miss_decay"],
                "data_driven_gate_alpha": 99.0,
                "data_driven_gate_fit": "quadratic",
                "data_driven_gate_k": round(sigk["k"], 3),
                "mahal_gate": round(mahal_gate, 4),
                "use_gpem_r": True,
                "gpem_fit": "polar",
                "metric_weight_w": round(wp, 4),
            },
            # ── per-derivation detail (formula inputs, §2.2 letters) ────────
            "derived": {
                "W_protocol_weight": {
                    "value": round(wp, 4),
                    "formula": ("post-gate FP-mass-weighted GT bin co-occupancy "
                                "(association pollution, fixed point with gate)"
                                if protocol == "v2v" else
                                "(L_fp + T_coast)/L_fp coast-tail amplification"),
                    "inputs": (w_v2v if protocol == "v2v" else w_ours),
                    "plateau_check": plateau,
                },
                "F_score_floor": {"value": round(floor["floor"], 4), **floor},
                "L_confirm": {
                    "m_star": conf["m_star"],
                    "confirm_log_odds": round(confirm_lo, 4),
                    "p_birth_logit": round(math.log(p_birth / (1 - p_birth)), 4),
                    "step_cost": round(conf["step_cost"], 1),
                    "fp_price": round(conf["fp_price"], 1),
                    "d_sweep": round(conf["d_sweep"], 6),
                    "q_fp_survival": round(conf["q"], 4),
                    "j_table": {str(m): round(j, 1)
                                for m, j in conf["j_table"].items()},
                },
                "C_coast": {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in coast.items()},
                "K_sigmoid": {"value": round(sigk["k"], 3),
                              "sigma_sum": round(sigk["sigma_sum"], 4)},
                "G_mahal_gate": {"value": round(mahal_gate, 4),
                                 "alpha": MAHAL_ALPHA, "df": MAHAL_DF},
                "Rng_max_range": {"value": max_range, "min_tp": RNG_MIN_TP},
            },
            # ── §2.1 solver outputs (complete, per source) ──────────────────
            "birth_gate": {
                bn: {
                    "scalar": o["scalar"],
                    "per_range": o["per_range"],
                    "fits": o["fits"],
                    "n_bins": o["n_bins"],
                    "polar_csv": f"{bn}_polar.csv",   # written next to the JSON
                } for bn, o in per_source.items()
            },
            "solver_lifecycle_fleet": solver_lifecycle,
            "solver_lifecycle_per_source": {
                bn: o["lifecycle"] for bn, o in per_source.items()},
            "fleet_stats": {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in stats.items()},
            "_per_source_polar": {bn: o["per_bin"] for bn, o in per_source.items()},
            "derivation_wall_clock_seconds": round(
                t_shared + (time.perf_counter() - tp0), 4),
        }
        configs[protocol] = config
    return configs


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_config(config: Dict, out_root: Path) -> Path:
    """Write instant_config.json + per-source polar gate CSVs."""
    out_dir = out_root / config["detector"] / config["protocol"]
    out_dir.mkdir(parents=True, exist_ok=True)
    polar = config.pop("_per_source_polar")
    for bn, per_bin in polar.items():
        if not per_bin:
            continue
        with open(out_dir / f"{bn}_polar.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(per_bin[0].keys()))
            wr.writeheader()
            wr.writerows(per_bin)
    path = out_dir / "instant_config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return path


def _print_table(all_configs: Dict[str, Dict[str, Dict]]) -> None:
    hdr = (f"{'detector':<18s} {'proto':<5s} {'w':>7s} {'floor':>6s} "
           f"{'gate':>6s} {'mh':>3s} {'ma':>3s} {'k':>5s} {'mahal':>6s} "
           f"{'rng':>5s} {'conf_lo':>8s} {'p_birth':>7s} {'sec':>6s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for det, by_proto in all_configs.items():
        for proto, cfg in by_proto.items():
            tp = cfg["tracker_params"]
            scalars = [b["scalar"] for b in cfg["birth_gate"].values()
                       if b["scalar"] is not None]
            gate = min(scalars) if scalars else float("nan")
            print(f"{det:<18s} {proto:<5s} {tp['metric_weight_w']:>7.3f} "
                  f"{tp['score_threshold']:>6.3f} {gate:>6.3f} "
                  f"{tp['ab3dmot_min_hits']:>3d} {tp['ab3dmot_max_age']:>3d} "
                  f"{tp['data_driven_gate_k']:>5.1f} {tp['mahal_gate']:>6.2f} "
                  f"{tp['detector_max_range']:>5.0f} {tp['confirm_log_odds']:>8.3f} "
                  f"{tp['p_tp_birth_gate']:>7.3f} "
                  f"{cfg['derivation_wall_clock_seconds']:>6.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detectors", nargs="*", default=list(DETECTOR_SOURCES),
                    help="Subset of detectors (default: all three).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"Output root (default {DEFAULT_OUT}).")
    args = ap.parse_args()

    t0 = time.perf_counter()
    all_configs: Dict[str, Dict[str, Dict]] = {}
    for det in args.detectors:
        if det not in DETECTOR_SOURCES:
            raise SystemExit(f"unknown detector {det!r} "
                             f"(choose from {list(DETECTOR_SOURCES)})")
        print(f"\n=== {det} ===")
        all_configs[det] = derive_for_detector(det)
        for proto, cfg in all_configs[det].items():
            path = write_config(cfg, args.out)
            print(f"  wrote {path}  "
                  f"({cfg['derivation_wall_clock_seconds']:.3f}s derivation)")

    total = time.perf_counter() - t0
    _print_table(all_configs)
    print(f"\ntotal wall-clock (all detectors × protocols, incl. I/O): "
          f"{total:.2f}s")

    # (W) plateau verdicts
    print("\n(W) vs empirical w-sweep plateau:")
    for det, by_proto in all_configs.items():
        for proto, cfg in by_proto.items():
            chk = cfg["derived"]["W_protocol_weight"]["plateau_check"]
            print(f"  {det:<18s} {proto:<5s} w={cfg['tracker_params']['metric_weight_w']:>7.3f} "
                  f"→ {chk['verdict']}"
                  + (f" (best w={chk.get('best_w')}, plateau={chk.get('plateau_w')})"
                     if chk.get("best_w") is not None else ""))


if __name__ == "__main__":
    main()
