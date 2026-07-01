#!/usr/bin/env python3
"""AMOTA-aware closed-form birth-gate solver.

Reads the new 69-column `polar_binned_errors.csv` from each detector's
characterization run (under `work_dirs/<run>/error_analysis*/`), solves the
closed-form quadratic per bin to find the AMOTA-optimal score threshold s*,
then fits four forms — static / linear / quadratic / polar — per detector
in pure GPEM style.

Per-bin solve
=============
For each bin with TP/FP score Gaussians and counts, the AMOTA-cost-weighted
optimum threshold satisfies:

    n_TP · N(s | μ_TP, σ_TP) / σ_TP  =  λ · n_FP · N(s | μ_FP, σ_FP) / σ_FP

where  λ_bin = fp_lifetime_mean / tp_lifetime_mean  is the cost ratio
(frames added per accepted FP / frames lost per rejected TP), derived
directly from the per-bin trajectory statistics.

Taking logs gives a quadratic  a·s² + b·s + c = 0  in s, with closed-form
roots. Pick the larger root in [μ_FP, μ_TP] (the meaningful crossing on
the upper side of the score distributions).

Fits
====
- static:    count-weighted median of all per-bin s*
- linear:    s(r) = a·r + b  (count-weighted least squares vs range_mid)
- quadratic: s(r) = a·r² + b·r + c
- polar:     per-(range, angle) bin lookup (no smoothing — same as GPEM polar)

Output: results/birth_gate_calibration/amota_bayes/<detector>_*.csv
        results/birth_gate_calibration/amota_bayes/recommended_gates.csv
        results/birth_gate_calibration/amota_bayes/birth_gate_curves.csv

Usage:
    python scripts/solve_amota_bayes_gate.py [--w 1.0]
        # Optional metric-preference weight w (λ_effective = w · λ_data).
        # Default 1.0 (data-derived MOTA-symmetric).
"""
import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
WORK_DIRS = (REPO / "data" / "v2v4real_inputs" / "ours")
# Sibling mmdetection3d clone — holds the freshly-refit aligned
# error_analysis dirs for the V2V4Real detectors (PP per-vehicle + CPZS).
# These are the CONFIG-MATCHING detector names the production log_odds
# tracker looks up (pointpillar_v2v4real_{vehicle}, centerpoint_100m_on_v2v4real).
MMDET3D = REPO.parent / "mmdetection3d" / "work_dirs"
# OUT_DIR override via env var SOLVER_OUT_DIR — lets us run alternate
# aggregations (e.g. P50) to a separate location without overwriting
# the production P25 outputs.
_OUT_BASE = Path(os.environ.get(
    "SOLVER_OUT_DIR",
    str(REPO / "results" / "birth_gate_calibration")))
OUT_DIR = _OUT_BASE / "amota_bayes"

# ---------------------------------------------------------------------------
# Detector → (calibration source dir, canonical name) mapping.
# Mirrors what the user reported from the 9 detector × split combos.
# ---------------------------------------------------------------------------

DETECTORS: Dict[str, Path] = {
    # nuScenes trainval
    "centerpoint":  WORK_DIRS / "centerpoint_100m_full"  / "error_analysis_trainval",
    "bev_fusion":   WORK_DIRS / "bevfusion_100m_full"    / "error_analysis_trainval",
    "detr3d":       WORK_DIRS / "detr3d_100m_full"       / "error_analysis_trainval",
    # V2V4Real fine-tuned CP, per vehicle
    "centerpoint_54m_v2v4real_finetune_tesla":  WORK_DIRS / "centerpoint_finetune_v2v4real_54m_tesla"  / "error_analysis_trainval",
    "centerpoint_54m_v2v4real_finetune_astuff": WORK_DIRS / "centerpoint_finetune_v2v4real_54m_astuff" / "error_analysis_trainval",
    # Tracker outputs (DMSTrack + CoBEVT) per vehicle
    "dmstrack_tesla":       WORK_DIRS / "dmstrack_train" / "error_analysis_tesla",
    "dmstrack_astuff":      WORK_DIRS / "dmstrack_train" / "error_analysis_astuff",
    "cobevt_tracker_tesla": WORK_DIRS / "cobevt_train"   / "error_analysis_tesla",
    "cobevt_tracker_astuff":WORK_DIRS / "cobevt_train"   / "error_analysis_astuff",
    # V2V4Real PointPillars per vehicle (score≥0 characterization with Block A-G)
    "pointpillar_tesla":    WORK_DIRS / "pointpillar_v2v4real" / "error_analysis_tesla_score00",
    "pointpillar_astuff":   WORK_DIRS / "pointpillar_v2v4real" / "error_analysis_astuff_score00",
    # ── Config-matching canonical names ──────────────────────────────────────
    # The production log_odds tracker looks up its gate + lifecycle by these
    # exact detector_name keys (see configs/v2v4real_experiments.py and
    # src/lifecycle_params.py / src/birth_gate_curves.py). They point at the
    # freshly-refit ALIGNED characterization under the sibling mmdetection3d
    # clone (757-row polar_binned_errors.csv each). The legacy
    # pointpillar_{vehicle} / centerpoint entries above stay intact: they still
    # back the DETECTOR_ALIASES rows used by the mixed-detector / CPft configs.
    "pointpillar_v2v4real_astuff":   MMDET3D / "reexport_pp_astuff"     / "error_analysis",
    "pointpillar_v2v4real_tesla":    MMDET3D / "reexport_pp_tesla"      / "error_analysis",
    "centerpoint_100m_on_v2v4real":  MMDET3D / "preds_cpzs_100m_tesla"  / "error_analysis",
}

# Bin-quality filters (skip bins that don't have enough signal)
MIN_N_TP = 5
MIN_N_FP = 5
MIN_TP_LIFETIME = 1.0   # frames
MIN_FP_LIFETIME = 1.0
MIN_SIGMA = 1e-3
LAMBDA_FLOOR = 1e-4
LAMBDA_CEIL  = 1e3


# ---------------------------------------------------------------------------
# Closed-form solver
# ---------------------------------------------------------------------------

def solve_amota_bayes_gate(n_tp: float, mu_tp: float, sig_tp: float,
                            n_fp: float, mu_fp: float, sig_fp: float,
                            lam: float) -> Optional[float]:
    """Return the AMOTA-cost-weighted optimal score threshold s* for one bin.

    Solves:  n_TP · N(s | μ_TP, σ_TP) / σ_TP = λ · n_FP · N(s | μ_FP, σ_FP) / σ_FP

    Quadratic in s:
        a·s² + b·s + c = 0
        a = 0.5 · (1/σ_FP² − 1/σ_TP²)
        b = μ_TP/σ_TP² − μ_FP/σ_FP²
        c = 0.5 · (μ_FP²/σ_FP² − μ_TP²/σ_TP²)
            + log(λ · n_FP / σ_FP) − log(n_TP / σ_TP)

    Returns the larger root in [0, 1] (clipped). Falls back to midpoint
    on degenerate cases. None if inputs are degenerate enough that even
    the midpoint isn't meaningful.
    """
    if n_tp <= 0 or n_fp <= 0 or lam <= 0:
        return None

    sig_tp = max(sig_tp, MIN_SIGMA)
    sig_fp = max(sig_fp, MIN_SIGMA)

    # ── Principled algorithm ──
    # Optimum at:  n_TP · N(s | μ_TP, σ_TP) = λ · n_FP · N(s | μ_FP, σ_FP)
    #
    # Three cases (NO heuristic blends):
    #   A. Quadratic has a real root in [μ_FP, μ_TP]    → that root is s*
    #   B. No root in [μ_FP, μ_TP] AND TP density >
    #         λ·FP density at the overlap midpoint       → s* = 0   (no gate)
    #   C. No root in [μ_FP, μ_TP] AND TP density ≤
    #         λ·FP density at the overlap midpoint       → s* = 1   (gate all)

    def _log_tp_density(s: float) -> float:
        return (math.log(n_tp) - math.log(sig_tp)
                - 0.5 * ((s - mu_tp) / sig_tp) ** 2)

    def _log_lam_fp_density(s: float) -> float:
        return (math.log(lam * n_fp) - math.log(sig_fp)
                - 0.5 * ((s - mu_fp) / sig_fp) ** 2)

    def _dominance_answer() -> float:
        """Return s* by dominance test at the overlap midpoint."""
        midpoint = 0.5 * (mu_tp + mu_fp)
        if _log_tp_density(midpoint) >= _log_lam_fp_density(midpoint):
            return 0.0   # Case B: TP dominates → no gate
        return 1.0       # Case C: FP dominates (cost-weighted) → gate all

    # Quadratic coefficients: a·s² + b·s + c = 0
    a = 0.5 * (1.0 / sig_fp ** 2 - 1.0 / sig_tp ** 2)
    b = mu_tp / sig_tp ** 2 - mu_fp / sig_fp ** 2
    c = (0.5 * (mu_fp ** 2 / sig_fp ** 2 - mu_tp ** 2 / sig_tp ** 2)
         + math.log(lam * n_fp / sig_fp)
         - math.log(n_tp / sig_tp))

    # σ_TP == σ_FP → quadratic degenerates to linear b·s + c = 0
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return _dominance_answer()
        s = -c / b
        # Same overlap-region check as the quadratic case below
        lo, hi = (mu_fp, mu_tp) if mu_fp < mu_tp else (mu_tp, mu_fp)
        if lo <= s <= hi:
            return max(0.0, min(1.0, s))
        return _dominance_answer()

    disc = b * b - 4 * a * c
    if disc < 0:
        return _dominance_answer()                     # no real roots
    sqd = math.sqrt(disc)
    roots = [(-b - sqd) / (2 * a), (-b + sqd) / (2 * a)]

    # Case A: pick the root inside [μ_FP, μ_TP] (the meaningful crossing)
    lo, hi = (mu_fp, mu_tp) if mu_fp < mu_tp else (mu_tp, mu_fp)
    valid = [s for s in roots if lo <= s <= hi]
    if valid:
        return max(0.0, min(1.0, max(valid)))

    # Cases B/C: roots exist but outside the overlap region
    return _dominance_answer()


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    v = (row.get(key, "") or "").strip()
    if v == "" or v.lower() == "nan":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def load_polar_binned_errors(path: Path) -> List[Dict]:
    """Read polar_binned_errors.csv, return per-bin dicts with the fields we need."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "range_lo":            _f(r, "range_lo"),
                "range_hi":            _f(r, "range_hi"),
                "angle_lo":            _f(r, "angle_lo"),
                "angle_hi":            _f(r, "angle_hi"),
                "gt_count":            _f(r, "gt_count"),
                "missed":              _f(r, "missed"),
                "n_fp":                _f(r, "n_fp"),
                "score_mean":          _f(r, "score_mean"),
                "score_std":           _f(r, "score_std"),
                "fp_score_mean":       _f(r, "fp_score_mean"),
                "fp_score_std":        _f(r, "fp_score_std"),
                "tp_lifetime_mean":    _f(r, "tp_lifetime_mean"),
                "fp_lifetime_mean":    _f(r, "fp_lifetime_mean"),
                "tp_lifetime_std":     _f(r, "tp_lifetime_std"),
                "fp_lifetime_std":     _f(r, "fp_lifetime_std"),
                "n_gt_tracks":         _f(r, "n_gt_tracks"),
                "n_fp_clusters":       _f(r, "n_fp_clusters"),
                # Lifecycle-autotune fields (Block D/E):
                # tp_p_match_per_track_mean = fraction of within-lifetime
                # frames where the TP track has a detection (= 1 - p_miss_TP).
                # fp_p_match_per_cluster_mean ≡ 1.0 by definition (FP clusters
                # are continuous streaks) — load anyway for completeness.
                "tp_p_match":          _f(r, "tp_p_match_per_track_mean"),
                "fp_p_match":          _f(r, "fp_p_match_per_cluster_mean"),
                "tp_fragments":        _f(r, "tp_fragments_per_track_mean"),
                "fp_fragments":        _f(r, "fp_fragments_per_cluster_mean"),
            })
    return rows


def _pooled_mean_std(triples: List[Tuple[float, float, float]]) -> Tuple[float, float]:
    """Pool (mean, std, count) tuples into a single (μ, σ) using the
    second-moment identity:
        σ²_pool = (Σ nᵢ·(σᵢ² + μᵢ²)) / Σnᵢ  −  μ²_pool
    Returns (0, 0) if total weight is zero.
    """
    n_tot = sum(n for _, _, n in triples)
    if n_tot <= 0:
        return 0.0, 0.0
    mu_pool = sum(mu * n for mu, _, n in triples) / n_tot
    sec_mom = sum(n * (sig * sig + mu * mu) for mu, sig, n in triples) / n_tot
    var = max(0.0, sec_mom - mu_pool * mu_pool)
    return mu_pool, math.sqrt(var)


def merge_bins(rows: List[Dict], range_step: float, angle_step: float) -> List[Dict]:
    """Coarsen 5m×10° bins into range_step × angle_step bins.

    Aggregation rules per merged group:
      - counts (gt_count, missed, n_fp, n_gt_tracks, n_fp_clusters): sum
      - score_mean / score_std       : pooled by n_tp = (gt_count − missed)
      - fp_score_mean / fp_score_std : pooled by n_fp
      - tp_lifetime_mean / tp_lifetime_std : pooled by n_gt_tracks (or n_tp if 0)
      - fp_lifetime_mean / fp_lifetime_std : pooled by n_fp_clusters (or n_fp if 0)

    Note: n_gt_tracks / n_fp_clusters become slight overcounts if the same
    trajectory crosses a sub-bin boundary inside a merged cell. Acceptable
    for our weighting/regression — the trajectory itself is fully captured.
    """
    groups: Dict[Tuple[float, float], List[Dict]] = defaultdict(list)
    for r in rows:
        rk = math.floor(r["range_lo"] / range_step) * range_step
        ak = math.floor(r["angle_lo"] / angle_step) * angle_step
        groups[(rk, ak)].append(r)

    out: List[Dict] = []
    for (rk, ak), group in sorted(groups.items()):
        gt_count = sum(g["gt_count"] for g in group)
        missed = sum(g["missed"] for g in group)
        n_fp = sum(g["n_fp"] for g in group)
        n_gt_tracks = sum(g["n_gt_tracks"] for g in group)
        n_fp_clusters = sum(g["n_fp_clusters"] for g in group)

        # TP score Gaussian: pool by per-bin n_tp = (gt_count − missed)
        tp_triples = [(g["score_mean"], g["score_std"],
                       max(0.0, g["gt_count"] - g["missed"])) for g in group]
        tp_mu, tp_sig = _pooled_mean_std(tp_triples)

        fp_triples = [(g["fp_score_mean"], g["fp_score_std"], g["n_fp"]) for g in group]
        fp_mu, fp_sig = _pooled_mean_std(fp_triples)

        tp_life_triples = [(g["tp_lifetime_mean"], g["tp_lifetime_std"],
                            g["n_gt_tracks"] if g["n_gt_tracks"] > 0
                            else max(0.0, g["gt_count"] - g["missed"]))
                           for g in group]
        tp_life_mu, tp_life_sig = _pooled_mean_std(tp_life_triples)

        fp_life_triples = [(g["fp_lifetime_mean"], g["fp_lifetime_std"],
                            g["n_fp_clusters"] if g["n_fp_clusters"] > 0 else g["n_fp"])
                           for g in group]
        fp_life_mu, fp_life_sig = _pooled_mean_std(fp_life_triples)

        out.append({
            "range_lo":         rk,
            "range_hi":         rk + range_step,
            "angle_lo":         ak,
            "angle_hi":         ak + angle_step,
            "gt_count":         gt_count,
            "missed":           missed,
            "n_fp":             n_fp,
            "score_mean":       tp_mu,
            "score_std":        tp_sig,
            "fp_score_mean":    fp_mu,
            "fp_score_std":     fp_sig,
            "tp_lifetime_mean": tp_life_mu,
            "fp_lifetime_mean": fp_life_mu,
            "tp_lifetime_std":  tp_life_sig,
            "fp_lifetime_std":  fp_life_sig,
            "n_gt_tracks":      n_gt_tracks,
            "n_fp_clusters":    n_fp_clusters,
        })
    return out


# ---------------------------------------------------------------------------
# Adaptive ring-expansion (hole-fill) pooling
#
# Mirrors merge_bins for the actual pooling math (reuses _pooled_mean_std),
# but expands neighborhood progressively per-cell only when needed. Each
# native 5m × 10° cell tries native first; if the solver returns case-A AND
# the cell has ≥ native_min_tp/fp samples, the native answer is kept. Else
# we expand to Chebyshev ring 1 (3×3 = 9 cells), ring 2 (5×5 = 25), ring 3
# (7×7 = 49, max) and re-solve. Stop the moment we get case-A; if max-ring
# still gives case-B/C, trust it as real signal (not sparsity).
# ---------------------------------------------------------------------------

def _neighbors_at_ring(ri: int, ai: int, ring: int, n_angle_bins: int):
    """Yield (ri', ai') for cells at exact Chebyshev distance `ring`.

    Range index `ri` may go out of bounds (caller filters by lookup miss);
    angle wraps modulo `n_angle_bins` (−180° ≡ 180°). Ring 0 yields just
    the anchor.
    """
    if ring == 0:
        yield (ri, ai % n_angle_bins)
        return
    for dr in range(-ring, ring + 1):
        for da in range(-ring, ring + 1):
            if max(abs(dr), abs(da)) != ring:
                continue
            yield (ri + dr, (ai + da) % n_angle_bins)


def _pool_rows_into_stats(rows: List[Dict]) -> Optional[Dict]:
    """Pool a list of native bin rows into a single stats dict.

    Same aggregation rules as merge_bins (counts summed; means/stds pooled
    via _pooled_mean_std with appropriate weights). Returns None if rows
    list is empty.
    """
    if not rows:
        return None
    gt_count = sum(r["gt_count"] for r in rows)
    missed = sum(r["missed"] for r in rows)
    n_fp = sum(r["n_fp"] for r in rows)
    n_gt_tracks = sum(r["n_gt_tracks"] for r in rows)
    n_fp_clusters = sum(r["n_fp_clusters"] for r in rows)

    tp_triples = [(r["score_mean"], r["score_std"],
                   max(0.0, r["gt_count"] - r["missed"])) for r in rows]
    tp_mu, tp_sig = _pooled_mean_std(tp_triples)

    fp_triples = [(r["fp_score_mean"], r["fp_score_std"], r["n_fp"]) for r in rows]
    fp_mu, fp_sig = _pooled_mean_std(fp_triples)

    tp_life_triples = [(r["tp_lifetime_mean"], r["tp_lifetime_std"],
                        r["n_gt_tracks"] if r["n_gt_tracks"] > 0
                        else max(0.0, r["gt_count"] - r["missed"]))
                       for r in rows]
    tp_life_mu, tp_life_sig = _pooled_mean_std(tp_life_triples)

    fp_life_triples = [(r["fp_lifetime_mean"], r["fp_lifetime_std"],
                        r["n_fp_clusters"] if r["n_fp_clusters"] > 0 else r["n_fp"])
                       for r in rows]
    fp_life_mu, fp_life_sig = _pooled_mean_std(fp_life_triples)

    return {
        "gt_count":         gt_count,
        "missed":           missed,
        "n_fp":             n_fp,
        "score_mean":       tp_mu,
        "score_std":        tp_sig,
        "fp_score_mean":    fp_mu,
        "fp_score_std":     fp_sig,
        "tp_lifetime_mean": tp_life_mu,
        "fp_lifetime_mean": fp_life_mu,
        "tp_lifetime_std":  tp_life_sig,
        "fp_lifetime_std":  fp_life_sig,
        "n_gt_tracks":      n_gt_tracks,
        "n_fp_clusters":    n_fp_clusters,
    }


def _solve_with_stats(stats: Dict, lifecycle_cap: bool,
                       l_fp_lifecycle: float, w: float
                       ) -> Optional[Tuple[float, str, float, float, float]]:
    """Run the AMOTA-Bayes solver on a pooled stats dict.

    Returns (s_star, case, lam_data, lam_eff, L_fp_used) or None if the
    pooled stats fail the basic sample/lifetime filters (same MIN_N_TP /
    MIN_N_FP / MIN_TP_LIFETIME / MIN_FP_LIFETIME as the legacy loop).
    """
    n_tp = max(0.0, stats["gt_count"] - stats["missed"])
    n_fp = stats["n_fp"]
    if n_tp < MIN_N_TP or n_fp < MIN_N_FP:
        return None
    L_tp = stats["tp_lifetime_mean"]
    L_fp_raw = stats["fp_lifetime_mean"]
    if L_tp < MIN_TP_LIFETIME or L_fp_raw < MIN_FP_LIFETIME:
        return None

    L_fp = min(L_fp_raw, l_fp_lifecycle) if lifecycle_cap else L_fp_raw
    lam_data = L_fp / L_tp
    lam = max(LAMBDA_FLOOR, min(LAMBDA_CEIL, w * lam_data))

    s_star = solve_amota_bayes_gate(
        n_tp, stats["score_mean"], stats["score_std"],
        n_fp, stats["fp_score_mean"], stats["fp_score_std"],
        lam,
    )
    if s_star is None:
        return None
    case = "A" if 1e-6 < s_star < 1.0 - 1e-6 else ("B" if s_star <= 1e-6 else "C")
    return s_star, case, lam_data, lam, L_fp


def adaptive_pool_solve(rows: List[Dict],
                         range_step: float,
                         angle_step: float,
                         native_min_tp: float,
                         native_min_fp: float,
                         max_ring: int,
                         lifecycle_cap: bool,
                         l_fp_lifecycle: float,
                         w: float) -> List[Dict]:
    """Per-native-cell adaptive ring expansion + solve.

    For each native cell at (r_lo, a_lo):
      1. Try native (ring 0). Keep if case-A AND native passes threshold.
      2. Else try ring 1, ring 2, ... up to max_ring. Stop at first case-A.
      3. If max_ring still case-B/C, keep that as real signal.

    Returns per-native-cell records (one per native cell that successfully
    solves at some ring), with fields matching the legacy per_bin schema
    PLUS native_n_tp, native_n_fp, ring_used.
    """
    # Index native cells by (range_idx, angle_idx). Angle index is relative
    # to −180° so angle_lo=−180 ↔ ai=0, angle_lo=170 ↔ ai=35.
    by_idx: Dict[Tuple[int, int], Dict] = {}
    for r in rows:
        ri = int(round(r["range_lo"] / range_step))
        ai = int(round((r["angle_lo"] + 180.0) / angle_step))
        by_idx[(ri, ai)] = r
    n_angle_bins = max(1, int(round(360.0 / angle_step)))

    out: List[Dict] = []
    for (ri, ai), anchor in by_idx.items():
        native_n_tp = max(0.0, anchor["gt_count"] - anchor["missed"])
        native_n_fp = anchor["n_fp"]
        native_thresh_met = (native_n_tp >= native_min_tp and
                             native_n_fp >= native_min_fp)

        chosen_stats: Optional[Dict] = None
        chosen_solver: Optional[Tuple[float, str, float, float, float]] = None
        chosen_ring: int = -1

        # Try rings 0..max_ring; stop early on case-A. Keep deepest result
        # as fallback if no case-A is found.
        for ring in range(0, max_ring + 1):
            # Cumulative pool: all cells at Chebyshev distance ≤ ring.
            acc: List[Dict] = []
            for r_check in range(0, ring + 1):
                for (nri, nai) in _neighbors_at_ring(ri, ai, r_check, n_angle_bins):
                    cell = by_idx.get((nri, nai))
                    if cell is not None:
                        acc.append(cell)

            stats = _pool_rows_into_stats(acc)
            if stats is None:
                continue
            solver_out = _solve_with_stats(stats, lifecycle_cap, l_fp_lifecycle, w)
            if solver_out is None:
                continue

            s_star, case, lam_data, lam_eff, L_fp_used = solver_out

            if ring == 0:
                # Native wins only if case-A AND native thresholds met.
                if case == "A" and native_thresh_met:
                    chosen_stats, chosen_solver, chosen_ring = stats, solver_out, 0
                    break
                # Otherwise record as fallback and try ring 1.
                chosen_stats, chosen_solver, chosen_ring = stats, solver_out, 0
                continue

            # Ring ≥ 1: greedy — keep first case-A.
            chosen_stats, chosen_solver, chosen_ring = stats, solver_out, ring
            if case == "A":
                break

        if chosen_stats is None or chosen_solver is None:
            continue

        s_star, case, lam_data, lam_eff, L_fp_used = chosen_solver
        pooled_n_tp = max(0.0, chosen_stats["gt_count"] - chosen_stats["missed"])
        pooled_n_fp = chosen_stats["n_fp"]

        out.append({
            "range_lo":     anchor["range_lo"],
            "range_hi":     anchor["range_hi"],
            "angle_lo":     anchor["angle_lo"],
            "angle_hi":     anchor["angle_hi"],
            "n_tp":         pooled_n_tp,
            "n_fp":         pooled_n_fp,
            "native_n_tp":  native_n_tp,
            "native_n_fp":  native_n_fp,
            "ring_used":    chosen_ring,
            "tp_mu":        round(chosen_stats["score_mean"], 4),
            "tp_sigma":     round(chosen_stats["score_std"],  4),
            "fp_mu":        round(chosen_stats["fp_score_mean"], 4),
            "fp_sigma":     round(chosen_stats["fp_score_std"],  4),
            "L_tp_mean":    round(chosen_stats["tp_lifetime_mean"], 3),
            "L_fp_mean":    round(L_fp_used, 3),
            "lambda_data":  round(lam_data, 4),
            "lambda_eff":   round(lam_eff, 4),
            "s_star":       round(s_star, 4),
            # Regression weight = 1.0 per native cell ("each native cell
            # counted once"). Sample-count weighting drags the median to
            # case-B for detectors like cobevt where high-FP regions yield
            # case-B cells with disproportionate n_fp mass; uniform-per-cell
            # gives each gate-curve data point equal vote.
            "weight":       1.0,
            "case":         case,
        })
    return out


# ---------------------------------------------------------------------------
# Per-detector aggregation: per-bin solve + four fits
# ---------------------------------------------------------------------------

def _wpercentile(items: List[Tuple[float, float]], q: float = 0.5) -> Optional[float]:
    """Weighted q-th percentile (0 ≤ q ≤ 1). q=0.5 = weighted median."""
    items = sorted(items)
    tot = sum(w for _, w in items)
    if tot <= 0:
        if not items: return None
        # uniform-weight fallback at the same percentile.
        vals = [s for s, _ in items]
        idx = min(len(vals) - 1, max(0, int(q * len(vals))))
        return vals[idx]
    target = tot * q
    cum = 0.0
    for s, w in items:
        cum += w
        if cum >= target:
            return s
    return items[-1][0]


def _wmedian(items: List[Tuple[float, float]]) -> Optional[float]:
    """Backwards-compat alias — weighted median (P50)."""
    return _wpercentile(items, q=0.5)


def _smooth_polar_bins_inplace(per_bin: List[Dict],
                                 range_step: float = 5.0,
                                 angle_step: float = 10.0,
                                 neighborhood_ring: int = 1) -> None:
    """Polar smoothing in two passes:

    Pass 1 — local-P25 smoothing for case-A and case-B cells:
        Replace each cell's s_star with the weighted P25 of its Chebyshev
        neighborhood (default 3×3 = ring 1). Eliminates the piecewise step
        discontinuities while applying the same recall-asymmetry correction
        we use globally — but locally, respecting per-(range, angle)
        structure.

    Pass 2 — case-C override to global P25:
        For cells that were originally case-C ("FP dominates, reject all"
        per the per-bin Bayes solve), the local-P25 won't help — their
        neighborhoods are typically also case-C. Per the AMOTA-vs-Bayes
        analysis in docs/P25_GATE_AGGREGATION.md, case-C bins should be
        treated as "use the global recall-biased default," not as "reject
        everything." So we replace their s_star with the global P25 of
        case-A+B bins (= the static fit value).

    Original s_star kept in `s_star_raw` for audit. The `case` field is
    updated to reflect the smoothed surface.
    """
    # Pre-compute global P25 from the original case-A+B bins (before smoothing
    # changes anything).
    ab_pairs = [(float(b["s_star"]), float(b.get("weight", 1.0)))
                for b in per_bin if b["case"] in ("A", "B")]
    global_p25 = _wpercentile(ab_pairs, q=GATE_PERCENTILE) if ab_pairs else 0.0
    if global_p25 is None:
        global_p25 = 0.0

    # Index bins by integer (range, angle) grid coordinates.
    by_idx: Dict[Tuple[int, int], Dict] = {}
    for b in per_bin:
        ri = int(round(b["range_lo"] / range_step))
        ai = int(round((b["angle_lo"] + 180.0) / angle_step))
        by_idx[(ri, ai)] = b
    n_angle_bins = max(1, int(round(360.0 / angle_step)))

    # Decide the new s_star per cell in a single sweep (read original values,
    # write the updated ones afterwards to avoid feedback).
    new_values: List[Tuple[Dict, float]] = []
    for b in per_bin:
        if b["case"] == "C":
            # Case-C override: replace with global P25 (= the static fit value).
            new_values.append((b, float(global_p25)))
            continue

        # Local-P25 smoothing for case-A / case-B cells.
        ri = int(round(b["range_lo"] / range_step))
        ai = int(round((b["angle_lo"] + 180.0) / angle_step))
        nbrs: List[Tuple[float, float]] = []
        for dr in range(-neighborhood_ring, neighborhood_ring + 1):
            for da in range(-neighborhood_ring, neighborhood_ring + 1):
                nri = ri + dr
                nai = (ai + da) % n_angle_bins
                cell = by_idx.get((nri, nai))
                if cell is not None:
                    nbrs.append((float(cell["s_star"]),
                                 float(cell.get("weight", 1.0))))
        if not nbrs:
            new_values.append((b, float(b["s_star"])))
            continue
        s_local_p25 = _wpercentile(nbrs, q=GATE_PERCENTILE)
        new_values.append((b,
                           s_local_p25 if s_local_p25 is not None
                           else float(b["s_star"])))

    # Apply: keep raw value for audit, update s_star and re-label case.
    for b, s_new in new_values:
        b["s_star_raw"] = b["s_star"]
        b["s_star"] = round(s_new, 4)
        if 1e-6 < s_new < 1.0 - 1e-6:
            b["case"] = "A"
        elif s_new <= 1e-6:
            b["case"] = "B"
        else:
            b["case"] = "C"


# Aggregation percentile across case-A+B bins. Median (P50) lands at the
# Bayes-optimal per-bin classification threshold averaged across bins —
# which is biased high relative to the AMOTA-optimal gate. P25 of the
# case-A+B s* distribution empirically matches expert-tuned gates: for
# CP-zs P25=0.295 ≈ master's hand-tune 0.30, while case-B-dominant
# detectors stay at 0 (the 25th percentile of "0+0+0+...+s_A" is still 0).
# This expresses the asymmetric AMOTA cost: false negatives integrate over
# the recall sweep, while false positives only count above each threshold.
GATE_PERCENTILE = float(os.environ.get("GATE_PERCENTILE", "0.25"))


def _solve_2x2(a, b, c, d, e, f):
    det = a * d - b * c
    if abs(det) < 1e-12:
        return 0.0, e / a if abs(a) > 1e-12 else 0.0
    return (e * d - b * f) / det, (a * f - e * c) / det


def _solve_3x3(M, v):
    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
              - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
              + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    D = det3(M)
    if abs(D) < 1e-12:
        return [0.0, 0.0, sum(v) / 3.0]
    out = []
    for i in range(3):
        Mi = [row[:] for row in M]
        for r in range(3):
            Mi[r][i] = v[r]
        out.append(det3(Mi) / D)
    return out


def fit_linear(pts: List[Tuple[float, float, float]]) -> Tuple[float, float]:
    """Weighted LS y = a·x + b. pts = [(x, y, w), ...]."""
    if len(pts) < 2:
        return 0.0, (pts[0][1] if pts else 0.0)
    sw = sum(w for _, _, w in pts)
    sx = sum(x * w for x, _, w in pts)
    sy = sum(y * w for _, y, w in pts)
    sxx = sum(x * x * w for x, _, w in pts)
    sxy = sum(x * y * w for x, y, w in pts)
    a, b = _solve_2x2(sxx, sx, sx, sw, sxy, sy)
    return a, b


def fit_quadratic(pts: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Weighted LS y = a·x² + b·x + c. pts = [(x, y, w), ...]."""
    if len(pts) < 3:
        a, b = fit_linear(pts)
        return 0.0, a, b
    sw = sum(w for _, _, w in pts)
    sx = sum(x * w for x, _, w in pts)
    sx2 = sum(x * x * w for x, _, w in pts)
    sx3 = sum(x ** 3 * w for x, _, w in pts)
    sx4 = sum(x ** 4 * w for x, _, w in pts)
    sy = sum(y * w for _, y, w in pts)
    sxy = sum(x * y * w for x, y, w in pts)
    sx2y = sum(x * x * y * w for x, y, w in pts)
    M = [[sx4, sx3, sx2], [sx3, sx2, sx], [sx2, sx, sw]]
    v = [sx2y, sxy, sy]
    return tuple(_solve_3x3(M, v))


def _lifecycle_l_fp_cap(p_birth: float, kill_log_odds: float,
                         log_lr_miss: float) -> float:
    """Analytic upper bound on FP track lifetime given the lifecycle config.

    A track born with `p_birth` log-odds will be killed by the lifecycle once
    its log-odds decays below `kill_log_odds`, with `|log_lr_miss|` decay per
    miss frame:

        L_FP_lifecycle = (logit(p_birth) − kill_log_odds) / |log_lr_miss|

    For our defaults (p_birth=0.7, kill=−2.5, miss=−0.7), L_FP_lifecycle ≈ 4.8
    frames. This caps the cost of accepting an FP regardless of how persistent
    the noise process is in raw data — because the lifecycle WILL kill it
    within this many frames.
    """
    p = max(1e-6, min(1.0 - 1e-6, p_birth))
    logit = math.log(p / (1.0 - p))
    decay = max(1e-6, abs(log_lr_miss))
    return max(1.0, (logit - kill_log_odds) / decay)


# ---------------------------------------------------------------------------
# Tier 1 lifecycle autotuning — per-detector log_lr_miss derivation.
#
# Math (Bayesian log-odds tracker):
#   On each miss the track's S_t += log_lr_miss, where
#     log_lr_miss = log[ P(miss | TP) / P(miss | FP_alive) ]
#
# From characterization data:
#   p_miss_TP = 1 - tp_p_match_per_track_mean
#   p_miss_FP_alive ≈ FP_MISS_PRIOR (= 0.7; FPs that the tracker is keeping
#     alive miss more often than TPs because they don't track a real object).
#     This is a prior, not derived from data because fp_p_match_per_cluster
#     is structurally 1.0 (FP clusters are defined as continuous streaks —
#     a "miss" ends the cluster in the data, so there's no in-cluster miss
#     rate to measure).
#
# kill_log_odds stays at the canonical -3.0 (= log(1/20) — P_real < 0.048
# threshold). p_birth comes from the AMOTA-Bayes gate output (the user's
# already-tuned anchor); fallback to 0.7 when the gate is 0 or undefined.
#
# Per-detector predictions on V2V4Real characterization:
#   cobevt  (p_miss=0.43 → log_lr_miss = log(0.43/0.7) = -0.49) — gentle
#   CPft    (p_miss=0.09 → log_lr_miss = log(0.09/0.7) = -2.05) — aggressive
#   CP-zs   (p_miss=0.54 → log_lr_miss = log(0.54/0.7) = -0.26) — very gentle
#
# Rationale: detectors whose TPs miss often (cobevt cooperative noise +
# clustering, CP-zs short-lifetime nuScenes transfer) need permissive
# lifecycle so tracks survive their natural miss rate; detectors whose
# TPs are reliable (CPft) can afford strict decay because each miss is
# strong FP evidence.
# ---------------------------------------------------------------------------

P_MISS_TP_FLOOR = 0.05       # avoid log_lr_miss → -∞ when p_miss is tiny
LIFECYCLE_LOG_LR_MISS_MIN = -3.0   # clamp; beyond this is over-aggressive
LIFECYCLE_LOG_LR_MISS_MAX = -0.05  # always at least slight decay
LIFECYCLE_KILL_LOG_ODDS   = -3.0   # canonical kill threshold; held fixed


def solve_lifecycle_params(rows: List[Dict],
                            p_birth_gate: Optional[float]) -> Dict[str, float]:
    """Per-detector log_odds lifecycle params from characterization rows.

    log_lr_miss derivation:
      For a TP track of natural lifetime L_TP frames with miss rate p_miss_TP,
      the *expected* total log-odds decay over the track's life is
          E[decay] = L_TP · p_miss_TP · |log_lr_miss|
      We require this not exceed the kill budget |logit(p_birth) - kill|,
      so that TPs survive their natural miss rate over their natural lifetime.
      Solving with equality gives the most-aggressive lifecycle that still
      preserves TPs:
          |log_lr_miss| = |logit(p_birth) - kill_log_odds| / (L_TP · p_miss_TP)
      Clamped to [LIFECYCLE_LOG_LR_MISS_MIN, LIFECYCLE_LOG_LR_MISS_MAX].

    Worked examples (V2V4Real defaults p_birth=0.7, kill=-3.0 → budget=3.85):
      cobevt (L_TP=57, p_miss=0.30):    |lr| = 3.85/(57·0.30) = 0.23 — gentle
      CPft   (L_TP=57, p_miss=0.09):    |lr| = 3.85/(57·0.09) = 0.75 — moderate
      CP-zs  (L_TP=3.1, p_miss=0.41):   |lr| = 3.85/(3.1·0.41) = 3.03 — strong
      bev_f  (L_TP=3.1, p_miss=0.65):   |lr| = 3.85/(3.1·0.65) = 1.91 — strong

    Returns dict with keys: p_birth, log_lr_miss, kill_log_odds,
    p_miss_tp_observed, l_tp_observed, n_tracks_observed.
    """
    # Sample-count-weighted aggregation across bins, using n_gt_tracks
    # as the weight on each (p_match, L_TP) statistic.
    match_triples = []
    life_triples = []
    for r in rows:
        n_tracks = r.get("n_gt_tracks", 0.0)
        if n_tracks <= 0:
            continue
        tp_match = r.get("tp_p_match", 0.0)
        if tp_match > 0.0:
            match_triples.append((tp_match, n_tracks))
        tp_life = r.get("tp_lifetime_mean", 0.0)
        if tp_life > 0.0:
            life_triples.append((tp_life, n_tracks))

    DEFAULT_CONFIRM = math.log(20.0)           # +2.996 — canonical
    DEFAULT_TAPE_DECAY = 0.5                    # current global default

    if not match_triples or not life_triples:
        return {
            "p_birth":            0.7,
            "log_lr_miss":        math.log(0.30 / 0.70),  # canonical default
            "kill_log_odds":      LIFECYCLE_KILL_LOG_ODDS,
            "confirm_log_odds":   DEFAULT_CONFIRM,
            "tape_score_miss_decay": DEFAULT_TAPE_DECAY,
            "p_miss_tp_observed": float("nan"),
            "l_tp_observed":      float("nan"),
            "n_tracks_observed":  0,
        }
    n_total_m = sum(n for _, n in match_triples)
    n_total_l = sum(n for _, n in life_triples)
    p_match_avg = sum(p * n for p, n in match_triples) / n_total_m
    l_tp_avg    = sum(L * n for L, n in life_triples)  / n_total_l
    p_miss_tp = max(P_MISS_TP_FLOOR, 1.0 - p_match_avg)
    l_tp = max(1.0, l_tp_avg)

    # AMOTA-Bayes gate as p_birth when available; canonical 0.7 otherwise.
    p_birth = float(p_birth_gate) if p_birth_gate and p_birth_gate > 1e-6 else 0.7
    p_b = max(1e-6, min(1.0 - 1e-6, p_birth))
    logit_p_birth = math.log(p_b / (1.0 - p_b))
    kill_budget = logit_p_birth - LIFECYCLE_KILL_LOG_ODDS
    # Most-aggressive decay that still preserves natural TP lifetimes.
    log_lr_miss_abs = kill_budget / (l_tp * p_miss_tp)
    log_lr_miss = -log_lr_miss_abs
    log_lr_miss = max(LIFECYCLE_LOG_LR_MISS_MIN,
                      min(LIFECYCLE_LOG_LR_MISS_MAX, log_lr_miss))

    # ── Tier 1.5: confirm_log_odds ─────────────────────────────────────────
    # Derive from the per-detector p_match_TP. Each TP match boosts log-odds
    # by roughly logit(p_match). We want tracks to be confirmed after
    # CONFIRM_N_MATCHES effective matches (~3 matches across detector
    # families gives consistent "burn-in" output behavior).
    #
    # confirm_log_odds = logit(p_birth) + N_confirm · avg_log_lr_hit
    #
    # For detectors with low p_match (cobevt 0.66, CP-zs 0.46), avg_log_lr_hit
    # is small, so confirm_log_odds collapses near logit(p_birth) — clamp to
    # avoid below-birth thresholds (which would mean tracks confirm at birth).
    CONFIRM_N_MATCHES = 3.0
    CONFIRM_MIN_ABSOLUTE = 0.5      # P_real > 0.62 minimum confirm threshold
    CONFIRM_MAX_ABSOLUTE = math.log(50.0)  # P_real > 0.98 cap
    avg_log_lr_hit = math.log(max(P_MISS_TP_FLOOR, p_match_avg) /
                              max(P_MISS_TP_FLOOR, 1.0 - p_match_avg))
    # Only positive log_lr_hit accumulates "real" evidence. Detectors with
    # p_match < 0.5 (avg_log_lr_hit < 0) have a degenerate log_odds tracker —
    # matches DECREASE log-odds rather than build evidence. For those we fall
    # back to the canonical confirm threshold; no useful per-detector tuning.
    if avg_log_lr_hit > 0:
        confirm_log_odds = logit_p_birth + CONFIRM_N_MATCHES * avg_log_lr_hit
    else:
        confirm_log_odds = DEFAULT_CONFIRM
    # Absolute clamp: always require confirm > +0.5 (P_real > 0.62) so tracks
    # can't auto-confirm at birth. Cap at log(50) to avoid runaway.
    confirm_log_odds = max(CONFIRM_MIN_ABSOLUTE,
                            min(CONFIRM_MAX_ABSOLUTE, confirm_log_odds))

    # ── Tier 1.5: tape_score_miss_decay ───────────────────────────────────
    # The tape_score (used for AMOTA recall-sweep ordering of unmatched
    # coasting tracks) decays multiplicatively per miss frame. For detectors
    # whose TPs miss often (high p_miss_TP), aggressive decay (0.5) under-
    # counts those TPs in the recall sweep. For reliable detectors (low
    # p_miss_TP), default 0.5 is fine.
    #
    # decay = 1 − p_miss_TP, clamped to [0.30, 0.95]. Examples:
    #   cobevt  (p_miss=0.30): decay=0.70  (gentle — coasting tracks survive)
    #   CPft    (p_miss=0.08): decay=0.92  (very gentle)
    #   CP-zs   (p_miss=0.41): decay=0.59  (moderate)
    tape_decay = max(0.30, min(0.95, 1.0 - p_miss_tp))

    return {
        "p_birth":               p_birth,
        "log_lr_miss":           log_lr_miss,
        "kill_log_odds":         LIFECYCLE_KILL_LOG_ODDS,
        "confirm_log_odds":      round(confirm_log_odds, 4),
        "tape_score_miss_decay": round(tape_decay, 4),
        "p_miss_tp_observed":    round(p_miss_tp, 4),
        "l_tp_observed":         round(l_tp, 2),
        "n_tracks_observed":     int(n_total_m),
    }


def analyze_detector(name: str, src: Path, w: float,
                     lifecycle_cap: bool = False,
                     l_fp_lifecycle: float = float("inf"),
                     bin_merge_range: Optional[float] = None,
                     bin_merge_angle: Optional[float] = None,
                     adaptive_pooling: bool = False,
                     native_min_tp: float = 20.0,
                     native_min_fp: float = 20.0,
                     max_ring: int = 3) -> Optional[Dict]:
    csv_path = src / "polar_binned_errors.csv"
    if not csv_path.exists():
        print(f"  [skip] {name}: no polar_binned_errors.csv at {csv_path}")
        return None

    rows = load_polar_binned_errors(csv_path)
    n_fine = len(rows)
    if adaptive_pooling:
        # Native 5m × 10° with per-cell ring expansion. Pre-resolves the
        # solver internally, so the per_bin loop below is bypassed.
        per_bin = adaptive_pool_solve(
            rows,
            range_step=5.0, angle_step=10.0,
            native_min_tp=native_min_tp, native_min_fp=native_min_fp,
            max_ring=max_ring,
            lifecycle_cap=lifecycle_cap, l_fp_lifecycle=l_fp_lifecycle, w=w,
        )
        # Ring-usage breakdown for visibility into how aggressive the
        # expansion was per detector.
        ring_counts = defaultdict(int)
        for b in per_bin:
            ring_counts[b["ring_used"]] += 1
        ring_summary = " ".join(f"r{k}={ring_counts[k]}" for k in sorted(ring_counts))
        print(f"    adaptive: {n_fine} native → {len(per_bin)} solved cells  "
              f"[{ring_summary}]")
    else:
        if bin_merge_range and bin_merge_angle:
            rows = merge_bins(rows, bin_merge_range, bin_merge_angle)
            print(f"    merged {n_fine} fine bins → {len(rows)} coarse "
                  f"({bin_merge_range:.0f}m × {bin_merge_angle:.0f}°)")
        per_bin = []  # type: List[Dict]

        for r in rows:
            n_tp = max(0.0, r["gt_count"] - r["missed"])
            n_fp = r["n_fp"]
            if n_tp < MIN_N_TP or n_fp < MIN_N_FP:
                continue

            L_tp = r["tp_lifetime_mean"]
            L_fp_raw = r["fp_lifetime_mean"]
            if L_tp < MIN_TP_LIFETIME or L_fp_raw < MIN_FP_LIFETIME:
                continue

            # Lifecycle-conditioned L_FP: see merge_bins / adaptive paths.
            L_fp = min(L_fp_raw, l_fp_lifecycle) if lifecycle_cap else L_fp_raw

            lam_data = L_fp / L_tp
            lam = max(LAMBDA_FLOOR, min(LAMBDA_CEIL, w * lam_data))

            s_star = solve_amota_bayes_gate(
                n_tp, r["score_mean"], r["score_std"],
                n_fp, r["fp_score_mean"], r["fp_score_std"],
                lam,
            )
            if s_star is None:
                continue

            is_case_a = 1e-6 < s_star < 1.0 - 1e-6
            # Legacy (non-adaptive) path: "native" == this row, ring=0 by definition.
            # Weight = 1.0 per cell to match the adaptive path's uniform-per-cell
            # weighting policy.
            per_bin.append({
                "range_lo":    r["range_lo"],
                "range_hi":    r["range_hi"],
                "angle_lo":    r["angle_lo"],
                "angle_hi":    r["angle_hi"],
                "n_tp":        n_tp,
                "n_fp":        n_fp,
                "native_n_tp": n_tp,
                "native_n_fp": n_fp,
                "ring_used":   0,
                "tp_mu":       round(r["score_mean"], 4),
                "tp_sigma":    round(r["score_std"],  4),
                "fp_mu":       round(r["fp_score_mean"], 4),
                "fp_sigma":    round(r["fp_score_std"],  4),
                "L_tp_mean":   round(L_tp, 3),
                "L_fp_mean":   round(L_fp, 3),
                "lambda_data": round(lam_data, 4),
                "lambda_eff":  round(lam, 4),
                "s_star":      round(s_star, 4),
                "weight":      1.0,
                "case":        "A" if is_case_a else ("B" if s_star <= 1e-6 else "C"),
            })

    if not per_bin:
        return None

    # ── Polar smoothing: replace per-bin Bayes-optimal s* with the local P25
    # of a 3×3 (range, angle) neighborhood. This combines two corrections:
    #
    #   1. SMOOTHING — eliminates the piecewise-step discontinuities (case-C
    #      cliffs at gate=1, case-B drops to 0) that hurt polar mode in the
    #      DDAB_P25 benchmark (polar consistently -3 to -10 V2V vs static).
    #
    #   2. RECALL-BIAS — same P25-vs-P50 correction we apply globally, but
    #      now LOCAL. Each cell's gate is the 25th percentile of its
    #      neighborhood's per-bin Bayes thresholds. AMOTA's recall-asymmetric
    #      objective is captured locally rather than across the entire detector.
    #
    # Result: a smooth (range, angle) gate surface that respects local
    # structure but doesn't suffer the Bayes-vs-AMOTA mismatch. Runtime
    # bilinear interpolation (in birth_gate_curves.py:_lookup_polar) gives
    # continuous values between cell centers.
    _smooth_polar_bins_inplace(per_bin)

    # ── Polar table (used as runtime ground truth — keeps ALL bins, including
    #    case-B/C dominance answers).
    # ── Static / linear / quadratic FITS use case-A AND case-B bins:
    #     - case-A (s* ∈ (0,1)): real crossings in the overlap region.
    #     - case-B (s*=0): "TP dominates, no gate needed in this region" —
    #       a legitimate downward constraint on the smooth gate, NOT noise.
    #       Excluding it was the cobevt/CPft regression source: detectors
    #       whose per-bin optimum is mostly "no gate" had their scalars
    #       pulled UP by the few case-A bins (which sit at high score),
    #       overgating the rest. Including case-B lets the count-weighted
    #       median reflect the true global optimum.
    #     - case-C (s*=1): "FP dominates, reject everything" — saturating
    #       answer, asymptotic. Including it would peg the median at 1.0
    #       for detr3d-style detectors (79% case-C on V2V4Real); polar
    #       handles those per-bin directly. Smooth fits skip case-C.
    fit_bins = [b for b in per_bin if b["case"] in ("A", "B")]

    # Scalar gate: weighted P25 over A+B bins (NOT median).
    # P50 (median) gives the per-bin Bayes-optimal classification threshold
    # averaged across bins, which is biased high relative to AMOTA-optimal
    # gating. P25 matches expert hand-tuning empirically (e.g. CP-zs P25 =
    # 0.295 ≈ master's 0.30). Case-B-dominant detectors stay at 0 because
    # P25 of "0+0+0+...+s_A" is still 0.
    if fit_bins:
        weighted_for_scalar = [(b["s_star"], b["weight"]) for b in fit_bins]
        scalar_gate = _wpercentile(weighted_for_scalar, q=GATE_PERCENTILE)
    else:
        # All bins are case-C → smooth fit is undefined (detector says
        # "reject everywhere"). Polar mode still works per-bin at runtime.
        scalar_gate = None

    # Per-range bucket (count-weighted P25 across angles within A+B bins)
    by_range_ab: Dict[Tuple[float, float], List[Tuple[float, float]]] = defaultdict(list)
    for b in fit_bins:
        by_range_ab[(b["range_lo"], b["range_hi"])].append((b["s_star"], b["weight"]))
    per_range = {k: _wpercentile(v, q=GATE_PERCENTILE) for k, v in sorted(by_range_ab.items())}

    # Curve fits on per-range medians (gate vs range_mid) — A+B bins
    fit_pts_per_range = [
        (0.5 * (lo + hi), s, sum(w for _, w in by_range_ab[(lo, hi)]))
        for (lo, hi), s in per_range.items() if s is not None
    ]
    a_lin, b_lin = fit_linear(fit_pts_per_range)
    a_q, b_q, c_q = fit_quadratic(fit_pts_per_range)

    # Tier 1 lifecycle autotune — runs over the raw native rows (not the
    # pooled/adaptive output, since we want detector-wide aggregate p_match).
    lifecycle = solve_lifecycle_params(rows, p_birth_gate=scalar_gate)

    return {
        "detector":   name,
        "n_bins":     len(per_bin),
        "scalar":     round(scalar_gate, 4) if scalar_gate is not None else None,
        "per_range":  {f"{lo:.0f}-{hi:.0f}m": round(s, 4) for (lo, hi), s in per_range.items() if s is not None},
        "fits": {
            "lin_a":  round(a_lin, 6),  "lin_b":  round(b_lin, 6),
            "quad_a": round(a_q,   6),  "quad_b": round(b_q,   6),  "quad_c": round(c_q, 6),
            "n_range_bins": len(fit_pts_per_range),
        },
        "lifecycle":  lifecycle,
        "per_bin": per_bin,   # full polar table for runtime polar lookup
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _merge_write_csv(path: Path, header: List[str], current_rows: List[List],
                     current_detectors: set) -> None:
    """Write ``header`` + merged rows to ``path``, keyed on the ``detector`` column.

    The parent ``results/birth_gate_calibration/`` aggregates (birth_gate_curves
    .csv, recommended_lifecycle.csv) hold rows for MANY detectors across many
    runs. The solver is run per-detector-subset (e.g. just the PP + CPZS
    V2V4Real detectors), so it must NOT clobber rows for detectors it didn't
    touch this run (centerpoint_54m_*, cobevt_*, dmstrack_*, mix_*, …).

    Merge semantics (same contract as derive_optimal_birth_gate._merge_write_csv,
    pinned by tests/test_birth_gate_merge.py): read any existing file, drop rows
    whose ``detector`` (column 0) is in ``current_detectors`` — they're being
    regenerated — keep all other rows, then append the current run's rows. A
    pre-existing file with a mismatched header is treated as stale and fully
    replaced.
    """
    kept: List[List[str]] = []
    if path.exists():
        with open(path, newline="") as f:
            existing = list(csv.reader(f))
        if existing:
            existing_header, existing_body = existing[0], existing[1:]
            if existing_header == header:
                kept = [row for row in existing_body
                        if row and row[0] not in current_detectors]
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(kept)
        wr.writerows(current_rows)


def write_outputs(results: List[Dict], w: float) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-detector full polar bin table
    for r in results:
        out = OUT_DIR / f"{r['detector']}_polar.csv"
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(r["per_bin"][0].keys()))
            wr.writeheader()
            wr.writerows(r["per_bin"])

    # Recommended scalar (static) gates per detector
    rec = OUT_DIR / "recommended_gates.csv"
    with open(rec, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["detector", "n_bins", "scalar", "metric_weight_w"])
        for r in sorted(results, key=lambda x: x["detector"]):
            wr.writerow([r["detector"], r["n_bins"], r["scalar"], w])

    # Curve fits per detector — emit THREE rows per detector, one per fit
    # form, all consumable via existing birth_gate_curves.py loader keyed by
    # float alpha. The rows encode each fit form into the (a_quad, b_quad,
    # c_quad) triple so runtime always evaluates as quadratic:
    #   alpha=97.0  → static    : a=b=0, c=scalar           (constant gate)
    #   alpha=98.0  → linear    : a=0, b=lin_a, c=lin_b     (linear in r)
    #   alpha=99.0  → quadratic : (quad_a, quad_b, quad_c)  (true quad)
    # Detector aliases also written so configs that name model variants
    # without their own characterization data fall back to a parent's curve.
    # Detector-name aliases: configs that use a `{vehicle}`-substituted
    # detector_name (e.g., "mix_cpzs_cpft_{vehicle}") get separate astuff/tesla
    # lookups. The aliases route those names back to the closest available
    # characterization data (same model, possibly different deployment).
    DETECTOR_ALIASES = {
        # cp_zeroshot uses centerpoint nuScenes data (same model)
        "centerpoint": [
            "centerpoint_100m_on_v2v4real",
            "u_cp_cp_astuff", "u_cp_cp_tesla",
            "u_cp_cpft_astuff",     # CP-zs side of CP×CPft mix
            "u_cpft_cp_tesla",
            "mix_cpzs_cpft_astuff",
            "mix_cpft_cpzs_tesla",
            "mix_cpzs_pp_astuff", "mix_pp_cpzs_tesla",  # CP-zs in PP-CP mixes
            # ── mix_pp_cpzs_astuff and mix_cpzs_pp_tesla intentionally NOT
            # listed here. Those are PP-running vehicles in heterogeneous mixes.
            # PointPillars has its own error characteristics (different x_std/
            # y_std/yaw_std vs range), and the GPEM cov path correctly loads
            # mix_pp_cpzs_*_polar_calibration.csv per-vehicle. But there's no
            # PP-specific birth gate characterization (PP's Block D-G schema
            # never run). Forcing centerpoint's gate (0.41) on the PP-vehicle
            # over-gates real TPs — master.csv's u_pp_cp_lf uses NO birth gate
            # at all and hits 39.35 V2V; our dd_ab variants forced the gate
            # and dropped to 27.5. Leaving these unaliased so PP-vehicle gets
            # gate=None at runtime, matching master.
        ],
        "centerpoint_54m_v2v4real_finetune_astuff": [
            "u_cpft_cpft_astuff",
            "u_cpft_cp_astuff",
            "mix_cpft_cpzs_astuff",
            "mix_cpft_pp_astuff",   # CPft in PP-CPft mixes
        ],
        "centerpoint_54m_v2v4real_finetune_tesla": [
            "u_cpft_cpft_tesla",
            "u_cp_cpft_tesla",
            "mix_cpzs_cpft_tesla",
            "mix_pp_cpft_tesla",
        ],
        # PointPillars per-vehicle (score≥0). Routes PP-vehicle slots in
        # heterogeneous mixes back to their detector's own characterization,
        # not centerpoint's. Names match the `mix_pp_cpft_*`/`mix_cpft_pp_*`/
        # `mix_pp_cpzs_*`/`mix_cpzs_pp_*` detector_name expansion convention:
        # the FIRST detector in the mix name is on the FIRST listed vehicle
        # (astuff), and the SECOND is on tesla (for the `mix_pp_cpzs` pair).
        "pointpillar_astuff": [
            "mix_pp_cpzs_astuff",   # astuff=PP in PP+CP-zs mix
            "mix_pp_cpft_astuff",   # astuff=PP in PP+CPft mix
            "u_pp_cp_astuff",       # uniform PP+CP fleet (astuff side)
            "u_pp_cpft_astuff",
            "u_pp_pp_astuff",       # both-PP fleet
        ],
        "pointpillar_tesla": [
            "mix_cpzs_pp_tesla",    # tesla=PP in CP-zs+PP mix
            "mix_cpft_pp_tesla",    # tesla=PP in CPft+PP mix
            "u_pp_cp_tesla",
            "u_pp_cpft_tesla",
            "u_pp_pp_tesla",
        ],
    }
    # Track every detector_name written this run (primary names + aliases).
    # Used as the merge key so the parent-dir aggregates only drop/replace
    # rows for detectors this run actually touched (see _merge_write_csv).
    current_detectors: set = set()
    for r in results:
        current_detectors.add(r["detector"])
        current_detectors.update(DETECTOR_ALIASES.get(r["detector"], []))

    # Write to BOTH the amota_bayes subdir (canonical solver output location)
    # AND the parent birth_gate_calibration dir (where birth_gate_curves.py
    # loader looks by default — _DEFAULT_CURVE_CSV in src/birth_gate_curves.py).
    # The SUBDIR copy is the solver's own scratch output — full overwrite is
    # fine there. The PARENT copy is a shared aggregate consumed at runtime, so
    # it is MERGE-SAFE: rows for detectors not in this run are preserved.
    fits_path = OUT_DIR / "birth_gate_curves.csv"
    parent_curves = OUT_DIR.parent / "birth_gate_curves.csv"
    fits_header = ["detector", "alpha", "fit_type",
                   "a_quad", "b_quad", "c_quad",
                   "a_lin",  "b_lin",
                   "n_range_bins"]
    fits_rows: List[List] = []
    for r in sorted(results, key=lambda x: x["detector"]):
        fit = r["fits"]
        scalar = r["scalar"] if r["scalar"] is not None else 0.5
        rows_for_detector = [
            # alpha, fit_type, a_quad, b_quad, c_quad, a_lin, b_lin
            (97.0, f"amota_bayes_static_w{w}",
             0.0, 0.0, scalar,
             0.0, scalar),
            (98.0, f"amota_bayes_linear_w{w}",
             0.0, fit["lin_a"], fit["lin_b"],
             fit["lin_a"], fit["lin_b"]),
            (99.0, f"amota_bayes_quad_w{w}",
             fit["quad_a"], fit["quad_b"], fit["quad_c"],
             fit["lin_a"], fit["lin_b"]),
            # Polar = per-(range, angle) bin lookup at runtime; the
            # per-detector polar table lives at <detector>_polar.csv.
            # Coefficients here are a fallback (= scalar) for when the
            # polar table can't be loaded (sparse-bin fallback at runtime).
            (96.0, f"amota_bayes_polar_w{w}",
             0.0, 0.0, scalar,
             0.0, scalar),
        ]
        for alpha, ft, qa, qb, qc, la, lb in rows_for_detector:
            # Primary detector row
            fits_rows.append([r["detector"], alpha, ft, qa, qb, qc, la, lb,
                              fit["n_range_bins"]])
            # Alias rows (e.g., cp_zeroshot uses centerpoint data)
            for alias in DETECTOR_ALIASES.get(r["detector"], []):
                fits_rows.append([alias, alpha, ft, qa, qb, qc, la, lb,
                                  fit["n_range_bins"]])

    # Subdir: full overwrite (solver-owned scratch).
    with open(fits_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(fits_header)
        wr.writerows(fits_rows)

    # Also copy each polar table CSV to the alias names for direct loader lookup
    import shutil as _sh
    for r in results:
        src = OUT_DIR / f"{r['detector']}_polar.csv"
        if not src.exists():
            continue
        for alias in DETECTOR_ALIASES.get(r["detector"], []):
            dst = OUT_DIR / f"{alias}_polar.csv"
            _sh.copyfile(src, dst)

    # Parent: MERGE-SAFE — drop/replace only this run's detectors, keep the
    # rest (centerpoint_54m_*, cobevt_*, dmstrack_*, mix_*, …) intact. This is
    # the runtime-loaded copy (src/birth_gate_curves.py:_DEFAULT_CURVE_CSV).
    _merge_write_csv(parent_curves, fits_header, fits_rows, current_detectors)

    # Tier 1 lifecycle autotune output. Written to BOTH the amota_bayes
    # subdir and the parent calibration dir (mirrors birth_gate_curves.csv).
    # Uses the same DETECTOR_ALIASES so cooperative/mix configs find their
    # lifecycle row by detector_name.
    rec_life = OUT_DIR / "recommended_lifecycle.csv"
    parent_life = OUT_DIR.parent / "recommended_lifecycle.csv"
    lifecycle_header = ["detector", "p_birth", "log_lr_miss", "kill_log_odds",
                        "confirm_log_odds", "tape_score_miss_decay",
                        "p_miss_tp_observed", "l_tp_observed",
                        "n_tracks_observed"]
    lifecycle_rows: List[List] = []
    for r in sorted(results, key=lambda x: x["detector"]):
        lc = r["lifecycle"]
        base_row = [r["detector"], lc["p_birth"], lc["log_lr_miss"],
                    lc["kill_log_odds"],
                    lc.get("confirm_log_odds", math.log(20.0)),
                    lc.get("tape_score_miss_decay", 0.5),
                    lc["p_miss_tp_observed"],
                    lc["l_tp_observed"], lc["n_tracks_observed"]]
        lifecycle_rows.append(base_row)
        # Alias rows: replicate lifecycle for every detector_name
        # variant that maps to this detector's characterization.
        for alias in DETECTOR_ALIASES.get(r["detector"], []):
            lifecycle_rows.append([alias] + base_row[1:])

    # Subdir: full overwrite. Parent: merge-safe (same contract as curves).
    with open(rec_life, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(lifecycle_header)
        wr.writerows(lifecycle_rows)
    _merge_write_csv(parent_life, lifecycle_header, lifecycle_rows,
                     current_detectors)

    print(f"  wrote {rec_life}")
    print(f"  wrote {parent_life} (merge-safe)")

    # SUMMARY.txt human-readable
    txt = OUT_DIR / "SUMMARY.txt"
    with open(txt, "w") as f:
        f.write(f"AMOTA-cost-weighted closed-form birth gate (w={w})\n")
        f.write("=" * 60 + "\n\n")
        f.write("Per-bin formula:\n")
        f.write("  λ_bin = w · (fp_lifetime_mean / tp_lifetime_mean)\n")
        f.write("  s* = quadratic root of\n")
        f.write("    n_TP · N(s|μ_TP,σ_TP)/σ_TP = λ · n_FP · N(s|μ_FP,σ_FP)/σ_FP\n\n")
        f.write(f"{'detector':<48s} {'scalar':>8s} {'n_bins':>7s}\n")
        f.write("-" * 65 + "\n")
        for r in sorted(results, key=lambda x: x["detector"]):
            sc = r["scalar"] if r["scalar"] is not None else float("nan")
            f.write(f"{r['detector']:<48s} {sc:>8.3f} {r['n_bins']:>7d}\n")

        f.write("\n\nPer-range scalar (count-weighted median across angles)\n")
        f.write("-" * 65 + "\n")
        all_ranges = sorted({k for r in results for k in r["per_range"]})
        f.write(f"{'detector':<48s}")
        for k in all_ranges:
            f.write(f" {k:>10s}")
        f.write("\n")
        for r in sorted(results, key=lambda x: x["detector"]):
            f.write(f"{r['detector']:<48s}")
            for k in all_ranges:
                v = r["per_range"].get(k)
                f.write(f" {v:>10.3f}" if v is not None else f"{'—':>11s}")
            f.write("\n")

        f.write("\n\nQuadratic curve fits  s(r) = a·r² + b·r + c\n")
        f.write(f"{'detector':<48s} {'a':>10s} {'b':>10s} {'c':>10s}\n")
        f.write("-" * 80 + "\n")
        for r in sorted(results, key=lambda x: x["detector"]):
            f.write(f"{r['detector']:<48s} {r['fits']['quad_a']:>10.5f} "
                    f"{r['fits']['quad_b']:>10.5f} {r['fits']['quad_c']:>10.5f}\n")

    print(f"\n  wrote {rec}")
    print(f"  wrote {fits_path}")
    print(f"  wrote {txt}")
    print(f"  wrote per-detector polar CSVs to {OUT_DIR}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w", type=float, default=1.0,
                    help="Metric-preference weight (default 1.0 = MOTA-symmetric). "
                         "<1 = recall bias, >1 = precision bias.")
    ap.add_argument("--lifecycle-cap", action="store_true",
                    help="Cap L_FP by analytic lifecycle-killing time. "
                         "L_FP_eff = min(L_FP_data, L_FP_lifecycle), where "
                         "L_FP_lifecycle = (logit(p_tp_birth) − kill_log_odds) "
                         "/ |log_lr_miss| ≈ 4.8 frames for our defaults. "
                         "Required for detectors with persistent cooperative FPs "
                         "(e.g. CoBEVT) where raw L_FP overestimates the actual "
                         "cost of accepting an FP.")
    ap.add_argument("--p-tp-birth", type=float, default=0.7,
                    help="Birth p_tp assumed for lifecycle cap (default 0.7).")
    ap.add_argument("--kill-log-odds", type=float, default=-2.5,
                    help="kill_log_odds for lifecycle cap (default -2.5).")
    ap.add_argument("--log-lr-miss", type=float, default=-0.7,
                    help="log_lr_miss for lifecycle cap (default -0.7).")
    ap.add_argument("--detectors", nargs="*", default=None,
                    help="Subset of detectors to analyze (default: all).")
    ap.add_argument("--error-analysis-dir", action="append", default=None,
                    metavar="NAME=PATH",
                    help="Override the hardcoded error_analysis source dir for "
                         "detector NAME with PATH (repeatable). Lets the bucket "
                         "rebuild point the solver at the persistent "
                         "<bucket>/gpem_calibration/_error_analysis/<name> dir "
                         "instead of the legacy DETECTORS map. PATH must exist "
                         "and contain polar_binned_errors.csv.")
    ap.add_argument("--bin-merge-range", type=float, default=None,
                    help="Coarsen the input range bins to this width (m). "
                         "Source CSVs are 5m × 10° — for V2V4Real's smaller "
                         "training set, 10m bins (2x merge) yield ~4× more "
                         "samples per cell, raising case-A% above the noise "
                         "floor. Set together with --bin-merge-angle. "
                         "Mutually exclusive with --adaptive-pooling.")
    ap.add_argument("--bin-merge-angle", type=float, default=None,
                    help="Coarsen angle bins to this width (deg). 20° is the "
                         "natural V2V4Real default (2x merge).")
    ap.add_argument("--adaptive-pooling", action="store_true",
                    help="Adaptive ring-expansion (hole-fill). Each native "
                         "5m × 10° cell tries native first; if not case-A or "
                         "below --native-min-tp/fp thresholds, expand to "
                         "Chebyshev rings 1..--max-ring (3×3 → 5×5 → 7×7). "
                         "Greedy-stops at first case-A. Strictly better than "
                         "blanket --bin-merge for sparse datasets.")
    ap.add_argument("--native-min-tp", type=float, default=20.0,
                    help="Min native n_TP for a cell to skip expansion (default 20). "
                         "4× the global 5/5 filter — aggressive smoothing.")
    ap.add_argument("--native-min-fp", type=float, default=20.0,
                    help="Min native n_FP for a cell to skip expansion (default 20).")
    ap.add_argument("--max-ring", type=int, default=3,
                    help="Max Chebyshev ring for adaptive pooling (default 3 = "
                         "7×7 = 49 cells, ≈35m × 70° at native granularity).")
    args = ap.parse_args()
    if (args.bin_merge_range is None) ^ (args.bin_merge_angle is None):
        ap.error("--bin-merge-range and --bin-merge-angle must be set together.")
    if args.adaptive_pooling and (args.bin_merge_range or args.bin_merge_angle):
        ap.error("--adaptive-pooling is mutually exclusive with --bin-merge-*.")

    # Parse --error-analysis-dir NAME=PATH overrides. For an overridden NAME we
    # resolve its error_analysis source dir from PATH instead of the hardcoded
    # DETECTORS map; the map stays the default for everything else.
    overrides: Dict[str, Path] = {}
    for item in (args.error_analysis_dir or []):
        if "=" not in item:
            ap.error(f"--error-analysis-dir expects NAME=PATH, got {item!r}")
        name, _, raw_path = item.partition("=")
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            ap.error(f"--error-analysis-dir expects NAME=PATH, got {item!r}")
        path = Path(raw_path)
        if not path.exists():
            ap.error(
                f"--error-analysis-dir {name}: path does not exist: {path}\n"
                f"  Fix: run the gpem refit step first so it writes the "
                f"persistent error_analysis dir, then re-run this gate step."
            )
        if not (path / "polar_binned_errors.csv").is_file():
            ap.error(
                f"--error-analysis-dir {name}: {path} lacks polar_binned_errors.csv\n"
                f"  Fix: the gpem refit must run evaluate_errors with --plot so it "
                f"emits polar_binned_errors.csv into this dir."
            )
        overrides[name] = path

    targets = list(DETECTORS.keys()) if not args.detectors else args.detectors
    if args.lifecycle_cap:
        l_fp_lifecycle = _lifecycle_l_fp_cap(args.p_tp_birth, args.kill_log_odds,
                                              args.log_lr_miss)
        print(f"Analyzing {len(targets)} detector(s) with metric-weight w={args.w}, "
              f"lifecycle-cap ON (L_FP_lifecycle ≈ {l_fp_lifecycle:.2f} frames "
              f"@ p_birth={args.p_tp_birth}, kill={args.kill_log_odds}, miss={args.log_lr_miss}):")
    else:
        l_fp_lifecycle = float("inf")
        print(f"Analyzing {len(targets)} detector(s) with metric-weight w={args.w} "
              f"(no lifecycle cap — using raw L_FP_data):")

    results: List[Dict] = []
    for name in targets:
        # Prefer an explicit --error-analysis-dir override; else the hardcoded map.
        if name in overrides:
            src = overrides[name]
        elif name in DETECTORS:
            src = DETECTORS[name]
        else:
            print(f"  [skip] {name}: unknown detector (not in DETECTORS map "
                  f"and no --error-analysis-dir override)")
            continue
        out = analyze_detector(name, src, args.w,
                                lifecycle_cap=args.lifecycle_cap,
                                l_fp_lifecycle=l_fp_lifecycle,
                                bin_merge_range=args.bin_merge_range,
                                bin_merge_angle=args.bin_merge_angle,
                                adaptive_pooling=args.adaptive_pooling,
                                native_min_tp=args.native_min_tp,
                                native_min_fp=args.native_min_fp,
                                max_ring=args.max_ring)
        if out is None:
            print(f"  [skip] {name}: no usable bins")
            continue
        sc = out["scalar"]
        sc_str = f"{sc:.3f}" if sc is not None else "  —  "
        print(f"  {name:<48s} scalar={sc_str}  bins={out['n_bins']}")
        results.append(out)

    if results:
        write_outputs(results, args.w)


if __name__ == "__main__":
    main()
