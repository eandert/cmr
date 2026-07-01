#!/usr/bin/env python3
"""Derive a GPEM-style data-driven birth gate per detector from calibration data.

Same predictors as GPEM regression: range only (linear/quad). Uses the same
per-bin TP/FP score statistics already in the polar_calibration CSV files.

For each detector and each alpha ∈ {1.0, 2.0} we compute, per (range, angle) bin:

    fp_upper = μ_fp + alpha · σ_fp     # upper edge of FP score distribution
    tp_lower = μ_tp - alpha · σ_tp     # lower edge of TP score distribution
    bin_gate = min(fp_upper, tp_lower) # TP-preserving gate

Aggregations (count-weighted on n_tp + n_fp):

    1) per-range bucket   → median across angles within each range bin
    2) per-detector scalar→ median across all bins
    3) linear curve fit   → gate(r) = b + a·r
    4) quadratic curve fit→ gate(r) = c + b·r + a·r²

For comparison we also emit the Bayes-optimal gate (where P(TP|score)=0.5)
per bin and per scalar — empirically too aggressive at birth, kept for
context.

Output:
    results/birth_gate_calibration/
        <detector>_optimal_gate.csv          # per-bin (one row per range×angle bin)
        recommended_gates.csv                 # detector × {bayes, tp_preserve_α1, tp_preserve_α2}
        birth_gate_curves.csv                 # per-detector linear+quad fits (α=1 and α=2)
        SUMMARY.txt                           # human-readable + per-range table
"""
import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
SENSOR_MODELS = REPO / "src" / "data" / "sensor_models"
OUT_DIR = REPO / "results" / "birth_gate_calibration"

_SCORE_GRID = [i / 1000.0 for i in range(1001)]
_LAPLACE = 0.5
_MIN_SIGMA = 1e-3
_ALPHAS = (1.0, 2.0)


# ---------------------------------------------------------------------------
# Bayesian P(TP|score) — kept for context / comparison; empirically too harsh.
# ---------------------------------------------------------------------------

def _gauss(x: float, mu: float, sigma: float) -> float:
    sigma = max(sigma, _MIN_SIGMA)
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def _bayes_p_tp(score: float, n_tp: float, n_fp: float,
                mu_tp: float, sig_tp: float,
                mu_fp: float, sig_fp: float) -> float:
    n_tp = n_tp + _LAPLACE
    n_fp = n_fp + _LAPLACE
    L_tp = _gauss(score, mu_tp, sig_tp)
    L_fp = _gauss(score, mu_fp, sig_fp)
    num = n_tp * L_tp
    den = num + n_fp * L_fp
    if den > 0:
        return num / den
    sig_tp = max(sig_tp, _MIN_SIGMA); sig_fp = max(sig_fp, _MIN_SIGMA)
    z_tp = (score - mu_tp) / sig_tp;  z_fp = (score - mu_fp) / sig_fp
    log_lr = (-0.5 * z_tp * z_tp - math.log(sig_tp)) - \
             (-0.5 * z_fp * z_fp - math.log(sig_fp))
    log_post = math.log(n_tp / n_fp) + log_lr
    return 1.0 / (1.0 + math.exp(-log_post)) if log_post >= 0 else \
           math.exp(log_post) / (1.0 + math.exp(log_post))


def _bayes_optimal_score_for_bin(n_tp: float, n_fp: float,
                                 mu_tp: float, sig_tp: float,
                                 mu_fp: float, sig_fp: float) -> Optional[float]:
    if n_tp <= 0 or n_fp <= 0:
        return None
    last = _bayes_p_tp(_SCORE_GRID[0], n_tp, n_fp, mu_tp, sig_tp, mu_fp, sig_fp)
    for s in _SCORE_GRID[1:]:
        cur = _bayes_p_tp(s, n_tp, n_fp, mu_tp, sig_tp, mu_fp, sig_fp)
        if last < 0.5 <= cur:
            return s
        last = cur
    return 0.0 if last >= 0.5 else 1.0


# ---------------------------------------------------------------------------
# TP-preserving gate (α-tunable): default formula used at runtime.
# ---------------------------------------------------------------------------

def _tp_preserving_score_for_bin(mu_tp: float, sig_tp: float,
                                 mu_fp: float, sig_fp: float,
                                 alpha: float) -> float:
    fp_upper = mu_fp + alpha * sig_fp
    tp_lower = mu_tp - alpha * sig_tp
    return max(0.0, min(1.0, min(fp_upper, tp_lower)))


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def _opt_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _row_int(row: Dict[str, str], key: str) -> int:
    v = _opt_float(row.get(key, ""))
    return int(v) if v is not None else 0


def _wmedian(items: List[Tuple[float, float]]) -> Optional[float]:
    items = sorted(items)
    tot = sum(w for _, w in items)
    if tot <= 0:
        return median([s for s, _ in items]) if items else None
    cum = 0.0
    for s, w in items:
        cum += w
        if cum >= tot / 2:
            return s
    return items[-1][0]


def _wmean(items: List[Tuple[float, float]]) -> Optional[float]:
    if not items:
        return None
    tot = sum(w for _, w in items)
    if tot <= 0:
        return sum(s for s, _ in items) / len(items)
    return sum(s * w for s, w in items) / tot


# ---------------------------------------------------------------------------
# Linear & quadratic regression (no numpy dep — keep this script lightweight)
# ---------------------------------------------------------------------------

def _solve_2x2(a, b, c, d, e, f) -> Tuple[float, float]:
    """Solve [[a,b],[c,d]] [x,y] = [e,f]."""
    det = a * d - b * c
    if abs(det) < 1e-12:
        return 0.0, e / a if abs(a) > 1e-12 else 0.0
    return (e * d - b * f) / det, (a * f - e * c) / det


def _fit_linear(pts: List[Tuple[float, float, float]]) -> Tuple[float, float]:
    """Weighted least squares y = a·x + b. pts = [(x, y, w), ...]."""
    if len(pts) < 2:
        if pts:
            return 0.0, pts[0][1]
        return 0.0, 0.0
    sw = sum(w for _, _, w in pts)
    sx = sum(x * w for x, _, w in pts)
    sy = sum(y * w for _, y, w in pts)
    sxx = sum(x * x * w for x, _, w in pts)
    sxy = sum(x * y * w for x, y, w in pts)
    a, b = _solve_2x2(sxx, sx, sx, sw, sxy, sy)
    return a, b


def _fit_quadratic(pts: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Weighted least squares y = a·x² + b·x + c. pts = [(x, y, w), ...]."""
    if len(pts) < 3:
        a, b = _fit_linear(pts)
        return 0.0, a, b
    sw = sum(w for _, _, w in pts)
    sx = sum(x * w for x, _, w in pts)
    sx2 = sum(x * x * w for x, _, w in pts)
    sx3 = sum(x ** 3 * w for x, _, w in pts)
    sx4 = sum(x ** 4 * w for x, _, w in pts)
    sy = sum(y * w for _, y, w in pts)
    sxy = sum(x * y * w for x, y, w in pts)
    sx2y = sum(x * x * y * w for x, y, w in pts)
    # Normal equations: [sx4 sx3 sx2; sx3 sx2 sx; sx2 sx sw] [a;b;c] = [sx2y; sxy; sy]
    M = [[sx4, sx3, sx2], [sx3, sx2, sx], [sx2, sx, sw]]
    v = [sx2y, sxy, sy]
    return tuple(_solve_3x3(M, v))


def _solve_3x3(M: List[List[float]], v: List[float]) -> List[float]:
    """Cramer's rule for 3x3."""
    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
              - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
              + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    D = det3(M)
    if abs(D) < 1e-12:
        # Singular — fall back to mean
        m = sum(v) / 3.0
        return [0.0, 0.0, m]
    out = []
    for i in range(3):
        Mi = [row[:] for row in M]
        for r in range(3):
            Mi[r][i] = v[r]
        out.append(det3(Mi) / D)
    return out


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_detector(name: str) -> Optional[Dict]:
    csv_path = SENSOR_MODELS / f"{name}_polar_calibration.csv"
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # Pool global TP / FP score stats for fallback when a bin is empty.
    tp_for_global, fp_for_global = [], []
    for r in rows:
        m = _opt_float(r.get("score_mean")); s = _opt_float(r.get("score_std"))
        n = _row_int(r, "gt_count") - _row_int(r, "missed")
        if m is not None and n > 0:
            tp_for_global.append((m, max(s or 0.05, _MIN_SIGMA), n))
        m = _opt_float(r.get("fp_score_mean")); s = _opt_float(r.get("fp_score_std"))
        n = _row_int(r, "n_fp")
        if m is not None and n > 0:
            fp_for_global.append((m, max(s or 0.05, _MIN_SIGMA), n))

    def _global(items, col):
        if not items:
            return None
        tot = sum(x[2] for x in items)
        return sum(x[col] * x[2] for x in items) / max(tot, 1)
    g_tp_mu = _global(tp_for_global, 0); g_tp_sig = _global(tp_for_global, 1) or 0.1
    g_fp_mu = _global(fp_for_global, 0); g_fp_sig = _global(fp_for_global, 1) or 0.1

    per_bin: List[Dict] = []
    by_range: Dict[Tuple[float, float], Dict[str, List[Tuple[float, float]]]] = {}
    scalar_pool: Dict[str, List[Tuple[float, float]]] = {
        "bayes": [],
        "tp_preserve_a1": [],
        "tp_preserve_a2": [],
    }

    for r in rows:
        n_tp = max(0, _row_int(r, "gt_count") - _row_int(r, "missed"))
        n_fp = _row_int(r, "n_fp")
        if n_tp + n_fp == 0:
            continue
        mu_tp = _opt_float(r.get("score_mean"))    or g_tp_mu
        sig_tp = max(_opt_float(r.get("score_std")) or g_tp_sig, _MIN_SIGMA)
        mu_fp = _opt_float(r.get("fp_score_mean")) or g_fp_mu
        sig_fp = max(_opt_float(r.get("fp_score_std")) or g_fp_sig, _MIN_SIGMA)

        bayes = _bayes_optimal_score_for_bin(n_tp, n_fp, mu_tp, sig_tp, mu_fp, sig_fp)
        tp1   = _tp_preserving_score_for_bin(mu_tp, sig_tp, mu_fp, sig_fp, 1.0) \
                if n_tp > 0 and n_fp > 0 else None
        tp2   = _tp_preserving_score_for_bin(mu_tp, sig_tp, mu_fp, sig_fp, 2.0) \
                if n_tp > 0 and n_fp > 0 else None

        rng_lo = _opt_float(r.get("range_lo")) or 0.0
        rng_hi = _opt_float(r.get("range_hi")) or 0.0
        ang_lo = _opt_float(r.get("angle_lo")) or 0.0
        ang_hi = _opt_float(r.get("angle_hi")) or 0.0
        weight = float(n_tp + n_fp)
        per_bin.append({
            "range_lo": rng_lo, "range_hi": rng_hi,
            "angle_lo": ang_lo, "angle_hi": ang_hi,
            "n_tp": n_tp, "n_fp": n_fp,
            "tp_mu": round(mu_tp, 4), "tp_sigma": round(sig_tp, 4),
            "fp_mu": round(mu_fp, 4), "fp_sigma": round(sig_fp, 4),
            "bayes_gate":         round(bayes, 4) if bayes is not None else "",
            "tp_preserve_alpha1": round(tp1, 4) if tp1 is not None else "",
            "tp_preserve_alpha2": round(tp2, 4) if tp2 is not None else "",
            "weight": weight,
        })

        rng_key = (rng_lo, rng_hi)
        bucket = by_range.setdefault(rng_key, {"bayes": [], "tp_preserve_a1": [], "tp_preserve_a2": []})
        if bayes is not None:
            bucket["bayes"].append((bayes, weight))
            scalar_pool["bayes"].append((bayes, weight))
        if tp1 is not None:
            bucket["tp_preserve_a1"].append((tp1, weight))
            scalar_pool["tp_preserve_a1"].append((tp1, weight))
        if tp2 is not None:
            bucket["tp_preserve_a2"].append((tp2, weight))
            scalar_pool["tp_preserve_a2"].append((tp2, weight))

    if not per_bin:
        return None

    # Per-range gates (count-weighted median across angles, per range bucket).
    per_range: Dict[Tuple[float, float], Dict[str, Optional[float]]] = {}
    fit_pts: Dict[str, List[Tuple[float, float, float]]] = {
        "bayes": [], "tp_preserve_a1": [], "tp_preserve_a2": []
    }
    for rng, by in sorted(by_range.items()):
        rng_mid = 0.5 * (rng[0] + rng[1])
        per_range[rng] = {}
        for key in ("bayes", "tp_preserve_a1", "tp_preserve_a2"):
            v = _wmedian(by[key])
            per_range[rng][key] = v
            if v is not None:
                tot_w = sum(w for _, w in by[key])
                fit_pts[key].append((rng_mid, v, tot_w))

    # Curve fits per gate flavor (linear + quadratic, gate vs range_mid).
    fits = {}
    for key, pts in fit_pts.items():
        a_lin, b_lin = _fit_linear(pts)
        a_q, b_q, c_q = _fit_quadratic(pts)
        fits[key] = {
            "lin_a": round(a_lin, 6), "lin_b": round(b_lin, 6),
            "quad_a": round(a_q, 6), "quad_b": round(b_q, 6), "quad_c": round(c_q, 6),
            "n_range_bins": len(pts),
        }

    # Scalar (count-weighted median across all bins).
    scalar = {key: _wmedian(items) for key, items in scalar_pool.items()}

    return {
        "detector": name,
        "n_bins": len(per_bin),
        "scalar":     {k: (round(v, 4) if v is not None else None) for k, v in scalar.items()},
        "per_range":  per_range,
        "fits":       fits,
        "per_bin":    per_bin,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _merge_write_csv(path: Path, header: List[str], current_rows: List[List],
                     current_detectors: set) -> None:
    """Write ``header`` + merged rows to ``path``, keyed on the ``detector`` column.

    This script is run per-bucket, so an aggregate CSV that holds rows for many
    detectors must NOT be clobbered by a run that only touches a subset. Merge
    semantics: read any existing file, drop rows whose ``detector`` is in the
    current run (they are being regenerated), keep all other rows, then append
    the current run's rows. The ``detector`` column is assumed to be column 0.
    """
    kept: List[List[str]] = []
    if path.exists():
        with open(path, newline="") as f:
            reader = csv.reader(f)
            existing = list(reader)
        if existing:
            existing_header, existing_body = existing[0], existing[1:]
            # Only trust an existing file whose header matches; otherwise the
            # old schema is stale and gets fully replaced.
            if existing_header == header:
                kept = [row for row in existing_body
                        if row and row[0] not in current_detectors]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)
        w.writerows(current_rows)


def write_outputs(results: List[Dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    current_detectors = {r["detector"] for r in results}

    # Per-detector per-bin CSV (one file per detector — overwrite is fine)
    for r in results:
        out = OUT_DIR / f"{r['detector']}_optimal_gate.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r["per_bin"][0].keys()))
            w.writeheader()
            w.writerows(r["per_bin"])

    # Recommended-gate scalar summary (all flavors). MERGED: this is a shared
    # aggregate across all detectors, so keep other detectors' rows intact.
    rec = OUT_DIR / "recommended_gates.csv"
    rec_header = ["detector", "n_bins",
                  "bayes_scalar",
                  "tp_preserve_alpha1_scalar",
                  "tp_preserve_alpha2_scalar"]
    rec_rows = [[r["detector"], r["n_bins"],
                 r["scalar"]["bayes"],
                 r["scalar"]["tp_preserve_a1"],
                 r["scalar"]["tp_preserve_a2"]]
                for r in sorted(results, key=lambda x: x["detector"])]
    _merge_write_csv(rec, rec_header, rec_rows, current_detectors)

    # Curve fits per detector (drop-in for runtime gate(distance) lookup).
    # MERGED: shared aggregate across all detectors.
    fits_path = OUT_DIR / "birth_gate_curves.csv"
    fits_header = ["detector", "alpha", "fit_type",
                   "a_quad", "b_quad", "c_quad",
                   "a_lin", "b_lin",
                   "n_range_bins"]
    fits_rows: List[List] = []
    for r in sorted(results, key=lambda x: x["detector"]):
        for alpha_key, alpha_val in (("tp_preserve_a1", 1.0),
                                     ("tp_preserve_a2", 2.0),
                                     ("bayes",          None)):
            fit = r["fits"][alpha_key]
            fits_rows.append([r["detector"], alpha_val, alpha_key,
                              fit["quad_a"], fit["quad_b"], fit["quad_c"],
                              fit["lin_a"],  fit["lin_b"],
                              fit["n_range_bins"]])
    _merge_write_csv(fits_path, fits_header, fits_rows, current_detectors)

    # Human-readable summary
    txt = OUT_DIR / "SUMMARY.txt"
    with open(txt, "w") as f:
        f.write("GPEM-derived data-driven birth gate per detector\n")
        f.write("=" * 68 + "\n\n")
        f.write("Three formulas reported per detector:\n")
        f.write("  bayes              : score where P(TP|s)=0.5 (symmetric — too harsh at birth)\n")
        f.write("  tp_preserve_alpha1 : min(μ_fp + 1σ, μ_tp − 1σ)   (~84% TP retention)\n")
        f.write("  tp_preserve_alpha2 : min(μ_fp + 2σ, μ_tp − 2σ)   (~97.7% TP retention)\n\n")
        f.write(f"{'detector':<48s} {'bayes':>8s} {'tp_α1':>8s} {'tp_α2':>8s} "
                f"{'n_bins':>7s}\n")
        f.write("-" * 84 + "\n")
        for r in sorted(results, key=lambda x: x["detector"]):
            sc = r["scalar"]
            def fmt(v): return f"{v:.3f}" if v is not None else "  —  "
            f.write(f"{r['detector']:<48s} {fmt(sc['bayes']):>8s} "
                    f"{fmt(sc['tp_preserve_a1']):>8s} "
                    f"{fmt(sc['tp_preserve_a2']):>8s} {r['n_bins']:>7d}\n")

        f.write("\n\nPer-range (TP-preserve α=1) gates per detector\n")
        f.write("-" * 68 + "\n")
        all_ranges = sorted({rng for r in results for rng in r["per_range"]})
        f.write(f"{'detector':<48s}")
        for lo, hi in all_ranges:
            f.write(f" {lo:>3.0f}-{hi:<3.0f}m")
        f.write("\n")
        for r in sorted(results, key=lambda x: x["detector"]):
            f.write(f"{r['detector']:<48s}")
            for rng in all_ranges:
                v = r["per_range"].get(rng, {}).get("tp_preserve_a1")
                f.write(f" {v:>7.3f}" if v is not None else f"{'—':>8s}")
            f.write("\n")

        f.write("\n\nCurve fits gate(r) = a·r² + b·r + c (TP-preserve α=1)\n")
        f.write(f"{'detector':<48s} {'a_quad':>10s} {'b_quad':>10s} {'c_quad':>10s} "
                f"{'a_lin':>10s} {'b_lin':>10s}\n")
        f.write("-" * 100 + "\n")
        for r in sorted(results, key=lambda x: x["detector"]):
            fit = r["fits"]["tp_preserve_a1"]
            f.write(f"{r['detector']:<48s} {fit['quad_a']:>10.5f} {fit['quad_b']:>10.5f} "
                    f"{fit['quad_c']:>10.5f} {fit['lin_a']:>10.5f} {fit['lin_b']:>10.5f}\n")

    print(f"\n  wrote {rec}")
    print(f"  wrote {fits_path}")
    print(f"  wrote {txt}")
    print(f"  wrote per-detector bin CSVs to {OUT_DIR}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    global SENSOR_MODELS, OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("detectors", nargs="*",
                    help="Detector names (without the _polar_calibration.csv suffix). Default: all.")
    ap.add_argument("--sensor-models-dir", type=Path, default=None,
                    help=f"Calibration CSV directory (default: {SENSOR_MODELS}).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"Output directory (default: {OUT_DIR}).")
    args = ap.parse_args()
    if args.sensor_models_dir:
        SENSOR_MODELS = args.sensor_models_dir
    if args.out_dir:
        OUT_DIR = args.out_dir
    names = args.detectors or sorted({
        p.stem.replace("_polar_calibration", "")
        for p in SENSOR_MODELS.glob("*_polar_calibration.csv")
    })
    print(f"Analyzing {len(names)} detector(s):")
    results: List[Dict] = []
    for name in names:
        r = analyze_detector(name)
        if r is None:
            print(f"  [skip] {name} — no calibration CSV or no bins with TP+FP")
            continue
        sc = r["scalar"]
        print(f"  {name:<48s} bayes={sc['bayes']:.3f}  "
              f"α1={sc['tp_preserve_a1']:.3f}  α2={sc['tp_preserve_a2']:.3f}  "
              f"bins={r['n_bins']}")
        results.append(r)
    if results:
        write_outputs(results)


if __name__ == "__main__":
    main()
