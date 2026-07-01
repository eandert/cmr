"""GPEM-derived data-driven birth-gate curves.

Loaded from `results/birth_gate_calibration/birth_gate_curves.csv` (produced by
`scripts/derive_optimal_birth_gate.py`). For each detector, fits a quadratic
(and linear) curve through the per-range count-weighted-median of the
TP-preserving score gate at α ∈ {1.0, 2.0}.

The TP-preserving formula is:
    gate(range) = min(μ_fp + α·σ_fp, μ_tp − α·σ_tp)
which keeps roughly 84% of TPs at α=1 (~97.7% at α=2) while rejecting
detections whose score sits in the FP-dominated lower tail. Symmetric to the
GPEM std-vs-range regression used for covariance — same predictors, same fit
shape, same calibration data.

Usage:
    from birth_gate_curves import load_birth_gate_curve
    curve = load_birth_gate_curve("cobevt_tracker_tesla", alpha=1.0)
    if curve is not None:
        gate = curve.evaluate(distance_m)   # score floor at this range
"""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default location — same place the script writes to.
_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_CURVE_CSV = _REPO / "results" / "birth_gate_calibration" / "birth_gate_curves.csv"
_DEFAULT_POLAR_DIR = _REPO / "results" / "birth_gate_calibration" / "amota_bayes"


@dataclass
class BirthGateCurve:
    """Per-detector, per-alpha gate curve.

    For static/linear/quadratic fits, evaluate(distance) returns the curve
    value (clamped to [0, 1]). For polar fit (per-(range, angle) bin lookup),
    polar_table is populated and evaluate(distance, angle_rad) does the lookup.
    """
    detector: str
    alpha: float
    fit_type: str
    a_quad: float
    b_quad: float
    c_quad: float
    a_lin: float
    b_lin: float
    n_range_bins: int
    # Per-(range, angle) bin table for polar mode.
    # Each tuple: (range_lo, range_hi, angle_lo_deg, angle_hi_deg, s_star).
    polar_table: Optional[List[Tuple[float, float, float, float, float]]] = None

    def evaluate(self, distance_m: float,
                 angle_rad: Optional[float] = None,
                 fit: str = "quadratic") -> float:
        """Return the gate score at the given range. Clamped to [0, 1].

        fit values:
          - "static" or "quadratic": evaluate as quadratic (static is encoded
             with a=b=0, so quadratic returns c_quad regardless of distance).
          - "linear": evaluate as linear in distance.
          - "polar": per-(range, angle) bin lookup; falls back to quadratic
             if polar_table is not loaded or angle_rad is None.
        """
        if fit == "polar":
            if self.polar_table is None or angle_rad is None:
                # Graceful fallback to the quadratic encoding.
                return self._eval_quadratic(distance_m)
            return self._lookup_polar(distance_m, angle_rad)
        elif fit == "linear":
            v = self.a_lin * distance_m + self.b_lin
            return max(0.0, min(1.0, v))
        else:  # "static" / "quadratic" — quadratic encoding handles static via a=b=0
            return self._eval_quadratic(distance_m)

    def _eval_quadratic(self, d: float) -> float:
        v = (self.a_quad * d * d + self.b_quad * d + self.c_quad)
        return max(0.0, min(1.0, v))

    def _lookup_polar(self, d: float, angle_rad: float) -> float:
        """Bilinear interpolation over the smoothed polar bin grid.

        Each cell's s_star is already a local-P25 smoothed value (computed
        by `solve_amota_bayes_gate.py:_smooth_polar_bins_inplace`). At
        lookup time we interpolate continuously between the 4 nearest cell
        centers in (range, angle) space — eliminates step discontinuities
        between bins and the gate=1 cliffs that hurt the per-bin variant.
        """
        import math
        # Normalize angle to [-180, 180] degrees to match calibration bins.
        ang_deg = math.degrees(angle_rad)
        ang_deg = ((ang_deg + 180.0) % 360.0) - 180.0

        # Cache cell-center index on first call. Keyed by (r_mid, a_mid) → s.
        # Also stash grid step sizes / extents for the interpolator.
        if not hasattr(self, "_polar_idx") or self._polar_idx is None:
            idx: Dict[Tuple[float, float], float] = {}
            for r_lo, r_hi, a_lo, a_hi, s in self.polar_table:
                idx[(0.5 * (r_lo + r_hi), 0.5 * (a_lo + a_hi))] = s
            rs = sorted({r for r, _ in idx.keys()})
            as_ = sorted({a for _, a in idx.keys()})
            # Assume uniform spacing (true for 5m × 10° characterization grid).
            self._polar_idx = idx
            self._r_step = rs[1] - rs[0] if len(rs) > 1 else 5.0
            self._a_step = as_[1] - as_[0] if len(as_) > 1 else 10.0
            self._r_min, self._r_max = rs[0], rs[-1]
            self._a_min, self._a_max = as_[0], as_[-1]
            self._n_a = max(1, int(round(360.0 / self._a_step)))

        # Map (d, ang_deg) → fractional cell-center coordinates.
        r_f = (d - self._r_min) / self._r_step
        a_f = (ang_deg - self._a_min) / self._a_step
        # Clamp range to grid extent (no extrapolation beyond min/max).
        if r_f < 0.0:
            r_f = 0.0
        max_r_idx = (self._r_max - self._r_min) / self._r_step
        if r_f > max_r_idx:
            r_f = max_r_idx
        # Angle wraps modulo grid.
        a_f = a_f % self._n_a

        r_lo_i = int(r_f)
        r_hi_i = min(r_lo_i + 1, int(max_r_idx))
        a_lo_i = int(a_f) % self._n_a
        a_hi_i = (a_lo_i + 1) % self._n_a

        w_r = r_f - r_lo_i
        w_a = a_f - int(a_f)

        r_lo = self._r_min + r_lo_i * self._r_step
        r_hi = self._r_min + r_hi_i * self._r_step
        a_lo = self._a_min + a_lo_i * self._a_step
        a_hi = self._a_min + a_hi_i * self._a_step

        s_ll = self._polar_idx.get((r_lo, a_lo))
        s_lh = self._polar_idx.get((r_lo, a_hi))
        s_hl = self._polar_idx.get((r_hi, a_lo))
        s_hh = self._polar_idx.get((r_hi, a_hi))

        # Any missing corner → fall back to quadratic fit at this distance.
        if None in (s_ll, s_lh, s_hl, s_hh):
            return self._eval_quadratic(d)

        s = ((1.0 - w_r) * (1.0 - w_a) * s_ll +
             (1.0 - w_r) *        w_a  * s_lh +
                    w_r  * (1.0 - w_a) * s_hl +
                    w_r  *        w_a  * s_hh)
        return max(0.0, min(1.0, s))


_cache: Dict[Tuple[str, float], BirthGateCurve] = {}
_loaded_path: Optional[Path] = None


def _load_csv(path: Path) -> Dict[Tuple[str, float], BirthGateCurve]:
    out: Dict[Tuple[str, float], BirthGateCurve] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                alpha = float(row["alpha"]) if row["alpha"] else None
            except (ValueError, KeyError):
                continue
            if alpha is None:
                continue   # skip the bayes row (no alpha)
            try:
                curve = BirthGateCurve(
                    detector=row["detector"],
                    alpha=alpha,
                    fit_type=row.get("fit_type", ""),
                    a_quad=float(row.get("a_quad", 0.0) or 0.0),
                    b_quad=float(row.get("b_quad", 0.0) or 0.0),
                    c_quad=float(row.get("c_quad", 0.0) or 0.0),
                    a_lin=float(row.get("a_lin", 0.0) or 0.0),
                    b_lin=float(row.get("b_lin", 0.0) or 0.0),
                    n_range_bins=int(float(row.get("n_range_bins", 0) or 0)),
                )
            except (ValueError, KeyError):
                continue
            out[(curve.detector, curve.alpha)] = curve
    return out


def load_birth_gate_curve(detector_name: str,
                          alpha: float,
                          curve_csv: Optional[Path] = None,
                          polar_dir: Optional[Path] = None) -> Optional[BirthGateCurve]:
    """Look up the curve for (detector_name, alpha). Returns None if absent.

    For polar mode (fit_type contains 'polar'), also lazily loads the
    per-detector polar bin table from `<polar_dir>/<detector_name>_polar.csv`
    on first request.
    """
    global _cache, _loaded_path
    csv_path = Path(curve_csv) if curve_csv else _DEFAULT_CURVE_CSV
    if _loaded_path != csv_path or not _cache:
        _cache = _load_csv(csv_path)
        _loaded_path = csv_path
    curve = _cache.get((detector_name, float(alpha)))
    if curve is None:
        return None

    # Polar fit: also attach the per-bin table on first lookup.
    if "polar" in (curve.fit_type or "") and curve.polar_table is None:
        pdir = Path(polar_dir) if polar_dir else _DEFAULT_POLAR_DIR
        polar_path = pdir / f"{detector_name}_polar.csv"
        if polar_path.exists():
            curve.polar_table = _load_polar_table(polar_path)
    return curve


def _load_polar_table(path: Path) -> List[Tuple[float, float, float, float, float]]:
    """Read a per-detector polar.csv into a list of (r_lo, r_hi, a_lo, a_hi, s_star)."""
    table: List[Tuple[float, float, float, float, float]] = []
    if not path.exists():
        return table
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                table.append((
                    float(r["range_lo"]),
                    float(r["range_hi"]),
                    float(r["angle_lo"]),
                    float(r["angle_hi"]),
                    float(r["s_star"]),
                ))
            except (ValueError, KeyError):
                continue
    return table
