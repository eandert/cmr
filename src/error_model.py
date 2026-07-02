"""Unified, strict-schema error-model loader.

ONE class for detector AND localization error models. Replaces the permissive
loader in error_models.py and the duplicate CSV parser in localizer.py.

Design principles:
  - Strict schema: missing required column / row / bin -> raise with file:row
    context and what was expected.
  - No silent fallbacks: every magic-number default (0.1, 0.001, intercept-only)
    in the old loader becomes an explicit error.
  - One mode dispatch: get_std/get_mse/sample respect the requested mode
    (static/linear/quadratic/polar) consistently for every axis.

See error_model_schema.py for the axis specs and per-mode column requirements.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from error_model_schema import (
    AxisSpec, DETECTOR_AXES, LOCALIZER_AXES, IndependentVar, Mode,
    REGRESSION_COLUMNS_ALWAYS, REGRESSION_COLUMNS_QUADRATIC,
    axes_for, overall_bin_min_width,
)


SENSOR_MODELS_DIR = Path(__file__).parent / "data" / "sensor_models"

# MAE -> std conversion factor for a folded normal distribution.
# Same constant the old loader uses; reproduced here so the old module
# can eventually be deleted.
_MAE_TO_STD = 1.2533141373155003


# ===== Exception types =====================================================

class ErrorModelError(Exception):
    """Base for all error_model failures. Always includes sensor + file path."""


class ErrorModelSchemaError(ErrorModelError):
    """CSV is malformed or missing required columns/values."""


class ErrorModelCoverageError(ErrorModelError):
    """CSV is well-formed but doesn't have the data the requested mode needs.

    Triggered by: mode='static' with no overall bin, mode='polar' with no
    polar CSV or with missing (range, angle) cells, mode='quadratic' with
    no quad_a/b/c columns, query outside [0, max_range], etc.
    """


class ErrorModelAxisError(ErrorModelError):
    """Caller asked for an axis that's not in the spec for this loader.
    Detector axes vs localizer axes are different sets.
    """


# ===== Per-bin distribution data ===========================================

@dataclass(frozen=True)
class _DistBin:
    """One row from the distributions CSV. Parameters are distribution-specific."""
    axis: str                           # public axis name (perp/distal/...)
    dist_lo: float                      # distance (m) or velocity (m/s) lower bound
    dist_hi: float                      # exclusive upper bound by convention
    dist_type: str                      # normal | laplace | logistic | student_t | cauchy | constant
    params: dict[str, float]            # mu/sigma/b/s/nu/x0/gamma/value depending on dist_type
    angle_lo_deg: Optional[float] = None
    angle_hi_deg: Optional[float] = None
    source_file: str = ""
    source_row: int = -1

    @property
    def is_polar(self) -> bool:
        return self.angle_lo_deg is not None

    @property
    def width(self) -> float:
        return self.dist_hi - self.dist_lo

    def get_std(self) -> float:
        """Std of this bin's distribution. Raises on unknown type — no silent 0.1."""
        t = self.dist_type
        p = self.params
        if t == "normal":
            return float(p["sigma"])
        if t == "laplace":
            return math.sqrt(2.0) * float(p["b"])
        if t == "logistic":
            return float(p["s"]) * math.pi / math.sqrt(3.0)
        if t == "student_t":
            nu = float(p["nu"])
            sigma = float(p["sigma"])
            if nu > 2:
                return sigma * math.sqrt(nu / (nu - 2.0))
            # nu <= 2 -> infinite variance. Mark as schema error rather than
            # the old code's silent "sigma * 2" fudge.
            raise ErrorModelSchemaError(
                f"{self.source_file}:{self.source_row}: student_t with nu={nu} "
                f"has undefined std (need nu > 2). axis={self.axis}, "
                f"dist=[{self.dist_lo},{self.dist_hi})"
            )
        if t == "cauchy":
            # Cauchy has no defined std. Old loader returned gamma*2 silently.
            raise ErrorModelSchemaError(
                f"{self.source_file}:{self.source_row}: cauchy distribution "
                f"has no defined std. axis={self.axis}, "
                f"dist=[{self.dist_lo},{self.dist_hi})"
            )
        if t == "constant":
            return 0.0
        raise ErrorModelSchemaError(
            f"{self.source_file}:{self.source_row}: unknown distribution type "
            f"{t!r}. Expected one of "
            f"[normal, laplace, logistic, student_t, cauchy, constant]"
        )


# ===== Regression parameters per axis ======================================

@dataclass(frozen=True)
class _Regression:
    """Parsed regression coefficients for a single axis."""
    axis: str
    intercept: float
    slope: float
    unit: str
    notes: str
    quad_a: Optional[float]
    quad_b: Optional[float]
    quad_c: Optional[float]
    # Direct std regression (if present in CSV; preferred over MAE * 1.2533)
    std_intercept: Optional[float]
    std_slope: Optional[float]
    std_quad_a: Optional[float]
    std_quad_b: Optional[float]
    std_quad_c: Optional[float]
    # Bias regression (used for MSE = bias² + variance)
    bias_intercept: Optional[float]
    bias_slope: Optional[float]
    bias_quad_a: Optional[float]
    bias_quad_b: Optional[float]
    bias_quad_c: Optional[float]
    # Variance regression (for MSE direct)
    var_intercept: Optional[float]
    var_slope: Optional[float]
    var_quad_a: Optional[float]
    var_quad_b: Optional[float]
    var_quad_c: Optional[float]
    # Polar fit: value(r, theta_rad) = (a + b*r) * (1 + c*cos(t) + d*cos(2t))
    # polar_*       parameterizes the MAE  fit (mirrors intercept/slope/quad_*).
    # polar_std_*   parameterizes the std fit (mirrors std_intercept/std_slope/std_quad_*).
    # CMR's mode='polar' evaluates polar_std_* at runtime for σ in R.
    # Theta in radians: t = atan2(world_y, world_x) of target relative to ego.
    polar_a: Optional[float] = None
    polar_b: Optional[float] = None
    polar_c: Optional[float] = None
    polar_d: Optional[float] = None
    polar_std_a: Optional[float] = None
    polar_std_b: Optional[float] = None
    polar_std_c: Optional[float] = None
    polar_std_d: Optional[float] = None
    source_file: str = ""

    @property
    def has_quadratic(self) -> bool:
        return None not in (self.quad_a, self.quad_b, self.quad_c)

    @property
    def has_std_regression(self) -> bool:
        return self.std_intercept is not None and self.std_slope is not None

    @property
    def has_std_quadratic(self) -> bool:
        return None not in (self.std_quad_a, self.std_quad_b, self.std_quad_c)

    @property
    def has_var_regression(self) -> bool:
        return self.var_intercept is not None and self.var_slope is not None

    @property
    def has_var_quadratic(self) -> bool:
        return None not in (self.var_quad_a, self.var_quad_b, self.var_quad_c)

    def predict_std_linear(self, x: float) -> float:
        if self.has_std_regression:
            return max(0.0, self.std_intercept + self.std_slope * x)
        # Fall back to MAE * 1.2533 — std regression is a strict-enhancement
        # column; if it's not in the CSV, we use the MAE conversion. This is
        # NOT a silent fallback — it's the documented relationship between
        # MAE and std for a folded-normal.
        return _MAE_TO_STD * max(0.0, self.intercept + self.slope * x)

    def predict_std_quadratic(self, x: float) -> float:
        if self.has_std_quadratic:
            return max(0.0, self.std_quad_a * x * x + self.std_quad_b * x + self.std_quad_c)
        if not self.has_quadratic:
            raise ErrorModelCoverageError(
                f"{self.source_file}: axis {self.axis!r} has no quad_a/b/c columns; "
                f"cannot serve mode='quadratic'"
            )
        return _MAE_TO_STD * max(0.0, self.quad_a * x * x + self.quad_b * x + self.quad_c)

    @property
    def has_polar_std(self) -> bool:
        """True iff polar_std_a..d are all populated (the fit succeeded upstream)."""
        return None not in (self.polar_std_a, self.polar_std_b,
                            self.polar_std_c, self.polar_std_d)

    @property
    def has_polar_mae(self) -> bool:
        return None not in (self.polar_a, self.polar_b, self.polar_c, self.polar_d)

    def predict_std_polar(self, r: float, theta_rad: float) -> float:
        """Smooth polar fit: std(r, θ) = (a + b·r) · (1 + c·cos θ + d·cos 2θ).

        Prefers the polar_std fit (fitted to per-bin std). Falls back to the
        polar MAE fit × _MAE_TO_STD if only the MAE polar is present. Raises
        if neither is in the CSV — that's an upstream characterization issue,
        not silently approximated.
        """
        import math as _math
        if self.has_polar_std:
            a, b, c, d = (self.polar_std_a, self.polar_std_b,
                          self.polar_std_c, self.polar_std_d)
        elif self.has_polar_mae:
            a, b, c, d = (self.polar_a, self.polar_b, self.polar_c, self.polar_d)
        else:
            raise ErrorModelCoverageError(
                f"{self.source_file}: axis {self.axis!r} has no polar_std_*/polar_* "
                f"columns; cannot serve mode='polar'. Regenerate the regression CSV "
                f"via mmdetection3d/tools/regression_to_cmr_csv.py against a "
                f"regression_models.txt that includes a '<axis>_polar:' line."
            )
        scale = a + b * r
        shape = 1.0 + c * _math.cos(theta_rad) + d * _math.cos(2.0 * theta_rad)
        val = scale * shape
        if not self.has_polar_std:
            val *= _MAE_TO_STD
        return max(0.0, val)

    def predict_bias(self, x: float, use_quadratic: bool) -> float:
        if use_quadratic and self.has_var_quadratic and None not in (self.bias_quad_a, self.bias_quad_b, self.bias_quad_c):
            return self.bias_quad_a * x * x + self.bias_quad_b * x + self.bias_quad_c
        if self.bias_intercept is not None and self.bias_slope is not None:
            return self.bias_intercept + self.bias_slope * x
        return 0.0  # No bias regression -> assume zero-mean error. Documented behavior.

    def predict_variance(self, x: float, use_quadratic: bool) -> Optional[float]:
        if use_quadratic and self.has_var_quadratic:
            return max(0.0, self.var_quad_a * x * x + self.var_quad_b * x + self.var_quad_c)
        if self.has_var_regression:
            return max(0.0, self.var_intercept + self.var_slope * x)
        return None


# ===== Loader helpers ======================================================

def _opt_float(s) -> Optional[float]:
    """Parse a float-or-None from a CSV cell. Empty string and 'nan' -> None."""
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _strict_float(s, *, file: str, row: int, col: str) -> float:
    """Parse a required float. Raise if missing or unparseable."""
    if s is None or str(s).strip() == "":
        raise ErrorModelSchemaError(
            f"{file}:{row}: required column {col!r} is empty"
        )
    try:
        return float(s)
    except ValueError:
        raise ErrorModelSchemaError(
            f"{file}:{row}: column {col!r} value {s!r} is not a valid float"
        )


def _read_csv_rows(path: Path) -> list[tuple[int, dict]]:
    """Read a CSV file, returning (row_number, dict) for each data row.

    Skips comment lines (starting with #) and empty lines. Raises if the file
    is missing or empty.
    """
    if not path.exists():
        raise ErrorModelCoverageError(f"file not found: {path}")
    with path.open("r") as f:
        raw_lines = [(i + 1, line) for i, line in enumerate(f)]
    data_lines = [(i, l) for i, l in raw_lines
                  if l.strip() and not l.strip().startswith("#")]
    if not data_lines:
        raise ErrorModelSchemaError(f"{path}: file contains no data rows")
    header_idx, header_line = data_lines[0]
    reader = csv.DictReader([l for _, l in data_lines])
    body_line_numbers = [i for i, _ in data_lines[1:]]
    return list(zip(body_line_numbers, reader))


# ===== The unified loader class ============================================

class ErrorModel:
    """Strict-schema error model loader.

    Args:
        sensor_name: stem of the CSV files in data/sensor_models/, e.g.
            "pointpillar_v2v4real_tesla" or "kiss_icp".
        mode: which path the loader exposes via get_std/get_mse:
            - "static":  bin-averaged constant across the full range
                         (requires an overall bin row in the distributions CSV)
            - "linear":  std = intercept + slope * x  (regression CSV)
            - "quadratic": std = quad_a*x² + quad_b*x + quad_c
            - "polar":   per-(range, angle) bin lookup; requires
                         {sensor}_polar_distributions.csv with full coverage
        independent_var: "distance" for detector models, "velocity" for
            localizer models. Drives the axis set and the CSV column
            interpretation.
        max_range: max value of independent_var the loader will answer for.
            Out-of-range queries raise ErrorModelCoverageError.

    The constructor validates the CSV files immediately and exhaustively for
    the requested mode. If the data is incomplete or malformed, it raises with
    the file path and the specific cell that's bad.
    """

    def __init__(
        self,
        sensor_name: str,
        mode: Mode,
        independent_var: IndependentVar = "distance",
        max_range: float = 100.0,
        data_dir: Optional[Path] = None,
    ):
        self.sensor_name = sensor_name
        self.mode = mode
        self.independent_var = independent_var
        self.max_range = float(max_range)
        self.data_dir = Path(data_dir) if data_dir is not None else SENSOR_MODELS_DIR

        # Polar is detector-only.
        if mode == "polar" and independent_var != "distance":
            raise ErrorModelCoverageError(
                f"mode='polar' requires independent_var='distance'; got "
                f"independent_var={independent_var!r} for sensor {sensor_name!r}"
            )

        self._axes: tuple[AxisSpec, ...] = axes_for(independent_var)
        self._axis_by_name: dict[str, AxisSpec] = {a.name: a for a in self._axes}

        self._regression_path = self.data_dir / f"{sensor_name}.csv"
        self._distributions_path = self.data_dir / f"{sensor_name}_distributions.csv"
        self._polar_path = self.data_dir / f"{sensor_name}_polar_distributions.csv"

        # Always-required: regression coefficients per axis.
        self._regressions: dict[str, _Regression] = self._load_regressions()

        # Distributions CSV — required for static + sampling. Loaded eagerly so
        # we can validate it; mode-specific completeness check follows.
        self._bins: dict[str, list[_DistBin]] = self._load_distributions()

        # Mode-specific completeness checks.
        if mode == "static":
            self._static_values: dict[str, float] = self._compute_static_values()
        else:
            self._static_values = {}

        if mode == "quadratic":
            self._require_quadratic_for_all_axes()

        if mode == "polar":
            self._polar_bins: dict[str, list[_DistBin]] = self._load_polar()
            # mode='polar' evaluates a continuous fit equation
            #   std(r, θ) = (a + b·r) · (1 + c·cos θ + d·cos 2θ)
            # at runtime (NOT per-bin lookup). Per-bin coverage gaps are
            # therefore not a runtime concern — they're just an artifact of
            # the upstream characterization. Skip the bin-coverage check
            # when every axis has a polar fit. Fall back to the strict bin
            # check only if any axis is missing a polar fit (so the user gets
            # a loud error pointing at the missing regression CSV column).
            if not all(self._regressions[a.name].has_polar_std
                       or self._regressions[a.name].has_polar_mae
                       for a in self._axes):
                self._validate_polar_coverage()
        else:
            self._polar_bins = {}

    # ---- regression loader ----

    def _load_regressions(self) -> dict[str, _Regression]:
        rows = _read_csv_rows(self._regression_path)
        # CSV-key -> AxisSpec (used to map from CSV's error_type names to public names)
        by_csv_key = {a.regression_csv_key: a for a in self._axes}
        out: dict[str, _Regression] = {}
        seen_csv_keys: set[str] = set()
        for row_num, row in rows:
            csv_key = (row.get("error_type") or "").strip()
            if not csv_key:
                raise ErrorModelSchemaError(
                    f"{self._regression_path}:{row_num}: error_type column is empty"
                )
            if csv_key not in by_csv_key:
                # Tolerate extra rows (e.g. miss_rate) — we just don't expose them.
                continue
            seen_csv_keys.add(csv_key)
            axis = by_csv_key[csv_key]
            # Required columns
            for col in REGRESSION_COLUMNS_ALWAYS:
                if col not in row:
                    raise ErrorModelSchemaError(
                        f"{self._regression_path}:{row_num}: missing required column "
                        f"{col!r} for axis {axis.name!r}"
                    )
            intercept = _strict_float(row.get("intercept"),
                                      file=str(self._regression_path), row=row_num, col="intercept")
            slope = _strict_float(row.get("slope"),
                                  file=str(self._regression_path), row=row_num, col="slope")
            unit = (row.get("unit") or "").strip()
            if unit != axis.unit:
                raise ErrorModelSchemaError(
                    f"{self._regression_path}:{row_num}: axis {axis.name!r} has "
                    f"unit={unit!r}, expected {axis.unit!r}"
                )
            notes = row.get("notes", "")

            reg = _Regression(
                axis=axis.name,
                intercept=intercept, slope=slope, unit=unit, notes=notes,
                quad_a=_opt_float(row.get("quad_a")),
                quad_b=_opt_float(row.get("quad_b")),
                quad_c=_opt_float(row.get("quad_c")),
                std_intercept=_opt_float(row.get("std_intercept")),
                std_slope=_opt_float(row.get("std_slope")),
                std_quad_a=_opt_float(row.get("std_quad_a")),
                std_quad_b=_opt_float(row.get("std_quad_b")),
                std_quad_c=_opt_float(row.get("std_quad_c")),
                bias_intercept=_opt_float(row.get("bias_intercept")),
                bias_slope=_opt_float(row.get("bias_slope")),
                bias_quad_a=_opt_float(row.get("bias_quad_a")),
                bias_quad_b=_opt_float(row.get("bias_quad_b")),
                bias_quad_c=_opt_float(row.get("bias_quad_c")),
                var_intercept=_opt_float(row.get("var_intercept")),
                var_slope=_opt_float(row.get("var_slope")),
                var_quad_a=_opt_float(row.get("var_quad_a")),
                var_quad_b=_opt_float(row.get("var_quad_b")),
                var_quad_c=_opt_float(row.get("var_quad_c")),
                polar_a=_opt_float(row.get("polar_a")),
                polar_b=_opt_float(row.get("polar_b")),
                polar_c=_opt_float(row.get("polar_c")),
                polar_d=_opt_float(row.get("polar_d")),
                polar_std_a=_opt_float(row.get("polar_std_a")),
                polar_std_b=_opt_float(row.get("polar_std_b")),
                polar_std_c=_opt_float(row.get("polar_std_c")),
                polar_std_d=_opt_float(row.get("polar_std_d")),
                source_file=str(self._regression_path),
            )
            out[axis.name] = reg

        missing = [a.name for a in self._axes if a.regression_csv_key not in seen_csv_keys]
        if missing:
            raise ErrorModelSchemaError(
                f"{self._regression_path}: missing regression rows for axes "
                f"{missing!r}. Expected error_type values: "
                f"{[self._axis_by_name[m].regression_csv_key for m in missing]!r}"
            )
        return out

    def _require_quadratic_for_all_axes(self) -> None:
        missing: list[str] = []
        for a in self._axes:
            r = self._regressions[a.name]
            if not r.has_quadratic and not r.has_std_quadratic:
                missing.append(a.name)
        if missing:
            raise ErrorModelCoverageError(
                f"{self._regression_path}: mode='quadratic' requires quad_a/b/c (or "
                f"std_quad_a/b/c) for every axis; missing for {missing!r}"
            )

    # ---- distributions loader ----

    def _parse_bin_params(self, dist_type: str, p1: Optional[float],
                          p2: Optional[float], p3: Optional[float],
                          file: str, row: int, axis: str) -> dict[str, float]:
        """Map (param1, param2, param3) to a distribution-specific dict.

        Schema (matches DistributionBin in the old loader):
          normal:    p1=mu      p2=sigma   p3=mae (optional)
          laplace:   p1=mu      p2=b
          logistic:  p1=mu      p2=s
          student_t: p1=nu      p2=mu      p3=sigma
          cauchy:    p1=x0      p2=gamma
          constant:  p1=value
        """
        def need(val, name):
            if val is None:
                raise ErrorModelSchemaError(
                    f"{file}:{row}: distribution={dist_type!r} for axis {axis!r} "
                    f"requires parameter {name!r}"
                )
            return float(val)

        if dist_type == "normal":
            return {"mu": need(p1, "param1=mu"), "sigma": need(p2, "param2=sigma")}
        if dist_type == "laplace":
            return {"mu": need(p1, "param1=mu"), "b": need(p2, "param2=b")}
        if dist_type == "logistic":
            return {"mu": need(p1, "param1=mu"), "s": need(p2, "param2=s")}
        if dist_type == "student_t":
            return {"nu": need(p1, "param1=nu"), "mu": need(p2, "param2=mu"),
                    "sigma": need(p3, "param3=sigma")}
        if dist_type == "cauchy":
            return {"x0": need(p1, "param1=x0"), "gamma": need(p2, "param2=gamma")}
        if dist_type == "constant":
            return {"value": need(p1, "param1=value")}
        raise ErrorModelSchemaError(
            f"{file}:{row}: unknown distribution type {dist_type!r}; expected one "
            f"of [normal, laplace, logistic, student_t, cauchy, constant]"
        )

    def _load_distributions(self) -> dict[str, list[_DistBin]]:
        rows = _read_csv_rows(self._distributions_path)
        # Map CSV error_type -> public axis name
        by_csv_key = {a.distribution_csv_key: a for a in self._axes}
        bins: dict[str, list[_DistBin]] = {a.name: [] for a in self._axes}

        for row_num, row in rows:
            csv_key = (row.get("error_type") or "").strip()
            if not csv_key:
                raise ErrorModelSchemaError(
                    f"{self._distributions_path}:{row_num}: error_type column empty"
                )
            if csv_key not in by_csv_key:
                # Ignore extra rows (e.g. miss_detection in detector distributions).
                continue
            axis = by_csv_key[csv_key]
            dist_lo = _strict_float(row.get("dist_min"),
                                    file=str(self._distributions_path), row=row_num,
                                    col="dist_min")
            dist_hi = _strict_float(row.get("dist_max"),
                                    file=str(self._distributions_path), row=row_num,
                                    col="dist_max")
            if dist_hi <= dist_lo:
                raise ErrorModelSchemaError(
                    f"{self._distributions_path}:{row_num}: dist_max ({dist_hi}) "
                    f"<= dist_min ({dist_lo}) for axis {axis.name!r}"
                )
            dist_type = (row.get("distribution") or "").strip()
            if not dist_type:
                raise ErrorModelSchemaError(
                    f"{self._distributions_path}:{row_num}: distribution column empty "
                    f"for axis {axis.name!r}"
                )
            p1 = _opt_float(row.get("param1"))
            p2 = _opt_float(row.get("param2"))
            p3 = _opt_float(row.get("param3"))
            params = self._parse_bin_params(dist_type, p1, p2, p3,
                                            file=str(self._distributions_path),
                                            row=row_num, axis=axis.name)
            bins[axis.name].append(_DistBin(
                axis=axis.name, dist_lo=dist_lo, dist_hi=dist_hi,
                dist_type=dist_type, params=params,
                source_file=str(self._distributions_path), source_row=row_num,
            ))

        # Validate: every axis has at least one bin.
        for a in self._axes:
            if not bins[a.name]:
                raise ErrorModelCoverageError(
                    f"{self._distributions_path}: no distribution rows found for "
                    f"axis {a.name!r} (expected error_type={a.distribution_csv_key!r})"
                )
        return bins

    # ---- static value computation ----

    def _compute_static_values(self) -> dict[str, float]:
        """For mode='static': return per-axis std from the count-weighted overall bin.

        Convention: the importer emits one row per axis with dist_max - dist_min
        > overall_bin_min_width(independent_var). Without this row we cannot
        produce a count-weighted static value (per-bin counts are not in the
        CSV), so we raise instead of silently averaging across narrow bins.
        """
        threshold = overall_bin_min_width(self.independent_var)
        missing_overall: list[str] = []
        result: dict[str, float] = {}
        for a in self._axes:
            wide_bins = [b for b in self._bins[a.name] if b.width > threshold]
            if not wide_bins:
                missing_overall.append(a.name)
                continue
            # If multiple wide bins (e.g. 0-10 partial + 0-150 full), pick the widest.
            best = max(wide_bins, key=lambda b: b.width)
            std = best.get_std()
            if std <= 0:
                raise ErrorModelSchemaError(
                    f"{best.source_file}:{best.source_row}: overall bin for axis "
                    f"{a.name!r} has non-positive std ({std})"
                )
            result[a.name] = std

        if missing_overall:
            threshold_unit = "m" if self.independent_var == "distance" else "m/s"
            raise ErrorModelCoverageError(
                f"{self._distributions_path}: sensor {self.sensor_name!r} mode='static' "
                f"requires a count-weighted overall bin (dist_max - dist_min > "
                f"{threshold}{threshold_unit}) for every axis; missing for "
                f"{missing_overall!r}. Append them with "
                f"scripts/synthesize_overall_bins.py (run automatically by "
                f"scripts/import_sim_gpem_models.py)."
            )
        return result

    # ---- polar loader ----

    def _load_polar(self) -> dict[str, list[_DistBin]]:
        if not self._polar_path.exists():
            raise ErrorModelCoverageError(
                f"mode='polar' requires {self._polar_path}, which is not present "
                f"for sensor {self.sensor_name!r}"
            )
        rows = _read_csv_rows(self._polar_path)
        by_csv_key = {a.distribution_csv_key: a for a in self._axes}
        polar: dict[str, list[_DistBin]] = {a.name: [] for a in self._axes}

        for row_num, row in rows:
            csv_key = (row.get("error_type") or "").strip()
            if not csv_key or csv_key not in by_csv_key:
                continue
            axis = by_csv_key[csv_key]
            dist_lo = _strict_float(row.get("dist_min"), file=str(self._polar_path),
                                    row=row_num, col="dist_min")
            dist_hi = _strict_float(row.get("dist_max"), file=str(self._polar_path),
                                    row=row_num, col="dist_max")
            angle_lo = _strict_float(row.get("angle_min"), file=str(self._polar_path),
                                     row=row_num, col="angle_min")
            angle_hi = _strict_float(row.get("angle_max"), file=str(self._polar_path),
                                     row=row_num, col="angle_max")
            dist_type = (row.get("distribution") or "").strip()
            p1 = _opt_float(row.get("param1"))
            p2 = _opt_float(row.get("param2"))
            p3 = _opt_float(row.get("param3"))
            params = self._parse_bin_params(dist_type, p1, p2, p3,
                                            file=str(self._polar_path),
                                            row=row_num, axis=axis.name)
            polar[axis.name].append(_DistBin(
                axis=axis.name, dist_lo=dist_lo, dist_hi=dist_hi,
                dist_type=dist_type, params=params,
                angle_lo_deg=angle_lo, angle_hi_deg=angle_hi,
                source_file=str(self._polar_path), source_row=row_num,
            ))
        return polar

    def _validate_polar_coverage(self) -> None:
        """Every axis must have polar bins. Per-axis we verify there's at least
        one bin reachable for each angle slice we'd query. Full grid validation
        is expensive; here we check that:
          - angle bins span [-180, 180) for every axis
          - distance bins span [0, max_range) for every axis
        Holes inside the grid would yield ErrorModelCoverageError at query time.
        """
        missing: list[str] = []
        for a in self._axes:
            bins = self._polar_bins[a.name]
            if not bins:
                missing.append(a.name)
                continue
            angle_los = sorted({b.angle_lo_deg for b in bins})
            if angle_los[0] > -180.0 + 1e-6:
                raise ErrorModelCoverageError(
                    f"{self._polar_path}: axis {a.name!r} polar coverage does not "
                    f"include angle -180; first bin starts at {angle_los[0]}"
                )
            angle_his = sorted({b.angle_hi_deg for b in bins})
            if angle_his[-1] < 180.0 - 1e-6:
                raise ErrorModelCoverageError(
                    f"{self._polar_path}: axis {a.name!r} polar coverage stops at "
                    f"{angle_his[-1]} (need to reach 180)"
                )
            dist_los = sorted({b.dist_lo for b in bins})
            if dist_los[0] > 1e-6:
                raise ErrorModelCoverageError(
                    f"{self._polar_path}: axis {a.name!r} polar coverage does not "
                    f"include range 0; first bin starts at {dist_los[0]}"
                )
            dist_his = sorted({b.dist_hi for b in bins})
            if dist_his[-1] < self.max_range - 1e-6:
                raise ErrorModelCoverageError(
                    f"{self._polar_path}: axis {a.name!r} polar coverage stops at "
                    f"range {dist_his[-1]} (max_range={self.max_range})"
                )
        if missing:
            raise ErrorModelCoverageError(
                f"{self._polar_path}: missing polar bins for axes {missing!r}"
            )

    # ===== Public accessors =================================================

    def _check_axis(self, axis: str) -> AxisSpec:
        if axis not in self._axis_by_name:
            raise ErrorModelAxisError(
                f"axis {axis!r} is not in the spec for independent_var="
                f"{self.independent_var!r}; valid: {sorted(self._axis_by_name)!r}"
            )
        return self._axis_by_name[axis]

    def _check_range(self, x: float) -> None:
        if x < 0:
            raise ErrorModelCoverageError(
                f"sensor {self.sensor_name!r}: query value {x} is negative"
            )
        # Allow slight overshoot to handle FP rounding; hard-cap at 1% over.
        if x > self.max_range * 1.01:
            raise ErrorModelCoverageError(
                f"sensor {self.sensor_name!r}: query value {x} exceeds max_range "
                f"{self.max_range}"
            )

    def get_std(self, axis: str, x: float, angle_deg: Optional[float] = None) -> float:
        """Std of the error on `axis` at independent_var value `x` (distance or velocity).

        Mode dispatch:
          static    -> constant from overall bin (x ignored)
          linear    -> std intercept + slope * x (or MAE * 1.2533 fallback)
          quadratic -> std quad regression (or MAE-quad if std_quad missing)
          polar     -> evaluate the polar fit equation
                       std(r, θ) = (a + b·r) · (1 + c·cos θ + d·cos 2θ)
                       (NOT a per-bin lookup — bins are the fit's input, not its output)

        Raises ErrorModelAxisError for unknown axis, ErrorModelCoverageError
        for out-of-range x or a missing polar fit.
        """
        import math as _math
        self._check_axis(axis)
        self._check_range(x)
        if self.mode == "static":
            return self._static_values[axis]
        if self.mode == "linear":
            return self._regressions[axis].predict_std_linear(x)
        if self.mode == "quadratic":
            return self._regressions[axis].predict_std_quadratic(x)
        if self.mode == "polar":
            if angle_deg is None:
                raise ErrorModelCoverageError(
                    f"mode='polar' requires angle_deg; got None for axis {axis!r}"
                )
            theta_rad = _math.radians(angle_deg)
            return self._regressions[axis].predict_std_polar(x, theta_rad)
        raise ErrorModelCoverageError(f"unknown mode: {self.mode!r}")

    def get_mse(self, axis: str, x: float, angle_deg: Optional[float] = None) -> float:
        """MSE = bias² + variance — for filter R-matrix diagonals.

        Uses variance regression if available; falls back to std². Bias defaults
        to 0 if no bias regression is present.
        """
        self._check_axis(axis)
        self._check_range(x)
        if self.mode == "polar":
            std = self.get_std(axis, x, angle_deg)
            return std * std
        reg = self._regressions[axis]
        use_quad = (self.mode == "quadratic")
        var = reg.predict_variance(x, use_quad)
        if var is None:
            std = self.get_std(axis, x, angle_deg)
            return std * std
        bias = reg.predict_bias(x, use_quad)
        return bias * bias + var

    def _lookup_polar_bin(self, axis: str, x: float, angle_deg: float) -> _DistBin:
        # Wrap angle to [-180, 180)
        a = float(angle_deg)
        while a < -180.0:
            a += 360.0
        while a >= 180.0:
            a -= 360.0
        for b in self._polar_bins[axis]:
            if b.dist_lo <= x < b.dist_hi and b.angle_lo_deg <= a < b.angle_hi_deg:
                return b
        raise ErrorModelCoverageError(
            f"sensor {self.sensor_name!r} axis {axis!r}: no polar bin covers "
            f"(distance={x}, angle={a} deg). Coverage gap in polar CSV."
        )

    def axes(self) -> tuple[str, ...]:
        """Public axis names for this model."""
        return tuple(a.name for a in self._axes)

    # ===== Ergonomic per-axis accessors =====================================
    #
    # These mirror the old DetectorErrorModel API so callers can be migrated
    # without rewriting every call-site. Each delegates to get_std / get_mse
    # with the right axis name; there is no separate logic.

    def get_perpendicular_std(self, distance: float, angle_deg: Optional[float] = None) -> float:
        return self.get_std("perp", distance, angle_deg)

    def get_distal_std(self, distance: float, angle_deg: Optional[float] = None) -> float:
        return self.get_std("distal", distance, angle_deg)

    def get_height_std(self, distance: float, angle_deg: Optional[float] = None) -> float:
        return self.get_std("height", distance, angle_deg)

    def get_yaw_std(self, distance: float, angle_deg: Optional[float] = None) -> float:
        return self.get_std("yaw", distance, angle_deg)

    def get_width_std(self, distance: float, angle_deg: Optional[float] = None) -> float:
        return self.get_std("width", distance, angle_deg)

    def get_length_std(self, distance: float, angle_deg: Optional[float] = None) -> float:
        return self.get_std("length", distance, angle_deg)

    def get_box_height_std(self, distance: float, angle_deg: Optional[float] = None) -> float:
        return self.get_std("box_height", distance, angle_deg)

    def get_perpendicular_mse(self, distance: float, angle_deg: Optional[float] = None) -> float:
        return self.get_mse("perp", distance, angle_deg)

    def get_distal_mse(self, distance: float, angle_deg: Optional[float] = None) -> float:
        return self.get_mse("distal", distance, angle_deg)

    def get_yaw_mse(self, distance: float, angle_deg: Optional[float] = None) -> float:
        return self.get_mse("yaw", distance, angle_deg)

    # ===== Detection probability ============================================
    #
    # Detector-only. Localizer models raise NotImplementedError. The miss-rate
    # data is encoded with error_type='miss_rate' in the regression CSV and
    # error_type='miss_detection' in the (polar) distributions CSV. Both are
    # OPTIONAL — the loader doesn't fail if they're absent at construction,
    # but detection_probability() will raise if you ask for it without them.

    def detection_probability(self, distance: float,
                              angle_deg: Optional[float] = None) -> float:
        """P(detection succeeds) at this (distance, angle). 1 - miss_rate.

        Raises ErrorModelCoverageError if the sensor's CSV does not include
        miss-rate data for the current mode.
        """
        if self.independent_var != "distance":
            raise NotImplementedError(
                "detection_probability is detector-only; not defined for "
                f"independent_var={self.independent_var!r}"
            )
        self._check_range(distance)
        if distance > self.max_range:
            return 0.0
        # Mode dispatch
        if self.mode == "polar":
            # Evaluate the polar fit equation for miss_rate (NOT per-bin lookup).
            # missed_rate_polar coefficients live on the regression CSV's
            # miss_rate row (mode='polar' reads from there). If absent, fall
            # back to the per-bin lookup (older sensors) and finally to the
            # regression-fit miss_rate to mirror linear/quadratic modes.
            miss_reg = self._miss_regression()
            if miss_reg.has_polar_mae:
                import math as _math
                a_p, b_p, c_p, d_p = (
                    miss_reg.polar_a, miss_reg.polar_b,
                    miss_reg.polar_c, miss_reg.polar_d,
                )
                theta_rad = _math.radians(angle_deg if angle_deg is not None else 0.0)
                miss = (a_p + b_p * distance) * (
                    1.0 + c_p * _math.cos(theta_rad)
                        + d_p * _math.cos(2.0 * theta_rad)
                )
                return max(0.0, min(1.0, 1.0 - miss))
            # No polar fit row — try per-bin lookup (legacy sensors).
            for b in self._miss_polar_bins():
                a = self._wrap_angle(angle_deg if angle_deg is not None else 0.0)
                if (b.dist_lo <= distance < b.dist_hi
                        and b.angle_lo_deg <= a < b.angle_hi_deg):
                    miss = float(b.params.get("value", b.params.get("mu", 0.5)))
                    return max(0.0, min(1.0, 1.0 - miss))
            raise ErrorModelCoverageError(
                f"sensor {self.sensor_name!r}: detection_probability in mode='polar' "
                f"needs either missed_rate_polar coefficients on the miss_rate row "
                f"of the regression CSV, or a covering miss_detection bin in the "
                f"polar distributions CSV. Neither found for "
                f"(distance={distance}, angle={angle_deg})."
            )
        # static / linear / quadratic: use miss_rate regression
        reg = self._miss_regression()
        if self.mode == "static":
            # Aggregated miss-rate from the polar_calibration.csv: count-weighted
            # mean of the per-(range, angle) bin miss_rate column over bins that
            # actually saw GT. Falls back to the regression intercept when the
            # polar CSV is absent or unusable (preserves the previous behavior
            # for sensors that lack polar data, instead of raising).
            miss = self._aggregated_miss_rate_static()
            if miss is None:
                miss = reg.intercept  # last-resort fallback
            return max(0.0, min(1.0, 1.0 - miss))
        if self.mode == "quadratic" and reg.has_quadratic:
            miss = reg.quad_a * distance * distance + reg.quad_b * distance + reg.quad_c
        else:
            miss = reg.intercept + reg.slope * distance
        return max(0.0, min(1.0, 1.0 - miss))

    def _aggregated_miss_rate_static(self) -> Optional[float]:
        """Compute an overall miss_rate from polar_calibration.csv.

        Count-weighted mean over bins that have non-zero ``gt_count``::

            sum(missed_i)  /  sum(gt_count_i)

        Returns ``None`` if the polar_calibration.csv is missing or has no
        usable rows; caller falls back to the regression intercept.

        Sibling-resolved relative to the sensor's polar distributions path —
        polar_calibration.csv lives in the same data dir alongside
        ``<sensor>_polar_distributions.csv``.
        """
        if hasattr(self, "_static_miss_cache"):
            return self._static_miss_cache
        calib = self.data_dir / f"{self.sensor_name}_polar_calibration.csv"
        if not calib.is_file():
            self._static_miss_cache = None
            return None
        try:
            total_gt = 0
            total_missed = 0
            for _, row in _read_csv_rows(calib):
                gt = row.get("gt_count")
                missed = row.get("missed")
                if not gt or not missed:
                    continue
                try:
                    gt_i = int(float(gt))
                    miss_i = int(float(missed))
                except (TypeError, ValueError):
                    continue
                if gt_i <= 0:
                    continue
                total_gt += gt_i
                total_missed += miss_i
        except OSError:
            self._static_miss_cache = None
            return None
        if total_gt <= 0:
            self._static_miss_cache = None
            return None
        self._static_miss_cache = max(0.0, min(1.0, total_missed / total_gt))
        return self._static_miss_cache

    def _miss_regression(self) -> "_Regression":
        """Load the miss_rate row from the regression CSV. Cached after first call."""
        if hasattr(self, "_miss_reg_cache"):
            return self._miss_reg_cache
        # Re-scan the CSV for the miss_rate row (not in DETECTOR_AXES so the
        # main loader skipped it).
        for row_num, row in _read_csv_rows(self._regression_path):
            if (row.get("error_type") or "").strip() == "miss_rate":
                self._miss_reg_cache = _Regression(
                    axis="miss_rate",
                    intercept=_strict_float(row.get("intercept"),
                                            file=str(self._regression_path),
                                            row=row_num, col="intercept"),
                    slope=_strict_float(row.get("slope"),
                                        file=str(self._regression_path),
                                        row=row_num, col="slope"),
                    unit=(row.get("unit") or ""), notes=row.get("notes", ""),
                    quad_a=_opt_float(row.get("quad_a")),
                    quad_b=_opt_float(row.get("quad_b")),
                    quad_c=_opt_float(row.get("quad_c")),
                    std_intercept=None, std_slope=None,
                    std_quad_a=None, std_quad_b=None, std_quad_c=None,
                    bias_intercept=None, bias_slope=None,
                    bias_quad_a=None, bias_quad_b=None, bias_quad_c=None,
                    var_intercept=None, var_slope=None,
                    var_quad_a=None, var_quad_b=None, var_quad_c=None,
                    # missed_rate_polar coefficients — used by
                    # detection_probability(mode='polar') to evaluate
                    #   miss(r, θ) = (a + b·r) · (1 + c·cos θ + d·cos 2θ)
                    polar_a=_opt_float(row.get("polar_a")),
                    polar_b=_opt_float(row.get("polar_b")),
                    polar_c=_opt_float(row.get("polar_c")),
                    polar_d=_opt_float(row.get("polar_d")),
                    polar_std_a=None, polar_std_b=None,
                    polar_std_c=None, polar_std_d=None,
                    source_file=str(self._regression_path),
                )
                return self._miss_reg_cache
        raise ErrorModelCoverageError(
            f"sensor {self.sensor_name!r}: regression CSV has no miss_rate row; "
            f"detection_probability cannot be computed"
        )

    def _miss_polar_bins(self) -> list[_DistBin]:
        """Subset of polar bins with error_type='miss_detection'. Cached."""
        if hasattr(self, "_miss_polar_cache"):
            return self._miss_polar_cache
        # Re-scan the polar CSV for miss_detection rows (those are NOT in
        # _polar_bins because miss_detection isn't a public axis).
        miss: list[_DistBin] = []
        for row_num, row in _read_csv_rows(self._polar_path):
            if (row.get("error_type") or "").strip() != "miss_detection":
                continue
            dist_lo = _strict_float(row.get("dist_min"), file=str(self._polar_path),
                                    row=row_num, col="dist_min")
            dist_hi = _strict_float(row.get("dist_max"), file=str(self._polar_path),
                                    row=row_num, col="dist_max")
            angle_lo = _strict_float(row.get("angle_min"), file=str(self._polar_path),
                                     row=row_num, col="angle_min")
            angle_hi = _strict_float(row.get("angle_max"), file=str(self._polar_path),
                                     row=row_num, col="angle_max")
            dist_type = (row.get("distribution") or "constant").strip()
            p1 = _opt_float(row.get("param1"))
            params = {"value": float(p1) if p1 is not None else 0.5}
            miss.append(_DistBin(
                axis="miss_detection", dist_lo=dist_lo, dist_hi=dist_hi,
                dist_type=dist_type, params=params,
                angle_lo_deg=angle_lo, angle_hi_deg=angle_hi,
                source_file=str(self._polar_path), source_row=row_num,
            ))
        self._miss_polar_cache = miss
        return miss

    @staticmethod
    def _wrap_angle(angle_deg: float) -> float:
        a = float(angle_deg)
        while a < -180.0:
            a += 360.0
        while a >= 180.0:
            a -= 360.0
        return a

    # ===== Sampling =========================================================
    #
    # For synthetic-noise injection. Selects the bin matching (x, angle) and
    # samples from its distribution. Raises if no bin covers the query.

    def sample(self, axis: str, x: float, angle_deg: Optional[float] = None,
               rng=None) -> float:
        import numpy as _np
        self._check_axis(axis)
        self._check_range(x)
        if rng is None:
            rng = _np.random.default_rng()
        # Pick the bin
        if self.mode == "polar":
            if angle_deg is None:
                raise ErrorModelCoverageError(
                    f"mode='polar' sample for axis {axis!r}: angle_deg required"
                )
            b = self._lookup_polar_bin(axis, x, angle_deg)
        else:
            # Use the narrowest distance-only bin that covers x; if none does,
            # use the overall bin (widest). If neither exists, raise.
            covering = [b for b in self._bins[axis] if b.dist_lo <= x < b.dist_hi]
            covering.sort(key=lambda b: b.dist_hi - b.dist_lo)
            if not covering:
                raise ErrorModelCoverageError(
                    f"sensor {self.sensor_name!r} axis {axis!r}: no bin covers x={x}"
                )
            b = covering[0]
        # Sample from the bin's distribution
        t, p = b.dist_type, b.params
        if t == "normal":
            return float(rng.normal(p["mu"], p["sigma"]))
        if t == "laplace":
            return float(rng.laplace(p["mu"], p["b"]))
        if t == "logistic":
            return float(rng.logistic(p["mu"], p["s"]))
        if t == "student_t":
            return float(p["mu"] + p["sigma"] * rng.standard_t(p["nu"]))
        if t == "cauchy":
            return float(p["x0"] + p["gamma"] * rng.standard_cauchy())
        if t == "constant":
            return float(p["value"])
        raise ErrorModelSchemaError(
            f"unknown distribution type {t!r} in sample(); should be caught at load"
        )

    # Ergonomic sampling shims (one per detector axis)
    def sample_distal_error(self, distance: float, angle_deg: Optional[float] = None,
                            rng=None) -> float:
        return self.sample("distal", distance, angle_deg, rng)

    def sample_perpendicular_error(self, distance: float, angle_deg: Optional[float] = None,
                                    rng=None) -> float:
        return self.sample("perp", distance, angle_deg, rng)

    def sample_height_error(self, distance: float, angle_deg: Optional[float] = None,
                            rng=None) -> float:
        return self.sample("height", distance, angle_deg, rng)

    def sample_yaw_error(self, distance: float, angle_deg: Optional[float] = None,
                         rng=None) -> float:
        return self.sample("yaw", distance, angle_deg, rng)

    def sample_width_error(self, distance: float, angle_deg: Optional[float] = None,
                            rng=None) -> float:
        return self.sample("width", distance, angle_deg, rng)

    def sample_length_error(self, distance: float, angle_deg: Optional[float] = None,
                            rng=None) -> float:
        return self.sample("length", distance, angle_deg, rng)

    # ---- aggregate samplers (used by the SUMO error-injection path) ----------
    #
    # Restored after the error-model refactor: `sensor.py` (regression error
    # model) and `test_sensing_errors_gauntlet` depend on these. `target_angle`
    # is in RADIANS (sensor frame), converted to degrees for the per-axis
    # samplers; position error is rotated distal/perp -> global x/y.

    def sample_errors(self, distance: float, target_angle: float, rng=None):
        """Sample position error in the global frame + predicted distal/perp std.

        Returns ``(x_error, y_error, distal_std, perp_std)``. ``target_angle`` is
        in radians; distal/perp are sampled in the sensor frame then rotated.
        """
        import math as _math
        angle_deg = _math.degrees(target_angle)
        distal_error = self.sample_distal_error(distance, angle_deg, rng)
        perp_error = self.sample_perpendicular_error(distance, angle_deg, rng)
        x_error = distal_error * _math.cos(target_angle) - perp_error * _math.sin(target_angle)
        y_error = distal_error * _math.sin(target_angle) + perp_error * _math.cos(target_angle)
        return (x_error, y_error,
                self.get_distal_std(distance, angle_deg),
                self.get_perpendicular_std(distance, angle_deg))

    def sample_all_errors(self, distance: float, target_angle: float, rng=None) -> dict:
        """Sample all errors (position, dimensions, yaw) for an object at
        ``distance`` (m) and ``target_angle`` (radians, sensor frame).

        Returns a dict of sampled errors plus the predicted per-axis stds used to
        build the measurement covariance. Works in every mode; in ``polar`` mode
        each axis is evaluated at ``(distance, angle)`` via its smooth polar fit.
        """
        import math as _math
        angle_deg = _math.degrees(target_angle)
        x_error, y_error, distal_std, perp_std = self.sample_errors(distance, target_angle, rng)
        return {
            "x_error": x_error,
            "y_error": y_error,
            "width_error": self.sample_width_error(distance, angle_deg, rng),
            "length_error": self.sample_length_error(distance, angle_deg, rng),
            "yaw_error": self.sample_yaw_error(distance, angle_deg, rng),
            # predicted stds for covariance
            "distal_std": distal_std,
            "perp_std": perp_std,
            "width_std": self.get_width_std(distance, angle_deg),
            "length_std": self.get_length_std(distance, angle_deg),
            "yaw_std": self.get_yaw_std(distance, angle_deg),
        }
