"""Schema specification for the unified error-model loader.

Declares the axes each error-model kind exposes and the column requirements
each mode (static / linear / quadratic / polar) imposes on the CSV files.

See error_model.py for the loader that consumes these specs and enforces them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


IndependentVar = Literal["distance", "velocity"]
Mode = Literal["static", "linear", "quadratic", "polar"]


@dataclass(frozen=True)
class AxisSpec:
    name: str                       # public axis identifier (e.g. "perp", "yaw")
    independent_var: IndependentVar
    unit: str                       # "meters" | "radians" — checked against CSV unit
    regression_csv_key: str         # error_type string in {sensor}.csv
    distribution_csv_key: str       # error_type string in {sensor}_distributions.csv


# Detector axes — measurement noise of a 3D box detection vs ego.
# regression_csv_key reflects the convention used in upstream characterization
# (radial/lateral/vertical map to distal/perpendicular/height in the bin data).
DETECTOR_AXES: tuple[AxisSpec, ...] = (
    AxisSpec("perp",       "distance", "meters",  "lateral",    "perpendicular"),
    AxisSpec("distal",     "distance", "meters",  "radial",     "distal"),
    AxisSpec("height",     "distance", "meters",  "vertical",   "height"),
    AxisSpec("yaw",        "distance", "radians", "yaw",        "yaw"),
    AxisSpec("width",      "distance", "meters",  "width",      "width"),
    AxisSpec("length",     "distance", "meters",  "length",     "length"),
    AxisSpec("box_height", "distance", "meters",  "box_height", "box_height"),
)


# Localizer axes — ego pose drift vs ground truth, parameterized by velocity.
# Same regression/distribution column conventions as detector.
LOCALIZER_AXES: tuple[AxisSpec, ...] = (
    AxisSpec("longitudinal", "velocity", "meters",  "longitudinal", "longitudinal"),
    AxisSpec("lateral",      "velocity", "meters",  "lateral",      "lateral"),
    AxisSpec("vertical",     "velocity", "meters",  "vertical",     "vertical"),
    AxisSpec("roll",         "velocity", "radians", "roll",         "roll"),
    AxisSpec("pitch",        "velocity", "radians", "pitch",        "pitch"),
    AxisSpec("yaw",          "velocity", "radians", "yaw",          "yaw"),
)


def axes_for(independent_var: IndependentVar) -> tuple[AxisSpec, ...]:
    if independent_var == "distance":
        return DETECTOR_AXES
    if independent_var == "velocity":
        return LOCALIZER_AXES
    raise ValueError(f"unknown independent_var: {independent_var!r}")


# Per-mode column requirements on the regression CSV (always required).
# An overall-bin row in the distributions CSV is required by mode="static".
REGRESSION_COLUMNS_ALWAYS: tuple[str, ...] = (
    "error_type", "intercept", "slope", "unit", "notes",
)

REGRESSION_COLUMNS_QUADRATIC: tuple[str, ...] = (
    "quad_a", "quad_b", "quad_c",
)

# Polar mode is bin-based and does NOT need regression beyond the linear
# fallback used for out-of-coverage queries (we raise on those, but the
# loader still needs the linear coefs to validate the CSV is well-formed).


# "Wide" overall-bin detector: any bin whose width exceeds this threshold
# (in the unit of dist_min/dist_max — meters for detector, m/s for localizer)
# is treated as the count-weighted aggregate row.
OVERALL_BIN_MIN_WIDTH_M = 50.0   # detector: dist_max - dist_min > 50m
OVERALL_BIN_MIN_WIDTH_VEL = 10.0  # localizer: vel_max - vel_min > 10 m/s


def overall_bin_min_width(independent_var: IndependentVar) -> float:
    if independent_var == "distance":
        return OVERALL_BIN_MIN_WIDTH_M
    if independent_var == "velocity":
        return OVERALL_BIN_MIN_WIDTH_VEL
    raise ValueError(f"unknown independent_var: {independent_var!r}")
