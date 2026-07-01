"""Per-vehicle physical attributes for the V2V4Real CAVs (tesla, astuff).

We don't have V2V4Real's raw per-frame sensor-calibration yamls on disk; the
practical alternative is to **derive the LiDAR vertical mount offset from the
frozen base GT itself**, which is the calibration encoded in the dataset:
every car-on-ground GT row obeys ``Y_center == mount_offset + h/2`` in tesla's
cam frame (V2V4Real convention). Averaging ``Y − h/2`` across all rows
recovers the tesla-frame LiDAR-to-ground offset to ~mm.

Surfaces here:

1. ``LIDAR_MOUNT_HEIGHT_M`` — per-vehicle vertical mount offset. tesla's
   value is derived from the GT at import time; astuff falls back to tesla's
   because V2V4Real ships GT only in tesla's frame (no astuff-frame ground
   truth to estimate from). If a future calibration source becomes available
   override the dict here.

2. ``DEFAULT_DIMS`` — fallback (length, width, height) tuples for the case
   where ego_state.csv per-frame dims aren't present (shouldn't happen for
   tesla/astuff). The augmenter ASSERTS the per-frame dims match these within
   ``DIMS_ASSERTION_TOL_M`` — any drift surfaces as a loud failure rather
   than a silent substitution.
"""

from __future__ import annotations

import statistics
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

# Reach the src/ neighbours of this configs/ dir so eval_config imports cleanly
# regardless of where the caller imports us from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval_config import BASE_GT_LABEL_DIR, DIM_COLS, POS_COLS  # noqa: E402


# Cached tesla-frame LiDAR mount offset, derived from the frozen, sha256-pinned
# base GT. Used as a fallback when those GT labels aren't on disk (a fresh clone
# or CI, where third_party/AB3DMOT is absent) so importing this module never
# hard-requires that external data. When the GT is present the derived value
# equals this constant.
_TESLA_MOUNT_M_FALLBACK = -1.8666935010914854


def _derive_tesla_mount_height() -> float:
    """Empirical tesla-frame LiDAR mount offset from the frozen base GT.

    For every car GT row in ``v2v4real_val_label/*.txt`` (V2V4Real ships these
    in tesla's cam frame), a car sitting on the ground obeys
    ``Y_center = LIDAR_MOUNT_HEIGHT_M[tesla] + h/2`` (the +h/2 lifts from
    ground to box center). Inverting and averaging across all rows gives an
    estimate of the mount offset to within per-car ground-elevation noise.
    """
    h_col, _w_col, _l_col = DIM_COLS
    _x_col, y_col, _z_col = POS_COLS
    mounts = []
    for label_file in sorted(Path(BASE_GT_LABEL_DIR).glob("*.txt")):
        for line in label_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            f = line.split()
            if len(f) < 17:
                continue
            try:
                h = float(f[h_col])
                y = float(f[y_col])
            except ValueError:
                continue
            if h > 0:
                # Y_center - h/2 = mount offset for cars on flat ground
                mounts.append(y - h / 2.0)
    if not mounts:
        warnings.warn(
            f"v2v4real_vehicles: no parseable GT rows under {BASE_GT_LABEL_DIR}; "
            f"using cached tesla mount {_TESLA_MOUNT_M_FALLBACK} m (frozen base-GT "
            "value). Expected in a fresh clone / CI where third_party/AB3DMOT is absent.",
            RuntimeWarning, stacklevel=2)
        return _TESLA_MOUNT_M_FALLBACK
    return float(statistics.mean(mounts))


# Derived once at import (the base GT is frozen + sha256-pinned).
_TESLA_MOUNT_M = _derive_tesla_mount_height()

LIDAR_MOUNT_HEIGHT_M: Dict[str, float] = {
    "tesla":  _TESLA_MOUNT_M,
    # V2V4Real has no astuff-frame GT to derive from; we treat the two
    # platforms as having the same LiDAR-to-ground geometry. Per-frame
    # elevation differences (e.g. astuff on a hill above tesla) are handled
    # SEPARATELY in world_to_local_kitti() via the world-z delta — this dict
    # only encodes the static sensor mount, not where the vehicle is.
    "astuff": _TESLA_MOUNT_M,
}

DEFAULT_DIMS: Dict[str, Tuple[float, float, float]] = {
    "tesla":  (4.97, 1.96, 1.44),
    "astuff": (5.18, 2.03, 1.77),
}

# Tolerance (metres) for asserting per-frame ego dims against DEFAULT_DIMS.
DIMS_ASSERTION_TOL_M = 0.05
