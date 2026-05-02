"""AB3DMOT-style Kalman filter — constant velocity, fixed Q/R, vanilla update.

Mirrors the KalmanBoxTracker design from Weng et al., "AB3DMOT: A Baseline
for 3D Multi-Object Tracking and New Evaluation Metrics" (IROS 2020,
https://github.com/xinshuoweng/AB3DMOT/blob/master/AB3DMOT_libs/model.py).
We reproduce the planar (x, y) subset of their 10D model — z, theta, l,
w, h are tracked simply as the latest measurement, and the heading
(theta) does NOT enter the dynamics (CV, not CTRV).

State (4-D):     [x, y, vx, vy]
Observation:     [x, y]

Q (process):     diag([1, 1, 0.01, 0.01])   — high position process noise,
                                              low velocity (CV smooths).
R (measurement): diag([1, 1])               — AB3DMOT's published `R = I`.
P_0:             diag([10, 10, 1000, 1000]) — high initial velocity
                                              uncertainty (no prior).

Differences from our other filters:
  - No GPEM per-detection R: ignores `match.covariance` and uses fixed R.
  - No CTRV: pure CV. AB3DMOT's published baseline assumes constant
    velocity is sufficient for nuScenes/KITTI-style highway/urban driving.
  - No trust scoring, no source-reliability tracking, no batch fusion —
    measurements processed sequentially per Kalman update.

This filter is intentionally minimal — the goal is to validate that our
modular architecture can drop in a published filter and reproduce the
reference number, then layer back in the calibrated GPEM/P(TP)/log_odds
modules to show what each contributes.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from filters.base import FilterBase


# AB3DMOT's published values (KalmanBoxTracker.__init__):
#   self.kf.Q = np.eye(10)
#   self.kf.Q[7:, 7:] *= 0.01     # velocity component
#   self.kf.R = np.eye(7)         # measurement noise
#   self.kf.P[7:, 7:] *= 1000.    # initial velocity uncertainty
# Replicated for our 4-D CV slice:
DEFAULT_Q_DIAG = np.array([1.0, 1.0, 0.01, 0.01])
DEFAULT_R_DIAG = np.array([1.0, 1.0])
DEFAULT_P0_DIAG = np.array([10.0, 10.0, 1000.0, 1000.0])


class AB3DMOTKalman(FilterBase):
    """4-D constant-velocity Kalman with AB3DMOT-style fixed Q, R, and P_0.

    Construction signature matches the other filters in this module so
    Fusion can drop it in via the `filter_class=` parameter.
    """

    def __init__(self, time: float, x: float, y: float, fusion_mode: int = 0,
                 initial_width: float = 2.0, initial_length: float = 4.5,
                 use_trust_scoring: bool = False,
                 passthrough_covariance: bool = False):
        # Fusion-side common attributes (FilterBase contract).
        self.x = float(x)
        self.y = float(y)
        self.dx = 0.0
        self.dy = 0.0
        self.last_time = float(time)
        self.width = float(initial_width)
        self.length = float(initial_length)

        # Internal Kalman state — 4-D CV: [x, y, vx, vy].
        self._mu = np.array([self.x, self.y, 0.0, 0.0], dtype=float)
        self._P = np.diag(DEFAULT_P0_DIAG)
        self._Q = np.diag(DEFAULT_Q_DIAG)
        self._R = np.diag(DEFAULT_R_DIAG)
        # Observation matrix: observe [x, y].
        self._H = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0]], dtype=float)

        # FilterBase outputs (populated each fusion()):
        self.error_covariance = self._P[:2, :2].copy()
        self.d_covariance = self._P[2:, 2:].copy()
        self.trupercept_list: List = []
        self.error_tracker_temp: List = []
        self.localTrackersIDList: List = []

        # Knobs we accept-but-ignore for cross-filter compatibility:
        self._fusion_mode = fusion_mode
        self._use_trust_scoring = use_trust_scoring
        self._passthrough_covariance = passthrough_covariance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _F(self, dt: float) -> np.ndarray:
        """Transition matrix for a CV step of duration dt."""
        return np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

    def _predict(self, time: float) -> tuple:
        """Apply the CV motion model to (mu, P), returning the predicted pair."""
        dt = max(0.0, time - self.last_time)
        F = self._F(dt)
        mu_pred = F @ self._mu
        P_pred = F @ self._P @ F.T + self._Q * dt
        return mu_pred, P_pred

    def _update(self, mu_pred: np.ndarray, P_pred: np.ndarray,
                z: np.ndarray) -> tuple:
        """Standard Kalman measurement update with fixed R."""
        H = self._H
        R = self._R
        innovation = z - H @ mu_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        mu = mu_pred + K @ innovation
        P = (np.eye(4) - K @ H) @ P_pred
        return mu, P

    # ------------------------------------------------------------------
    # FilterBase API
    # ------------------------------------------------------------------

    def fusion(self, measurement_list, time: float, monitor: bool,
               participant_trust_scores: Optional[Dict[int, float]] = None) -> None:
        """Predict to `time`, then sequentially apply each measurement
        with the fixed AB3DMOT R. Width/length are taken as the latest
        matched measurement's value (no Kalman tracking on dimensions
        — matches AB3DMOT's behaviour where dim is observed each frame
        and the prior is barely used)."""
        # Predict to the current frame.
        mu, P = self._predict(time)

        # Apply each measurement.
        sensor_ids = []
        for m in measurement_list:
            z = np.array([float(m.x), float(m.y)], dtype=float)
            mu, P = self._update(mu, P, z)
            self.width = float(m.width)
            self.length = float(m.length)
            sensor_ids.append(m.id)

        self._mu = mu
        self._P = P
        self.last_time = float(time)

        self.x = float(mu[0])
        self.y = float(mu[1])
        self.dx = float(mu[2])
        self.dy = float(mu[3])
        self.error_covariance = P[:2, :2].copy()
        self.d_covariance = P[2:, 2:].copy()
        self.localTrackersIDList = sensor_ids
        # Trupercept / error-monitor outputs unused by AB3DMOT-style
        # tracking, but FilterBase contract expects them present.
        self.trupercept_list = []
        self.error_tracker_temp = []

    def getKalmanPred(self, time: float):
        """Return (x, y, a, b, phi) — predicted position + 2σ ellipse."""
        mu_pred, P_pred = self._predict(time)
        # 2σ ellipse parameters from the position-block covariance.
        Pxy = P_pred[:2, :2]
        eigvals, eigvecs = np.linalg.eigh(Pxy)
        # Guard against tiny numerical negatives:
        eigvals = np.clip(eigvals, 1e-9, None)
        a = 2.0 * math.sqrt(float(eigvals[1]))
        b = 2.0 * math.sqrt(float(eigvals[0]))
        phi = math.atan2(float(eigvecs[1, 1]), float(eigvecs[0, 1]))
        return float(mu_pred[0]), float(mu_pred[1]), a, b, phi

    def getKalmanPredWithCovariance(self, time: float):
        mu_pred, P_pred = self._predict(time)
        return float(mu_pred[0]), float(mu_pred[1]), P_pred[:2, :2].copy()
