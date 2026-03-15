"""
Covariance Intersection (CI) filter for track-level fusion.

=== WHY CI INSTEAD OF STANDARD KALMAN? ===

The standard Kalman filter assumes that the prediction error and measurement
error are UNCORRELATED.  In cooperative perception, this assumption breaks down:
  - Multiple vehicles observe the same object and share detections
  - The ego's localizer error affects ALL perceived object positions
  - Detections from the same sensor in consecutive frames share systematic bias

When errors are correlated but the filter assumes independence, the standard
Kalman update becomes OVERCONFIDENT — the fused covariance is too small, and
the filter starts ignoring measurements in favor of its own prediction. This
is the "overconfidence feedback loop" observed with the precision-weighted
fusion approach in ResizableKalman (Part 3 experiment).

Covariance Intersection avoids this entirely by finding the TIGHTEST CONSISTENT
bound on the fused estimate, regardless of what the unknown cross-correlations
might be.  The trade-off is that CI is more conservative than the optimal Kalman
update — but it is NEVER overconfident, which makes it robust in practice.

=== CI FORMULA ===

Given two estimates (xa, Pa) and (xb, Pb) with unknown cross-correlation:

    P_fused = inv(ω · inv(Pa) + (1-ω) · inv(Pb))
    x_fused = P_fused · (ω · inv(Pa) · xa + (1-ω) · inv(Pb) · xb)

where ω ∈ [0, 1] is chosen to minimize trace(P_fused) (or det(P_fused)).

Key properties:
  1. P_fused ≤ Pa  AND  P_fused ≤ Pb  in the PSD sense (at the optimal ω)
     → the fused covariance is always smaller than either input
  2. P_fused is always a valid (consistent) covariance bound regardless of
     the true cross-correlation between a and b
  3. When Pa and Pb ARE uncorrelated, CI gives a larger covariance than
     the optimal Kalman update — but this conservatism is bounded

=== ARCHITECTURE ===

This filter is a DROP-IN REPLACEMENT for ResizableKalman. It:
  - Accepts the same constructor arguments
  - Exposes the same public attributes (x, y, dx, dy, error_covariance, etc.)
  - Implements the same FilterBase interface (fusion, getKalmanPred, getKalmanPredWithCovariance)
  - Uses the same motion models (CV, CA, CTRV) for the prediction step
  - Only differs in the MEASUREMENT UPDATE step: CI fusion instead of Kalman gain

To use it, pass `filter_class=CovarianceIntersectionFilter` to GlobalTracked
or SensorFusion.

=== REFERENCES ===

[1] Julier & Uhlmann, "A non-divergent estimation algorithm in the presence
    of unknown correlations", ACC 1997.
[2] Reinhardt et al., "Closed-Form Optimization of Covariance Intersection
    for Low-Dimensional Matrices", Fusion 2012 (fast trace-optimal CI for 2D).
[3] KIT-ISAS implementation: github.com/KIT-ISAS/data-fusion
"""

import math
import numpy as np
from typing import Dict, Optional, Tuple, List
from scipy.optimize import minimize_scalar

import utils
from filters.base import FilterBase

# Shared constant: must match the value in sensor_fusion.py
# Used to extract participant_id from tracker_id for trust scoring
max_id = 10000


# =============================================================================
# CORE CI FUSION FUNCTIONS (stateless, tested independently)
# =============================================================================

def _ci_fuse(xa: np.ndarray, Pa: np.ndarray,
             xb: np.ndarray, Pb: np.ndarray,
             minimize: str = 'trace') -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fuse two estimates using Covariance Intersection (CI).

    This is the core CI algorithm. Given two state estimates with their
    covariances, it finds the optimal ω that minimizes the fused covariance
    and returns the fused estimate.

    Args:
        xa: State vector a, shape (n,) or (n,1). E.g. predicted position [x, y].
        Pa: Covariance of estimate a, shape (n, n). Must be symmetric positive definite.
        xb: State vector b, shape (n,) or (n,1). E.g. measurement [x, y].
        Pb: Covariance of estimate b, shape (n, n). Must be symmetric positive definite.
        minimize: Optimization criterion.
            'trace' — minimize trace(P_fused), equivalent to minimizing total variance.
                      This is the most common choice and has good numerical properties.
            'det'   — minimize det(P_fused), equivalent to minimizing the volume of
                      the uncertainty ellipsoid. Can be better for anisotropic covariances.

    Returns:
        x_fused: Fused state vector, shape (n,)
        P_fused: Fused covariance, shape (n, n). Guaranteed PSD and consistent.
        omega:   Optimal weight ω ∈ [0, 1].
                 ω → 1 means estimate a (prediction) is trusted more.
                 ω → 0 means estimate b (measurement) is trusted more.

    Mathematical guarantee:
        For ANY true cross-correlation ρ between a and b:
            E[(x_true - x_fused)(x_true - x_fused)^T] ≤ P_fused  (in PSD sense)

    Notes:
        - If Pa is much smaller than Pb, omega → 1 (trust prediction)
        - If Pb is much smaller than Pa, omega → 0 (trust measurement)
        - If Pa ≈ Pb, omega ≈ 0.5 (equal trust)
        - scipy.optimize.minimize_scalar with 'bounded' method uses Brent's algorithm,
          which converges superlinearly and typically needs 5-15 function evaluations
    """
    # Flatten to 1D for consistent math
    xa = np.asarray(xa, dtype=float).ravel()
    xb = np.asarray(xb, dtype=float).ravel()
    Pa = np.asarray(Pa, dtype=float)
    Pb = np.asarray(Pb, dtype=float)

    # Pre-compute inverses once (used in both objective and final fusion)
    # These must succeed — if Pa or Pb is singular, the input is invalid
    Pa_inv = np.linalg.inv(Pa)
    Pb_inv = np.linalg.inv(Pb)

    def objective(omega):
        """
        Compute trace or determinant of P_fused as a function of omega.
        This function is CONVEX on [0, 1] for both trace and det criteria
        (proven in [1]), so Brent's method will find the global minimum.
        """
        P_fused_inv = omega * Pa_inv + (1.0 - omega) * Pb_inv
        P_fused = np.linalg.inv(P_fused_inv)
        if minimize == 'trace':
            return np.trace(P_fused)
        else:
            return np.linalg.det(P_fused)

    # Find optimal omega using bounded scalar optimization (Brent's method)
    # bounds=[0, 1] because omega is the convex combination weight
    result = minimize_scalar(objective, bounds=(0.0, 1.0), method='bounded')
    omega = result.x

    # Compute the fused estimate at the optimal omega
    P_fused_inv = omega * Pa_inv + (1.0 - omega) * Pb_inv
    P_fused = np.linalg.inv(P_fused_inv)
    x_fused = P_fused @ (omega * Pa_inv @ xa + (1.0 - omega) * Pb_inv @ xb)

    return x_fused, P_fused, omega


def _ci_fuse_multi(estimates: List[Tuple[np.ndarray, np.ndarray]],
                   minimize: str = 'trace') -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse N ≥ 1 estimates using sequential pairwise CI.

    When multiple measurements arrive in the same frame, we fuse them
    pairwise-sequentially: fuse(m1, m2) → fuse(result, m3) → ...

    NOTE: Sequential pairwise CI is NOT the same as batch CI with N-way
    omega optimization. The batch version finds N weights ω_1..ω_N with
    Σω_i = 1, which is tighter but requires N-dimensional optimization.
    Sequential pairwise is simpler and still guarantees consistency.
    For our use case (typically 1-3 measurements per frame), the difference
    is negligible.

    Args:
        estimates: List of (x, P) tuples. Each x is shape (n,) or (n,1),
                   each P is shape (n, n).
        minimize: 'trace' or 'det' (passed to _ci_fuse)

    Returns:
        x_fused: Fused state, shape (n,)
        P_fused: Fused covariance, shape (n, n)
    """
    if len(estimates) == 0:
        raise ValueError("No estimates to fuse")
    if len(estimates) == 1:
        return estimates[0][0].ravel(), estimates[0][1]

    # Start with first estimate, sequentially fuse in each subsequent one
    x_acc, P_acc = estimates[0]
    x_acc = np.asarray(x_acc, dtype=float).ravel()
    P_acc = np.asarray(P_acc, dtype=float)

    for x_i, P_i in estimates[1:]:
        x_acc, P_acc, _ = _ci_fuse(x_acc, P_acc, x_i.ravel(), P_i, minimize)

    return x_acc, P_acc


# =============================================================================
# CI FILTER CLASS
# =============================================================================

class CovarianceIntersectionFilter(FilterBase):
    """
    Track-level fusion filter using Covariance Intersection for measurement updates.

    === HOW IT WORKS (per frame) ===

    1. PREDICTION: Same as ResizableKalman.
       - Apply motion model (CV/CA/CTRV) to propagate state forward
       - Grow covariance: P_pred = F·P·F^T + Q

    2. MEASUREMENT COLLECTION: Same as ResizableKalman.
       - Gather all matched measurements (from cooperating vehicles)
       - Apply trust scoring to filter/scale covariances

    3. MEASUREMENT UPDATE (THIS IS WHERE CI DIFFERS):
       Standard Kalman:
           K = P_pred · H^T · inv(H · P_pred · H^T + R)
           x_new = x_pred + K · (z - H · x_pred)
           P_new = (I - K·H) · P_pred
       → Assumes P_pred and R are uncorrelated. Overconfident if they're not.

       CI update:
           z_pred = H · x_pred             (prediction in measurement space)
           R_pred = H · P_pred · H^T       (prediction cov in measurement space)
           z_meas, R_meas = CI_fuse_multi(measurements)  (fuse all measurements)
           z_fused, P_fused = CI_fuse(z_pred, R_pred, z_meas, R_meas)
       → No assumption about correlation. Conservative but consistent.

    4. VELOCITY UPDATE: Same as ResizableKalman.
       - Estimate velocity from position change (exponential moving average)
       - This is a heuristic — CI only directly updates position/heading

    === WHY CI SHOULD OUTPERFORM IN GPEM EXPERIMENTS ===

    The standard Kalman's overconfidence problem causes it to weight
    all measurements nearly equally regardless of their covariance,
    because the Kalman gain saturates when P_pred shrinks.

    CI maintains a larger (more honest) prediction covariance, so:
    - Measurements with smaller R (closer objects, better GPEM estimates)
      get genuinely higher weight via the CI omega
    - The filter remains responsive to measurements instead of locking
      onto its own prediction
    - GPEM's distance-dependent covariance has a real effect on the fused position

    === CONSTRUCTOR ===

    Same signature as ResizableKalman for drop-in compatibility:
        CovarianceIntersectionFilter(time, x, y, fusion_mode,
                                      initial_width, initial_length,
                                      use_trust_scoring, passthrough_covariance)
    """

    def __init__(self, time, x, y, fusion_mode, initial_width=2.0,
                 initial_length=4.5, use_trust_scoring=True,
                 passthrough_covariance=False):
        """
        Initialize the CI filter.

        Args:
            time: Initial timestamp (seconds). Used as reference for elapsed time.
            x: Initial x position (meters, global frame).
            y: Initial y position (meters, global frame).
            fusion_mode: Motion model selection:
                0 = Constant Velocity (CV): state = [x, y, vx, vy] (4D)
                1 = Constant Acceleration (CA): state = [x, y, vx, vy, ax, ay] (6D)
                2 = Constant Turn Rate and Velocity (CTRV): state = [x, y, v, ψ, ψ̇] (5D)
            initial_width: Initial estimated object width (meters). Default 2.0.
            initial_length: Initial estimated object length (meters). Default 4.5.
            use_trust_scoring: If True, apply trust-based measurement filtering.
                Measurements from participants with trust ≤ 0.67 are dropped.
                Others are scaled: cov_adjusted = cov / trust². Default True.
            passthrough_covariance: If True, after CI update, override the position
                covariance with the raw measurement covariance. This preserves GPEM
                predictions in the output covariance. Default False.
        """
        # === MEASUREMENT LISTS (rebuilt every frame in addFrames()) ===
        # These lists hold one entry per matched measurement in the current frame
        self.localTrackersHList = []            # Observation matrix type per measurement
        self.localTrackersMeasurementList = []   # Position measurements [x,y] or [x,y,heading]
        self.localTrackersCovarianceList = []    # Measurement covariances (2x2 or 3x3)
        self.localTrackersIDList = []            # Tracker IDs (encodes participant + local ID)
        self.localTrackersExtraList = []         # [confidence, trust_score] per measurement
        self.localTrackersWidthList = []         # Width measurements for 1D Kalman
        self.localTrackersLengthList = []        # Length measurements for 1D Kalman
        self.localTrackersWidthStdList = []      # Width std (from GPEM) for measurement noise
        self.localTrackersLengthStdList = []     # Length std (from GPEM) for measurement noise

        # === CONFIGURATION FLAGS ===
        self.use_trust_scoring = use_trust_scoring
        self.passthrough_covariance = passthrough_covariance
        # Note: Unlike ResizableKalman, CI does NOT have a separate
        # precision_weighted_fusion flag. The CI algorithm naturally handles
        # multi-measurement fusion without the overconfidence issue.

        # === OUTPUT STATE (read by sensor_fusion.py after each fusion() call) ===
        # These are the public interface that GlobalTracked reads
        self.error_covariance = np.array([[1.0, 0.0], [0.0, 1.0]], dtype='float')  # 2x2 position cov
        self.d_covariance = np.array([[2.0, 0.0], [0.0, 2.0]], dtype='float')      # 2x2 velocity cov
        self.x = x              # Current estimated x position
        self.y = y              # Current estimated y position
        self.dx = 0.0           # Current estimated x velocity
        self.dy = 0.0           # Current estimated y velocity
        self.error_tracker_temp = []  # Per-sensor error records (for monitoring)
        self.trupercept_list = []     # Cooperative monitoring data

        # === INTERNAL STATE ===
        self.min_size = 0.5           # Minimum object size for matching
        self.last_measurement = time  # Time of last measurement (for track removal)
        self.last_update = time       # Time of last fusion (for elapsed time computation)
        self.idx = 0                  # Frame counter (0 = first frame, uses averaging not CI)
        self.time_until_removal = 0.3 # Seconds before track is dropped for no measurements
        self.fusion_mode = fusion_mode
        self.last_measurement_cov = None  # 2x2 position cov from last measurement batch

        # === 1D KALMAN FOR WIDTH/LENGTH ===
        # Object dimensions are tracked with simple scalar Kalman filters.
        # This is identical to ResizableKalman — dimensions don't benefit from CI
        # because they come from single measurements (no correlation issue).
        self.width = initial_width
        self.length = initial_length
        self.width_variance = 1.0           # Initial width uncertainty
        self.length_variance = 1.0          # Initial length uncertainty
        self.width_process_noise = 0.01     # Dimensions change slowly (parked vs moving)
        self.length_process_noise = 0.01
        # Note: measurement noise for width/length comes from GPEM-predicted stds

        # === FULL STATE VECTOR AND COVARIANCE ===
        # State dimension depends on fusion_mode:
        #   CV (mode 0):   [x, y, vx, vy]        → 4D
        #   CA (mode 1):   [x, y, vx, vy, ax, ay] → 6D
        #   CTRV (mode 2): [x, y, v, ψ, ψ̇]      → 5D
        if self.fusion_mode == 0:
            self.F_t_len = 4
        elif self.fusion_mode == 1:
            self.F_t_len = 6
        else:
            self.F_t_len = 5

        # Initial state: position from first detection, velocity/acceleration zero
        self.X_hat_t = np.zeros((self.F_t_len, 1), dtype='float')
        self.X_hat_t[0, 0] = x
        self.X_hat_t[1, 0] = y

        # Initial covariance: identity for position (will be overwritten on first frame),
        # large values for velocity (we have no velocity information yet)
        self.P_hat_t = np.identity(self.F_t_len)

        # === PROCESS NOISE TUNING ===
        # sigma_a = standard deviation of random acceleration (m/s²)
        # This controls how fast P_pred grows between frames:
        #   - Too small → filter trusts prediction too much, ignores measurements
        #   - Too large → filter ignores its own prediction, tracks are jittery
        # 2.0 m/s² corresponds to typical urban driving acceleration changes.
        # For CI, this value is LESS CRITICAL than for standard Kalman because
        # CI's omega optimization automatically balances prediction vs measurement
        # trust. With Kalman, bad Q leads to either overconfidence or jitter;
        # with CI, Q mainly affects prediction covariance magnitude, and omega
        # compensates for the rest.
        self.sigma_a = 2.0  # m/s² — acceleration noise standard deviation

        # Mode-specific initial covariance for non-position states
        if self.fusion_mode == 0:
            # CV: large initial velocity uncertainty (no velocity info yet)
            self.P_hat_t[2][2] = 10.0  # vx variance
            self.P_hat_t[3][3] = 10.0  # vy variance
        elif self.fusion_mode == 1:
            # CA: large velocity and moderate acceleration uncertainty
            self.P_hat_t[2][2] = 10.0  # vx variance
            self.P_hat_t[3][3] = 10.0  # vy variance
            self.P_hat_t[4][4] = 5.0   # ax variance
            self.P_hat_t[5][5] = 5.0   # ay variance
        else:
            # CTRV: large speed, moderate heading, small yaw rate uncertainty
            self.P_hat_t[2][2] = 10.0  # speed variance (m²/s²)
            self.P_hat_t[3][3] = 1.0   # heading variance (rad²) ≈ 57° std
            self.P_hat_t[4][4] = 0.5   # yaw rate variance (rad²/s²)

        # Placeholder Q (recomputed each frame based on actual elapsed time)
        self.Q_t = self._compute_Q(0.1)
        # Control input (not used — no external control input in cooperative perception)
        self.B_t = np.array([[0]] * self.F_t_len, dtype='float')
        self.U_t = 0

    # =========================================================================
    # PROCESS NOISE COMPUTATION
    # =========================================================================

    def _compute_Q(self, dt):
        """
        Compute the process noise covariance matrix Q for a given timestep dt.

        Uses the standard discrete white noise acceleration model:
            The continuous-time acceleration noise has PSD = sigma_a².
            After integration over dt, the discrete position/velocity noise is:

            For CV (mode 0):
                Q = sigma_a² * [[dt⁴/4,  0,     dt³/2, 0    ],
                                [0,      dt⁴/4, 0,     dt³/2],
                                [dt³/2,  0,     dt²,   0    ],
                                [0,      dt³/2, 0,     dt²  ]]

            For CA (mode 1): same pattern extended to acceleration states.

            For CTRV (mode 2): diagonal with velocity, heading, yaw rate noise.

        Why this model:
            - Physically motivated: acceleration changes randomly over time
            - dt-dependent: longer prediction intervals → more uncertainty growth
            - Cross-correlated: position noise is correlated with velocity noise
              because random acceleration affects both

        For CI, Q controls how fast the prediction covariance P_pred grows.
        If Q is too small, P_pred stays small and omega → 1 (trusting prediction),
        which reduces CI's benefit. If Q is too large, P_pred is huge and omega → 0
        (trusting measurement only), losing temporal smoothing. sigma_a = 2.0 m/s²
        is a good balance for urban driving at 10-50 km/h.

        Args:
            dt: Time step in seconds (typically 0.1s at 10 Hz SUMO sim rate)

        Returns:
            Q: Process noise covariance matrix, shape (F_t_len, F_t_len)
        """
        sigma_a = self.sigma_a
        sigma_a2 = sigma_a * sigma_a

        if self.fusion_mode == 0:
            # CV model: state = [x, y, vx, vy]
            dt2 = dt * dt
            dt3 = dt2 * dt
            dt4 = dt3 * dt
            return sigma_a2 * np.array([
                [dt4/4,  0,      dt3/2,  0    ],
                [0,      dt4/4,  0,      dt3/2],
                [dt3/2,  0,      dt2,    0    ],
                [0,      dt3/2,  0,      dt2  ]
            ], dtype='float')
        elif self.fusion_mode == 1:
            # CA model: state = [x, y, vx, vy, ax, ay]
            # Uses jerk noise model (sigma_a is actually sigma_jerk here)
            dt2 = dt * dt
            dt3 = dt2 * dt
            dt4 = dt3 * dt
            dt5 = dt4 * dt
            return sigma_a2 * np.array([
                [dt5/20, 0,      dt4/8,  0,      dt3/6,  0    ],
                [0,      dt5/20, 0,      dt4/8,  0,      dt3/6],
                [dt4/8,  0,      dt3/3,  0,      dt2/2,  0    ],
                [0,      dt4/8,  0,      dt3/3,  0,      dt2/2],
                [dt3/6,  0,      dt2/2,  0,      dt,     0    ],
                [0,      dt3/6,  0,      dt2/2,  0,      dt   ]
            ], dtype='float')
        else:
            # CTRV model: state = [x, y, v, ψ, ψ̇]
            # Process noise on velocity, heading, and yaw rate (independent)
            sigma_v = sigma_a * dt       # velocity change per timestep
            sigma_psi = 0.1 * dt         # heading drift per timestep
            sigma_psi_dot = 0.5 * dt     # yaw rate change per timestep
            return np.diag([sigma_v**2, sigma_v**2, sigma_v**2,
                           sigma_psi**2, sigma_psi_dot**2])

    # =========================================================================
    # STATE TRANSITION (MOTION MODEL)
    # =========================================================================

    def _build_F(self, elapsed):
        """
        Build the state transition matrix F for a given elapsed time.

        This is the linearized motion model used in the prediction step:
            X_pred = F · X_hat
            P_pred = F · P_hat · F^T + Q

        For CV and CA, F is exact (linear models).
        For CTRV, F is the Jacobian of the nonlinear CTRV equations evaluated
        at the current state — this is the Extended Kalman Filter linearization.

        Args:
            elapsed: Time since last update, in seconds.

        Returns:
            F: State transition matrix, shape (F_t_len, F_t_len)
        """
        if self.fusion_mode == 0:
            # Constant Velocity: x_new = x + vx·dt, y_new = y + vy·dt
            return np.array([[1, 0, elapsed, 0],
                            [0, 1, 0, elapsed],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype='float')
        elif self.fusion_mode == 1:
            # Constant Acceleration: x_new = x + vx·dt + 0.5·ax·dt²
            return np.array([
                [1, 0, elapsed, 0, elapsed**2, 0],
                [0, 1, 0, elapsed, 0, elapsed**2],
                [0, 0, 1, 0, elapsed, 0],
                [0, 0, 0, 1, 0, elapsed],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]], dtype='float')
        else:
            # CTRV: Nonlinear model, F is the Jacobian at current state
            # State = [x, y, v, ψ, ψ̇]
            # x_new = x + (v/ψ̇)·(sin(ψ+ψ̇·dt) - sin(ψ))
            # y_new = y + (v/ψ̇)·(-cos(ψ+ψ̇·dt) + cos(ψ))
            # Degenerates to straight line when ψ̇ ≈ 0
            v = float(self.X_hat_t[2, 0])
            phi = float(self.X_hat_t[3, 0])
            phi_dot = float(self.X_hat_t[4, 0])
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)

            if abs(phi_dot) < 1e-5:
                # Near-zero yaw rate: use Taylor expansion to avoid division by zero
                v_x = cos_phi * elapsed
                v_y = sin_phi * elapsed
                phi_x = -v * sin_phi * elapsed
                phi_y = v * cos_phi * elapsed
                phi_dot_x = -0.5 * v * sin_phi * elapsed**2
                phi_dot_y = 0.5 * v * cos_phi * elapsed**2
            else:
                phi_delta = phi + elapsed * phi_dot
                sin_phi_delta = math.sin(phi_delta)
                cos_phi_delta = math.cos(phi_delta)
                # Partial derivatives of position w.r.t. state variables
                v_x = (1.0 / phi_dot) * (-sin_phi + sin_phi_delta)
                v_y = (1.0 / phi_dot) * (cos_phi - cos_phi_delta)
                phi_x = (v / phi_dot) * (-cos_phi + cos_phi_delta)
                phi_y = (v / phi_dot) * (-sin_phi + sin_phi_delta)
                phi_dot_x = (v * elapsed / phi_dot) * cos_phi_delta - \
                            (v / phi_dot**2) * (-sin_phi + sin_phi_delta)
                phi_dot_y = (v * elapsed / phi_dot) * sin_phi_delta - \
                            (v / phi_dot**2) * (cos_phi - cos_phi_delta)

            return np.array([[1, 0, v_x, phi_x, phi_dot_x],
                            [0, 1, v_y, phi_y, phi_dot_y],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 1, elapsed],
                            [0, 0, 0, 0, 1]], dtype='float')

    # =========================================================================
    # OBSERVATION MODEL
    # =========================================================================

    def h_t(self, h_t_type):
        """
        Observation matrix H: maps full state to measurement space.

        For type 0 (standard measurements):
            CV/CA:  H extracts [x, y] from state (we observe position only)
            CTRV:   H extracts [x, y, ψ] from state (we observe position + heading)

        Args:
            h_t_type: 0 for standard observation, other values reserved for radar.

        Returns:
            H: Observation matrix, shape (m, F_t_len) where m = measurement dimension
        """
        if h_t_type == 0:
            if self.fusion_mode == 0:
                # CV: observe [x, y] from [x, y, vx, vy]
                return np.array([[1., 0., 0., 0.],
                                [0., 1., 0., 0.]], dtype='float')
            elif self.fusion_mode == 1:
                # CA: observe [x, y] from [x, y, vx, vy, ax, ay]
                return np.array([[1., 0., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0., 0.]], dtype='float')
            elif self.fusion_mode == 2:
                # CTRV: observe [x, y, ψ] from [x, y, v, ψ, ψ̇]
                return np.array([[1., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0.],
                                [0., 0., 0., 1., 0.]], dtype='float')
        # Fallback: zero observation (no information from this measurement type)
        return np.zeros((2, self.F_t_len), dtype='float')

    # =========================================================================
    # MEASUREMENT INGESTION
    # =========================================================================

    def addFrames(self, measurement_list):
        """
        Rebuild measurement lists from the current frame's match list.

        Called at the start of each fusion() call. Clears old measurements
        and populates fresh ones from the matched detections.

        For CTRV mode, constructs 3x3 covariance (adding heading variance)
        from the 2x2 position covariance in each detection.

        Args:
            measurement_list: List of MatchClass objects (from sensor_fusion.py).
                Each has: id, x, y, covariance, angle, confidence, trust_score,
                          width, length, width_std, length_std
        """
        self.localTrackersCovarianceList = []
        self.localTrackersMeasurementList = []
        self.localTrackersHList = []
        self.localTrackersIDList = []
        self.localTrackersExtraList = []
        self.localTrackersWidthList = []
        self.localTrackersLengthList = []
        self.localTrackersWidthStdList = []
        self.localTrackersLengthStdList = []

        for match in measurement_list:
            self.localTrackersIDList.append(match.id)

            if self.fusion_mode == 2:
                # CTRV: measurement is [x, y, heading]
                self.localTrackersMeasurementList.append(
                    np.array([match.x, match.y, match.angle]))
                # Expand 2x2 position cov to 3x3 with heading variance
                match_cov = np.asarray(match.covariance, dtype='float')
                cov_3x3 = np.eye(3, dtype='float')
                if match_cov.ndim == 2 and match_cov.shape[0] >= 2 and match_cov.shape[1] >= 2:
                    cov_3x3[0:2, 0:2] = match_cov[0:2, 0:2]
                else:
                    cov_3x3[0:2, 0:2] = np.eye(2, dtype='float') * 0.25  # Fallback
                cov_3x3[2, 2] = 0.1  # ~18° std heading measurement noise
                self.localTrackersCovarianceList.append(cov_3x3)
            else:
                # CV/CA: measurement is [x, y]
                self.localTrackersMeasurementList.append(
                    np.array([match.x, match.y]))
                self.localTrackersCovarianceList.append(match.covariance)

            self.localTrackersHList.append(0)
            self.localTrackersExtraList.append(
                [match.confidence, match.trust_score])
            self.localTrackersWidthList.append(match.width)
            self.localTrackersLengthList.append(match.length)
            self.localTrackersWidthStdList.append(match.width_std)
            self.localTrackersLengthStdList.append(match.length_std)

    # =========================================================================
    # FIRST-FRAME INITIALIZATION
    # =========================================================================

    def averageMeasurementsFirstFrame(self):
        """
        Precision-weighted average for first frame initialization.

        On the very first frame, we have no prior state to fuse against,
        so we just compute a precision-weighted average of all measurements.
        This is identical to ResizableKalman's first-frame behavior.

        Returns:
            (pos, cov): Fused position and 2x2 covariance
        """
        if len(self.localTrackersCovarianceList) == 1:
            return self.localTrackersMeasurementList[0], self.localTrackersCovarianceList[0]

        precision_sum = np.zeros((2, 2))
        weighted_pos = np.zeros(2)
        for meas, cov in zip(self.localTrackersMeasurementList,
                             self.localTrackersCovarianceList):
            P_inv = np.linalg.inv(cov[:2, :2])
            precision_sum += P_inv
            weighted_pos += P_inv @ meas[:2]

        fused_cov = np.linalg.inv(precision_sum)
        fused_pos = fused_cov @ weighted_pos
        return fused_pos, fused_cov

    # =========================================================================
    # MAIN FUSION (called once per frame per tracked object)
    # =========================================================================

    def fusion(self, measurement_list, time, monitor,
               participant_trust_scores: Optional[Dict[int, float]] = None):
        """
        Main fusion method: predict → collect measurements → CI update.

        This is called by sensor_fusion.py (GlobalTracked.update) once per frame
        for each tracked object. After this call, the public attributes
        (x, y, dx, dy, error_covariance, etc.) are updated.

        Args:
            measurement_list: List of MatchClass objects matched to this track.
            time: Current simulation timestamp in seconds.
            monitor: If True, compute per-sensor error stats and trupercept data.
            participant_trust_scores: Optional dict {participant_id: trust_score}.
                Used to filter out untrustworthy participants and scale their
                covariances. Trust score ∈ (0, 2], with 1.0 = nominal trust.

        Flow:
            Frame 0: Simple average initialization (no prediction yet)
            Frame 1+: Predict → Filter measurements → CI fuse → Update velocity
        """
        self.addFrames(measurement_list)

        if self.idx == 0:
            # === FIRST FRAME: Initialize from measurements ===
            # No prior state exists, so we just average all measurements.
            # CI requires two estimates (prediction + measurement) so we can't
            # use it until frame 1.
            pos, cov = self.averageMeasurementsFirstFrame()
            self.x = pos[0]
            self.y = pos[1]
            self.error_covariance = np.array([
                [cov[0, 0], cov[0, 1]],
                [cov[1, 0], cov[1, 1]]
            ], dtype='float')

            # Initialize full state vector based on fusion mode
            if self.fusion_mode == 0:
                self.X_hat_t = np.array(
                    [[self.x], [self.y], [0], [0]], dtype='float')
            elif self.fusion_mode == 1:
                self.X_hat_t = np.array(
                    [[self.x], [self.y], [0], [0], [0], [0]], dtype='float')
            else:
                # CTRV: use heading from measurement if available
                init_heading = pos[2] if len(pos) > 2 else 0.0
                self.X_hat_t = np.array(
                    [[self.x], [self.y], [0], [init_heading], [0]], dtype='float')
                if cov.shape[0] > 2:
                    self.P_hat_t[3][3] = cov[2, 2]

            # Seed position covariance from measurement
            self.P_hat_t[0][0] = self.error_covariance[0][0]
            self.P_hat_t[0][1] = self.error_covariance[0][1]
            self.P_hat_t[1][0] = self.error_covariance[1][0]
            self.P_hat_t[1][1] = self.error_covariance[1][1]
            self.last_update = time
            self.dx = 0.0
            self.dy = 0.0
            self.idx += 1
        else:
            # === FRAMES 1+: Predict → CI Update ===

            # --- STEP 1: PREDICTION ---
            # Propagate state and covariance forward using motion model.
            # This is identical to ResizableKalman — the innovation is in the update.
            elapsed = time - self.last_update
            if elapsed <= 0.0:
                elapsed = 0.1  # Fallback for duplicate timestamps

            F_t = self._build_F(elapsed)
            self.Q_t = self._compute_Q(elapsed)
            # X_pred = F · X_hat + B · U  (B·U = 0, no control input)
            # P_pred = F · P_hat · F^T + Q
            self.X_hat_t, self.P_hat_t = utils.kalman_prediction(
                self.X_hat_t, self.P_hat_t, F_t, self.B_t, self.U_t, self.Q_t)

            # --- STEP 2: COLLECT VALID MEASUREMENTS ---
            # Filter out NaN/Inf, apply trust scoring, add minimum covariance floor
            valid_measurements = []
            valid_covariances = []

            if len(self.localTrackersMeasurementList) != 0:
                for mu, cov, h_t_type, tracker_id in zip(
                        self.localTrackersMeasurementList,
                        self.localTrackersCovarianceList,
                        self.localTrackersHList,
                        self.localTrackersIDList):
                    # Skip corrupt measurements
                    if np.any(np.isnan(mu)) or np.any(np.isnan(cov)):
                        continue
                    if np.any(np.isinf(mu)) or np.any(np.isinf(cov)):
                        continue

                    # Trust scoring: filter bad participants, scale covariance
                    # participant_id is encoded as tracker_id // max_id
                    if self.use_trust_scoring and participant_trust_scores is not None:
                        participant_id = tracker_id // max_id
                        trust_score = participant_trust_scores.get(participant_id, 1.0)
                        if np.isnan(trust_score) or np.isinf(trust_score):
                            trust_score = 1.0
                        # Drop measurements from participants with trust ≤ 0.67
                        if trust_score <= 0.67:
                            continue
                        # Scale covariance inversely with trust²
                        # High trust → smaller cov → more trusted in CI fusion
                        clamped_score = max(0.67, min(2.0, trust_score))
                        adjusted_cov = cov / (clamped_score ** 2)
                    else:
                        adjusted_cov = cov

                    # Add tiny diagonal to prevent exactly singular covariances
                    # (CI requires invertible covariance matrices)
                    min_cov = 1e-11
                    adjusted_cov = adjusted_cov + min_cov * np.eye(adjusted_cov.shape[0])

                    valid_measurements.append(mu)
                    valid_covariances.append(adjusted_cov)

                added = len(valid_measurements)

                if added > 0:
                    # --- STEP 3: CI UPDATE ---
                    # This is the KEY DIFFERENCE from ResizableKalman.
                    #
                    # Instead of: K = P·H^T·(H·P·H^T + R)^-1  (assumes P and R uncorrelated)
                    # We use: CI-fuse(prediction_in_meas_space, combined_measurement)
                    #
                    # Why project prediction to measurement space?
                    # CI fuses two estimates of the SAME quantity. The prediction is in
                    # full state space [x,y,vx,vy,...] but measurements are only [x,y].
                    # We project the prediction to measurement space so both inputs to CI
                    # represent the same quantity (position, or position+heading for CTRV).

                    # Project prediction to measurement space
                    H = self.h_t(0)
                    z_pred = (H @ self.X_hat_t).ravel()  # predicted measurement (e.g. [x, y])
                    R_pred = H @ self.P_hat_t @ H.T      # predicted measurement covariance

                    # Combine all measurements using CI (handles unknown inter-measurement correlation)
                    # In cooperative perception, different vehicles' measurements may be correlated
                    # through shared localizer errors, so CI is appropriate here too.
                    meas_dim = len(z_pred)
                    meas_estimates = []
                    for mu, cov in zip(valid_measurements, valid_covariances):
                        meas_estimates.append((mu[:meas_dim], cov[:meas_dim, :meas_dim]))

                    if len(meas_estimates) == 1:
                        # Single measurement: no need to fuse measurements together
                        z_meas, R_meas = meas_estimates[0]
                    else:
                        # Multiple measurements: CI-fuse them into one combined measurement
                        z_meas, R_meas = _ci_fuse_multi(meas_estimates)

                    # CI-fuse prediction with combined measurement
                    # omega → 1: prediction trusted more (prediction cov smaller)
                    # omega → 0: measurement trusted more (measurement cov smaller)
                    z_fused, P_fused_meas, omega = _ci_fuse(
                        z_pred, R_pred, z_meas, R_meas)

                    # --- FULL STATE UPDATE from CI result ---
                    # We need to update the full state (including velocity and
                    # cross-covariances) based on the CI-fused position.
                    #
                    # Approach: Use the MEASUREMENT (not CI result) as the Kalman
                    # update input, with the measurement covariance R_meas. This is
                    # a standard Kalman update that properly corrects all state
                    # components and cross-covariances. The CI influence comes
                    # through R_meas: smaller R (from GPEM) → larger Kalman gain →
                    # measurement pulls estimate more.
                    #
                    # Why not use CI's P_fused as R? Because that would double-count
                    # the prediction: CI already balanced pred vs meas, then Kalman
                    # would re-balance the CI result against the same prediction.
                    #
                    # The CI fusion was used to determine the measurement weighting
                    # when multiple measurements exist. For the final state update,
                    # we use the CI-combined measurement z_meas with R_meas.
                    Z_meas = z_meas.reshape(-1, 1)
                    self.X_hat_t, self.P_hat_t = utils.kalman_update(
                        self.X_hat_t, self.P_hat_t, Z_meas, R_meas, H)

                    # Store the fused position for velocity estimation
                    z_fused_pos = np.array([self.X_hat_t[0, 0], self.X_hat_t[1, 0]])

                    # Store measurement covariance for passthrough mode
                    self.last_measurement_cov = R_meas[:2, :2].copy() if R_meas.shape[0] >= 2 else R_meas.copy()

                    # Passthrough mode: override CI-fused covariance with raw measurement cov
                    # This ensures the output covariance reflects GPEM predictions directly
                    # rather than the (more conservative) CI-fused covariance.
                    if self.passthrough_covariance and self.last_measurement_cov is not None:
                        self.P_hat_t[0:2, 0:2] = self.last_measurement_cov

                    # --- STEP 4: VELOCITY UPDATE (supplementary) ---
                    # The Kalman update above already updates velocity through the
                    # gain K. But for additional velocity estimation from position
                    # change (useful when Kalman velocity estimate lags), we blend
                    # in a position-based velocity estimate with small weight.
                    if hasattr(self, '_last_fused_pos') and self._last_fused_pos is not None:
                        if elapsed > 0.001:
                            vel_x = (z_fused_pos[0] - self._last_fused_pos[0]) / elapsed
                            vel_y = (z_fused_pos[1] - self._last_fused_pos[1]) / elapsed
                            alpha = 0.15  # Smaller weight since Kalman already updates velocity
                            if self.fusion_mode == 0 or self.fusion_mode == 1:
                                self.X_hat_t[2, 0] = alpha * vel_x + (1 - alpha) * self.X_hat_t[2, 0]
                                self.X_hat_t[3, 0] = alpha * vel_y + (1 - alpha) * self.X_hat_t[3, 0]
                            elif self.fusion_mode == 2:
                                speed = math.hypot(vel_x, vel_y)
                                self.X_hat_t[2, 0] = alpha * speed + (1 - alpha) * self.X_hat_t[2, 0]
                    self._last_fused_pos = z_fused_pos

            # --- NO MEASUREMENTS: prediction-only (zero-gain Kalman update) ---
            if len(self.localTrackersMeasurementList) == 0 or len(valid_measurements) == 0:
                # No measurements this frame. Covariance grows via prediction only.
                # Use zero-gain Kalman update (H=0) to keep state vector consistent.
                if self.fusion_mode == 0:
                    nothing_cov = np.eye(2, dtype='float')
                    measure = np.zeros(2, dtype='float')
                    nothing_Ht = np.zeros((2, 4), dtype='float')
                elif self.fusion_mode == 1:
                    nothing_cov = np.eye(2, dtype='float')
                    measure = np.zeros(2, dtype='float')
                    nothing_Ht = np.zeros((2, 6), dtype='float')
                elif self.fusion_mode == 2:
                    nothing_cov = np.eye(3, dtype='float')
                    measure = np.zeros(3, dtype='float')
                    nothing_Ht = np.zeros((3, 5), dtype='float')

                Z_t = measure.reshape(-1, 1)
                self.X_hat_t, self.P_hat_t = utils.kalman_update(
                    self.X_hat_t, self.P_hat_t, Z_t, nothing_cov, nothing_Ht)

            # --- MONITORING: compute per-sensor error statistics ---
            self.error_tracker_temp = []
            length = len(self.localTrackersIDList)
            if monitor:
                if self.P_hat_t[0][0] != 0.0 and self.P_hat_t[0][1] != 0.0:
                    global_error_x, global_error_y, global_error_angle = utils.ellipsify(
                        [[self.P_hat_t[0][0], self.P_hat_t[0][1]],
                         [self.P_hat_t[1][0], self.P_hat_t[1][1]]], 1.0)
                else:
                    global_error_x = 0.0
                    global_error_y = 0.0
                    global_error_angle = 0.0

                for id_val, mu, cov, h_t_type in zip(
                        self.localTrackersIDList,
                        self.localTrackersMeasurementList,
                        self.localTrackersCovarianceList,
                        self.localTrackersHList):
                    Z_t = mu.transpose()
                    Z_t = Z_t.reshape(Z_t.shape[0], -1)
                    y_t_temp = Z_t - self.h_t(h_t_type).dot(self.X_hat_t)
                    location_error = math.hypot(y_t_temp[0], y_t_temp[1])
                    expected_a, expected_b, expected_angle = utils.ellipsify(cov, 1.0)
                    expected_x = utils.calculateRadiusAtAngle(
                        expected_a, expected_b, expected_angle, math.radians(0))
                    expected_y = utils.calculateRadiusAtAngle(
                        expected_a, expected_b, expected_angle, math.radians(90))
                    expected_location_error = math.hypot(expected_x, expected_y) - \
                        math.hypot(global_error_x, global_error_y)
                    if expected_location_error <= 0.0 or math.isnan(expected_location_error):
                        expected_location_error = max(0.01, math.hypot(expected_x, expected_y))
                    location_error_std = location_error / expected_location_error
                    if math.isnan(location_error_std) or math.isinf(location_error_std):
                        continue
                    self.error_tracker_temp.append([id_val, location_error_std, length])

                trupercept_list = []
                for id_test, mu_test, confidence_test in zip(
                        self.localTrackersIDList,
                        self.localTrackersMeasurementList,
                        self.localTrackersExtraList):
                    trupercept_list.append([id_test, confidence_test[0]])
                self.trupercept_list = trupercept_list

            # === UPDATE PUBLIC STATE ===
            self.last_update = time
            self.x = self.X_hat_t[0][0]
            self.y = self.X_hat_t[1][0]
            if self.fusion_mode == 2:
                # CTRV: velocity components from speed and heading
                self.dx = self.X_hat_t[2][0] * math.cos(self.X_hat_t[3][0])
                self.dy = self.X_hat_t[2][0] * math.sin(self.X_hat_t[3][0])
            else:
                # CV/CA: velocity components are directly in state
                self.dx = self.X_hat_t[2][0]
                self.dy = self.X_hat_t[3][0]
            self.idx += 1

            # Extract 2x2 position and velocity covariance blocks for external consumers
            if self.P_hat_t[0][0] != 0.0 or self.P_hat_t[0][1] != 0.0:
                self.error_covariance = np.array(
                    [[self.P_hat_t[0][0], self.P_hat_t[0][1]],
                     [self.P_hat_t[1][0], self.P_hat_t[1][1]]], dtype='float')
                self.d_covariance = np.array(
                    [[self.P_hat_t[2][2], self.P_hat_t[2][3]],
                     [self.P_hat_t[3][2], self.P_hat_t[3][3]]], dtype='float')
            else:
                self.error_covariance = np.array([[1.0, 0.0], [0.0, 1.0]], dtype='float')
                self.d_covariance = np.array([[2.0, 0.0], [0.0, 2.0]], dtype='float')

            # === 1D KALMAN FOR OBJECT DIMENSIONS ===
            # Width and length are tracked with simple scalar Kalman filters.
            # Process noise grows each frame; measurement noise from GPEM std.
            self.width_variance += self.width_process_noise
            self.length_variance += self.length_process_noise

            if len(self.localTrackersWidthList) > 0:
                for w_meas, l_meas, w_std, l_std in zip(
                        self.localTrackersWidthList, self.localTrackersLengthList,
                        self.localTrackersWidthStdList, self.localTrackersLengthStdList):
                    # Width update
                    w_noise = max(1e-4, w_std**2)  # Measurement variance from GPEM
                    width_gain = self.width_variance / (self.width_variance + w_noise)
                    self.width = self.width + width_gain * (w_meas - self.width)
                    self.width_variance = (1 - width_gain) * self.width_variance

                    # Length update
                    l_noise = max(1e-4, l_std**2)
                    length_gain = self.length_variance / (self.length_variance + l_noise)
                    self.length = self.length + length_gain * (l_meas - self.length)
                    self.length_variance = (1 - length_gain) * self.length_variance

    # =========================================================================
    # PREDICTION METHODS (for matching and track management)
    # =========================================================================

    def getKalmanPred(self, time):
        """
        Predict position and uncertainty ellipse at a future time.

        Used by sensor_fusion.py for matching new detections to existing tracks.
        The predicted position and uncertainty ellipse define the gating region.

        Args:
            time: Future timestamp to predict to.

        Returns:
            (x, y, a, b, phi): Predicted position and uncertainty ellipse params
                a, b = semi-axes of 1-sigma ellipse
                phi = rotation angle of ellipse
        """
        elapsed = time - self.last_update
        if elapsed < 0:
            predicted_x = self.X_hat_t[0][0]
            predicted_y = self.X_hat_t[1][0]
            predicted_P = self.P_hat_t
        else:
            F_pred = self._build_F(elapsed)

            if self.fusion_mode == 2:
                # CTRV: use nonlinear position prediction (not just F·X)
                v = float(self.X_hat_t[2, 0])
                psi = float(self.X_hat_t[3, 0])
                psi_dot = float(self.X_hat_t[4, 0])
                if abs(psi_dot) < 1e-5:
                    predicted_x = self.X_hat_t[0, 0] + v * math.cos(psi) * elapsed
                    predicted_y = self.X_hat_t[1, 0] + v * math.sin(psi) * elapsed
                else:
                    psi_new = psi + psi_dot * elapsed
                    predicted_x = self.X_hat_t[0, 0] + (v / psi_dot) * (math.sin(psi_new) - math.sin(psi))
                    predicted_y = self.X_hat_t[1, 0] + (v / psi_dot) * (-math.cos(psi_new) + math.cos(psi))
            else:
                # CV/CA: linear prediction X_pred = F · X
                predicted_X_hat, _ = utils.kalman_prediction(
                    self.X_hat_t, self.P_hat_t, F_pred, self.B_t, self.U_t, self.Q_t)
                predicted_x = predicted_X_hat[0][0]
                predicted_y = predicted_X_hat[1][0]

            # Covariance prediction: P_pred = F · P · F^T + Q
            _, predicted_P = utils.kalman_prediction(
                self.X_hat_t, self.P_hat_t, F_pred, self.B_t, self.U_t, self.Q_t)

        # Convert 2x2 position covariance to ellipse parameters
        a, b, phi = utils.ellipsify(predicted_P[0:2, 0:2], 1.0)
        return predicted_x, predicted_y, a, b, phi

    def getKalmanPredWithCovariance(self, time):
        """
        Predict position and full state covariance at a future time.

        Used for Mahalanobis distance computation in Hungarian matching.
        Returns the full state covariance (not just position block).

        Args:
            time: Future timestamp to predict to.

        Returns:
            (x, y, P): Predicted position and full state covariance matrix
        """
        elapsed = time - self.last_update
        if elapsed < 0:
            predicted_x = self.X_hat_t[0][0]
            predicted_y = self.X_hat_t[1][0]
            predicted_P = self.P_hat_t
        else:
            F_pred = self._build_F(elapsed)

            if self.fusion_mode == 2:
                v = float(self.X_hat_t[2, 0])
                psi = float(self.X_hat_t[3, 0])
                psi_dot = float(self.X_hat_t[4, 0])
                if abs(psi_dot) < 1e-5:
                    predicted_x = self.X_hat_t[0, 0] + v * math.cos(psi) * elapsed
                    predicted_y = self.X_hat_t[1, 0] + v * math.sin(psi) * elapsed
                else:
                    psi_new = psi + psi_dot * elapsed
                    predicted_x = self.X_hat_t[0, 0] + (v / psi_dot) * (math.sin(psi_new) - math.sin(psi))
                    predicted_y = self.X_hat_t[1, 0] + (v / psi_dot) * (-math.cos(psi_new) + math.cos(psi))
            else:
                predicted_X_hat, _ = utils.kalman_prediction(
                    self.X_hat_t, self.P_hat_t, F_pred, self.B_t, self.U_t, self.Q_t)
                predicted_x = predicted_X_hat[0][0]
                predicted_y = predicted_X_hat[1][0]

            _, predicted_P = utils.kalman_prediction(
                self.X_hat_t, self.P_hat_t, F_pred, self.B_t, self.U_t, self.Q_t)

        return predicted_x, predicted_y, predicted_P
