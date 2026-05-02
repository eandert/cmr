import math
import numpy as np
from typing import Dict, Optional

import utils
from filters.base import FilterBase

# Shared constant: must match the value in sensor_fusion.py
max_id = 10000


class ResizableKalman(FilterBase):
    def __init__(self, time, x, y, fusion_mode, initial_width=2.0, initial_length=4.5, use_trust_scoring=True, passthrough_covariance=False):
        # This list will take
        self.localTrackersHList = []
        self.localTrackersMeasurementList = []
        self.localTrackersCovarianceList = []
        self.localTrackersIDList = []
        # Width and length measurements for 1D Kalman tracking
        self.localTrackersWidthList = []
        self.localTrackersLengthList = []
        self.localTrackersWidthStdList = []
        self.localTrackersLengthStdList = []

        # Flag to enable/disable trust-based filtering and covariance scaling
        self.use_trust_scoring = use_trust_scoring

        # Flag to pass through measurement covariance directly instead of Kalman-filtered covariance
        # When True, the position covariance block is overwritten with the last measurement covariance
        # after each update. This ensures GPEM predictions flow through without Kalman filtering.
        self.passthrough_covariance = passthrough_covariance

        # Flag to use precision-weighted mean for position fusion instead of sequential Kalman updates
        # When True: position = precision-weighted mean of all measurements, covariance = sum of precisions inverted
        # Kalman is still used for velocity estimation via motion model
        self.precision_weighted_fusion = passthrough_covariance  # Enable when passthrough is enabled

        # Init the covariance to some value
        self.error_covariance = np.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype='float')
        self.d_covariance = np.array([[2.0, 0.0], [0.0, 2.0]], dtype='float')

        # Store the first value for matching purposes
        self.x = x
        self.y = y

        # Store the covariance vs. expected
        self.error_tracker_temp = []

        # Arbitrary min tracking size so we are not too small to match to
        self.min_size = .5

        # Process varaition guess
        process_variation = 0.1

        # Track the time of the last track
        self.last_measurement = time
        self.last_update = time # Initialize last_update here

        # Track the number of times this Kalman filter has been used
        self.idx = 0

        # The amount of time before a tracker is removed from the list
        self.time_until_removal = .3

        # Store the fusion mode
        self.fusion_mode = fusion_mode

        # Trupercept stuff
        self.trupercept_list = []

        process_noise_estimate = 0.1

        # 1D Kalman filters for width and length
        # State: [value], Process noise: low (dimensions don't change quickly)
        # Measurement noise: moderate (sensor noise in dimension estimation)
        self.width = initial_width
        self.length = initial_length
        self.width_variance = 1.0  # Initial uncertainty
        self.length_variance = 1.0
        self.width_process_noise = 0.01  # Dimensions change slowly
        self.length_process_noise = 0.01
        self.dimension_measurement_noise = 0.5  # Measurement uncertainty

        # Set up the Kalman filter
        # Initial State cov
        if self.fusion_mode == 0:
            # Set up the Kalman filter
            self.F_t_len = 4
        elif self.fusion_mode == 1:
            # Setup for x_hat = x + dx + dxdx,  y_hat = y + dy + dydy
            self.F_t_len = 6
        else:
            # model from https://journals.sagepub.com/doi/abs/10.1177/0959651820975523
            self.F_t_len = 5

        # Initialize Kalman state with the initial position
        # This ensures getKalmanPred() returns meaningful values before first fusion call
        self.X_hat_t = np.zeros((self.F_t_len, 1), dtype='float')
        self.X_hat_t[0, 0] = x  # Initial x position
        self.X_hat_t[1, 0] = y  # Initial y position
        # Velocity and other states remain zero until first fusion
        self.P_hat_t = np.identity(self.F_t_len)

        # Store acceleration noise std dev for Q matrix computation (updated each timestep)
        from filters.filter_config import get_sigma_a
        self.sigma_a = get_sigma_a("ekf")

        # Adaptive process noise: store last measurement covariance for position block of Q
        # This makes Q adapt to GPEM predictions - if sensor says uncertainty is high, Q is high
        self.last_measurement_cov = None  # 2x2 position covariance from last measurement

        # Prediction cache: getKalmanPredWithCovariance stores its result here so that
        # the immediately-following fusion() call can skip re-running the identical
        # F computation and kalman_prediction matrix multiply.
        self._pred_cache_time = None
        self._pred_cache_X = None
        self._pred_cache_P = None

        # Now apply mode-specific initializations
        if self.fusion_mode == 0:
            # Setup for x_hat = x + dx,  y_hat = y + dy
            # Initial State cov - velocity is unknown
            self.P_hat_t[2][2] = 10.0  # large initial velocity uncertainty
            self.P_hat_t[3][3] = 10.0
            # Q_t will be computed each timestep in _compute_Q
            self.Q_t = self._compute_Q(0.1)  # placeholder with dt=0.1
            # Control matrix
            self.B_t = np.array([[0]] * self.F_t_len, dtype='float')
            # Control vector
            self.U_t = 0
        elif self.fusion_mode == 1:
            # Setup for x_hat = x + dx + dxdx,  y_hat = y + dy + dydy
            self.P_hat_t[2][2] = 10.0  # velocity uncertainty
            self.P_hat_t[3][3] = 10.0
            self.P_hat_t[4][4] = 5.0   # acceleration uncertainty
            self.P_hat_t[5][5] = 5.0
            # Q_t will be computed each timestep in _compute_Q
            self.Q_t = self._compute_Q(0.1)
            # Control matrix
            self.B_t = np.array([[0]] * self.F_t_len, dtype='float')
            # Control vector
            self.U_t = 0
        else:
            # model from https://journals.sagepub.com/doi/abs/10.1177/0959651820975523
            self.P_hat_t[2][2] = 10.0  # velocity uncertainty
            self.P_hat_t[3][3] = 1.0   # heading uncertainty (rad)
            self.P_hat_t[4][4] = 0.5   # yaw rate uncertainty
            # Q_t will be computed each timestep in _compute_Q
            self.Q_t = self._compute_Q(0.1)
            # Control matrix
            self.B_t = np.array([[0]] * self.F_t_len, dtype='float')
            # Control vector
            self.U_t = 0

    def _compute_Q(self, dt):
        """
        Compute process noise matrix Q for given timestep dt.
        Standard discrete white noise acceleration model.
        """
        sigma_a = self.sigma_a
        sigma_a2 = sigma_a * sigma_a

        if self.fusion_mode == 0:
            # State: [x, y, vx, vy] - constant velocity with acceleration noise
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
            # State: [x, y, vx, vy, ax, ay] - constant acceleration with jerk noise
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
            # CTRV model: [x, y, v, psi, psi_dot]
            # Process noise on velocity and yaw rate
            sigma_v = sigma_a * dt  # velocity change std
            sigma_psi = 0.1 * dt    # heading change std
            sigma_psi_dot = 0.5 * dt  # yaw rate change std
            return np.diag([sigma_v**2, sigma_v**2, sigma_v**2, sigma_psi**2, sigma_psi_dot**2])

    def addFrames(self, measurement_list):
        # Rebuild the lists every time because measurements come and go
        self.localTrackersCovarianceList = []
        self.localTrackersMeasurementList = []
        self.localTrackersHList = []
        self.localTrackersIDList = []
        self.localTrackersExtraList = []
        self.localTrackersWidthList = []
        self.localTrackersLengthList = []
        self.localTrackersWidthStdList = []
        self.localTrackersLengthStdList = []
        # Check if there are more sensors in the area that have not been added
        for match in measurement_list:
            self.localTrackersIDList.append(match.id)

            if self.fusion_mode == 2:
                # CTRV: observe [x, y, heading] - angle is measured
                self.localTrackersMeasurementList.append(
                    np.array([match.x, match.y, match.angle]))
                # Expand 2x2 position covariance to 3x3 with heading variance
                angle_variance = match.yaw_variance if match.yaw_variance is not None else 0.1
                cov_3x3 = np.eye(3, dtype='float')
                # Ensure we extract 2x2 from match.covariance even if it's larger
                match_cov = np.asarray(match.covariance, dtype='float')
                if match_cov.ndim == 2 and match_cov.shape[0] >= 2 and match_cov.shape[1] >= 2:
                    cov_3x3[0:2, 0:2] = match_cov[0:2, 0:2]
                else:
                    cov_3x3[0:2, 0:2] = np.eye(2, dtype='float') * 0.25  # Fallback
                cov_3x3[2, 2] = angle_variance
                self.localTrackersCovarianceList.append(cov_3x3)
            else:
                # CV models: observe [x, y] only
                self.localTrackersMeasurementList.append(
                    np.array([match.x, match.y]))
                self.localTrackersCovarianceList.append(match.covariance)

            self.localTrackersHList.append(0)
            self.localTrackersExtraList.append(
                [match.confidence, match.trust_score])
            # Add width and length measurements and their predicted stds
            self.localTrackersWidthList.append(match.width)
            self.localTrackersLengthList.append(match.length)
            self.localTrackersWidthStdList.append(match.width_std)
            self.localTrackersLengthStdList.append(match.length_std)

    def h_t(self, h_t_type):
        if h_t_type == 0:
            if self.fusion_mode == 0:
                # CV model: observe [x, y] from state [x, y, vx, vy]
                return np.array([[1., 0., 0., 0.],
                                [0., 1., 0., 0.]], dtype='float')
            elif self.fusion_mode == 1:
                # CA model: observe [x, y] from state [x, y, vx, vy, ax, ay]
                return np.array([[1., 0., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0., 0.]], dtype='float')
            elif self.fusion_mode == 2:
                # CTRV model: observe [x, y, ψ] from state [x, y, v, ψ, ψ̇]
                return np.array([[1., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0.],
                                [0., 0., 0., 1., 0.]], dtype='float')
        else:
            # Radar observation model (not currently used)
            return np.array([[0, 0., 0., 0.],
                            [0., 0, 0., 0.]], dtype='float')

    def averageMeasurementsFirstFrame(self):
        if len(self.localTrackersCovarianceList) == 1:
            return self.localTrackersMeasurementList[0], self.localTrackersCovarianceList[0]

        # Batch precision-weighted fusion: P_fused = (Σ P_i^-1)^-1, x_fused = P_fused · Σ(P_i^-1 · x_i)
        cov_stack = np.array([c[:2, :2] for c in self.localTrackersCovarianceList]) + 1e-8 * np.eye(2)
        pos_stack = np.array([m[:2] for m in self.localTrackersMeasurementList])
        prec_stack = np.linalg.inv(cov_stack)  # (N, 2, 2) batch inverse
        precision_sum = prec_stack.sum(axis=0)
        weighted_pos = np.einsum('nij,nj->i', prec_stack, pos_stack)

        fused_cov = np.linalg.inv(precision_sum)
        fused_pos = fused_cov @ weighted_pos

        return fused_pos, fused_cov


    def fusion(self, measurement_list, time, monitor, participant_trust_scores: Dict[int, float] = None):
        """
        Fuse measurements using Kalman filter.

        Args:
            measurement_list: List of MatchClass objects
            time: Current timestamp
            monitor: Whether to collect cooperative monitoring data
            participant_trust_scores: Dict mapping participant_id to trust score (for global fusion)
                                     If None, trust scoring is disabled or uses default 1.0
        """
        # Set the kalman variables and resize the arrays dynalically (if needed
        self.addFrames(measurement_list)
        # Do the kalman thing!
        if self.idx == 0:
            # We have no prior detection so we need to just output what we have but store for later
            # Do a Naive average to get the starting position
            pos, cov = self.averageMeasurementsFirstFrame()

            # Store so that next fusion is better
            self.x = pos[0]
            self.y = pos[1]
            # error_covariance is always 2x2 (position only) for bbox matching/ellipsify
            self.error_covariance = np.array([
                [cov[0, 0], cov[0, 1]],
                [cov[1, 0], cov[1, 1]]
            ], dtype='float')
            self.idx += 1

            if self.fusion_mode == 0:
                # State: [x, y, dx, dy] - velocity initialized to 0, will be inferred from position
                self.X_hat_t = np.array(
                    [[self.x], [self.y], [0], [0]], dtype='float')
            elif self.fusion_mode == 1:
                # State: [x, y, dx, dy, ddx, ddy] - velocity/accel initialized to 0
                self.X_hat_t = np.array(
                    [[self.x], [self.y], [0], [0], [0], [0]], dtype='float')
            else:
                # CTRV model: [x, y, v, psi, psi_dot]
                # Initialize heading from measurement if available (pos has 3 elements for CTRV)
                init_heading = pos[2] if len(pos) > 2 else 0.0
                self.X_hat_t = np.array(
                    [[self.x], [self.y], [0], [init_heading], [0]], dtype='float')
                # Seed heading covariance from measurement
                if cov.shape[0] > 2:
                    self.P_hat_t[3][3] = cov[2, 2]

            # Seed the position covariance values directly from the measurement
            self.P_hat_t[0][0] = self.error_covariance[0][0]
            self.P_hat_t[0][1] = self.error_covariance[0][1]
            self.P_hat_t[1][0] = self.error_covariance[1][0]
            self.P_hat_t[1][1] = self.error_covariance[1][1]
            self.last_update = time
            self.x = self.x
            self.y = self.y
            self.dx = 0.0
            self.dy = 0.0
            self.idx += 1
        else:
            elapsed = time - self.last_update
            if elapsed <= 0.0:
                elapsed = 0.1

            if self.fusion_mode == 0:
                self.F_t = np.array([[1, 0, elapsed, 0],
                                    [0, 1, 0, elapsed],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype='float')

            elif self.fusion_mode == 1:
                # setup for x, dx, dxdx
                self.F_t = np.array([[1, 0, elapsed, 0, elapsed*elapsed, 0],
                                    [0, 1, 0, elapsed, 0, elapsed*elapsed],
                                    [0, 0, 1, 0, elapsed, 0],
                                    [0, 0, 0, 1, 0, elapsed],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]], dtype='float')

            else:
                # setup for CTRV model (https://journals.sagepub.com/doi/abs/10.1177/0959651820975523)
                # State: [x, y, v, psi, psi_dot]
                v_1 = float(self.X_hat_t[2, 0])
                phi_1 = float(self.X_hat_t[3, 0])
                phi_dot_1 = float(self.X_hat_t[4, 0])

                # Pre-calculate trig terms
                sin_phi = math.sin(phi_1)
                cos_phi = math.cos(phi_1)

                # Handle cases where phi_dot is near zero (constant velocity limit)
                if abs(phi_dot_1) < 1e-5:
                    v_x = cos_phi * elapsed
                    v_y = sin_phi * elapsed
                    phi_x = -v_1 * sin_phi * elapsed
                    phi_y = v_1 * cos_phi * elapsed
                    phi_dot_x = -0.5 * v_1 * sin_phi * elapsed**2
                    phi_dot_y = 0.5 * v_1 * cos_phi * elapsed**2
                else:
                    phi_delta = phi_1 + elapsed * phi_dot_1
                    sin_phi_delta = math.sin(phi_delta)
                    cos_phi_delta = math.cos(phi_delta)

                    v_x = (1.0 / phi_dot_1) * (-sin_phi + sin_phi_delta)
                    v_y = (1.0 / phi_dot_1) * (cos_phi - cos_phi_delta)

                    phi_x = (v_1 / phi_dot_1) * (-cos_phi + cos_phi_delta)
                    phi_y = (v_1 / phi_dot_1) * (-sin_phi + sin_phi_delta)

                    phi_dot_x = (v_1 * elapsed / phi_dot_1) * cos_phi_delta - \
                                (v_1 / phi_dot_1**2) * (-sin_phi + sin_phi_delta)
                    phi_dot_y = (v_1 * elapsed / phi_dot_1) * sin_phi_delta - \
                                (v_1 / phi_dot_1**2) * (cos_phi - cos_phi_delta)

                self.F_t = np.array([[1, 0, v_x, phi_x, phi_dot_x],
                                    [0, 1, v_y, phi_y, phi_dot_y],
                                    [0, 0, 1, 0, 0],
                                    [0, 0, 0, 1, elapsed],
                                    [0, 0, 0, 0, 1]], dtype='float')

            # Reuse prediction from getKalmanPredWithCovariance if it was called
            # for this same timestamp (avoids recomputing identical F and matrix multiply).
            if self._pred_cache_time == time:
                self.X_hat_t = self._pred_cache_X
                self.P_hat_t = self._pred_cache_P
                self._pred_cache_time = None  # consumed; invalidate
            else:
                self.Q_t = self._compute_Q(elapsed)
                self.X_hat_t, self.P_hat_t = utils.kalman_prediction(
                    self.X_hat_t, self.P_hat_t, self.F_t, self.B_t, self.U_t, self.Q_t)

            # Process measurements
            added = 0
            valid_measurements = []
            valid_covariances = []

            if len(self.localTrackersMeasurementList) != 0:
                for mu, cov, h_t_type, tracker_id in zip(self.localTrackersMeasurementList, self.localTrackersCovarianceList, self.localTrackersHList, self.localTrackersIDList):
                    # Skip if measurement or covariance contains NaN/Inf
                    if np.any(np.isnan(mu)) or np.any(np.isnan(cov)):
                        continue
                    if np.any(np.isinf(mu)) or np.any(np.isinf(cov)):
                        continue

                    # Trust scoring: filter bad participants and scale covariance
                    if self.use_trust_scoring and participant_trust_scores is not None:
                        participant_id = tracker_id // max_id
                        trust_score = participant_trust_scores.get(participant_id, 1.0)
                        if np.isnan(trust_score) or np.isinf(trust_score):
                            trust_score = 1.0
                        if trust_score <= 0.67:
                            continue
                        clamped_score = max(0.67, min(2.0, trust_score))
                        adjusted_cov = cov / (clamped_score ** 2)
                    else:
                        adjusted_cov = cov

                    # Ensure covariance has minimum values to prevent singularity
                    min_cov = 1e-11
                    adjusted_cov = adjusted_cov + min_cov * np.eye(adjusted_cov.shape[0])

                    valid_measurements.append(mu)
                    valid_covariances.append(adjusted_cov)

                added = len(valid_measurements)

                if added > 0:
                    if self.precision_weighted_fusion:
                        # Precision-weighted mean fusion: directly compute weighted average
                        # Position = (Σ R_i^-1)^-1 · Σ(R_i^-1 · z_i)
                        # This gives GPEM covariances direct control over weighting.
                        # Note: prediction is NOT included as a term here to avoid an
                        # overconfidence feedback loop where fused_cov shrinks each frame,
                        # causing the prediction to dominate measurements.

                        # Extract 2D position measurements and covariances
                        positions = [mu[:2] for mu in valid_measurements]
                        pos_covs = [cov[:2, :2] if cov.shape[0] >= 2 else cov for cov in valid_covariances]

                        # Batch precision-weighted fusion: vectorized matrix inversions
                        cov_stack = np.array(pos_covs)  # (N, 2, 2)
                        pos_stack = np.array(positions)  # (N, 2)
                        # Regularize near-singular covariances
                        cov_stack += 1e-8 * np.eye(2)
                        prec_stack = np.linalg.inv(cov_stack)  # (N, 2, 2) batch inverse
                        precision_sum = prec_stack.sum(axis=0)  # (2, 2)
                        weighted_sum = np.einsum('nij,nj->i', prec_stack, pos_stack)  # (2,)

                        # Fused covariance = inverse of summed precisions
                        try:
                            fused_cov = np.linalg.inv(precision_sum)
                        except np.linalg.LinAlgError:
                            fused_cov = np.eye(2) * 0.1

                        # Fused position
                        fused_pos = fused_cov @ weighted_sum

                        # Update state: set position directly, keep velocity from prediction
                        self.X_hat_t[0, 0] = fused_pos[0]
                        self.X_hat_t[1, 0] = fused_pos[1]

                        # Update position covariance block
                        self.P_hat_t[0:2, 0:2] = fused_cov

                        # Store for reference
                        self.last_measurement_cov = fused_cov.copy()

                        # Update velocity estimate from position change
                        if hasattr(self, '_last_fused_pos') and self._last_fused_pos is not None:
                            if elapsed > 0.001:
                                vel_x = (fused_pos[0] - self._last_fused_pos[0]) / elapsed
                                vel_y = (fused_pos[1] - self._last_fused_pos[1]) / elapsed
                                alpha = 0.3  # Weight for new velocity estimate
                                if self.fusion_mode == 0 or self.fusion_mode == 1:
                                    self.X_hat_t[2, 0] = alpha * vel_x + (1 - alpha) * self.X_hat_t[2, 0]
                                    self.X_hat_t[3, 0] = alpha * vel_y + (1 - alpha) * self.X_hat_t[3, 0]
                                elif self.fusion_mode == 2:
                                    speed = math.hypot(vel_x, vel_y)
                                    self.X_hat_t[2, 0] = alpha * speed + (1 - alpha) * self.X_hat_t[2, 0]
                        self._last_fused_pos = fused_pos.copy()
                    else:
                        # Sequential Kalman updates (original behavior)
                        for mu, adjusted_cov in zip(valid_measurements, valid_covariances):
                            Z_t = mu.transpose()
                            Z_t = Z_t.reshape(Z_t.shape[0], -1)
                            self.X_hat_t, self.P_hat_t = utils.kalman_update(
                                self.X_hat_t, self.P_hat_t, Z_t, adjusted_cov, self.h_t(0))

                            if adjusted_cov.shape[0] >= 2:
                                self.last_measurement_cov = adjusted_cov[0:2, 0:2].copy()

                        # Passthrough mode: override Kalman covariance
                        if self.passthrough_covariance and self.last_measurement_cov is not None:
                            self.P_hat_t[0:2, 0:2] = self.last_measurement_cov

            if len(self.localTrackersMeasurementList) == 0 or added == 0:
                # No measurements - do a zero-gain update (effectively just prediction)
                if self.fusion_mode == 0:
                    nothing_cov = np.eye(2, dtype='float')
                    measure = np.zeros(2, dtype='float')
                    nothing_Ht = np.zeros((2, 4), dtype='float')
                elif self.fusion_mode == 1:
                    nothing_cov = np.eye(2, dtype='float')
                    measure = np.zeros(2, dtype='float')
                    nothing_Ht = np.zeros((2, 6), dtype='float')
                elif self.fusion_mode == 2:
                    # CTRV: 3x3 covariance, 3-element measurement, 3x5 H
                    nothing_cov = np.eye(3, dtype='float')
                    measure = np.zeros(3, dtype='float')
                    nothing_Ht = np.zeros((3, 5), dtype='float')

                Z_t = measure.reshape(-1, 1)
                self.X_hat_t, self.P_hat_t = utils.kalman_update(
                    self.X_hat_t, self.P_hat_t, Z_t, nothing_cov, nothing_Ht)

            # Lets check the accuracy of each sensing platform
            self.error_tracker_temp = []
            length = len(self.localTrackersIDList)
            if monitor:
                # Our method
                if self.P_hat_t[0][0] != 0.0 and self.P_hat_t[0][1] != 0.0:
                    global_error_x, global_error_y, global_error_angle = utils.ellipsify(
                        [[self.P_hat_t[0][0], self.P_hat_t[0][1]], [self.P_hat_t[1][0], self.P_hat_t[1][1]]], 1.0)
                else:
                    global_error_x = 0.0
                    global_error_y = 0.0
                    global_error_angle = 0.0
                for id, mu, cov, h_t_type in zip(self.localTrackersIDList, self.localTrackersMeasurementList, self.localTrackersCovarianceList, self.localTrackersHList):
                    Z_t = (mu).transpose()
                    Z_t = Z_t.reshape(Z_t.shape[0], -1)
                    y_t_temp = Z_t - self.h_t(h_t_type).dot(self.X_hat_t)
                    location_error = math.hypot(y_t_temp[0], y_t_temp[1])
                    expected_a, expected_b, expected_angle = utils.ellipsify(
                        cov, 1.0)
                    expected_x = utils.calculateRadiusAtAngle(
                        expected_a, expected_b, expected_angle, math.radians(0))
                    expected_y = utils.calculateRadiusAtAngle(
                        expected_a, expected_b, expected_angle, math.radians(90))
                    expected_location_error = math.hypot(
                        expected_x, expected_y) - math.hypot(global_error_x, global_error_y)

                    # Protect against division by zero or invalid values
                    if expected_location_error <= 0.0 or math.isnan(expected_location_error):
                        # Use a fallback: just the expected ellipse size
                        expected_location_error = max(0.01, math.hypot(expected_x, expected_y))

                    location_error_std = location_error / expected_location_error

                    # Skip if result is NaN or Inf
                    if math.isnan(location_error_std) or math.isinf(location_error_std):
                        continue

                    self.error_tracker_temp.append(
                        [id, location_error_std, length])

                trupercept_list = []
                for id_test, mu_test, confidence_test in zip(self.localTrackersIDList, self.localTrackersMeasurementList, self.localTrackersExtraList):
                    trupercept_list.append([id_test, confidence_test[0]])
                self.trupercept_list = trupercept_list

            self.last_update = time
            self.x = self.X_hat_t[0][0]
            self.y = self.X_hat_t[1][0]
            if self.fusion_mode == 2:
                # Different setup for fusion mode 2 need to calculate vector from angle and velocity
                self.dx = self.X_hat_t[2][0] * math.cos(self.X_hat_t[3][0])
                self.dy = self.X_hat_t[2][0] * math.sin(self.X_hat_t[3][0])
            else:
                self.dx = self.X_hat_t[2][0]
                self.dy = self.X_hat_t[3][0]
            self.idx += 1
            if self.P_hat_t[0][0] != 0.0 or self.P_hat_t[0][1] != 0.0:
                self.error_covariance = np.array([[self.P_hat_t[0][0], self.P_hat_t[0][1]], [
                                                 self.P_hat_t[1][0], self.P_hat_t[1][1]]], dtype='float')
                self.d_covariance = np.array([[self.P_hat_t[2][2], self.P_hat_t[2][3]], [
                                             self.P_hat_t[3][2], self.P_hat_t[3][3]]], dtype='float')
            else:
                self.error_covariance = np.array(
                    [[1.0, 0.0], [0.0, 1.0]], dtype='float')
                self.d_covariance = np.array(
                    [[2.0, 0.0], [0.0, 2.0]], dtype='float')

            # 1D Kalman update for width and length
            # Prediction step: state stays same (dimensions don't change), variance increases
            self.width_variance += self.width_process_noise
            self.length_variance += self.length_process_noise

            # Update step: fuse all measurements
            if len(self.localTrackersWidthList) > 0:
                for w_meas, l_meas, w_std, l_std in zip(self.localTrackersWidthList, self.localTrackersLengthList, self.localTrackersWidthStdList, self.localTrackersLengthStdList):
                    # Width Kalman update using GPEM-predicted std
                    # Use predicted std squared as measurement noise
                    w_noise = max(1e-4, w_std**2)
                    width_gain = self.width_variance / (self.width_variance + w_noise)
                    self.width = self.width + width_gain * (w_meas - self.width)
                    self.width_variance = (1 - width_gain) * self.width_variance

                    # Length Kalman update using GPEM-predicted std
                    l_noise = max(1e-4, l_std**2)
                    length_gain = self.length_variance / (self.length_variance + l_noise)
                    self.length = self.length + length_gain * (l_meas - self.length)
                    self.length_variance = (1 - length_gain) * self.length_variance


    def getKalmanPred(self, time):
        # Use the current Kalman state to predict the position at the given time
        elapsed = time - self.last_update
        if elapsed < 0:
            # If the requested time is before the last update, return the last updated position
            # Or handle as an error/special case, for now, just return current state
            predicted_x = self.X_hat_t[0][0]
            predicted_y = self.X_hat_t[1][0]
            predicted_P = self.P_hat_t
        else:
            # Predict the state forward
            if self.fusion_mode == 0:
                F_pred = np.array([[1, 0, elapsed, 0],
                                   [0, 1, 0, elapsed],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype='float')
            elif self.fusion_mode == 1:
                F_pred = np.array([[1, 0, elapsed, 0, elapsed*elapsed, 0],
                                   [0, 1, 0, elapsed, 0, elapsed*elapsed],
                                   [0, 0, 1, 0, elapsed, 0],
                                   [0, 0, 0, 1, 0, elapsed],
                                   [0, 0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 1]], dtype='float')
            else:
                # CTRV model: State = [x, y, v, psi, psi_dot]
                # Extrapolate using current velocity and heading
                v = float(self.X_hat_t[2, 0])
                psi = float(self.X_hat_t[3, 0])
                psi_dot = float(self.X_hat_t[4, 0])

                if abs(psi_dot) < 1e-5:
                    # Straight-line motion
                    predicted_x = self.X_hat_t[0, 0] + v * math.cos(psi) * elapsed
                    predicted_y = self.X_hat_t[1, 0] + v * math.sin(psi) * elapsed
                else:
                    # Turning motion
                    psi_new = psi + psi_dot * elapsed
                    predicted_x = self.X_hat_t[0, 0] + (v / psi_dot) * (math.sin(psi_new) - math.sin(psi))
                    predicted_y = self.X_hat_t[1, 0] + (v / psi_dot) * (-math.cos(psi_new) + math.cos(psi))

                # Build F_pred for covariance propagation (linearized around current state)
                cos_psi = math.cos(psi)
                sin_psi = math.sin(psi)
                if abs(psi_dot) < 1e-5:
                    v_x = cos_psi * elapsed
                    v_y = sin_psi * elapsed
                    psi_x = -v * sin_psi * elapsed
                    psi_y = v * cos_psi * elapsed
                    psi_dot_x = -0.5 * v * sin_psi * elapsed**2
                    psi_dot_y = 0.5 * v * cos_psi * elapsed**2
                else:
                    psi_new = psi + elapsed * psi_dot
                    sin_psi_new = math.sin(psi_new)
                    cos_psi_new = math.cos(psi_new)
                    v_x = (1.0 / psi_dot) * (-sin_psi + sin_psi_new)
                    v_y = (1.0 / psi_dot) * (cos_psi - cos_psi_new)
                    psi_x = (v / psi_dot) * (-cos_psi + cos_psi_new)
                    psi_y = (v / psi_dot) * (-sin_psi + sin_psi_new)
                    psi_dot_x = (v * elapsed / psi_dot) * cos_psi_new - (v / psi_dot**2) * (-sin_psi + sin_psi_new)
                    psi_dot_y = (v * elapsed / psi_dot) * sin_psi_new - (v / psi_dot**2) * (cos_psi - cos_psi_new)

                F_pred = np.array([[1, 0, v_x, psi_x, psi_dot_x],
                                   [0, 1, v_y, psi_y, psi_dot_y],
                                   [0, 0, 1, 0, 0],
                                   [0, 0, 0, 1, elapsed],
                                   [0, 0, 0, 0, 1]], dtype='float')

            predicted_X_hat, predicted_P = utils.kalman_prediction(
                self.X_hat_t, self.P_hat_t, F_pred, self.B_t, self.U_t, self.Q_t)
            predicted_x = predicted_X_hat[0][0]
            predicted_y = predicted_X_hat[1][0]

        a, b, phi = utils.ellipsify(predicted_P[0:2, 0:2], 1.0)
        return predicted_x, predicted_y, a, b, phi

    def getKalmanPredWithCovariance(self, time):
        """
        Extended version of getKalmanPred that also returns the predicted covariance matrix.
        Used for Mahalanobis distance computation in Hungarian matching.
        """
        elapsed = time - self.last_update
        if elapsed < 0:
            predicted_x = self.X_hat_t[0][0]
            predicted_y = self.X_hat_t[1][0]
            predicted_P = self.P_hat_t
        else:
            if self.fusion_mode == 0:
                F_pred = np.array([[1, 0, elapsed, 0],
                                   [0, 1, 0, elapsed],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype='float')
            elif self.fusion_mode == 1:
                F_pred = np.array([[1, 0, elapsed, 0, elapsed*elapsed, 0],
                                   [0, 1, 0, elapsed, 0, elapsed*elapsed],
                                   [0, 0, 1, 0, elapsed, 0],
                                   [0, 0, 0, 1, 0, elapsed],
                                   [0, 0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 1]], dtype='float')
            else:
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

                cos_psi = math.cos(psi)
                sin_psi = math.sin(psi)
                if abs(psi_dot) < 1e-5:
                    v_x = cos_psi * elapsed
                    v_y = sin_psi * elapsed
                    psi_x = -v * sin_psi * elapsed
                    psi_y = v * cos_psi * elapsed
                    psi_dot_x = -0.5 * v * sin_psi * elapsed**2
                    psi_dot_y = 0.5 * v * cos_psi * elapsed**2
                else:
                    psi_new = psi + elapsed * psi_dot
                    sin_psi_new = math.sin(psi_new)
                    cos_psi_new = math.cos(psi_new)
                    v_x = (1.0 / psi_dot) * (-sin_psi + sin_psi_new)
                    v_y = (1.0 / psi_dot) * (cos_psi - cos_psi_new)
                    psi_x = (v / psi_dot) * (-cos_psi + cos_psi_new)
                    psi_y = (v / psi_dot) * (-sin_psi + sin_psi_new)
                    psi_dot_x = (v * elapsed / psi_dot) * cos_psi_new - (v / psi_dot**2) * (-sin_psi + sin_psi_new)
                    psi_dot_y = (v * elapsed / psi_dot) * sin_psi_new - (v / psi_dot**2) * (cos_psi - cos_psi_new)

                F_pred = np.array([[1, 0, v_x, psi_x, psi_dot_x],
                                   [0, 1, v_y, psi_y, psi_dot_y],
                                   [0, 0, 1, 0, 0],
                                   [0, 0, 0, 1, elapsed],
                                   [0, 0, 0, 0, 1]], dtype='float')

            predicted_X_hat, predicted_P = utils.kalman_prediction(
                self.X_hat_t, self.P_hat_t, F_pred, self.B_t, self.U_t, self.Q_t)
            predicted_x = predicted_X_hat[0][0]
            predicted_y = predicted_X_hat[1][0]

            # Cache so fusion() can skip re-computing the identical prediction step
            self._pred_cache_time = time
            self._pred_cache_X = predicted_X_hat
            self._pred_cache_P = predicted_P

        return predicted_x, predicted_y, predicted_P
