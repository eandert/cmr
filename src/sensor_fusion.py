import math
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from scipy.optimize import linear_sum_assignment
import bisect
from typing import Dict, Tuple, List, Optional

import utils
from sensor import DetectedObject


max_id = 10000


class MatchClass:
    def __init__(self, id, x, y, covariance, dx, dy, d_confidence, confidence, trust_score, object_type, time, width, length, angle, width_std=0.5, length_std=0.5):
        self.x = x
        self.y = y
        self.covariance = covariance
        self.dx = dx
        self.dy = dy
        self.velocity_confidence = d_confidence
        self.type = object_type
        self.last_measurement = time
        self.id = id
        self.confidence = confidence
        self.trust_score = trust_score
        self.width = width
        self.length = length
        self.angle = angle
        self.width_std = width_std
        self.length_std = length_std


class ResizableKalman:
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
        self.sigma_a = 2.0  # m/s^2 - standard deviation of random acceleration
        
        # Adaptive process noise: store last measurement covariance for position block of Q
        # This makes Q adapt to GPEM predictions - if sensor says uncertainty is high, Q is high
        self.last_measurement_cov = None  # 2x2 position covariance from last measurement
        
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
                angle_variance = 0.1  # ~18 deg std dev for heading measurement
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
            # TODO: implement radar type
            return np.array([[0, 0., 0., 0.],
                            [0., 0, 0., 0.]], dtype='float')

    def averageMeasurementsFirstFrame(self):
        if len(self.localTrackersCovarianceList) == 1:
            return self.localTrackersMeasurementList[0], self.localTrackersCovarianceList[0]

        # Simple mean average for position
        temporary_mu = np.mean(self.localTrackersMeasurementList, axis=0)
        
        # FUSION: The covariance of the mean of N independent measurements is (1/N^2) * sum(Sigma_i)
        # If they are all roughly the same, this is Sigma/N.
        # Previously this was np.mean (Sigma), which was N times too large.
        num_measurements = len(self.localTrackersCovarianceList)
        temporary_c = np.mean(self.localTrackersCovarianceList, axis=0) / num_measurements

        return temporary_mu, temporary_c


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
            # try:
            # We have valid data
            # Transition matrix
            elapsed = time - self.last_update  # FIXED: new_time - old_time = positive
            if elapsed <= 0.0:
                # print( "Error time elapsed is incorrect! " + str(elapsed) )
                # Set to arbitrary time
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

            # Compute process noise Q for this timestep
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
                        # Position = (Σ P_i^-1)^-1 · Σ(P_i^-1 · x_i)
                        # This gives GPEM covariances direct control over weighting
                        
                        # Extract 2D position measurements and covariances
                        positions = [mu[:2] for mu in valid_measurements]
                        pos_covs = [cov[:2, :2] if cov.shape[0] >= 2 else cov for cov in valid_covariances]
                        
                        # Compute precisions (inverse covariances)
                        precisions = []
                        for pc in pos_covs:
                            try:
                                prec = np.linalg.inv(pc)
                                precisions.append(prec)
                            except np.linalg.LinAlgError:
                                prec = np.linalg.inv(pc + 0.01 * np.eye(2))
                                precisions.append(prec)
                        
                        # Sum of precisions = fused precision
                        precision_sum = sum(precisions)
                        
                        # Fused covariance = inverse of summed precisions
                        try:
                            fused_cov = np.linalg.inv(precision_sum)
                        except np.linalg.LinAlgError:
                            fused_cov = np.eye(2) * 0.1
                        
                        # Fused position = fused_cov @ Σ(precision_i @ position_i)
                        weighted_sum = sum(prec @ pos for prec, pos in zip(precisions, positions))
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
                    # print(y_t_temp)
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
                    # print(" error: ", id, location_error, expected_location_error, location_error_std)

                # TruPercept
                trupercept_list = []
                # print(self.localTrackersIDList, self.localTrackersMeasurementList, self.localTrackersExtraList)
                for id_test, mu_test, confidence_test in zip(self.localTrackersIDList, self.localTrackersMeasurementList, self.localTrackersExtraList):
                    # print(self.localTrackersIDList, self.localTrackersMeasurementList, self.localTrackersExtraList)
                    # trupercept_sub_list = []
                    # for id, mu, confidence in zip(self.localTrackersIDList, self.localTrackersMeasurementList, self.localTrackersExtraList):
                    #     if id_test != id:
                    # iou = 1 - shared_math.computeDistanceBBox([mu_test[0], mu_test[1], self.min_size, self.min_size, 0], [mu[0], mu[1], self.min_size, self.min_size, 0])
                    trupercept_list.append([id_test, confidence_test[0]])
                    # trupercept_list.append(trupercept_sub_list)
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

            # except Exception as e:
            #     print ( " Exception: " + str(e) )

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

        return predicted_x, predicted_y, predicted_P


class GlobalTracked:
    def __init__(self, detected_object, time, track_id, fusion_mode, use_trust_scoring=True, passthrough_covariance=False):
        self.x = detected_object.centroid[0]
        self.y = detected_object.centroid[1]
        self.passthrough_covariance = passthrough_covariance
        self.dx = 0
        self.dy = 0
        self.error_covariance = np.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype='float')
        self.last_measurement = time
        self.last_update = time
        self.id = track_id
        self.idx = 0
        self.min_size = 0.5
        self.track_count = 0
        self.d_covariance = np.array([[2.0, 0.0], [0.0, 2.0]], dtype='float')
        self.match_list = []
        self.fusion_steps = 0
        self.error_monitor = []
        self.num_trackers = 0
        self.width = detected_object.dimensions[0]
        self.length = detected_object.dimensions[1]
        self.angle = detected_object.angle

        # Type voting: track counts of each type seen
        # Uses a sliding window approach with exponential decay
        self.type_votes = {}  # {type_value: vote_count}
        self.type = detected_object.type  # Current best estimate
        self._update_type_vote(detected_object.type)

        # Trupercept stuff
        self.trupercept_list = []

        # Add this first match
        new_match = MatchClass(detected_object.vehicle_id, detected_object.centroid[0], detected_object.centroid[1], detected_object.error_covariance,
                               detected_object.velocity_vector[0], detected_object.velocity_vector[1], detected_object.error_covariance, 1.0, 1.0, 
                               detected_object.type, time, 
                               detected_object.dimensions[0], detected_object.dimensions[1], detected_object.angle,
                               detected_object.width_std, detected_object.length_std)
        self.match_list.append(new_match)

        # Kalman stuff
        self.fusion_mode = fusion_mode
        self.kalman = ResizableKalman(time, self.x, self.y, fusion_mode, 
                                       initial_width=self.width, initial_length=self.length,
                                       use_trust_scoring=use_trust_scoring,
                                       passthrough_covariance=passthrough_covariance)
    
    def _update_type_vote(self, detected_type):
        """Update type voting with a new detection's type classification."""
        if detected_type is None:
            return
        
        # Decay existing votes (exponential decay for recency weighting)
        decay_factor = 0.95
        for t in self.type_votes:
            self.type_votes[t] *= decay_factor
        
        # Add vote for this detection's type
        if detected_type in self.type_votes:
            self.type_votes[detected_type] += 1.0
        else:
            self.type_votes[detected_type] = 1.0
        
        # Determine winning type (majority vote with recency weighting)
        if self.type_votes:
            self.type = max(self.type_votes, key=self.type_votes.get)

    def update(self, detected_object, time):
        new_match = MatchClass(detected_object.vehicle_id, detected_object.centroid[0], detected_object.centroid[1], detected_object.error_covariance,
                               detected_object.velocity_vector[0], detected_object.velocity_vector[1], detected_object.error_covariance, 1.0, 1.0, 
                               detected_object.type, time, 
                               detected_object.dimensions[0], detected_object.dimensions[1], detected_object.angle,
                               detected_object.width_std, detected_object.length_std)
        self.match_list.append(new_match)
        
        # Update type voting
        self._update_type_vote(detected_object.type)

        self.last_measurement = time

    # Gets our position in an array form so we can use it in the BallTree
    def getPosition(self):
        # ORDER: [x, y, width, length, angle]
        # REVERTED: Matching logic expects width first
        return [
            [self.x, self.y, self.width, self.length, self.angle]
        ]

    def getPositionPredicted(self, timestamp):
        # Call the Kalman filter's prediction method
        predicted_x, predicted_y, a, b, phi = self.kalman.getKalmanPred(timestamp)

        # Dynamically adjust the bounding box size based on Kalman filter's uncertainty
        # Use 2 standard deviations for the matching gate to help catch detections 
        # that might be slightly offset from prediction
        adjusted_width = self.width + (a * 2.0)
        adjusted_length = self.length + (b * 2.0)

        # Return the predicted position with the dynamically adjusted width and length
        # ORDER: [x, y, width, length, angle]
        # REVERTED: Matching logic expects width first
        return [
            [predicted_x, predicted_y, adjusted_width, adjusted_length, self.angle]
        ]
    
    def getPositionPredictedWithCovariance(self, timestamp):
        """
        Get predicted position with full covariance matrix for Hungarian matching.
        Returns position, bounding box, and prediction covariance.
        """
        predicted_x, predicted_y, P_pred = self.kalman.getKalmanPredWithCovariance(timestamp)
        
        # Ellipse parameters for bbox adjustment
        a, b, phi = utils.ellipsify(P_pred[0:2, 0:2], 2.0)
        adjusted_width = self.width + (a * 2.0)
        adjusted_length = self.length + (b * 2.0)
        
        return {
            'x': predicted_x,
            'y': predicted_y,
            'P_pred': P_pred,
            'bbox': [predicted_x, predicted_y, adjusted_width, adjusted_length, self.angle],
            'base_bbox': [predicted_x, predicted_y, self.width, self.length, self.angle]
        }

    def fusion(self, time, monitor, participant_trust_scores: Dict[int, float] = None):
        """
        Fuse all matches for this track using Kalman filter.
        
        Args:
            time: Current timestamp
            monitor: Whether to collect cooperative monitoring data
            participant_trust_scores: Dict mapping participant_id to trust score (for global fusion)
        """
        self.kalman.fusion(self.match_list, time, monitor, participant_trust_scores)
        self.x = self.kalman.x
        self.y = self.kalman.y
        self.error_covariance = self.kalman.error_covariance
        self.dx = self.kalman.dx
        self.dy = self.kalman.dy
        self.d_covariance = self.kalman.d_covariance
        self.error_monitor = self.kalman.error_tracker_temp
        self.num_trackers = len(self.kalman.localTrackersIDList)
        self.trupercept_list = self.kalman.trupercept_list
        self.fusion_steps += 1
        
        # Get Kalman-tracked width and length
        self.width = self.kalman.width
        self.length = self.kalman.length
        
        # Use measured angle from detections (weighted circular mean)
        # Velocity direction != vehicle heading (especially during turns)
        if len(self.match_list) > 0:
            if len(self.match_list) == 1:
                self.angle = self.match_list[0].angle
            else:
                # Weighted circular mean for angle (handles wraparound)
                sin_sum = 0.0
                cos_sum = 0.0
                total_weight = 0.0
                for i, match in enumerate(self.match_list):
                    weight = 2.0 ** i  # More recent = higher weight
                    sin_sum += weight * math.sin(match.angle)
                    cos_sum += weight * math.cos(match.angle)
                    total_weight += weight
                if total_weight > 0:
                    self.angle = math.atan2(sin_sum / total_weight, cos_sum / total_weight)

    def clearLastFrame(self):
        self.match_list = []


class Fusion:
    # Fusion is a special class for matching and fusing detections for a variety of sources.
    # The input is scalable and therefore must be generated before being fed into this class.
    # A unique list of detections is required from each individual sensor or pre-fused device
    # output or it will not be matched. Detections too close to each other may be combined.
    # This is a modified version of the frame-by-frame tracker seen in:
    # https://github.com/eandert/Jetson_Nano_Camera_Vehicle_Tracker
    def __init__(self, id, use_trust_scoring=True, passthrough_covariance=False):
        # Set other parameters for the class
        self.tracked_list = []
        self.id = id
        self.current_track_id = 0
        self.min_size = 0.5
        self.trackShowThreshold = 4
        self.fusion_mode = 2  # CTRV model: uses heading measurement for better turn tracking
        self.use_trust_scoring = use_trust_scoring  # Flag to enable/disable trust-based filtering
        self.passthrough_covariance = passthrough_covariance  # Pass measurement covariance directly
        
        # Trust scoring: per-participant trust scores (default 1.0 = fully trusted)
        # Key: participant_id (extracted from vehicle_id // max_id)
        # Value: trust score (1.0 = normal, >1.2 = anomalous/untrusted)
        self.participant_trust_scores: Dict[int, float] = {}
    
    def get_participant_trust_score(self, vehicle_id: int) -> float:
        """
        Get the trust score for a participant based on their vehicle_id.
        
        Args:
            vehicle_id: The universal ID (participant_id * max_id + track_id)
            
        Returns:
            Trust score (1.0 = normal, >1.0 = less trusted)
        """
        participant_id = vehicle_id // max_id
        return self.participant_trust_scores.get(participant_id, 1.0)
    
    def update_trust_scores(self, scores: Dict[int, float]) -> None:
        """
        Update trust scores from PerceptionScorer output.
        
        Args:
            scores: Dict mapping participant_id to their normalized trust score
        """
        self.participant_trust_scores.update(scores)

    def fuseDetectionFrame(self, time, monitor=False):
        result = []
        cooperative_monitoring = []
        trupercept_monitoring = []
        detected_objects = []  # New list to store DetectedObject instances

        for track in self.tracked_list:
            # Pass participant trust scores to enable trust-based filtering/weighting
            trust_scores = self.participant_trust_scores if self.use_trust_scoring else None
            track.fusion(time, monitor, trust_scores)
            if track.fusion_steps > self.trackShowThreshold:
                result.append([track.id, track.x, track.y, track.error_covariance.tolist(
                ), track.dx, track.dy, track.d_covariance.tolist(), track.num_trackers])
                if track.error_monitor:
                    cooperative_monitoring.append(track.error_monitor)
                    trupercept_monitoring.append(track.trupercept_list)

                # print("Track ID: ", track.id, "X: ", track.x, "Y: ", track.y, "DX: ", track.dx, "DY: ", track.dy)
                
                # Create a DetectedObject instance and append to the list
                detected_object = DetectedObject(
                    vehicle_id=track.id,
                    vehicle_type=track.type,  # Use voted type from track
                    detected_bbox=None,  # Bounding box will be calculated by the property
                    centroid=[track.x, track.y],
                    width=track.width,
                    length=track.length,
                    angle=track.angle,  # Use the averaged angle
                    expected_error_gaussian=None,
                    error_covariance=track.error_covariance,
                    velocity_vector=[track.dx, track.dy]
                )
                detected_objects.append(detected_object)

                # Now use the property to get the bounding box corners
                bbox = detected_object.detected_bbox_corners

                # If you need to use the bbox directly later, you can update the detected_bbox attribute
                detected_object.detected_bbox = bbox

            track.clearLastFrame()

        if len(self.tracked_list) >= 1:
            self.cleanDetections()

        return result, detected_objects, cooperative_monitoring, trupercept_monitoring

    def processDetectionFrame(self, timestamp, observations, cleanupTime, source_participant_id: int = 0):
        """
        Process a frame of detections from a single participant (CAV/CIS).
        
        Args:
            timestamp: Current simulation time
            observations: List of DetectedObject instances
            cleanupTime: Time threshold for removing stale tracks
            source_participant_id: ID of the CAV/CIS that produced these detections
                                   Used for trust scoring (participant_id * max_id + track_id)
        """
        detections_list_positions = []
        detections_list = []
        for idx, det in enumerate(observations):
            # ORDER: [x, y, width, length, angle]
            # REVERTED: Matching logic expects width first
            detections_list_positions.append(
                [det.centroid[0], det.centroid[1], det.dimensions[0], det.dimensions[1], det.angle])
            
            # Encode source participant ID into the detection for trust scoring
            # This allows us to track which participant contributed each detection
            # Format: encoded_id = source_participant_id * max_id + detection_index
            # Note: vehicle_id may be a string (SUMO ID), so we use detection index instead
            det.vehicle_id = source_participant_id * max_id + idx
            
            detections_list.append(det)

        # Call the matching function to modify our detections in tracked_list
        self.matchDetections(detections_list_positions, detections_list,
                             timestamp, cleanupTime)

    def matchDetections(self, detections_list_positions, detection_list, timestamp, cleanupTime):
        """
        Match detections to existing tracks using the Hungarian algorithm with
        a hybrid cost function combining Mahalanobis distance and IOU.
        
        This replaces the greedy BallTree approach with globally optimal assignment.
        """
        # Check if there are any detections
        if len(detections_list_positions) > 0:
            
            # Check if there are any existing tracks
            if len(self.tracked_list) > 0:
                num_tracks = len(self.tracked_list)
                num_detections = len(detections_list_positions)
                
                # Build cost matrix: rows = tracks, cols = detections
                # Using a large finite value for "impossible" assignments (gated out)
                # Note: scipy's linear_sum_assignment can't handle inf values
                IMPOSSIBLE_COST = 1e9
                cost_matrix = np.full((num_tracks, num_detections), IMPOSSIBLE_COST)
                
                # Pre-compute track predictions with covariance
                track_predictions = []
                for track in self.tracked_list:
                    pred = track.getPositionPredictedWithCovariance(timestamp)
                    track_predictions.append(pred)
                
                # Build cost matrix with hybrid Mahalanobis + IOU metric
                for t_idx, (track, pred) in enumerate(zip(self.tracked_list, track_predictions)):
                    for d_idx, det in enumerate(detection_list):
                        det_pos = [det.centroid[0], det.centroid[1]]
                        det_bbox = detections_list_positions[d_idx]
                        det_cov = det.error_covariance
                        
                        # Ensure det_cov is 2x2
                        if det_cov is None or np.array(det_cov).size < 4:
                            det_cov = np.eye(2)
                        else:
                            det_cov = np.array(det_cov).reshape(2, 2)
                        
                        # Compute hybrid cost
                        cost, mahal_dist, iou = utils.compute_hybrid_cost(
                            detection_pos=det_pos,
                            detection_cov=det_cov,
                            predicted_pos=[pred['x'], pred['y']],
                            P_pred=pred['P_pred'],
                            det_bbox=det_bbox,
                            pred_bbox=pred['bbox'],
                            mahal_weight=0.6,  # Weight Mahalanobis more for far detections
                            iou_weight=0.4,    # IOU helps when boxes overlap
                            mahal_gate=13.82   # 99.9% chi-squared with 2 DOF
                        )
                        
                        cost_matrix[t_idx, d_idx] = cost
                
                # Run Hungarian algorithm for optimal assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Filter out assignments that exceed cost threshold
                matched_track_indices = set()
                matched_det_indices = set()
                
                for t_idx, d_idx in zip(row_ind, col_ind):
                    if cost_matrix[t_idx, d_idx] < IMPOSSIBLE_COST:
                        # Valid match
                        self.tracked_list[t_idx].update(detection_list[d_idx], timestamp)
                        matched_track_indices.add(t_idx)
                        matched_det_indices.add(d_idx)
                
                # Create new tracks for unmatched detections
                for d_idx in range(num_detections):
                    if d_idx not in matched_det_indices:
                        new = GlobalTracked(
                            detection_list[d_idx],
                            timestamp, 
                            (max_id * self.id) + self.current_track_id, 
                            self.fusion_mode,
                            use_trust_scoring=self.use_trust_scoring,
                            passthrough_covariance=self.passthrough_covariance
                        )
                        if self.current_track_id < max_id:
                            self.current_track_id += 1
                        else:
                            self.current_track_id = 0
                        self.tracked_list.append(new)

            else:
                # If there are no existing tracks, create new tracks for all detections
                for dl in detection_list:
                    new = GlobalTracked(dl, timestamp, (max_id * self.id) + self.current_track_id, self.fusion_mode,
                                        use_trust_scoring=self.use_trust_scoring,
                                        passthrough_covariance=self.passthrough_covariance)
                    if self.current_track_id < max_id:
                        self.current_track_id += 1
                    else:
                        self.current_track_id = 0
                    self.tracked_list.append(new)

        # Clean up old tracks
        remove = []
        for idx, track in enumerate(self.tracked_list):
            track.relations = []
            if track.last_measurement <= (timestamp - cleanupTime):
                remove.append(idx)

        for delete in reversed(remove):
            self.tracked_list.pop(delete)

    def cleanDetections(self):
        """
        Remove duplicate tracks that represent the same physical object.
        
        Uses Mahalanobis distance combined with IOU to identify overlapping tracks,
        then keeps the track with more fusion history (more stable).
        """
        if len(self.tracked_list) < 2:
            return
        
        num_tracks = len(self.tracked_list)
        
        # Build pairwise similarity matrix
        # Only consider upper triangle since similarity is symmetric
        merge_pairs = []  # List of (i, j, cost) where i < j
        
        for i in range(num_tracks):
            track_i = self.tracked_list[i]
            # Get position with covariance inflation
            a_i, b_i, _ = utils.ellipsify(track_i.error_covariance, 2.0)
            bbox_i = [track_i.x, track_i.y, track_i.width + a_i, track_i.length + b_i, track_i.angle]
            
            for j in range(i + 1, num_tracks):
                track_j = self.tracked_list[j]
                a_j, b_j, _ = utils.ellipsify(track_j.error_covariance, 2.0)
                bbox_j = [track_j.x, track_j.y, track_j.width + a_j, track_j.length + b_j, track_j.angle]
                
                # Compute IOU
                iou = utils.rotated_box_iou(bbox_i, bbox_j)
                
                # Compute Mahalanobis distance using combined covariance
                combined_cov = track_i.error_covariance + track_j.error_covariance
                try:
                    S_inv = np.linalg.inv(combined_cov)
                except np.linalg.LinAlgError:
                    S_inv = np.linalg.inv(combined_cov + 1e-6 * np.eye(2))
                
                pos_i = [track_i.x, track_i.y]
                pos_j = [track_j.x, track_j.y]
                mahal_dist = utils.mahalanobis_distance(pos_i, pos_j, S_inv)
                
                # Consider merging if:
                # 1. IOU > 0.3 (decent overlap), OR
                # 2. Mahalanobis distance < 4.0 (statistically close given uncertainties)
                if iou > 0.3 or mahal_dist < 4.0:
                    # Combined score (lower = more similar)
                    cost = (1.0 - iou) * 0.5 + (mahal_dist / 10.0) * 0.5
                    merge_pairs.append((i, j, cost))
        
        # Sort by cost (most similar first)
        merge_pairs.sort(key=lambda x: x[2])
        
        # Greedily merge: keep track with more fusion history
        remove_set = set()
        for i, j, cost in merge_pairs:
            if i in remove_set or j in remove_set:
                continue
            
            track_i = self.tracked_list[i]
            track_j = self.tracked_list[j]
            
            # Only merge if at least one track is established
            if track_i.fusion_steps < self.trackShowThreshold and track_j.fusion_steps < self.trackShowThreshold:
                continue
            
            # Keep the track with more history (more fusion steps)
            # Tiebreaker: more recent last_measurement
            if track_i.fusion_steps > track_j.fusion_steps:
                remove_set.add(j)
            elif track_j.fusion_steps > track_i.fusion_steps:
                remove_set.add(i)
            elif track_i.last_measurement >= track_j.last_measurement:
                remove_set.add(j)
            else:
                remove_set.add(i)
        
        # Remove merged tracks (in reverse order to maintain indices)
        for idx in sorted(remove_set, reverse=True):
            self.tracked_list.pop(idx)
