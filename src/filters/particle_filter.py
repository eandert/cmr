"""
Particle Filter — Sequential Importance Resampling (SIR).

A Monte Carlo approach to Bayesian state estimation. Maintains a weighted
set of particles (state hypotheses) and updates them via:
  1. Predict: propagate each particle through the motion model + process noise
  2. Update: weight each particle by measurement likelihood using R
  3. Resample: when effective sample size drops, resample to avoid degeneracy

Reference: Arulampalam, M.S. et al. (2002). "A tutorial on particle filters
for online nonlinear/non-Gaussian Bayesian tracking." IEEE TSP.
"""

import math
import numpy as np
from typing import Dict, Optional

import utils
from filters.base import FilterBase
from filters.kalman_ctrv import max_id


class ParticleFilter(FilterBase):
    """
    SIR Particle Filter implementing the same FilterBase interface as
    ResizableKalman, AdaptiveKalman, and CovarianceIntersectionFilter.
    """

    N_PARTICLES = 500
    RESAMPLE_THRESHOLD_RATIO = 0.75  # resample more aggressively

    def __init__(self, time, x, y, fusion_mode, initial_width=2.0,
                 initial_length=4.5, use_trust_scoring=True,
                 passthrough_covariance=False):
        self.fusion_mode = fusion_mode
        self.use_trust_scoring = use_trust_scoring
        self.passthrough_covariance = passthrough_covariance

        # State dimension
        if fusion_mode == 0:
            self.F_t_len = 4   # [x, y, vx, vy]
        elif fusion_mode == 1:
            self.F_t_len = 6   # [x, y, vx, vy, ax, ay]
        else:
            self.F_t_len = 5   # [x, y, v, psi, psi_dot]

        # Particles: (N, state_dim)
        self.particles = np.zeros((self.N_PARTICLES, self.F_t_len))
        self.particles[:, 0] = x
        self.particles[:, 1] = y
        # Small initial spread — tight around initial position
        self.particles[:, 0] += np.random.normal(0, 0.02, self.N_PARTICLES)
        self.particles[:, 1] += np.random.normal(0, 0.02, self.N_PARTICLES)

        self.weights = np.ones(self.N_PARTICLES) / self.N_PARTICLES

        # Process noise parameter — lower than Kalman because each particle
        # gets independent noise, so the ensemble spread grows faster
        from filters.filter_config import get_sigma_a
        self.sigma_a = get_sigma_a("pf")

        # State estimate (weighted mean)
        self.X_hat_t = np.zeros((self.F_t_len, 1))
        self.X_hat_t[0, 0] = x
        self.X_hat_t[1, 0] = y
        self.P_hat_t = np.eye(self.F_t_len)

        # Required interface attributes
        self.x = x
        self.y = y
        self.dx = 0.0
        self.dy = 0.0
        self.error_covariance = np.eye(2)
        self.d_covariance = np.array([[2.0, 0.0], [0.0, 2.0]])
        self.error_tracker_temp = []
        self.trupercept_list = []
        self.localTrackersIDList = []

        # Dimension tracking (1D Kalman, same as other filters)
        self.width = initial_width
        self.length = initial_length
        self.width_variance = 1.0
        self.length_variance = 1.0
        self.width_process_noise = 0.01
        self.length_process_noise = 0.01

        # Timing
        self.last_update = time
        self.last_measurement = time
        self.idx = 0
        self.time_until_removal = 0.3
        self.min_size = 0.5

        # Measurement list storage (for monitoring / bookkeeping)
        self.localTrackersMeasurementList = []
        self.localTrackersCovarianceList = []
        self.localTrackersHList = []
        self.localTrackersExtraList = []
        self.localTrackersWidthList = []
        self.localTrackersLengthList = []
        self.localTrackersWidthStdList = []
        self.localTrackersLengthStdList = []

        # Last measurement covariance for passthrough mode
        self.last_measurement_cov = None

        # Process noise matrix placeholder
        self.Q_t = self._compute_Q(0.1)
        self.B_t = np.array([[0]] * self.F_t_len, dtype='float')
        self.U_t = 0

    # ------------------------------------------------------------------
    # Motion model helpers (reused from AdaptiveKalman)
    # ------------------------------------------------------------------
    def _build_F(self, elapsed):
        """Build the state transition matrix for the current elapsed time."""
        if self.fusion_mode == 0:
            return np.array([[1, 0, elapsed, 0],
                             [0, 1, 0, elapsed],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype='float')
        elif self.fusion_mode == 1:
            return np.array([[1, 0, elapsed, 0, elapsed*elapsed, 0],
                             [0, 1, 0, elapsed, 0, elapsed*elapsed],
                             [0, 0, 1, 0, elapsed, 0],
                             [0, 0, 0, 1, 0, elapsed],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1]], dtype='float')
        else:
            # CTRV — use weighted mean state for linearization
            x_mean = np.average(self.particles, weights=self.weights, axis=0)
            v_1 = x_mean[2]
            phi_1 = x_mean[3]
            phi_dot_1 = x_mean[4]
            sin_phi = math.sin(phi_1)
            cos_phi = math.cos(phi_1)

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

            return np.array([[1, 0, v_x, phi_x, phi_dot_x],
                             [0, 1, v_y, phi_y, phi_dot_y],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, elapsed],
                             [0, 0, 0, 0, 1]], dtype='float')

    def _compute_Q(self, dt):
        """Compute process noise matrix Q for given timestep dt."""
        sigma_a = self.sigma_a
        sigma_a2 = sigma_a * sigma_a

        if self.fusion_mode == 0:
            dt2 = dt * dt
            dt3 = dt2 * dt
            dt4 = dt3 * dt
            return sigma_a2 * np.array([
                [dt4/4, 0,     dt3/2, 0    ],
                [0,     dt4/4, 0,     dt3/2],
                [dt3/2, 0,     dt2,   0    ],
                [0,     dt3/2, 0,     dt2  ]
            ], dtype='float')
        elif self.fusion_mode == 1:
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
            sigma_v = sigma_a * dt
            sigma_psi = 0.1 * dt
            sigma_psi_dot = 0.5 * dt
            return np.diag([sigma_v**2, sigma_v**2, sigma_v**2,
                            sigma_psi**2, sigma_psi_dot**2])

    def _h_t(self):
        """Observation matrix H (position-only for CV/CA, position+heading for CTRV)."""
        if self.fusion_mode == 0:
            return np.array([[1., 0., 0., 0.],
                             [0., 1., 0., 0.]], dtype='float')
        elif self.fusion_mode == 1:
            return np.array([[1., 0., 0., 0., 0., 0.],
                             [0., 1., 0., 0., 0., 0.]], dtype='float')
        else:
            return np.array([[1., 0., 0., 0., 0.],
                             [0., 1., 0., 0., 0.],
                             [0., 0., 0., 1., 0.]], dtype='float')

    # ------------------------------------------------------------------
    # Measurement frame management (same pattern as ResizableKalman)
    # ------------------------------------------------------------------
    def _addFrames(self, measurement_list):
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
                self.localTrackersMeasurementList.append(
                    np.array([match.x, match.y, match.angle]))
                match_cov = np.asarray(match.covariance, dtype='float')
                cov_3x3 = np.eye(3, dtype='float')
                if match_cov.ndim == 2 and match_cov.shape[0] >= 2 and match_cov.shape[1] >= 2:
                    cov_3x3[0:2, 0:2] = match_cov[0:2, 0:2]
                else:
                    cov_3x3[0:2, 0:2] = np.eye(2) * 0.25
                cov_3x3[2, 2] = match.yaw_variance if match.yaw_variance is not None else 0.1
                self.localTrackersCovarianceList.append(cov_3x3)
            else:
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

    # ------------------------------------------------------------------
    # Core: systematic resampling
    # ------------------------------------------------------------------
    @staticmethod
    def _systematic_resample(weights):
        N = len(weights)
        cumsum = np.cumsum(weights)
        # Guard against floating-point: ensure last element is exactly 1
        cumsum[-1] = 1.0
        u = (np.arange(N) + np.random.uniform()) / N
        indices = np.searchsorted(cumsum, u)
        return indices

    def _compute_n_eff(self):
        return 1.0 / np.sum(self.weights ** 2)

    # ------------------------------------------------------------------
    # Predict particles through motion model
    # ------------------------------------------------------------------
    def _predict_particles(self, elapsed):
        """Propagate all particles through motion model with process noise."""
        Q = self._compute_Q(elapsed)
        self.Q_t = Q

        if self.fusion_mode == 2:
            # CTRV: nonlinear prediction — vectorized
            noise = np.random.multivariate_normal(
                np.zeros(self.F_t_len), Q, self.N_PARTICLES)
            v = self.particles[:, 2]
            psi = self.particles[:, 3]
            psi_dot = self.particles[:, 4]
            psi_new = psi + psi_dot * elapsed

            # Split into near-zero and non-zero yaw rate
            small = np.abs(psi_dot) < 1e-5
            dx = np.empty(self.N_PARTICLES)
            dy = np.empty(self.N_PARTICLES)
            # Near-zero yaw rate: straight-line
            dx[small] = v[small] * np.cos(psi[small]) * elapsed
            dy[small] = v[small] * np.sin(psi[small]) * elapsed
            # Non-zero yaw rate: circular arc
            nz = ~small
            dx[nz] = (v[nz] / psi_dot[nz]) * (np.sin(psi_new[nz]) - np.sin(psi[nz]))
            dy[nz] = (v[nz] / psi_dot[nz]) * (-np.cos(psi_new[nz]) + np.cos(psi[nz]))

            self.particles[:, 0] += dx
            self.particles[:, 1] += dy
            self.particles[:, 3] = psi_new
            self.particles += noise
        else:
            # CV / CA: linear model F @ x + noise
            F = self._build_F(elapsed)
            noise = np.random.multivariate_normal(
                np.zeros(self.F_t_len), Q, self.N_PARTICLES)
            self.particles = (F @ self.particles.T).T + noise

    # ------------------------------------------------------------------
    # Update weights from measurement likelihood
    # ------------------------------------------------------------------
    def _update_weights(self, valid_measurements, valid_covariances):
        """Weight each particle by Gaussian measurement likelihood.

        Fully vectorized across both particles (N) and measurements (M):
          - Batch R inversions: (M, z_dim, z_dim)
          - Batch innovations: (M, N, z_dim)
          - Batch Mahalanobis: einsum over all at once
        """
        if not valid_measurements:
            return

        H = self._h_t()
        z_dim = H.shape[0]
        log_weights = np.log(self.weights + 1e-300)

        # Predicted observations for all particles — computed once (N, z_dim)
        z_pred_all = (H @ self.particles.T).T  # (N, z_dim)

        # Stack all measurements and covariances
        M = len(valid_measurements)
        z_stack = np.array([z[:z_dim] for z in valid_measurements])  # (M, z_dim)
        R_stack = np.array([
            (R[:z_dim, :z_dim] if R.shape[0] >= z_dim else R) + 1e-11 * np.eye(z_dim)
            for R in valid_covariances
        ])  # (M, z_dim, z_dim)

        # Batch invert all R matrices at once
        try:
            R_inv_stack = np.linalg.inv(R_stack)  # (M, z_dim, z_dim)
            log_det_stack = np.linalg.slogdet(R_stack)[1]  # (M,)
        except np.linalg.LinAlgError:
            # Fallback: process one at a time
            for z, R in zip(valid_measurements, valid_covariances):
                z_vec = z[:z_dim]
                R_use = (R[:z_dim, :z_dim] if R.shape[0] >= z_dim else R) + 1e-11 * np.eye(z_dim)
                try:
                    R_inv = np.linalg.inv(R_use)
                    log_det = np.linalg.slogdet(R_use)[1]
                except np.linalg.LinAlgError:
                    continue
                innovations = z_vec - z_pred_all
                mahal = np.sum(innovations @ R_inv * innovations, axis=1)
                log_weights += -0.5 * (mahal + log_det)
            self._normalize_log_weights(log_weights)
            return

        # Batch innovations: (M, N, z_dim)
        # z_stack is (M, z_dim), z_pred_all is (N, z_dim)
        innovations = z_stack[:, np.newaxis, :] - z_pred_all[np.newaxis, :, :]  # (M, N, z_dim)

        # Batch Mahalanobis: for each measurement m, compute d_m^T R_m^{-1} d_m for all N particles
        # innovations @ R_inv gives (M, N, z_dim) @ (M, z_dim, z_dim) -> need einsum
        # mahal[m, n] = innovations[m,n,:] @ R_inv[m,:,:] @ innovations[m,n,:]
        mahal = np.einsum('mni,mij,mnj->mn', innovations, R_inv_stack, innovations)  # (M, N)

        # Accumulate log-likelihoods across all measurements
        log_weights += np.sum(-0.5 * (mahal + log_det_stack[:, np.newaxis]), axis=0)  # (N,)

        self._normalize_log_weights(log_weights)

    def _normalize_log_weights(self, log_weights):
        """Normalize log weights to proper probability distribution."""

        # Normalize in log space for numerical stability
        max_log = np.max(log_weights)
        log_weights -= max_log
        self.weights = np.exp(log_weights)
        w_sum = np.sum(self.weights)
        if w_sum > 0:
            self.weights /= w_sum
        else:
            self.weights = np.ones(self.N_PARTICLES) / self.N_PARTICLES

    # ------------------------------------------------------------------
    # Extract state estimate from weighted particles
    # ------------------------------------------------------------------
    def _estimate_state(self):
        """Compute weighted mean and covariance from particles."""
        x_hat = np.average(self.particles, weights=self.weights, axis=0)
        diff = self.particles - x_hat  # (N, state_dim)
        # Vectorized weighted covariance: P = diff^T @ diag(weights) @ diff
        weighted_diff = diff * self.weights[:, np.newaxis]  # (N, state_dim)
        P = weighted_diff.T @ diff  # (state_dim, state_dim)

        self.X_hat_t = x_hat.reshape(-1, 1)
        self.P_hat_t = P

        # Position and velocity extraction
        self.x = x_hat[0]
        self.y = x_hat[1]
        if self.fusion_mode == 2:
            self.dx = x_hat[2] * math.cos(x_hat[3])
            self.dy = x_hat[2] * math.sin(x_hat[3])
        else:
            self.dx = x_hat[2]
            self.dy = x_hat[3] if self.F_t_len > 3 else 0.0

        # 2x2 position covariance
        self.error_covariance = P[0:2, 0:2].copy()
        # 2x2 velocity covariance
        self.d_covariance = P[2:4, 2:4].copy()

    # ------------------------------------------------------------------
    # FilterBase: fusion
    # ------------------------------------------------------------------
    def fusion(self, measurement_list, time, monitor,
               participant_trust_scores: Optional[Dict[int, float]] = None):
        self._addFrames(measurement_list)

        if self.idx == 0:
            # First frame: initialize from measurements
            if len(self.localTrackersMeasurementList) > 0:
                positions = [m[:2] for m in self.localTrackersMeasurementList]
                pos = np.mean(positions, axis=0)
                self.x = pos[0]
                self.y = pos[1]

                # Spread particles tightly around first measurement
                self.particles[:, 0] = pos[0] + np.random.normal(0, 0.02, self.N_PARTICLES)
                self.particles[:, 1] = pos[1] + np.random.normal(0, 0.02, self.N_PARTICLES)

                if self.fusion_mode == 2 and len(self.localTrackersMeasurementList[0]) > 2:
                    heading = self.localTrackersMeasurementList[0][2]
                    self.particles[:, 3] = heading + np.random.normal(0, 0.02, self.N_PARTICLES)

                # Seed covariance from measurement
                if len(self.localTrackersCovarianceList) > 0:
                    cov = self.localTrackersCovarianceList[0]
                    self.error_covariance = np.array([
                        [cov[0, 0], cov[0, 1]],
                        [cov[1, 0], cov[1, 1]]], dtype='float')

            self.X_hat_t[0, 0] = self.x
            self.X_hat_t[1, 0] = self.y
            self.last_update = time
            self.dx = 0.0
            self.dy = 0.0
            self.idx += 1
            return

        # --- Elapsed time ---
        elapsed = time - self.last_update
        if elapsed <= 0.0:
            elapsed = 0.1

        # --- Predict ---
        self._predict_particles(elapsed)

        # --- Collect valid measurements ---
        valid_measurements = []
        valid_covariances = []
        for mu, cov, tracker_id in zip(
                self.localTrackersMeasurementList,
                self.localTrackersCovarianceList,
                self.localTrackersIDList):
            if np.any(np.isnan(mu)) or np.any(np.isnan(cov)):
                continue
            if np.any(np.isinf(mu)) or np.any(np.isinf(cov)):
                continue

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

            min_cov = 1e-11
            adjusted_cov = adjusted_cov + min_cov * np.eye(adjusted_cov.shape[0])
            valid_measurements.append(mu)
            valid_covariances.append(adjusted_cov)

        # --- Update weights ---
        if len(valid_measurements) > 0:
            self._update_weights(valid_measurements, valid_covariances)

            # Store last measurement covariance
            if valid_covariances[-1].shape[0] >= 2:
                self.last_measurement_cov = valid_covariances[-1][0:2, 0:2].copy()

        # --- Resample if needed ---
        n_eff = self._compute_n_eff()
        if n_eff < self.N_PARTICLES * self.RESAMPLE_THRESHOLD_RATIO:
            indices = self._systematic_resample(self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.N_PARTICLES) / self.N_PARTICLES
            # Add small jitter to prevent degeneracy
            jitter_std = 0.002
            self.particles += np.random.normal(0, jitter_std,
                                               self.particles.shape)

        # --- Extract state estimate ---
        self._estimate_state()

        # --- Passthrough covariance mode ---
        if self.passthrough_covariance and self.last_measurement_cov is not None:
            self.error_covariance = self.last_measurement_cov.copy()
            self.P_hat_t[0:2, 0:2] = self.last_measurement_cov

        # --- Monitoring / bookkeeping ---
        self.error_tracker_temp = []
        length = len(self.localTrackersIDList)
        if monitor:
            H = self._h_t()
            if self.P_hat_t[0][0] != 0.0 and self.P_hat_t[0][1] != 0.0:
                global_error_x, global_error_y, global_error_angle = utils.ellipsify(
                    self.P_hat_t[0:2, 0:2], 1.0)
            else:
                global_error_x = global_error_y = global_error_angle = 0.0

            for id_, mu, cov in zip(
                    self.localTrackersIDList,
                    self.localTrackersMeasurementList,
                    self.localTrackersCovarianceList):
                z_dim = H.shape[0]
                z_pred = (H @ self.X_hat_t).flatten()
                residual = mu[:z_dim] - z_pred
                location_error = math.hypot(residual[0], residual[1])
                expected_a, expected_b, expected_angle = utils.ellipsify(cov, 1.0)
                expected_x = utils.calculateRadiusAtAngle(
                    expected_a, expected_b, expected_angle, math.radians(0))
                expected_y = utils.calculateRadiusAtAngle(
                    expected_a, expected_b, expected_angle, math.radians(90))
                expected_location_error = math.hypot(
                    expected_x, expected_y) - math.hypot(global_error_x, global_error_y)
                if expected_location_error <= 0.0 or math.isnan(expected_location_error):
                    expected_location_error = max(0.01, math.hypot(expected_x, expected_y))
                location_error_std = location_error / expected_location_error
                if math.isnan(location_error_std) or math.isinf(location_error_std):
                    continue
                self.error_tracker_temp.append([id_, location_error_std, length])

            trupercept_list = []
            for id_test, confidence_test in zip(
                    self.localTrackersIDList, self.localTrackersExtraList):
                trupercept_list.append([id_test, confidence_test[0]])
            self.trupercept_list = trupercept_list

        # --- Width/length 1D Kalman ---
        self.width_variance += self.width_process_noise
        self.length_variance += self.length_process_noise
        if len(self.localTrackersWidthList) > 0:
            for w_meas, l_meas, w_std, l_std in zip(
                    self.localTrackersWidthList, self.localTrackersLengthList,
                    self.localTrackersWidthStdList, self.localTrackersLengthStdList):
                w_noise = max(1e-4, w_std**2)
                width_gain = self.width_variance / (self.width_variance + w_noise)
                self.width += width_gain * (w_meas - self.width)
                self.width_variance = (1 - width_gain) * self.width_variance
                l_noise = max(1e-4, l_std**2)
                length_gain = self.length_variance / (self.length_variance + l_noise)
                self.length += length_gain * (l_meas - self.length)
                self.length_variance = (1 - length_gain) * self.length_variance

        self.last_update = time
        self.idx += 1

    # ------------------------------------------------------------------
    # FilterBase: getKalmanPred
    # ------------------------------------------------------------------
    def getKalmanPred(self, time):
        elapsed = time - self.last_update
        if elapsed <= 0:
            px, py = self.x, self.y
            P = self.P_hat_t
        else:
            # Propagate mean state (not particles — too expensive for gating)
            F = self._build_F(elapsed)
            Q = self._compute_Q(elapsed)
            X_pred = F @ self.X_hat_t
            P = F @ self.P_hat_t @ F.T + Q
            px = float(X_pred[0, 0])
            py = float(X_pred[1, 0])

        a, b, phi = utils.ellipsify(P[0:2, 0:2], 1.0)
        return px, py, a, b, phi

    # ------------------------------------------------------------------
    # FilterBase: getKalmanPredWithCovariance
    # ------------------------------------------------------------------
    def getKalmanPredWithCovariance(self, time):
        elapsed = time - self.last_update
        if elapsed <= 0:
            return self.x, self.y, self.P_hat_t
        else:
            F = self._build_F(elapsed)
            Q = self._compute_Q(elapsed)
            X_pred = F @ self.X_hat_t
            P = F @ self.P_hat_t @ F.T + Q
            return float(X_pred[0, 0]), float(X_pred[1, 0]), P
