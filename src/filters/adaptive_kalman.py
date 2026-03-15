"""
Adaptive Kalman Filter — Innovation-Based Adaptive Estimation (IAE).

Extends the standard ResizableKalman filter by adapting the process noise
magnitude (sigma_a) online based on the innovation (measurement residual)
sequence.  When the filter is well-tuned, the normalised innovation squared
(NIS) should average to the measurement dimension.  If innovations are
consistently larger than predicted, the filter is overconfident and sigma_a
is increased; if smaller, sigma_a is decreased.

Reference: Mehra, R. (1970). "On the identification of variances and
adaptive Kalman filtering." IEEE Transactions on Automatic Control.
"""

import math
import numpy as np
from collections import deque
from typing import Dict, Optional

import utils
from filters.kalman_ctrv import ResizableKalman


class AdaptiveKalman(ResizableKalman):
    """
    Drop-in replacement for ResizableKalman with online Q adaptation.

    The innovation sequence {nu_k} is tracked in a sliding window.
    Every fusion step we compute the sample innovation covariance and
    compare it to the filter-predicted innovation covariance S_k.
    sigma_a is scaled so that the two converge.
    """

    # --- tunables ---
    WINDOW_SIZE = 20          # number of recent innovations to keep
    ADAPT_ALPHA = 0.15        # exponential smoothing for sigma_a updates
    SIGMA_A_MIN = 0.2         # lower bound on sigma_a (m/s^2)
    SIGMA_A_MAX = 15.0        # upper bound on sigma_a (m/s^2)
    WARMUP_STEPS = 5          # use default sigma_a until this many updates

    def __init__(self, time, x, y, fusion_mode, initial_width=2.0,
                 initial_length=4.5, use_trust_scoring=True,
                 passthrough_covariance=False):
        super().__init__(time, x, y, fusion_mode,
                         initial_width=initial_width,
                         initial_length=initial_length,
                         use_trust_scoring=use_trust_scoring,
                         passthrough_covariance=passthrough_covariance)

        # Innovation history (each entry: (nu, S) where nu is innovation vector, S is predicted cov)
        self._innovation_window = deque(maxlen=self.WINDOW_SIZE)
        self._adapt_step = 0

    # ------------------------------------------------------------------
    # Override fusion to inject innovation tracking + Q adaptation
    # ------------------------------------------------------------------
    def fusion(self, measurement_list, time, monitor,
               participant_trust_scores: Dict[int, float] = None):
        """
        Same contract as ResizableKalman.fusion(), with added Q adaptation.
        """
        # On the very first frame, delegate entirely to the parent (initialisation)
        if self.idx == 0:
            super().fusion(measurement_list, time, monitor, participant_trust_scores)
            return

        # ---- Prediction step (replicates parent logic so we can capture pre-update state) ----
        self.addFrames(measurement_list)

        elapsed = time - self.last_update
        if elapsed <= 0.0:
            elapsed = 0.1

        # Build F_t (transition matrix) — same logic as parent
        self.F_t = self._build_F(elapsed)

        # Compute Q with current (possibly adapted) sigma_a
        self.Q_t = self._compute_Q(elapsed)

        # Predict
        self.X_hat_t, self.P_hat_t = utils.kalman_prediction(
            self.X_hat_t, self.P_hat_t, self.F_t, self.B_t, self.U_t, self.Q_t)

        # ---- Collect valid measurements (same filtering as parent) ----
        valid_measurements = []
        valid_covariances = []

        for mu, cov, h_t_type, tracker_id in zip(
                self.localTrackersMeasurementList,
                self.localTrackersCovarianceList,
                self.localTrackersHList,
                self.localTrackersIDList):
            if np.any(np.isnan(mu)) or np.any(np.isnan(cov)):
                continue
            if np.any(np.isinf(mu)) or np.any(np.isinf(cov)):
                continue

            if self.use_trust_scoring and participant_trust_scores is not None:
                from filters.kalman_ctrv import max_id
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

        added = len(valid_measurements)

        # ---- Innovation tracking (before the Kalman update changes X_hat/P_hat) ----
        if added > 0:
            H = self.h_t(0)
            z_pred = H.dot(self.X_hat_t).flatten()
            S_pred = H.dot(self.P_hat_t).dot(H.T)

            # Use average measurement as the representative innovation source
            z_avg = np.mean([mu[:H.shape[0]] for mu in valid_measurements], axis=0)
            R_avg = np.mean([c[:H.shape[0], :H.shape[0]] for c in valid_covariances], axis=0)
            S_total = S_pred + R_avg

            nu = z_avg - z_pred  # innovation vector
            self._innovation_window.append((nu, S_total))
            self._adapt_step += 1

        # ---- Measurement update ----
        if added > 0:
            if self.precision_weighted_fusion:
                self._precision_weighted_update(valid_measurements, valid_covariances, elapsed)
            else:
                for mu, adjusted_cov in zip(valid_measurements, valid_covariances):
                    Z_t = mu.transpose().reshape(-1, 1)
                    self.X_hat_t, self.P_hat_t = utils.kalman_update(
                        self.X_hat_t, self.P_hat_t, Z_t, adjusted_cov, self.h_t(0))
                    if adjusted_cov.shape[0] >= 2:
                        self.last_measurement_cov = adjusted_cov[0:2, 0:2].copy()

                if self.passthrough_covariance and self.last_measurement_cov is not None:
                    self.P_hat_t[0:2, 0:2] = self.last_measurement_cov
        else:
            # No measurements — zero-gain update
            self._zero_gain_update()

        # ---- Adapt sigma_a from innovation statistics ----
        if self._adapt_step >= self.WARMUP_STEPS and len(self._innovation_window) >= 3:
            self._adapt_process_noise()

        # ---- Post-update bookkeeping (same as parent) ----
        self._post_update_bookkeeping(time, monitor, elapsed, valid_measurements, valid_covariances, added)

    # ------------------------------------------------------------------
    # Q adaptation core
    # ------------------------------------------------------------------
    def _adapt_process_noise(self):
        """
        Adapt sigma_a based on normalised innovation squared (NIS).

        NIS_k = nu_k^T  S_k^{-1}  nu_k

        For a well-tuned filter, E[NIS] = dim(z).  If the observed average
        NIS is larger, we increase sigma_a (filter is too confident).
        """
        dim_z = self._innovation_window[0][0].shape[0]
        nis_values = []

        for nu, S in self._innovation_window:
            try:
                S_inv = np.linalg.inv(S + 1e-10 * np.eye(S.shape[0]))
                nis = float(nu.T @ S_inv @ nu)
                if not (np.isnan(nis) or np.isinf(nis)):
                    nis_values.append(nis)
            except np.linalg.LinAlgError:
                continue

        if len(nis_values) < 3:
            return

        avg_nis = np.mean(nis_values)
        # Ratio: >1 means filter is overconfident, <1 means underconfident
        ratio = avg_nis / dim_z

        # Scale sigma_a: if ratio > 1, increase; if < 1, decrease
        # Use sqrt because Q ~ sigma_a^2, so NIS ~ sigma_a^2 contribution
        scale = math.sqrt(max(ratio, 0.01))

        # Exponential smoothing
        new_sigma_a = self.sigma_a * (1.0 - self.ADAPT_ALPHA) + \
                      (self.sigma_a * scale) * self.ADAPT_ALPHA

        # Clamp
        self.sigma_a = max(self.SIGMA_A_MIN, min(self.SIGMA_A_MAX, new_sigma_a))

    # ------------------------------------------------------------------
    # Helper: build transition matrix F (extracted from parent to avoid duplication)
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
            # CTRV model
            v_1 = float(self.X_hat_t[2, 0])
            phi_1 = float(self.X_hat_t[3, 0])
            phi_dot_1 = float(self.X_hat_t[4, 0])
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

    # ------------------------------------------------------------------
    # Helpers extracted from parent's fusion() for reuse
    # ------------------------------------------------------------------
    def _precision_weighted_update(self, valid_measurements, valid_covariances, elapsed):
        """Precision-weighted mean fusion (same logic as parent)."""
        positions = [mu[:2] for mu in valid_measurements]
        pos_covs = [cov[:2, :2] if cov.shape[0] >= 2 else cov for cov in valid_covariances]

        precision_sum = np.zeros((2, 2))
        weighted_sum = np.zeros(2)
        for pc, pos in zip(pos_covs, positions):
            try:
                prec = np.linalg.inv(pc)
            except np.linalg.LinAlgError:
                prec = np.linalg.inv(pc + 0.01 * np.eye(2))
            precision_sum += prec
            weighted_sum += prec @ pos

        try:
            fused_cov = np.linalg.inv(precision_sum)
        except np.linalg.LinAlgError:
            fused_cov = np.eye(2) * 0.1

        fused_pos = fused_cov @ weighted_sum
        self.X_hat_t[0, 0] = fused_pos[0]
        self.X_hat_t[1, 0] = fused_pos[1]
        self.P_hat_t[0:2, 0:2] = fused_cov
        self.last_measurement_cov = fused_cov.copy()

        if hasattr(self, '_last_fused_pos') and self._last_fused_pos is not None:
            if elapsed > 0.001:
                vel_x = (fused_pos[0] - self._last_fused_pos[0]) / elapsed
                vel_y = (fused_pos[1] - self._last_fused_pos[1]) / elapsed
                alpha = 0.3
                if self.fusion_mode == 0 or self.fusion_mode == 1:
                    self.X_hat_t[2, 0] = alpha * vel_x + (1 - alpha) * self.X_hat_t[2, 0]
                    self.X_hat_t[3, 0] = alpha * vel_y + (1 - alpha) * self.X_hat_t[3, 0]
                elif self.fusion_mode == 2:
                    speed = math.hypot(vel_x, vel_y)
                    self.X_hat_t[2, 0] = alpha * speed + (1 - alpha) * self.X_hat_t[2, 0]
        self._last_fused_pos = fused_pos.copy()

    def _zero_gain_update(self):
        """Zero-gain update when no measurements available."""
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

    def _post_update_bookkeeping(self, time, monitor, elapsed,
                                  valid_measurements, valid_covariances, added):
        """Post-update state extraction and monitoring (same as parent)."""
        self.error_tracker_temp = []
        length = len(self.localTrackersIDList)
        if monitor:
            if self.P_hat_t[0][0] != 0.0 and self.P_hat_t[0][1] != 0.0:
                global_error_x, global_error_y, global_error_angle = utils.ellipsify(
                    [[self.P_hat_t[0][0], self.P_hat_t[0][1]],
                     [self.P_hat_t[1][0], self.P_hat_t[1][1]]], 1.0)
            else:
                global_error_x = global_error_y = global_error_angle = 0.0
            for id_, mu, cov, h_t_type in zip(
                    self.localTrackersIDList, self.localTrackersMeasurementList,
                    self.localTrackersCovarianceList, self.localTrackersHList):
                Z_t = mu.transpose().reshape(-1, 1)
                y_t_temp = Z_t - self.h_t(h_t_type).dot(self.X_hat_t)
                location_error = math.hypot(y_t_temp[0], y_t_temp[1])
                expected_a, expected_b, expected_angle = utils.ellipsify(cov, 1.0)
                expected_x = utils.calculateRadiusAtAngle(expected_a, expected_b, expected_angle, math.radians(0))
                expected_y = utils.calculateRadiusAtAngle(expected_a, expected_b, expected_angle, math.radians(90))
                expected_location_error = math.hypot(expected_x, expected_y) - math.hypot(global_error_x, global_error_y)
                if expected_location_error <= 0.0 or math.isnan(expected_location_error):
                    expected_location_error = max(0.01, math.hypot(expected_x, expected_y))
                location_error_std = location_error / expected_location_error
                if math.isnan(location_error_std) or math.isinf(location_error_std):
                    continue
                self.error_tracker_temp.append([id_, location_error_std, length])

            trupercept_list = []
            for id_test, mu_test, confidence_test in zip(
                    self.localTrackersIDList, self.localTrackersMeasurementList,
                    self.localTrackersExtraList):
                trupercept_list.append([id_test, confidence_test[0]])
            self.trupercept_list = trupercept_list

        self.last_update = time
        self.x = self.X_hat_t[0][0]
        self.y = self.X_hat_t[1][0]
        if self.fusion_mode == 2:
            self.dx = self.X_hat_t[2][0] * math.cos(self.X_hat_t[3][0])
            self.dy = self.X_hat_t[2][0] * math.sin(self.X_hat_t[3][0])
        else:
            self.dx = self.X_hat_t[2][0]
            self.dy = self.X_hat_t[3][0]
        self.idx += 1
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

        # 1D Kalman update for width and length
        self.width_variance += self.width_process_noise
        self.length_variance += self.length_process_noise
        if len(self.localTrackersWidthList) > 0:
            for w_meas, l_meas, w_std, l_std in zip(
                    self.localTrackersWidthList, self.localTrackersLengthList,
                    self.localTrackersWidthStdList, self.localTrackersLengthStdList):
                w_noise = max(1e-4, w_std**2)
                width_gain = self.width_variance / (self.width_variance + w_noise)
                self.width = self.width + width_gain * (w_meas - self.width)
                self.width_variance = (1 - width_gain) * self.width_variance
                l_noise = max(1e-4, l_std**2)
                length_gain = self.length_variance / (self.length_variance + l_noise)
                self.length = self.length + length_gain * (l_meas - self.length)
                self.length_variance = (1 - length_gain) * self.length_variance
