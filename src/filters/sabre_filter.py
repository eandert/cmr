"""
SABRE — Source-Adaptive Batch Reliability Estimation.

Extends BICI with per-source NIS tracking via SourceReliabilityLUT.
Each source (participant) gets its own innovation history, which informs
the omega weight initialization for the next frame's batch ICI optimization.

Sources that consistently produce large innovations (unreliable) get
down-weighted; consistent sources get up-weighted. New sources start
with neutral priors and build history over ~15 frames.

When no history exists (first frames, or single-source tracks), SABRE
degrades gracefully to standard BICI behavior.

Key relationship:
  - GPEM provides initial R calibration from offline characterization
  - SABRE refines fusion weights online from actual innovation performance
  - Together: GPEM says "this source should have R=0.25"; SABRE checks
    "does it actually behave like R=0.25?" and adjusts accordingly

References:
  - Liu et al., "BICI and IMM", Applied Sciences 2021
  - Ajgl & Straka, "ICI Fusion of Multiple Estimates", FUSION 2020
  - Noack et al., "Inverse Covariance Intersection", Fusion 2017
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import minimize as scipy_minimize

from filters.bici_filter import BICIFilter
from filters.covariance_intersection import _ici_fuse, _ci_fuse_multi
from filters.source_reliability_lut import SourceReliabilityLUT


def _sabre_fuse(estimates: List[Tuple[np.ndarray, np.ndarray]],
                omega_priors: List[float],
                minimize: str = 'trace') -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch ICI with reliability-informed omega initialization.

    Same N-way ICI formula as _bici_fuse_multi, but starts the optimizer
    from informed priors instead of uniform weights.

    Args:
        estimates: List of (x, P) tuples.
        omega_priors: Reliability scores per source (positive, not necessarily summing to 1).
        minimize: 'trace' or 'det'.

    Returns:
        (x_fused, P_fused)
    """
    if len(estimates) == 0:
        raise ValueError("No estimates to fuse")
    if len(estimates) == 1:
        return estimates[0][0].ravel(), estimates[0][1]

    # N=2: use ICI directly (omega priors bias the initial guess)
    if len(estimates) == 2:
        x_f, P_f, _ = _ici_fuse(estimates[0][0], estimates[0][1],
                                  estimates[1][0], estimates[1][1], minimize)
        return x_f, P_f

    N = len(estimates)
    xs = [np.asarray(x, dtype=float).ravel() for x, _ in estimates]
    Ps = [np.asarray(P, dtype=float) for _, P in estimates]
    P_invs = [np.linalg.inv(P) for P in Ps]
    sum_P_invs = sum(P_invs)

    def objective(omegas_free):
        omega_last = 1.0 - np.sum(omegas_free)
        if omega_last < 0:
            return 1e12
        omegas = np.append(omegas_free, omega_last)
        if np.any(omegas < -1e-10):
            return 1e12
        try:
            P_weighted = sum(w * P for w, P in zip(omegas, Ps))
            common_info = np.linalg.inv(P_weighted)
            P_fused_inv = sum_P_invs - common_info
            P_fused = np.linalg.inv(P_fused_inv)
            if np.any(np.linalg.eigvalsh(P_fused) < -1e-10):
                return 1e12
            return np.trace(P_fused) if minimize == 'trace' else np.linalg.det(P_fused)
        except np.linalg.LinAlgError:
            return 1e12

    # Informed initialization from reliability priors
    priors = np.array(omega_priors[:N], dtype=float)
    priors = np.maximum(priors, 0.01)  # Floor to prevent zeros
    priors = priors / priors.sum()  # Normalize to simplex
    w0 = priors[:N - 1]  # Free parameters (last = 1 - sum)

    bounds = [(0.001, 0.999)] * (N - 1)
    result = scipy_minimize(objective, w0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 50, 'ftol': 1e-8})

    omegas_free = result.x
    omega_last = max(1.0 - np.sum(omegas_free), 0.001)
    omegas = np.append(omegas_free, omega_last)
    omegas = omegas / omegas.sum()

    try:
        P_weighted = sum(w * P for w, P in zip(omegas, Ps))
        common_info = np.linalg.inv(P_weighted)
        P_fused_inv = sum_P_invs - common_info
        P_fused = np.linalg.inv(P_fused_inv)
        if np.any(np.linalg.eigvalsh(P_fused) < -1e-10):
            raise np.linalg.LinAlgError("Non-PSD")

        # Vectorized BICI gain computation
        P_inv_stack = np.array(P_invs)  # (N, d, d)
        xs_stack = np.array(xs)  # (N, d)
        K_terms = P_fused @ (P_inv_stack - omegas[:, np.newaxis, np.newaxis] * common_info)
        x_fused = np.einsum('nij,nj->i', K_terms, xs_stack)

        return x_fused, P_fused
    except np.linalg.LinAlgError:
        return _ci_fuse_multi(estimates, minimize)


class SABREFilter(BICIFilter):
    """
    SABRE: Source-Adaptive Batch Reliability Estimation.

    Extends BICI with per-source NIS tracking via SourceReliabilityLUT.
    The LUT maintains a rolling window of per-source innovation statistics
    and converts them to omega weight priors for the batch ICI optimization.

    Inherits from BICIFilter (which inherits from CovarianceIntersectionFilter):
      - CTRV/CV/CA motion models
      - ICI for prediction-vs-measurement fusion
      - Trust scoring, passthrough, monitoring

    sigma_a is read from filter_config["sabre"].
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from filters.filter_config import get_sigma_a
        self.sigma_a = get_sigma_a("sabre")
        meas_dim = 3 if self.fusion_mode == 2 else 2
        self._reliability_lut = SourceReliabilityLUT(meas_dim=meas_dim)
        # Store last prediction for post-fusion NIS computation
        self._last_z_pred = None
        self._last_S_pred = None
        self._last_source_ids = None
        self._last_valid_measurements = None

    def _fuse_measurements(self, meas_estimates, source_ids=None):
        """
        Override: use reliability-informed omega priors from LUT.

        Falls back to standard BICI when no source IDs available or
        when all sources have equal reliability (no history yet).
        """
        if len(meas_estimates) <= 1:
            return meas_estimates[0] if meas_estimates else (None, None)

        if source_ids is None or len(source_ids) != len(meas_estimates):
            return super()._fuse_measurements(meas_estimates)

        omega_priors = self._reliability_lut.get_omega_priors(source_ids)

        # If all priors are equal (no history), fall back to standard BICI
        if len(set(f'{p:.4f}' for p in omega_priors)) <= 1:
            return super()._fuse_measurements(meas_estimates)

        return _sabre_fuse(meas_estimates, omega_priors)

    def _post_fusion_update(self, source_ids, valid_measurements, z_pred, S_pred):
        """
        After fusion: compute per-source NIS and update the reliability LUT.

        Called from the CI fusion pipeline after the Kalman update.
        Each source gets its own NIS value based on how much its measurement
        deviated from the prediction.
        """
        if S_pred is None or z_pred is None:
            return

        meas_dim = len(z_pred)
        try:
            S_inv = np.linalg.inv(S_pred + 1e-10 * np.eye(meas_dim))
        except np.linalg.LinAlgError:
            return

        # Vectorized NIS computation for all sources
        n_valid = min(len(source_ids), len(valid_measurements))
        if n_valid == 0:
            return
        mu_stack = np.array([valid_measurements[i][:meas_dim] for i in range(n_valid)])  # (M, d)
        nu_stack = mu_stack - z_pred  # (M, d)
        nis_all = np.einsum('mi,ij,mj->m', nu_stack, S_inv, nu_stack)  # (M,)

        for i in range(n_valid):
            self._reliability_lut.update(source_ids[i], float(nis_all[i]))

        self._reliability_lut.step()

    @property
    def reliability_log(self):
        """Get current LUT state for monitoring."""
        return self._reliability_lut.get_log()
