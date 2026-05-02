"""
Source Reliability LUT for SABRE (Source-Adaptive Batch Reliability Estimation).

Tracks per-source innovation history and computes reliability scores
for omega weight initialization in batch ICI fusion.

Each tracked object maintains its own LUT — the same source may have
different reliability for different tracks depending on distance, angle,
and occlusion conditions.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional


class SourceReliabilityLUT:
    """
    Lookup table mapping source participant IDs to reliability scores.

    Reliability is derived from Normalized Innovation Squared (NIS):
      - NIS ≈ dim(z) → source is well-calibrated → reliability ≈ 1.0
      - NIS >> dim(z) → source surprises filter → reliability < 1.0
      - NIS << dim(z) → source is very predictable → reliability > 1.0

    Scores are smoothed with exponential averaging to prevent oscillation.
    Stale sources (not seen recently) are evicted to bound memory.
    """

    def __init__(self, window_size: int = 10, stale_timeout: int = 50,
                 adapt_rate: float = 0.1, meas_dim: int = 2):
        """
        Args:
            window_size: NIS history length per source.
            stale_timeout: Frames without observation before evicting a source.
            adapt_rate: Exponential smoothing rate for reliability updates.
            meas_dim: Measurement dimension (2 for [x,y], 3 for [x,y,psi]).
        """
        self._sources: Dict[int, dict] = {}
        self._frame_count: int = 0
        self.window_size = window_size
        self.stale_timeout = stale_timeout
        self.adapt_rate = adapt_rate
        self.meas_dim = meas_dim

    def get_omega_priors(self, source_ids: List[int]) -> List[float]:
        """
        Get reliability-based omega priors for a list of source IDs.

        Returns a list of positive floats (one per source) that can be
        normalized to form the initial simplex point for BICI optimization.
        Unknown sources get the default prior (1.0).

        Args:
            source_ids: List of participant IDs, one per measurement.

        Returns:
            List of reliability scores, same length as source_ids.
        """
        priors = []
        for sid in source_ids:
            entry = self._sources.get(sid)
            if entry is not None and entry['n_observations'] >= 3:
                priors.append(entry['reliability'])
            else:
                priors.append(1.0)  # Default for new/unknown sources
        return priors

    def update(self, source_id: int, nis: float, reported_cov: float = None):
        """
        Update a source's reliability based on a new NIS observation.

        Args:
            source_id: Participant ID of the source.
            nis: Normalized Innovation Squared value for this measurement.
            reported_cov: Optional average diagonal of reported R (for monitoring).
        """
        if not np.isfinite(nis) or nis < 0:
            return  # Skip invalid values

        if source_id not in self._sources:
            self._sources[source_id] = {
                'nis_history': deque(maxlen=self.window_size),
                'reliability': 1.0,
                'last_seen': self._frame_count,
                'n_observations': 0,
                'avg_covariance': reported_cov or 0.0,
            }

        entry = self._sources[source_id]
        entry['nis_history'].append(nis)
        entry['last_seen'] = self._frame_count
        entry['n_observations'] += 1

        if reported_cov is not None:
            # Running average of reported covariance
            alpha = 1.0 / entry['n_observations']
            entry['avg_covariance'] = (1 - alpha) * entry['avg_covariance'] + alpha * reported_cov

        # Recompute reliability from NIS history
        if len(entry['nis_history']) >= 3:
            avg_nis = float(np.mean(list(entry['nis_history'])))
            ratio = avg_nis / max(self.meas_dim, 1)
            raw_reliability = 1.0 / max(ratio, 0.1)
            raw_reliability = float(np.clip(raw_reliability, 0.1, 10.0))

            # Exponential smoothing
            entry['reliability'] = (
                (1.0 - self.adapt_rate) * entry['reliability'] +
                self.adapt_rate * raw_reliability
            )

    def step(self):
        """Advance frame counter and evict stale sources."""
        self._frame_count += 1
        self._evict_stale()

    def _evict_stale(self):
        """Remove sources not seen in stale_timeout frames."""
        stale = [sid for sid, entry in self._sources.items()
                 if self._frame_count - entry['last_seen'] > self.stale_timeout]
        for sid in stale:
            del self._sources[sid]

    def get_log(self) -> dict:
        """Return full LUT state for telemetry/debugging."""
        return {
            'frame': self._frame_count,
            'n_sources': len(self._sources),
            'sources': {
                sid: {
                    'reliability': entry['reliability'],
                    'n_observations': entry['n_observations'],
                    'avg_nis': float(np.mean(list(entry['nis_history']))) if entry['nis_history'] else 0,
                    'last_seen': entry['last_seen'],
                    'avg_covariance': entry['avg_covariance'],
                }
                for sid, entry in self._sources.items()
            }
        }

    def get_reliability(self, source_id: int) -> float:
        """Get reliability score for a single source. Returns 1.0 if unknown."""
        entry = self._sources.get(source_id)
        if entry is not None and entry['n_observations'] >= 3:
            return entry['reliability']
        return 1.0

    @property
    def active_sources(self) -> int:
        """Number of currently tracked sources."""
        return len(self._sources)
