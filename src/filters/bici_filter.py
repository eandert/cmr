"""
Batch Inverse Covariance Intersection (BICI) filter.

Extends the standard CI filter by replacing sequential pairwise measurement
fusion with N-way simultaneous Batch ICI. When multiple detections arrive in
one frame, BICI optimizes all N weights jointly — producing tighter (less
conservative) covariance bounds than sequential pairwise CI.

This lets high-quality close-range GPEM detections maintain more influence
in the fusion, rather than being diluted by pairwise sequential averaging.

For N=1 or N=2 measurements, BICI is equivalent to the standard CI/ICI path.
The advantage appears with N≥3 measurements per frame (common at higher
CAV penetration rates).

References:
  - Liu, Deng & Hu, "Multi-Sensor Fusion Positioning Based on BICI and IMM",
    Applied Sciences 2021 (BICI improves over SICI by 14.3%)
  - Ajgl & Straka, "Inverse Covariance Intersection Fusion of Multiple
    Estimates", FUSION 2020 (N-way ICI generalization)
  - Noack et al., "Inverse Covariance Intersection: New Insights and
    Properties", Fusion 2017 (foundational ICI)
"""

from filters.covariance_intersection import (
    CovarianceIntersectionFilter,
    _bici_fuse_multi,
)


class BICIFilter(CovarianceIntersectionFilter):
    """
    BICI filter — identical to CI filter except multi-measurement fusion
    uses Batch ICI (N-way simultaneous optimization) instead of sequential
    pairwise CI.

    Inherits everything from CovarianceIntersectionFilter:
      - CTRV/CV/CA motion models
      - ICI for prediction-vs-measurement fusion
      - Trust scoring, passthrough, monitoring

    sigma_a is read from filter_config["bici"], independently tunable from CI.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from filters.filter_config import get_sigma_a
        self.sigma_a = get_sigma_a("bici")

    def _fuse_measurements(self, meas_estimates, source_ids=None):
        """
        Override: use Batch ICI for N-way simultaneous fusion.

        For N=1: passthrough (no fusion needed).
        For N=2: delegates to _ici_fuse (equivalent to batch with 2 sources).
        For N≥3: N-dimensional simplex optimization over all weights jointly.
        """
        if len(meas_estimates) == 1:
            return meas_estimates[0]
        return _bici_fuse_multi(meas_estimates)
