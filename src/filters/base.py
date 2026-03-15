from abc import ABC, abstractmethod
from typing import Dict, Optional


class FilterBase(ABC):
    """
    Abstract base class for track-level fusion filters.

    A filter is responsible for maintaining the state estimate (position,
    velocity, covariance) of a single tracked object and fusing new
    measurements into that estimate.

    All concrete filters must expose the following attributes after each
    call to fusion():
        x, y          - estimated position
        dx, dy        - estimated velocity components
        error_covariance    - 2x2 position covariance (numpy array)
        d_covariance        - 2x2 velocity covariance (numpy array)
        width, length       - estimated object dimensions
        trupercept_list     - cooperative monitoring list (may be empty)
        error_tracker_temp  - per-sensor error records (may be empty)
        localTrackersIDList - IDs of sensors that contributed this frame
    """

    @abstractmethod
    def fusion(self, measurement_list, time: float, monitor: bool,
               participant_trust_scores: Optional[Dict[int, float]] = None) -> None:
        """
        Fuse measurement_list into the current state estimate.

        Args:
            measurement_list: list of MatchClass objects for this frame
            time: current simulation timestamp
            monitor: if True, populate error_tracker_temp / trupercept_list
            participant_trust_scores: optional dict mapping participant_id → score
        """

    @abstractmethod
    def getKalmanPred(self, time: float):
        """
        Return a predicted position at *time*.

        Returns:
            (x, y, a, b, phi) where a/b are ellipse semi-axes and phi is angle.
        """

    @abstractmethod
    def getKalmanPredWithCovariance(self, time: float):
        """
        Return a predicted position *and* full covariance at *time*.

        Returns:
            (x, y, P) where P is the full state covariance matrix.
        """
