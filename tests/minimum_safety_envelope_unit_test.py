import sys
import unittest
from pathlib import Path

# Make src/ importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ground_truth import GroundTruthObject  # noqa: E402
from metrics.minimum_safety_envelope import calculate_mse_violations  # noqa: E402

# Unit tests
class TestCalculateMSEViolations(unittest.TestCase):
    def setUp(self):
        self.ground_truth_objects = [
            GroundTruthObject("veh1", "veh_passenger", (100, 100), (0, 0), [], 0, 1.8, 5.0),
            GroundTruthObject("veh2", "veh_passenger", (200, 200), (0, 0), [], 0, 1.8, 5.0),
            GroundTruthObject("veh3", "veh_passenger", (300, 300), (0, 0), [], 0, 1.8, 5.0)
        ]
        self.ego_vehicle = GroundTruthObject("veh1", "veh_passenger", (100, 100), (0, 0), [], 0, 1.8, 5.0)

    def test_no_violations(self):
        violations = calculate_mse_violations(self.ego_vehicle, self.ground_truth_objects, category="Aggressive")
        self.assertEqual(violations, 0)

    def test_one_violation(self):
        self.ground_truth_objects[1].centroid = (105, 100)  # Close to ego vehicle
        violations = calculate_mse_violations(self.ego_vehicle, self.ground_truth_objects, category="Aggressive")
        self.assertEqual(violations, 1)

    def test_multiple_violations(self):
        self.ground_truth_objects[1].centroid = (105, 100)  # Close to ego vehicle
        self.ground_truth_objects[2].centroid = (110, 100)  # Close to ego vehicle
        violations = calculate_mse_violations(self.ego_vehicle, self.ground_truth_objects, category="Aggressive")
        self.assertEqual(violations, 2)

    def test_different_category(self):
        self.ground_truth_objects[1].centroid = (105, 100)  # Close to ego vehicle
        violations = calculate_mse_violations(self.ego_vehicle, self.ground_truth_objects, category="Conservative")
        self.assertEqual(violations, 1)

if __name__ == "__main__":
    unittest.main()