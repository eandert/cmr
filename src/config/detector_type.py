from enum import Enum

class DetectorType(Enum):
    """
    Enum for detector types (nuScenes-oriented). PERFECT is no error; others use
    regression-tested error models loaded from CSV in data/sensor_models:
    {error_model_name}.csv and {error_model_name}_distributions.csv.
    Replace those CSVs to get new values (new process load, or call error_models.reload_all_models()).
    Polynomial values here are placeholders when error_model_name is set.
    """
    PERFECT = (1, [0.000001], [0.000001], [0.000001], [.999999], None)
    # PointPillars (KITTI-trained, still valid for nuScenes)
    POINTPILLARS_KITTI = (6, [0.01, 0.005], [0.01, 0.005], [0.005, 0.002], [-0.002, 0.9], "pointpillars_kitti")
    # Below models are from mmdetection3d (nuScenes, from mmdetection3d)
    BEV_FUSION = (7, [0.01, 0.005], [0.01, 0.005], [0.005, 0.002], [-0.002, 0.9], "bev_fusion")
    DETR3D = (8, [0.01, 0.005], [0.01, 0.005], [0.005, 0.002], [-0.002, 0.9], "detr3d")   # camera only
    CENTERPOINT = (9, [0.01, 0.005], [0.01, 0.005], [0.005, 0.002], [-0.002, 0.9], "centerpoint")  # lidar only

    def __init__(self, id, centroid_radial_error_polynomial, centroid_distance_error_polynomial, 
                 bounding_box_error_polynomial, detection_probability_polynomial, error_model_name=None):
        """
        Initialize the detector type with its properties.
        
        Args:
            id (int): The ID of the detector type.
            centroid_radial_error_polynomial (list): Coefficients for the centroid radial error polynomial.
            centroid_distance_error_polynomial (list): Coefficients for the centroid distance error polynomial.
            bounding_box_error_polynomial (list): Coefficients for the bounding box error polynomial.
            detection_probability_polynomial (list): Coefficients for the detection probability polynomial.
            error_model_name (str, optional): Name of regression-tested error model to use instead of polynomials.
        """
        self.id = id
        self.centroid_radial_error_polynomial = centroid_radial_error_polynomial
        self.centroid_distance_error_polynomial = centroid_distance_error_polynomial
        self.bounding_box_error_polynomial = bounding_box_error_polynomial
        self.detection_probability_polynomial = detection_probability_polynomial
        self.error_model_name = error_model_name