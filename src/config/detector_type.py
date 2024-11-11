from enum import Enum

class DetectorType(Enum):
    """
    Enum for different types of detectors with their properties.
    
    Attributes:
        YOLO (tuple): YOLO detector with (id, centroid_radial_error_polynomial, centroid_distance_error_polynomial, bounding_box_error_polynomial, detection_probability_polynomial).
        SSD (tuple): SSD detector with (id, centroid_radial_error_polynomial, centroid_distance_error_polynomial, bounding_box_error_polynomial, detection_probability_polynomial).
        FASTER_RCNN (tuple): Faster R-CNN detector with (id, centroid_radial_error_polynomial, centroid_distance_error_polynomial, bounding_box_error_polynomial, detection_probability_polynomial).
    """
    PERFECT = (1, [0.000001], [0.000001], [0.000001], [.999999]) # (id, centroid_radial_error_polynomial, centroid_distance_error_polynomial, bounding_box_error_polynomial, detection_probability_polynomial)
    YOLO = (5, [0.009, 0.02], [0.009, 0.02], [0.0009, 0.01], [-0.002, 0.7])
    SSD = (3, [0.2, 0.02, 0.002], [0.2, 0.02, 0.002], [0.2, 0.02, 0.002], [0.8, -0.003, 0.0])
    FASTER_RCNN = (4, [0.3, 0.03, 0.003], [0.3, 0.03, 0.003], [0.3, 0.03, 0.003], [0.7, -0.004, 0.0])
    POINT_PILLARS = (5, [0.0018, 0.02], [0.0018, 0.02], [0.0009, 0.01], [-0.002, 0.9])

    def __init__(self, id, centroid_radial_error_polynomial, centroid_distance_error_polynomial, bounding_box_error_polynomial, detection_probability_polynomial):
        """
        Initialize the detector type with its properties.
        
        Args:
            id (int): The ID of the detector type.
            centroid_radial_error_polynomial (list): Coefficients for the centroid radial error polynomial.
            centroid_distance_error_polynomial (list): Coefficients for the centroid distance error polynomial.
            bounding_box_error_polynomial (list): Coefficients for the bounding box error polynomial.
            detection_probability_polynomial (list): Coefficients for the detection probability polynomial.
        """
        self.id = id
        self.centroid_radial_error_polynomial = centroid_radial_error_polynomial
        self.centroid_distance_error_polynomial = centroid_distance_error_polynomial
        self.bounding_box_error_polynomial = bounding_box_error_polynomial
        self.detection_probability_polynomial = detection_probability_polynomial