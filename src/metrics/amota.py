from shapely.geometry import Polygon

def calculate_amota(detected_objects, ground_truth_objects, iou_threshold=0.5):
    """
    Calculate the Average Multi-Object Tracking Accuracy (AMOTA).
    
    Parameters:
    detected_objects (list): List of detected objects.
    ground_truth_objects (list): List of ground truth objects.
    iou_threshold (float): Intersection over Union (IoU) threshold to consider a match.
    
    Returns:
    float: The AMOTA score.
    """
    total_matches = 0
    total_ground_truth = len(ground_truth_objects)
    total_detected = len(detected_objects)
    
    if total_ground_truth == 0:
        return 0.0

    # Create a list to keep track of matched ground truth objects
    matched_gt = [False] * total_ground_truth

    # Example: Count matches based on some criteria (e.g., IoU threshold)
    for det in detected_objects:
        for i, gt in enumerate(ground_truth_objects):
            _iou_value = iou(gt.bbox, det.detected_bbox)
            if not matched_gt[i] and _iou_value > iou_threshold:
                total_matches += 1
                matched_gt[i] = True
                break

    # Calculate precision and recall
    precision = total_matches / total_detected if total_detected > 0 else 0
    recall = total_matches / total_ground_truth

    # Calculate AMOTA score
    amota = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return amota

def iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Parameters:
    bbox1 (list): Bounding box 1 coordinates.
    bbox2 (list): Bounding box 2 coordinates.
    
    Returns:
    float: IoU score.
    """
    poly1 = Polygon(bbox1)
    poly2 = Polygon(bbox2)
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    return intersection_area / union_area if union_area != 0 else 0