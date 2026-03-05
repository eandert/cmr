import numpy as np

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


def calculate_amotp(detected_objects, ground_truth_objects, iou_threshold=0.5):
    """
    Calculate the Average Multi-Object Tracking Precision (AMOTP).
    
    AMOTP measures precision - how many detected objects are correct matches.
    Unlike AMOTA which measures recall (how many true objects are detected),
    AMOTP focuses on the quality of detections.
    
    Parameters:
    detected_objects (list): List of detected objects.
    ground_truth_objects (list): List of ground truth objects.
    iou_threshold (float): Intersection over Union (IoU) threshold to consider a match.
    
    Returns:
    float: The AMOTP score (0.0 to 1.0), where 1.0 is perfect precision.
    """
    total_matches = 0
    total_detected = len(detected_objects)
    
    if total_detected == 0:
        return 0.0

    # Create a list to keep track of matched ground truth objects
    matched_gt = [False] * len(ground_truth_objects)

    # Count matches based on IoU threshold
    for det in detected_objects:
        for i, gt in enumerate(ground_truth_objects):
            _iou_value = iou(gt.bbox, det.detected_bbox)
            if not matched_gt[i] and _iou_value > iou_threshold:
                total_matches += 1
                matched_gt[i] = True
                break

    # AMOTP is simply the precision: correct detections / total detections
    amotp = total_matches / total_detected
    
    return amotp


def polygon_area_signed(vertices):
    """
    Calculate the signed area of a polygon using the shoelace formula.
    Positive area = counter-clockwise winding
    Negative area = clockwise winding
    """
    n = len(vertices)
    if n < 3:
        return 0.0
    
    vertices = np.asarray(vertices, dtype=np.float64)
    x = vertices[:, 0]
    y = vertices[:, 1]
    
    # Signed area (positive for CCW, negative for CW)
    area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area


def polygon_area(vertices):
    """
    Calculate the absolute area of a polygon.
    """
    return abs(polygon_area_signed(vertices))


def ensure_ccw(vertices):
    """
    Ensure polygon vertices are in counter-clockwise order.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if len(vertices) < 3:
        return vertices
    
    # Check winding order using signed area
    signed_area = polygon_area_signed(vertices)
    
    # If clockwise (negative area), reverse the order
    if signed_area < 0:
        return vertices[::-1].copy()
    return vertices


def line_intersection(p1, p2, p3, p4):
    """
    Find the intersection point of two line segments.
    """
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return np.array([x, y], dtype=np.float64)


def sutherland_hodgman_clip(subject_polygon, clip_polygon):
    """
    Sutherland-Hodgman polygon clipping algorithm.
    Clips the subject polygon against the clip polygon.
    Both polygons should be in CCW order.
    """
    def inside_edge(point, edge_start, edge_end):
        """Check if point is on the inside (left side for CCW) of the edge."""
        return (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - \
               (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0]) >= -1e-10
    
    # Ensure both polygons are CCW
    subject = ensure_ccw(subject_polygon)
    clip = ensure_ccw(clip_polygon)
    
    if len(subject) == 0 or len(clip) == 0:
        return np.array([])
    
    output = list(subject)
    
    for i in range(len(clip)):
        if len(output) == 0:
            return np.array([])
        
        input_list = output
        output = []
        
        edge_start = clip[i]
        edge_end = clip[(i + 1) % len(clip)]
        
        for j in range(len(input_list)):
            current = np.asarray(input_list[j], dtype=np.float64)
            previous = np.asarray(input_list[j - 1], dtype=np.float64)
            
            current_inside = inside_edge(current, edge_start, edge_end)
            previous_inside = inside_edge(previous, edge_start, edge_end)
            
            if current_inside:
                if not previous_inside:
                    intersection = line_intersection(previous, current, edge_start, edge_end)
                    if intersection is not None:
                        output.append(intersection)
                output.append(current)
            elif previous_inside:
                intersection = line_intersection(previous, current, edge_start, edge_end)
                if intersection is not None:
                    output.append(intersection)
    
    if len(output) < 3:
        return np.array([])
    
    return np.array(output, dtype=np.float64)


def convex_polygon_intersection_area(poly1, poly2):
    """
    Calculate the intersection area of two convex polygons.
    Uses Sutherland-Hodgman clipping algorithm.
    """
    intersection = sutherland_hodgman_clip(poly1, poly2)
    
    if len(intersection) < 3:
        return 0.0
    
    return polygon_area(intersection)


def iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Pure numpy implementation.
    
    Parameters:
    bbox1 (list): Bounding box 1 coordinates (list of 4 corner tuples).
    bbox2 (list): Bounding box 2 coordinates (list of 4 corner tuples).
    
    Returns:
    float: IoU score.
    """
    # Handle None or invalid inputs
    if bbox1 is None or bbox2 is None:
        return 0.0
    
    try:
        # Convert to numpy arrays
        poly1 = np.asarray(bbox1, dtype=np.float64)
        poly2 = np.asarray(bbox2, dtype=np.float64)
        
        if len(poly1) < 3 or len(poly2) < 3:
            return 0.0
        
        # Calculate areas
        area1 = polygon_area(poly1)
        area2 = polygon_area(poly2)
        
        if area1 <= 0 or area2 <= 0:
            return 0.0
        
        # Find intersection area
        intersection_area = convex_polygon_intersection_area(poly1, poly2)
        
        # Calculate union area
        union_area = area1 + area2 - intersection_area
        
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area
        
    except Exception:
        # If anything goes wrong, return 0
        return 0.0
