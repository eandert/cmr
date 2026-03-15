import numpy as np
import math


def _match_detections_to_gt(detected_objects, ground_truth_objects, iou_threshold=0.1,
                            bbox_tolerance=0.5):
    """
    Match detected objects to ground truth using greedy IoU matching.

    Bounding boxes are expanded by bbox_tolerance (as a fraction of each dimension)
    before computing IoU, to accommodate localization error proportionally to vehicle size.
    E.g., bbox_tolerance=0.5 adds 50% to width and height on each side.

    Returns:
        matches: list of (det_idx, gt_idx) pairs
        unmatched_dets: list of det indices (false positives)
        unmatched_gts: list of gt indices (false negatives)
    """
    matched_gt = set()
    matches = []
    unmatched_dets = []

    for d_idx, det in enumerate(detected_objects):
        best_iou = -1.0
        best_gt_idx = -1
        det_bbox = _expand_bbox(det.detected_bbox, det, bbox_tolerance)
        for g_idx, gt in enumerate(ground_truth_objects):
            if g_idx in matched_gt:
                continue
            gt_bbox = _expand_bbox(gt.bbox, gt, bbox_tolerance)
            _iou_value = iou(gt_bbox, det_bbox)
            if _iou_value > iou_threshold and _iou_value > best_iou:
                best_iou = _iou_value
                best_gt_idx = g_idx
        if best_gt_idx >= 0:
            matches.append((d_idx, best_gt_idx))
            matched_gt.add(best_gt_idx)
        else:
            unmatched_dets.append(d_idx)

    unmatched_gts = [i for i in range(len(ground_truth_objects)) if i not in matched_gt]
    return matches, unmatched_dets, unmatched_gts


def _expand_bbox(bbox, obj, tolerance):
    """
    Expand a bounding box by a percentage tolerance.
    Uses the object's centroid, width/length, and angle to rebuild an expanded box.
    Falls back to the original bbox if attributes are missing.
    """
    if bbox is None or tolerance <= 0:
        return bbox
    try:
        cx, cy = obj.centroid[0], obj.centroid[1]
        # Try width/length (GT objects) or width/length from detected objects
        w = getattr(obj, 'width', None)
        l = getattr(obj, 'length', None)
        if w is None or l is None:
            # Try dimensions tuple (some GT objects)
            dims = getattr(obj, 'dimensions', None)
            if dims is not None:
                w, l = dims[0], dims[1]
        if w is None or l is None:
            return bbox
        angle = getattr(obj, 'angle', 0.0) or 0.0
        hw = (w * (1.0 + tolerance)) / 2.0
        hl = (l * (1.0 + tolerance)) / 2.0
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        corners = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
        return [(cx + cos_a * x - sin_a * y, cy + sin_a * x + cos_a * y) for x, y in corners]
    except (AttributeError, TypeError):
        return bbox


def calculate_amota(detected_objects, ground_truth_objects, iou_threshold=0.1,
                    bbox_tolerance=0.5):
    """
    Calculate per-frame MOTA (Multi-Object Tracking Accuracy).

    MOTA = 1 - (FP + FN) / max(1, GT_total)

    Uses IoU matching with bounding box tolerance expansion so the matching
    threshold naturally scales with vehicle size.

    Returns:
        float: MOTA score (can be negative if FP+FN > GT). None if no GT.
    """
    total_ground_truth = len(ground_truth_objects)

    if total_ground_truth == 0:
        return None

    matches, unmatched_dets, unmatched_gts = _match_detections_to_gt(
        detected_objects, ground_truth_objects, iou_threshold,
        bbox_tolerance=bbox_tolerance)

    tp = len(matches)
    fp = len(unmatched_dets)
    fn = len(unmatched_gts)

    # MOTA formula: 1 - (FP + FN + IDSW) / GT
    # Not clamped — negative values indicate more errors than GT objects
    mota = 1.0 - (fp + fn) / max(1, total_ground_truth)

    return mota


def calculate_amotp(detected_objects, ground_truth_objects, iou_threshold=0.1,
                    bbox_tolerance=0.5):
    """
    Calculate per-frame MOTP (Multi-Object Tracking Precision).

    MOTP = average Euclidean center distance of matched (TP) pairs.
    Lower is better. Returns 0.0 if no matches.

    This measures localization quality — how close matched detections are
    to their ground truth. GPEM should directly improve this metric.

    Returns:
        float: Average center distance of matched pairs (meters). 0.0 if no matches.
    """
    if len(detected_objects) == 0 or len(ground_truth_objects) == 0:
        return 0.0

    matches, _, _ = _match_detections_to_gt(
        detected_objects, ground_truth_objects, iou_threshold,
        bbox_tolerance=bbox_tolerance)

    if len(matches) == 0:
        return 0.0

    total_distance = 0.0
    for d_idx, g_idx in matches:
        det = detected_objects[d_idx]
        gt = ground_truth_objects[g_idx]
        dx = det.centroid[0] - gt.centroid[0]
        dy = det.centroid[1] - gt.centroid[1]
        total_distance += math.sqrt(dx * dx + dy * dy)

    return total_distance / len(matches)


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
