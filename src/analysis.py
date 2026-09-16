import uuid
import datetime
from typing import List, Dict, Any, Optional, Tuple
from src.config import CLASS_DISPLAY_NAMES, DEFAULT_IOU_THRESHOLD

def calculate_iou(boxA: Dict[str, float], boxB: Dict[str, float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA['x1'], boxB['x1'])
    yA = max(boxA['y1'], boxB['y1'])
    xB = min(boxA['x2'], boxB['x2'])
    yB = min(boxA['y2'], boxB['y2'])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea == 0.0:
        return 0.0

    boxAArea = (boxA['x2'] - boxA['x1']) * (boxA['y2'] - boxA['y1'])
    boxBArea = (boxB['x2'] - boxB['x1']) * (boxB['y2'] - boxB['y1'])

    denom = boxAArea + boxBArea - interArea
    if denom <= 0.0:
        return 0.0
    return interArea / denom

def compute_spatial_location(bbox: Dict[str, float], img_width: int, img_height: int) -> Tuple[str, str]:
    """
    Deterministically computes the relative image location based on the center of the bounding box
    divided into a standard 3x3 photographic coordinate grid:
    - X-axis: left (< 33.3%), center (33.3% - 66.7%), right (> 66.7%)
    - Y-axis: top/front (< 33.3%), center (33.3% - 66.7%), bottom/rear (> 66.7%)
    """
    if img_width <= 0 or img_height <= 0:
        return ("center", "Detected near the center region of the image.")

    center_x = (bbox['x1'] + bbox['x2']) / 2.0
    center_y = (bbox['y1'] + bbox['y2']) / 2.0

    rel_x = center_x / float(img_width)
    rel_y = center_y / float(img_height)

    # Determine horizontal position
    if rel_x < 0.333:
        h_pos = "left"
    elif rel_x > 0.667:
        h_pos = "right"
    else:
        h_pos = "center"

    # Determine vertical position
    if rel_y < 0.333:
        v_pos = "front"
    elif rel_y > 0.667:
        v_pos = "rear"
    else:
        v_pos = "center"

    if v_pos == "center" and h_pos == "center":
        loc_key = "center"
    elif v_pos == "center":
        loc_key = f"center-{h_pos}"
    elif h_pos == "center":
        loc_key = f"{v_pos}-center"
    else:
        loc_key = f"{v_pos}-{h_pos}"

    description = f"Detected near the {loc_key} region of the image."
    return (loc_key, description)

def group_duplicate_detections(
    detections: List[Dict[str, Any]],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Groups overlapping detections of the same or different damage classes using IoU.
    Retains the detection with highest confidence as primary while noting grouped count.
    """
    if not detections:
        return []

    # Sort descending by confidence
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    kept_detections: List[Dict[str, Any]] = []
    visited = [False] * len(sorted_dets)

    for i in range(len(sorted_dets)):
        if visited[i]:
            continue
        primary = dict(sorted_dets[i])
        visited[i] = True
        overlap_group = [primary]

        for j in range(i + 1, len(sorted_dets)):
            if visited[j]:
                continue
            iou = calculate_iou(primary['bbox'], sorted_dets[j]['bbox'])
            if iou >= iou_threshold:
                visited[j] = True
                overlap_group.append(sorted_dets[j])

        primary['overlap_count'] = len(overlap_group)
        primary['is_grouped'] = len(overlap_group) > 1
        kept_detections.append(primary)

    return kept_detections

class DamageAnalysisEngine:
    """
    Deterministic damage analysis engine that operates strictly on the model detections.
    Adheres to safety constraints: never invents parts, repair costs, or structural severity.
    """

    @staticmethod
    def analyze(
        detections: List[Dict[str, Any]],
        img_width: int,
        img_height: int,
        group_duplicates: bool = False,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        model_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        report_id = f"VDA-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Apply spatial location calculation to all detections
        processed_detections: List[Dict[str, Any]] = []
        for idx, det in enumerate(detections, 1):
            loc_key, loc_desc = compute_spatial_location(det['bbox'], img_width, img_height)
            class_name = det['class_name']
            display_name = CLASS_DISPLAY_NAMES.get(class_name, class_name.replace('_', ' ').title())

            # Bounding box relative coverage percentage
            box_w = det['bbox']['width']
            box_h = det['bbox']['height']
            box_area = box_w * box_h
            total_img_area = max(1.0, float(img_width * img_height))
            area_coverage_pct = round((box_area / total_img_area) * 100, 2)

            processed_detections.append({
                'id': idx,
                'class_name': class_name,
                'display_name': display_name,
                'confidence': det['confidence'],
                'confidence_pct': f"{round(det['confidence'] * 100, 1)}%",
                'bbox': det['bbox'],
                'location': loc_key,
                'location_description': loc_desc,
                'area_coverage_pct': area_coverage_pct,
                # Responsible severity classification: never fabricate unverified claims
                'severity': "Requires physical inspection",
                'detection_status': "Model detected"
            })

        # Duplicate grouping if requested
        if group_duplicates and len(processed_detections) > 1:
            final_detections = group_duplicate_detections(processed_detections, iou_threshold)
            # Re-index
            for idx, det in enumerate(final_detections, 1):
                det['id'] = idx
        else:
            final_detections = processed_detections

        # Damage Breakdown aggregation
        category_map: Dict[str, Dict[str, Any]] = {}
        for det in final_detections:
            c_name = det['class_name']
            if c_name not in category_map:
                category_map[c_name] = {
                    'class_name': c_name,
                    'display_name': det['display_name'],
                    'count': 0,
                    'highest_confidence': 0.0,
                    'locations': []
                }
            category_map[c_name]['count'] += 1
            if det['confidence'] > category_map[c_name]['highest_confidence']:
                category_map[c_name]['highest_confidence'] = det['confidence']
            if det['location'] not in category_map[c_name]['locations']:
                category_map[c_name]['locations'].append(det['location'])

        breakdown = []
        for cat in category_map.values():
            breakdown.append({
                'class_name': cat['class_name'],
                'display_name': cat['display_name'],
                'count': cat['count'],
                'highest_confidence': round(cat['highest_confidence'], 4),
                'highest_confidence_pct': f"{round(cat['highest_confidence'] * 100, 1)}%",
                'regions_present': cat['locations']
            })
        # Sort breakdown by count descending then highest confidence
        breakdown.sort(key=lambda x: (x['count'], x['highest_confidence']), reverse=True)

        # Overall summary statistics
        total_detections = len(final_detections)
        damage_categories_count = len(category_map)
        highest_conf = max([d['confidence'] for d in final_detections], default=0.0)

        # Image quality / resolution assessment
        if img_width >= 1280 and img_height >= 720:
            quality_label = "High Definition (HD+)"
        elif img_width >= 640 and img_height >= 480:
            quality_label = "Standard Resolution"
        else:
            quality_label = "Low Resolution"

        summary = {
            'total_detections': total_detections,
            'damage_categories_count': damage_categories_count,
            'highest_confidence': round(highest_conf, 4),
            'highest_confidence_pct': f"{round(highest_conf * 100, 1)}%" if total_detections > 0 else "N/A",
            'image_quality': quality_label,
            'image_dimensions': f"{img_width} x {img_height} px"
        }

        # Executive summary narrative (strictly deterministic & factual)
        if total_detections == 0:
            executive_summary = (
                "The inspection model analyzed the uploaded vehicle image and detected no visible damage regions "
                "meeting or exceeding the active confidence threshold."
            )
        else:
            cat_phrases = [f"{b['count']} {b['display_name'].lower()}(s)" for b in breakdown]
            categories_str = ", ".join(cat_phrases)
            executive_summary = (
                f"The model detected {total_detections} damage region(s) across {damage_categories_count} damage category/categories: "
                f"{categories_str}. Peak detection confidence is {summary['highest_confidence_pct']}."
            )

        limitations = [
            "Image-based 2D computer vision detection cannot determine sub-surface or internal structural vehicle damage.",
            "Severity and physical repair costs cannot be reliably determined from image detection alone.",
            "Lighting conditions, reflections, dirt, and camera angles may affect detection accuracy."
        ]

        recommendations = [
            "A comprehensive in-person inspection by a certified automotive technician is recommended to verify damage depth and structural integrity.",
            "Capture additional high-resolution images from multiple orthogonal angles and close-up perspectives for comprehensive assessment."
        ]

        return {
            'report_id': report_id,
            'created_at': created_at,
            'image': {
                'width': img_width,
                'height': img_height,
                'resolution': f"{img_width}x{img_height}",
                'quality': quality_label
            },
            'summary': summary,
            'executive_summary': executive_summary,
            'breakdown': breakdown,
            'detections': final_detections,
            'limitations': limitations,
            'recommendations': recommendations,
            'model_info': model_info or {}
        }
