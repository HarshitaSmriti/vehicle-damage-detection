import io
import unittest
import numpy as np
from PIL import Image, ImageDraw
import torch
from fastapi.testclient import TestClient

from src.config import CLASS_NAMES, DEFAULT_CONFIDENCE_THRESHOLD
from src.inference import DamageDetector
from src.analysis import DamageAnalysisEngine, calculate_iou, compute_spatial_location, group_duplicate_detections
from src.annotator import draw_detections_on_image, image_to_base64_uri
from src.pdf_generator import generate_pdf_report
from src.app import app

class TestVehicleDamageDetectionSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = DamageDetector.get_instance()
        cls.client = TestClient(app)

    def test_01_model_loading_and_classes(self):
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.processor)
        self.assertEqual(len(self.detector.id2label), 4)
        expected_classes = {"crack_or_hole", "deformation", "scratch", "windshield_damage"}
        self.assertEqual(set(self.detector.id2label.values()), expected_classes)

    def test_02_spatial_grid_mapping(self):
        # Test 9 grid regions
        w, h = 900, 600
        # 1. Front-left (top-left)
        loc1, desc1 = compute_spatial_location({'x1': 50, 'y1': 50, 'x2': 150, 'y2': 150}, w, h)
        self.assertEqual(loc1, "front-left")
        self.assertIn("front-left", desc1)

        # 2. Center
        loc2, desc2 = compute_spatial_location({'x1': 400, 'y1': 250, 'x2': 500, 'y2': 350}, w, h)
        self.assertEqual(loc2, "center")

        # 3. Rear-right (bottom-right)
        loc3, desc3 = compute_spatial_location({'x1': 700, 'y1': 450, 'x2': 850, 'y2': 550}, w, h)
        self.assertEqual(loc3, "rear-right")

    def test_03_iou_and_duplicate_grouping(self):
        box1 = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
        box2 = {'x1': 110, 'y1': 110, 'x2': 205, 'y2': 205} # High overlap
        box3 = {'x1': 500, 'y1': 500, 'x2': 600, 'y2': 600} # No overlap

        iou_12 = calculate_iou(box1, box2)
        iou_13 = calculate_iou(box1, box3)
        self.assertGreater(iou_12, 0.7)
        self.assertEqual(iou_13, 0.0)

        dets = [
            {'id': 1, 'class_name': 'scratch', 'confidence': 0.85, 'bbox': box1, 'location': 'front-left', 'display_name': 'Scratch'},
            {'id': 2, 'class_name': 'scratch', 'confidence': 0.75, 'bbox': box2, 'location': 'front-left', 'display_name': 'Scratch'},
            {'id': 3, 'class_name': 'deformation', 'confidence': 0.90, 'bbox': box3, 'location': 'rear-right', 'display_name': 'Deformation'}
        ]
        grouped = group_duplicate_detections(dets, iou_threshold=0.6)
        self.assertEqual(len(grouped), 2)
        # Primary scratch should have overlap_count = 2
        scratch_det = next(d for d in grouped if d['class_name'] == 'scratch')
        self.assertEqual(scratch_det['overlap_count'], 2)
        self.assertEqual(scratch_det['confidence'], 0.85)

    def test_04_zero_detections_scenario(self):
        report = DamageAnalysisEngine.analyze(
            detections=[],
            img_width=640,
            img_height=480,
            model_info=self.detector.get_model_info()
        )
        self.assertEqual(report['summary']['total_detections'], 0)
        self.assertEqual(report['summary']['damage_categories_count'], 0)
        self.assertEqual(len(report['breakdown']), 0)
        self.assertIn("detected no visible damage", report['executive_summary'])

    def test_05_large_resolution_image_inference(self):
        # 4K resolution image test
        large_img = Image.new("RGB", (3840, 2160), color=(150, 150, 150))
        draw = ImageDraw.Draw(large_img)
        draw.rectangle([500, 500, 1200, 1000], fill=(20, 20, 20))
        
        detections = self.detector.predict(large_img, confidence_threshold=0.01)
        self.assertIsInstance(detections, list)
        
        # Verify bounding box mapping scales up to 4K dimensions
        for d in detections:
            self.assertLessEqual(d['bbox']['x2'], 3840)
            self.assertLessEqual(d['bbox']['y2'], 2160)

    def test_06_annotation_rendering(self):
        img = Image.new("RGB", (800, 600), color=(200, 200, 200))
        mock_detections = [
            {
                'id': 1,
                'class_name': 'scratch',
                'confidence': 0.92,
                'bbox': {'x1': 100, 'y1': 100, 'x2': 300, 'y2': 250, 'width': 200, 'height': 150}
            }
        ]
        annotated = draw_detections_on_image(img, mock_detections)
        self.assertEqual(annotated.size, img.size)
        uri = image_to_base64_uri(annotated)
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))

    def test_07_pdf_generation(self):
        img = Image.new("RGB", (640, 480), color=(180, 180, 180))
        report = DamageAnalysisEngine.analyze(
            detections=[{
                'id': 1,
                'class_name': 'deformation',
                'confidence': 0.88,
                'bbox': {'x1': 150, 'y1': 100, 'x2': 400, 'y2': 350, 'width': 250, 'height': 250}
            }],
            img_width=640,
            img_height=480,
            model_info=self.detector.get_model_info()
        )
        annotated = draw_detections_on_image(img, report['detections'])
        pdf_bytes = generate_pdf_report(report, img, annotated)
        self.assertGreater(len(pdf_bytes), 2000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_08_api_predict_and_error_handling(self):
        # 1. Valid prediction
        img = Image.new("RGB", (640, 480), color=(120, 120, 120))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        
        response = self.client.post(
            "/api/predict",
            files={"file": ("test.jpg", buf, "image/jpeg")},
            data={"confidence_threshold": "0.10"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("report", data)
        self.assertIn("annotated_image", data)

        # 2. Invalid file upload
        bad_response = self.client.post(
            "/api/predict",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
        )
        self.assertEqual(bad_response.status_code, 400)

    def test_09_safety_and_no_hallucinations(self):
        # Verify the analysis engine never invents severity or fake cost
        report = DamageAnalysisEngine.analyze(
            detections=[{
                'id': 1,
                'class_name': 'crack_or_hole',
                'confidence': 0.95,
                'bbox': {'x1': 50, 'y1': 50, 'x2': 150, 'y2': 150, 'width': 100, 'height': 100}
            }],
            img_width=500,
            img_height=500
        )
        # Severity must be standard and safe
        self.assertEqual(report['detections'][0]['severity'], "Requires physical inspection")
        # Ensure no fake repair dollar amounts or fabricated price estimates exist
        self.assertNotIn("estimated_cost", report)
        self.assertNotIn("price", report)
        self.assertNotIn("$", str(report))

if __name__ == "__main__":
    unittest.main(verbosity=2)
