import os
import sys
import json
from PIL import Image, ImageDraw

def validate():
    print("=" * 50)
    print("RUNNING VEHICLE DAMAGE DETECTION VALIDATION SUITE")
    print("=" * 50)

    # 1. Config Loading
    try:
        from src.config import APP_CONFIG, CONFIG_JSON_PATH
        assert APP_CONFIG is not None
        assert "num_classes" in APP_CONFIG or "checkpoint_base" in APP_CONFIG
        print("Config: OK")
    except Exception as e:
        print(f"Config: FAILED ({e})")
        sys.exit(1)

    # 2. Classes Loading
    try:
        from src.config import CLASS_NAMES
        assert len(CLASS_NAMES) >= 4
        assert 0 in CLASS_NAMES or "0" in CLASS_NAMES
        print("Classes: OK")
    except Exception as e:
        print(f"Classes: FAILED ({e})")
        sys.exit(1)

    # 3. Model Loading
    try:
        from src.inference import DamageDetector
        detector = DamageDetector.get_instance()
        assert detector.model is not None
        assert detector.processor is not None
        print("Model: OK")
    except Exception as e:
        print(f"Model: FAILED ({e})")
        sys.exit(1)

    # 4. Inference on Sample Image
    try:
        # Create a synthetic test image with vehicle-like dimensions
        test_img = Image.new("RGB", (800, 600), color=(100, 110, 120))
        draw = ImageDraw.Draw(test_img)
        # Add shapes simulating damage regions
        draw.rectangle([150, 150, 350, 300], fill=(70, 70, 70), outline=(200, 50, 50))
        draw.line([400, 200, 600, 350], fill=(220, 220, 220), width=4)
        
        detections = detector.predict(test_img, confidence_threshold=0.01)
        assert isinstance(detections, list)
        print("Inference: OK")
    except Exception as e:
        print(f"Inference: FAILED ({e})")
        sys.exit(1)

    # 5. Postprocessing & Spatial Analysis
    try:
        from src.analysis import DamageAnalysisEngine
        report = DamageAnalysisEngine.analyze(
            detections=detections,
            img_width=test_img.width,
            img_height=test_img.height,
            model_info=detector.get_model_info()
        )
        assert "report_id" in report
        assert "summary" in report
        assert "detections" in report
        assert "breakdown" in report
        
        # Verify bounding box validity if detections exist
        for det in report["detections"]:
            bbox = det["bbox"]
            assert 0 <= bbox["x1"] <= test_img.width
            assert 0 <= bbox["y1"] <= test_img.height
            assert bbox["x2"] >= bbox["x1"]
            assert bbox["y2"] >= bbox["y1"]
            assert "location" in det
            assert "location_description" in det
        print("Postprocessing: OK")
    except Exception as e:
        print(f"Postprocessing: FAILED ({e})")
        sys.exit(1)

    # 6. Report Generation & PDF Validation
    try:
        from src.annotator import draw_detections_on_image
        from src.pdf_generator import generate_pdf_report
        
        annotated_img = draw_detections_on_image(test_img, report["detections"])
        pdf_bytes = generate_pdf_report(report, test_img, annotated_img)
        assert len(pdf_bytes) > 1000
        assert pdf_bytes.startswith(b"%PDF")
        print("Report generation: OK")
    except Exception as e:
        print(f"Report generation: FAILED ({e})")
        sys.exit(1)

    print("=" * 50)
    print("ALL VALIDATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    validate()
