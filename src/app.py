import io
import gc
import time
import logging
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.config import STATIC_DIR, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_IOU_THRESHOLD, APP_CONFIG, OUTPUTS_DIR
from src.inference import DamageDetector
from src.analysis import DamageAnalysisEngine
from src.annotator import draw_detections_on_image, image_to_base64_uri
from src.pdf_generator import generate_pdf_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("vehicle_damage_detector.app")

app = FastAPI(
    title="Vehicle Damage Detection & Assessment API",
    description="Production-grade AI Vehicle Damage Detection and Audit Reporting powered by RT-DETR.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing DamageDetector singleton on server startup...")
    DamageDetector.get_instance()
    logger.info("DamageDetector ready for inference.")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/api/health")
async def health_check():
    detector = DamageDetector.get_instance()
    return {
        "status": "ok",
        "device": str(detector.device),
        "model_loaded": detector.model is not None,
        "classes": detector.id2label
    }

@app.get("/api/model-info")
async def get_model_info():
    detector = DamageDetector.get_instance()
    return {
        "success": True,
        "info": detector.get_model_info()
    }

@app.post("/api/predict")
async def predict_damage(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Form(None),
    group_duplicates: Optional[bool] = Form(False),
    iou_threshold: Optional[float] = Form(DEFAULT_IOU_THRESHOLD)
):
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

        try:
            image = Image.open(io.BytesIO(contents))
            image.load()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")

        detector = DamageDetector.get_instance()
        threshold = confidence_threshold if confidence_threshold is not None else DEFAULT_CONFIDENCE_THRESHOLD
        
        # 1. Run inference
        start_time = time.time()
        raw_detections = detector.predict(image, confidence_threshold=threshold)
        inference_latency_ms = round((time.time() - start_time) * 1000, 2)

        # 2. Deterministic damage analysis & spatial mapping
        model_info = detector.get_model_info()
        report = DamageAnalysisEngine.analyze(
            detections=raw_detections,
            img_width=image.width,
            img_height=image.height,
            group_duplicates=bool(group_duplicates),
            iou_threshold=float(iou_threshold or DEFAULT_IOU_THRESHOLD),
            model_info=model_info
        )
        report["inference_latency_ms"] = inference_latency_ms

        # 3. Generate annotated image
        annotated_image = draw_detections_on_image(image, report["detections"])

        # 4. Generate base64 representations
        orig_b64 = image_to_base64_uri(image, format="JPEG", quality=85)
        annot_b64 = image_to_base64_uri(annotated_image, format="JPEG", quality=85)

        del image
        del annotated_image
        gc.collect()

        return {
            "success": True,
            "report": report,
            "original_image": orig_b64,
            "annotated_image": annot_b64
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during damage prediction")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/api/report/pdf")
async def export_pdf_report(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Form(None),
    group_duplicates: Optional[bool] = Form(False),
    iou_threshold: Optional[float] = Form(DEFAULT_IOU_THRESHOLD)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")

        detector = DamageDetector.get_instance()
        threshold = confidence_threshold if confidence_threshold is not None else DEFAULT_CONFIDENCE_THRESHOLD
        
        raw_detections = detector.predict(image, confidence_threshold=threshold)
        model_info = detector.get_model_info()
        
        report = DamageAnalysisEngine.analyze(
            detections=raw_detections,
            img_width=image.width,
            img_height=image.height,
            group_duplicates=bool(group_duplicates),
            iou_threshold=float(iou_threshold or DEFAULT_IOU_THRESHOLD),
            model_info=model_info
        )

        annotated_image = draw_detections_on_image(image, report["detections"])
        pdf_bytes = generate_pdf_report(report, image, annotated_image)

        filename = f"Vehicle_Damage_Report_{report['report_id']}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.exception("Error generating PDF report")
        raise HTTPException(status_code=500, detail=f"PDF generation error: {str(e)}")

# Mount static files and evaluation assets
if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
