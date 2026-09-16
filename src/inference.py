import os
import gc
import time
import logging
import urllib.request
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from src.config import BEST_MODEL_DIR, DEFAULT_CONFIDENCE_THRESHOLD, CLASS_NAMES, APP_CONFIG

# Optimize PyTorch CPU memory and threads for cloud/container deployment
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

logger = logging.getLogger("vehicle_damage_detector.inference")

MODEL_WEIGHTS_URL = "https://github.com/HarshitaSmriti/vehicle-damage-detection/releases/download/v1.0.0/model.safetensors"

def ensure_model_weights(model_dir: Union[str, Path]):
    model_dir_path = Path(model_dir)
    target_safetensors = model_dir_path / "model.safetensors"
    target_bin = model_dir_path / "pytorch_model.bin"

    if not target_safetensors.exists() and not target_bin.exists():
        logger.info(f"Model weights not found in {model_dir_path}. Downloading from GitHub Releases...")
        model_dir_path.mkdir(parents=True, exist_ok=True)
        
        temp_file = model_dir_path / "model.safetensors.download"
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [("User-Agent", "Mozilla/5.0")]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(MODEL_WEIGHTS_URL, temp_file)
            temp_file.rename(target_safetensors)
            logger.info(f"Model weights downloaded successfully ({target_safetensors.stat().st_size / (1024*1024):.1f} MB)")
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            logger.exception("Failed to download model weights automatically")
            raise RuntimeError(f"Could not download model weights: {e}")

class DamageDetector:
    _instance: Optional['DamageDetector'] = None

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = str(model_path or BEST_MODEL_DIR)
        
        ensure_model_weights(self.model_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing DamageDetector on device: {self.device}")
        
        start_time = time.time()
        self.processor = RTDetrImageProcessor.from_pretrained(self.model_path)
        
        # Load in evaluation mode with low CPU memory overhead
        self.model = RTDetrForObjectDetection.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        self.model.eval()
        self.load_time_seconds = time.time() - start_time
        
        self.id2label = getattr(self.model.config, "id2label", CLASS_NAMES)
        self.id2label = {int(k): str(v) for k, v in self.id2label.items()}
        logger.info(f"Model loaded successfully in {self.load_time_seconds:.2f}s with classes: {self.id2label}")
        
        gc.collect()

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None) -> 'DamageDetector':
        if cls._instance is None:
            cls._instance = cls(model_path=model_path)
        return cls._instance

    def predict(
        self,
        image: Image.Image,
        confidence_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        threshold = confidence_threshold if confidence_threshold is not None else DEFAULT_CONFIDENCE_THRESHOLD
        
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        orig_width, orig_height = image.size
        
        # Process tensor inputs
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
            
        target_sizes = torch.tensor([[orig_height, orig_width]], device=self.device)
        
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )[0]
        
        detections: List[Dict[str, Any]] = []
        
        scores = results["scores"].detach().cpu().tolist()
        labels = results["labels"].detach().cpu().tolist()
        boxes = results["boxes"].detach().cpu().tolist()
        
        for score, label_idx, box in zip(scores, labels, boxes):
            xmin, ymin, xmax, ymax = box
            xmin = max(0.0, min(float(orig_width), float(xmin)))
            ymin = max(0.0, min(float(orig_height), float(ymin)))
            xmax = max(0.0, min(float(orig_width), float(xmax)))
            ymax = max(0.0, min(float(orig_height), float(ymax)))
            
            if xmax <= xmin or ymax <= ymin:
                continue
                
            class_name = self.id2label.get(label_idx, f"class_{label_idx}")
            
            detections.append({
                "class_id": int(label_idx),
                "class_name": class_name,
                "confidence": round(float(score), 4),
                "bbox": {
                    "x1": round(xmin, 1),
                    "y1": round(ymin, 1),
                    "x2": round(xmax, 1),
                    "y2": round(ymax, 1),
                    "width": round(xmax - xmin, 1),
                    "height": round(ymax - ymin, 1)
                }
            })
            
        # Free tensor outputs memory
        del outputs
        del inputs
        gc.collect()
        
        return detections

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "architecture": self.model.__class__.__name__,
            "base_checkpoint": APP_CONFIG.get("checkpoint_base", "PekingU/rtdetr_r50vd_coco_o365"),
            "backbone": "ResNet-50-vd (rt_detr_resnet)",
            "num_classes": len(self.id2label),
            "classes": self.id2label,
            "input_resolution": f"{APP_CONFIG.get('image_size', 640)}x{APP_CONFIG.get('image_size', 640)}",
            "framework": f"PyTorch {torch.__version__} / HuggingFace Transformers",
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "default_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "trained_epochs": APP_CONFIG.get("trained_epochs", 21),
            "test_mAP_50": APP_CONFIG.get("test_mAP_50", 0.5898)
        }
