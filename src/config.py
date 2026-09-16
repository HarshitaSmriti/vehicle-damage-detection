import os
import json
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
BEST_MODEL_DIR = MODELS_DIR / 'best_model'
CONFIG_JSON_PATH = MODELS_DIR / 'config.json'
CLASS_NAMES_JSON_PATH = MODELS_DIR / 'class_names.json'
OUTPUTS_DIR = BASE_DIR / 'outputs'
STATIC_DIR = BASE_DIR / 'static'

# Default fallbacks if files not found
DEFAULT_CLASSES = {
    0: 'crack_or_hole',
    1: 'deformation',
    2: 'scratch',
    3: 'windshield_damage'
}

CLASS_DISPLAY_NAMES = {
    'crack_or_hole': 'Crack or Hole',
    'deformation': 'Deformation / Dent',
    'scratch': 'Scratch',
    'windshield_damage': 'Windshield Damage'
}

# Distinct, professional color scheme (Hex & RGB) for visualization
CLASS_COLORS = {
    'crack_or_hole': {'hex': '#EF4444', 'rgb': (239, 68, 68)},        # Red
    'deformation': {'hex': '#F59E0B', 'rgb': (245, 158, 11)},          # Amber
    'scratch': {'hex': '#3B82F6', 'rgb': (59, 130, 246)},              # Blue
    'windshield_damage': {'hex': '#8B5CF6', 'rgb': (139, 92, 246)},    # Purple
    'default': {'hex': '#10B981', 'rgb': (16, 185, 129)}               # Emerald
}

def load_config() -> Dict[str, Any]:
    if CONFIG_JSON_PATH.exists():
        with open(CONFIG_JSON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif (BEST_MODEL_DIR / 'config_summary.json').exists():
        with open(BEST_MODEL_DIR / 'config_summary.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'checkpoint_base': 'PekingU/rtdetr_r50vd_coco_o365',
        'num_classes': 4,
        'image_size': 640,
        'confidence_threshold': 0.1
    }

def load_class_names() -> Dict[int, str]:
    target_path = CLASS_NAMES_JSON_PATH
    if not target_path.exists() and (MODELS_DIR / 'class_names (2).json').exists():
        target_path = MODELS_DIR / 'class_names (2).json'
    elif not target_path.exists() and (BEST_MODEL_DIR / 'class_names.json').exists():
        target_path = BEST_MODEL_DIR / 'class_names.json'

    if target_path.exists():
        with open(target_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'id2label' in data:
                return {int(k): v for k, v in data['id2label'].items()}
            return {int(k): v for k, v in data.items()}
    return DEFAULT_CLASSES

APP_CONFIG = load_config()
CLASS_NAMES = load_class_names()
DEFAULT_CONFIDENCE_THRESHOLD = float(APP_CONFIG.get('confidence_threshold', 0.1))
DEFAULT_IOU_THRESHOLD = 0.6
