# Vehicle Damage Detection & Assessment System

An end-to-end Computer Vision system for automated vehicle damage detection, spatial analysis, and audit report generation powered by a fine-tuned Real-Time DEtection TRansformer (**RT-DETR ResNet-50-vd**).

---

## Features

- **Fine-Tuned RT-DETR Detection**: Real-time object detection across 4 damage classes:
  - `crack_or_hole`
  - `deformation` (dents, bent panels)
  - `scratch`
  - `windshield_damage`
- **Deterministic Damage Analysis Engine**:
  - $3 \times 3$ photographic spatial grid location mapping (`front-left`, `center`, `rear-right`, etc.).
  - Overlap deduplication & grouping via Intersection-over-Union (IoU).
  - Damage breakdown statistics and peak confidence tracking.
  - Responsible severity handling (*"Requires physical inspection"*) avoiding fabricated claims.
- **Interactive Web Dashboard**:
  - Drag-and-drop image upload with live preview.
  - Confidence threshold slider and duplicate grouping toggles.
  - Comparison viewer (*Annotated Image*, *Original Image*, *Side-by-Side*).
  - Structured formal inspection audit report view.
  - Model architecture & training analytics viewer (training curves, confusion matrix, class distribution).
- **Automated Audit PDF Generator**:
  - Downloadable multi-page vector PDF inspection report with inspection metadata, side-by-side evidence, tables, and inspection guidance.
- **RESTful API**:
  - FastAPI backend with `/api/predict`, `/api/report/pdf`, `/api/model-info`, `/api/health`.

---

## Project Structure

```
vehichle_damage_detection/
├── models/
│   ├── best_model/              # RT-DETR weights, processor & configs
│   ├── last_model/
│   ├── class_names.json         # id2label class mapping
│   └── config.json              # Training configuration metadata
├── outputs/                     # Training curves, confusion matrix, predictions
├── src/
│   ├── config.py                # Configuration and color schemes
│   ├── inference.py             # Singleton RT-DETR inference engine
│   ├── analysis.py              # Deterministic spatial analysis engine
│   ├── annotator.py             # Bounding box visualizer
│   ├── pdf_generator.py         # Multi-page ReportLab PDF generator
│   └── app.py                   # FastAPI REST API & static server
├── static/
│   ├── index.html               # Responsive web dashboard
│   ├── styles.css               # Styling and theme
│   └── app.js                   # Client interactivity & API calls
├── validate_model.py            # CLI validation script
├── test_suite.py                # Automated unit & integration tests
├── run.py                       # Application launcher
└── requirements.txt             # Project dependencies
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HarshitaSmriti/vehicle-damage-detection.git
   cd vehicle-damage-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python run.py
   ```
   Open **http://localhost:8000** in your browser.

---

## Validation & Testing

Run the validation script:
```bash
python validate_model.py
```

Run the complete test suite:
```bash
python test_suite.py
```

---

## License
MIT
