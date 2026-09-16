import io
import json
import streamlit as st
from PIL import Image

from src.config import CLASS_NAMES, CLASS_COLORS, CLASS_DISPLAY_NAMES, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_IOU_THRESHOLD, APP_CONFIG
from src.inference import DamageDetector
from src.analysis import DamageAnalysisEngine
from src.annotator import draw_detections_on_image
from src.pdf_generator import generate_pdf_report

# Page Config
st.set_page_config(
    page_title="AI Vehicle Damage Assessment",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    return DamageDetector.get_instance()

def main():
    st.markdown('<h1 class="main-title">🚗 Vehicle Damage Assessment</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Real-Time Computer Vision Detection & Structured Audit Reporting (RT-DETR ResNet-50)</p>', unsafe_allow_html=True)

    with st.spinner("Initializing AI Detection Model..."):
        detector = load_detector()

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Detection Settings")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(DEFAULT_CONFIDENCE_THRESHOLD),
            step=0.05,
            help="Minimum model confidence required to register a damage region."
        )
        group_duplicates = st.checkbox(
            "Group Overlapping Detections",
            value=False,
            help="Merge overlapping nearby boxes with IoU >= 0.60"
        )
        
        st.markdown("---")
        st.subheader("🎯 Model Classes")
        for cls_name, display in CLASS_DISPLAY_NAMES.items():
            color = CLASS_COLORS.get(cls_name, {}).get("hex", "#3B82F6")
            st.markdown(f"<span style='color:{color}; font-weight:600;'>●</span> {display}", unsafe_allow_html=True)

    # Main Tabs
    tab_detect, tab_report, tab_analytics = st.tabs(["🔍 Damage Detection", "📋 Audit Report", "📊 Model Analytics"])

    with tab_detect:
        uploaded_file = st.file_uploader(
            "Upload Vehicle Image (JPEG, PNG, WebP)",
            type=["jpg", "jpeg", "png", "webp", "bmp"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            if max(image.size) > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            if image.mode != "RGB":
                image = image.convert("RGB")

            col_btn, col_space = st.columns([1, 2])
            with col_btn:
                run_btn = st.button("🚀 Detect Vehicle Damage", type="primary")

            if run_btn or "last_report" in st.session_state:
                if run_btn:
                    with st.spinner("Analyzing image with RT-DETR neural network..."):
                        raw_detections = detector.predict(image, confidence_threshold=conf_threshold)
                        model_info = detector.get_model_info()
                        report = DamageAnalysisEngine.analyze(
                            detections=raw_detections,
                            img_width=image.width,
                            img_height=image.height,
                            group_duplicates=group_duplicates,
                            model_info=model_info
                        )
                        annotated_img = draw_detections_on_image(image, report["detections"])
                        pdf_bytes = generate_pdf_report(report, image, annotated_img)

                        st.session_state["last_image"] = image
                        st.session_state["last_annotated"] = annotated_img
                        st.session_state["last_report"] = report
                        st.session_state["last_pdf"] = pdf_bytes

                report = st.session_state.get("last_report")
                orig_img = st.session_state.get("last_image", image)
                annot_img = st.session_state.get("last_annotated", image)
                pdf_bytes = st.session_state.get("last_pdf", b"")

                if report:
                    # Top Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Damage Regions", report["summary"]["total_detections"])
                    with m2:
                        st.metric("Categories Found", report["summary"]["damage_categories_count"])
                    with m3:
                        st.metric("Peak Confidence", report["summary"]["highest_confidence_pct"])
                    with m4:
                        st.metric("Resolution", report["summary"]["image_dimensions"])

                    # Image Viewers
                    st.subheader("Visual Inspection")
                    view_mode = st.radio("View Mode", ["Side-by-Side", "Annotated Only", "Original Only"], horizontal=True)

                    if view_mode == "Side-by-Side":
                        c1, c2 = st.columns(2)
                        with c1:
                            st.caption("Original Image")
                            st.image(orig_img, use_container_width=True)
                        with c2:
                            st.caption("Annotated AI Detections")
                            st.image(annot_img, use_container_width=True)
                    elif view_mode == "Annotated Only":
                        st.image(annot_img, use_container_width=True)
                    else:
                        st.image(orig_img, use_container_width=True)

                    # Breakdown
                    st.subheader("Damage Category Breakdown")
                    if report["breakdown"]:
                        bk_data = []
                        for b in report["breakdown"]:
                            bk_data.append({
                                "Damage Type": b["display_name"],
                                "Count": b["count"],
                                "Peak Confidence": b["highest_confidence_pct"],
                                "Observed Regions": ", ".join(b.get("regions_present", [])) or "center"
                            })
                        st.table(bk_data)
                    else:
                        st.info("No damage regions detected above the active confidence threshold.")

                    # Individual Detections
                    st.subheader("Individual Detections Inventory")
                    if report["detections"]:
                        det_data = []
                        for d in report["detections"]:
                            bbox = d["bbox"]
                            det_data.append({
                                "#": d["id"],
                                "Class": d["display_name"],
                                "Confidence": d["confidence_pct"],
                                "Image Region": d["location"],
                                "Bounding Box": f"[{bbox['x1']}, {bbox['y1']}, {bbox['x2']}, {bbox['y2']}]",
                                "Severity Status": d["severity"]
                            })
                        st.dataframe(det_data, use_container_width=True)

                    # Actions
                    st.subheader("Actions & Export")
                    act1, act2 = st.columns(2)
                    with act1:
                        st.download_button(
                            label="📄 Download Official PDF Audit Report",
                            data=pdf_bytes,
                            file_name=f"Vehicle_Damage_Report_{report['report_id']}.pdf",
                            mime="application/pdf",
                            type="primary"
                        )
                    with act2:
                        st.download_button(
                            label="💾 Export Structured JSON Data",
                            data=json.dumps(report, indent=2),
                            file_name=f"Damage_Report_{report['report_id']}.json",
                            mime="application/json"
                        )
        else:
            st.info("👆 Please upload a vehicle image above to begin damage inspection.")

    with tab_report:
        report = st.session_state.get("last_report")
        if report:
            st.markdown(f"### Inspection ID: `{report['report_id']}`")
            st.caption(f"Audit Date: {report['created_at'][:19].replace('T', ' ')} UTC")
            
            st.markdown("#### 1. Executive Summary")
            st.info(report["executive_summary"])

            st.markdown("#### 2. Methodological Limitations & Safety Disclaimers")
            for lim in report.get("limitations", []):
                st.warning(f"• {lim}")

            st.markdown("#### 3. Recommended Next Steps")
            for rec in report.get("recommendations", []):
                st.success(f"• {rec}")
        else:
            st.info("Run damage detection first to view the full audit report.")

    with tab_analytics:
        st.subheader("Model Architecture & Training Artifacts")
        m_info = detector.get_model_info()
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Architecture", m_info["architecture"])
        with c2:
            st.metric("Base Checkpoint", m_info["base_checkpoint"].split("/")[-1])
        with c3:
            st.metric("Trained Epochs", m_info.get("trained_epochs", 21))
        with c4:
            st.metric("Test mAP@50", f"{(m_info.get('test_mAP_50', 0.5898)*100):.1f}%")

        st.markdown("---")
        a1, a2 = st.columns(2)
        with a1:
            st.caption("Training & Validation Curves")
            st.image("outputs/training_curves.png", use_container_width=True)
        with a2:
            st.caption("Confusion Matrix")
            st.image("outputs/confusion_matrix.png", use_container_width=True)
        
        st.caption("Dataset Class Distribution")
        st.image("outputs/class_distribution.png", use_container_width=True)

if __name__ == "__main__":
    main()
