import io
import os
import tempfile
from typing import Dict, Any, Optional
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas
from src.config import CLASS_COLORS

class NumberedCanvas(canvas.Canvas):
    """Canvas that performs two passes to dynamically print total page count and footer."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header line (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "Vehicle Damage Assessment Report — Inspection Audit Document")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Footer line
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 40, 558, 40)

        # Footer text
        footer_text = "CONFIDENTIAL & PROPRIETARY — Automated Computer Vision Vehicle Assessment"
        self.drawString(54, 28, footer_text)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 28, page_str)
        self.restoreState()

def generate_pdf_report(
    report_data: Dict[str, Any],
    orig_img: Image.Image,
    annot_img: Image.Image
) -> bytes:
    """
    Generates a high-quality, professional inspection report in PDF format.
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A")
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#64748B")
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )
    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569")
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1E293B")
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0F172A")
    )

    story = []

    # Title & Header
    story.append(Paragraph("VEHICLE DAMAGE ASSESSMENT", title_style))
    story.append(Paragraph(f"Inspection ID: <b>{report_data.get('report_id')}</b> &nbsp;|&nbsp; Date: <b>{report_data.get('created_at', '')[:19].replace('T', ' ')} UTC</b>", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=10))

    # Executive Summary Card
    summary_data = report_data.get('summary', {})
    exec_summary_text = report_data.get('executive_summary', '')
    
    summary_box_data = [
        [
            Paragraph("<b>EXECUTIVE SUMMARY</b>", table_cell_bold),
            Paragraph("<b>SUMMARY STATISTICS</b>", table_cell_bold)
        ],
        [
            Paragraph(exec_summary_text, body_style),
            Paragraph(
                f"• Damage Regions: <b>{summary_data.get('total_detections', 0)}</b><br/>"
                f"• Damage Categories: <b>{summary_data.get('damage_categories_count', 0)}</b><br/>"
                f"• Highest Confidence: <b>{summary_data.get('highest_confidence_pct', 'N/A')}</b><br/>"
                f"• Image Resolution: <b>{summary_data.get('image_dimensions', 'N/A')}</b>",
                table_cell
            )
        ]
    ]
    summary_table = Table(summary_box_data, colWidths=[3.5 * inch, 3.5 * inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # Visual Inspection Images (Original vs Annotated)
    story.append(Paragraph("VISUAL INSPECTION & MODEL ANNOTATION", section_heading))
    
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_path = os.path.join(tmpdir, "orig.jpg")
        annot_path = os.path.join(tmpdir, "annot.jpg")
        
        orig_img.save(orig_path, format="JPEG", quality=85)
        annot_img.save(annot_path, format="JPEG", quality=85)
        
        img_w = 3.35 * inch
        img_h = 2.2 * inch
        
        rl_orig = RLImage(orig_path, width=img_w, height=img_h)
        rl_annot = RLImage(annot_path, width=img_w, height=img_h)
        
        img_table_data = [
            [Paragraph("<b>Original Uploaded Image</b>", table_cell_bold), Paragraph("<b>AI Annotated Damage Overlay</b>", table_cell_bold)],
            [rl_orig, rl_annot]
        ]
        img_table = Table(img_table_data, colWidths=[3.5 * inch, 3.5 * inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 2),
            ('RIGHTPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 12))

        # Damage Category Breakdown Table
        story.append(Paragraph("DAMAGE CATEGORY BREAKDOWN", section_heading))
        breakdown_items = report_data.get('breakdown', [])
        
        if breakdown_items:
            bk_header = [
                Paragraph("<b>Damage Type</b>", table_cell_bold),
                Paragraph("<b>Count</b>", table_cell_bold),
                Paragraph("<b>Peak Confidence</b>", table_cell_bold),
                Paragraph("<b>Observed Regions</b>", table_cell_bold)
            ]
            bk_rows = [bk_header]
            for item in breakdown_items:
                regions_str = ", ".join(item.get('regions_present', [])) or "center"
                bk_rows.append([
                    Paragraph(item.get('display_name', item.get('class_name')), table_cell),
                    Paragraph(str(item.get('count', 1)), table_cell),
                    Paragraph(item.get('highest_confidence_pct', ''), table_cell),
                    Paragraph(regions_str, table_cell)
                ])
            bk_table = Table(bk_rows, colWidths=[2.2 * inch, 1.0 * inch, 1.4 * inch, 2.4 * inch])
            bk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(bk_table)
        else:
            story.append(Paragraph("<i>No damage regions identified above threshold.</i>", callout_style))
        story.append(Spacer(1, 12))

        # Detailed Detections Inventory Table
        story.append(Paragraph("DETAILED DETECTION INVENTORY", section_heading))
        detections = report_data.get('detections', [])
        
        if detections:
            det_header = [
                Paragraph("<b>#</b>", table_cell_bold),
                Paragraph("<b>Damage Class</b>", table_cell_bold),
                Paragraph("<b>Confidence</b>", table_cell_bold),
                Paragraph("<b>Image Region</b>", table_cell_bold),
                Paragraph("<b>Bounding Box (x1, y1, x2, y2)</b>", table_cell_bold),
                Paragraph("<b>Severity Status</b>", table_cell_bold)
            ]
            det_rows = [det_header]
            for det in detections:
                bbox = det.get('bbox', {})
                bbox_str = f"[{bbox.get('x1', 0):.0f}, {bbox.get('y1', 0):.0f}, {bbox.get('x2', 0):.0f}, {bbox.get('y2', 0):.0f}]"
                det_rows.append([
                    Paragraph(str(det.get('id', '')), table_cell),
                    Paragraph(det.get('display_name', det.get('class_name')), table_cell),
                    Paragraph(det.get('confidence_pct', ''), table_cell),
                    Paragraph(det.get('location', ''), table_cell),
                    Paragraph(bbox_str, table_cell),
                    Paragraph(det.get('severity', 'Requires inspection'), table_cell)
                ])
            det_table = Table(det_rows, colWidths=[0.35 * inch, 1.6 * inch, 0.9 * inch, 1.2 * inch, 1.55 * inch, 1.4 * inch])
            det_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(det_table)
        story.append(Spacer(1, 12))

        # Limitations & Next Steps (Responsible AI & Inspection Guidance)
        advisory_elements = []
        advisory_elements.append(Paragraph("LIMITATIONS & RECOMMENDED NEXT STEPS", section_heading))
        
        limitations = report_data.get('limitations', [])
        rec_text = "<b>Methodological Limitations:</b><br/>" + "<br/>".join([f"• {lim}" for lim in limitations])
        
        recommendations = report_data.get('recommendations', [])
        rec_text += "<br/><br/><b>Recommended Actions:</b><br/>" + "<br/>".join([f"• {rec}" for rec in recommendations])
        
        advisory_table = Table([[Paragraph(rec_text, callout_style)]], colWidths=[7.0 * inch])
        advisory_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FEF3C7")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#F59E0B")),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        advisory_elements.append(advisory_table)
        advisory_elements.append(Spacer(1, 8))

        # Model Provenance & Metadata
        model_info = report_data.get('model_info', {})
        if model_info:
            meta_str = (
                f"Model Architecture: <b>{model_info.get('architecture', 'RTDetrForObjectDetection')}</b> &nbsp;|&nbsp; "
                f"Base: <b>{model_info.get('base_checkpoint', 'rtdetr_r50vd')}</b> &nbsp;|&nbsp; "
                f"Resolution: <b>{model_info.get('input_resolution', '640x640')}</b> &nbsp;|&nbsp; "
                f"Device: <b>{model_info.get('device', 'CPU')}</b>"
            )
            advisory_elements.append(Paragraph(meta_str, subtitle_style))

        story.append(KeepTogether(advisory_elements))

        # Build PDF with dynamic 2-pass NumberedCanvas
        doc.build(story, canvasmaker=NumberedCanvas)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
