import io
import base64
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
from src.config import CLASS_COLORS, CLASS_DISPLAY_NAMES

def get_font(size: int = 14) -> ImageFont.ImageFont:
    """Attempt to load a standard truetype font or fallback to default."""
    try:
        # Standard fonts on Windows
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except IOError:
            return ImageFont.load_default()

def draw_detections_on_image(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    line_width: Optional[int] = None
) -> Image.Image:
    """
    Renders clean, professional bounding box annotations on the image.
    Preserves exact aspect ratio and original image dimensions.
    """
    if image.mode != "RGB":
        annotated = image.convert("RGB").copy()
    else:
        annotated = image.copy()

    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    # Dynamically scale line width and font size based on image resolution
    diag = (width**2 + height**2)**0.5
    calc_line_width = max(2, int(round(diag / 450))) if line_width is None else line_width
    font_size = max(12, int(round(diag / 60)))
    font = get_font(font_size)

    for det in detections:
        bbox = det["bbox"]
        class_name = det["class_name"]
        confidence = det["confidence"]
        display_name = CLASS_DISPLAY_NAMES.get(class_name, class_name.replace("_", " ").title())

        # Coordinates
        x1 = max(0, min(width - 1, bbox["x1"]))
        y1 = max(0, min(height - 1, bbox["y1"]))
        x2 = max(0, min(width - 1, bbox["x2"]))
        y2 = max(0, min(height - 1, bbox["y2"]))

        color_info = CLASS_COLORS.get(class_name, CLASS_COLORS["default"])
        rgb_color = color_info["rgb"]

        # 1. Draw outer bounding box rectangle
        for i in range(calc_line_width):
            draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=rgb_color)

        # 2. Prepare badge label text
        label_text = f"{display_name} {int(round(confidence * 100))}%"

        # Calculate text bounding box
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        padding_x = 6
        padding_y = 3
        badge_w = text_w + 2 * padding_x
        badge_h = text_h + 2 * padding_y

        # Badge position: above box if space permits, else inside box
        if y1 - badge_h >= 0:
            badge_y1 = y1 - badge_h
            badge_y2 = y1
        else:
            badge_y1 = y1
            badge_y2 = y1 + badge_h

        badge_x1 = max(0, min(width - badge_w, x1))
        badge_x2 = badge_x1 + badge_w

        # Draw solid badge background
        draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=rgb_color)

        # Draw readable white text
        draw.text(
            (badge_x1 + padding_x, badge_y1 + padding_y),
            label_text,
            fill=(255, 255, 255),
            font=font
        )

    return annotated

def image_to_base64_uri(image: Image.Image, format: str = "JPEG", quality: int = 90) -> str:
    """Converts a PIL Image to a base64 data URI string."""
    buffered = io.BytesIO()
    if format.upper() in ("JPG", "JPEG"):
        image.save(buffered, format="JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        image.save(buffered, format="PNG")
        mime = "image/png"
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{img_b64}"
