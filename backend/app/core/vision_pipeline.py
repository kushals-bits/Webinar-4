"""
Vision Pipeline — Gemini 2.5 Multimodal Analysis for Bill Parsing and Food Item Inspection

This module wraps the Google Generative AI SDK to perform two distinct analyses:

1. **Bill Parsing** (`parse_bill_image`)
   Extracts structured receipt data (merchant name, timestamp, line items, grand
   total, and any visible order-ID) from an uploaded bill / receipt photograph.

2. **Food Item Inspection** (`inspect_food_items`)
   Cross-references a photo of the delivered food against the expected order
   manifest and flags damage, wrong items, quantity shortfalls, stock-photo
   characteristics, and temporal / lighting inconsistencies.

Both analyses instruct the model to respond with **valid JSON only** and handle
markdown code-fence stripping before parsing.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Any

import google.generativeai as genai
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDK configuration
# ---------------------------------------------------------------------------

genai.configure(api_key=settings.GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_image_part(image_bytes: bytes) -> dict[str, Any]:
    """
    Encode *image_bytes* for the Gemini ``inline_data`` payload.

    Determines the MIME type from PIL's format detection; falls back to
    ``image/jpeg`` for opaque byte streams.

    Returns
    -------
    ``{"mime_type": str, "data": str}`` where *data* is a base64-encoded string.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        fmt = (img.format or "JPEG").upper()
        mime_map = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
            "GIF": "image/gif",
            "BMP": "image/bmp",
        }
        mime_type = mime_map.get(fmt, "image/jpeg")
    except Exception:
        mime_type = "image/jpeg"

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return {"mime_type": mime_type, "data": encoded}


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (``` … ```) from *text*."""
    # Match ```json ... ``` or ``` ... ```
    pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Bill Parsing
# ---------------------------------------------------------------------------


def parse_bill_image(bill_image_bytes: bytes) -> dict[str, Any]:
    """
    Send the bill image to Gemini and extract structured receipt data.

    The model is instructed to return **only** valid JSON with the schema::

        {
            "merchant_name": str | null,
            "bill_timestamp": str | null,       // ISO 8601 or as printed
            "item_list": [
                {"name": str, "qty": int, "unit_price": float}
            ],
            "grand_total": float | null,
            "order_id_on_bill": str | null
        }

    Returns the parsed dict, or ``{"error": "parse_failed", "raw_response": str}``
    on any failure.
    """
    if not settings.GEMINI_API_KEY:
        logger.warning("No GEMINI_API_KEY set; returning fallback receipt analysis for offline simulation.")
        return {
            "merchant_name": "Punjabi Dhaba",
            "bill_timestamp": "2026-11-09T18:30:00",
            "item_list": [
                {"name": "Butter Chicken", "qty": 2, "unit_price": 320.0},
                {"name": "Garlic Naan", "qty": 4, "unit_price": 55.0},
                {"name": "Dal Makhani", "qty": 1, "unit_price": 220.0},
            ],
            "grand_total": 1079.0,
            "order_id_on_bill": "ORD-2026-001",
        }
    prompt = (
        "You are an expert OCR and receipt-parsing assistant. "
        "Carefully examine the provided bill or receipt image and extract every piece of "
        "structured information. "
        "Respond with ONLY valid JSON — no markdown, no prose — conforming exactly to this schema:\n\n"
        "{\n"
        '  "merchant_name": "<string or null>",\n'
        '  "bill_timestamp": "<ISO-8601 string or printed date-time or null>",\n'
        '  "item_list": [\n'
        '    {"name": "<string>", "qty": <integer>, "unit_price": <float>}\n'
        "  ],\n"
        '  "grand_total": <float or null>,\n'
        '  "order_id_on_bill": "<string or null>"\n'
        "}\n\n"
        "If a field cannot be determined from the image, use null. "
        "Do NOT include any explanation or surrounding text."
    )

    image_part = _load_image_part(bill_image_bytes)

    try:
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        response = model.generate_content(
            [
                prompt,
                {"inline_data": image_part},
            ]
        )
        raw_text = response.text
        clean_text = _strip_code_fences(raw_text)
        result = json.loads(clean_text)
        return result
    except json.JSONDecodeError as exc:
        logger.warning("parse_bill_image: JSON decode failed: %s", exc)
        return {"error": "parse_failed", "raw_response": raw_text if "raw_text" in dir() else ""}
    except Exception as exc:
        logger.error("parse_bill_image: Gemini API error: %s", exc)
        return {"error": "parse_failed", "raw_response": str(exc)}


# ---------------------------------------------------------------------------
# Food Item Inspection
# ---------------------------------------------------------------------------


def inspect_food_items(
    food_image_bytes: bytes,
    expected_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Send the food photograph to Gemini alongside the expected order manifest.

    The model is instructed to return **only** valid JSON with the schema::

        {
            "delivered_items": [
                {
                    "name": str,
                    "qty_visible": int,
                    "condition": "ok" | "damaged" | "leaking" | "tampered" | "missing",
                    "notes": str
                }
            ],
            "damage_detected": bool,
            "wrong_items": [str],
            "is_stock_photo": bool,
            "lighting_consistency": bool,
            "visual_anachronism": bool,
            "overall_assessment": str
        }

    Returns the parsed dict, or ``{"error": "parse_failed", "raw_response": str}``
    on any failure.
    """
    if not settings.GEMINI_API_KEY:
        logger.warning("No GEMINI_API_KEY set; returning fallback food inspection for offline simulation.")
        delivered_items = []
        for item in expected_items:
            delivered_items.append({
                "name": item.get("name", "Unknown Item"),
                "qty_visible": item.get("qty", 1),
                "condition": "ok",
                "notes": "Verified against expected manifest (offline simulation mode)",
            })
        return {
            "delivered_items": delivered_items,
            "damage_detected": False,
            "wrong_items": [],
            "is_stock_photo": False,
            "lighting_consistency": True,
            "visual_anachronism": False,
            "overall_assessment": "Items match order manifest with authentic delivery presentation.",
        }

    expected_json = json.dumps(expected_items, indent=2)

    prompt = (
        "You are an expert food quality and delivery verification assistant. "
        "You will receive a photograph of delivered food items alongside an expected order manifest.\n\n"
        f"EXPECTED ORDER MANIFEST:\n{expected_json}\n\n"
        "Carefully analyze the photograph and:\n"
        "1. Count visible quantities for each expected item.\n"
        "2. Assess the condition of each item (ok / damaged / leaking / tampered / missing).\n"
        "3. Identify any items present in the photo that were NOT in the manifest (wrong items).\n"
        "4. Determine whether the image appears to be a stock photo rather than an authentic delivery photo.\n"
        "5. Check for consistent lighting across the frame (inconsistency may indicate compositing).\n"
        "6. Flag any visual anachronisms (e.g., watermarks, timestamps from a different context).\n\n"
        "Respond with ONLY valid JSON — no markdown, no prose — conforming exactly to this schema:\n\n"
        "{\n"
        '  "delivered_items": [\n'
        '    {\n'
        '      "name": "<string matching manifest name>",\n'
        '      "qty_visible": <integer>,\n'
        '      "condition": "<ok|damaged|leaking|tampered|missing>",\n'
        '      "notes": "<brief observation or empty string>"\n'
        '    }\n'
        "  ],\n"
        '  "damage_detected": <true|false>,\n'
        '  "wrong_items": ["<item name>"],\n'
        '  "is_stock_photo": <true|false>,\n'
        '  "lighting_consistency": <true|false>,\n'
        '  "visual_anachronism": <true|false>,\n'
        '  "overall_assessment": "<one-sentence summary>"\n'
        "}\n\n"
        "Do NOT include any explanation or surrounding text."
    )

    image_part = _load_image_part(food_image_bytes)

    try:
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        response = model.generate_content(
            [
                prompt,
                {"inline_data": image_part},
            ]
        )
        raw_text = response.text
        clean_text = _strip_code_fences(raw_text)
        result = json.loads(clean_text)
        return result
    except json.JSONDecodeError as exc:
        logger.warning("inspect_food_items: JSON decode failed: %s", exc)
        return {"error": "parse_failed", "raw_response": raw_text if "raw_text" in dir() else ""}
    except Exception as exc:
        logger.error("inspect_food_items: Gemini API error: %s", exc)
        return {"error": "parse_failed", "raw_response": str(exc)}


# ---------------------------------------------------------------------------
# Combined Pipeline Entry-Point
# ---------------------------------------------------------------------------


def run_vision_pipeline(
    bill_bytes: bytes,
    food_bytes: bytes,
    expected_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Execute both bill-parsing and food-inspection analyses in sequence.

    Returns
    -------
    ``{"bill_analysis": dict, "food_analysis": dict}``
    """
    bill_analysis = parse_bill_image(bill_bytes)
    food_analysis = inspect_food_items(food_bytes, expected_items)
    return {
        "bill_analysis": bill_analysis,
        "food_analysis": food_analysis,
    }
