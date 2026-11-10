"""
Fraud Detection Engine — EXIF Metadata Analysis, Timestamp Verification,
Device Fingerprinting, and Multi-tier Fraud Score Computation

Pipeline
--------
1. Extract EXIF metadata from the uploaded image.
2. Validate the image capture timestamp against the order delivery time.
3. Inspect software/make fields for AI-synthesis or emulator markers.
4. Build a device fingerprint and check for multi-account abuse.
5. Run perceptual-hash duplicate check and reverse-image-search mock.
6. Run screen-capture artifact detection.
7. Aggregate weighted fraud score and enforce a tiered decision.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from datetime import datetime, timezone
from typing import Any

from dateutil import parser as dateutil_parser
from PIL import ExifTags, Image

from app.config import settings
from app.database import mock_db
from app.security import hashing_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EXIF Extraction
# ---------------------------------------------------------------------------

# Known editing / synthesis software markers (lowercase)
_SYNTHESIS_SOFTWARE_MARKERS = [
    "photoshop",
    "gimp",
    "lightroom",
    "snapseed",
    "faceapp",
    "canva",
    "picsart",
    "adobe",
]

# Virtual device / emulator markers in the Make field (lowercase)
_EMULATOR_MAKE_MARKERS = [
    "unknown",
    "generic",
    "android sdk",
]


def extract_exif_metadata(image_bytes: bytes) -> dict[str, Any]:
    """
    Open *image_bytes* and attempt to extract relevant EXIF fields.

    Returns
    -------
    A dict with keys:
      ``datetime_original``, ``make``, ``model``, ``software``,
      ``has_gps``, ``raw_exif_present``.
    All values are None (and ``raw_exif_present`` is False) when EXIF is absent.
    """
    empty_result: dict[str, Any] = {
        "datetime_original": None,
        "make": None,
        "model": None,
        "software": None,
        "has_gps": False,
        "raw_exif_present": False,
    }

    try:
        img = Image.open(io.BytesIO(image_bytes))
        raw_exif = None
        if hasattr(img, "getexif"):
            exif_obj = img.getexif()
            if exif_obj:
                raw_exif = dict(exif_obj.items())
        if not raw_exif and hasattr(img, "_getexif"):
            raw_exif = img._getexif()
    except Exception as exc:
        logger.warning("extract_exif_metadata: could not open image: %s", exc)
        return empty_result

    if not raw_exif:
        return empty_result

    # Build human-readable tag map
    tag_map: dict[str, Any] = {
        ExifTags.TAGS.get(tag_id, str(tag_id)): value
        for tag_id, value in raw_exif.items()
    }

    return {
        "datetime_original": tag_map.get("DateTimeOriginal"),
        "make": tag_map.get("Make"),
        "model": tag_map.get("Model"),
        "software": tag_map.get("Software"),
        "has_gps": "GPSInfo" in tag_map,
        "raw_exif_present": True,
    }


# ---------------------------------------------------------------------------
# Timestamp Validation
# ---------------------------------------------------------------------------


def validate_timestamp(
    image_datetime_str: str | None,
    order_delivery_time_str: str,
    max_delta_seconds: int = 7200,
) -> dict[str, Any]:
    """
    Compare the image capture timestamp against the order delivery time.

    Rules
    -----
    - Missing timestamp              → MISSING_EXIF_TIMESTAMP
    - Image taken > *max_delta_seconds* after delivery → POST_DELIVERY_WINDOW_EXCEEDED
    - Image taken > 60 s before delivery → PREDATES_DELIVERY
    - Otherwise                     → valid

    Returns
    -------
    ``{"valid": bool, "flag": str | None, "delta_seconds": float | None}``
    """
    if not image_datetime_str:
        return {"valid": False, "flag": "MISSING_EXIF_TIMESTAMP", "delta_seconds": None}

    try:
        image_dt = dateutil_parser.parse(image_datetime_str)
        delivery_dt = dateutil_parser.parse(order_delivery_time_str)
    except Exception as exc:
        logger.warning("validate_timestamp: parse error: %s", exc)
        return {"valid": False, "flag": "MISSING_EXIF_TIMESTAMP", "delta_seconds": None}

    # Normalise to UTC for comparison
    if image_dt.tzinfo is None:
        image_dt = image_dt.replace(tzinfo=timezone.utc)
    if delivery_dt.tzinfo is None:
        delivery_dt = delivery_dt.replace(tzinfo=timezone.utc)

    delta_seconds = (image_dt - delivery_dt).total_seconds()

    if delta_seconds > max_delta_seconds:
        return {
            "valid": False,
            "flag": "POST_DELIVERY_WINDOW_EXCEEDED",
            "delta_seconds": delta_seconds,
        }

    if delta_seconds < -60:
        return {
            "valid": False,
            "flag": "PREDATES_DELIVERY",
            "delta_seconds": delta_seconds,
        }

    return {"valid": True, "flag": None, "delta_seconds": delta_seconds}


# ---------------------------------------------------------------------------
# Synthesis / Emulator Marker Inspection
# ---------------------------------------------------------------------------


def inspect_software_synthesis_markers(exif_meta: dict[str, Any]) -> dict[str, Any]:
    """
    Scan EXIF ``software`` and ``make`` fields for editing-tool or emulator
    markers that indicate a synthesised / manipulated image.

    Returns
    -------
    ``{"synthesis_detected": bool, "markers_found": list[str]}``
    """
    markers_found: list[str] = []

    software = (exif_meta.get("software") or "").lower()
    make = (exif_meta.get("make") or "").lower()

    for marker in _SYNTHESIS_SOFTWARE_MARKERS:
        if marker in software:
            markers_found.append(f"software:{marker}")

    for marker in _EMULATOR_MAKE_MARKERS:
        if marker in make:
            markers_found.append(f"make:{marker}")

    return {
        "synthesis_detected": len(markers_found) > 0,
        "markers_found": markers_found,
    }


# ---------------------------------------------------------------------------
# Device Fingerprint
# ---------------------------------------------------------------------------


def build_device_fingerprint(exif_meta: dict[str, Any]) -> str:
    """
    Derive a stable device fingerprint from the EXIF ``make``, ``model``,
    and ``software`` fields.

    Returns the first 16 hex characters of a SHA-256 digest.
    """
    make = (exif_meta.get("make") or "unknown").strip().lower()
    model = (exif_meta.get("model") or "unknown").strip().lower()
    software = (exif_meta.get("software") or "unknown").strip().lower()
    combined = f"{make}|{model}|{software}"
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return digest[:16]


# ---------------------------------------------------------------------------
# Full Fraud-Detection Pipeline
# ---------------------------------------------------------------------------

# Weighted contribution of each fraud signal to the overall risk score
_FRAUD_WEIGHTS: dict[str, float] = {
    "MISSING_EXIF_TIMESTAMP": 0.15,
    "PREDATES_DELIVERY": 0.30,
    "POST_DELIVERY_WINDOW_EXCEEDED": 0.20,
    "synthesis_detected": 0.25,
    "multi_account_device": 0.35,
    "is_duplicate_hash": 0.40,
    "is_indexed_online": 0.30,
    "is_screen_capture": 0.25,
}


def run_fraud_detection(
    image_bytes: bytes,
    order_id: str,
    customer_id: str,
    image_type: str = "food",
) -> dict[str, Any]:
    """
    Execute the complete fraud-detection pipeline for a single uploaded image.

    Returns a comprehensive result dict containing all extracted signals and
    the aggregated ``fraud_risk_score`` (0.0 – 1.0).
    """
    order = mock_db.get_order(order_id)
    delivery_time_str = order.get("delivery_time", datetime.now(tz=timezone.utc).isoformat()) if order else datetime.now(tz=timezone.utc).isoformat()

    # 1. EXIF extraction
    exif_meta = extract_exif_metadata(image_bytes)

    # 2. Timestamp validation
    ts_result = validate_timestamp(
        image_datetime_str=exif_meta.get("datetime_original"),
        order_delivery_time_str=delivery_time_str,
        max_delta_seconds=settings.MAX_TIMESTAMP_DELTA_SECONDS,
    )

    # 3. Synthesis / emulator markers
    synthesis_result = inspect_software_synthesis_markers(exif_meta)

    # 4. Device fingerprint & multi-account check
    fingerprint = build_device_fingerprint(exif_meta)
    other_acc = mock_db.check_device_multi_account(fingerprint, customer_id)
    if isinstance(other_acc, str):
        other_accounts = [other_acc]
        multi_account_detected = True
    elif isinstance(other_acc, list):
        other_accounts = other_acc
        multi_account_detected = len(other_acc) > 0
    else:
        other_accounts = []
        multi_account_detected = False

    # 5. pHash computation + DB cross-check
    try:
        phash_str = hashing_service.compute_phash(image_bytes)
    except Exception as exc:
        logger.warning("run_fraud_detection: phash failed: %s", exc)
        phash_str = ""

    hash_check = hashing_service.check_hash_against_db(
        phash_str,
        config_limit=settings.PHASH_HAMMING_DISTANCE_LIMIT,
    )

    # 6. Reverse-image-search mock
    reverse_image = hashing_service.generate_reverse_image_webhook_mock(phash_str)

    # 7. Screen-capture artifact detection
    screen_capture = hashing_service.detect_screen_capture_artifacts(image_bytes)

    # ------------------------------------------------------------------ #
    # Assemble fraud flags & compute weighted risk score
    # ------------------------------------------------------------------ #
    fraud_flags: list[str] = []
    fraud_risk_score: float = 0.0

    # Timestamp flags
    if not ts_result["valid"] and ts_result["flag"]:
        fraud_flags.append(ts_result["flag"])
        fraud_risk_score += _FRAUD_WEIGHTS.get(ts_result["flag"], 0.0)

    # Synthesis markers
    if synthesis_result["synthesis_detected"]:
        fraud_flags.append("SYNTHESIS_MARKERS_DETECTED")
        fraud_risk_score += _FRAUD_WEIGHTS["synthesis_detected"]

    # Multi-account device
    if multi_account_detected:
        fraud_flags.append("MULTI_ACCOUNT_DEVICE")
        fraud_risk_score += _FRAUD_WEIGHTS["multi_account_device"]

    # Duplicate perceptual hash
    if hash_check["is_duplicate"]:
        fraud_flags.append("DUPLICATE_IMAGE_HASH")
        fraud_risk_score += _FRAUD_WEIGHTS["is_duplicate_hash"]

    # Online indexed image
    if reverse_image["is_indexed_online"]:
        fraud_flags.append("IMAGE_INDEXED_ONLINE")
        fraud_risk_score += _FRAUD_WEIGHTS["is_indexed_online"]

    # Screen-capture
    if screen_capture["is_screen_capture"]:
        fraud_flags.append("SCREEN_CAPTURE_DETECTED")
        # Weight proportionally to detection confidence
        fraud_risk_score += _FRAUD_WEIGHTS["is_screen_capture"] * screen_capture["confidence"]

    fraud_risk_score = round(min(fraud_risk_score, 1.0), 4)

    return {
        "image_type": image_type,
        "order_id": order_id,
        "customer_id": customer_id,
        "exif_metadata": exif_meta,
        "timestamp_check": ts_result,
        "synthesis_check": synthesis_result,
        "device_fingerprint": fingerprint,
        "multi_account_check": {
            "detected": multi_account_detected,
            "other_customer_ids": other_accounts,
        },
        "phash": phash_str,
        "hash_check": hash_check,
        "reverse_image_check": reverse_image,
        "screen_capture_check": screen_capture,
        "fraud_flags": fraud_flags,
        "fraud_risk_score": fraud_risk_score,
    }


# ---------------------------------------------------------------------------
# Tiered Decision Enforcement
# ---------------------------------------------------------------------------


def enforce_fraud_decision(
    fraud_result: dict[str, Any],
    order_id: str,
    customer_id: str,
) -> dict[str, Any]:
    """
    Map the aggregated *fraud_risk_score* to one of three tiered outcomes:

    - **HARD_REJECT**   (score ≥ 0.8) – Logs to suspicious-activity log.
    - **MANUAL_REVIEW** (0.4 ≤ score < 0.8) – Queues for human review.
    - **CLEARED**       (score < 0.4) – Security checks passed.
    """
    score: float = fraud_result.get("fraud_risk_score", 0.0)
    flags: list[str] = fraud_result.get("fraud_flags", [])
    phash: str = fraud_result.get("phash", "")

    if score >= settings.FRAUD_HARD_REJECT_THRESHOLD:
        log_entry = {
            "order_id": order_id,
            "customer_id": customer_id,
            "fraud_risk_score": score,
            "fraud_flags": flags,
            "phash": phash,
            "action": "HARD_REJECT",
        }
        mock_db.add_to_suspicious_log(log_entry)
        if phash:
            mock_db.register_hash(
                phash,
                {"customer_id": customer_id, "order_id": order_id, "flagged_at": datetime.now(tz=timezone.utc).isoformat()},
            )
        return {
            "action": "HARD_REJECT",
            "message": "Unable to verify proof of delivery.",
            "fraud_risk_score": score,
        }

    if score >= settings.FRAUD_MANUAL_REVIEW_THRESHOLD:
        queue_entry = {
            "order_id": order_id,
            "customer_id": customer_id,
            "fraud_risk_score": score,
            "fraud_flags": flags,
            "phash": phash,
            "action": "MANUAL_REVIEW",
        }
        mock_db.add_to_admin_queue(queue_entry)
        return {
            "action": "MANUAL_REVIEW",
            "message": "Claim flagged for manual review.",
            "fraud_risk_score": score,
        }

    return {
        "action": "CLEARED",
        "message": "Security checks passed.",
        "fraud_risk_score": score,
    }
