"""
Fraud Detection Pipeline
Covers EXIF extraction, timestamp validation, pHash deduplication,
screen-capture detection, scoring, and enforcement.
"""
from __future__ import annotations

import hashlib
import io
import math
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


# ---------------------------------------------------------------------------
# EXIF helpers
# ---------------------------------------------------------------------------

def _try_piexif(img_bytes: bytes) -> Optional[Dict[str, Any]]:
    """Attempt to read EXIF via piexif; return None if unavailable."""
    try:
        import piexif  # type: ignore
        exif_dict = piexif.load(img_bytes)
        zeroth = exif_dict.get("0th", {})
        exif_ifd = exif_dict.get("Exif", {})

        def _decode(v: Any) -> Optional[str]:
            if isinstance(v, bytes):
                return v.rstrip(b"\x00").decode("utf-8", errors="replace")
            return str(v) if v else None

        result: Dict[str, Any] = {
            "raw_exif_present": bool(zeroth or exif_ifd),
        }

        # piexif tag IDs
        MAKE = 271
        MODEL = 272
        SOFTWARE = 305
        DT_ORIGINAL = 36867  # DateTimeOriginal in Exif IFD

        result["make"] = _decode(zeroth.get(MAKE))
        result["model"] = _decode(zeroth.get(MODEL))
        result["software"] = _decode(zeroth.get(SOFTWARE))
        raw_dt = _decode(exif_ifd.get(DT_ORIGINAL))
        result["datetime_original"] = raw_dt

        # Parse datetime
        result["datetime_parsed"] = None
        if raw_dt:
            try:
                result["datetime_parsed"] = datetime.strptime(raw_dt, "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
        return result
    except Exception:
        return None


def extract_exif_metadata(img_bytes: bytes) -> Dict[str, Any]:
    """
    Extract EXIF metadata from raw image bytes.

    Returns a dict with keys:
        raw_exif_present (bool)
        make (str|None)
        model (str|None)
        software (str|None)
        datetime_original (str|None)  – raw string from EXIF
        datetime_parsed (datetime|None)
    """
    result = _try_piexif(img_bytes)
    if result is not None:
        return result

    # Fallback: use Pillow's _getexif / getexif
    defaults: Dict[str, Any] = {
        "raw_exif_present": False,
        "make": None,
        "model": None,
        "software": None,
        "datetime_original": None,
        "datetime_parsed": None,
    }
    try:
        img = Image.open(io.BytesIO(img_bytes))
        exif_data: Optional[Dict] = None
        if hasattr(img, "getexif"):
            exif_data = img.getexif()
        elif hasattr(img, "_getexif"):
            exif_data = img._getexif()

        if not exif_data:
            return defaults

        # PIL tag IDs
        TAG_MAKE = 271
        TAG_MODEL = 272
        TAG_SOFTWARE = 305
        TAG_DT_ORIGINAL = 36867

        def _d(v: Any) -> Optional[str]:
            if isinstance(v, bytes):
                return v.rstrip(b"\x00").decode("utf-8", errors="replace")
            return str(v) if v else None

        defaults["raw_exif_present"] = True
        defaults["make"] = _d(exif_data.get(TAG_MAKE))
        defaults["model"] = _d(exif_data.get(TAG_MODEL))
        defaults["software"] = _d(exif_data.get(TAG_SOFTWARE))
        raw_dt = _d(exif_data.get(TAG_DT_ORIGINAL))
        defaults["datetime_original"] = raw_dt
        if raw_dt:
            try:
                defaults["datetime_parsed"] = datetime.strptime(raw_dt, "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
        return defaults
    except Exception:
        return defaults


# ---------------------------------------------------------------------------
# Timestamp validation
# ---------------------------------------------------------------------------

class TimestampFlag:
    VALID = "VALID"
    PREDATES_DELIVERY = "PREDATES_DELIVERY"
    POST_DELIVERY_WINDOW_EXCEEDED = "POST_DELIVERY_WINDOW_EXCEEDED"
    MISSING_EXIF_TIMESTAMP = "MISSING_EXIF_TIMESTAMP"


def validate_image_timestamp(
    image_datetime: Optional[datetime],
    delivery_datetime: datetime,
    max_window_hours: float = 2.0,
) -> Dict[str, Any]:
    """
    Validate that the photo timestamp falls within the allowed post-delivery window.

    Returns:
        flag   (str)   – one of TimestampFlag constants
        valid  (bool)
        delta_minutes (float|None) – positive = after delivery
    """
    if image_datetime is None:
        return {"flag": TimestampFlag.MISSING_EXIF_TIMESTAMP, "valid": False, "delta_minutes": None}

    delta: timedelta = image_datetime - delivery_datetime
    delta_minutes = delta.total_seconds() / 60.0

    if delta_minutes < 0:
        return {"flag": TimestampFlag.PREDATES_DELIVERY, "valid": False, "delta_minutes": delta_minutes}

    if delta_minutes > max_window_hours * 60:
        return {
            "flag": TimestampFlag.POST_DELIVERY_WINDOW_EXCEEDED,
            "valid": False,
            "delta_minutes": delta_minutes,
        }

    return {"flag": TimestampFlag.VALID, "valid": True, "delta_minutes": delta_minutes}


# ---------------------------------------------------------------------------
# Perceptual hash (pHash)
# ---------------------------------------------------------------------------

def _dct_1d(values: List[float]) -> List[float]:
    """Naive 1-D DCT-II (slow but dependency-free)."""
    n = len(values)
    result = []
    for k in range(n):
        s = sum(values[j] * math.cos(math.pi * k * (2 * j + 1) / (2 * n)) for j in range(n))
        result.append(s)
    return result


def compute_phash(img_bytes: bytes, hash_size: int = 8) -> str:
    """
    Compute a perceptual hash (pHash) of the image.

    Returns a hex string of length hash_size*2 (default 16 chars).
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    # Resize to (hash_size*4) × (hash_size*4) for DCT
    dct_size = hash_size * 4
    img = img.resize((dct_size, dct_size), Image.LANCZOS)
    pixels = list(img.getdata())

    # 2-D DCT via two passes of 1-D DCT
    rows = [pixels[r * dct_size:(r + 1) * dct_size] for r in range(dct_size)]
    dct_rows = [_dct_1d([float(v) for v in row]) for row in rows]
    # Transpose and DCT columns
    dct_cols = [[dct_rows[r][c] for r in range(dct_size)] for c in range(dct_size)]
    dct_2d = [_dct_1d(col) for col in dct_cols]

    # Extract top-left hash_size × hash_size (excluding DC component)
    flat = []
    for c in range(hash_size):
        for r in range(hash_size):
            flat.append(dct_2d[c][r])

    # Median threshold
    sorted_vals = sorted(flat)
    median = sorted_vals[len(sorted_vals) // 2]

    bits = [1 if v >= median else 0 for v in flat]

    # Pack bits into bytes
    byte_vals = []
    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]
        while len(chunk) < 8:
            chunk.append(0)
        byte_vals.append(int("".join(str(b) for b in chunk), 2))

    return bytes(byte_vals).hex()


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute bit-level Hamming distance between two hex hashes."""
    if len(hash1) != len(hash2):
        raise ValueError("Hash lengths must match")
    dist = 0
    for c1, c2 in zip(hash1, hash2):
        xor = int(c1, 16) ^ int(c2, 16)
        dist += bin(xor).count("1")
    return dist


def check_duplicate_hash(
    phash: str,
    known_hashes: List[str],
    max_hamming: int = 5,
) -> Dict[str, Any]:
    """
    Check whether *phash* is within *max_hamming* bits of any known fraud hash.

    Returns:
        is_duplicate (bool)
        matched_hash (str|None)
        distance (int|None)
    """
    for known in known_hashes:
        try:
            dist = hamming_distance(phash, known)
            if dist <= max_hamming:
                return {"is_duplicate": True, "matched_hash": known, "distance": dist}
        except ValueError:
            continue
    return {"is_duplicate": False, "matched_hash": None, "distance": None}


# ---------------------------------------------------------------------------
# Screen-capture detection
# ---------------------------------------------------------------------------

def detect_screen_capture(img_bytes: bytes, border_px: int = 15, uniformity_threshold: float = 0.98) -> Dict[str, Any]:
    """
    Detect whether an image is a screen capture by checking border uniformity.

    A solid-color border of *border_px* pixels is a strong indicator that
    the photo was taken of a screen rather than a real food item.

    Returns:
        is_screen_capture (bool)
        confidence (float 0-1)
        reason (str)
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    width, height = img.size

    if width < border_px * 2 + 10 or height < border_px * 2 + 10:
        return {"is_screen_capture": False, "confidence": 0.0, "reason": "image_too_small"}

    # Collect border pixels
    border_pixels: List[Tuple[int, int, int]] = []
    for x in range(width):
        for y in range(border_px):  # top
            border_pixels.append(img.getpixel((x, y)))
        for y in range(height - border_px, height):  # bottom
            border_pixels.append(img.getpixel((x, y)))
    for y in range(border_px, height - border_px):
        for x in range(border_px):  # left
            border_pixels.append(img.getpixel((x, y)))
        for x in range(width - border_px, width):  # right
            border_pixels.append(img.getpixel((x, y)))

    if not border_pixels:
        return {"is_screen_capture": False, "confidence": 0.0, "reason": "no_border_pixels"}

    # Measure uniformity: fraction of pixels matching the most common colour
    from collections import Counter
    counts = Counter(border_pixels)
    most_common_count = counts.most_common(1)[0][1]
    uniformity = most_common_count / len(border_pixels)

    is_sc = uniformity >= uniformity_threshold
    return {
        "is_screen_capture": is_sc,
        "confidence": round(uniformity, 4),
        "reason": "uniform_border" if is_sc else "natural_variation",
    }


# ---------------------------------------------------------------------------
# Reverse image / online-index webhook (mock)
# ---------------------------------------------------------------------------

def check_reverse_image_webhook(phash: str) -> Dict[str, Any]:
    """
    Simulate a reverse-image-search webhook to detect stock/online photos.

    Real implementation would call an external API; here we use a deterministic
    rule based on the first nibble of the hash to allow testability.

    Hash starting with 'a' or 'd' → flagged as found online.
    """
    first_char = phash[0].lower() if phash else "z"
    is_indexed = first_char in {"a", "d"}
    return {
        "is_indexed_online": is_indexed,
        "source": "mock_webhook",
        "phash": phash,
    }


# ---------------------------------------------------------------------------
# Fraud scoring & enforcement
# ---------------------------------------------------------------------------

HARD_REJECT_THRESHOLD = 0.75
MANUAL_REVIEW_THRESHOLD = 0.40


@dataclass
class FraudResult:
    fraud_risk_score: float
    action: str
    flags: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def compute_fraud_score(
    timestamp_flag: str,
    is_duplicate: bool,
    is_screen_capture: bool,
    is_indexed_online: bool,
    screen_capture_confidence: float = 0.0,
) -> float:
    """
    Combine individual fraud signals into a 0-1 risk score.
    """
    score = 0.0
    if timestamp_flag == TimestampFlag.PREDATES_DELIVERY:
        score += 0.40
    elif timestamp_flag == TimestampFlag.MISSING_EXIF_TIMESTAMP:
        score += 0.15
    elif timestamp_flag == TimestampFlag.POST_DELIVERY_WINDOW_EXCEEDED:
        score += 0.20

    if is_duplicate:
        score += 0.35
    if is_screen_capture:
        score += 0.25 * screen_capture_confidence
    if is_indexed_online:
        score += 0.30

    return min(score, 1.0)


def enforce_fraud_decision(
    fraud_risk_score: float,
    flags: List[str],
    suspicious_log: Optional[List[Dict]] = None,
    order_id: Optional[str] = None,
) -> FraudResult:
    """
    Given a risk score, determine the enforcement action.

    Thresholds:
        score >= 0.75  → HARD_REJECT  (logged to suspicious_log)
        score >= 0.40  → MANUAL_REVIEW
        score <  0.40  → CLEARED
    """
    if fraud_risk_score >= HARD_REJECT_THRESHOLD:
        action = "HARD_REJECT"
    elif fraud_risk_score >= MANUAL_REVIEW_THRESHOLD:
        action = "MANUAL_REVIEW"
    else:
        action = "CLEARED"

    result = FraudResult(
        fraud_risk_score=fraud_risk_score,
        action=action,
        flags=flags,
    )

    if action == "HARD_REJECT" and suspicious_log is not None:
        suspicious_log.append(
            {
                "order_id": order_id,
                "fraud_risk_score": fraud_risk_score,
                "action": action,
                "flags": flags,
            }
        )

    return result
