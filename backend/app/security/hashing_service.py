"""
Perceptual Hashing Service — Anti-Plagiarism & Recycled Image Detection

Provides:
  - compute_phash            : Compute a 64-bit perceptual hash from raw image bytes.
  - hamming_distance         : Bit-level distance between two hex pHash strings.
  - check_hash_against_db    : Cross-reference a pHash with the known-fraud database.
  - generate_reverse_image_webhook_mock : Simulate an external reverse-image-search API.
  - detect_screen_capture_artifacts    : Heuristic analysis for screenshot-based fraud.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
from typing import Any

import imagehash
from PIL import Image

from app.database import mock_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Hashing
# ---------------------------------------------------------------------------


def compute_phash(image_bytes: bytes) -> str:
    """
    Open *image_bytes* with PIL and compute a 64-bit perceptual hash.

    Returns the hash as a lowercase hex string (16 characters).
    """
    img = Image.open(io.BytesIO(image_bytes))
    phash = imagehash.phash(img)
    return str(phash)  # imagehash already formats as hex


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Compute the Hamming (bit-level) distance between two hex pHash strings.

    ``imagehash`` overloads ``-`` to return the Hamming distance.
    """
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return abs(h1 - h2)


# ---------------------------------------------------------------------------
# Database Cross-Check
# ---------------------------------------------------------------------------


def check_hash_against_db(
    phash_str: str,
    config_limit: int = 4,
) -> dict[str, Any]:
    """
    Compare *phash_str* against every entry in ``HISTORICAL_REFUND_HASHES``.

    Returns a dict with keys:
      - ``is_duplicate``   – True if any stored hash is within *config_limit* bits.
      - ``matched_entry``  – The matching DB entry dict, or None.
      - ``hamming_distance`` – The Hamming distance to the best match, or None.
    """
    best_distance: int | None = None
    best_entry: dict | None = None
    best_hash: str | None = None

    for stored_hash, entry in mock_db.HISTORICAL_REFUND_HASHES.items():
        try:
            dist = hamming_distance(phash_str, stored_hash)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not compare hashes '%s' vs '%s': %s", phash_str, stored_hash, exc)
            continue

        if best_distance is None or dist < best_distance:
            best_distance = dist
            best_entry = entry
            best_hash = stored_hash

    if best_distance is not None and best_distance <= config_limit:
        return {
            "is_duplicate": True,
            "matched_entry": {**best_entry, "_stored_hash": best_hash},  # type: ignore[arg-type]
            "hamming_distance": best_distance,
        }

    return {
        "is_duplicate": False,
        "matched_entry": None,
        "hamming_distance": best_distance,  # closest distance even when not matching
    }


# ---------------------------------------------------------------------------
# Reverse-Image Webhook Mock
# ---------------------------------------------------------------------------


def generate_reverse_image_webhook_mock(phash_str: str) -> dict[str, Any]:
    """
    Simulate the response of an external reverse-image-search API.

    Heuristic: hashes starting with 'a' or 'd' are assumed to be indexed
    online (mimicking the two seeded fraud hashes that begin with 'a1…'
    and 'de…').
    """
    first_char = phash_str[0].lower() if phash_str else ""
    if first_char in ("a", "d"):
        return {
            "is_indexed_online": True,
            "source_platforms": ["Google Images", "Pinterest"],
            "confidence": 0.91,
        }
    return {
        "is_indexed_online": False,
        "source_platforms": [],
        "confidence": 0.05,
    }


# ---------------------------------------------------------------------------
# Screen-Capture Artifact Detection
# ---------------------------------------------------------------------------


def detect_screen_capture_artifacts(image_bytes: bytes) -> dict[str, Any]:
    """
    Use heuristics to detect whether *image_bytes* is a screenshot rather
    than an authentic photograph.

    Indicators checked
    ------------------
    1. Aspect ratio matches common screen ratios (16:9, 4:3).
    2. Uniform pixel border (device bezel / status-bar pattern).
    3. High-frequency pixel variance in sampled blocks (Moiré / LCD grid).

    Returns
    -------
    ``{"is_screen_capture": bool, "confidence": float, "indicators": list[str]}``
    """
    indicators: list[str] = []
    confidence_score: float = 0.0

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning("detect_screen_capture_artifacts: could not open image: %s", exc)
        return {"is_screen_capture": False, "confidence": 0.0, "indicators": ["image_load_failed"]}

    width, height = img.size
    if width == 0 or height == 0:
        return {"is_screen_capture": False, "confidence": 0.0, "indicators": []}

    # ------------------------------------------------------------------ #
    # 1. Aspect-ratio check
    # ------------------------------------------------------------------ #
    from math import gcd
    divisor = gcd(width, height)
    ratio_w = width // divisor
    ratio_h = height // divisor

    SCREEN_RATIOS = {(16, 9), (9, 16), (4, 3), (3, 4), (16, 10), (10, 16)}
    if (ratio_w, ratio_h) in SCREEN_RATIOS:
        indicators.append(f"screen_aspect_ratio_{ratio_w}_{ratio_h}")
        confidence_score += 0.20

    # ------------------------------------------------------------------ #
    # 2. Border uniformity (top-5 rows, left-5 cols)
    # ------------------------------------------------------------------ #
    border_pixels: list[tuple[int, int, int]] = []

    # Top 5 rows
    for y in range(min(5, height)):
        for x in range(width):
            border_pixels.append(img.getpixel((x, y)))  # type: ignore[arg-type]

    # Left 5 cols
    for x in range(min(5, width)):
        for y in range(height):
            border_pixels.append(img.getpixel((x, y)))  # type: ignore[arg-type]

    if border_pixels:
        # Compute variance per channel
        n = len(border_pixels)
        mean_r = sum(p[0] for p in border_pixels) / n
        mean_g = sum(p[1] for p in border_pixels) / n
        mean_b = sum(p[2] for p in border_pixels) / n
        var_r = sum((p[0] - mean_r) ** 2 for p in border_pixels) / n
        var_g = sum((p[1] - mean_g) ** 2 for p in border_pixels) / n
        var_b = sum((p[2] - mean_b) ** 2 for p in border_pixels) / n
        avg_variance = (var_r + var_g + var_b) / 3.0

        # Very low variance → uniform border → bezel / status bar
        border_uniformity = max(0.0, 1.0 - (avg_variance / 2000.0))
        if border_uniformity > 0.85:
            indicators.append("uniform_border_detected")
            confidence_score += 0.30

    # ------------------------------------------------------------------ #
    # 3. High-frequency pixel variation (Moiré / LCD-grid pattern)
    # ------------------------------------------------------------------ #
    block_size = 10
    sample_step_x = max(1, width // 10)
    sample_step_y = max(1, height // 10)
    high_freq_blocks = 0
    total_blocks = 0

    for bx in range(0, width - block_size, sample_step_x):
        for by in range(0, height - block_size, sample_step_y):
            block_pixels = [
                img.getpixel((bx + dx, by + dy))  # type: ignore[arg-type]
                for dx in range(block_size)
                for dy in range(block_size)
            ]
            lum = [(p[0] + p[1] + p[2]) / 3 for p in block_pixels]
            avg_lum = sum(lum) / len(lum)
            var_lum = sum((l - avg_lum) ** 2 for l in lum) / len(lum)
            total_blocks += 1
            if var_lum > 1500:
                high_freq_blocks += 1

    if total_blocks > 0:
        hf_ratio = high_freq_blocks / total_blocks
        if hf_ratio > 0.4:
            indicators.append(f"high_frequency_pixel_variation_{hf_ratio:.2f}")
            confidence_score += 0.20 * min(hf_ratio / 0.6, 1.0)

    confidence_score = min(confidence_score, 1.0)
    is_screen_capture = confidence_score >= 0.30

    return {
        "is_screen_capture": is_screen_capture,
        "confidence": round(confidence_score, 4),
        "indicators": indicators,
    }
