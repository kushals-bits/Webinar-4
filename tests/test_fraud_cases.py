"""
Fraud Detection Test Suite
Tests the full fraud detection pipeline with standard vs. malicious payloads.
"""
import io
import os
import sys
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# sys.path so we can import from backend/app without installing anything
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.core.fraud_detection import (  # noqa: E402
    TimestampFlag,
    check_duplicate_hash,
    check_reverse_image_webhook,
    compute_fraud_score,
    compute_phash,
    detect_screen_capture,
    enforce_fraud_decision,
    extract_exif_metadata,
    hamming_distance,
    validate_image_timestamp,
)


# ===========================================================================
# Helper
# ===========================================================================

def make_test_image(
    width: int = 640,
    height: int = 480,
    color: tuple = (255, 100, 50),
    fmt: str = "JPEG",
    add_exif: bool = True,
    exif_datetime: Optional[str] = None,
    make: str = "Apple",
    model: str = "iPhone 15",
    software: str = "iOS 17.0",
) -> bytes:
    """
    Create a solid-colour PIL image and return it as raw bytes.

    If *add_exif* is True the function tries to embed minimal EXIF data
    using the ``piexif`` library.  If piexif is not installed the image is
    still returned – just without EXIF.
    """
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()

    if add_exif:
        try:
            import piexif  # type: ignore

            dt_str = exif_datetime or "2026:11:09 18:35:00"
            dt_bytes = dt_str.encode("ascii")

            zeroth_ifd = {
                piexif.ImageIFD.Make: make.encode("ascii"),
                piexif.ImageIFD.Model: model.encode("ascii"),
                piexif.ImageIFD.Software: software.encode("ascii"),
            }
            exif_ifd = {
                piexif.ExifIFD.DateTimeOriginal: dt_bytes,
            }
            exif_bytes = piexif.dump({"0th": zeroth_ifd, "Exif": exif_ifd})
            img.save(buf, format=fmt, exif=exif_bytes)
        except Exception:
            # piexif unavailable or failed – save without EXIF
            img.save(buf, format=fmt)
    else:
        img.save(buf, format=fmt)

    return buf.getvalue()


# ===========================================================================
# TestExifMetadataExtraction
# ===========================================================================

class TestExifMetadataExtraction:
    def test_extract_valid_exif_metadata(self):
        """Image with EXIF should return make / model / software fields."""
        img_bytes = make_test_image(add_exif=True, make="Apple", model="iPhone 15", software="iOS 17.0")
        result = extract_exif_metadata(img_bytes)

        # raw_exif_present should be True when EXIF is embedded
        # (piexif required; if not installed the test degrades gracefully)
        try:
            import piexif  # noqa: F401
            assert result["raw_exif_present"] is True, "Expected raw_exif_present=True when EXIF embedded"
            assert result["make"] is not None, "Expected make to be present"
            assert result["model"] is not None, "Expected model to be present"
            assert result["software"] is not None, "Expected software to be present"
            assert "Apple" in result["make"]
            assert "iPhone 15" in result["model"]
            assert "iOS 17.0" in result["software"]
        except ImportError:
            pytest.skip("piexif not installed; skipping strict EXIF assertion")

    def test_extract_missing_exif_returns_defaults(self):
        """Image without EXIF should return raw_exif_present=False."""
        img_bytes = make_test_image(add_exif=False)
        result = extract_exif_metadata(img_bytes)
        assert result["raw_exif_present"] is False
        assert result["make"] is None
        assert result["model"] is None
        assert result["datetime_parsed"] is None


# ===========================================================================
# TestTimestampValidation
# ===========================================================================

class TestTimestampValidation:
    _DELIVERY = datetime(2026, 11, 9, 18, 0, 0)  # 18:00:00

    def test_valid_timestamp_within_window(self):
        """Photo taken 30 min after delivery → VALID."""
        image_dt = self._DELIVERY + timedelta(minutes=30)
        result = validate_image_timestamp(image_dt, self._DELIVERY)
        assert result["flag"] == TimestampFlag.VALID
        assert result["valid"] is True
        assert result["delta_minutes"] == pytest.approx(30.0, abs=0.1)

    def test_timestamp_predates_delivery(self):
        """Photo taken 10 min BEFORE delivery → PREDATES_DELIVERY."""
        image_dt = self._DELIVERY - timedelta(minutes=10)
        result = validate_image_timestamp(image_dt, self._DELIVERY)
        assert result["flag"] == TimestampFlag.PREDATES_DELIVERY
        assert result["valid"] is False
        assert result["delta_minutes"] < 0

    def test_timestamp_exceeds_window(self):
        """Photo taken 3 hours after delivery → POST_DELIVERY_WINDOW_EXCEEDED."""
        image_dt = self._DELIVERY + timedelta(hours=3)
        result = validate_image_timestamp(image_dt, self._DELIVERY, max_window_hours=2.0)
        assert result["flag"] == TimestampFlag.POST_DELIVERY_WINDOW_EXCEEDED
        assert result["valid"] is False
        assert result["delta_minutes"] == pytest.approx(180.0, abs=0.1)

    def test_missing_timestamp_flags_correctly(self):
        """None timestamp → MISSING_EXIF_TIMESTAMP."""
        result = validate_image_timestamp(None, self._DELIVERY)
        assert result["flag"] == TimestampFlag.MISSING_EXIF_TIMESTAMP
        assert result["valid"] is False
        assert result["delta_minutes"] is None


# ===========================================================================
# TestPHashService
# ===========================================================================

class TestPHashService:
    def test_phash_computation_returns_hex_string(self):
        """compute_phash should return a 16-character hex string (hash_size=8)."""
        img_bytes = make_test_image(color=(200, 150, 100))
        phash = compute_phash(img_bytes)
        assert isinstance(phash, str)
        assert len(phash) == 16, f"Expected 16-char hex; got len={len(phash)}: {phash!r}"
        # Must be valid hex
        int(phash, 16)

    def test_identical_images_zero_hamming_distance(self):
        """The same image hashed twice should have Hamming distance == 0."""
        img_bytes = make_test_image(color=(100, 200, 50))
        h1 = compute_phash(img_bytes)
        h2 = compute_phash(img_bytes)
        assert hamming_distance(h1, h2) == 0

    def test_different_images_nonzero_hamming_distance(self):
        """Two distinctly coloured images should have Hamming distance > 0."""
        img_a = make_test_image(color=(255, 0, 0))    # red
        img_b = make_test_image(color=(0, 0, 255))    # blue
        h_a = compute_phash(img_a)
        h_b = compute_phash(img_b)
        assert hamming_distance(h_a, h_b) > 0

    def test_known_fraud_hash_detected(self):
        """
        Insert a known pHash into the mock DB list, then verify that the same
        image is flagged as a duplicate.
        """
        img_bytes = make_test_image(color=(123, 45, 67))
        known_hash = compute_phash(img_bytes)  # treat this as already in DB
        mock_db_hashes = [known_hash]

        new_phash = compute_phash(img_bytes)   # re-hash the identical image
        result = check_duplicate_hash(new_phash, mock_db_hashes)
        assert result["is_duplicate"] is True
        assert result["matched_hash"] == known_hash
        assert result["distance"] == 0

    def test_clean_hash_not_flagged(self):
        """A fresh image whose hash is not in the DB should NOT be flagged."""
        img_bytes = make_test_image(color=(10, 20, 30))
        fresh_hash = compute_phash(img_bytes)
        mock_db_hashes: list = []  # empty fraud DB
        result = check_duplicate_hash(fresh_hash, mock_db_hashes)
        assert result["is_duplicate"] is False
        assert result["matched_hash"] is None


# ===========================================================================
# TestScreenCaptureDetection
# ===========================================================================

class TestScreenCaptureDetection:
    def test_uniform_border_detected_as_screen_capture(self):
        """
        Image with a solid black 20-px border around coloured content
        should be flagged as a screen capture.
        """
        width, height = 320, 240
        border = 20
        img = Image.new("RGB", (width, height), (0, 0, 0))  # all black

        # Paint the interior with a vivid colour so the content differs
        for x in range(border, width - border):
            for y in range(border, height - border):
                img.putpixel((x, y), (255, 100, 50))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        result = detect_screen_capture(img_bytes, border_px=20, uniformity_threshold=0.90)
        assert result["is_screen_capture"] is True, (
            f"Expected screen capture; got confidence={result['confidence']}"
        )

    def test_natural_food_photo_clears_check(self):
        """
        An image with randomly coloured pixels should NOT be flagged as a
        screen capture (border is not uniform).
        """
        import random
        random.seed(42)
        width, height = 320, 240
        img = Image.new("RGB", (width, height))
        pixels = [
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(width * height)
        ]
        img.putdata(pixels)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        img_bytes = buf.getvalue()

        result = detect_screen_capture(img_bytes, border_px=15, uniformity_threshold=0.98)
        assert result["is_screen_capture"] is False, (
            f"Did not expect screen-capture flag; confidence={result['confidence']}"
        )


# ===========================================================================
# TestFraudEnforcement
# ===========================================================================

class TestFraudEnforcement:
    def test_score_above_threshold_hard_reject(self):
        """fraud_risk_score=0.9 should produce HARD_REJECT."""
        result = enforce_fraud_decision(0.9, flags=["PREDATES_DELIVERY", "DUPLICATE_HASH"])
        assert result.action == "HARD_REJECT"
        assert result.fraud_risk_score == pytest.approx(0.9)

    def test_score_in_manual_range_queued(self):
        """fraud_risk_score=0.6 should produce MANUAL_REVIEW."""
        result = enforce_fraud_decision(0.6, flags=["POST_DELIVERY_WINDOW_EXCEEDED"])
        assert result.action == "MANUAL_REVIEW"

    def test_score_below_threshold_clears(self):
        """fraud_risk_score=0.2 should produce CLEARED."""
        result = enforce_fraud_decision(0.2, flags=[])
        assert result.action == "CLEARED"

    def test_hard_reject_logged_to_suspicious_activity(self):
        """After HARD_REJECT, the suspicious_log list should contain an entry."""
        suspicious_log: list = []
        enforce_fraud_decision(
            0.85,
            flags=["PREDATES_DELIVERY"],
            suspicious_log=suspicious_log,
            order_id="ORD-TEST-999",
        )
        assert len(suspicious_log) == 1, "Expected exactly one log entry"
        entry = suspicious_log[0]
        assert entry["order_id"] == "ORD-TEST-999"
        assert entry["action"] == "HARD_REJECT"
        assert entry["fraud_risk_score"] == pytest.approx(0.85)


# ===========================================================================
# TestReverseImageWebhookMock
# ===========================================================================

class TestReverseImageWebhookMock:
    def test_hash_starting_with_a_flagged_online(self):
        """Hash starting with 'a' → is_indexed_online=True."""
        phash = "a" + "0" * 15
        result = check_reverse_image_webhook(phash)
        assert result["is_indexed_online"] is True

    def test_hash_starting_with_d_flagged_online(self):
        """Hash starting with 'd' → is_indexed_online=True."""
        phash = "d" + "f" * 15
        result = check_reverse_image_webhook(phash)
        assert result["is_indexed_online"] is True

    def test_clean_hash_not_online(self):
        """Hash starting with 'z' → is_indexed_online=False."""
        phash = "z" + "1" * 15
        result = check_reverse_image_webhook(phash)
        assert result["is_indexed_online"] is False
