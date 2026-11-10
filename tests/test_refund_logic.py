"""
Refund Engine Test Suite
Tests deterministic payout allocation under all standard refund scenarios.
"""
import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# sys.path so we can import from backend/app without installing anything
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.core.refund_engine import calculate_refund, make_vision_result  # noqa: E402
from app.database.mock_db import (  # noqa: E402
    add_to_admin_queue,
    add_to_suspicious_log,
    check_device_multi_account,
    get_admin_queue,
    get_order,
    get_suspicious_log,
    register_device_fingerprint,
    reset_all,
)


# ===========================================================================
# Helpers
# ===========================================================================

def make_mock_vision_result(
    delivered_items: List[str],
    damage_detected: bool = False,
    wrong_items: Optional[List[str]] = None,
    is_stock_photo: bool = False,
    item_conditions: Optional[Dict[str, str]] = None,
    qty_visible: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Thin wrapper around ``make_vision_result`` that mirrors the vision pipeline
    output format expected by ``calculate_refund``.
    """
    return make_vision_result(
        delivered_items=delivered_items,
        damage_detected=damage_detected,
        wrong_items=wrong_items,
        is_stock_photo=is_stock_photo,
        item_conditions=item_conditions,
        qty_visible=qty_visible,
    )


_CLEARED_FRAUD = {"action": "CLEARED", "fraud_risk_score": 0.1}


# ===========================================================================
# TestRefundCalculation
# ===========================================================================

class TestRefundCalculation:
    def test_wrong_item_full_refund_plus_delivery_waiver(self):
        """
        Wrong item detected → full item refund (qty×unit_price) + delivery waiver.
        Order: Butter Chicken × 2 @ ₹320 each.
        Expected: item_refund=640.0, delivery_waiver=49.0, total=689.0
        """
        order_items = [{"name": "Butter Chicken", "qty": 2, "unit_price": 320.0}]
        vision = make_mock_vision_result(
            delivered_items=[],
            wrong_items=["Butter Chicken"],
            damage_detected=False,
        )
        result = calculate_refund(order_items, vision, _CLEARED_FRAUD, delivery_fee=49.0)

        assert result["status"] == "CLEARED"
        breakdown = result["breakdown"]
        assert breakdown is not None
        assert breakdown["item_refunds"]["Butter Chicken"] == pytest.approx(640.0)
        assert breakdown["delivery_fee_waiver"] == pytest.approx(49.0)
        assert result["total_refund"] == pytest.approx(689.0)

    def test_spillage_full_item_refund(self):
        """
        Damage detected with condition 'leaking' → full refund for that item.
        Order: Dal Makhani × 1 @ ₹220.
        Expected: item_refund=220.0
        """
        order_items = [{"name": "Dal Makhani", "qty": 1, "unit_price": 220.0}]
        vision = make_mock_vision_result(
            delivered_items=["Dal Makhani"],
            damage_detected=True,
            item_conditions={"Dal Makhani": "leaking"},
        )
        result = calculate_refund(order_items, vision, _CLEARED_FRAUD, delivery_fee=49.0)

        assert result["status"] == "CLEARED"
        breakdown = result["breakdown"]
        assert breakdown["item_refunds"]["Dal Makhani"] == pytest.approx(220.0)

    def test_missing_quantity_linear_refund(self):
        """
        2 of 4 Garlic Naan missing → refund for 2 × ₹55 = ₹110.
        """
        order_items = [{"name": "Garlic Naan", "qty": 4, "unit_price": 55.0}]
        vision = make_mock_vision_result(
            delivered_items=["Garlic Naan"],
            damage_detected=False,
            qty_visible={"Garlic Naan": 2},
        )
        result = calculate_refund(order_items, vision, _CLEARED_FRAUD, delivery_fee=49.0)

        assert result["status"] == "CLEARED"
        breakdown = result["breakdown"]
        assert breakdown["item_refunds"]["Garlic Naan"] == pytest.approx(110.0)

    def test_full_order_delivered_no_refund(self):
        """
        All items delivered correctly → total_refund = 0.0.
        """
        order_items = [
            {"name": "Butter Chicken", "qty": 2, "unit_price": 320.0},
            {"name": "Garlic Naan", "qty": 4, "unit_price": 55.0},
        ]
        vision = make_mock_vision_result(
            delivered_items=["Butter Chicken", "Garlic Naan"],
            damage_detected=False,
        )
        result = calculate_refund(order_items, vision, _CLEARED_FRAUD, delivery_fee=49.0)

        assert result["status"] == "CLEARED"
        assert result["total_refund"] == pytest.approx(0.0)
        breakdown = result["breakdown"]
        for item_name, refund_amt in breakdown["item_refunds"].items():
            assert refund_amt == pytest.approx(0.0), f"{item_name} should have 0 refund"
        assert breakdown["delivery_fee_waiver"] == pytest.approx(0.0)

    def test_multiple_items_mixed_issues(self):
        """
        Mixed order:
          Butter Chicken (2×320) → wrong item → 640.0
          Garlic Naan (4×55) → qty_visible=2 → 110.0
          Dal Makhani (1×220) → ok → 0.0
          delivery waiver → 49.0
          total → 799.0
        """
        order_items = [
            {"name": "Butter Chicken", "qty": 2, "unit_price": 320.0},
            {"name": "Garlic Naan", "qty": 4, "unit_price": 55.0},
            {"name": "Dal Makhani", "qty": 1, "unit_price": 220.0},
        ]
        vision = make_mock_vision_result(
            delivered_items=["Garlic Naan", "Dal Makhani"],
            wrong_items=["Butter Chicken"],
            damage_detected=False,
            qty_visible={"Garlic Naan": 2},
        )
        result = calculate_refund(order_items, vision, _CLEARED_FRAUD, delivery_fee=49.0)

        assert result["status"] == "CLEARED"
        breakdown = result["breakdown"]
        assert breakdown["item_refunds"]["Butter Chicken"] == pytest.approx(640.0)
        assert breakdown["item_refunds"]["Garlic Naan"] == pytest.approx(110.0)
        assert breakdown["item_refunds"]["Dal Makhani"] == pytest.approx(0.0)
        assert breakdown["delivery_fee_waiver"] == pytest.approx(49.0)
        assert result["total_refund"] == pytest.approx(799.0)

    def test_fraud_rejected_claim_returns_zero_refund(self):
        """HARD_REJECT fraud result → breakdown=None, status=REJECTED, total=0."""
        order_items = [{"name": "Butter Chicken", "qty": 2, "unit_price": 320.0}]
        vision = make_mock_vision_result(
            delivered_items=[],
            wrong_items=["Butter Chicken"],
        )
        fraud = {"action": "HARD_REJECT", "fraud_risk_score": 0.95}

        result = calculate_refund(order_items, vision, fraud, delivery_fee=49.0)

        assert result["status"] == "REJECTED"
        assert result["breakdown"] is None
        assert result["total_refund"] == pytest.approx(0.0)

    def test_manual_review_claim_returns_pending_status(self):
        """MANUAL_REVIEW fraud result → status=PENDING_REVIEW, breakdown=None."""
        order_items = [{"name": "Dal Makhani", "qty": 1, "unit_price": 220.0}]
        vision = make_mock_vision_result(delivered_items=["Dal Makhani"])
        fraud = {"action": "MANUAL_REVIEW", "fraud_risk_score": 0.6}

        result = calculate_refund(order_items, vision, fraud, delivery_fee=49.0)

        assert result["status"] == "PENDING_REVIEW"
        assert result["breakdown"] is None

    def test_refund_percentage_calculated_correctly(self):
        """
        Order gross = 1079.0  (2×320 + 4×55 + 1×220 – but we control the numbers here)
        We use Butter Chicken 2×320 wrong → refund=640, delivery=49, total=689.
        gross_total = 640, delivery = 49 → denominator = 689.
        refund_pct = 689/689 * 100 = 100.0

        Instead test a scenario where refund < total:
          Order: Garlic Naan 4×55=220, Dal Makhani 1×220=220, total gross=440+delivery49=489
          Naan qty_visible=2 → refund 110, delivery 49, total 159
          pct = 159/489 * 100 ≈ 32.5
        """
        order_items = [
            {"name": "Garlic Naan", "qty": 4, "unit_price": 55.0},
            {"name": "Dal Makhani", "qty": 1, "unit_price": 220.0},
        ]
        vision = make_mock_vision_result(
            delivered_items=["Dal Makhani", "Garlic Naan"],
            qty_visible={"Garlic Naan": 2},
        )
        result = calculate_refund(order_items, vision, _CLEARED_FRAUD, delivery_fee=49.0)

        # refund = 110 (naan) + 49 (waiver) = 159
        # denominator = (220+220) + 49 = 489
        expected_pct = round(159.0 / 489.0 * 100, 1)
        assert result["refund_percentage"] == pytest.approx(expected_pct, abs=0.2)
        assert result["total_refund"] == pytest.approx(159.0)


# ===========================================================================
# TestMockDatabaseOperations
# ===========================================================================

class TestMockDatabaseOperations:
    """Tests for the in-memory mock database layer."""

    def setup_method(self):
        """Reset mutable DB state before every test."""
        reset_all()

    def test_get_valid_order_returns_data(self):
        """Seeded order ORD-2026-001 should be retrievable."""
        order = get_order("ORD-2026-001")
        assert order is not None
        assert order["order_id"] == "ORD-2026-001"
        assert "items" in order
        assert len(order["items"]) > 0

    def test_get_invalid_order_returns_none(self):
        """Non-existent order ID should return None."""
        order = get_order("ORD-NONEXISTENT")
        assert order is None

    def test_add_to_suspicious_log_persists(self):
        """Entry added to suspicious log should appear in get_suspicious_log()."""
        entry = {
            "order_id": "ORD-FRAUD-001",
            "fraud_risk_score": 0.92,
            "action": "HARD_REJECT",
            "flags": ["PREDATES_DELIVERY"],
        }
        add_to_suspicious_log(entry)
        log = get_suspicious_log()

        assert len(log) == 1
        saved = log[0]
        assert saved["order_id"] == "ORD-FRAUD-001"
        assert saved["action"] == "HARD_REJECT"
        assert saved["fraud_risk_score"] == pytest.approx(0.92)

    def test_add_to_admin_queue_persists(self):
        """Entry added to admin queue should appear in get_admin_queue()."""
        entry = {
            "order_id": "ORD-REVIEW-002",
            "fraud_risk_score": 0.55,
            "action": "MANUAL_REVIEW",
            "flags": ["MISSING_EXIF_TIMESTAMP"],
        }
        add_to_admin_queue(entry)
        queue = get_admin_queue()

        assert len(queue) == 1
        saved = queue[0]
        assert saved["order_id"] == "ORD-REVIEW-002"
        assert saved["action"] == "MANUAL_REVIEW"

    def test_device_fingerprint_multi_account_detection(self):
        """
        Register the same device fingerprint under two different customer IDs.
        check_device_multi_account should return the *other* customer's ID.
        """
        fp = "FP-DEVICE-XYZ-9876"
        register_device_fingerprint(fp, "CUST-101")
        register_device_fingerprint(fp, "CUST-202")

        # From CUST-101's perspective, the other account is CUST-202
        other_for_101 = check_device_multi_account(fp, "CUST-101")
        assert other_for_101 == "CUST-202"

        # From CUST-202's perspective, the other account is CUST-101
        other_for_202 = check_device_multi_account(fp, "CUST-202")
        assert other_for_202 == "CUST-101"

    def test_device_fingerprint_single_account_no_flag(self):
        """A fingerprint registered to only one customer should not trigger the check."""
        fp = "FP-CLEAN-DEVICE-001"
        register_device_fingerprint(fp, "CUST-303")

        other = check_device_multi_account(fp, "CUST-303")
        assert other is None

    def test_suspicious_log_isolated_between_tests(self):
        """
        After reset_all() the log should be empty; proves test isolation.
        """
        log = get_suspicious_log()
        assert log == [], f"Log should be empty after reset; got {log}"

    def test_multiple_entries_in_suspicious_log(self):
        """Multiple add calls should stack chronologically."""
        add_to_suspicious_log({"order_id": "ORD-A", "action": "HARD_REJECT"})
        add_to_suspicious_log({"order_id": "ORD-B", "action": "HARD_REJECT"})
        log = get_suspicious_log()
        assert len(log) == 2
        assert log[0]["order_id"] == "ORD-A"
        assert log[1]["order_id"] == "ORD-B"
