"""
Refund Engine — Deterministic Payout Allocation Based on Visual Verification Results

This module translates the outputs of the Vision Pipeline into an auditable,
line-item refund breakdown and produces the final claim decision document that
the FastAPI layer returns to the client.

Refund Tiers
------------
- Full rejection  : Fraud hard-reject → 0 payout, claim closed.
- Pending review  : Fraud manual-review → claim held, no immediate payout.
- Computed refund : Fraud cleared → deterministic per-item calculation:
    * Wrong item delivered               → 100 % of item value
    * Item visually damaged / leaking    → 100 % of item value
    * Quantity shortfall                 → pro-rated by missing fraction
    * Item ok and correct quantity       → 0 % (no refund)
  Plus optional waivers of delivery fee and platform fee.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Line-Item Refund Calculation
# ---------------------------------------------------------------------------


def calculate_item_refund(
    order_items: list[dict[str, Any]],
    food_analysis: dict[str, Any],
    bill_analysis: dict[str, Any],  # kept for future cross-validation use
) -> dict[str, Any]:
    """
    Compute a detailed, line-item refund breakdown.

    Parameters
    ----------
    order_items:
        The ``items`` list from the original order record, each with
        ``name``, ``qty``, and ``unit_price`` keys.
    food_analysis:
        The structured output from the vision pipeline with keys:
          - ``delivered_items``: list of dicts with ``name``, ``condition``,
            ``qty_visible``
          - ``wrong_items``: list of item name strings
          - ``damage_detected``: bool
    bill_analysis:
        The structured output from the bill image parser
        (reserved for cross-validation; not yet used in scoring).

    Returns
    -------
    A dict with keys:
      ``line_items``, ``item_refund_total``, ``delivery_fee_waiver``,
      ``platform_fee_waiver``, ``total_refund``, ``refund_percentage``.
    """
    delivered_items: list[dict] = food_analysis.get("delivered_items", [])
    wrong_items: list[str] = [w.lower() for w in food_analysis.get("wrong_items", [])]
    damage_detected: bool = food_analysis.get("damage_detected", False)

    # Build a lookup: item name (lower) -> vision analysis entry
    delivered_lookup: dict[str, dict] = {
        d["name"].lower(): d for d in delivered_items if "name" in d
    }

    line_items: list[dict[str, Any]] = []
    item_refund_total: float = 0.0
    has_wrong_item = False

    for order_item in order_items:
        name: str = order_item.get("name", "")
        expected_qty: int = int(order_item.get("qty", 1))
        unit_price: float = float(order_item.get("unit_price", 0.0))
        full_item_value: float = expected_qty * unit_price

        name_lower = name.lower()
        refund_amount: float = 0.0
        refund_reason: str = "none"

        if name_lower in wrong_items:
            # Wrong item delivered -> full refund
            refund_amount = full_item_value
            refund_reason = "wrong_item_delivered"
            has_wrong_item = True
        else:
            vision_entry = delivered_lookup.get(name_lower, {})
            condition: str = vision_entry.get("condition", "ok").lower()
            qty_visible: int = int(vision_entry.get("qty_visible", expected_qty))

            if damage_detected and condition in ("damaged", "leaking", "tampered"):
                # Damaged / compromised packaging -> full refund
                refund_amount = full_item_value
                refund_reason = f"item_{condition}"
            elif qty_visible < expected_qty:
                # Quantity shortfall -> pro-rated
                missing = expected_qty - qty_visible
                refund_amount = round((missing / expected_qty) * full_item_value, 2)
                refund_reason = f"quantity_shortfall_{missing}_of_{expected_qty}"
            else:
                refund_amount = 0.0
                refund_reason = "none"

        line_items.append(
            {
                "name": name,
                "expected_qty": expected_qty,
                "unit_price": unit_price,
                "full_item_value": round(full_item_value, 2),
                "refund_amount": round(refund_amount, 2),
                "refund_reason": refund_reason,
            }
        )
        item_refund_total += refund_amount

    item_refund_total = round(item_refund_total, 2)

    # Waive delivery fee when any wrong item was delivered
    delivery_fee_waiver: float = settings.DELIVERY_FEE if has_wrong_item else 0.0

    # Waive platform fee when refund covers more than half the order value
    order_total = sum(i["full_item_value"] for i in line_items)
    platform_fee_waiver: float = (
        settings.PLATFORM_FEE if order_total > 0 and (item_refund_total / order_total) >= 0.5 else 0.0
    )

    total_refund = round(item_refund_total + delivery_fee_waiver + platform_fee_waiver, 2)
    grand_order_value = round(order_total + settings.DELIVERY_FEE + settings.PLATFORM_FEE, 2)
    refund_percentage = round((total_refund / grand_order_value * 100) if grand_order_value > 0 else 0.0, 2)

    return {
        "line_items": line_items,
        "item_refund_total": item_refund_total,
        "delivery_fee_waiver": delivery_fee_waiver,
        "platform_fee_waiver": platform_fee_waiver,
        "total_refund": total_refund,
        "refund_percentage": refund_percentage,
    }


# ---------------------------------------------------------------------------
# Final Claim Decision Generator
# ---------------------------------------------------------------------------


def generate_refund_decision(
    order_id: str,
    order_data: dict[str, Any],
    vision_result: dict[str, Any],
    fraud_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Produce the authoritative refund decision document for a claim.

    Decision flow
    -------------
    1. Fraud **HARD_REJECT** -> status ``REJECTED``, zero refund.
    2. Fraud **MANUAL_REVIEW** -> status ``PENDING_REVIEW``, no refund yet.
    3. Fraud **CLEARED** -> status ``APPROVED`` (possibly with $0 refund if
       the vision analysis found no issues), full breakdown included.

    Returns
    -------
    A dict matching::

        {
            "order_id": str,
            "status": "REJECTED" | "PENDING_REVIEW" | "APPROVED",
            "fraud_check": dict,
            "refund_breakdown": dict | None,
            "vision_summary": dict,
            "timestamp": str   // ISO-8601 UTC
        }
    """
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    action = fraud_result.get("action", "CLEARED")

    vision_summary = {
        "bill_merchant": vision_result.get("bill_analysis", {}).get("merchant_name"),
        "bill_total": vision_result.get("bill_analysis", {}).get("grand_total"),
        "bill_order_id": vision_result.get("bill_analysis", {}).get("order_id_on_bill"),
        "food_overall_assessment": vision_result.get("food_analysis", {}).get("overall_assessment"),
        "food_damage_detected": vision_result.get("food_analysis", {}).get("damage_detected", False),
        "food_wrong_items": vision_result.get("food_analysis", {}).get("wrong_items", []),
        "food_is_stock_photo": vision_result.get("food_analysis", {}).get("is_stock_photo", False),
    }

    if action == "HARD_REJECT":
        return {
            "order_id": order_id,
            "status": "REJECTED",
            "fraud_check": fraud_result,
            "refund_breakdown": None,
            "vision_summary": vision_summary,
            "timestamp": now_iso,
        }

    if action == "MANUAL_REVIEW":
        return {
            "order_id": order_id,
            "status": "PENDING_REVIEW",
            "fraud_check": fraud_result,
            "refund_breakdown": None,
            "vision_summary": vision_summary,
            "timestamp": now_iso,
        }

    # Fraud CLEARED -> compute deterministic refund
    order_items: list[dict] = order_data.get("items", [])
    food_analysis: dict = vision_result.get("food_analysis", {})
    bill_analysis: dict = vision_result.get("bill_analysis", {})

    refund_breakdown = calculate_item_refund(order_items, food_analysis, bill_analysis)

    return {
        "order_id": order_id,
        "status": "APPROVED",
        "fraud_check": fraud_result,
        "refund_breakdown": refund_breakdown,
        "vision_summary": vision_summary,
        "timestamp": now_iso,
    }


# ---------------------------------------------------------------------------
# Test & Integration Helpers
# ---------------------------------------------------------------------------


def make_vision_result(
    delivered_items: list[str],
    damage_detected: bool = False,
    wrong_items: list[str] | None = None,
    is_stock_photo: bool = False,
    item_conditions: dict[str, str] | None = None,
    qty_visible: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Helper creating a vision pipeline output structure for testing/mocking."""
    wrong_items = wrong_items or []
    item_conditions = item_conditions or {}
    qty_visible = qty_visible or {}

    deliv_entries: list[dict[str, Any]] = []
    for item_name in delivered_items:
        deliv_entries.append(
            {
                "name": item_name,
                "condition": item_conditions.get(item_name, "ok"),
                "qty_visible": qty_visible.get(item_name, None),
            }
        )

    return {
        "bill_analysis": {},
        "food_analysis": {
            "delivered_items": deliv_entries,
            "damage_detected": damage_detected,
            "wrong_items": wrong_items,
            "is_stock_photo": is_stock_photo,
            "item_conditions": item_conditions,
            "qty_visible": qty_visible,
        },
    }


def calculate_refund(
    order_items: list[dict[str, Any]],
    vision: dict[str, Any],
    fraud_result: dict[str, Any],
    delivery_fee: float = 49.0,
) -> dict[str, Any]:
    """
    Evaluate fraud status and compute refund breakdown for test scenarios.
    """
    action = fraud_result.get("action", "CLEARED")
    if action == "HARD_REJECT":
        return {
            "status": "REJECTED",
            "breakdown": None,
            "total_refund": 0.0,
            "refund_percentage": 0.0,
        }
    if action == "MANUAL_REVIEW":
        return {
            "status": "PENDING_REVIEW",
            "breakdown": None,
            "total_refund": 0.0,
            "refund_percentage": 0.0,
        }

    food = vision.get("food_analysis", vision)
    delivered_items = food.get("delivered_items", [])
    wrong_items = [w.lower() for w in food.get("wrong_items", [])]
    damage_detected = food.get("damage_detected", False)
    item_conditions = food.get("item_conditions", {})
    qty_visible_map = food.get("qty_visible", {})

    delivered_lookup: dict[str, dict] = {}
    for d in delivered_items:
        if isinstance(d, dict) and "name" in d:
            delivered_lookup[d["name"].lower()] = d
        elif isinstance(d, str):
            delivered_lookup[d.lower()] = {
                "name": d,
                "condition": item_conditions.get(d, "ok"),
                "qty_visible": qty_visible_map.get(d, None),
            }

    item_refunds: dict[str, float] = {}
    item_refund_total = 0.0
    has_wrong = bool(wrong_items)
    has_shortfall = False
    order_gross = 0.0

    for order_item in order_items:
        name = order_item.get("name", "")
        expected_qty = int(order_item.get("qty", 1))
        unit_price = float(order_item.get("unit_price", 0.0))
        full_value = expected_qty * unit_price
        order_gross += full_value

        name_lower = name.lower()
        refund_amt = 0.0

        if name_lower in wrong_items:
            refund_amt = full_value
            has_wrong = True
        else:
            vision_entry = delivered_lookup.get(name_lower, {})
            cond = item_conditions.get(name, vision_entry.get("condition", "ok")).lower()
            q_vis = qty_visible_map.get(name, vision_entry.get("qty_visible", expected_qty))
            if q_vis is None:
                q_vis = expected_qty

            if damage_detected and cond in ("damaged", "leaking", "tampered"):
                refund_amt = full_value
            elif q_vis < expected_qty:
                has_shortfall = True
                missing = expected_qty - q_vis
                refund_amt = round((missing / expected_qty) * full_value, 2)
            else:
                refund_amt = 0.0

        item_refunds[name] = round(refund_amt, 2)
        item_refund_total += refund_amt

    item_refund_total = round(item_refund_total, 2)

    # Delivery fee waiver when wrong item or missing item occurred
    if (has_wrong or has_shortfall) and item_refund_total > 0:
        delivery_fee_waiver = delivery_fee
    else:
        delivery_fee_waiver = 0.0

    total_refund = round(item_refund_total + delivery_fee_waiver, 2)
    grand_order_value = order_gross + delivery_fee
    refund_pct = round((total_refund / grand_order_value * 100) if grand_order_value > 0 else 0.0, 1)

    return {
        "status": "CLEARED",
        "total_refund": total_refund,
        "refund_percentage": refund_pct,
        "breakdown": {
            "item_refunds": item_refunds,
            "delivery_fee_waiver": delivery_fee_waiver,
            "platform_fee_waiver": 0.0,
            "item_refund_total": item_refund_total,
        },
    }
