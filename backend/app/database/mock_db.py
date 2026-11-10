"""
In-memory mock database for the Zomato Replica Refund System.

Provides:
  - Active orders store
  - Historical fraud perceptual hashes blacklist
  - Suspicious-activity log
  - Admin review queue
  - Device-fingerprint multi-account registry
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

ACTIVE_ORDERS: Dict[str, Dict[str, Any]] = {
    "ORD-2026-001": {
        "order_id": "ORD-2026-001",
        "customer_id": "CUST-101",
        "restaurant": "Punjabi Dhaba",
        "restaurant_id": "REST-55",
        "items": [
            {"name": "Butter Chicken", "qty": 2, "unit_price": 320.0},
            {"name": "Garlic Naan", "qty": 4, "unit_price": 55.0},
            {"name": "Dal Makhani", "qty": 1, "unit_price": 220.0},
        ],
        "delivery_fee": 49.0,
        "grand_total": 1079.0,
        "delivery_time": "2026-11-09T18:30:00+05:30",
        "status": "DELIVERED",
    },
    "ORD-2026-002": {
        "order_id": "ORD-2026-002",
        "customer_id": "CUST-202",
        "restaurant": "Burger Barn",
        "restaurant_id": "REST-12",
        "items": [
            {"name": "Classic Burger", "qty": 1, "unit_price": 199.0},
            {"name": "Cheese Fries", "qty": 1, "unit_price": 149.0},
            {"name": "Coke 500ml", "qty": 2, "unit_price": 60.0},
        ],
        "delivery_fee": 49.0,
        "grand_total": 468.0,
        "delivery_time": "2026-11-09T19:45:00+05:30",
        "status": "DELIVERED",
    },
    "ORD-2026-003": {
        "order_id": "ORD-2026-003",
        "customer_id": "CUST-303",
        "restaurant": "Sushi Palace",
        "restaurant_id": "REST-88",
        "items": [
            {"name": "Dragon Roll", "qty": 2, "unit_price": 450.0},
            {"name": "Miso Soup", "qty": 2, "unit_price": 120.0},
            {"name": "Edamame", "qty": 1, "unit_price": 180.0},
        ],
        "delivery_fee": 49.0,
        "grand_total": 1320.0,
        "delivery_time": "2026-11-09T20:15:00+05:30",
        "status": "DELIVERED",
    },
}

_ORDERS = ACTIVE_ORDERS

HISTORICAL_REFUND_HASHES: Dict[str, Dict[str, Any]] = {
    "a1b2c3d4e5f6a7b8": {
        "customer_id": "CUST-999",
        "order_id": "ORD-2025-FRAUD-01",
        "flagged_at": "2026-01-15T10:00:00Z",
    },
    "deadbeefcafebabe": {
        "customer_id": "CUST-888",
        "order_id": "ORD-2025-FRAUD-02",
        "flagged_at": "2026-02-20T14:30:00Z",
    },
}

SUSPICIOUS_ACTIVITY_LOG: List[Dict[str, Any]] = []
_SUSPICIOUS_LOG = SUSPICIOUS_ACTIVITY_LOG

ADMIN_REVIEW_QUEUE: List[Dict[str, Any]] = []
_ADMIN_QUEUE = ADMIN_REVIEW_QUEUE

# Maps device_fingerprint -> list of customer_ids
DEVICE_FINGERPRINT_REGISTRY: Dict[str, List[str]] = {}
_DEVICE_REGISTRY = DEVICE_FINGERPRINT_REGISTRY


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_order(order_id: str) -> Optional[Dict[str, Any]]:
    """Return a deep copy of the order dict, or None if not found."""
    order = ACTIVE_ORDERS.get(order_id)
    return deepcopy(order) if order else None


def add_order(order: Dict[str, Any]) -> None:
    ACTIVE_ORDERS[order["order_id"]] = deepcopy(order)


def get_suspicious_log() -> List[Dict[str, Any]]:
    return list(SUSPICIOUS_ACTIVITY_LOG)


def add_to_suspicious_log(entry: Dict[str, Any]) -> None:
    SUSPICIOUS_ACTIVITY_LOG.append(deepcopy(entry))


def get_admin_queue() -> List[Dict[str, Any]]:
    return list(ADMIN_REVIEW_QUEUE)


def add_to_admin_queue(entry: Dict[str, Any]) -> None:
    ADMIN_REVIEW_QUEUE.append(deepcopy(entry))


def register_hash(hash_str: str, metadata: Dict[str, Any]) -> None:
    """Store a hash in the known fraud blacklist."""
    HISTORICAL_REFUND_HASHES[hash_str] = deepcopy(metadata)


def register_device_fingerprint(fingerprint: str, customer_id: str) -> None:
    """Associate a device fingerprint with a customer ID."""
    if fingerprint not in DEVICE_FINGERPRINT_REGISTRY:
        DEVICE_FINGERPRINT_REGISTRY[fingerprint] = []
    if customer_id not in DEVICE_FINGERPRINT_REGISTRY[fingerprint]:
        DEVICE_FINGERPRINT_REGISTRY[fingerprint].append(customer_id)


def check_device_multi_account(fingerprint: str, customer_id: str) -> Optional[str]:
    """
    Return the other customer_id that shares this fingerprint, or None.
    If multiple accounts share the fingerprint the first other one is returned.
    """
    others = [cid for cid in DEVICE_FINGERPRINT_REGISTRY.get(fingerprint, []) if cid != customer_id]
    return others[0] if others else None


def reset_all() -> None:
    """Reset all mutable in-memory state (useful in tests)."""
    SUSPICIOUS_ACTIVITY_LOG.clear()
    ADMIN_REVIEW_QUEUE.clear()
    DEVICE_FINGERPRINT_REGISTRY.clear()
