"""
In-memory mock database for the Zomato Replica Refund System.

Provides:
  - Customers table (Primary key customer_id, trust tiers, order history, comments)
  - Active orders store (Foreign key customer_id, bill_id, order_date, comments)
  - Historical fraud perceptual hashes blacklist
  - Suspicious-activity log
  - Admin review queue
  - Device-fingerprint multi-account registry
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 1. Customers Table
# ---------------------------------------------------------------------------

CUSTOMERS: Dict[str, Dict[str, Any]] = {
    "CUST-101": {
        "customer_id": "CUST-101",
        "customer_name": "Rahul Sharma",
        "email": "rahul.sharma@example.com",
        "phone": "+91-9810123456",
        "account_created_at": "2024-03-15T10:00:00Z",
        "trust_tier": "VIP_GOLD",
        "trust_score": 0.92,
        "lifetime_orders_count": 18,
        "dispute_count": 1,
        "refund_rate_percent": 5.5,
        "comment": "Verified Gold Member; high order volume; prompt delivery receiver; 1 past dispute resolved amicably.",
        "order_history": [
            {
                "order_id": "ORD-2026-001",
                "bill_id": "BILL-2026-001",
                "order_date": "2026-11-09T17:45:00+05:30",
                "restaurant": "Punjabi Dhaba",
                "grand_total": 1079.0,
                "status": "DELIVERED",
                "dispute_filed": True,
                "dispute_status": "OPEN",
                "comment": "Customer reported 2 Garlic Naan missing from the bag.",
            },
            {
                "order_id": "ORD-2026-PAST-01",
                "bill_id": "BILL-2026-8812",
                "order_date": "2026-10-28T20:10:00+05:30",
                "restaurant": "Chai Point",
                "grand_total": 340.0,
                "status": "DELIVERED",
                "dispute_filed": False,
                "dispute_status": "NONE",
                "comment": "Delivered on time, 5-star rating left.",
            },
            {
                "order_id": "ORD-2026-PAST-02",
                "bill_id": "BILL-2026-7731",
                "order_date": "2026-10-12T13:15:00+05:30",
                "restaurant": "Biryani By Kilo",
                "grand_total": 890.0,
                "status": "DELIVERED",
                "dispute_filed": False,
                "dispute_status": "NONE",
                "comment": "Order delivered intact.",
            },
        ],
        "associated_device_fingerprints": ["a9f4c3d2e1b0"],
    },
    "CUST-202": {
        "customer_id": "CUST-202",
        "customer_name": "Priya Patel",
        "email": "priya.patel@example.com",
        "phone": "+91-9820234567",
        "account_created_at": "2025-01-10T14:30:00Z",
        "trust_tier": "REGULAR_SILVER",
        "trust_score": 0.88,
        "lifetime_orders_count": 7,
        "dispute_count": 0,
        "refund_rate_percent": 0.0,
        "comment": "Regular Silver account; zero previous disputes; fast digital payment method.",
        "order_history": [
            {
                "order_id": "ORD-2026-002",
                "bill_id": "BILL-2026-002",
                "order_date": "2026-11-09T19:00:00+05:30",
                "restaurant": "Burger Barn",
                "grand_total": 468.0,
                "status": "DELIVERED",
                "dispute_filed": True,
                "dispute_status": "OPEN",
                "comment": "Customer reported crushed burger box and spillage during rain delivery.",
            },
            {
                "order_id": "ORD-2026-PAST-03",
                "bill_id": "BILL-2026-6641",
                "order_date": "2026-10-22T19:45:00+05:30",
                "restaurant": "Subway",
                "grand_total": 420.0,
                "status": "DELIVERED",
                "dispute_filed": False,
                "dispute_status": "NONE",
                "comment": "Delivered successfully.",
            },
        ],
        "associated_device_fingerprints": ["b8e3d2c1f0a9"],
    },
    "CUST-303": {
        "customer_id": "CUST-303",
        "customer_name": "Amit Verma",
        "email": "amit.verma@example.com",
        "phone": "+91-9830345678",
        "account_created_at": "2024-08-20T11:15:00Z",
        "trust_tier": "REGULAR_SILVER",
        "trust_score": 0.85,
        "lifetime_orders_count": 24,
        "dispute_count": 2,
        "refund_rate_percent": 8.3,
        "comment": "Frequent weekend gourmet ordering; 2 past minor packaging spillage refunds.",
        "order_history": [
            {
                "order_id": "ORD-2026-003",
                "bill_id": "BILL-2026-003",
                "order_date": "2026-11-09T19:30:00+05:30",
                "restaurant": "Sushi Palace",
                "grand_total": 1320.0,
                "status": "DELIVERED",
                "dispute_filed": False,
                "dispute_status": "NONE",
                "comment": "Order delivered on time without incident.",
            },
        ],
        "associated_device_fingerprints": ["c7d2e1b0a9f8"],
    },
    "CUST-442": {
        "customer_id": "CUST-442",
        "customer_name": "Vikram Malhotra",
        "email": "vikram.m@example.com",
        "phone": "+91-9840456789",
        "account_created_at": "2024-11-01T09:00:00Z",
        "trust_tier": "VIP_GOLD",
        "trust_score": 0.94,
        "lifetime_orders_count": 31,
        "dispute_count": 1,
        "refund_rate_percent": 3.2,
        "comment": "High-LTV Gold member; submitted active wrong food delivery dispute (billed for Butter Chicken & Naan, received Masala Dosa).",
        "order_history": [
            {
                "order_id": "ORD-2026-904",
                "bill_id": "BILL-2026-9041",
                "order_date": "2026-11-09T18:00:00+05:30",
                "restaurant": "Royal Tandoor Kitchen",
                "grand_total": 1129.0,
                "status": "DELIVERED",
                "dispute_filed": True,
                "dispute_status": "UNDER_REVIEW",
                "comment": "Customer reported rider swapped bags: received Masala Dosa instead of Royal Tandoor items.",
            },
        ],
        "associated_device_fingerprints": ["d6c1b0a9f8e7"],
    },
    "CUST-999": {
        "customer_id": "CUST-999",
        "customer_name": "Karan Singh (Flagged Sybil)",
        "email": "karan.999@dispostable.com",
        "phone": "+91-9899999999",
        "account_created_at": "2026-01-14T08:00:00Z",
        "trust_tier": "FLAGGED_HIGH_RISK",
        "trust_score": 0.08,
        "lifetime_orders_count": 3,
        "dispute_count": 3,
        "refund_rate_percent": 100.0,
        "comment": "High-risk serial fraudster; 100% dispute rate; submitted recycled duplicate hash (pHash a1b2c3d4e5f6a7b8) across multiple phone accounts.",
        "order_history": [
            {
                "order_id": "ORD-2025-FRAUD-01",
                "bill_id": "BILL-2025-FRAUD-01",
                "order_date": "2026-01-15T09:15:00Z",
                "restaurant": "Delhi Darbar",
                "grand_total": 750.0,
                "status": "REJECTED",
                "dispute_filed": True,
                "dispute_status": "REJECTED_FRAUD",
                "comment": "Submitted photo previously used under CUST-888. Blacklisted.",
            },
        ],
        "associated_device_fingerprints": ["emu_device_8831", "shared_mac_9912"],
    },
    "CUST-888": {
        "customer_id": "CUST-888",
        "customer_name": "Rohit Mehra (Flagged Sybil)",
        "email": "rohit.m888@dispostable.com",
        "phone": "+91-9888888888",
        "account_created_at": "2026-02-18T12:00:00Z",
        "trust_tier": "FLAGGED_HIGH_RISK",
        "trust_score": 0.12,
        "lifetime_orders_count": 2,
        "dispute_count": 2,
        "refund_rate_percent": 100.0,
        "comment": "Flagged account; colluded with CUST-999 sharing identical emulator hardware footprint; pHash deadbeefcafebabe.",
        "order_history": [
            {
                "order_id": "ORD-2025-FRAUD-02",
                "bill_id": "BILL-2025-FRAUD-02",
                "order_date": "2026-02-20T13:45:00Z",
                "restaurant": "Pizza Express",
                "grand_total": 680.0,
                "status": "REJECTED",
                "dispute_filed": True,
                "dispute_status": "REJECTED_FRAUD",
                "comment": "Submitted Pinterest stock image. Blacklisted.",
            },
        ],
        "associated_device_fingerprints": ["emu_device_8831"],
    },
}

_CUSTOMERS = CUSTOMERS


# ---------------------------------------------------------------------------
# 2. Active Orders Store (With Customer ID FK, Bill ID, Order Date & Comments)
# ---------------------------------------------------------------------------

ACTIVE_ORDERS: Dict[str, Dict[str, Any]] = {
    "ORD-2026-001": {
        "order_id": "ORD-2026-001",
        "customer_id": "CUST-101",
        "customer_name": "Rahul Sharma",
        "bill_id": "BILL-2026-001",
        "order_date": "2026-11-09T17:45:00+05:30",
        "delivery_time": "2026-11-09T18:30:00+05:30",
        "restaurant": "Punjabi Dhaba",
        "restaurant_id": "REST-55",
        "items": [
            {"name": "Butter Chicken", "qty": 2, "unit_price": 320.0, "total": 640.0},
            {"name": "Garlic Naan", "qty": 4, "unit_price": 55.0, "total": 220.0},
            {"name": "Dal Makhani", "qty": 1, "unit_price": 220.0, "total": 220.0},
        ],
        "delivery_fee": 49.0,
        "platform_fee": 5.0,
        "grand_total": 1079.0,
        "status": "DELIVERED",
        "comment": "Contactless delivery requested; leave bag at apartment door.",
        "dispute_status": "OPEN",
    },
    "ORD-2026-002": {
        "order_id": "ORD-2026-002",
        "customer_id": "CUST-202",
        "customer_name": "Priya Patel",
        "bill_id": "BILL-2026-002",
        "order_date": "2026-11-09T19:00:00+05:30",
        "delivery_time": "2026-11-09T19:45:00+05:30",
        "restaurant": "Burger Barn",
        "restaurant_id": "REST-12",
        "items": [
            {"name": "Classic Burger", "qty": 1, "unit_price": 199.0, "total": 199.0},
            {"name": "Cheese Fries", "qty": 1, "unit_price": 149.0, "total": 149.0},
            {"name": "Coke 500ml", "qty": 2, "unit_price": 60.0, "total": 120.0},
        ],
        "delivery_fee": 49.0,
        "platform_fee": 5.0,
        "grand_total": 468.0,
        "status": "DELIVERED",
        "comment": "Urgent dinner order; rider noted heavy rainfall during transit.",
        "dispute_status": "OPEN",
    },
    "ORD-2026-003": {
        "order_id": "ORD-2026-003",
        "customer_id": "CUST-303",
        "customer_name": "Amit Verma",
        "bill_id": "BILL-2026-003",
        "order_date": "2026-11-09T19:30:00+05:30",
        "delivery_time": "2026-11-09T20:15:00+05:30",
        "restaurant": "Sushi Palace",
        "restaurant_id": "REST-88",
        "items": [
            {"name": "Dragon Roll", "qty": 2, "unit_price": 450.0, "total": 900.0},
            {"name": "Miso Soup", "qty": 2, "unit_price": 120.0, "total": 240.0},
            {"name": "Edamame", "qty": 1, "unit_price": 180.0, "total": 180.0},
        ],
        "delivery_fee": 49.0,
        "platform_fee": 5.0,
        "grand_total": 1320.0,
        "status": "DELIVERED",
        "comment": "Please include extra soy sauce and chopstick sets.",
        "dispute_status": "NONE",
    },
    "ORD-2026-904": {
        "order_id": "ORD-2026-904",
        "customer_id": "CUST-442",
        "customer_name": "Vikram Malhotra",
        "bill_id": "BILL-2026-9041",
        "order_date": "2026-11-09T18:00:00+05:30",
        "delivery_time": "2026-11-09T18:35:00+05:30",
        "restaurant": "Royal Tandoor Kitchen",
        "restaurant_id": "REST-91",
        "items": [
            {"name": "Butter Chicken", "qty": 2, "unit_price": 320.0, "total": 640.0},
            {"name": "Garlic Naan", "qty": 4, "unit_price": 55.0, "total": 220.0},
            {"name": "Dal Makhani", "qty": 1, "unit_price": 220.0, "total": 220.0},
        ],
        "delivery_fee": 49.0,
        "platform_fee": 5.0,
        "grand_total": 1129.0,
        "status": "DELIVERED",
        "comment": "Wrong food delivered dispute: Customer uploaded paper bill BILL-2026-9041 and photo of Masala Dosa.",
        "dispute_status": "UNDER_REVIEW",
    },
}

_ORDERS = ACTIVE_ORDERS


# ---------------------------------------------------------------------------
# 3. Fraud Blacklist & Audit Logs
# ---------------------------------------------------------------------------

HISTORICAL_REFUND_HASHES: Dict[str, Dict[str, Any]] = {
    "a1b2c3d4e5f6a7b8": {
        "customer_id": "CUST-999",
        "order_id": "ORD-2025-FRAUD-01",
        "bill_id": "BILL-2025-FRAUD-01",
        "flagged_at": "2026-01-15T10:00:00Z",
        "reason": "Recycled duplicate image hash",
    },
    "deadbeefcafebabe": {
        "customer_id": "CUST-888",
        "order_id": "ORD-2025-FRAUD-02",
        "bill_id": "BILL-2025-FRAUD-02",
        "flagged_at": "2026-02-20T14:30:00Z",
        "reason": "Pinterest stock image hash",
    },
}

SUSPICIOUS_ACTIVITY_LOG: List[Dict[str, Any]] = []
_SUSPICIOUS_LOG = SUSPICIOUS_ACTIVITY_LOG

ADMIN_REVIEW_QUEUE: List[Dict[str, Any]] = []
_ADMIN_QUEUE = ADMIN_REVIEW_QUEUE

# Maps device_fingerprint -> list of customer_ids
DEVICE_FINGERPRINT_REGISTRY: Dict[str, List[str]] = {
    "emu_device_8831": ["CUST-999", "CUST-888"],
    "a9f4c3d2e1b0": ["CUST-101"],
    "b8e3d2c1f0a9": ["CUST-202"],
    "c7d2e1b0a9f8": ["CUST-303"],
    "d6c1b0a9f8e7": ["CUST-442"],
}
_DEVICE_REGISTRY = DEVICE_FINGERPRINT_REGISTRY


# ---------------------------------------------------------------------------
# 4. Public Database Access API
# ---------------------------------------------------------------------------

# --- Customer Operations ---

def get_customer(customer_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve customer profile by Primary Key customer_id."""
    cust = CUSTOMERS.get(customer_id)
    return deepcopy(cust) if cust else None


def get_all_customers() -> List[Dict[str, Any]]:
    """Return all customer profiles in the database."""
    return [deepcopy(c) for c in CUSTOMERS.values()]


def add_customer(customer: Dict[str, Any]) -> None:
    """Insert or update a customer record."""
    CUSTOMERS[customer["customer_id"]] = deepcopy(customer)


def get_customer_orders(customer_id: str) -> List[Dict[str, Any]]:
    """Return all active orders associated with a foreign key customer_id."""
    return [deepcopy(o) for o in ACTIVE_ORDERS.values() if o.get("customer_id") == customer_id]


# --- Order & Bill Operations ---

def get_order(order_id: str) -> Optional[Dict[str, Any]]:
    """Return a deep copy of the order dict, or None if not found."""
    order = ACTIVE_ORDERS.get(order_id)
    return deepcopy(order) if order else None


def get_all_orders() -> List[Dict[str, Any]]:
    """Return all active orders in the database."""
    return [deepcopy(o) for o in ACTIVE_ORDERS.values()]


def get_order_by_bill_id(bill_id: str) -> Optional[Dict[str, Any]]:
    """Look up an order by its unique physical bill/receipt ID."""
    for order in ACTIVE_ORDERS.values():
        if order.get("bill_id") == bill_id:
            return deepcopy(order)
    return None


def add_order(order: Dict[str, Any]) -> None:
    """Store an order in the active orders table."""
    ACTIVE_ORDERS[order["order_id"]] = deepcopy(order)


# --- Audit & Security Registry Operations ---

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
    """Reset all mutable in-memory state (useful in test suites)."""
    SUSPICIOUS_ACTIVITY_LOG.clear()
    ADMIN_REVIEW_QUEUE.clear()
