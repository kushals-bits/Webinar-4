"""
Customer Database & Schema Integrity Test Suite
Tests customer table operations, foreign key relations, order histories,
bill ID lookups, and FastAPI customer endpoints.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest
from fastapi.testclient import TestClient

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.database import mock_db  # noqa: E402
from app.main import app  # noqa: E402


client = TestClient(app)


class TestCustomerDatabase:
    """Tests for the mock_db CUSTOMERS and ACTIVE_ORDERS relational store."""

    def test_customer_schema_fields(self):
        """Verify all required schema fields are present in every customer record."""
        customers = mock_db.get_all_customers()
        assert len(customers) >= 5

        required_fields = {
            "customer_id",
            "customer_name",
            "email",
            "phone",
            "account_created_at",
            "trust_tier",
            "trust_score",
            "lifetime_orders_count",
            "dispute_count",
            "refund_rate_percent",
            "comment",
            "order_history",
            "associated_device_fingerprints",
        }

        for cust in customers:
            missing = required_fields - set(cust.keys())
            assert not missing, f"Customer {cust.get('customer_id')} is missing fields: {missing}"
            assert isinstance(cust["customer_id"], str)
            assert isinstance(cust["customer_name"], str)
            assert isinstance(cust["order_history"], list)
            assert isinstance(cust["comment"], str)

    def test_order_history_item_schema(self):
        """Verify the nested order_history items have required relational fields."""
        cust = mock_db.get_customer("CUST-101")
        assert cust is not None
        assert len(cust["order_history"]) >= 2

        order_history_fields = {
            "order_id",
            "bill_id",
            "order_date",
            "restaurant",
            "grand_total",
            "status",
            "dispute_filed",
            "dispute_status",
            "comment",
        }

        for hist in cust["order_history"]:
            missing = order_history_fields - set(hist.keys())
            assert not missing, f"Order history entry missing fields: {missing}"
            assert hist["bill_id"].startswith("BILL-")
            assert "T" in hist["order_date"]  # ISO timestamp

    def test_active_orders_foreign_key_link(self):
        """Verify ACTIVE_ORDERS correctly link to customer_id as FK and have bill_id, order_date, comment."""
        orders = mock_db.get_all_orders()
        assert len(orders) >= 4

        order_fields = {
            "order_id",
            "customer_id",
            "customer_name",
            "bill_id",
            "order_date",
            "restaurant",
            "items",
            "delivery_fee",
            "platform_fee",
            "grand_total",
            "status",
            "comment",
            "dispute_status",
        }

        for order in orders:
            missing = order_fields - set(order.keys())
            assert not missing, f"Order {order.get('order_id')} is missing fields: {missing}"

            # Verify foreign key customer exists in CUSTOMERS table
            cust_id = order["customer_id"]
            cust = mock_db.get_customer(cust_id)
            assert cust is not None, f"Foreign key customer_id {cust_id} not found in CUSTOMERS table"
            assert cust["customer_name"] == order["customer_name"]

    def test_lookup_order_by_bill_id(self):
        """Lookup order by physical bill receipt ID."""
        order = mock_db.get_order_by_bill_id("BILL-2026-9041")
        assert order is not None
        assert order["order_id"] == "ORD-2026-904"
        assert order["customer_id"] == "CUST-442"
        assert order["customer_name"] == "Vikram Malhotra"
        assert "BILL-2026-9041" in order["comment"]

        # Nonexistent bill
        missing = mock_db.get_order_by_bill_id("BILL-DOES-NOT-EXIST")
        assert missing is None

    def test_get_customer_orders_fk(self):
        """Retrieve all active orders by customer foreign key."""
        orders_c101 = mock_db.get_customer_orders("CUST-101")
        assert len(orders_c101) >= 1
        assert orders_c101[0]["customer_id"] == "CUST-101"
        assert orders_c101[0]["bill_id"] == "BILL-2026-001"


class TestCustomerFastAPIRoutes:
    """Tests for customer and bill REST API endpoints in FastAPI."""

    def test_get_all_customers_endpoint(self):
        res = client.get("/api/v1/customers")
        assert res.status_code == 200
        data = res.json()
        assert "customers" in data
        assert data["total"] >= 5

    def test_get_single_customer_endpoint(self):
        res = client.get("/api/v1/customers/CUST-442")
        assert res.status_code == 200
        cust = res.json()
        assert cust["customer_id"] == "CUST-442"
        assert cust["customer_name"] == "Vikram Malhotra"
        assert cust["trust_tier"] == "VIP_GOLD"
        assert cust["trust_score"] == pytest.approx(0.94)

    def test_get_nonexistent_customer_returns_404(self):
        res = client.get("/api/v1/customers/CUST-NONEXISTENT")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

    def test_get_customer_orders_endpoint(self):
        res = client.get("/api/v1/customers/CUST-442/orders")
        assert res.status_code == 200
        data = res.json()
        assert data["customer_id"] == "CUST-442"
        assert len(data["orders"]) >= 1
        assert data["orders"][0]["order_id"] == "ORD-2026-904"

    def test_get_bill_order_endpoint(self):
        res = client.get("/api/v1/bills/BILL-2026-9041")
        assert res.status_code == 200
        data = res.json()
        assert data["bill_id"] == "BILL-2026-9041"
        assert data["order"]["customer_id"] == "CUST-442"

    def test_get_nonexistent_bill_returns_404(self):
        res = client.get("/api/v1/bills/BILL-FAKE-9999")
        assert res.status_code == 404
