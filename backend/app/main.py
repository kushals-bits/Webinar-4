"""
Zomato Replica Refund System — FastAPI Application Entry-Point

Endpoints
---------
POST  /api/v1/claims/submit               Submit a refund claim with bill & food images.
GET   /api/v1/customers                   Retrieve all customer profiles.
GET   /api/v1/customers/{customer_id}     Retrieve a customer profile by Primary Key.
GET   /api/v1/customers/{customer_id}/orders Retrieve orders associated with a customer.
GET   /api/v1/bills/{bill_id}             Look up an order by printed bill ID.
GET   /api/v1/admin/queue                 Retrieve the manual-review queue.
GET   /api/v1/admin/suspicious-log        Retrieve the suspicious-activity log.
GET   /api/v1/orders/{order_id}           Retrieve order details.
GET   /health                             Service health-check.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.refund_engine import generate_refund_decision
from app.core.vision_pipeline import run_vision_pipeline
from app.database import mock_db
from app.security.fraud_detector import enforce_fraud_decision, run_fraud_detection

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Zomato Replica Refund System — Backend started ===")
    logger.info("Seeded %d orders, %d fraud hashes.", len(mock_db.ACTIVE_ORDERS), len(mock_db.HISTORICAL_REFUND_HASHES))
    yield
    logger.info("=== Backend shutting down ===")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Zomato Replica Refund System",
    description=(
        "AI-powered food-delivery refund adjudication platform featuring "
        "multimodal Gemini vision analysis, perceptual-hash fraud detection, "
        "EXIF metadata forensics, and deterministic refund computation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — permissive for development; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class ClaimRequest(BaseModel):
    """Body model for the claim-submission endpoint (documented only; actual
    endpoint uses Form + File parameters for multipart/form-data)."""

    order_id: str
    customer_id: str
    issue_description: str


class RefundLineItem(BaseModel):
    name: str
    expected_qty: int
    unit_price: float
    full_item_value: float
    refund_amount: float
    refund_reason: str


class RefundBreakdown(BaseModel):
    line_items: list[RefundLineItem]
    item_refund_total: float
    delivery_fee_waiver: float
    platform_fee_waiver: float
    total_refund: float
    refund_percentage: float


class FraudCheck(BaseModel):
    action: str
    message: str
    fraud_risk_score: float


class VisionSummary(BaseModel):
    bill_merchant: str | None
    bill_total: float | None
    bill_order_id: str | None
    food_overall_assessment: str | None
    food_damage_detected: bool
    food_wrong_items: list[str]
    food_is_stock_photo: bool


class RefundResponse(BaseModel):
    order_id: str
    status: str
    fraud_check: dict[str, Any]
    refund_breakdown: dict[str, Any] | None
    vision_summary: dict[str, Any]
    timestamp: str


class AdminQueueItem(BaseModel):
    order_id: str
    customer_id: str
    fraud_risk_score: float
    fraud_flags: list[str]
    phash: str
    action: str


class CreateCustomerRequest(BaseModel):
    customer_name: str
    customer_id: str | None = None
    email: str | None = None
    phone: str | None = None
    trust_tier: str = "REGULAR_SILVER"
    trust_score: float = 0.85
    comment: str = "Registered via API"
    device_fingerprint: str | None = None


class OrderItemInput(BaseModel):
    name: str
    qty: int = 1
    unit_price: float


class CreateOrderRequest(BaseModel):
    customer_id: str
    restaurant: str
    items: list[OrderItemInput]
    order_id: str | None = None
    bill_id: str | None = None
    delivery_fee: float = 49.0
    platform_fee: float = 5.0
    comment: str = ""


class AdminQueueResponse(BaseModel):
    queue: list[dict[str, Any]]
    total: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["System"])
async def health_check() -> dict[str, str]:
    """Service liveness probe."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/v1/customers", tags=["Customers"])
async def get_customers() -> dict[str, Any]:
    """Retrieve all customer profiles."""
    customers = mock_db.get_all_customers()
    return {"customers": customers, "total": len(customers)}


@app.post("/api/v1/customers", tags=["Customers"], status_code=201)
async def create_customer(payload: CreateCustomerRequest) -> dict[str, Any]:
    """Register and store a new customer profile in the database."""
    cust = mock_db.save_new_customer(
        customer_name=payload.customer_name,
        customer_id=payload.customer_id,
        email=payload.email,
        phone=payload.phone,
        trust_tier=payload.trust_tier,
        trust_score=payload.trust_score,
        comment=payload.comment,
        device_fingerprint=payload.device_fingerprint,
    )
    return {"message": "Customer created successfully", "customer": cust}


@app.get("/api/v1/customers/{customer_id}", tags=["Customers"])
async def get_customer(customer_id: str) -> dict[str, Any]:
    """Retrieve a specific customer profile by Primary Key."""
    cust = mock_db.get_customer(customer_id)
    if cust is None:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found.")
    return cust


@app.get("/api/v1/customers/{customer_id}/orders", tags=["Customers"])
async def get_customer_orders(customer_id: str) -> dict[str, Any]:
    """Retrieve all active orders associated with a customer (FK relationship)."""
    cust = mock_db.get_customer(customer_id)
    if cust is None:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found.")
    orders = mock_db.get_customer_orders(customer_id)
    return {"customer_id": customer_id, "orders": orders, "total": len(orders)}


@app.get("/api/v1/bills/{bill_id}", tags=["Bills"])
async def get_bill_order(bill_id: str) -> dict[str, Any]:
    """Look up an order by its unique physical bill/receipt ID."""
    order = mock_db.get_order_by_bill_id(bill_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Bill '{bill_id}' not found.")
    return {"bill_id": bill_id, "order": order}


@app.get("/api/v1/orders/{order_id}", tags=["Orders"])
async def get_order(order_id: str) -> dict[str, Any]:
    """Retrieve full order details by order ID."""
    order = mock_db.get_order(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order '{order_id}' not found.")
    return {"order_id": order_id, **order}


@app.post("/api/v1/orders", tags=["Orders"], status_code=201)
async def create_order(payload: CreateOrderRequest) -> dict[str, Any]:
    """Store a new order in the database and automatically synchronize the customer's history."""
    cust = mock_db.get_customer(payload.customer_id)
    if not cust:
        raise HTTPException(
            status_code=404,
            detail=f"Customer '{payload.customer_id}' not found. Please create customer first."
        )

    order = mock_db.create_new_order(
        customer_id=payload.customer_id,
        restaurant=payload.restaurant,
        items=[it.model_dump() for it in payload.items],
        order_id=payload.order_id,
        bill_id=payload.bill_id,
        delivery_fee=payload.delivery_fee,
        platform_fee=payload.platform_fee,
        comment=payload.comment,
    )
    return {"message": "Order created successfully", "order": order}


@app.get("/api/v1/admin/queue", response_model=AdminQueueResponse, tags=["Admin"])
async def get_admin_queue() -> AdminQueueResponse:
    """Return all claims currently awaiting manual review."""
    queue = mock_db.get_admin_queue()
    return AdminQueueResponse(queue=queue, total=len(queue))


@app.get("/api/v1/admin/suspicious-log", tags=["Admin"])
async def get_suspicious_log() -> dict[str, Any]:
    """Return the full suspicious-activity log."""
    log = mock_db.get_suspicious_log()
    return {"log": log, "total": len(log)}


@app.post("/api/v1/claims/submit", response_model=RefundResponse, tags=["Claims"])
async def submit_claim(
    order_id: str = Form(..., description="The order ID associated with the claim."),
    customer_id: str = Form(..., description="The customer's unique identifier."),
    issue_description: str = Form(..., description="Free-text description of the issue."),
    bill_image: UploadFile = File(..., description="Photograph or scan of the delivery receipt / bill."),
    food_image: UploadFile = File(..., description="Photograph of the delivered food items."),
) -> RefundResponse:
    """
    Submit a refund claim.

    Processing pipeline
    -------------------
    1. Validate order exists in the database.
    2. Read both uploaded images into memory.
    3. Run fraud detection on each image independently.
    4. Merge fraud results (take the higher risk score).
    5. Enforce a tiered fraud decision (HARD_REJECT / MANUAL_REVIEW / CLEARED).
    6. If not hard-rejected, run the Gemini vision pipeline.
    7. Generate the final refund decision document and return it.
    """
    # ------------------------------------------------------------------ #
    # 1. Validate order
    # ------------------------------------------------------------------ #
    order_data = mock_db.get_order(order_id)
    if order_data is None:
        raise HTTPException(status_code=404, detail=f"Order '{order_id}' not found.")

    logger.info("Claim submitted | order=%s customer=%s", order_id, customer_id)

    # ------------------------------------------------------------------ #
    # 2. Read image bytes
    # ------------------------------------------------------------------ #
    try:
        bill_bytes = await bill_image.read()
        food_bytes = await food_image.read()
    except Exception as exc:
        logger.error("Image read failed: %s", exc)
        raise HTTPException(status_code=400, detail="Failed to read uploaded images.") from exc

    if not bill_bytes or not food_bytes:
        raise HTTPException(status_code=400, detail="Both bill_image and food_image must be non-empty.")

    # ------------------------------------------------------------------ #
    # 3. Fraud detection — bill image
    # ------------------------------------------------------------------ #
    try:
        bill_fraud = run_fraud_detection(
            image_bytes=bill_bytes,
            order_id=order_id,
            customer_id=customer_id,
            image_type="bill",
        )
    except Exception as exc:
        logger.error("Fraud detection failed for bill image: %s", exc)
        bill_fraud = {
            "fraud_risk_score": 0.0,
            "fraud_flags": ["FRAUD_DETECTION_ERROR"],
            "phash": "",
            "action": "CLEARED",
        }

    # ------------------------------------------------------------------ #
    # 4. Fraud detection — food image
    # ------------------------------------------------------------------ #
    try:
        food_fraud = run_fraud_detection(
            image_bytes=food_bytes,
            order_id=order_id,
            customer_id=customer_id,
            image_type="food",
        )
    except Exception as exc:
        logger.error("Fraud detection failed for food image: %s", exc)
        food_fraud = {
            "fraud_risk_score": 0.0,
            "fraud_flags": ["FRAUD_DETECTION_ERROR"],
            "phash": "",
            "action": "CLEARED",
        }

    # ------------------------------------------------------------------ #
    # 5. Merge fraud results — take worst-case score
    # ------------------------------------------------------------------ #
    merged_score = max(
        bill_fraud.get("fraud_risk_score", 0.0),
        food_fraud.get("fraud_risk_score", 0.0),
    )
    merged_flags = list(
        set(bill_fraud.get("fraud_flags", []) + food_fraud.get("fraud_flags", []))
    )
    merged_phash = food_fraud.get("phash", "") or bill_fraud.get("phash", "")

    merged_fraud_result = {
        **food_fraud,  # keep all forensic detail from food image
        "fraud_risk_score": round(min(merged_score, 1.0), 4),
        "fraud_flags": merged_flags,
        "phash": merged_phash,
        "bill_fraud_score": bill_fraud.get("fraud_risk_score", 0.0),
        "food_fraud_score": food_fraud.get("fraud_risk_score", 0.0),
    }

    # ------------------------------------------------------------------ #
    # 6. Enforce tiered fraud decision
    # ------------------------------------------------------------------ #
    fraud_decision = enforce_fraud_decision(merged_fraud_result, order_id, customer_id)

    # ------------------------------------------------------------------ #
    # 7. Vision pipeline (skipped on hard reject)
    # ------------------------------------------------------------------ #
    if fraud_decision["action"] == "HARD_REJECT":
        vision_result: dict[str, Any] = {
            "bill_analysis": {"error": "skipped_due_to_fraud_rejection"},
            "food_analysis": {"error": "skipped_due_to_fraud_rejection"},
        }
    else:
        try:
            vision_result = run_vision_pipeline(
                bill_bytes=bill_bytes,
                food_bytes=food_bytes,
                expected_items=order_data.get("items", []),
            )
        except Exception as exc:
            logger.error("Vision pipeline failed: %s", exc)
            vision_result = {
                "bill_analysis": {"error": "vision_pipeline_error", "detail": str(exc)},
                "food_analysis": {
                    "delivered_items": [],
                    "damage_detected": False,
                    "wrong_items": [],
                    "is_stock_photo": False,
                    "lighting_consistency": True,
                    "visual_anachronism": False,
                    "overall_assessment": "Vision analysis unavailable.",
                },
            }

    # ------------------------------------------------------------------ #
    # 8. Generate final refund decision
    # ------------------------------------------------------------------ #
    final_decision = generate_refund_decision(
        order_id=order_id,
        order_data=order_data,
        vision_result=vision_result,
        fraud_result=fraud_decision,
    )

    logger.info(
        "Claim processed | order=%s status=%s refund=%.2f",
        order_id,
        final_decision.get("status"),
        (final_decision.get("refund_breakdown") or {}).get("total_refund", 0.0),
    )

    return RefundResponse(**final_decision)
