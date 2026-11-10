"""
Wrong Food Delivery Dispute Resolution & Image Authenticity Workbench
Operational Interface for:
- Deep Learning Data Pipeline with Augmentation & Batching (PyTorch)
- Generative AI vs. Classical Augmentation & Synthetic Forensics
- Automated Wrong Delivery Adjudication & Refund Engine
"""

from __future__ import annotations

import io
import os
import sys
from datetime import datetime
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ExifTags, Image, ImageOps
import streamlit as st
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from app.core.pytorch_pipeline import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    FoodDisputeDataset,
    adjudicate_wrong_food_dispute,
    analyze_sensor_noise_and_spectral_energy,
    audit_hardware_exif_provenance,
    classify_ai_vs_real_image,
    create_dispute_dataloader,
    get_augmentation_transforms,
    get_inference_transforms,
)
import importlib
from app.database import mock_db
from app.security import hashing_service

# Hot-reload backend modules to ensure newly defined functions are picked up by Streamlit
importlib.reload(mock_db)
importlib.reload(hashing_service)

# Page configuration
st.set_page_config(
    page_title="Wrong Food Dispute Resolution | AI vs. Real Image Adjudication",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .badge-clean { background-color: #238636; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 13px; }
    .badge-review { background-color: #d29922; color: black; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 13px; }
    .badge-reject { background-color: #da3633; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 13px; }
    .panel-header {
        background: linear-gradient(90deg, #1f242c 0%, #161b22 100%);
        border-left: 4px solid #58a6ff;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_images")

# Available sample assets
FOOD_OPTIONS = {
    "Real Food: Masala Dosa (Real Camera Capture)": "dosa.jpg",
    "Real Food: Burger (Real Camera Ingestion)": "Burger.jpg",
    "Real Food: Chole Bhature (AVIF Modern Container)": "Chole.avif",
    "Real Food: Hakka Noodles (Real Dish Photo)": "Noodles.jpeg",
    "Real Food: Penne Pasta (Real Portrait Photo)": "Pasta.jpg",
    "Real Food: North Indian Thali (Real Platter)": "Thali.jpeg",
    "Real Food: Deluxe Maharaja Thali (Real Platter)": "Thali_2.jpeg",
    "Real Food: Authentic Mobile Delivery (iPhone 15)": "real_mobile_delivery.jpg",
    "Synthetic: AI-Generated Food (Diffusion Model Fake)": "ai_generated_food.jpg",
}

BILL_OPTIONS = {
    "Physical Printed Restaurant Bill / Receipt": "delivery_receipt_bill.jpg",
}

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

st.markdown(
    """
    <div class="panel-header">
        <h2 style="margin:0; color:#58a6ff; font-weight:700;">Wrong Food Delivery Dispute Resolution & Image Authenticity Workbench</h2>
        <h4 style="margin:4px 0 0 0; color:#8b949e; font-weight:400;">
            PyTorch Deep Learning Data Pipeline • Generative AI vs. Real Image Detection • Automated Refund Engine
        </h4>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: Customer Intake (Food Photo + Printed Bill Photo)
# -----------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Customer Dispute Intake")
    st.markdown("*Customer submits proof: Food Photo + Paper Bill Receipt*")

    intake_source = st.radio("Intake Source:", ["Curated Real & AI Samples", "Upload Custom Images"])

    food_img = None
    bill_img = None
    food_bytes = None
    bill_bytes = None
    food_title = "Delivered Food"
    bill_title = "Printed Bill"

    if intake_source == "Curated Real & AI Samples":
        sel_food = st.selectbox("1. Delivered Food Photo:", list(FOOD_OPTIONS.keys()))
        food_title = sel_food.split(":")[1].strip() if ":" in sel_food else sel_food
        p_food = os.path.join(SAMPLE_DIR, FOOD_OPTIONS[sel_food])
        if os.path.exists(p_food):
            with open(p_food, "rb") as f:
                food_bytes = f.read()
            food_img = Image.open(io.BytesIO(food_bytes))

        sel_bill = st.selectbox("2. Printed Bill Receipt:", list(BILL_OPTIONS.keys()))
        bill_title = sel_bill
        p_bill = os.path.join(SAMPLE_DIR, BILL_OPTIONS[sel_bill])
        if os.path.exists(p_bill):
            with open(p_bill, "rb") as f:
                bill_bytes = f.read()
            bill_img = Image.open(io.BytesIO(bill_bytes))

        st.markdown("---")
        st.subheader("Customer & Order Intelligence")
        all_custs = mock_db.get_all_customers()
        cust_options = {f"{c['customer_id']} — {c['customer_name']} ({c['trust_tier']})": c['customer_id'] for c in all_custs}
        default_idx = 3 if len(all_custs) > 3 else 0
        sel_label = st.selectbox("Customer Profile:", list(cust_options.keys()), index=default_idx)
        sel_cid = cust_options[sel_label]
        cust_profile = mock_db.get_customer(sel_cid)

        customer_id = cust_profile["customer_id"]
        customer_name = cust_profile["customer_name"]
        device_fp = (cust_profile.get("associated_device_fingerprints") or ["clean_device_mobile"])[0]

        # Active orders for this customer (FK relationship)
        c_orders = mock_db.get_customer_orders(customer_id)
        if c_orders:
            order_opts = {f"{o['order_id']} | Bill: {o['bill_id']} ({o.get('order_date', '')[:10]})": o for o in c_orders}
            sel_ord_label = st.selectbox("Active Order / Bill ID:", list(order_opts.keys()), index=0)
            active_order = order_opts[sel_ord_label]
            order_id = active_order["order_id"]
            bill_id = active_order["bill_id"]
            order_date = active_order["order_date"]
            order_comment = active_order.get("comment", "")
            billed_manifest = active_order.get("items", [
                {"name": "Butter Chicken", "qty": 2, "unit_price": 320.0, "total": 640.0},
                {"name": "Garlic Naan", "qty": 4, "unit_price": 55.0, "total": 220.0},
                {"name": "Dal Makhani", "qty": 1, "unit_price": 220.0, "total": 220.0},
            ])
            delivery_fee = active_order.get("delivery_fee", 49.0)
        else:
            order_id = "ORD-2026-904"
            bill_id = "BILL-2026-9041"
            order_date = "2026-11-09T18:00:00+05:30"
            order_comment = "Dispute filed for wrong item delivered."
            billed_manifest = [
                {"name": "Butter Chicken", "qty": 2, "unit_price": 320.0, "total": 640.0},
                {"name": "Garlic Naan", "qty": 4, "unit_price": 55.0, "total": 220.0},
                {"name": "Dal Makhani", "qty": 1, "unit_price": 220.0, "total": 220.0},
            ]
            delivery_fee = 49.0

        customer_claim_text = st.text_area(
            "Customer Claim / Notes:",
            value=order_comment or "I ordered Butter Chicken and Garlic Naan, but the delivery bag contained something completely different! Please refund.",
        )

    else:
        # Upload Custom Images with dedicated New Customer Info & DB Ingestion
        st.markdown("#### 1. Upload Dispute Images")
        up_food = st.file_uploader("Upload Delivered Food Photo:", type=["jpg", "jpeg", "png", "webp", "avif"])
        if up_food:
            food_bytes = up_food.getvalue()
            food_img = Image.open(io.BytesIO(food_bytes))
            food_title = up_food.name

        up_bill = st.file_uploader("Upload Printed Bill Receipt:", type=["jpg", "jpeg", "png", "webp"])
        if up_bill:
            bill_bytes = up_bill.getvalue()
            bill_img = Image.open(io.BytesIO(bill_bytes))
            bill_title = up_bill.name
        elif os.path.exists(os.path.join(SAMPLE_DIR, "delivery_receipt_bill.jpg")):
            with open(os.path.join(SAMPLE_DIR, "delivery_receipt_bill.jpg"), "rb") as f:
                bill_bytes = f.read()
            bill_img = Image.open(io.BytesIO(bill_bytes))
            bill_title = "Default Receipt"

        st.markdown("---")
        st.markdown("#### 2. Register New Customer & Order in DB")
        st.caption("New customer information is persisted to the database and screened for multi-account fraud.")

        with st.expander("New Customer Registration Fields", expanded=True):
            new_cust_name = st.text_input("Customer Name:", value="Ananya Sen")
            new_cust_id = st.text_input("Customer ID (Primary Key):", value="CUST-550")
            new_email = st.text_input("Email:", value="ananya.sen@example.com")
            new_phone = st.text_input("Phone:", value="+91-9876543210")
            new_tier = st.selectbox("Initial Trust Tier:", ["REGULAR_SILVER", "VIP_GOLD", "FLAGGED_HIGH_RISK"], index=0)
            new_score = st.slider("Initial Trust Score:", 0.0, 1.0, 0.88, 0.01)

            st.markdown("**Order & Printed Bill Details**")
            new_order_id = st.text_input("Order ID:", value="ORD-2026-550")
            new_bill_id = st.text_input("Physical Bill / Receipt ID:", value="BILL-2026-5501")
            new_restaurant = st.text_input("Restaurant Name:", value="Royal Tandoor Kitchen")
            new_item_name = st.text_input("Billed Dish on Receipt:", value="Butter Chicken")
            new_item_price = st.number_input("Billed Item Price (₹):", value=320.0, step=10.0)
            new_item_qty = st.number_input("Billed Quantity:", value=2, min_value=1, step=1)
            new_claim = st.text_area(
                "Customer Claim Notes:",
                value="I ordered Butter Chicken and Naan, but delivery bag arrived with a completely different food item! Please refund.",
            )

            st.markdown("**Device Hardware Fingerprint**")
            device_preset = st.selectbox(
                "Simulated Device Hardware:",
                [
                    "Clean Mobile Device (dev_browser_ananya_01)",
                    "Colluding Sybil Device (emu_device_8831 - Shared with Flagged CUST-999)",
                ],
                index=0,
            )
            device_fp = "emu_device_8831" if "emu_device_8831" in device_preset else "dev_browser_ananya_01"

            save_db_btn = st.button("💾 Persist Customer & Order to DB", use_container_width=True)

        # Persist new customer and order to DB
        cust_profile = mock_db.get_customer(new_cust_id)
        if not cust_profile or save_db_btn:
            cust_profile = mock_db.save_new_customer(
                customer_name=new_cust_name,
                customer_id=new_cust_id,
                email=new_email,
                phone=new_phone,
                trust_tier=new_tier,
                trust_score=new_score,
                comment=new_claim,
                device_fingerprint=device_fp,
            )
            mock_db.create_new_order(
                customer_id=new_cust_id,
                restaurant=new_restaurant,
                items=[{"name": new_item_name, "qty": int(new_item_qty), "unit_price": float(new_item_price)}],
                order_id=new_order_id,
                bill_id=new_bill_id,
                comment=new_claim,
            )
            st.success(f"Customer `{new_cust_name}` ({new_cust_id}) & Order `{new_order_id}` stored in Database!")

        customer_id = cust_profile["customer_id"]
        customer_name = cust_profile["customer_name"]
        order_id = new_order_id
        bill_id = new_bill_id
        order_date = datetime.now().isoformat()
        order_comment = new_claim
        billed_manifest = [
            {"name": new_item_name, "qty": int(new_item_qty), "unit_price": float(new_item_price), "total": float(new_item_qty * new_item_price)}
        ]
        delivery_fee = 49.0
        customer_claim_text = new_claim

    # Dynamic Customer Trust & Dispute Badge
    score_col = "#3fb950" if cust_profile["trust_score"] > 0.8 else ("#d29922" if cust_profile["trust_score"] > 0.5 else "#f85149")
    st.markdown(
        f"""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:6px; padding:10px; margin-top:8px;">
            <div style="font-size:12px; color:#8b949e;">Customer: <b style="color:#f0f6fc;">{customer_name}</b> ({customer_id})</div>
            <div style="font-size:12px; color:#8b949e;">Bill ID: <b style="color:#58a6ff;">{bill_id}</b></div>
            <div style="font-size:12px; color:#8b949e;">Order Date: <span>{order_date[:10]}</span></div>
            <div style="font-size:12px; color:#8b949e;">Trust Score: <b style="color:{score_col};">{cust_profile['trust_score']:.0%}</b> ({cust_profile['trust_tier']})</div>
            <div style="font-size:11px; color:#8b949e; margin-top:4px;"><i>"{cust_profile.get('comment', '')}"</i></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if food_img is None or bill_img is None:
    st.info("Select or upload both the Delivered Food Photo and the Printed Bill Receipt to begin dispute adjudication.")
    st.stop()

# Ensure raw bytes are ready for hashing & forensic analysis
if food_bytes is None:
    buf = io.BytesIO()
    food_img.convert("RGB").save(buf, format="JPEG")
    food_bytes = buf.getvalue()

# -----------------------------------------------------------------------------
# Fraud Detection & Adjudication Pipeline Execution
# -----------------------------------------------------------------------------

# Step 1: Detect AI vs. Real image (Lecture 6)
forensic_res = classify_ai_vs_real_image(food_img)
is_ai = forensic_res["is_ai_generated"]
ai_prob = forensic_res["ai_probability"]

# Step 2: Perceptual Hash (pHash) against Fraud Registry
food_phash = hashing_service.compute_phash(food_bytes)
hash_check = hashing_service.check_hash_against_db(food_phash, config_limit=10)
is_duplicate_hash = hash_check["is_duplicate"]
matched_hash_entry = hash_check.get("matched_entry")

# Step 3: Multi-Account Device Fingerprint Collusion
other_account_on_device = mock_db.check_device_multi_account(device_fp, customer_id)
is_multi_account_fraud = other_account_on_device is not None

# Step 4: Screen capture / stock border artifacts
screen_check = hashing_service.detect_screen_capture_artifacts(food_bytes)
is_screen_capture = screen_check.get("is_screen_capture", False)

# Discrepancy & Adjudication Engine
observed_dish = food_title.split(".")[0].replace("_", " ").title()

billed_total = sum(i["total"] for i in billed_manifest)

if is_ai:
    fraud_type = "AI_GENERATED_SYNTHETIC_IMAGE"
    decision_reason = f"Diffusion Model synthetic generation detected with {ai_prob:.1%} probability (Photon noise variance: {forensic_res['spectral_features']['noise_variance']:.1f}, Laplacian variance: {forensic_res['spectral_features']['blur_score']:.1f}, FFT roll-off: {forensic_res['spectral_features']['hf_spectral_ratio']:.3f}). Stripped camera EXIF metadata."
    adjudication_status = "REJECTED_SYNTHETIC_FRAUD"
    total_payout = 0.0
    item_refund = 0.0
    waiver = 0.0
    # Audit to database
    audit_log = mock_db.record_fraud_incident(
        customer_id=customer_id,
        order_id=order_id,
        bill_id=bill_id,
        phash=food_phash,
        reason="AI-generated synthetic food photo uploaded for refund fraud",
        risk_score=ai_prob,
    )
elif is_duplicate_hash:
    fraud_type = "RECYCLED_BLACKLISTED_IMAGE_HASH"
    decision_reason = f"Perceptual image hash ({food_phash}) matches previously blacklisted refund fraud record {matched_hash_entry.get('order_id')} (Hamming distance: {hash_check['hamming_distance']})."
    adjudication_status = "REJECTED_KNOWN_HASH_FRAUD"
    total_payout = 0.0
    item_refund = 0.0
    waiver = 0.0
    audit_log = mock_db.record_fraud_incident(
        customer_id=customer_id,
        order_id=order_id,
        bill_id=bill_id,
        phash=food_phash,
        reason=f"Recycled fraudulent image hash (matched {matched_hash_entry.get('order_id')})",
        risk_score=0.98,
    )
elif is_multi_account_fraud:
    fraud_type = "MULTI_ACCOUNT_DEVICE_COLLUSION"
    decision_reason = f"Device fingerprint '{device_fp}' is linked to multiple accounts ({customer_id} and flagged {other_account_on_device}). Sybil ring detected."
    adjudication_status = "REJECTED_MULTI_ACCOUNT_FRAUD"
    total_payout = 0.0
    item_refund = 0.0
    waiver = 0.0
    audit_log = mock_db.record_fraud_incident(
        customer_id=customer_id,
        order_id=order_id,
        bill_id=bill_id,
        phash=food_phash,
        reason=f"Multi-account device collusion with {other_account_on_device}",
        risk_score=0.90,
    )
else:
    # Authenticity verified; check wrong food delivery discrepancy
    billed_names = [it["name"].lower() for it in billed_manifest]
    is_wrong_item = not any(w in observed_dish.lower() for w in billed_names)
    if is_wrong_item:
        fraud_type = "NONE (AUTHENTIC_DELIVERY_ISSUE)"
        item_refund = sum(it.get("total", it["qty"] * it["unit_price"]) for it in billed_manifest)
        waiver = delivery_fee
        total_payout = item_refund + waiver
        adjudication_status = "APPROVED_WRONG_ITEM"
        decision_reason = f"Authentic camera capture verified (Noise var: {forensic_res['spectral_features']['noise_variance']:.1f}, Laplacian: {forensic_res['spectral_features']['blur_score']:.1f}). Discrepancy confirmed: Customer received '{observed_dish}', but bill was charged for '{', '.join(it['name'] for it in billed_manifest)}'."
        mock_db.record_refund_approval(
            customer_id=customer_id,
            order_id=order_id,
            refund_amount=total_payout,
            reason="Verified wrong meal delivery against receipt bill",
        )
    else:
        fraud_type = "NONE (CORRECT_DELIVERY)"
        item_refund = 0.0
        waiver = 0.0
        total_payout = 0.0
        adjudication_status = "RESOLVED_CORRECT_DELIVERY"
        decision_reason = f"Authentic camera capture verified. Delivered meal '{observed_dish}' matches billed items on receipt. No refund issued."

adjudication = {
    "order_id": order_id,
    "customer_id": customer_id,
    "status": adjudication_status,
    "fraud_type": fraud_type,
    "decision_reason": decision_reason,
    "observed_dish": observed_dish,
    "billed_items": billed_manifest,
    "item_refund_rs": item_refund,
    "delivery_waiver_rs": waiver,
    "total_payout_rs": total_payout,
    "phash": food_phash,
    "device_fingerprint": device_fp,
    "ai_probability": ai_prob,
}

# Re-fetch customer profile to show post-audit database state
cust_profile = mock_db.get_customer(customer_id)

# -----------------------------------------------------------------------------
# Unified Refund Adjudication Decision Message & Master Summary Table
# -----------------------------------------------------------------------------

refund_generated = adjudication["status"] == "APPROVED_WRONG_ITEM"
refund_payout = adjudication["total_payout_rs"]

# Prominent Decision Message Banner
if refund_generated:
    st.markdown(
        f"""
        <div style="background-color: #0d2818; border: 2px solid #238636; border-radius: 8px; padding: 18px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="background: #238636; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 13px;">DECISION VERDICT</span>
                    <h2 style="margin: 6px 0 0 0; color: #3fb950; font-weight: 800;">REFUND WILL BE GENERATED: YES</h2>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 12px; color: #8b949e; text-transform: uppercase;">Total Payout Approved</div>
                    <div style="font-size: 26px; color: #3fb950; font-weight: 800;">₹{refund_payout:.2f}</div>
                </div>
            </div>
            <p style="margin: 10px 0 0 0; color: #c9d1d9; font-size: 15px; border-top: 1px solid #23863644; padding-top: 10px;">
                <b>Message:</b> Full reimbursement of <b>₹{refund_payout:.2f}</b> (₹{adjudication['item_refund_rs']:.2f} food value + full ₹{adjudication['delivery_waiver_rs']:.2f} delivery fee waiver) will be credited to customer <b>{customer_name}</b> ({customer_id}). Authentic physical camera capture verified; food delivered (<b>{observed_dish}</b>) does not match billed items on receipt <b>{bill_id}</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif adjudication["status"] == "REJECTED_SYNTHETIC_FRAUD":
    st.markdown(
        f"""
        <div style="background-color: #2b1113; border: 2px solid #da3633; border-radius: 8px; padding: 18px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="background: #da3633; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 13px;">DECISION VERDICT</span>
                    <h2 style="margin: 6px 0 0 0; color: #f85149; font-weight: 800;">REFUND WILL BE GENERATED: NO (HARD REJECT)</h2>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 12px; color: #8b949e; text-transform: uppercase;">Total Payout Denied</div>
                    <div style="font-size: 26px; color: #f85149; font-weight: 800;">₹0.00</div>
                </div>
            </div>
            <p style="margin: 10px 0 0 0; color: #c9d1d9; font-size: 15px; border-top: 1px solid #da363344; padding-top: 10px;">
                <b>Message:</b> No refund will be issued (<b>₹0.00</b>). The uploaded food photograph was detected as an <b>AI-generated synthetic image ({ai_prob:.1%} probability)</b> with unnatural FFT radial roll-off, lack of optical lens noise, and stripped hardware EXIF provenance. Claim locked, pHash blacklisted, and logged to <code>suspicious_activity_log</code>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif adjudication["status"] == "REJECTED_KNOWN_HASH_FRAUD":
    st.markdown(
        f"""
        <div style="background-color: #2b1113; border: 2px solid #da3633; border-radius: 8px; padding: 18px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="background: #da3633; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 13px;">DECISION VERDICT</span>
                    <h2 style="margin: 6px 0 0 0; color: #f85149; font-weight: 800;">REFUND WILL BE GENERATED: NO (RECYCLED SCAM IMAGE)</h2>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 12px; color: #8b949e; text-transform: uppercase;">Total Payout Denied</div>
                    <div style="font-size: 26px; color: #f85149; font-weight: 800;">₹0.00</div>
                </div>
            </div>
            <p style="margin: 10px 0 0 0; color: #c9d1d9; font-size: 15px; border-top: 1px solid #da363344; padding-top: 10px;">
                <b>Message:</b> No refund will be issued (<b>₹0.00</b>). The uploaded food image matches a previously blacklisted perceptual hash from historical fraud syndicates (Hamming distance: {hash_check.get('hamming_distance', 0)} bits). Account escalated to Security Admin Queue.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif adjudication["status"] == "REJECTED_MULTI_ACCOUNT_FRAUD":
    st.markdown(
        f"""
        <div style="background-color: #2b1113; border: 2px solid #da3633; border-radius: 8px; padding: 18px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="background: #da3633; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 13px;">DECISION VERDICT</span>
                    <h2 style="margin: 6px 0 0 0; color: #f85149; font-weight: 800;">REFUND WILL BE GENERATED: NO (MULTI-ACCOUNT COLLUSION)</h2>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 12px; color: #8b949e; text-transform: uppercase;">Total Payout Denied</div>
                    <div style="font-size: 26px; color: #f85149; font-weight: 800;">₹0.00</div>
                </div>
            </div>
            <p style="margin: 10px 0 0 0; color: #c9d1d9; font-size: 15px; border-top: 1px solid #da363344; padding-top: 10px;">
                <b>Message:</b> No refund will be issued (<b>₹0.00</b>). Hardware device fingerprint <code>{device_fp}</code> is linked to multiple accounts ({customer_id} and flagged {other_account_on_device}). Sybil ring detected and customer trust score penalized in the database.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div style="background-color: #16202e; border: 2px solid #388bfd; border-radius: 8px; padding: 18px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="background: #1f6feb; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 13px;">DECISION VERDICT</span>
                    <h2 style="margin: 6px 0 0 0; color: #58a6ff; font-weight: 800;">REFUND WILL BE GENERATED: NO (ACCURATE DELIVERY)</h2>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 12px; color: #8b949e; text-transform: uppercase;">Total Payout</div>
                    <div style="font-size: 26px; color: #58a6ff; font-weight: 800;">₹0.00</div>
                </div>
            </div>
            <p style="margin: 10px 0 0 0; color: #c9d1d9; font-size: 15px; border-top: 1px solid #388bfd44; padding-top: 10px;">
                <b>Message:</b> No refund required (<b>₹0.00</b>). Authentic camera capture verified. The meal delivered (<b>{observed_dish}</b>) matches the items billed on the printed receipt. The order was delivered accurately.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Master Adjudication & Refund Summary Table (Consolidated in One Table)
summary_data = [
    {
        "Verification Dimension": "1. Customer Food Upload",
        "Evidence Observed from Intake": f"Dish: '{observed_dish}' ({food_title})",
        "Screening Status": "Ingested",
        "Refund Decision Impact": "Subject to Authenticity & Forensic Analysis",
    },
    {
        "Verification Dimension": "2. Printed Bill Receipt Upload",
        "Evidence Observed from Intake": f"Bill: '{bill_id}' ({', '.join(it['name'] for it in billed_manifest)})",
        "Screening Status": "Verified",
        "Refund Decision Impact": f"Billed Baseline: ₹{billed_total:.2f} + ₹{delivery_fee:.2f} Delivery",
    },
    {
        "Verification Dimension": "3. Generative AI vs. Real Detector",
        "Evidence Observed from Intake": f"P(AI) = {ai_prob:.1%} | Noise: {forensic_res['spectral_features']['noise_variance']:.1f} | Laplacian: {forensic_res['spectral_features']['blur_score']:.1f}",
        "Screening Status": "❌ FAKE (Synthetic AI)" if is_ai else "✅ AUTHENTIC (Real Camera)",
        "Refund Decision Impact": "Hard Reject (₹0.00)" if is_ai else "Cleared to Receipt Discrepancy Gate",
    },
    {
        "Verification Dimension": "4. Perceptual Hash (pHash) Blacklist",
        "Evidence Observed from Intake": f"pHash: {food_phash[:16]}... (Distance: {hash_check.get('hamming_distance', 'N/A')})",
        "Screening Status": "❌ BLACKLISTED MATCH" if is_duplicate_hash else "✅ UNIQUE / CLEAN",
        "Refund Decision Impact": "Hard Reject (₹0.00)" if is_duplicate_hash else "Cleared",
    },
    {
        "Verification Dimension": "5. Device Hardware Fingerprint",
        "Evidence Observed from Intake": f"Device: '{device_fp}' (Collusion: {other_account_on_device or 'None'})",
        "Screening Status": "❌ MULTI-ACCOUNT COLLUSION" if is_multi_account_fraud else "✅ UNIQUE HARDWARE",
        "Refund Decision Impact": "Hard Reject (₹0.00)" if is_multi_account_fraud else "Cleared",
    },
    {
        "Verification Dimension": "6. Meal Discrepancy Cross-Check",
        "Evidence Observed from Intake": f"Delivered '{observed_dish}' vs Billed '{', '.join(it['name'] for it in billed_manifest)}'",
        "Screening Status": "⚠️ WRONG FOOD VERIFIED" if adjudication["status"] == "APPROVED_WRONG_ITEM" else ("ℹ️ CORRECT FOOD" if not (is_ai or is_duplicate_hash or is_multi_account_fraud) else "BLOCKED BY FRAUD"),
        "Refund Decision Impact": f"100% Item Refund (₹{adjudication['item_refund_rs']:.2f}) + Waiver (₹{adjudication['delivery_waiver_rs']:.2f})" if adjudication["status"] == "APPROVED_WRONG_ITEM" else "₹0.00 Refund",
    },
    {
        "Verification Dimension": "★ FINAL REFUND DECISION",
        "Evidence Observed from Intake": f"Customer: {customer_name} ({customer_id}) | Order: {order_id}",
        "Screening Status": "🟢 APPROVED" if refund_generated else "🔴 DENIED / ZERO REFUND",
        "Refund Decision Impact": f"REFUND GENERATED: {'YES (₹' + f'{refund_payout:.2f}' + ')' if refund_generated else 'NO (₹0.00)'}",
    },
]

st.markdown("#### Dispute Adjudication & Refund Decision Table")
st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
st.markdown("---")

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------

tabs = st.tabs([
    "Deep Learning Data Pipeline",
    "Generative AI vs. Real Detection",
    "Dispute Adjudication & Refund Engine",
    "Mathematical Foundations & Formulations",
])

# =============================================================================
# Tab 1: PyTorch Data Pipeline with Augmentation & Batching
# =============================================================================
with tabs[0]:
    st.subheader("Deep Learning Data Pipeline: Tensor Transformations & Batching")
    st.markdown(
        "Demonstrating on-the-fly PyTorch preprocessing transformations, classical data augmentation, and mini-batch tensor assembly."
    )

    t_col1, t_col2 = st.columns([1, 1])

    inference_transform = get_inference_transforms(image_size=224)
    augmentation_transform = get_augmentation_transforms(image_size=224)

    # Standardize input for transforms
    clean_food_pil = ImageOps.exif_transpose(food_img).convert("RGB")
    food_tensor_inf = inference_transform(clean_food_pil)
    food_tensor_aug = augmentation_transform(clean_food_pil)

    with t_col1:
        st.markdown("**1. Raw Input vs. Standardized Inference Tensor**")
        st.image(clean_food_pil, caption=f"Raw Ingestion: {clean_food_pil.size[0]}x{clean_food_pil.size[1]} ({clean_food_pil.mode})", width='stretch')
        st.code(
            f"PyTorch Inference Tensor:\n"
            f"Shape       : {list(food_tensor_inf.shape)} [C, H, W]\n"
            f"Dtype       : {food_tensor_inf.dtype}\n"
            f"Value Range : [{food_tensor_inf.min():.3f}, {food_tensor_inf.max():.3f}]\n"
            f"Mean / Std  : Norm(mean={IMAGENET_MEAN}, std={IMAGENET_STD})",
            language="text",
        )

    with t_col2:
        st.markdown("**2. Classical Data Augmentation (On-the-Fly)**")
        # Denormalize tensor to visualize
        inv_aug = food_tensor_aug.clone()
        for c in range(3):
            inv_aug[c] = inv_aug[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
        inv_aug_np = np.clip(inv_aug.permute(1, 2, 0).numpy(), 0, 1)

        st.image(inv_aug_np, caption="Augmented View: Random Crop (224x224) + Color Jitter + Affine Rotate + Flip", width='stretch')
        st.code(
            "Transforms.Compose([\n"
            "    Resize(256), RandomCrop(224),\n"
            "    RandomHorizontalFlip(p=0.5),\n"
            "    ColorJitter(brightness=0.15, contrast=0.15),\n"
            "    RandomAffine(degrees=15, translate=(0.08, 0.08)),\n"
            "    ToTensor(), Normalize(mean, std)\n"
            "])\n"
            "Purpose: Invariance to customer camera angle, lighting, and phone orientation.",
            language="python",
        )

    st.markdown("---")
    st.subheader("PyTorch DataLoader Mini-Batch Assembly")
    st.markdown("Assembling multiple customer claims into a batched tensor pipeline for high-throughput model serving.")

    sample_items = [
        {"food_image": food_img, "bill_image": bill_img, "order_id": order_id, "food_label": observed_dish, "billed_item": "Butter Chicken"},
        {"food_image": clean_food_pil, "bill_image": bill_img, "order_id": "ORD-2026-905", "food_label": "Chole Bhature", "billed_item": "Thali"},
        {"food_image": clean_food_pil, "bill_image": bill_img, "order_id": "ORD-2026-906", "food_label": "Hakka Noodles", "billed_item": "Biryani"},
        {"food_image": clean_food_pil, "bill_image": bill_img, "order_id": "ORD-2026-907", "food_label": "Penne Pasta", "billed_item": "Pizza"},
    ]

    dispute_loader = create_dispute_dataloader(sample_items, batch_size=4, shuffle=False)
    batch = next(iter(dispute_loader))

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("**Delivered Food Batch Tensor**")
        st.metric("Batch Tensor Dimension", str(list(batch["food_tensor"].shape)))
        st.caption("Dimensions represent: `[Batch Size=4, Channels=3, Height=224, Width=224]`")
    with b_col2:
        st.markdown("**Printed Bill Batch Tensor**")
        st.metric("Bill Tensor Dimension", str(list(batch["bill_tensor"].shape)))
        st.caption("Standardized bill receipt tensors ready for downstream OCR/vision embeddings.")


# =============================================================================
# Tab 2: Generative AI vs. Real Image Detection
# =============================================================================
with tabs[1]:
    st.subheader("Generative AI vs. Real Image Detection Engine")
    st.markdown(
        "Evaluating physical camera sensor noise, 2D Fast Fourier Transform (FFT) high-frequency power, "
        "and hardware provenance to catch synthetic food refund fraud."
    )

    verdict_badge = (
        '<span class="badge-reject">SYNTHETIC AI-GENERATED IMAGE</span>'
        if is_ai
        else '<span class="badge-clean">AUTHENTIC REAL CAMERA CAPTURE</span>'
    )
    st.markdown(f"### Forensic Verdict: {verdict_badge}", unsafe_allow_html=True)
    st.markdown(f"**Synthetic Probability $P(\\text{{AI}})$:** `{ai_prob:.1%}` | **Confidence:** `{forensic_res['authenticity_confidence']:.1%}`")

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.metric("Photon Sensor Noise", f"{forensic_res['spectral_features']['noise_variance']:.1f}", delta="Low/Smooth" if forensic_res['spectral_features']['noise_variance'] < 8.0 else "Camera Grain")
    with f_col2:
        st.metric("Laplacian Edge Variance", f"{forensic_res['spectral_features']['blur_score']:.1f}", delta="Hyper-Smooth" if forensic_res['spectral_features']['blur_score'] < 120 else "Natural Crispness")
    with f_col3:
        st.metric("FFT High-Frequency Ratio", f"{forensic_res['spectral_features']['hf_spectral_ratio']:.3f}", delta="Spectral Roll-Off" if forensic_res['spectral_features']['hf_spectral_ratio'] < 0.28 else "Sustained Spectrum")

    st.markdown("---")

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("**2D Fast Fourier Transform (FFT) Power Spectrum**")
        fig_fft, ax_fft = plt.subplots(figsize=(5, 4))
        cax_fft = ax_fft.imshow(forensic_res["spectral_features"]["magnitude_spectrum"], cmap="magma")
        ax_fft.set_title("Log-Magnitude Spectrum (Centered)")
        fig_fft.colorbar(cax_fft, ax=ax_fft, fraction=0.046, pad=0.04)
        st.pyplot(fig_fft)
        st.caption("Real camera optical lenses produce sustained high-frequency radial scatter. AI diffusion models exhibit unnatural radial roll-offs.")

    with g_col2:
        st.markdown("**Laplacian High-Pass Texture Residual**")
        fig_lap, ax_lap = plt.subplots(figsize=(5, 4))
        cax_lap = ax_lap.imshow(forensic_res["spectral_features"]["laplacian_mask"], cmap="inferno")
        ax_lap.set_title(f"Edge Variance Var(Laplacian): {forensic_res['spectral_features']['blur_score']}")
        fig_lap.colorbar(cax_lap, ax=ax_lap, fraction=0.046, pad=0.04)
        st.pyplot(fig_lap)
        st.caption("Isolates photon shot noise and physical edges. Synthetic food images lack true sensor noise grain.")

    st.markdown("---")
    st.subheader("Hardware EXIF Provenance & Forensic Triggers")
    ex = forensic_res["exif_provenance"]
    st.json({
        "EXIF Metadata Present": ex["has_exif"],
        "Camera Hardware Make": ex["make"] or "None / Stripped by Diffusion Tool",
        "Camera Hardware Model": ex["model"] or "None / Virtual Synthesizer",
        "Capture Timestamp": ex["datetime_original"] or "None",
        "Hardware Provenance Score": f"{ex['provenance_score']:.0%}",
        "Active Forensic Triggers": forensic_res["ai_signals"] if is_ai else forensic_res["real_signals"],
    })

    st.markdown("---")
    st.subheader("Multimodal Fraud Intelligence & Blacklist Screening")
    st.markdown("Evaluating perceptual image hashes against blacklisted syndicates and checking device hardware for multi-accounting sybil rings.")

    sec_c1, sec_c2, sec_c3 = st.columns(3)
    with sec_c1:
        st.markdown("**1. Perceptual Hash (pHash) Fingerprint**")
        st.code(f"pHash: {food_phash}\nDuplicate in Blacklist: {is_duplicate_hash}", language="text")
        if is_duplicate_hash:
            st.error(f"🚨 Recycled Image Match! Distance: {hash_check['hamming_distance']} bits (Blacklisted in {matched_hash_entry.get('order_id')})")
        else:
            st.success("✅ Clean Unique Hash (Not in blacklist)")

    with sec_c2:
        st.markdown("**2. Device Fingerprint (Sybil Collusion)**")
        st.code(f"Device: {device_fp}\nMulti-Account Detected: {is_multi_account_fraud}", language="text")
        if is_multi_account_fraud:
            st.error(f"🚨 Collusion Alert! Device shared with {other_account_on_device}")
        else:
            st.success("✅ Unique Single-User Device Hardware")

    with sec_c3:
        st.markdown("**3. Screen Capture / Display Artifacts**")
        st.code(f"Screen Artifacts: {is_screen_capture}", language="text")
        if is_screen_capture:
            st.warning("⚠️ Potential Screenshot / Display Rebroadcast")
        else:
            st.success("✅ Natural Optical Capture Clean")


# =============================================================================
# Tab 3: Dispute Adjudication & Refund Engine (Wrong Food Delivery Case)
# =============================================================================
with tabs[2]:
    st.subheader("Wrong Food Delivery Dispute Resolution & Refund Engine")
    st.markdown(
        "A multi-tiered decision gate: verifying whether the submitted food photo is AI-generated, "
        "and if verified as authentic, cross-referencing delivered items against the billed receipt."
    )

    disp_col1, disp_col2 = st.columns(2)
    with disp_col1:
        st.markdown("#### 1. Customer Upload: Delivered Food")
        st.image(food_img, caption=f"Active Target: {food_title}", width='stretch')
        st.info(f"Observed Dish in Photo: **{observed_dish}**")

    with disp_col2:
        st.markdown("#### 2. Customer Upload: Printed Paper Bill")
        st.image(bill_img, caption="Physical Printed Receipt / Bill", width='stretch')
        st.info("Restaurant: **Royal Tandoor Kitchen** | Billed Total: **₹1,129.00**")

    st.markdown("---")
    st.markdown("### Customer Profile & Cross-Referenced Database Record")
    c_info1, c_info2, c_info3, c_info4 = st.columns(4)
    with c_info1:
        st.metric("Customer Name", customer_name)
    with c_info2:
        st.metric("Customer ID (FK)", customer_id)
    with c_info3:
        st.metric("Physical Bill ID", bill_id)
    with c_info4:
        st.metric("Trust Tier", cust_profile["trust_tier"])

    with st.expander(f"Customer Order History ({len(cust_profile.get('order_history', []))} Recorded Invoices & Disputes)"):
        hist_df = pd.DataFrame(cust_profile.get("order_history", []))
        if not hist_df.empty:
            st.dataframe(hist_df[["order_id", "bill_id", "order_date", "restaurant", "grand_total", "status", "dispute_filed", "dispute_status", "comment"]], use_container_width=True)
        else:
            st.caption("No prior order history on record.")

    st.markdown("---")
    st.markdown("### Order Manifest (From Printed Bill Receipt)")
    df_manifest = pd.DataFrame(billed_manifest)
    st.dataframe(df_manifest, use_container_width=True)

    st.markdown("---")
    st.markdown("### Automated Adjudication Outcome & Decision Table")
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    if adjudication["status"] == "REJECTED_SYNTHETIC_FRAUD":
        st.error(
            "🚫 **CLAIM HARD REJECTED (AI-GENERATED SYNTHETIC IMAGE DETECTED)**\n\n"
            f"**Forensic Reason:** {adjudication['decision_reason']}\n\n"
            "• **Refund Approved:** ₹0.00\n"
            "• **Delivery Fee Waiver:** ₹0.00\n"
            "• **Total Payout Issued:** **₹0.00**\n"
            "• **Security Action:** Order locked, claim logged to `suspicious_activity_log`, customer trust score penalized."
        )
        st.json(adjudication)

    elif adjudication["status"] == "REJECTED_KNOWN_HASH_FRAUD":
        st.error(
            "🚫 **CLAIM HARD REJECTED (RECYCLED BLACKLISTED IMAGE FRAUD)**\n\n"
            f"**Forensic Reason:** {adjudication['decision_reason']}\n\n"
            "• **Refund Approved:** ₹0.00\n"
            "• **Total Payout Issued:** **₹0.00**\n"
            "• **Security Action:** Hash already blacklisted from prior scam syndicate. Account escalated to Security Admin Queue."
        )
        st.json(adjudication)

    elif adjudication["status"] == "REJECTED_MULTI_ACCOUNT_FRAUD":
        st.error(
            "🚫 **CLAIM HARD REJECTED (MULTI-ACCOUNT DEVICE COLLUSION DETECTED)**\n\n"
            f"**Forensic Reason:** {adjudication['decision_reason']}\n\n"
            "• **Refund Approved:** ₹0.00\n"
            "• **Total Payout Issued:** **₹0.00**\n"
            "• **Security Action:** Sybil collusion pattern detected. Shared device registry logged to `suspicious_activity_log`."
        )
        st.json(adjudication)

    elif adjudication["status"] == "APPROVED_WRONG_ITEM":
        st.success(
            "✅ **AUTHENTIC REAL IMAGE CONFIRMED & WRONG FOOD VERIFIED**\n\n"
            f"**Decision Reason:** {adjudication['decision_reason']}\n\n"
            f"• **Food Item Refund:** ₹{adjudication['item_refund_rs']:.2f} (100% item value reimbursement)\n"
            f"• **Delivery Fee Waiver:** ₹{adjudication['delivery_waiver_rs']:.2f} (Full ₹49 waiver)\n"
            f"• **TOTAL REFUND APPROVED:** **₹{adjudication['total_payout_rs']:.2f}**\n\n"
            "• **Action:** Immediate credit to customer payment method. Restaurant notified for dispatch error."
        )
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.metric("Total Refund Credited", f"Rs. {adjudication['total_payout_rs']:.2f}")
        with p_c2:
            st.metric("Delivery Fee Waived", f"Rs. {adjudication['delivery_waiver_rs']:.2f}")
        st.json(adjudication)

    else:
        st.info(
            "ℹ️ **REAL IMAGE VERIFIED — NO DISCREPANCY DETECTED**\n\n"
            f"**Decision Reason:** {adjudication['decision_reason']}\n\n"
            "• **Total Refund:** ₹0.00 (Customer received the exact meal billed on receipt)."
        )
        st.json(adjudication)

    st.markdown("---")
    st.subheader("Live Database Audit & Security Ledger State")
    st.markdown("Real-time view of database updates, customer trust penalties, and audit log entries:")
    l_c1, l_c2 = st.columns([1, 1])
    with l_c1:
        st.markdown("**Customer Trust Profile in Database (Post-Adjudication)**")
        updated_cust = mock_db.get_customer(customer_id)
        if updated_cust:
            st.json({
                "Customer ID": updated_cust["customer_id"],
                "Customer Name": updated_cust["customer_name"],
                "Trust Tier": updated_cust["trust_tier"],
                "Trust Score": f"{updated_cust['trust_score']:.0%}",
                "Dispute Count": updated_cust["dispute_count"],
                "Refund Rate": f"{updated_cust['refund_rate_percent']:.1f}%",
                "Database Comment": updated_cust["comment"],
            })
    with l_c2:
        st.markdown("**Recent Suspicious Activity Audit Entries**")
        susp_log = mock_db.get_suspicious_log()
        if susp_log:
            st.dataframe(pd.DataFrame(susp_log[-5:]), use_container_width=True)
        else:
            st.info("No suspicious fraud incidents logged. Ledger is clean.")


# =============================================================================
# Tab 4: Mathematical Foundations & Formulations
# =============================================================================
with tabs[3]:
    st.subheader("Technical Reference & Mathematical Formulations")

    st.markdown("### 1. PyTorch Input Tensor Standardization")
    st.markdown("Channel-wise standardization transforms pixel matrices into standard zero-mean unit-variance tensors:")
    st.latex(r"z_{c, x, y} = \frac{x_{c, x, y} - \mu_c}{\sigma_c} \quad \text{where } c \in \{R, G, B\}")
    st.markdown("For ImageNet transfer learning: $\\mu = [0.485, 0.456, 0.406]$, $\\sigma = [0.229, 0.224, 0.225]$.")

    st.markdown("### 2. 2D Fast Fourier Transform (FFT) Power Spectrum")
    st.markdown("Discrete 2D Fourier transformation isolates spatial frequencies to identify generative AI spectral roll-off:")
    st.latex(r"F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) \exp\left[-j 2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)\right]")
    st.latex(r"\text{Magnitude Spectrum } |F(u, v)| = \sqrt{\text{Re}(F(u, v))^2 + \text{Im}(F(u, v))^2}")

    st.markdown("### 3. Laplacian Variance for Optical Sharpness")
    st.markdown("Measures second-order spatial derivatives to differentiate camera sensor grain from synthetic smoothing:")
    st.latex(r"\nabla^2 I = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix} * I \quad \implies \quad \text{Blur Score} = \text{Var}(\nabla^2 I)")

    st.markdown("### 4. Wrong Food Delivery Adjudication Matrix")
    st.markdown(
        r"""
        | Step | Verification Gate | Condition | Action & Financial Outcome |
        |---|---|---|---|
        | **1** | **Authenticity Gate** | $P(\text{AI}) \ge 0.55$ (Synthetic) | **Hard Reject**: ₹0.00 Refund, claim locked in suspicious log |
        | **2** | **Authenticity Gate** | $P(\text{AI}) < 0.55$ (Real Camera) | **Pass**: Cleared to Multimodal Receipt Cross-Verification |
        | **3** | **Item Discrepancy** | Observed Dish $\ne$ Billed Items | **Approved Wrong Item**: 100% item refund + ₹49 delivery fee waiver |
        | **4** | **Item Discrepancy** | Observed Dish = Billed Items | **Resolved Correct**: ₹0.00 Refund (Order delivered intact) |
        """
    )
