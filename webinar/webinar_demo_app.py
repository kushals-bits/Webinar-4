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

from app.core import pytorch_pipeline
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
importlib.reload(pytorch_pipeline)
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

def build_pipeline_graphviz_dot(
    food_title: str,
    bill_id: str,
    clean_food_pil: Image.Image,
    food_bytes: bytes,
    food_tensor_inf: torch.Tensor,
    forensic_res: dict[str, Any],
    is_ai: bool,
    ai_prob: float,
    food_phash: str,
    is_duplicate_hash: bool,
    is_screen_capture: bool,
    is_wrong_item: bool,
    observed_dish: str,
    billed_manifest: list[dict[str, Any]],
    billed_total: float,
    adjudication_status: str,
    total_payout: float,
    customer_id: str,
    trust_score: float,
) -> str:
    clean_food_title = food_title.replace('"', '').replace("'", "")
    clean_dish = observed_dish.replace('"', '').replace("'", "")
    billed_str = ", ".join(it.get("name", "") for it in billed_manifest).replace('"', '').replace("'", "")
    
    spec = forensic_res.get("spectral_features", {})
    exif = forensic_res.get("exif_provenance", {})
    exif_hw = f"{exif.get('make') or 'None'} {exif.get('model') or ''}".strip()
    if not exif_hw or exif_hw == "None":
        exif_hw = "None / Stripped by Synthesizer"

    peakiness = spec.get("hf_peakiness", 0.0)
    flat_noise = spec.get("flat_noise_variance", 0.0)
    hf_ratio = spec.get("hf_spectral_ratio", 0.0)

    c_neu_bg = "#161b22"
    c_neu_border = "#388bfd"
    c_neu_font = "#e6edf3"

    c_pass_bg = "#0d2818"
    c_pass_border = "#238636"
    c_pass_font = "#3fb950"

    c_fail_bg = "#3d1111"
    c_fail_border = "#da3633"
    c_fail_font = "#ff7b72"

    c_bypass_bg = "#21262d"
    c_bypass_border = "#484f58"
    c_bypass_font = "#8b949e"

    dot_lines = [
        "digraph BackendExecutionPipeline {",
        "  graph [rankdir=TB, bgcolor=\"transparent\", compound=true, nodesep=0.45, ranksep=0.42];",
        "  node [shape=box, style=\"filled,rounded\", fontname=\"Arial\", fontsize=10, margin=\"0.22,0.16\", penwidth=1.8];",
        "  edge [fontname=\"Arial\", fontsize=9, penwidth=1.6];",
        "",
        "  // Stage 1: Dual Image Intake",
        f"  S1 [label=\"STAGE 1: DUAL IMAGE INTAKE\\n\\n[INPUT]\\n* Food Proof: {clean_food_title} ({clean_food_pil.size[0]}x{clean_food_pil.size[1]} px, {len(food_bytes)//1024} KB)\\n* Printed Receipt: {bill_id}\\n\\n[BACKEND PROCESS]\\n* Tag 274 EXIF auto-orientation transpose\\n* Non-blocking RGB canvas normalization\\n\\n[OUTPUT]\\n* Normalized RGB Canvas (3 Channels)\\n* Staged In-Memory Binary Buffer\", fillcolor=\"{c_neu_bg}\", fontcolor=\"{c_neu_font}\", color=\"{c_neu_border}\"];",
        "",
        "  // Stage 2: PyTorch Deep Learning Ingestion",
        f"  S2 [label=\"STAGE 2: PYTORCH DATA PIPELINE (DL INGESTION)\\n\\n[INPUT]\\n* Normalized RGB Image Canvas\\n\\n[BACKEND PROCESS]\\n* Resize(256) -> CenterCrop(224)\\n* ToTensor() -> ImageNet Normalization (mean, std)\\n\\n[OUTPUT]\\n* FloatTensor: [3, 224, 224]\\n* Value Range: [{food_tensor_inf.min():.2f}, {food_tensor_inf.max():.2f}] (Zero-Mean Unit-Variance)\", fillcolor=\"{c_neu_bg}\", fontcolor=\"{c_neu_font}\", color=\"{c_neu_border}\"];",
        "",
        "  S1 -> S2 [color=\"#58a6ff\"];",
        "",
        "  // Stage 3: Generative AI vs Real Camera Forensics",
    ]

    if is_ai:
        s3_bg, s3_border, s3_font = c_fail_bg, c_fail_border, c_fail_font
        s3_verdict = f"SYNTHETIC AI GENERATION (P(AI) = {ai_prob:.1%})"
    else:
        s3_bg, s3_border, s3_font = c_pass_bg, c_pass_border, c_pass_font
        s3_verdict = f"AUTHENTIC REAL CAMERA (P(AI) = {ai_prob:.1%})"

    dot_lines.extend([
        f"  S3 [label=\"STAGE 3: GENERATIVE AI vs. REAL DETECTOR\\n\\n[INPUT]\\n* PyTorch Ingestion Tensor and Spatial Frequency Plane\\n\\n[BACKEND PROCESS]\\n* 2D FFT Nyquist Ring (r >= 0.55): HF Ratio = {hf_ratio:.3f}\\n* VAE Lattice Peakiness: {peakiness:.3f} (Threshold: 0.096)\\n* Flat Sensor Shot Noise: {flat_noise:.2f} Var\\n* Hardware EXIF Audit: {exif_hw}\\n\\n[OUTPUT]\\n* Verdict: {s3_verdict}\", fillcolor=\"{s3_bg}\", fontcolor=\"{s3_font}\", color=\"{s3_border}\"];",
        "",
        "  S2 -> S3 [color=\"#58a6ff\"];",
        "",
    ])

    if is_ai:
        dot_lines.extend([
            f"  S6_Fraud [label=\"STAGE 6: ADJUDICATION AND FRAUD LOCK\\n\\n[INPUT]\\n* P(AI) = {ai_prob:.1%} Synthetic Signal\\n* Customer ID: {customer_id} (Trust: {trust_score:.0%})\\n\\n[BACKEND PROCESS]\\n* Hard reject refund claim\\n* Record audit in suspicious_activity_log\\n* Slash customer trust score (-15 pts)\\n\\n[OUTPUT]\\n* Status: REJECTED_SYNTHETIC_FRAUD\\n* Total Payout: Rs 0.00\\n* Claim Locked and Prevented Loss: Rs {billed_total + 49.0:.2f}\", fillcolor=\"{c_fail_bg}\", fontcolor=\"{c_fail_font}\", color=\"{c_fail_border}\"];",
            "",
            f"  S_Bypass [label=\"STAGE 4 and 5: HASH INTEGRITY and RECEIPT VERIFICATION\\n\\n[STATUS: BYPASSED AND LOCKED]\\n* Receipt cross-verification skipped\\n* Financial payout blocked due to synthetic image fraud\", fillcolor=\"{c_bypass_bg}\", fontcolor=\"{c_bypass_font}\", color=\"{c_bypass_border}\", style=\"dashed,filled\"];",
            "",
            "  S3 -> S6_Fraud [label=\"Synthetic Diffusion Flagged (P(AI) >= 50%)\", color=\"#da3633\", fontcolor=\"#ff7b72\"];",
            "  S3 -> S_Bypass [style=\"dashed\", color=\"#484f58\", label=\"Bypassed\"];",
            "  S_Bypass -> S6_Fraud [style=\"dashed\", color=\"#484f58\"];",
            "}",
        ])
    else:
        s4_bg = c_fail_bg if is_duplicate_hash else c_pass_bg
        s4_border = c_fail_border if is_duplicate_hash else c_pass_border
        s4_font = c_fail_font if is_duplicate_hash else c_pass_font
        s4_verdict = "RECYCLED DUPLICATE MATCH" if is_duplicate_hash else "UNIQUE / CLEAN (No match in registry)"

        dot_lines.extend([
            f"  S4 [label=\"STAGE 4: DUPLICATE HASH and TAMPERING GATE\\n\\n[INPUT]\\n* Staged Image Byte Stream\\n\\n[BACKEND PROCESS]\\n* 64-bit DCT Perceptual Hash (pHash)\\n* Hamming distance check vs Fraud Blacklist (Thresh <= 10)\\n* Screen capture border uniformity audit\\n\\n[OUTPUT]\\n* pHash: {food_phash[:16]}...\\n* Registry Status: {s4_verdict}\\n* Screen Capture: {'SCREENSHOT' if is_screen_capture else 'DIRECT PHOTO'}\", fillcolor=\"{s4_bg}\", fontcolor=\"{s4_font}\", color=\"{s4_border}\"];",
            "",
            "  S3 -> S4 [label=\"Authentic Camera Cleared\", color=\"#238636\", fontcolor=\"#3fb950\"];",
            "",
        ])

        if is_duplicate_hash:
            dot_lines.extend([
                f"  S6_Hash [label=\"STAGE 6: ADJUDICATION AND FRAUD LOCK\\n\\n[INPUT]\\n* Recycled Fraudulent Hash Match\\n\\n[BACKEND PROCESS]\\n* Flag duplicate claim abuse\\n* Log incident to audit registry\\n\\n[OUTPUT]\\n* Status: REJECTED_KNOWN_HASH_FRAUD\\n* Total Payout: Rs 0.00\", fillcolor=\"{c_fail_bg}\", fontcolor=\"{c_fail_font}\", color=\"{c_fail_border}\"];",
                "",
                "  S4 -> S6_Hash [label=\"Recycled Hash Flagged\", color=\"#da3633\", fontcolor=\"#ff7b72\"];",
                "}",
            ])
        else:
            s5_bg = c_pass_bg if is_wrong_item else c_neu_bg
            s5_border = c_pass_border if is_wrong_item else c_neu_border
            s5_font = c_pass_font if is_wrong_item else c_neu_font
            s5_verdict = f"WRONG ITEM DELIVERED (Received {clean_dish})" if is_wrong_item else f"ORDER MATCHED (Delivered {clean_dish})"

            dot_lines.extend([
                f"  S5 [label=\"STAGE 5: MULTIMODAL RECEIPT CROSS-VERIFICATION\\n\\n[INPUT]\\n* Observed Delivered Dish: {clean_dish}\\n* Billed Order Receipt: {billed_str} (Rs {billed_total:.2f})\\n\\n[BACKEND PROCESS]\\n* Cross-modality receipt discrepancy comparison\\n* Manifest string and visual plate mapping\\n\\n[OUTPUT]\\n* Discrepancy Verdict: {s5_verdict}\", fillcolor=\"{s5_bg}\", fontcolor=\"{s5_font}\", color=\"{s5_border}\"];",
                "",
                "  S4 -> S5 [label=\"Hash Integrity Cleared\", color=\"#238636\", fontcolor=\"#3fb950\"];",
                "",
            ])

            if is_wrong_item:
                dot_lines.extend([
                    f"  S6_Settlement [label=\"STAGE 6: ADJUDICATION AND REFUND SETTLEMENT\\n\\n[INPUT]\\n* Verified Real Photo + Confirmed Order Discrepancy\\n* Customer Trust Score: {trust_score:.0%}\\n\\n[BACKEND PROCESS]\\n* Compute 100% item refund: Rs {billed_total:.2f}\\n* Compute 100% delivery waiver: Rs 49.00\\n* Record approval in customer ledger\\n\\n[OUTPUT]\\n* Status: APPROVED_WRONG_ITEM\\n* Total Payout: Rs {total_payout:.2f}\\n* Customer Trust Intact\", fillcolor=\"{c_pass_bg}\", fontcolor=\"{c_pass_font}\", color=\"{c_pass_border}\"];",
                    "",
                    "  S5 -> S6_Settlement [label=\"Discrepancy Verified\", color=\"#238636\", fontcolor=\"#3fb950\"];",
                    "}",
                ])
            else:
                dot_lines.extend([
                    f"  S6_Settlement [label=\"STAGE 6: ADJUDICATION (NO DISCREPANCY)\\n\\n[INPUT]\\n* Verified Real Photo + Matching Receipt Items\\n\\n[BACKEND PROCESS]\\n* Verify delivered items match billed order\\n* No compensation warranted\\n\\n[OUTPUT]\\n* Status: RESOLVED_CORRECT_DELIVERY\\n* Total Payout: Rs 0.00\\n* Dispute Closed\", fillcolor=\"{c_neu_bg}\", fontcolor=\"{c_neu_font}\", color=\"{c_neu_border}\"];",
                    "",
                    "  S5 -> S6_Settlement [label=\"Order Verified Correct\", color=\"#388bfd\", fontcolor=\"#58a6ff\"];",
                    "}",
                ])

    return "\n".join(dot_lines)

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
    "Synthetic: AI-Generated Indian Thali (User Case / WhatsApp Diffusion)": "whatsapp_ai_test.jpg",
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

# Stage 1 & 2: Canonical Ingestion & PyTorch Tensor Standardization
clean_food_pil = ImageOps.exif_transpose(food_img).convert("RGB")
inference_transform = get_inference_transforms(image_size=224)
food_tensor_inf = inference_transform(clean_food_pil)

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
billed_names = [it["name"].lower() for it in billed_manifest]
is_wrong_item = not any(w in observed_dish.lower() for w in billed_names)
billed_total = sum(i["total"] for i in billed_manifest)

if is_ai:
    fraud_type = "AI_GENERATED_SYNTHETIC_IMAGE"
    signals_summary = "; ".join(forensic_res["ai_signals"][:2]) if forensic_res.get("ai_signals") else "Synthetic diffusion characteristics"
    decision_reason = f"Diffusion Model synthetic generation detected with {ai_prob:.1%} probability. Key forensic markers: {signals_summary}."
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

if refund_generated:
    status_badge = '<span class="badge-clean">APPROVED (WRONG FOOD CONFIRMED)</span>'
elif "REJECTED" in adjudication["status"]:
    status_badge = f'<span class="badge-reject">{adjudication["status"]}</span>'
else:
    status_badge = '<span class="badge-review">RESOLVED (CORRECT DELIVERY)</span>'

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
        "Evidence Observed from Intake": f"P(AI) = {ai_prob:.1%} | HF Ratio: {forensic_res['spectral_features']['hf_spectral_ratio']:.3f} | Lattice: {forensic_res['spectral_features'].get('hf_peakiness', 0.0):.3f}",
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
    "Live Backend Decision Pipeline Flow",
    "Generative AI vs. Real Detection",
    "Dispute Adjudication & Refund Engine",
])

# =============================================================================
# Tab 0: Live Backend Decision Pipeline Flow (Dynamic Flowchart & Trace)
# =============================================================================
with tabs[0]:
    st.subheader("Live Backend Decision Pipeline Flow & Execution Trace")
    st.markdown(
        "Dynamic algorithmic pipeline graph and stage-by-stage execution trace illustrating the exact inputs, "
        "handout mathematical transformations, and output decisions generated for the uploaded claim."
    )

    # 1. Executive Claim Snapshot
    exec_status_class = "badge-clean" if adjudication_status == "APPROVED_WRONG_ITEM" else "badge-reject"
    st.markdown(
        f"""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px 18px; margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                <div>
                    <span style="font-size:12px; color:#8b949e;">ORDER ID:</span> <b style="color:#f0f6fc;">{order_id}</b> &nbsp;|&nbsp;
                    <span style="font-size:12px; color:#8b949e;">CUSTOMER:</span> <b style="color:#58a6ff;">{cust_profile['customer_name']}</b> ({customer_id}) &nbsp;|&nbsp;
                    <span style="font-size:12px; color:#8b949e;">DELIVERED FOOD:</span> <b style="color:#f0f6fc;">{observed_dish}</b>
                </div>
                <div>
                    <span style="font-size:12px; color:#8b949e;">ADJUDICATION:</span> <span class="{exec_status_class}">{adjudication_status}</span> &nbsp;|&nbsp;
                    <span style="font-size:12px; color:#8b949e;">TOTAL PAYOUT:</span> <b style="color:{'#3fb950' if refund_generated else '#f85149'}; font-size:16px;">₹{total_payout:.2f}</b>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Dynamic Pipeline Flowchart (Input -> Processing -> Output)
    st.markdown("#### End-to-End Decision Flowchart (Input ➔ Backend Processing ➔ Output)")
    pipeline_dot = build_pipeline_graphviz_dot(
        food_title=food_title,
        bill_id=bill_id,
        clean_food_pil=clean_food_pil,
        food_bytes=food_bytes,
        food_tensor_inf=food_tensor_inf,
        forensic_res=forensic_res,
        is_ai=is_ai,
        ai_prob=ai_prob,
        food_phash=food_phash,
        is_duplicate_hash=is_duplicate_hash,
        is_screen_capture=is_screen_capture,
        is_wrong_item=is_wrong_item,
        observed_dish=observed_dish,
        billed_manifest=billed_manifest,
        billed_total=billed_total,
        adjudication_status=adjudication_status,
        total_payout=total_payout,
        customer_id=customer_id,
        trust_score=cust_profile.get("trust_score", 0.85),
    )
    st.graphviz_chart(pipeline_dot, use_container_width=True)
    st.caption("Color Legend: 🟦 Blue = Ingestion & Tensor Standardization | 🟩 Green = Cleared / Authentic / Approved | 🟥 Red = Synthetic AI / Recycled Fraud / Locked | ⬛ Gray = Bypassed Gate")

    st.markdown("---")
    st.markdown("#### Stage-by-Stage Input, Handout Topic Processing & Output Breakdown")

    # Stage 1 Card
    with st.expander("Stage 1: Dual Image Intake & Format Normalization", expanded=True):
        s1_c1, s1_c2, s1_c3 = st.columns([1, 1, 1])
        with s1_c1:
            st.markdown("**📥 Stage Input**")
            st.markdown(f"• **Delivered Food:** `{food_title}`")
            st.markdown(f"• **Food Dimensions:** `{clean_food_pil.size[0]} x {clean_food_pil.size[1]} px`")
            st.markdown(f"• **Food Raw Bytes:** `{len(food_bytes):,} bytes`")
            st.markdown(f"• **Receipt Bill:** `{bill_id}` (`{bill_img.size[0]} x {bill_img.size[1]} px`)")
        with s1_c2:
            st.markdown("**⚙️ Backend Processing (Handout Topic)**")
            st.markdown("• **EXIF Auto-Transpose:** Orientation Tag 274 evaluated to normalize mobile device rotation.")
            st.markdown("• **Channel Canonicalization:** Strips alpha transparencies / CMYK and projects onto standard 3-channel RGB canvas.")
            st.markdown("• **Binary Buffer Staging:** Stores raw memory bytes for downstream hashing.")
        with s1_c3:
            st.markdown("**📤 Stage Output**")
            st.markdown(f"• **Standardized Canvas:** `PIL.Image RGB ({clean_food_pil.size[0]}x{clean_food_pil.size[1]})`")
            st.markdown("• **Staged Ingestion Buffer:** Ready for PyTorch torchvision tensor conversion.")
            st.markdown("• **Gate Status:** <span class='badge-clean'>INGESTED & NORMALIZED</span>", unsafe_allow_html=True)

    # Stage 2 Card
    with st.expander("Stage 2: PyTorch Deep Learning Ingestion (Tensor Standardization)", expanded=True):
        s2_c1, s2_c2, s2_c3 = st.columns([1, 1, 1])
        with s2_c1:
            st.markdown("**📥 Stage Input**")
            st.markdown(f"• **Normalized Canvas:** `RGB Image ({clean_food_pil.size[0]} x {clean_food_pil.size[1]})`")
            st.markdown("• **Target Spatial Resolution:** `224 x 224` pixels")
            st.markdown("• **Target Norm:** ImageNet $\\mu$ and $\\sigma$")
        with s2_c2:
            st.markdown("**⚙️ Backend Processing (Handout Topic)**")
            st.markdown("• `Resize(256)`: Preserves aspect ratio along shortest boundary.")
            st.markdown("• `CenterCrop(224)`: Standardizes spatial tensor geometry.")
            st.markdown("• `ToTensor()`: Converts $[0, 255]$ integer pixels to $[0.0, 1.0]$ float tensors.")
            st.markdown("• `Normalize()`: Channel-wise zero-mean standardization.")
        with s2_c3:
            st.markdown("**📤 Stage Output**")
            st.markdown(f"• **Ingestion Tensor:** `torch.FloatTensor {list(food_tensor_inf.shape)}`")
            st.markdown(f"• **Tensor Value Range:** `[{food_tensor_inf.min():.3f}, {food_tensor_inf.max():.3f}]`")
            st.markdown(f"• **Tensor Dtype:** `{food_tensor_inf.dtype}`")
            st.markdown("• **Gate Status:** <span class='badge-clean'>STANDARDIZED TENSOR</span>", unsafe_allow_html=True)

    # Stage 3 Card
    with st.expander("Stage 3: Generative AI vs. Real Image Forensic Detection Engine", expanded=True):
        s3_c1, s3_c2, s3_c3 = st.columns([1, 1, 1])
        with s3_c1:
            st.markdown("**📥 Stage Input**")
            st.markdown("• **PyTorch Tensor:** `[3, 224, 224]`")
            st.markdown("• **Grayscale 2D Matrix:** Spatial frequency analysis")
            st.markdown("• **HSV Chroma Plane:** Color saturation analysis")
            st.markdown("• **EXIF Metadata Table:** Hardware provenance tags")
        with s3_c2:
            st.markdown("**⚙️ Backend Processing (Handout Topic)**")
            spec = forensic_res["spectral_features"]
            ex = forensic_res["exif_provenance"]
            st.markdown(f"• **2D FFT Nyquist Ring ($r \\ge 0.55$):** Ratio = `{spec['hf_spectral_ratio']:.3f}`")
            st.markdown(f"• **VAE Lattice Peakiness:** `{spec.get('hf_peakiness', 0.0):.3f}` (Threshold: 0.096)")
            st.markdown(f"• **Flat-Region Sensor Noise (PRNU):** `{spec.get('flat_noise_variance', 0.0):.2f}` Var")
            st.markdown(f"• **Studio Saturation:** `{spec.get('saturation_mean', 0.0):.1f}` | Corr: `{spec.get('corr_rg', 0.0):.2f}`")
            st.markdown(f"• **Hardware EXIF:** `{ex.get('make') or 'None'} {ex.get('model') or ''}`")
        with s3_c3:
            st.markdown("**📤 Stage Output**")
            st.markdown(f"• **Synthetic Probability $P(\\text{{AI}}):$** `{ai_prob:.1%}`")
            st.markdown(f"• **Confidence:** `{forensic_res['authenticity_confidence']:.1%}`")
            if is_ai:
                st.markdown("• **Forensic Verdict:** <span class='badge-reject'>🚨 SYNTHETIC AI GENERATION</span>", unsafe_allow_html=True)
                st.markdown("• **Gating Action:** Hard Reject (Lock Dispute to ₹0.00)")
            else:
                st.markdown("• **Forensic Verdict:** <span class='badge-clean'>✅ AUTHENTIC REAL CAMERA</span>", unsafe_allow_html=True)
                st.markdown("• **Gating Action:** Cleared to Multimodal Receipt Gate")

    # Stage 4 Card
    with st.expander("Stage 4: Duplicate Hash & Tampering Integrity Audit", expanded=True):
        s4_c1, s4_c2, s4_c3 = st.columns([1, 1, 1])
        with s4_c1:
            st.markdown("**📥 Stage Input**")
            st.markdown(f"• **Image Stream:** `{len(food_bytes):,} bytes`")
            st.markdown(f"• **Device Fingerprint:** `{device_fp}`")
            st.markdown(f"• **Customer Account ID:** `{customer_id}`")
        with s4_c2:
            st.markdown("**⚙️ Backend Processing (Handout Topic)**")
            st.markdown("• **64-bit DCT pHash:** Low-frequency discrete cosine transform hashing.")
            st.markdown(r"• **Hamming Distance Match:** Compares against fraud hash registry (Threshold $\le 10$).")
            st.markdown("• **Screen Capture Edge Filter:** Analyzes image border pixel uniformity.")
            st.markdown("• **Device Sybil Detection:** Checks multi-account linkage on device ID.")
        with s4_c3:
            st.markdown("**📤 Stage Output**")
            st.markdown(f"• **Perceptual Hash:** `{food_phash[:16]}...`")
            st.markdown(f"• **Registry Match:** {'❌ BLACKLISTED' if is_duplicate_hash else '✅ UNIQUE / CLEAN'}")
            st.markdown(f"• **Screen Capture:** {'❌ SCREENSHOT' if is_screen_capture else '✅ DIRECT PHOTO'}")
            st.markdown(f"• **Device Collusion:** {'❌ SYBIL RING' if is_multi_account_fraud else '✅ SINGLE ACCOUNT'}")

    # Stage 5 Card
    with st.expander("Stage 5: Multimodal Receipt Cross-Verification (Discrepancy Gate)", expanded=True):
        s5_c1, s5_c2, s5_c3 = st.columns([1, 1, 1])
        with s5_c1:
            st.markdown("**📥 Stage Input**")
            st.markdown(f"• **Observed Food Dish:** `{observed_dish}`")
            st.markdown(f"• **Billed Receipt:** `{bill_id}`")
            st.markdown(f"• **Billed Items:** `{', '.join(it['name'] for it in billed_manifest)}`")
            st.markdown(f"• **Billed Item Total:** `₹{billed_total:.2f}`")
        with s5_c2:
            st.markdown("**⚙️ Backend Processing (Handout Topic)**")
            if is_ai:
                st.markdown("• **GATING LOCK:** Stage was bypassed and locked because Stage 3 identified synthetic image fraud.")
                st.markdown("• **Policy Rule:** Automated verification halts before receipt verification to prevent fraud payouts.")
            else:
                st.markdown("• **Multimodal Cross-Verification:** Compares observed dish against billed items manifest.")
                st.markdown("• **Discrepancy Formula:** `is_wrong_item = observed_dish not in billed_manifest`.")
                st.markdown("• **Financial Entitlement:** Wrong delivery entitles 100% item refund + ₹49 delivery waiver.")
        with s5_c3:
            st.markdown("**📤 Stage Output**")
            if is_ai:
                st.markdown("• **Discrepancy Status:** <span class='badge-review'>BYPASSED / LOCKED</span>", unsafe_allow_html=True)
                st.markdown("• **Entitlement:** ₹0.00 (Fraud Lock)")
            elif is_wrong_item:
                st.markdown("• **Discrepancy Status:** <span class='badge-clean'>WRONG FOOD CONFIRMED</span>", unsafe_allow_html=True)
                st.markdown(f"• **Entitlement:** Full Item Refund (₹{billed_total:.2f}) + Delivery Waiver (₹{delivery_fee:.2f})")
            else:
                st.markdown("• **Discrepancy Status:** <span class='badge-review'>ORDER MATCHED (NO ERROR)</span>", unsafe_allow_html=True)
                st.markdown("• **Entitlement:** ₹0.00 (Order fulfilled correctly)")

    # Stage 6 Card
    with st.expander("Stage 6: Final Adjudication Engine & Financial Settlement", expanded=True):
        s6_c1, s6_c2, s6_c3 = st.columns([1, 1, 1])
        with s6_c1:
            st.markdown("**📥 Stage Input**")
            st.markdown(f"• **Gated Forensic Verdict:** {'SYNTHETIC' if is_ai else 'AUTHENTIC'}")
            st.markdown(f"• **Receipt Discrepancy:** {'CONFIRMED WRONG' if is_wrong_item else 'MATCHED'}")
            st.markdown(f"• **Customer Trust Score:** `{cust_profile.get('trust_score', 0.85):.0%}` ({cust_profile.get('trust_tier', 'Verified')})")
        with s6_c2:
            st.markdown("**⚙️ Backend Processing (Handout Topic)**")
            st.markdown("• **Automated Dispute Resolution Policy Tree:**")
            st.markdown("  - *Synthetic or Blacklisted:* Hard Reject (₹0.00), log to suspicious activity, penalize trust score.")
            st.markdown("  - *Authentic Camera + Wrong Item:* Approve 100% item refund + 100% delivery waiver.")
            st.markdown("  - *Authentic Camera + Correct Item:* Resolve dispute with ₹0.00 (Order delivered intact).")
            st.markdown("• **Ledger Mutation:** Updates customer order history and active dispute record in database.")
        with s6_c3:
            st.markdown("**📤 Stage Output**")
            st.markdown(f"• **Final Status:** `{adjudication_status}`")
            st.markdown(f"• **Item Refund Approved:** `₹{item_refund:.2f}`")
            st.markdown(f"• **Delivery Waiver Approved:** `₹{waiver:.2f}`")
            st.markdown(f"• **Total Customer Payout:** `₹{total_payout:.2f}`")
            st.markdown(f"• **Executive Decision Reason:** *\"{decision_reason}\"*")

    with st.expander("View Raw Graphviz DOT Code"):
        st.code(pipeline_dot, language="dot")


# =============================================================================
# Tab 1: Generative AI vs. Real Image Detection
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

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        st.metric(
            "Nyquist HF Spectral Ratio",
            f"{forensic_res['spectral_features']['hf_spectral_ratio']:.3f}",
            delta="Spectral Deficient" if forensic_res['spectral_features']['hf_spectral_ratio'] < 0.69 else "Sustained Spectrum",
            delta_color="inverse" if forensic_res['spectral_features']['hf_spectral_ratio'] < 0.69 else "normal",
        )
    with f_col2:
        pk = forensic_res['spectral_features'].get('hf_peakiness', 0.0)
        st.metric(
            "HF Lattice Peakiness",
            f"{pk:.3f}",
            delta="Lattice Spikes (Diffusion VAE)" if pk > 0.096 else "Continuous Scatter",
            delta_color="inverse" if pk > 0.096 else "normal",
        )
    with f_col3:
        fn = forensic_res['spectral_features'].get('flat_noise_variance', 0.0)
        st.metric(
            "Flat-Region Shot Noise",
            f"{fn:.2f}",
            delta="Hypersmooth Synthetic" if fn < 0.10 else "Physical CMOS PRNU",
            delta_color="inverse" if fn < 0.10 else "normal",
        )
    with f_col4:
        sat = forensic_res['spectral_features'].get('saturation_mean', 0.0)
        st.metric(
            "Studio Glaze Saturation",
            f"{sat:.1f}",
            delta="Artificial Studio Sat" if sat > 120.0 else "Natural Daylight",
            delta_color="inverse" if sat > 120.0 else "normal",
        )

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
# Tab 2: Dispute Adjudication & Refund Engine (Wrong Food Delivery Case)
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
