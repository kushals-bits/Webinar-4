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
from app.database import mock_db

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
    food_title = "Delivered Food"
    bill_title = "Printed Bill"

    if intake_source == "Curated Real & AI Samples":
        sel_food = st.selectbox("1. Delivered Food Photo:", list(FOOD_OPTIONS.keys()))
        food_title = sel_food.split(":")[1].strip() if ":" in sel_food else sel_food
        p_food = os.path.join(SAMPLE_DIR, FOOD_OPTIONS[sel_food])
        if os.path.exists(p_food):
            food_img = Image.open(p_food)

        sel_bill = st.selectbox("2. Printed Bill Receipt:", list(BILL_OPTIONS.keys()))
        bill_title = sel_bill
        p_bill = os.path.join(SAMPLE_DIR, BILL_OPTIONS[sel_bill])
        if os.path.exists(p_bill):
            bill_img = Image.open(p_bill)
    else:
        up_food = st.file_uploader("1. Upload Delivered Food Photo:", type=["jpg", "jpeg", "png", "webp", "avif"])
        if up_food:
            food_img = Image.open(up_food)
            food_title = up_food.name

        up_bill = st.file_uploader("2. Upload Printed Bill Receipt:", type=["jpg", "jpeg", "png", "webp"])
        if up_bill:
            bill_img = Image.open(up_bill)
            bill_title = up_bill.name
        elif os.path.exists(os.path.join(SAMPLE_DIR, "delivery_receipt_bill.jpg")):
            bill_img = Image.open(os.path.join(SAMPLE_DIR, "delivery_receipt_bill.jpg"))
            bill_title = "Default Receipt"

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
    else:
        order_id = st.text_input("Order ID:", value="ORD-2026-904")
        bill_id = st.text_input("Bill ID:", value="BILL-2026-9041")
        order_date = "2026-11-09T18:00:00+05:30"
        order_comment = "Dispute filed for wrong item delivered."

    # Customer trust & dispute badge
    score_col = "#3fb950" if cust_profile["trust_score"] > 0.8 else ("#d29922" if cust_profile["trust_score"] > 0.5 else "#f85149")
    st.markdown(
        f"""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:6px; padding:10px; margin-top:8px;">
            <div style="font-size:12px; color:#8b949e;">Customer: <b style="color:#f0f6fc;">{customer_name}</b> ({customer_id})</div>
            <div style="font-size:12px; color:#8b949e;">Bill ID: <b style="color:#58a6ff;">{bill_id}</b></div>
            <div style="font-size:12px; color:#8b949e;">Order Date: <span>{order_date}</span></div>
            <div style="font-size:12px; color:#8b949e;">Trust Score: <b style="color:{score_col};">{cust_profile['trust_score']:.0%}</b> ({cust_profile['trust_tier']})</div>
            <div style="font-size:11px; color:#8b949e; margin-top:4px;"><i>"{cust_profile.get('comment', '')}"</i></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    customer_claim_text = st.text_area(
        "Customer Claim / Notes:",
        value=order_comment or "I ordered Butter Chicken and Garlic Naan, but the delivery bag contained something completely different! Please refund.",
    )

if food_img is None or bill_img is None:
    st.info("Select or upload both the Delivered Food Photo and the Printed Bill Receipt to begin dispute adjudication.")
    st.stop()

# -----------------------------------------------------------------------------
# Adjudication Pipeline Execution
# -----------------------------------------------------------------------------

# Step 1: Detect AI vs. Real image
forensic_res = classify_ai_vs_real_image(food_img)
is_ai = forensic_res["is_ai_generated"]
ai_prob = forensic_res["ai_probability"]

# Order Manifest (What the restaurant bill says was charged)
billed_manifest = [
    {"name": "Butter Chicken", "qty": 2, "unit_price": 320.0, "total": 640.0},
    {"name": "Garlic Naan", "qty": 4, "unit_price": 55.0, "total": 220.0},
    {"name": "Dal Makhani", "qty": 1, "unit_price": 220.0, "total": 220.0},
]
billed_total = sum(i["total"] for i in billed_manifest)
delivery_fee = 49.0

# Observed delivered dish
observed_dish = food_title.split("(")[0].strip()

# Run full dispute adjudication
adjudication = adjudicate_wrong_food_dispute(
    order_image=food_img,
    bill_image=bill_img,
    order_id=order_id,
    customer_id=customer_id,
    billed_items=billed_manifest,
    reported_dish_observed=observed_dish,
    delivery_fee=delivery_fee,
)

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
    st.markdown("### Automated Adjudication Outcome")

    if adjudication["status"] == "REJECTED_SYNTHETIC_FRAUD":
        st.error(
            "🚫 **CLAIM HARD REJECTED (AI-GENERATED IMAGE DETECTED)**\n\n"
            f"**Reason:** {adjudication['decision_reason']}\n\n"
            "• **Refund Approved:** ₹0.00\n"
            "• **Delivery Fee Waiver:** ₹0.00\n"
            "• **Total Payout Issued:** **₹0.00**\n"
            "• **Security Action:** Order locked, claim logged to `suspicious_activity_log`."
        )
        st.json(adjudication)

    elif adjudication["status"] == "APPROVED_WRONG_ITEM":
        st.success(
            "✅ **AUTHENTIC REAL IMAGE CONFIRMED & WRONG FOOD VERIFIED**\n\n"
            f"**Reason:** {adjudication['decision_reason']}\n\n"
            f"• **Food Item Refund:** ₹{adjudication['refund_amount_rs']:.2f} (100% reimbursement)\n"
            f"• **Delivery Fee Waiver:** ₹{adjudication['delivery_waiver_rs']:.2f} (Full ₹49 waiver)\n"
            f"• **TOTAL REFUND APPROVED:** **₹{adjudication['total_payout_rs']:.2f}**\n\n"
            "• **Action:** Immediate credit to original payment method. Apology notification dispatched."
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
            f"**Reason:** {adjudication['decision_reason']}\n\n"
            "• **Total Refund:** ₹0.00 (Customer received the exact meal billed on receipt)."
        )
        st.json(adjudication)


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
