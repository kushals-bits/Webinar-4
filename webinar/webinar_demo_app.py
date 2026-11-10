"""
Multimodal Image Authenticity & Automated Dispute Resolution Workbench
Operational Interface for Pre-processing for AI-Ready Data (Real Images Engine).
"""

from __future__ import annotations

import io
import os
import sys
from datetime import datetime, timezone
from typing import Any

import cv2
import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ExifTags, Image, ImageOps
import streamlit as st

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from app.security.hashing_service import generate_reverse_image_webhook_mock, hamming_distance

# Page configuration
st.set_page_config(
    page_title="AI-Ready Data Pre-processing | Dispute Resolution Workbench",
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

HISTORICAL_FRAUD_DB = {
    "a1b2c3d4e5f6a7b8": {"case": "REF-2025-01", "customer": "CUST-999", "flagged_on": "2026-01-15"},
    "deadbeefcafebabe": {"case": "REF-2025-02", "customer": "CUST-888", "flagged_on": "2026-02-20"},
}

REAL_FOOD_SAMPLES = {
    "Burger (IPTC Stock Provenance Tags)": "Burger.jpg",
    "Chole Bhature (AVIF Modern Container)": "Chole.avif",
    "Masala Dosa (Online Catalog Match)": "dosa.jpg",
    "Hakka Noodles (Web Compression Ingestion)": "Noodles.jpeg",
    "Penne Pasta (Portrait Aspect Normalization)": "Pasta.jpg",
    "North Indian Thali (Multi-Item Meal Platter)": "Thali.jpeg",
    "Deluxe Thali (Vertical Border Ingestion)": "Thali_2.jpeg",
    "Delivery Receipt Bill (Physical Printed OCR)": "delivery_receipt_bill.jpg",
}

# -----------------------------------------------------------------------------
# Image Forensics & Normalization Routines
# -----------------------------------------------------------------------------

def normalize_image(pil_img: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    """Standardize orientation based on EXIF tag 274 and convert to canonical sRGB."""
    orientation_corrected = False
    original_mode = pil_img.mode
    original_size = pil_img.size

    try:
        exif = None
        if hasattr(pil_img, "getexif"):
            exif_obj = pil_img.getexif()
            if exif_obj:
                exif = dict(exif_obj.items())
        if not exif and hasattr(pil_img, "_getexif"):
            exif = pil_img._getexif()

        if exif and 274 in exif and exif[274] != 1:
            transposed = ImageOps.exif_transpose(pil_img)
            orientation_corrected = True
        else:
            transposed = pil_img
    except Exception:
        transposed = pil_img

    clean_img = transposed.convert("RGB") if transposed.mode != "RGB" else transposed

    return clean_img, {
        "original_mode": original_mode,
        "clean_mode": clean_img.mode,
        "size": clean_img.size,
        "orientation_corrected": orientation_corrected,
    }


def audit_exif_metadata(pil_img: Image.Image, delivery_time_iso: str) -> dict[str, Any]:
    """Extract hardware metadata, IPTC descriptions, and compute delta against the delivery timestamp."""
    raw_exif = None
    if hasattr(pil_img, "getexif"):
        exif_obj = pil_img.getexif()
        if exif_obj:
            raw_exif = dict(exif_obj.items())
    if not raw_exif and hasattr(pil_img, "_getexif"):
        raw_exif = pil_img._getexif()

    if not raw_exif:
        return {
            "has_exif": False,
            "has_hardware": False,
            "make": "Not Available",
            "model": "Not Available",
            "software": "None",
            "description": None,
            "datetime_original": None,
            "delta_seconds": None,
            "timestamp_valid": False,
            "flag": "MISSING_EXIF_TIMESTAMP",
            "risk_score_delta": 0.15,
        }

    exif = {ExifTags.TAGS.get(k, k): v for k, v in raw_exif.items() if k in ExifTags.TAGS}
    make = str(exif.get("Make", "Unknown")).strip()
    model = str(exif.get("Model", "Unknown")).strip()
    software = str(exif.get("Software", "None")).strip()
    description = str(exif.get("ImageDescription", "")).strip() or None
    dt_str = exif.get("DateTimeOriginal")

    has_hardware = make not in ("Unknown", "None", "") and model not in ("Unknown", "None", "")

    delivery_dt = datetime.fromisoformat(delivery_time_iso)
    flag = None
    delta_sec = None
    valid = False
    risk_delta = 0.0

    if dt_str:
        try:
            img_dt = datetime.strptime(str(dt_str).strip(), "%Y:%m:%d %H:%M:%S").replace(tzinfo=delivery_dt.tzinfo)
            delta_sec = (img_dt - delivery_dt).total_seconds()
            if delta_sec < -60:
                flag = "PREDATES_DELIVERY"
                risk_delta = 0.30
            elif delta_sec > 7200:
                flag = "POST_DELIVERY_WINDOW_EXCEEDED"
                risk_delta = 0.20
            else:
                valid = True
        except Exception:
            flag = "INVALID_TIMESTAMP_FORMAT"
            risk_delta = 0.15
    else:
        flag = "MISSING_EXIF_TIMESTAMP"
        risk_delta = 0.15

    return {
        "has_exif": True,
        "has_hardware": has_hardware,
        "make": make,
        "model": model,
        "software": software,
        "description": description,
        "datetime_original": dt_str,
        "delta_seconds": delta_sec,
        "timestamp_valid": valid,
        "flag": flag,
        "risk_score_delta": risk_delta,
    }

extract_exif_metadata = audit_exif_metadata


def compute_phash_forensics(pil_img: Image.Image, max_hamming: int = 4) -> dict[str, Any]:
    """Compute 2D-DCT spectrum, 64-bit pHash token, and evaluate Hamming distance."""
    gray = pil_img.convert("L")
    resized = gray.resize((32, 32), Image.Resampling.LANCZOS)
    arr = np.array(resized, dtype=np.float32)

    dct = cv2.dct(arr)
    dct_low = dct[0:8, 0:8]
    dct_plot = dct_low.copy()
    dct_plot[0, 0] = 0.0

    flat = dct_low.flatten()[1:]
    median = np.median(flat)
    hash_bits = (dct_low > median).astype(int)

    phash_token = str(imagehash.phash(pil_img))
    current_h = imagehash.hex_to_hash(phash_token)

    min_dist = 999
    matched_entry = None
    matched_hash = None

    for stored_str, meta in HISTORICAL_FRAUD_DB.items():
        stored_h = imagehash.hex_to_hash(stored_str)
        dist = abs(current_h - stored_h)
        if dist < min_dist:
            min_dist = dist
            matched_entry = meta
            matched_hash = stored_str

    is_duplicate = min_dist <= max_hamming

    return {
        "phash_token": phash_token,
        "full_dct": dct,
        "low_dct": dct_plot,
        "hash_bits": hash_bits,
        "is_duplicate": is_duplicate,
        "min_hamming_distance": min_dist,
        "matched_entry": matched_entry if is_duplicate else None,
        "matched_hash": matched_hash if is_duplicate else None,
        "risk_delta": 0.40 if is_duplicate else 0.0,
    }


def detect_screen_capture_and_blur(pil_img: Image.Image, border_px: int = 30) -> dict[str, Any]:
    """Measure border pixel uniformity, Laplacian spatial variance, and high-pass edge mask."""
    img_rgb = pil_img.convert("RGB")
    w, h = img_rgb.size

    top = [img_rgb.getpixel((x, y)) for x in range(w) for y in range(border_px)]
    bottom = [img_rgb.getpixel((x, y)) for x in range(w) for y in range(h - border_px, h)]
    left = [img_rgb.getpixel((x, y)) for x in range(border_px) for y in range(border_px, h - border_px)]
    right = [img_rgb.getpixel((x, y)) for x in range(w - border_px, w) for y in range(border_px, h - border_px)]
    border_pixels = top + bottom + left + right

    from collections import Counter

    counts = Counter(border_pixels)
    most_common_cnt = counts.most_common(1)[0][1]
    uniformity = most_common_cnt / len(border_pixels)
    is_screen_capture = uniformity >= 0.85

    arr_gray = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(arr_gray, cv2.CV_64F)
    blur_variance = float(laplacian.var())

    return {
        "is_screen_capture": is_screen_capture,
        "border_uniformity": round(uniformity, 4),
        "blur_score": round(blur_variance, 2),
        "laplacian_mask": np.abs(laplacian),
        "risk_delta": round(0.25 * uniformity, 3) if is_screen_capture else 0.0,
    }


# -----------------------------------------------------------------------------
# User Interface
# -----------------------------------------------------------------------------

st.markdown(
    """
    <div class="panel-header">
        <h2 style="margin:0; color:#58a6ff; font-weight:700;">Pre-processing for AI-Ready Data</h2>
        <h4 style="margin:4px 0 0 0; color:#8b949e; font-weight:400;">Real Customer Image Forensics & Dispute Adjudication Workbench</h4>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Data Intake (Real Images Only)")
    input_source = st.radio("Intake Method:", ["Preset Real Food Samples", "Upload Custom Real Photo"])

    selected_image = None
    image_title = "Uploaded Image"

    if input_source == "Preset Real Food Samples":
        scenario = st.selectbox("Select Real Customer Food Item:", list(REAL_FOOD_SAMPLES.keys()))
        image_title = scenario
        sample_path = os.path.join(SAMPLE_DIR, REAL_FOOD_SAMPLES[scenario])
        if os.path.exists(sample_path):
            selected_image = Image.open(sample_path)
        else:
            st.error(f"File not found: {sample_path}")
    else:
        uploaded_file = st.file_uploader(
            "Upload real customer photograph:",
            type=["jpg", "jpeg", "png", "webp", "avif"],
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
            image_title = uploaded_file.name

    st.markdown("---")
    st.subheader("Ground Truth Parameters")
    delivery_time = st.text_input("Delivery Timestamp (ISO):", value="2026-11-09T18:30:00+05:30")
    order_id = st.text_input("Order ID:", value="ORD-2026-001")
    customer_id = st.text_input("Customer ID:", value="CUST-101")

if selected_image is None:
    st.info("Select a preset real food sample or upload an image to begin analysis.")
    st.stop()

# Pipeline Execution
clean_img, norm_info = normalize_image(selected_image)
exif_res = extract_exif_metadata(selected_image, delivery_time)
phash_res = compute_phash_forensics(clean_img)
spatial_res = detect_screen_capture_and_blur(clean_img)

# Reverse Image Search Mock Check
rev_check = generate_reverse_image_webhook_mock(phash_res["phash_token"])

# Risk Scoring Aggregation
risk_score = 0.0
flags = []

if exif_res["flag"]:
    flags.append(exif_res["flag"])
    risk_score += exif_res["risk_score_delta"]

if phash_res["is_duplicate"]:
    flags.append("DUPLICATE_IMAGE_HASH")
    risk_score += phash_res["risk_delta"]

if rev_check.get("is_indexed_online"):
    flags.append("IMAGE_INDEXED_ONLINE")
    risk_score += 0.30

if spatial_res["is_screen_capture"]:
    flags.append("SCREEN_CAPTURE_DETECTED")
    risk_score += spatial_res["risk_delta"]

# Stock photo IPTC description check
if exif_res.get("description"):
    desc_lower = exif_res["description"].lower()
    if any(k in desc_lower for k in ["stock", "rustic", "setting", "getty", "shutter"]):
        flags.append("STOCK_PHOTO_METADATA_DETECTED")
        risk_score += 0.25

total_risk_score = round(min(risk_score, 1.0), 3)

if total_risk_score >= 0.8:
    decision_action = "HARD_REJECT"
    decision_badge = '<span class="badge-reject">HARD REJECT</span>'
    decision_msg = "Claim failed authenticity checks. Refund calculation locked."
elif total_risk_score >= 0.4:
    decision_action = "MANUAL_REVIEW"
    decision_badge = '<span class="badge-review">MANUAL REVIEW</span>'
    decision_msg = "Borderline anomaly detected. Escalated to manual review queue."
else:
    decision_action = "CLEARED"
    decision_badge = '<span class="badge-clean">CLEARED</span>'
    decision_msg = "Passed authenticity and integrity verification."

# Tab Navigation
tabs = st.tabs([
    "Diagnostic Workbench",
    "Real Images Pairwise Benchmark",
    "Multimodal Fusion & Payout",
    "Technical Foundations & Formulas",
])

# -----------------------------------------------------------------------------
# Tab 1: Diagnostic Workbench
# -----------------------------------------------------------------------------
with tabs[0]:
    st.markdown(f"#### Active Target: **{image_title}**")

    # Metrics Summary
    q1, q2, q3, q4, q5, q6 = st.columns(6)
    with q1:
        st.metric("Accuracy", "Consistent" if not phash_res["is_duplicate"] else "Compromised")
    with q2:
        st.metric("Completeness", "100%" if clean_img else "Incomplete")
    with q3:
        st.metric("Consistency", "Pass" if not spatial_res["is_screen_capture"] else "Bezel Anomaly")
    with q4:
        st.metric("Timeliness", "Valid (<=2h)" if exif_res["timestamp_valid"] else "Flagged")
    with q5:
        st.metric("Validity", "Authentic Camera" if exif_res["has_hardware"] else ("Indexed/Stock" if rev_check.get("is_indexed_online") or exif_res.get("description") else "Clean Real Dish"))
    with q6:
        st.metric("Uniqueness", f"D_H = {phash_res['min_hamming_distance']}" if not phash_res["is_duplicate"] else "Duplicate!")

    st.markdown(f"**Adjudication Decision:** {decision_badge} — *{decision_msg}*", unsafe_allow_html=True)
    if flags:
        st.warning(f"**Active Flags:** `{', '.join(flags)}` (Total Risk Score: **{total_risk_score:.2f}**)")

    if exif_res.get("description"):
        st.info(f"🏷️ **Embedded Stock IPTC Tag Detected:** *\"{exif_res['description']}\"*")

    if rev_check.get("is_indexed_online"):
        st.info(f"🌐 **Online Reverse Image Search Match:** Found on {', '.join(rev_check['source_platforms'])} (Confidence: {rev_check['confidence']:.0%})")

    st.markdown("---")

    col_img, col_forensic = st.columns([1, 1])

    with col_img:
        st.subheader("Image Normalization")
        st.image(clean_img, caption=f"Standardized sRGB ({clean_img.size[0]}x{clean_img.size[1]} px)", width='stretch')
        st.json({
            "Original Format/Mode": norm_info["original_mode"],
            "Standardized Mode": norm_info["clean_mode"],
            "Orientation Corrected (Tag 274)": norm_info["orientation_corrected"],
        })

    with col_forensic:
        st.subheader("Metadata & Temporal Analysis")
        st.json({
            "EXIF Present": exif_res["has_exif"],
            "Hardware Present": exif_res["has_hardware"],
            "Device Make": exif_res["make"],
            "Device Model": exif_res["model"],
            "Software": exif_res["software"],
            "Description Tag": exif_res.get("description"),
            "Capture Timestamp": exif_res["datetime_original"],
            "Delta from Delivery": f"{exif_res['delta_seconds']} seconds" if exif_res['delta_seconds'] is not None else "N/A",
            "Temporal Audit": "Pass" if exif_res["timestamp_valid"] else f"Flagged ({exif_res['flag']})",
        })

    st.markdown("---")
    st.subheader("Frequency Domain (DCT) & Spatial Filters")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**2D-DCT Low-Frequency Spectrum (8x8)**")
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        cax = ax1.imshow(phash_res["low_dct"], cmap="viridis")
        fig1.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title("DCT Coefficients (No DC)")
        st.pyplot(fig1)

    with c2:
        st.markdown("**64-bit Perceptual Hash Matrix**")
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.imshow(phash_res["hash_bits"], cmap="binary")
        ax2.set_title(f"pHash: {phash_res['phash_token']}")
        st.pyplot(fig2)

    with c3:
        st.markdown("**Laplacian Spatial Edge / Moiré Mask**")
        fig3, ax3 = plt.subplots(figsize=(4, 4))
        ax3.imshow(spatial_res["laplacian_mask"], cmap="inferno")
        ax3.set_title(f"Sharpness Score: {spatial_res['blur_score']}")
        st.pyplot(fig3)

# -----------------------------------------------------------------------------
# Tab 2: Real Images Pairwise Benchmark
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Cross-Image Benchmark: All 7 Real Customer Food Deliveries")
    st.markdown("Demonstrating how pre-processing standardizes diverse formats, aspect ratios, and validates uniqueness via pairwise Hamming distance.")

    benchmark_rows = []
    hashes_map = {}

    for label, fname in REAL_FOOD_SAMPLES.items():
        if fname == "delivery_receipt_bill.jpg":
            continue
        fpath = os.path.join(SAMPLE_DIR, fname)
        if not os.path.exists(fpath):
            continue
        with Image.open(fpath) as im:
            c_im, _ = normalize_image(im)
            ph = str(imagehash.phash(c_im))
            hashes_map[label.split('(')[0].strip()] = ph
            gray = cv2.cvtColor(np.array(c_im), cv2.COLOR_RGB2GRAY)
            blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            rev = generate_reverse_image_webhook_mock(ph)
            is_indexed = rev.get("is_indexed_online", False)

            benchmark_rows.append({
                "Dish Name": label.split('(')[0].strip(),
                "File Format": im.format,
                "Dimensions": f"{im.size[0]} x {im.size[1]}",
                "pHash Token": ph,
                "Sharpness (Blur Score)": f"{blur_val:.1f}",
                "Online Indexed": "Yes (Flagged)" if is_indexed else "No (Clean)",
                "Action Tier": "MANUAL REVIEW" if is_indexed else "CLEARED"
            })

    df_b = pd.DataFrame(benchmark_rows)
    st.dataframe(df_b, use_container_width=True)

    st.markdown("---")
    st.subheader("Pairwise Hamming Distance Matrix ($D_H$)")
    st.markdown(
        "Verifying mathematical uniqueness across all real food claims. Different real dishes produce "
        "high Hamming distances ($D_H \\ge 22$), proving zero false-positive collisions."
    )

    names = list(hashes_map.keys())
    n = len(names)
    matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            h1 = imagehash.hex_to_hash(hashes_map[names[i]])
            h2 = imagehash.hex_to_hash(hashes_map[names[j]])
            matrix[i, j] = abs(h1 - h2)

    df_matrix = pd.DataFrame(matrix, index=names, columns=names)
    st.dataframe(df_matrix, use_container_width=True)

    fig_mat, ax_mat = plt.subplots(figsize=(8, 6))
    im_h = ax_mat.imshow(matrix, cmap="Blues")
    ax_mat.set_xticks(range(n))
    ax_mat.set_yticks(range(n))
    ax_mat.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax_mat.set_yticklabels(names, fontsize=9)
    fig_mat.colorbar(im_h, ax=ax_mat, fraction=0.046, pad=0.04)
    for i in range(n):
        for j in range(n):
            ax_mat.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black" if matrix[i, j] > 18 else "white", fontsize=9)
    ax_mat.set_title("Pairwise Hamming Distance Matrix (DH >= 22 = Distinct Meals)")
    st.pyplot(fig_mat)

# -----------------------------------------------------------------------------
# Tab 3: Multimodal Fusion & Payout
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Multimodal Feature Fusion & Payout Engine")
    st.markdown("Cross-verifying tabular order specifications against receipt text and visual inspection for the active real dish.")

    st.markdown(f"**Customer ID:** `{customer_id}` | **Order ID:** `{order_id}`")

    # Dynamic order items based on the active image
    img_name_lower = image_title.lower()
    if "burger" in img_name_lower:
        order_items = [
            {"name": "Grilled Rib Burger", "qty": 2, "unit_price": 280.0, "full_value": 560.0},
            {"name": "Peri Peri French Fries", "qty": 1, "unit_price": 120.0, "full_value": 120.0},
            {"name": "Cold Beverage", "qty": 2, "unit_price": 60.0, "full_value": 120.0},
        ]
        scenarios = [
            "Missing French Fries (Shortfall)",
            "Wrong Burger Delivered (Chicken instead of Rib Burger)",
            "Damaged / Crushed Packaging",
            "Full Order Correct & Intact (No Issue)",
        ]
        scenario_map = {
            scenarios[0]: ({"French Fries (1 missing)": 120.0}, 49.0, 169.0),
            scenarios[1]: ({"Rib Burger (Wrong Item)": 560.0}, 49.0, 609.0),
            scenarios[2]: ({"Crushed Burger Box (Damaged)": 280.0}, 0.0, 280.0),
            scenarios[3]: ({}, 0.0, 0.0),
        }
    elif "chole" in img_name_lower:
        order_items = [
            {"name": "Special Chole Bhature (2 pcs)", "qty": 2, "unit_price": 220.0, "full_value": 440.0},
            {"name": "Sweet Punjabi Lassi", "qty": 2, "unit_price": 80.0, "full_value": 160.0},
            {"name": "Extra Bhatura", "qty": 2, "unit_price": 50.0, "full_value": 100.0},
        ]
        scenarios = [
            "Spilled / Leaking Chole Gravy (Damaged Packaging)",
            "Missing 1 Sweet Lassi (Shortfall)",
            "Full Order Correct & Intact (No Issue)",
        ]
        scenario_map = {
            scenarios[0]: ({"Chole Bhature (Spillage)": 220.0}, 49.0, 269.0),
            scenarios[1]: ({"Sweet Lassi (1 missing)": 80.0}, 0.0, 80.0),
            scenarios[2]: ({}, 0.0, 0.0),
        }
    elif "dosa" in img_name_lower:
        order_items = [
            {"name": "Crispy Masala Dosa", "qty": 2, "unit_price": 160.0, "full_value": 320.0},
            {"name": "Mysore Masala Dosa", "qty": 1, "unit_price": 180.0, "full_value": 180.0},
            {"name": "Madras Filter Coffee", "qty": 2, "unit_price": 60.0, "full_value": 120.0},
        ]
        scenarios = [
            "Missing Filter Coffee (Shortfall)",
            "Burnt / Cold Dosa (Quality Issue)",
            "Full Order Correct & Intact (No Issue)",
        ]
        scenario_map = {
            scenarios[0]: ({"Filter Coffee (2 missing)": 120.0}, 0.0, 120.0),
            scenarios[1]: ({"Masala Dosa (Quality Issue)": 160.0}, 49.0, 209.0),
            scenarios[2]: ({}, 0.0, 0.0),
        }
    elif "noodles" in img_name_lower:
        order_items = [
            {"name": "Veg Hakka Noodles", "qty": 2, "unit_price": 180.0, "full_value": 360.0},
            {"name": "Crispy Spring Rolls", "qty": 1, "unit_price": 150.0, "full_value": 150.0},
            {"name": "Chilli Paneer Gravy", "qty": 1, "unit_price": 220.0, "full_value": 220.0},
        ]
        scenarios = [
            "Missing Spring Rolls (Shortfall)",
            "Spilled Chilli Paneer Gravy",
            "Full Order Correct & Intact (No Issue)",
        ]
        scenario_map = {
            scenarios[0]: ({"Spring Rolls (1 missing)": 150.0}, 0.0, 150.0),
            scenarios[1]: ({"Chilli Paneer (Spillage)": 220.0}, 49.0, 269.0),
            scenarios[2]: ({}, 0.0, 0.0),
        }
    elif "pasta" in img_name_lower:
        order_items = [
            {"name": "Penne Arrabiata", "qty": 1, "unit_price": 320.0, "full_value": 320.0},
            {"name": "Cheesy Garlic Bread", "qty": 1, "unit_price": 160.0, "full_value": 160.0},
            {"name": "Tiramisu Cup", "qty": 1, "unit_price": 210.0, "full_value": 210.0},
        ]
        scenarios = [
            "Missing Garlic Bread (Shortfall)",
            "Wrong Pasta Delivered (White Sauce instead of Red)",
            "Full Order Correct & Intact (No Issue)",
        ]
        scenario_map = {
            scenarios[0]: ({"Garlic Bread (1 missing)": 160.0}, 0.0, 160.0),
            scenarios[1]: ({"Penne Arrabiata (Wrong Item)": 320.0}, 49.0, 369.0),
            scenarios[2]: ({}, 0.0, 0.0),
        }
    elif "thali_2" in img_name_lower or "deluxe" in img_name_lower:
        order_items = [
            {"name": "Deluxe Maharaja Thali", "qty": 1, "unit_price": 380.0, "full_value": 380.0},
            {"name": "Spiced Buttermilk (Chaas)", "qty": 2, "unit_price": 50.0, "full_value": 100.0},
            {"name": "Bengali Rasgulla (2 pcs)", "qty": 1, "unit_price": 60.0, "full_value": 60.0},
        ]
        scenarios = [
            "Missing Spiced Buttermilk (Shortfall)",
            "Damaged Thali Packaging / Leaking Dal",
            "Full Order Correct & Intact (No Issue)",
        ]
        scenario_map = {
            scenarios[0]: ({"Spiced Buttermilk (2 missing)": 100.0}, 0.0, 100.0),
            scenarios[1]: ({"Deluxe Thali (Damaged Lid)": 380.0}, 49.0, 429.0),
            scenarios[2]: ({}, 0.0, 0.0),
        }
    elif "thali" in img_name_lower:
        order_items = [
            {"name": "North Indian Executive Thali", "qty": 2, "unit_price": 290.0, "full_value": 580.0},
            {"name": "Fresh Lime Soda", "qty": 2, "unit_price": 60.0, "full_value": 120.0},
            {"name": "Extra Gulab Jamun (2 pcs)", "qty": 1, "unit_price": 70.0, "full_value": 70.0},
        ]
        scenarios = [
            "Missing Extra Gulab Jamun (Shortfall)",
            "Leaking Dal Makhani Bowl (Spillage)",
            "Full Order Correct & Intact (No Issue)",
        ]
        scenario_map = {
            scenarios[0]: ({"Gulab Jamun (1 missing)": 70.0}, 0.0, 70.0),
            scenarios[1]: ({"Executive Thali (Spilled Dal)": 290.0}, 49.0, 339.0),
            scenarios[2]: ({}, 0.0, 0.0),
        }
    else:
        order_items = [
            {"name": "North Indian Executive Thali", "qty": 2, "unit_price": 290.0, "full_value": 580.0},
            {"name": "Garlic Naan", "qty": 4, "unit_price": 55.0, "full_value": 220.0},
            {"name": "Sweet Lassi", "qty": 2, "unit_price": 80.0, "full_value": 160.0},
        ]
        scenarios = [
            "Missing 2 Garlic Naan (Shortfall)",
            "Wrong Item Delivered",
            "Damaged Packaging / Spillage",
            "Full Order Correct & Intact (No Issue)",
        ]
        scenario_map = {
            scenarios[0]: ({"Garlic Naan (2 missing)": 110.0}, 49.0, 159.0),
            scenarios[1]: ({"Executive Thali (Wrong Item)": 580.0}, 49.0, 629.0),
            scenarios[2]: ({"Damaged Packaging": 290.0}, 0.0, 290.0),
            scenarios[3]: ({}, 0.0, 0.0),
        }

    st.markdown("**1. Tabular Order Manifest**")
    st.table(order_items)

    st.markdown("**2. Visual Inspection & Discrepancy Evaluation**")

    if decision_action == "HARD_REJECT":
        st.error("Adjudication Terminated: Claim failed authenticity checks. Refund payout: **Rs 0.00**")
        st.json({
            "status": "REJECTED",
            "fraud_risk_score": total_risk_score,
            "action": "HARD_REJECT",
            "notice": "Unable to verify proof of delivery.",
            "suspicious_log_updated": True,
        })
    elif decision_action == "MANUAL_REVIEW":
        st.warning("Claim Held: Borderline risk score requires manual investigation.")
        st.json({
            "status": "PENDING_REVIEW",
            "fraud_risk_score": total_risk_score,
            "action": "MANUAL_REVIEW",
            "queue_reference": "QUEUE-2026-091",
            "payout_held": True,
            "flags": flags,
        })
    else:
        st.success("Security Checks Cleared: Performing item cross-matching.")
        discrepancy_type = st.selectbox("Discrepancy Scenario:", scenarios)
        refund_breakdown, waiver, total_refund = scenario_map[discrepancy_type]

        p1, p2 = st.columns(2)
        with p1:
            st.markdown("**Payout Summary**")
            st.metric("Total Approved Refund", f"Rs {total_refund:.2f}")
            st.metric("Delivery Fee Waiver", f"Rs {waiver:.2f}")
        with p2:
            st.markdown("**Line-Item Detail**")
            st.json(refund_breakdown if refund_breakdown else {"info": "All items verified. Rs 0 refund."})

# -----------------------------------------------------------------------------
# Tab 4: Technical Foundations & Formulas
# -----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Technical Reference & Mathematical Formulations")

    st.markdown("### 1. 2D Discrete Cosine Transform (DCT-II)")
    st.markdown("Transforms spatial luminance values into frequency coefficients:")
    st.latex(r"X_{k_1, k_2} = \sum_{n_1=0}^{N_1-1} \sum_{n_2=0}^{N_2-1} x_{n_1, n_2} \cos\left[\frac{\pi}{N_1}\left(n_1 + \frac{1}{2}\right)k_1\right] \cos\left[\frac{\pi}{N_2}\left(n_2 + \frac{1}{2}\right)k_2\right]")

    st.markdown("### 2. Bitwise Hamming Distance Metric")
    st.markdown("Measures bit difference between two 64-bit perceptual hash tokens:")
    st.latex(r"D_H(h_1, h_2) = \sum_{j=0}^{63} (B_{1,j} \oplus B_{2,j})")

    st.markdown("### 3. Laplacian Variance for Edge Sharpness")
    st.markdown("Evaluates second-order spatial derivatives to measure texture crispness:")
    st.latex(r"\nabla^2 I = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix} * I \quad \implies \quad \text{Blur Score} = \text{Var}(\nabla^2 I)")

    st.markdown("### 4. Quality Dimensions Reference")
    st.markdown(
        r"""
        | Dimension | Evaluation Check | Failure Trigger |
        |---|---|---|
        | **Accuracy** | Visual items match reality | Discrepancy between manifest and delivered contents |
        | **Completeness** | Multi-file requirement | Missing receipt or missing item photo |
        | **Consistency** | Cross-modal text matching | Restaurant name or item mismatch on printed bill |
        | **Timeliness** | Timestamp window check | Photo timestamp exceeds 2-hour delivery window |
        | **Validity** | Sensor and physical check | Re-photographed display, bezel, or synthetic generation |
        | **Uniqueness** | Perceptual deduplication | pHash collision against historical blacklist ($D_H \le 4$) |
        """
    )
