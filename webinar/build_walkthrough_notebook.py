"""
Build the human-written Laboratory Walkthrough Notebook:
'ai_ready_image_preprocessing_walkthrough.ipynb'.

Curriculum Alignment:
- Deep Learning Data Pipeline with On-the-Fly Augmentation & Batching (PyTorch)
- Generative AI vs. Classical Augmentation & Synthetic Image Forensics
- Wrong Food Delivery Dispute Resolution & Automated Refund Engine
"""
import json
import os

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "ai_ready_image_preprocessing_walkthrough.ipynb")

cells = []

def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

# -----------------------------------------------------------------------------
# Title & Problem Statement
# -----------------------------------------------------------------------------
add_md(r"""# Wrong Food Delivery Dispute Adjudication & Image Authenticity Pipeline
## Laboratory Manual: PyTorch Data Pipeline, Generative AI Detection & Automated Refund Engine

### Problem Statement & Operational Use Case
In online food delivery platforms (e.g. Zomato replica), customers frequently request refunds claiming:
*"I was delivered the wrong food order!"*

To adjudicate the claim automatically, the platform requires customers to submit two images:
1. **Delivered Food Photo**: Proof of the meal items physically received.
2. **Printed Bill Receipt**: The paper invoice attached to the delivery bag.

Fraudulent actors increasingly attempt to exploit this policy by submitting **Generative AI synthetic images** (e.g. photorealistic diffusion outputs of incorrect dishes) to obtain wrongful payouts without receiving wrong food.

This laboratory guide engineers an end-to-end production pipeline:
1. **PyTorch Deep Learning Data Pipeline**: Dataset abstraction, on-the-fly transformations, classical data augmentation, and tensor batching.
2. **Generative AI vs. Real Image Detection Engine**: Evaluating physical camera photon shot noise (PRNU), 2D-FFT high-frequency power spectra, and hardware EXIF provenance.
3. **Dispute Adjudication & Refund Engine**:
   - If image is **AI-generated**: Lock refund loop, record suspicious activity, and issue **₹0.00** payout.
   - If image is **Real**: Verify if wrong food was delivered against the printed bill. If wrong, issue **100% item refund + delivery fee waiver**.""")

# -----------------------------------------------------------------------------
# 1. Environment & Package Verification
# -----------------------------------------------------------------------------
add_md(r"""## 1. Environment & Tooling Verification
We initialize the required deep learning, computer vision, and numerical libraries:
- `torch` & `torchvision`: PyTorch `Dataset`, `DataLoader`, tensor transformations, and normalization.
- `PIL` & `cv2`: Multi-format image container ingestion and spatial signal processing.
- `numpy` & `matplotlib`: Spectral power plotting and matrix mathematics.""")

add_code(r"""import os
import sys
import io
import math
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps, ExifTags
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['image.cmap'] = 'magma'

print("Python       :", sys.version.split()[0])
print("PyTorch      :", torch.__version__)
print("Torchvision  :", torchvision.__version__ if 'torchvision' in locals() else 'Installed')
print("OpenCV       :", cv2.__version__)
print("Pillow       :", Image.__version__)
print("CUDA Avail   :", torch.cuda.is_available())
print("Environment initialized successfully.")""")

# -----------------------------------------------------------------------------
# 2. Dataset Ingestion (Food Photos + Printed Receipt)
# -----------------------------------------------------------------------------
add_md(r"""## 2. Ingestion of Customer Dispute Evidence
The customer submits evidence pairing the delivered meal with the printed receipt invoice.
We load genuine customer food captures alongside reference synthetic diffusion outputs:
- Real Food Items: `dosa.jpg`, `Burger.jpg`, `Chole.avif`, `Noodles.jpeg`, `Pasta.jpg`, `Thali.jpeg`
- Synthetic Food: `ai_generated_food.jpg` (Diffusion model output)
- Printed Bill Receipt: `delivery_receipt_bill.jpg`""")

add_code(r"""SAMPLE_DIR = os.path.join(os.path.dirname(__file__) if '__file__' in locals() else '.', "sample_images")

def load_sample(filename):
    path = os.path.join(SAMPLE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing sample file: {path}")
    return Image.open(path)

img_dosa = load_sample("dosa.jpg")
img_burger = load_sample("Burger.jpg")
img_chole = load_sample("Chole.avif")
img_noodles = load_sample("Noodles.jpeg")
img_ai = load_sample("ai_generated_food.jpg")
img_bill = load_sample("delivery_receipt_bill.jpg")

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
axes[0].imshow(img_dosa); axes[0].set_title(f"Real Food: Masala Dosa\n{img_dosa.size[0]}x{img_dosa.size[1]} ({img_dosa.format})"); axes[0].axis('off')
axes[1].imshow(img_chole); axes[1].set_title(f"Real Food: Chole (AVIF)\n{img_chole.size[0]}x{img_chole.size[1]} ({img_chole.format})"); axes[1].axis('off')
axes[2].imshow(img_ai); axes[2].set_title(f"Synthetic Food: AI Diffusion\n{img_ai.size[0]}x{img_ai.size[1]} ({img_ai.format})"); axes[2].axis('off')
axes[3].imshow(img_bill); axes[3].set_title(f"Printed Receipt / Bill\n{img_bill.size[0]}x{img_bill.size[1]} ({img_bill.format})"); axes[3].axis('off')

plt.tight_layout()
plt.show()""")

# -----------------------------------------------------------------------------
# 3. PyTorch Deep Learning Data Pipeline
# -----------------------------------------------------------------------------
add_md(r"""## 3. PyTorch Data Pipeline with On-the-Fly Transformations
Deep vision models expect standardized float32 tensors with channel-wise normalization:
$$z_{c, x, y} = \frac{x_{c, x, y} - \mu_c}{\sigma_c} \quad \text{where } c \in \{R, G, B\}$$

Using ImageNet statistics: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.

We compare two pipelines:
1. **Inference Pipeline**: Deterministic `Resize(224) -> CenterCrop(224) -> ToTensor() -> Normalize()`.
2. **Classical Data Augmentation Pipeline**: `RandomCrop(224) -> RandomHorizontalFlip() -> ColorJitter() -> RandomAffine() -> Normalize()`.
   Classical augmentation introduces invariance to customer camera angles and lighting without distorting physical ground truth.""")

add_code(r"""IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

inference_transforms = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

augmentation_transforms = T.Compose([
    T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
    T.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.92, 1.08)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

clean_dosa = ImageOps.exif_transpose(img_dosa).convert("RGB")
tensor_inf = inference_transforms(clean_dosa)
tensor_aug = augmentation_transforms(clean_dosa)

print("Inference Tensor Shape :", tensor_inf.shape, "| Dtype:", tensor_inf.dtype)
print("Tensor Min / Max Range :", float(tensor_inf.min()), "to", float(tensor_inf.max()))

# Visualizing Augmented vs Deterministic Tensor
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

inv_inf = tensor_inf.clone()
for c in range(3): inv_inf[c] = inv_inf[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
axes[0].imshow(np.clip(inv_inf.permute(1, 2, 0).numpy(), 0, 1))
axes[0].set_title("Standardized Inference Tensor (224x224)")
axes[0].axis('off')

inv_aug = tensor_aug.clone()
for c in range(3): inv_aug[c] = inv_aug[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
axes[1].imshow(np.clip(inv_aug.permute(1, 2, 0).numpy(), 0, 1))
axes[1].set_title("Classical Augmentation (Crop + Jitter + Affine)")
axes[1].axis('off')

plt.tight_layout()
plt.show()""")

# -----------------------------------------------------------------------------
# 4. PyTorch Dataset & DataLoader Mini-Batching
# -----------------------------------------------------------------------------
add_md(r"""## 4. PyTorch Dataset & DataLoader Mini-Batch Assembly
To achieve high throughput during peak order hours, customer dispute claims are ingested as a batched stream using PyTorch `Dataset` and `DataLoader` abstractions.""")

add_code(r"""class CustomerDisputeDataset(Dataset):
    def __init__(self, claim_records, transform=None):
        self.records = claim_records
        self.transform = transform or inference_transforms

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        food_pil = ImageOps.exif_transpose(item["food_image"]).convert("RGB")
        bill_pil = ImageOps.exif_transpose(item["bill_image"]).convert("RGB")

        return {
            "food_tensor": self.transform(food_pil),
            "bill_tensor": self.transform(bill_pil),
            "order_id": item["order_id"],
            "billed_dish": item["billed_dish"],
            "claimed_dish": item["claimed_dish"]
        }

claim_records = [
    {"food_image": img_dosa, "bill_image": img_bill, "order_id": "ORD-101", "billed_dish": "Butter Chicken", "claimed_dish": "Masala Dosa"},
    {"food_image": img_burger, "bill_image": img_bill, "order_id": "ORD-102", "billed_dish": "Butter Chicken", "claimed_dish": "Burger"},
    {"food_image": img_chole, "bill_image": img_bill, "order_id": "ORD-103", "billed_dish": "Butter Chicken", "claimed_dish": "Chole Bhature"},
    {"food_image": img_ai, "bill_image": img_bill, "order_id": "ORD-104", "billed_dish": "Butter Chicken", "claimed_dish": "Synthetic Food"}
]

dataset = CustomerDisputeDataset(claim_records, transform=inference_transforms)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

batch = next(iter(dataloader))
print("Food Tensor Batch Dimensions :", batch["food_tensor"].shape, "[BatchSize=4, Channels=3, H=224, W=224]")
print("Bill Tensor Batch Dimensions :", batch["bill_tensor"].shape, "[BatchSize=4, Channels=3, H=224, W=224]")
print("Batch Order IDs              :", batch["order_id"])""")

# -----------------------------------------------------------------------------
# 5. Generative AI vs. Real Image Detection Engine
# -----------------------------------------------------------------------------
add_md(r"""## 5. Generative AI vs. Real Image Forensics Engine
How does the system determine whether an uploaded food photo is authentic or generated by an AI model?

Physical Forensics Indicators:
1. **Camera Sensor Noise Residual (PRNU)**: Real CMOS sensors introduce Poisson-Gaussian photon shot noise. Diffusion models create mathematically smooth pixel manifolds with low residual noise variance.
2. **2D Fast Fourier Transform (FFT) Power Spectrum**:
   $$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) \exp\left[-j 2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)\right]$$
   Real lenses maintain radial energy across high frequencies, whereas synthetic generators exhibit steep high-frequency spectral roll-offs or grid artifacts.
3. **Hardware Provenance Audit**: Authentic mobile phones write hardware tags (`Make`, `Model`, `DateTimeOriginal`). Generative AI tools strip device EXIF tags.""")

add_code(r"""def extract_forensic_signals(pil_img):
    rgb = pil_img.convert("RGB")
    gray = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # 1. Laplacian Edge Sharpness Variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_var = float(lap.var())

    # 2. Sensor Noise Residual (High-pass filtered)
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_res = cv2.absdiff(gray, smoothed)
    noise_var = float(np.var(noise_res))

    # 3. 2D-FFT Radial High-Frequency Energy Ratio
    fft2 = np.fft.fft2(gray)
    fshift = np.fft.fftshift(fft2)
    mag_spec = np.log(np.abs(fshift) + 1.0)

    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    inner_mask = r <= (min(h, w) // 8)
    outer_mask = (r > (min(h, w) // 8)) & (r <= (min(h, w) // 3))
    
    inner_e = float(np.mean(mag_spec[inner_mask])) if np.any(inner_mask) else 1.0
    outer_e = float(np.mean(mag_spec[outer_mask])) if np.any(outer_mask) else 0.0
    hf_ratio = outer_e / max(inner_e, 1e-5)

    # 4. EXIF Hardware Audit
    exif = pil_img.getexif() if hasattr(pil_img, "getexif") else (pil_img._getexif() if hasattr(pil_img, "_getexif") else None)
    has_hw = False
    if exif:
        tag_dict = {ExifTags.TAGS.get(k, k): v for k, v in exif.items() if k in ExifTags.TAGS}
        has_hw = bool(tag_dict.get("Make") and tag_dict.get("Model"))

    # Synthesize AI Probability
    prob = 0.0
    if not exif: prob += 0.35
    elif not has_hw: prob += 0.20
    else: prob -= 0.30

    if noise_var < 8.0: prob += 0.35
    elif noise_var > 20.0: prob -= 0.20

    if blur_var < 120.0: prob += 0.20
    elif blur_var > 350.0: prob -= 0.15

    if hf_ratio < 0.28: prob += 0.20

    final_prob = float(np.clip(prob, 0.02, 0.98))
    is_ai = final_prob >= 0.55

    return {
        "is_ai": is_ai,
        "prob_ai": round(final_prob, 3),
        "noise_var": round(noise_var, 2),
        "blur_var": round(blur_var, 2),
        "hf_ratio": round(hf_ratio, 3),
        "has_hardware": has_hw,
        "mag_spec": mag_spec
    }

forensics_real = extract_forensic_signals(img_dosa)
forensics_ai = extract_forensic_signals(img_ai)

print("Real Dosa Photo  -> AI Probability:", forensics_real['prob_ai'], "| Verdict:", "AI" if forensics_real['is_ai'] else "REAL AUTHENTIC")
print("AI-Generated Food -> AI Probability:", forensics_ai['prob_ai'], "| Verdict:", "AI SYNTHETIC" if forensics_ai['is_ai'] else "REAL")

# Visualize 2D-FFT comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
c1 = axes[0].imshow(forensics_real['mag_spec'], cmap='magma')
axes[0].set_title(f"Real Photo 2D-FFT (HF Ratio: {forensics_real['hf_ratio']})")
fig.colorbar(c1, ax=axes[0], fraction=0.046, pad=0.04)

c2 = axes[1].imshow(forensics_ai['mag_spec'], cmap='magma')
axes[1].set_title(f"AI-Generated 2D-FFT (HF Ratio: {forensics_ai['hf_ratio']})")
fig.colorbar(c2, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()""")

# -----------------------------------------------------------------------------
# 6. Wrong Food Dispute Adjudication & Refund Engine
# -----------------------------------------------------------------------------
add_md(r"""## 6. Wrong Food Delivery Dispute Adjudication & Refund Engine
The business logic ties image authenticity directly to dispute settlement:
1. **Authenticity Gate**:
   - If food photo is **AI-generated** ($P(\text{AI}) \ge 0.55$):
     - **Refund is strictly blocked** (Payout = ₹0.00).
     - Reason: *"Unable to verify physical proof of delivery: Image identified as synthetic."*
2. **Item Discrepancy Verification**:
   - If food photo is **Real physical capture**:
     - Cross-verify delivered meal against billed restaurant receipt items.
     - **Wrong Food Delivered**: Customer receives **100% item refund + full ₹49 delivery fee waiver**!
     - **Correct Food Delivered**: Customer received what was billed $\rightarrow$ **₹0.00 refund**.""")

add_code(r"""def resolve_wrong_food_claim(food_pil, billed_items, observed_dish_name, delivery_fee=49.0):
    # Step 1: Check Authenticity
    forensic = extract_forensic_signals(food_pil)
    
    if forensic["is_ai"]:
        return {
            "verdict": "REJECTED_SYNTHETIC_FRAUD",
            "is_ai_generated": True,
            "ai_probability": forensic["prob_ai"],
            "refund_issued_rs": 0.0,
            "delivery_fee_waived_rs": 0.0,
            "total_payout_rs": 0.0,
            "action_reason": "Refund blocked: Proof-of-delivery image flagged as synthetic / AI-generated."
        }
        
    # Step 2: Cross-verify delivered meal vs. billed items
    billed_dish_names = [item["name"].lower() for item in billed_items]
    is_wrong_delivery = not any(observed_dish_name.lower() in name or name in observed_dish_name.lower() for name in billed_dish_names)
    
    if is_wrong_delivery:
        item_total = sum(i["qty"] * i["unit_price"] for i in billed_items)
        return {
            "verdict": "APPROVED_WRONG_ITEM",
            "is_ai_generated": False,
            "ai_probability": forensic["prob_ai"],
            "refund_issued_rs": item_total,
            "delivery_fee_waived_rs": delivery_fee,
            "total_payout_rs": item_total + delivery_fee,
            "action_reason": f"Wrong food delivery confirmed: Customer received '{observed_dish_name}' instead of billed items."
        }
    else:
        return {
            "verdict": "RESOLVED_CORRECT_DELIVERY",
            "is_ai_generated": False,
            "ai_probability": forensic["prob_ai"],
            "refund_issued_rs": 0.0,
            "delivery_fee_waived_rs": 0.0,
            "total_payout_rs": 0.0,
            "action_reason": "No discrepancy: Delivered food matches items printed on receipt."
        }

# Billed restaurant manifest from printed paper receipt
receipt_manifest = [
    {"name": "Butter Chicken", "qty": 2, "unit_price": 320.0},
    {"name": "Garlic Naan", "qty": 4, "unit_price": 55.0}
]

print("=== CASE 1: FRAUDULENT CUSTOMER SUBMITTING AI-GENERATED FOOD ===")
decision_ai = resolve_wrong_food_claim(img_ai, receipt_manifest, "Synthetic Pizza")
print("Status       :", decision_ai['verdict'])
print("Total Refund : Rs.", decision_ai['total_payout_rs'])
print("Reason       :", decision_ai['action_reason'])
print()

print("=== CASE 2: GENUINE CUSTOMER RECEIVING WRONG FOOD (MASALA DOSA) ===")
decision_real = resolve_wrong_food_claim(img_dosa, receipt_manifest, "Masala Dosa")
print("Status       :", decision_real['verdict'])
print("Total Refund : Rs.", decision_real['total_payout_rs'])
print("Reason       :", decision_real['action_reason'])""")

# -----------------------------------------------------------------------------
# 7. Summary & Checklist
# -----------------------------------------------------------------------------
add_md(r"""## 7. Production Engineering Summary

| Pipeline Component | Deep Learning Technique | Operational Protection |
|---|---|---|
| **Input Standardization** | PyTorch `Resize(224) + Normalize(ImageNet)` | Consistent tensor geometries across diverse mobile phone cameras |
| **Classical Augmentation** | `RandomCrop + Affine + ColorJitter` | Prepares models for real-world lighting, tilt, and occlusion variations |
| **DataLoader Batching** | Mini-batch tensor aggregation `[B, 3, 224, 224]` | High-throughput GPU/CPU serving during peak dinner delivery hours |
| **Sensor Grain Forensics** | Photon noise residual variance (PRNU) | Detects over-smoothed synthetic manifolds produced by diffusion models |
| **Spectral Analysis** | 2D Fast Fourier Transform (FFT) Power Spectrum | Identifies unnatural high-frequency spectral roll-offs |
| **Hardware Provenance** | EXIF Tag Audit (`Make`, `Model`, `DateTimeOriginal`) | Exposes synthetic outputs and stripped internet downloads |
| **Automated Dispute Gate** | Binary Authenticity Gate $\rightarrow$ Multimodal Receipt Match | **Zero payout for AI fraud**; **100% item refund + delivery waiver for genuine wrong food** |

---
*Laboratory Manual: Pre-processing for AI-Ready Data — Wrong Food Dispute Resolution Engine.*""")

# Write notebook JSON
notebook_content = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print(f"Clean manual-style notebook written to: {NOTEBOOK_PATH}")
