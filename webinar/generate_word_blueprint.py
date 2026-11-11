"""
Comprehensive generator for the complete, professional Word Document (.docx):
'Complete Webinar Delivery Blueprint: From Zero Knowledge to Production AI Forensics'
Fully covers all curriculum topics from basic to advanced:
- 6 Data Quality Dimensions, Ingestion, Digital Image Representation
- PyTorch Data Pipeline, Aspect-Preserving Standardization, DataLoader Batching, ImageNet Normalization
- Temporal Windowing, Leakage Prevention, Display Screen Moiré & Bezel Forensics
- 64-bit DCT Perceptual Hashing (pHash), Hamming Distance Matching
- Generative AI vs Real Forensics: 2D FFT Diffraction, CMOS PRNU Shot Noise, VAE Lattice Peakiness, Studio Glaze
- International C2PA / IPTC XMP Metadata Provenance, Container Spoofing
- Multimodal Receipt Cross-Verification, Discrepancy Matching
- Automated Policy Tree, Sybil Device Fingerprinting, Trust Score Ledger Mutation
- 60-Minute Speaker Runbook & Interactive Student Checkpoint Questions
"""

from __future__ import annotations

import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, hex_color: str):
    """Set the background color of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tc_pr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
    """Set inner margins (padding) for a cell in twips (1 pt = 20 twips)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)

def set_callout_borders(cell, border_color="1B365D", border_size="24"):
    """Set thick left border and none for others."""
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="{border_size}" w:space="0" w:color="{border_color}"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tc_pr.append(borders)

def set_table_borders(table, border_color="CBD5E0"):
    """Set clean light gray borders on table."""
    tbl_pr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="{border_color}"/>'
        f'<w:left w:val="none"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="{border_color}"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{border_color}"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tbl_pr.append(borders)

def add_heading_1(doc, text: str, color_rgb=(27, 54, 93)):
    h = doc.add_heading(level=1)
    r = h.add_run(text)
    r.font.size = Pt(15.5)
    r.font.bold = True
    r.font.color.rgb = RGBColor(*color_rgb)
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(5)
    return h

def add_heading_2(doc, text: str, color_rgb=(43, 108, 176)):
    h = doc.add_heading(level=2)
    r = h.add_run(text)
    r.font.size = Pt(12.5)
    r.font.bold = True
    r.font.color.rgb = RGBColor(*color_rgb)
    h.paragraph_format.space_before = Pt(10)
    h.paragraph_format.space_after = Pt(3)
    return h

def add_callout(doc, title: str, text: str, border_hex="2B6CB0", bg_hex="F7FAFC", title_rgb=(27, 54, 93)):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = tbl.cell(0, 0)
    set_cell_background(c, bg_hex)
    set_callout_borders(c, border_hex, "24")
    set_cell_margins(c, top=120, bottom=120, left=160, right=160)
    p = c.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    rt = p.add_run(title + "\n")
    rt.bold = True
    rt.font.size = Pt(10)
    rt.font.color.rgb = RGBColor(*title_rgb)
    rb = p.add_run(text)
    rb.font.size = Pt(9.5)
    rb.font.color.rgb = RGBColor(45, 55, 72)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def build_full_blueprint_doc(output_path: str):
    doc = Document()

    # 1-inch margins
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    # Base styling
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Calibri"
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(45, 55, 72)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)

    # -------------------------------------------------------------
    # Document Cover / Header
    # -------------------------------------------------------------
    p_meta = doc.add_paragraph()
    r_meta = p_meta.add_run("ENTERPRISE ENGINEERING FIELD GUIDE & MASTERCLASS PLAYBOOK")
    r_meta.font.size = Pt(9)
    r_meta.font.bold = True
    r_meta.font.color.rgb = RGBColor(43, 108, 176)
    p_meta.paragraph_format.space_after = Pt(2)

    p_title = doc.add_paragraph()
    r_title = p_title.add_run("Complete Webinar Delivery Blueprint:\nFrom Zero Knowledge to Production AI Forensics")
    r_title.font.size = Pt(21)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(27, 54, 93)
    p_title.paragraph_format.space_after = Pt(4)

    p_sub = doc.add_paragraph()
    r_sub = p_sub.add_run("Comprehensive Curriculum: Data Quality, PyTorch Ingestion, 2D FFT Diffraction, CMOS PRNU Physics, C2PA Provenance & Multimodal Adjudication")
    r_sub.font.size = Pt(11)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(113, 128, 150)
    p_sub.paragraph_format.space_after = Pt(12)

    # Metadata Grid
    meta_table = doc.add_table(rows=2, cols=3)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        [("Audience Target", "Undergraduate & Graduate Students (Zero Prior AI Knowledge)"),
         ("Curriculum Scope", "Basic (Foundations) ➔ Intermediate (Pipelines) ➔ Advanced (Forensics)"),
         ("Live System", "Streamlit Forensics Workbench (port 8501)")],
        [("Core Stack", "PyTorch, OpenCV, NumPy, Matplotlib, C2PA XMP, pHash"),
         ("Verification", "56 Automated Unit & Forensic Tests (100% Pass)"),
         ("GitHub Repo", "https://github.com/kushals-bits/Webinar-4.git")]
    ]
    for r_idx, row in enumerate(meta_data):
        for c_idx, (k, v) in enumerate(row):
            cell = meta_table.cell(r_idx, c_idx)
            set_cell_background(cell, "F7FAFC")
            set_cell_margins(cell, top=70, bottom=70, left=90, right=90)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            rk = p.add_run(f"{k}: ")
            rk.bold = True
            rk.font.size = Pt(8.5)
            rk.font.color.rgb = RGBColor(27, 54, 93)
            rv = p.add_run(v)
            rv.font.size = Pt(8.5)
            rv.font.color.rgb = RGBColor(74, 85, 104)
    set_table_borders(meta_table, "CBD5E0")
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------
    # 1. Executive Summary & Curriculum Hierarchy
    # -------------------------------------------------------------
    add_heading_1(doc, "1. Executive Summary & Curriculum Architecture")
    p = doc.add_paragraph()
    p.add_run(
        "This field guide provides an exhaustive, end-to-end pedagogical blueprint designed to take students with zero knowledge of computer vision "
        "and guide them through the mathematical, statistical, and operational foundations of modern AI dispute adjudication. "
        "The curriculum is centered around an authentic, high-stakes industrial use case: automated wrong food delivery refund claims in food delivery platforms (e.g. Zomato replica).\n\n"
        "The delivery is divided into three rigorous pedagogical tiers:"
    )

    p_tiers = doc.add_paragraph()
    p_tiers.add_run(
        "1. Level 1: Basic (Foundations & Intuition) — The business dilemma, 6 Data Quality Dimensions, digital image representation, RGB light channels, and cryptographic vs. perceptual hashing.\n"
        "2. Level 2: Intermediate (Engineering Pipelines & Deduplication) — EXIF auto-transposition, temporal windowing, PyTorch dataset & dataloader mini-batching, aspect-preserving standardization, ImageNet normalization, screen capture bezel/Moiré detection, and 64-bit DCT pHash.\n"
        "3. Level 3: Advanced (Synthetic AI Forensics & Decision Policy) — 2D Fast Fourier Transform optical diffraction, CMOS sensor photon shot noise (PRNU), latent diffusion VAE lattice harmonics, international C2PA/IPTC XMP metadata provenance, container spoofing, and multimodal discrepancy adjudication."
    )

    add_callout(
        doc,
        "⏱️ Recommended 60-Minute Masterclass Timeline:",
        "• 00 – 15 Min: Level 1 (Basic) — The ₹100 Crore Problem, 6 Quality Dimensions, Digital Images, and Hashing.\n"
        "• 15 – 35 Min: Level 2 (Intermediate) — PyTorch Ingestion, Tensors, Screen Forensics, and Receipt Cross-Matching.\n"
        "• 35 – 55 Min: Level 3 (Advanced) — Catching GenAI: 2D FFT, CMOS PRNU Grain, VAE Peakiness, C2PA Provenance & Policy Trees.\n"
        "• 55 – 60 Min: Wrap-Up & Live Interactive Q&A — Real-World Deployment, Sybil Attack Resistance & Career Q&A.",
        border_hex="2B6CB0", bg_hex="EDF2F7", title_rgb=(27, 54, 93)
    )

    # -------------------------------------------------------------
    # 2. Level 1: Basic — Foundations, Image Representation & Business Context
    # -------------------------------------------------------------
    add_heading_1(doc, "2. Level 1: Basic — Foundations & Intuition (15 Mins)", color_rgb=(40, 116, 50))

    add_heading_2(doc, "Pointer 1.1: The Industrial Problem — The ₹100 Crore Refund Loophole")
    p = doc.add_paragraph()
    p.add_run(
        "Online food delivery platforms process millions of orders daily. A core operational feature is instant customer service: if a customer receives the wrong meal, a cold item, or a spilled container, "
        "they upload photographic evidence to receive immediate reimbursement. However, automated refund endpoints present a massive target for fraud syndicates and dishonest customers:\n"
        "• The Exploit: Bad actors upload photos of spilled food scraped from Google Images, recycled images from past orders, or generate synthetic meal photos using AI.\n"
        "• The Asymmetric Cost Trade-Off: Human agent review creates a 48-hour SLA that destroys customer goodwill, while blind automated payouts bankrupt the platform.\n"
        "• The Engineering Mandate: Build an automated, real-time vision adjudicator capable of verifying authenticity and line-item discrepancies in under 500 milliseconds."
    )

    add_callout(
        doc,
        "🎙️ Word-for-Word Speaker Script (The Opening Hook):",
        '"Imagine you order a Deluxe Maharaja Thali on your favorite food app. When the delivery rider arrives, you open the package and find a plain burger. '
        'You click Request Refund, upload a picture of the burger, and within 30 seconds your bank account is credited with ₹350. Great customer service, right?\n\n'
        'Now imagine 50,000 college students discover that if they prompt Meta AI or Midjourney for a picture of spilled curry on a kitchen floor, '
        'they can eat free lunch every single day. Food delivery companies lose tens of millions every year to automated refund fraud.\n\n'
        'If we inspect every claim with human agents, customers wait 48 hours and get furious. If we blindly auto-refund, platforms bleed out. '
        'Today, we are going to build an automated, real-time AI adjudicator that catches fake and recycled claims in under 500 milliseconds."',
        border_hex="28A745", bg_hex="F0FFF4", title_rgb=(34, 111, 48)
    )

    add_heading_2(doc, "Pointer 1.2: Foundations of AI-Ready Data — The 6 Data Quality Dimensions")
    p = doc.add_paragraph()
    p.add_run(
        "Before feeding unstructured data into neural vision models, production systems audit the six fundamental data quality dimensions:\n"
        "1. Accuracy: Does the image depict the true physical entity delivered, rather than a synthetic hallucination or a stock photo?\n"
        "2. Completeness: Are all required evidence modalities present (delivered dish photo + printed restaurant bill invoice)?\n"
        "3. Consistency: Is the image formatted as a canonical 3-channel RGB canvas, and do metadata timestamps align across time zones?\n"
        "4. Timeliness: Was the photograph captured within the delivery claim window (preventing claims submitted 3 weeks after delivery)?\n"
        "5. Validity: Does the file conform to genuine image encoding standards (JPEG, PNG, AVIF) without container spoofing?\n"
        "6. Uniqueness: Is the image novel, or is it a recycled duplicate of an existing claim in the fraud registry?"
    )

    add_heading_2(doc, "Pointer 1.3: What is a Digital Image to a Machine?")
    p = doc.add_paragraph()
    p.add_run(
        "Students often conceptualize computer vision as an optical eye looking through a camera lens. Demystify this immediately: computers have no eyes, only numerical tensors.\n"
        "• The Discrete Canvas: An image is a 3-dimensional matrix: Shape = [Height (H) × Width (W) × Channels (C)].\n"
        "• Primary Light Channels (RGB): Every color pixel is composed of 3 integer values ranging from 0 (completely dark) to 255 (maximum photon intensity):\n"
        "    - Pure Red: [255, 0, 0] | Pure Green: [0, 255, 0] | Pure Blue: [0, 0, 255]\n"
        "    - White: [255, 255, 255] | Black: [0, 0, 0]\n"
        "• Data Volume at Scale: A standard 12-megapixel smartphone photo (4032 × 3024 pixels) contains 12,192,768 pixels, each with 3 numbers—totaling 36,578,304 discrete integer values.\n"
        "• The Perceptual Paradox: If you modify a single pixel value from 255 to 254, human eyes perceive zero change. But mathematically, a computer calculates the difference instantly."
    )

    add_heading_2(doc, "Pointer 1.4: Cryptographic Hashing (MD5/SHA) vs. Perceptual Hashing (pHash)")
    p = doc.add_paragraph()
    p.add_run(
        "A common beginner question: 'Why can't we just compute an MD5 or SHA-256 hash to check if a photo was already uploaded?'"
    )

    hash_tbl = doc.add_table(rows=4, cols=3)
    hash_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    h_headers = ["Attribute", "Cryptographic Hashing (MD5 / SHA-256)", "Perceptual Hashing (64-bit DCT pHash)"]
    for idx, text in enumerate(h_headers):
        cell = hash_tbl.cell(0, idx)
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p = cell.paragraphs[0]
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    h_rows = [
        ("Design Principle", "Security & Integrity: Any 1-bit change produces an entirely distinct hash (Avalanche Effect).", "Visual Equivalence: Generates matching fingerprints for images that look visually similar."),
        ("WhatsApp Compression", "❌ Hash changes completely (100% false negative on recycled photos).", "✅ Preserves identical or near-identical hash (Hamming Distance ≤ 4)."),
        ("Computational Domain", "Byte-level cryptographic hash function.", "Frequency-domain 2D Discrete Cosine Transform (DCT-II).")
    ]
    for r_idx, (c0, c1, c2) in enumerate(h_rows, start=1):
        for c_idx, val in enumerate([c0, c1, c2]):
            cell = hash_tbl.cell(r_idx, c_idx)
            set_cell_background(cell, "F7FAFC" if r_idx % 2 == 1 else "FFFFFF")
            set_cell_margins(cell, top=70, bottom=70, left=90, right=90)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(val)
            r.font.size = Pt(8)
    set_table_borders(hash_tbl, "CBD5E0")
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------
    # 3. Level 2: Intermediate — Pipelines, Tensors, Forensics & Cross-Verification
    # -------------------------------------------------------------
    add_heading_1(doc, "3. Level 2: Intermediate — Pipelines, Tensors & Cross-Verification (20 Mins)", color_rgb=(43, 108, 176))

    add_heading_2(doc, "Pointer 2.1: The Ingestion Problem — Why Raw Phone Photos Break Deep Learning")
    p = doc.add_paragraph()
    p.add_run(
        "Production computer vision models cannot directly consume raw mobile phone uploads due to three major real-world discrepancies:\n"
        "• Orientation Variance: Smartphones contain internal gyroscopes that record camera orientation inside EXIF Tag 274. If unhandled, convolutional filters receive a portrait photo rotated 90 degrees sideways.\n"
        "• Color Space Inconsistencies: Uploads frequently contain alpha transparency channels (RGBA) from web screenshots or print CMYK profiles from scanned bills, causing fatal channel mismatch crashes in PyTorch.\n"
        "• Dimension Heterogeneity: One customer uploads a 4032 × 3024 iPhone photo; another uploads an 800 × 600 cropped screenshot. Deep neural networks require a fixed spatial canvas (e.g., 224 × 224 pixels)."
    )

    add_heading_2(doc, "Pointer 2.2: Aspect-Preserving Standardization & PyTorch Normalization")
    p = doc.add_paragraph()
    p.add_run(
        "To standardize image geometries without distorting circular food plates into unnatural ellipses, production pipelines use aspect-preserving resizing followed by center cropping and channel normalization:\n"
        "1. Resize(256): Scales the shortest boundary of the image to 256 pixels while preserving the natural aspect ratio.\n"
        "2. CenterCrop(224): Extracts the canonical 224 × 224 pixel central window where the delivered dish is concentrated.\n"
        "3. ToTensor(): Maps [0, 255] integer pixels into [0.0, 1.0] float32 tensors.\n"
        "4. Channel Standardization: Centers pixel distributions to zero mean and unit variance using standard ImageNet statistics:\n"
        "     z = (x - μ) / σ  where μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]\n"
        "Why Normalize? High integer pixel values cause activation explosions and erratic gradient updates in convolutional neural networks. Standardizing stabilizes backpropagation across all RGB channels."
    )

    add_heading_2(doc, "Pointer 2.3: Classical On-the-Fly Augmentation vs. Generative Augmentation")
    p = doc.add_paragraph()
    p.add_run(
        "During model training, vision models must become invariant to real-world camera tilt, ambient shadows, and zoom levels. We distinguish two approaches:\n"
        "• Classical Augmentation: Operations like RandomCrop(224), RandomHorizontalFlip(p=0.5), ColorJitter(), and RandomAffine() manipulate existing pixel geometries without altering semantic reality.\n"
        "• Generative Augmentation: Using diffusion models to generate synthetic variations alters semantic ground truth and introduces synthetic artifacts that corrupt training data.\n"
        "Conclusion: Classical augmentation preserves ground truth integrity while hardening models against real-world customer photo variations."
    )

    add_heading_2(doc, "Pointer 2.4: PyTorch Dataset & DataLoader Mini-Batch Assembly")
    p = doc.add_paragraph()
    p.add_run(
        "During peak dinner delivery hours, dispute adjudication systems handle tens of thousands of concurrent claims. Processing images sequentially causes severe latency bottlenecks. "
        "We implement custom PyTorch Dataset and DataLoader abstractions to assemble claims into vectorized mini-batches:\n"
        "• Batch Geometry: [Batch_Size, Channels=3, Height=224, Width=224] (e.g., [16, 3, 224, 224]).\n"
        "• Vectorized GPU Execution: High-throughput tensor processing allows parallel inference across multiple claims simultaneously."
    )

    add_heading_2(doc, "Pointer 2.5: Temporal Windowing & Retroactive Fraud Prevention")
    p = doc.add_paragraph()
    p.add_run(
        "Fraudsters frequently attempt to submit photos of damaged food that were captured days or weeks prior to the order delivery. "
        "The pipeline performs an automated temporal audit comparing EXIF DateTimeOriginal against the platform delivery timestamp:\n"
        "  Δt = t_photo - t_delivery\n"
        "• Valid Delivery Claim Window: Δt ∈ [0, 7200 seconds] (0 to 2 hours post-delivery).\n"
        "• Temporal Anomalies: If Δt < 0 (photo taken before the restaurant even cooked the food) or Δt > 7200s (photo taken hours/days later), the claim is flagged for temporal leakage fraud."
    )

    add_heading_2(doc, "Pointer 2.6: Display Screen Forensics (Bezel Uniformity & Moiré Interference)")
    p = doc.add_paragraph()
    p.add_run(
        "A common amateur fraud technique is displaying an image of spilled food on a computer monitor or iPad and photographing the screen with a mobile phone. "
        "The pipeline catches screen replays using two physical indicators:\n"
        "1. Perimeter Bezel Uniformity: Evaluates pixel standard deviation along the extreme image boundaries. Monitor and tablet bezels produce unnaturally uniform border pixel rows.\n"
        "2. Moiré Pattern Filtering: Camera sensors photographing digital displays capture optical interference grids between the monitor's subpixel matrix and the camera's Bayer filter, producing characteristic high-pass ripple frequencies."
    )

    add_heading_2(doc, "Pointer 2.7: 64-bit DCT Perceptual Hashing (pHash) Algorithm")
    p = doc.add_paragraph()
    p.add_run(
        "The perceptual hash compresses visual structural silhouettes into a 64-bit fingerprint via a 4-step algorithm:\n"
        "1. Downsample & Grayscale: Resize canvas to 32 × 32 pixels and convert to single-channel luminance.\n"
        "2. 2D Discrete Cosine Transform (DCT-II): Decomposes the spatial block into spatial frequency components, concentrating low-frequency energy at the top-left.\n"
        "3. Extract 8 × 8 Grid: Retain the top 64 lowest-frequency coefficients (ignoring high-frequency noise and compression artifacts).\n"
        "4. Median Binarization: Calculate the median coefficient. Set bit = 1 if above median, 0 if below. Result: A 16-character hexadecimal hash.\n"
        "• Hamming Distance Metric: Count bitwise differences: DH(hash_A, hash_B) ≤ 10 flags recycled fraud across historical claims."
    )

    add_heading_2(doc, "Pointer 2.8: Multimodal Discrepancy Cross-Verification (Plate vs. Receipt)")
    p = doc.add_paragraph()
    p.add_run(
        "Once image integrity is cleared, multimodal fusion compares the visual entity detected on the food plate against the OCR text parsed from the paper bill:\n"
        "  is_wrong_item = (observed_dish not in billed_manifest)\n"
        "• Financial Settlement Rule: If wrong food is confirmed (e.g. Maharaja Thali delivered instead of Masala Dosa), the engine issues a 100% item refund + full delivery fee waiver (₹49).\n"
        "• Correct Delivery Rule: If delivered food matches the billed order, dispute is closed with ₹0.00 payout."
    )

    # -------------------------------------------------------------
    # 4. Level 3: Advanced — Synthetic AI Forensics & Signal Physics
    # -------------------------------------------------------------
    add_heading_1(doc, "4. Level 3: Advanced — Synthetic AI Forensics & Signal Physics (20 Mins)", color_rgb=(197, 48, 48))

    add_heading_2(doc, "Pointer 3.1: The Modern GenAI Threat — When Scammers Stop Stealing Photos")
    p = doc.add_paragraph()
    p.add_run(
        "Perceptual hashing easily catches recycled photos from Google. But what happens when fraudsters prompt generative models on demand? "
        "Tools like Meta AI, Midjourney, and Stable Diffusion can generate a photo of spilled curry or an incorrect meal in under 5 seconds. "
        "To human support agents, the image looks 100% authentic and delicious. To detect synthetic media, we must look beyond visual semantics and analyze sensor physics, frequency mathematics, and cryptographic metadata."
    )

    add_heading_2(doc, "Pointer 3.2: 2D Fast Fourier Transform (FFT) & Optical Glass Diffraction")
    p = doc.add_paragraph()
    p.add_run(
        "• The Physical Law of Optical Diffraction: Light entering a real camera passes through physical glass elements and mechanical aperture blades. "
        "This optical diffraction sustains rich, continuous radial energy all the way to the sensor's Nyquist frequency boundary.\n"
        "• The Latent Diffusion Limitation: Latent diffusion models decode images from compact latent manifolds (e.g., 64 × 64). While they render smooth textures realistically, "
        "they mathematically fail to synthesize continuous optical diffraction, exhibiting an unnatural radial roll-off in the outer high-frequency ring (r ≥ 0.55):\n"
        "    F(u, v) = ∬ f(x, y) e^(-j 2π (ux + vy)) dx dy\n"
        "    HF_Ratio = [∬_Ring |F(u, v)|^2 du dv] / [∬_Total |F(u, v)|^2 du dv]\n"
        "• Real Camera Photo: High-frequency ratio ≥ 0.65.\n"
        "• Diffusion Synthetic Image: High-frequency ratio drops significantly."
    )

    add_heading_2(doc, "Pointer 3.3: CMOS Sensor Photon Shot Noise (PRNU) vs. Diffusion 'Studio Glaze'")
    p = doc.add_paragraph()
    p.add_run(
        "• Photon Shot Noise (PRNU): Physical CMOS sensors capture discrete photons. Quantum arrival fluctuations (Poisson noise) and microscopic silicon imperfections "
        "create Photo-Response Non-Uniformity (PRNU) in every real camera image. Even on flat, smooth table surfaces, real cameras exhibit subtle variance:\n"
        "    R = |I - GaussianBlur(I, 5)|  ==>  σ^2_noise = Var(R) > 1.0 (often > 20.0)\n"
        "• Diffusion Hyper-Smoothness: Diffusion denoising leaves flat surfaces unnaturally clean and smooth (σ^2_noise < 0.10).\n"
        "• Diffusion Studio Glaze: Synthetic models produce hyper-saturated lighting and unnaturally locked cross-channel correlations (r_RGB > 0.95)."
    )

    add_heading_2(doc, "Pointer 3.4: Latent Diffusion VAE Lattice Peakiness & Checkerboard Artifacts")
    p = doc.add_paragraph()
    p.add_run(
        "When latent diffusion architectures decode internal representations back into pixel space using convolutional transposed layers (upsampling), "
        "they create periodic checkerboard lattice harmonics. In the frequency domain, these appear as isolated periodic spikes in the outer Nyquist ring. "
        "By calculating the peak-to-mean spectral ratio (Peakiness > 0.096), our engine flags latent decoder harmonics even after aggressive social media compression."
    )

    add_heading_2(doc, "Pointer 3.5: C2PA & IPTC Standard Metadata Provenance & Container Spoofing")
    p = doc.add_paragraph()
    p.add_run(
        "• International C2PA Standard: Leading AI providers (Meta AI, Google Imagen, Adobe Firefly) embed international C2PA and IPTC metadata into the image XMP header:\n"
        '    iptcExt:DigitalSourceType = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"\n'
        '    dc:publisher = "Meta"\n'
        "• Why Standard EXIF Fails: Standard Pillow EXIF getters (getexif()) only read legacy TIFF tags and miss XMP streams entirely. "
        "Our engine inspects pil_img.getxmp() and raw byte streams: finding trainedAlgorithmicMedia triggers immediate 99.0% hard detection.\n"
        "• Container Format Spoofing: Web generators often output WebP images that users download and save with a disguised .jpg extension. "
        "The engine audits the underlying container format to flag MIME extension mismatches."
    )

    add_heading_2(doc, "Pointer 3.6: End-to-End Decision Policy Tree & Governance Architecture")
    p = doc.add_paragraph()
    p.add_run(
        "The dispute resolution system integrates all signals into an automated decision policy tree:\n"
        "1. Synthetic AI Flag (P(AI) ≥ 50%) or Blacklisted Hash Match:\n"
        "   ==> Immediate Hard Reject (₹0.00 Payout), bypasses receipt verification, logs incident to suspicious_activity_log, and penalizes customer trust score (-15 pts).\n"
        "2. Authentic Camera Photo + Confirmed Discrepancy:\n"
        "   ==> Approved Wrong Item: Issues 100% item refund + ₹49 delivery fee waiver.\n"
        "3. Authentic Camera Photo + Matching Order Items:\n"
        "   ==> Resolved Correct Delivery: Order verified correct, closed with ₹0.00 payout.\n"
        "• Sybil Device Fingerprinting: Clusters customer accounts by hardware device fingerprint to detect coordinated multi-account refund fraud rings."
    )

    # -------------------------------------------------------------
    # 5. Live Demonstration Guide & Instructor Script
    # -------------------------------------------------------------
    add_heading_1(doc, "5. Live Demonstration Script & Minute-by-Minute Runbook")
    p = doc.add_paragraph()
    p.add_run(
        "This instructor runbook maps each segment of the webinar to specific click actions and visual artifact demonstrations inside the live Streamlit workbench (http://localhost:8501):"
    )

    demo_tbl = doc.add_table(rows=6, cols=3)
    demo_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    dheaders = ["Timeline", "Application Action (http://localhost:8501)", "Speaker Narration & Pedagogical Focus"]
    for idx, text in enumerate(dheaders):
        cell = demo_tbl.cell(0, idx)
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p = cell.paragraphs[0]
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    demo_data = [
        ("Min 00 – 12", "Select 'Real Food: Authentic Mobile Delivery (iPhone 15)'", 
         "Show genuine camera hardware EXIF (Apple A16 Bionic), natural photon shot noise, and continuous frequency scatter. Establish baseline of authenticity."),
        ("Min 12 – 25", "Switch to Tab 0 ('Live Backend Decision Pipeline Flow')", 
         "Walk through the Visual Memory-Map Flowchart. Point out how each stage transforms the input image to an output visual artifact (crop, spectrum, hash)."),
        ("Min 25 – 38", "Select 'Synthetic: Meta AI Delivery Dish (User Case 2 / C2PA)'", 
         "Show how the dish looks appetizing to human eyes, but the red C2PA synthetic media warning triggers instantly, bypassing the receipt gate directly to ₹0.00 lock."),
        ("Min 38 – 48", "Select 'Synthetic: AI-Generated Indian Thali (User Case 1)'", 
         "Show stripped EXIF and display the 2D FFT Log-Magnitude Spectrum in magma colormap. Highlight the steep radial drop-off and lack of sensor grain."),
        ("Min 48 – 60", "Select 'Real Food: Deluxe Maharaja Thali (Real Platter)'", 
         "Demonstrate multimodal receipt matching: photo is authentic camera capture, but discrepancy engine identifies thali delivered instead of dosa. Approve ₹309 refund.")
    ]
    for r_idx, (c0, c1, c2) in enumerate(demo_data, start=1):
        for c_idx, val in enumerate([c0, c1, c2]):
            cell = demo_tbl.cell(r_idx, c_idx)
            set_cell_background(cell, "F7FAFC" if r_idx % 2 == 1 else "FFFFFF")
            set_cell_margins(cell, top=70, bottom=70, left=90, right=90)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(val)
            r.font.size = Pt(8)
    set_table_borders(demo_tbl, "CBD5E0")
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------
    # 6. Interactive Student Checkpoint Questions & Answers
    # -------------------------------------------------------------
    add_heading_1(doc, "6. Interactive Student Checkpoint Questions & Answers")

    qa_list = [
        ("Level 1 Checkpoint Question: Cryptographic vs. Perceptual Hashing",
         "Why can't a food delivery platform just compute an MD5 or SHA-256 hash to detect if a customer uploaded a duplicate photo from Google?",
         "Cryptographic hashes exhibit the Avalanche Effect: changing even one single bit produces an unrecognizable hash. When photos are shared via messaging apps, image compression and re-encoding alter thousands of bits without changing visual content. Perceptual Hashing (pHash) solves this by hashing low-frequency visual shapes instead of raw bytes."),
        ("Level 2 Checkpoint Question: PyTorch Tensor Normalization",
         "Why do deep vision networks require image tensors to be normalized with mean and standard deviation instead of raw [0, 255] pixels?",
         "High integer pixel ranges [0, 255] cause activation explosions and erratic gradient updates in convolutional neural networks. Normalizing to zero-mean and unit variance stabilizes backpropagation and ensures balanced gradient flow across color channels."),
        ("Level 2 Checkpoint Question: Temporal Windowing",
         "What is temporal data leakage in customer dispute evidence, and how does the system prevent it?",
         "Temporal leakage occurs when an uploaded photo was taken before the order was placed or weeks after delivery. The system audits the EXIF DateTimeOriginal tag against the delivery window: Δt = t_photo - t_delivery ∈ [0, 7200s]. Negative values or excessive delays flag retroactive claim fraud."),
        ("Level 3 Checkpoint Question: Generative AI Optical Diffraction",
         "If an AI food generator creates a picture that looks 100% photorealistic to human eyes, what physical laws allow our code to detect it?",
         "Optical diffraction through physical glass aperture blades and CMOS sensor Photo-Response Non-Uniformity (PRNU). Real cameras capture silicon photon noise grain and sustained high-frequency Nyquist diffraction, whereas diffusion decoders produce mathematical smoothness and periodic deconvolution checkerboard lattice harmonics."),
        ("Level 3 Checkpoint Question: C2PA Metadata Watermarking",
         "Why do standard Python image EXIF tools fail to detect Meta AI and Google Imagen synthetic provenance stamps?",
         "Standard EXIF tools only inspect legacy TIFF camera header tags. Modern generative models store C2PA provenance inside embedded XMP XML packets (e.g., iptcExt:DigitalSourceType). Detecting these requires explicit XMP parsing or raw byte stream inspection.")
    ]

    for q_title, q_text, a_text in qa_list:
        add_heading_2(doc, q_title)
        pq = doc.add_paragraph()
        rq = pq.add_run(f"Question: \"{q_text}\"")
        rq.bold = True
        rq.font.color.rgb = RGBColor(27, 54, 93)

        pa = doc.add_paragraph()
        ra_label = pa.add_run("Explanation for Students: ")
        ra_label.bold = True
        ra_label.font.color.rgb = RGBColor(40, 116, 50)
        pa.add_run(a_text)

    # -------------------------------------------------------------
    # 7. Verification & Repository Deliverables
    # -------------------------------------------------------------
    add_heading_1(doc, "7. Verification & Repository Deliverables")
    p = doc.add_paragraph()
    p.add_run(
        "All pipelines, models, test suites, and interactive demonstrators referenced in this blueprint are fully functional and verified:\n"
        "• Live Demonstration App: webinar/webinar_demo_app.py (Streamlit on port 8501)\n"
        "• Deep Learning & Forensic Core: backend/app/core/pytorch_pipeline.py\n"
        "• Unit & Forensic Test Suite: tests/test_fraud_cases.py (56/56 passing tests)\n"
        "• Hands-on Student Notebook: webinar/ai_ready_image_preprocessing_walkthrough.ipynb\n"
        "• Remote Git Repository: https://github.com/kushals-bits/Webinar-4.git (master & main branches)"
    )

    doc.save(output_path)
    print(f"Document successfully created at: {output_path}")

if __name__ == "__main__":
    out_dir = r"C:\Users\Roger\AG\zomato-replica-refund-system\webinar"
    out_file = os.path.join(out_dir, "Complete_Webinar_Delivery_Blueprint_From_Zero_Knowledge_to_Production_AI_Forensics.docx")
    build_full_blueprint_doc(out_file)
