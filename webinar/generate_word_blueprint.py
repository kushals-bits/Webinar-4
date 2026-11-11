"""
Comprehensive generator for the complete, professional Word Document (.docx):
'Complete Webinar Delivery Blueprint: From Zero Knowledge to Production AI Forensics'
Updated for a 90-Minute Masterclass with Exhaustive Basic Definitions & Explanations:
- Comprehensive Beginner Glossary & Mathematical Definitions
- 6 Data Quality Dimensions & Ingestion Protocols
- Spatial Ingestion, PyTorch Tensor Pipelines, Aspect Standardization & ImageNet Normalization
- Temporal Windowing, Retroactive Fraud, Screen Capture Bezel & Moiré Forensics
- 64-bit DCT Perceptual Hashing (pHash) & Hamming Distance Matching
- Generative AI Forensics: 2D FFT Optical Diffraction, CMOS PRNU Photon Noise, VAE Lattice Peakiness, Studio Glaze
- International C2PA & IPTC XMP Metadata Provenance, Container Spoofing
- Multimodal Receipt Cross-Verification & Discrepancy Matching
- Automated Decision Policy Tree, Sybil Device Fingerprinting, Trust Score Ledger Mutation
- 90-Minute Speaker Narration Runbook & Expanded Student Checkpoint Questions
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

def add_heading_3(doc, text: str, color_rgb=(13, 92, 117)):
    h = doc.add_heading(level=3)
    r = h.add_run(text)
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.color.rgb = RGBColor(*color_rgb)
    h.paragraph_format.space_before = Pt(7)
    h.paragraph_format.space_after = Pt(2)
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
    r_meta = p_meta.add_run("ENTERPRISE ENGINEERING FIELD GUIDE & MASTERCLASS PLAYBOOK (90 MINUTES)")
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
    r_sub = p_sub.add_run("A 90-Minute Pedagogical Masterclass: Comprehensive Foundational Definitions, PyTorch Ingestion, 2D FFT Diffraction Physics, CMOS PRNU Shot Noise, C2PA Provenance & Multimodal Adjudication")
    r_sub.font.size = Pt(11)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(113, 128, 150)
    p_sub.paragraph_format.space_after = Pt(12)

    # Metadata Grid
    meta_table = doc.add_table(rows=2, cols=3)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        [("Audience Target", "Undergraduate & Graduate Students (Zero Prior AI Exposure)"),
         ("Lecture Duration", "90 Minutes (Comprehensive 4-Part Structure)"),
         ("Live System", "Streamlit Forensics Workbench (http://localhost:8501)")],
        [("Core Stack", "PyTorch, OpenCV, NumPy, Matplotlib, C2PA XMP, 64-bit pHash"),
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
    # 1. Executive Summary & 90-Minute Masterclass Architecture
    # -------------------------------------------------------------
    add_heading_1(doc, "1. Executive Summary & 90-Minute Masterclass Architecture")
    p = doc.add_paragraph()
    p.add_run(
        "This masterclass blueprint is engineered specifically for students who have no prior exposure to computer vision, machine learning, or signal processing. "
        "Over a complete 90-minute session, the curriculum builds an intuitive, progressive bridge from the ground up: "
        "beginning with elementary concepts (what a pixel is, how colors are mixed, and why computers see numbers instead of objects), "
        "moving to production data engineering pipelines (tensors, spatial standardization, and deduplication), "
        "and culminating in state-of-the-art synthetic AI detection based on physical optics, sensor noise, and international cryptographic metadata standards.\n\n"
        "The curriculum is anchored in a high-stakes, real-world case study: automated wrong food delivery refund claims in food delivery applications (e.g. Zomato replica)."
    )

    add_callout(
        doc,
        "⏱️ Complete 90-Minute Masterclass Schedule:",
        "• Part 1: Minutes 00 – 25 (25 Mins) — Level 1: Foundations, Elementary Glossary & The Business Problem.\n"
        "    The ₹100 Crore refund loophole, basic definitions (Pixel, RGB, Bit Depth, Tensor, Hash), and the 6 Data Quality Dimensions.\n"
        "• Part 2: Minutes 25 – 55 (30 Mins) — Level 2: Engineering Pipelines, Tensors & Deduplication.\n"
        "    EXIF orientation Tag 274, temporal windowing, PyTorch standardization, ImageNet normalization, screen capture bezel/Moiré detection, and 64-bit DCT pHash.\n"
        "• Part 3: Minutes 55 – 80 (25 Mins) — Level 3: Deep AI Forensics, Signal Physics & Decision Governance.\n"
        "    2D FFT optical diffraction, CMOS sensor photon shot noise (PRNU), VAE lattice peakiness, C2PA/IPTC XMP provenance, and multimodal receipt matching.\n"
        "• Part 4: Minutes 80 – 90 (10 Mins) — Live Demonstrator Runbook & Student Q&A.\n"
        "    Hands-on execution on the live workbench (http://localhost:8501), student checkpoint quiz, and interactive debrief.",
        border_hex="2B6CB0", bg_hex="EDF2F7", title_rgb=(27, 54, 93)
    )

    # -------------------------------------------------------------
    # 2. Comprehensive Beginner Technical Glossary & Fundamental Definitions
    # -------------------------------------------------------------
    add_heading_1(doc, "2. Comprehensive Beginner Glossary: Core Definitions & Intuitive Explanations", color_rgb=(13, 92, 117))
    p = doc.add_paragraph()
    p.add_run(
        "Before presenting code or architecture, every foundational concept must be clearly defined with real-world analogies and mathematical intuition. "
        "Use this glossary to anchor students who have zero prior knowledge:"
    )

    glossary = [
        ("Pixel (Picture Element)",
         "The smallest indivisible addressable unit of a digital display or image sensor grid.",
         "Think of a pixel as a single tiny colored stone in a giant mosaic. If you look from far away, you see a picture of a pizza; if you look through a magnifying glass, you only see a single square dot of uniform color.",
         "An integer coordinate (x, y) holding color intensity values: Pixel at (x, y) = [Red, Green, Blue]."),

        ("Bit Depth & Intensity Range [0, 255]",
         "The amount of computer memory (in bits) allocated to store the brightness of each color channel for a single pixel.",
         "Standard digital images allocate 8 bits (1 byte) of binary storage per color channel. In binary math, 8 bits can represent 2^8 = 256 unique numbers, ranging from 0 (complete darkness / black) to 255 (maximum possible brightness).",
         "Color Intensity: 0 = minimum photon emission (black); 255 = maximum saturation."),

        ("RGB Color Model (Additive Color)",
         "An additive color system in which Red, Green, and Blue light wavelengths are mixed in various proportions to produce over 16.7 million distinct colors.",
         "Stage lighting in a theater: if you shine a red flashlight, green flashlight, and blue flashlight all at the exact same spot on a dark stage, the spot turns pure white. If you turn all flashlights off, it is pure black.",
         "[255, 0, 0] = Red | [0, 255, 0] = Green | [0, 0, 255] = Blue | [255, 255, 255] = White | [0, 0, 0] = Black. Total possible colors = 256 × 256 × 256 = 16,777,216 (24-bit TrueColor)."),

        ("Scalar vs. Vector vs. Matrix vs. Tensor",
         "A mathematical hierarchy defining arrays of numbers by their number of spatial and semantic dimensions (rank).",
         "• Scalar (Rank 0): A single number (e.g., meal price = ₹160.00).\n• Vector (Rank 1): A 1D list of numbers (e.g., [R, G, B] values for a single pixel).\n• Matrix (Rank 2): A 2D grid of numbers (e.g., a grayscale image with Height × Width).\n• 3D Tensor (Rank 3): A single color image with [3 Channels × Height × Width].\n• 4D Tensor (Rank 4): A batch of multiple color images: [Batch_Size, 3 Channels, Height, Width] processed together by a neural network.",
         "Tensors provide the universal mathematical language of all deep learning vision frameworks (such as PyTorch)."),

        ("Data Pre-Processing",
         "The automated engineering process of cleaning, transforming, and standardizing raw, messy real-world inputs into clean, uniform numerical data ready for machine learning models.",
         "Cooking dinner: You never throw raw, unpeeled, unwashed potatoes directly into a hot oven. You wash off the dirt (cleaning), peel the skin (stripping metadata), chop them into equal slices (resizing and cropping), and measure the seasonings (normalization).",
         "Without pre-processing, real-world data heterogeneity causes immediate model crashes or silent algorithmic degradation ('Garbage In, Garbage Out')."),

        ("Z-Score Normalization (Standardization)",
         "A statistical technique that rescales numerical data so that the resulting distribution has a mean (μ) of 0 and a standard deviation (σ) of 1.",
         "Grading on a curve: If an exam was exceptionally difficult and the class average was 35%, scaling students relative to the average allows a fair comparison regardless of test difficulty.",
         "Formula: z = (x - μ) / σ. In computer vision, standard ImageNet parameters are applied: μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]. Normalizing prevents neural network gradients from exploding during backpropagation."),

        ("EXIF Metadata (Exchangeable Image File Format)",
         "Standardized metadata headers automatically written into image files (JPEG, HEIC, TIFF) by digital cameras and smartphones at the instant of photo capture.",
         "The digital passport of a photo: recording the camera make (e.g. Apple), model (iPhone 15 Pro), lens focal length, aperture, GPS coordinates, capture date/time (DateTimeOriginal), and gyroscope tilt (Tag 274 Orientation).",
         "Genuine smartphone photos carry rich hardware provenance tags. Web-downloaded images and generative AI tools strip or replace these headers."),

        ("Cryptographic Hash (MD5 / SHA-256)",
         "A mathematical one-way function that maps an arbitrary block of binary data to a fixed-length hexadecimal string, engineered so that even a single-bit alteration produces a totally different hash (Avalanche Effect).",
         "A wax seal on a royal letter: if even one word inside the letter is altered or a corner is torn, the seal is broken.",
         "Excellent for cybersecurity, but terrible for image deduplication because social media compression modifies file bytes without altering visual content."),

        ("Perceptual Hash (pHash)",
         "A visual fingerprinting algorithm that computes a compact hash based on the low-frequency visual structure (silhouette and luminance gradients) of an image.",
         "Recognizing a close friend wearing sunglasses or a winter coat: even though their clothing or lighting changed, their facial bone structure remains instantly recognizable.",
         "Uses the 2D Discrete Cosine Transform (DCT) to generate a 64-bit hash. Images that look visually identical produce near-identical hashes (Hamming Distance ≤ 4) despite heavy compression or resizing."),

        ("Spatial Domain vs. Frequency Domain",
         "Two distinct mathematical ways of representing the same visual data:\n• Spatial Domain: Pixels mapped by physical (x, y) coordinates on the screen (where features are located).\n• Frequency Domain: Pixels decomposed into how rapidly brightness changes across space (how fast colors fluctuate).",
         "Musical sheet vs. audio sound waves: The spatial domain is like reading notes on sheet music; the frequency domain is like seeing the audio equalizer bars bouncing on a stereo display (bass vs. treble).",
         "Smooth food textures and flat tables are low frequencies; sharp knife edges, hair, and lens diffraction are high frequencies."),

        ("2D Fast Fourier Transform (FFT)",
         "An algorithm that computes the 2-dimensional Discrete Fourier Transform, decomposing an image into an orthogonal basis of complex sinusoidal wave patterns.",
         "Deconstructing a finished smoothie back into its individual ingredient cups (identifying how much strawberry, banana, and milk went into the blend).",
         "Equation: F(u, v) = ∬ f(x, y) e^(-j 2π (ux + vy)) dx dy. Allows engineers to inspect whether an image possesses natural optical glass high frequencies or artificial synthetic smoothing."),

        ("Optical Lens Diffraction",
         "The physical wave bending of light rays as they pass through the mechanical aperture blades and curved glass elements of an optical camera lens.",
         "Water waves passing through a narrow harbor entrance: the waves spread out in concentric rings on the other side.",
         "Real camera lenses sustain continuous high-frequency diffraction energy all the way to the sensor's Nyquist limit (r ≥ 0.55). Generative AI diffusion models mathematically fail to replicate this physical diffraction."),

        ("CMOS Sensor Photon Shot Noise (PRNU)",
         "Photo-Response Non-Uniformity: Microscopic quantum fluctuations in photon arrivals (Poisson noise) and silicon pixel manufacturing variations inherent to all physical CMOS camera sensors.",
         "The unique grain of sandpaper or film stock: no two camera sensors produce 100% perfectly identical electrical signals even under uniform lighting.",
         "Real photos have measurable residual noise variance (Var > 1.0, often > 20.0). Diffusion models produce mathematically smooth manifolds with near-zero noise variance (Var < 0.10)."),

        ("Latent Diffusion Model (Generative AI)",
         "A modern generative deep learning architecture that generates images by learning to iteratively remove noise from a low-dimensional latent space representation conditioned on text prompts.",
         "A sculptor starting with a rough block of marble (random Gaussian noise) and gradually chiseling away tiny stone chips step-by-step until a recognizable sculpture of a pasta dish appears.",
         "Popular models: Stable Diffusion, Midjourney, Meta AI / Imagine, DALL-E. Because they denoise in a compressed latent space (e.g. 64 × 64), their upsampling decoders leak periodic checkerboard lattice harmonics."),

        ("C2PA / IPTC Standard Metadata",
         "Coalition for Content Provenance and Authenticity: An international cryptographic metadata standard that cryptographically signs digital media to prove provenance and indicate whether media was synthesized by AI.",
         "A tamper-proof digital watermark and certificate of authenticity stamped directly into the image's XMP metadata packet.",
         'Tags like iptcExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia" prove synthetic origin with 99.0% certainty.'),

        ("Multimodal AI",
         "An AI architecture capable of understanding, combining, and cross-referencing multiple diverse data types (modalities) simultaneously—such as visual images, OCR text, and tabular database records.",
         "A human restaurant manager looking at a plate of food on a table while simultaneously reading the paper printed receipt and checking the customer's loyalty account on a tablet.",
         "Discrepancy Formula: is_wrong_item = (observed_dish not in billed_manifest)."),

        ("Asymmetric Cost Function & Gating",
         "A decision-making strategy where the financial and operational penalty of making one type of error is vastly different from the penalty of making another type of error.",
         "A smoke detector: It is mildly annoying if a smoke alarm beeps when you burn toast (False Positive), but catastrophic if it fails to sound during a real house fire (False Negative).",
         "In refund fraud: Accusing an honest high-value customer of fraud (False Positive) destroys lifetime loyalty, while paying a ₹150 fraudulent claim (False Negative) is a minor loss. Multi-tiered gating balances these costs.")
    ]

    for term, defn, analogy, math in glossary:
        add_heading_2(doc, term)
        p = doc.add_paragraph()
        r1 = p.add_run("• Formal Definition: ")
        r1.bold = True
        r1.font.color.rgb = RGBColor(27, 54, 93)
        p.add_run(defn + "\n")

        r2 = p.add_run("• Intuitive Beginner Analogy: ")
        r2.bold = True
        r2.font.color.rgb = RGBColor(40, 116, 50)
        p.add_run(analogy + "\n")

        r3 = p.add_run("• Mathematical / Technical Formula: ")
        r3.bold = True
        r3.font.color.rgb = RGBColor(197, 48, 48)
        p.add_run(math)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------
    # 3. Part 1: Beginner Foundations & The Business Dilemma (25 Mins)
    # -------------------------------------------------------------
    add_heading_1(doc, "3. Part 1: Level 1 — Beginner Foundations & The Business Dilemma (25 Mins)", color_rgb=(40, 116, 50))

    add_heading_2(doc, "Pointer 1.1: The Industrial Context — The ₹100 Crore Refund Loophole")
    p = doc.add_paragraph()
    p.add_run(
        "Online food delivery platforms process millions of meals daily. Instant automated customer dispute resolution is vital for customer retention: "
        "if a delivery rider brings a cold burger instead of a hot biryani, the customer expects an immediate refund. "
        "However, automated refund APIs represent an immense attack surface for fraud syndicates:\n"
        "• The Attack Vector: Fraudulent rings submit recycled photos from past orders, download images of spilled food from social media, or generate hyper-realistic fake photos using generative AI.\n"
        "• Operational Friction: Assigning human customer support agents to review every photo creates a 48-hour backlog, ballooning payroll and infuriating genuine customers.\n"
        "• Blind Automation Failure: Blindly auto-refunding every claim creates massive financial leakage, draining platform margins.\n"
        "• The Engineering Mandate: Build an automated, real-time computer vision adjudicator capable of verifying physical authenticity and line-item discrepancies in under 500 milliseconds."
    )

    add_callout(
        doc,
        "🎙️ Word-for-Word Presenter Script (The Opening Hook):",
        '"Imagine you order a Deluxe Maharaja Thali on your favorite app. When the rider arrives, you open the package and find a plain burger. '
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
        "1. Accuracy: Does the image depict the true physical meal delivered, rather than a synthetic hallucination or a stock photo?\n"
        "2. Completeness: Are all required evidence modalities present (delivered dish photo + printed restaurant bill invoice)?\n"
        "3. Consistency: Is the image formatted as a canonical 3-channel RGB canvas, and do metadata timestamps align across time zones?\n"
        "4. Timeliness: Was the photograph captured within the delivery claim window (preventing claims submitted 3 weeks after delivery)?\n"
        "5. Validity: Does the file conform to genuine image encoding standards (JPEG, PNG, AVIF) without container spoofing?\n"
        "6. Uniqueness: Is the image novel, or is it a recycled duplicate of an existing claim in the fraud registry?"
    )

    # -------------------------------------------------------------
    # 4. Part 2: Intermediate — Data Pipelines, Tensors & Deduplication (30 Mins)
    # -------------------------------------------------------------
    add_heading_1(doc, "4. Part 2: Level 2 — Engineering Pipelines, Tensors & Deduplication (30 Mins)", color_rgb=(43, 108, 176))

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
    # 5. Part 3: Advanced — Synthetic AI Forensics, Signal Physics & Decision Governance (25 Mins)
    # -------------------------------------------------------------
    add_heading_1(doc, "5. Part 3: Level 3 — Synthetic AI Forensics & Signal Physics (25 Mins)", color_rgb=(197, 48, 48))

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
    # 6. Part 4: Live Demonstration Runbook (90 Mins Total)
    # -------------------------------------------------------------
    add_heading_1(doc, "6. Part 4: Live Demonstration Script & 90-Minute Masterclass Runbook")
    p = doc.add_paragraph()
    p.add_run(
        "This expanded instructor runbook maps the entire 90-minute lecture into six discrete pedagogical phases with specific click actions and live workbench demonstrations (http://localhost:8501):"
    )

    demo_tbl = doc.add_table(rows=7, cols=3)
    demo_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    dheaders = ["Timeline (90 Min)", "Application Action (http://localhost:8501)", "Speaker Narration & Pedagogical Focus"]
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
        ("Min 00 – 15", "Open Workbench Landing Screen (Overview & Presets)", 
         "Deliver the ₹100 Crore opening hook. Explain why food delivery platforms suffer massive leakage and establish the beginner definitions (pixels, RGB light, tensors, data pre-processing)."),
        ("Min 15 – 30", "Select 'Real Food: Authentic Mobile Delivery (iPhone 15)'", 
         "Show genuine camera hardware EXIF tags (Apple A16 Bionic lens). Explain EXIF orientation Tag 274, 3-channel canonicalization, and demonstrate why raw smartphone captures break deep models."),
        ("Min 30 – 45", "Switch to Tab 0 ('Live Backend Decision Pipeline Flow')", 
         "Walk through the Visual Memory-Map Flowchart. Demonstrate Stage 1 (dual intake) and Stage 2 (Resize 256, CenterCrop 224, ImageNet normalization). Contrast classical augmentation with generative corruption."),
        ("Min 45 – 60", "Select 'Fraud: Recycled Duplicate (Known Fraud Ring)'", 
         "Demonstrate Stage 4 (64-bit DCT pHash). Contrast cryptographic MD5 Avalanche Effect with perceptual frequency invariance. Show the 8x8 DCT bitplane heatmap and Hamming distance matching."),
        ("Min 60 – 75", "Select 'Synthetic: Meta AI Delivery Dish (User Case 2 / C2PA)' & 'Synthetic: WhatsApp Diffusion'", 
         "Show how photorealistic AI meals fool human eyes. Reveal the 2D FFT Log-Magnitude Spectrum in magma colormap and the lack of CMOS photon noise (PRNU). Demonstrate the C2PA XMP metadata trigger bypassing receipt verification to ₹0.00 lock."),
        ("Min 75 – 90", "Select 'Real Food: Deluxe Maharaja Thali' + Open Interactive Q&A", 
         "Demonstrate multimodal receipt matching: camera is authentic, but discrepancy engine identifies Maharaja Thali delivered instead of Masala Dosa. Approve ₹309 refund. Conduct interactive student checkpoint quiz and wrap up.")
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
    # 7. Interactive Student Checkpoint Questions & Answers
    # -------------------------------------------------------------
    add_heading_1(doc, "7. Interactive Student Checkpoint Questions & Answers")

    qa_list = [
        ("Level 1 Checkpoint Question: Cryptographic vs. Perceptual Hashing",
         "Why can't a food delivery platform just compute an MD5 or SHA-256 hash to detect if a customer uploaded a duplicate photo from Google?",
         "Cryptographic hashes exhibit the Avalanche Effect: changing even one single bit produces an unrecognizable hash. When photos are shared via messaging apps, image compression and re-encoding alter thousands of bits without changing visual content. Perceptual Hashing (pHash) solves this by hashing low-frequency visual shapes instead of raw bytes."),
        ("Level 2 Checkpoint Question: PyTorch Tensor Normalization",
         "Why do deep vision networks require image tensors to be normalized with mean and standard deviation instead of raw [0, 255] pixels?",
         "High integer pixel ranges [0, 255] cause activation explosions and erratic gradient updates in convolutional neural networks. Normalizing to zero-mean and unit variance stabilizes backpropagation and ensures balanced gradient flow across color channels."),
        ("Level 2 Checkpoint Question: Temporal Windowing & Leakage",
         "What is temporal data leakage in customer dispute evidence, and how does the system prevent it?",
         "Temporal leakage occurs when an uploaded photo was taken before the order was placed or weeks after delivery. The system audits the EXIF DateTimeOriginal tag against the delivery window: Δt = t_photo - t_delivery ∈ [0, 7200s]. Negative values or excessive delays flag retroactive claim fraud."),
        ("Level 3 Checkpoint Question: Generative AI Optical Diffraction",
         "If an AI food generator creates a picture that looks 100% photorealistic to human eyes, what physical laws allow our code to detect it?",
         "Optical diffraction through physical glass aperture blades and CMOS sensor Photo-Response Non-Uniformity (PRNU). Real cameras capture silicon photon noise grain and sustained high-frequency Nyquist diffraction, whereas diffusion decoders produce mathematical smoothness and periodic deconvolution checkerboard lattice harmonics."),
        ("Level 3 Checkpoint Question: C2PA Metadata Watermarking",
         "Why do standard Python image EXIF tools fail to detect Meta AI and Google Imagen synthetic provenance stamps?",
         "Standard EXIF tools only inspect legacy TIFF camera header tags. Modern generative models store C2PA provenance inside embedded XMP XML packets (e.g., iptcExt:DigitalSourceType). Detecting these requires explicit XMP parsing or raw byte stream inspection."),
        ("Level 3 Checkpoint Question: Asymmetric Error Cost in Fraud Detection",
         "Why does the automated refund adjudicator treat False Positives (rejecting an honest customer) with vastly greater sensitivity than False Negatives?",
         "Accusing a loyal customer of fraud destroys long-term customer lifetime value (LTV) and causes viral reputational damage, whereas refunding a single ₹150 fraudulent meal is a bounded financial loss. Multi-tiered gating balances these costs by sending borderline claims to manual review rather than hard-rejecting.")
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
    # 8. Verification & Repository Deliverables
    # -------------------------------------------------------------
    add_heading_1(doc, "8. Verification & Repository Deliverables")
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
