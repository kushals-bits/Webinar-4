import os
import sys

sys.path.insert(0, r"C:\Users\Roger\AG\zomato-replica-refund-system\backend")

from PIL import Image
import cv2
import numpy as np
from app.security.fraud_detector import extract_exif_metadata, run_fraud_detection, enforce_fraud_decision
from app.security.hashing_service import compute_phash, check_hash_against_db, generate_reverse_image_webhook_mock, hamming_distance

folder = r"C:\Users\Roger\AG\zomato-replica-refund-system\Food images"
files = sorted(os.listdir(folder))

print(f"=== Found {len(files)} real customer food images in '{folder}' ===")

for fn in files:
    fp = os.path.join(folder, fn)
    with open(fp, "rb") as f:
        data = f.read()

    with Image.open(fp) as img:
        size = img.size
        fmt = img.format
        mode = img.mode
        gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    exif = extract_exif_metadata(data)
    phash = compute_phash(data)
    dup = check_hash_against_db(phash)
    rev = generate_reverse_image_webhook_mock(phash)

    report = run_fraud_detection(
        image_bytes=data,
        order_id="ORD-2026-001",
        customer_id=f"cust_{fn[:4].lower()}",
        image_type="food"
    )
    decision = enforce_fraud_decision(report, "ORD-2026-001", f"cust_{fn[:4].lower()}")

    print("-" * 65)
    print(f"File: {fn} [{fmt}, {size[0]}x{size[1]}, {mode}]")
    print(f"  pHash: {phash} (Duplicate in DB: {dup['is_duplicate']})")
    print(f"  Blur Sharpness: {blur:.2f}")
    print(f"  EXIF Present: {exif['raw_exif_present']} | Device: {exif.get('make')} {exif.get('model')} | DTO: {exif.get('datetime_original')}")
    print(f"  Flags: {report['fraud_flags']}")
    print(f"  Fraud Score: {report['fraud_risk_score']:.2f} -> DECISION: {decision['action']}")
    print(f"  User Notice: {decision['message']}")

print("\n=== Pairwise Hamming Distance Across Real Images ===")
hashes = [(fn, compute_phash(open(os.path.join(folder, fn), 'rb').read())) for fn in files]
for i in range(len(hashes)):
    for j in range(i + 1, len(hashes)):
        f1, h1 = hashes[i]
        f2, h2 = hashes[j]
        d = hamming_distance(h1, h2)
        print(f"  {f1} vs {f2}: Hamming Distance = {d}")
