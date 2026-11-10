"""
Generate Realistic Test Images for the AI-Ready Data Pre-processing Webinar.

Creates:
1. real_mobile_delivery.jpg   - Authentic mobile camera food photo with valid EXIF metadata
2. ai_generated_food.jpg      - AI-generated / synthetic food photo (hyper-smooth, no EXIF)
3. screen_capture_food.jpg    - Photograph of a screen with bezel and Moiré pattern
4. recycled_duplicate.jpg     - Previously filed duplicate image (known pHash)
5. delivery_receipt_bill.jpg  - Delivery bill receipt with order details
"""
import io
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import piexif

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_real_mobile_image():
    """Generates an authentic looking food photo with EXIF metadata (iPhone 15)."""
    width, height = 800, 600
    img = Image.new("RGB", (width, height), color=(240, 235, 225))
    draw = ImageDraw.Draw(img)

    # Background: wooden table texture
    for y in range(0, height, 4):
        tint = int(30 + 10 * math.sin(y / 15.0))
        draw.line([(0, y), (width, y)], fill=(180 - tint, 130 - tint, 90 - tint))

    # Plate (ceramic off-white)
    draw.ellipse([150, 80, 650, 520], fill=(245, 245, 240), outline=(210, 210, 205), width=6)
    draw.ellipse([180, 110, 620, 490], fill=(235, 235, 230))

    # Butter chicken curry bowl / curry splash
    draw.ellipse([240, 160, 560, 440], fill=(215, 80, 30))
    # Chicken pieces & butter glaze
    draw.ellipse([280, 200, 370, 290], fill=(235, 110, 45))
    draw.ellipse([400, 220, 500, 310], fill=(230, 105, 40))
    draw.ellipse([340, 300, 440, 380], fill=(240, 120, 50))
    # Cream swirl
    draw.arc([300, 210, 480, 370], start=30, end=270, fill=(255, 255, 235), width=8)

    # Naan bread on the side
    draw.polygon([(100, 350), (220, 300), (260, 440), (140, 490)], fill=(225, 195, 140))
    draw.ellipse([150, 340, 180, 370], fill=(120, 80, 40))  # char blister
    draw.ellipse([200, 390, 225, 415], fill=(110, 70, 35))

    # Add natural camera sensor noise (Gaussian noise)
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 7.0, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(arr)

    # Attach EXIF metadata
    zeroth_ifd = {
        piexif.ImageIFD.Make: b"Apple",
        piexif.ImageIFD.Model: b"iPhone 15",
        piexif.ImageIFD.Software: b"iOS 17.5.1",
        piexif.ImageIFD.DateTime: b"2026:11:09 18:42:15",
    }
    exif_ifd = {
        piexif.ExifIFD.DateTimeOriginal: b"2026:11:09 18:42:15",
        piexif.ExifIFD.DateTimeDigitized: b"2026:11:09 18:42:15",
        piexif.ExifIFD.LensModel: b"iPhone 15 back camera 5.96mm f/1.6",
    }
    exif_bytes = piexif.dump({"0th": zeroth_ifd, "Exif": exif_ifd})

    path = os.path.join(OUTPUT_DIR, "real_mobile_delivery.jpg")
    noisy_img.save(path, "JPEG", quality=92, exif=exif_bytes)
    print(f"Saved real mobile photo: {path}")


def create_ai_generated_image():
    """Generates a synthetic / AI-generated staged food photo (hyper-smooth, no EXIF)."""
    width, height = 800, 600
    img = Image.new("RGB", (width, height), color=(15, 15, 20))
    draw = ImageDraw.Draw(img)

    # Synthetic studio lighting backdrop: perfect radial gradient
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - width / 2) ** 2 + (y - height / 2) ** 2)
    gradient = 255 - np.clip(dist_from_center * 0.35, 0, 240)

    r_ch = (gradient * 0.15).astype(np.uint8)
    g_ch = (gradient * 0.18).astype(np.uint8)
    b_ch = (gradient * 0.25).astype(np.uint8)
    bg = np.dstack([r_ch, g_ch, b_ch])
    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)

    # Hyper-staged, symmetrical gourmet burger
    cx, cy = 400, 320
    # Perfect bun top
    draw.pieslice([cx - 200, cy - 180, cx + 200, cy + 30], start=180, end=360, fill=(210, 140, 60))
    # Sesame seeds perfectly spaced
    for sx, sy in [(330, 200), (370, 185), (410, 180), (450, 185), (480, 205), (380, 220), (430, 220)]:
        draw.ellipse([sx - 4, sy - 2, sx + 4, sy + 2], fill=(255, 250, 230))
    # Lettuce (neon green, sharp vectors)
    draw.rectangle([cx - 210, cy + 10, cx + 210, cy + 40], fill=(70, 200, 50))
    # Tomato (glossy red)
    draw.rectangle([cx - 190, cy + 40, cx + 190, cy + 70], fill=(230, 30, 20))
    # Cheese melt (curving yellow triangles)
    draw.polygon([(cx - 190, cy + 70), (cx + 190, cy + 70), (cx + 170, cy + 120), (cx - 150, cy + 115)], fill=(255, 200, 10))
    # Patty
    draw.rectangle([cx - 195, cy + 85, cx + 195, cy + 130], fill=(90, 45, 25))
    # Bun bottom
    draw.rectangle([cx - 185, cy + 130, cx + 185, cy + 175], fill=(200, 130, 50))

    # Hyper-smooth filtering: zero camera grain, perfectly uniform surfaces
    img = img.filter(ImageFilter.SMOOTH_MORE)

    # Notice: NO EXIF metadata attached whatsoever! (Synthetic generator behavior)
    path = os.path.join(OUTPUT_DIR, "ai_generated_food.jpg")
    img.save(path, "JPEG", quality=98)
    print(f"Saved AI-generated photo: {path}")


def create_screen_capture_image():
    """Generates a photo taken of a computer/tablet screen (bezel + Moiré pattern)."""
    width, height = 800, 600
    # Outer dark bezel of a laptop or tablet screen
    img = Image.new("RGB", (width, height), color=(20, 20, 22))
    draw = ImageDraw.Draw(img)

    # Inside display area (offset by 35px bezel on all sides)
    bezel = 40
    disp_w = width - 2 * bezel
    disp_h = height - 2 * bezel

    # Inner display content: web browser showing food
    display = Image.new("RGB", (disp_w, disp_h), color=(240, 240, 245))
    d_draw = ImageDraw.Draw(display)
    # Browser chrome mock
    d_draw.rectangle([0, 0, disp_w, 35], fill=(220, 220, 225))
    d_draw.ellipse([10, 10, 22, 22], fill=(240, 80, 80))
    d_draw.ellipse([30, 10, 42, 22], fill=(240, 180, 50))
    d_draw.ellipse([50, 10, 62, 22], fill=(80, 200, 80))
    d_draw.rectangle([80, 6, disp_w - 20, 28], fill=(255, 255, 255))

    # Pizza on screen
    d_draw.ellipse([120, 70, disp_w - 120, disp_h - 40], fill=(210, 150, 70))
    d_draw.ellipse([140, 90, disp_w - 140, disp_h - 60], fill=(240, 190, 80))
    for px, py in [(200, 180), (320, 150), (260, 260), (380, 230), (220, 320)]:
        d_draw.ellipse([px, py, px + 40, py + 40], fill=(180, 30, 25))

    # Paste display inside bezel
    img.paste(display, (bezel, bezel))

    # Superimpose high-frequency periodic Moiré interference pattern
    arr = np.array(img, dtype=np.float32)
    y_coords, x_coords = np.ogrid[:height, :width]
    # Interference sinusoidal grid mimicking camera sensor beating with LCD pixel grid
    moire = 18.0 * np.sin(x_coords / 2.3) * np.sin(y_coords / 2.3)
    arr = np.clip(arr + moire[:, :, np.newaxis], 0, 255).astype(np.uint8)
    final_img = Image.fromarray(arr)

    path = os.path.join(OUTPUT_DIR, "screen_capture_food.jpg")
    final_img.save(path, "JPEG", quality=90)
    print(f"Saved screen-capture photo: {path}")


def create_recycled_duplicate_image():
    """Generates an image that simulates a recycled / duplicate claim image."""
    width, height = 800, 600
    img = Image.new("RGB", (width, height), color=(245, 240, 230))
    draw = ImageDraw.Draw(img)

    # Sushi platter
    draw.rectangle([100, 120, 700, 480], fill=(40, 40, 45))
    # Sushi rolls
    for i, x in enumerate([180, 280, 380, 480, 580]):
        draw.ellipse([x, 220, x + 70, 340], fill=(245, 245, 250))
        draw.ellipse([x + 15, 245, x + 55, 315], fill=(220, 60, 50))  # salmon core
        draw.ellipse([x + 22, 265, x + 48, 295], fill=(80, 190, 80))  # avocado

    # Ginger & wasabi
    draw.ellipse([150, 380, 210, 420], fill=(240, 160, 180))  # pink pickled ginger
    draw.ellipse([240, 385, 290, 425], fill=(130, 210, 60))   # green wasabi

    path = os.path.join(OUTPUT_DIR, "recycled_duplicate.jpg")
    img.save(path, "JPEG", quality=90)
    print(f"Saved recycled duplicate photo: {path}")


def create_delivery_receipt_bill():
    """Generates a realistic printed restaurant receipt."""
    width, height = 500, 700
    img = Image.new("RGB", (width, height), color=(250, 250, 248))
    draw = ImageDraw.Draw(img)

    # Header
    draw.text((150, 40), "PUNJABI DHABA", fill=(20, 20, 20))
    draw.text((120, 65), "Authentic North Indian Cuisine", fill=(70, 70, 70))
    draw.text((140, 85), "Order # ORD-2026-001", fill=(20, 20, 20))
    draw.text((130, 105), "Date: 09-Nov-2026 18:25", fill=(60, 60, 60))

    # Divider line
    draw.line([(40, 135), (460, 135)], fill=(150, 150, 150), width=1)
    draw.text((45, 145), "ITEM                        QTY   PRICE", fill=(40, 40, 40))
    draw.line([(40, 170), (460, 170)], fill=(150, 150, 150), width=1)

    # Line items
    draw.text((45, 185), "Butter Chicken               2    640.00", fill=(30, 30, 30))
    draw.text((45, 215), "Garlic Naan                  4    220.00", fill=(30, 30, 30))
    draw.text((45, 245), "Dal Makhani                  1    220.00", fill=(30, 30, 30))

    draw.line([(40, 280), (460, 280)], fill=(150, 150, 150), width=1)
    draw.text((45, 295), "Item Total:                     1080.00", fill=(30, 30, 30))
    draw.text((45, 320), "Taxes & Packaging:                50.00", fill=(60, 60, 60))
    draw.text((45, 345), "Delivery Fee:                     49.00", fill=(60, 60, 60))
    draw.line([(40, 375), (460, 375)], fill=(80, 80, 80), width=2)
    draw.text((45, 390), "GRAND TOTAL:                   Rs 1079.00", fill=(10, 10, 10))
    draw.line([(40, 420), (460, 420)], fill=(80, 80, 80), width=2)

    # Barcode
    for bx in range(80, 420, 6):
        thickness = 2 if (bx % 12 == 0) else 4
        draw.line([(bx, 480), (bx, 540)], fill=(30, 30, 30), width=thickness)
    draw.text((170, 550), "* ORD-2026-001 *", fill=(60, 60, 60))

    # Paper texture subtle noise
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 4.0, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    final_img = Image.fromarray(arr)

    path = os.path.join(OUTPUT_DIR, "delivery_receipt_bill.jpg")
    final_img.save(path, "JPEG", quality=90)
    print(f"Saved delivery receipt bill: {path}")


if __name__ == "__main__":
    print("Generating webinar sample images...")
    create_real_mobile_image()
    create_ai_generated_image()
    create_screen_capture_image()
    create_recycled_duplicate_image()
    create_delivery_receipt_bill()
    print("All sample images generated successfully!")

