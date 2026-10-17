"""
Synthetic Postal Address & Envelope Generator.
Generates realistic postal address lines and full mail envelopes with stamps,
postmarks, textures, and line-level bounding box ground truths.
"""

import os
import random
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Realistic Postal Data Banks
TITLES = ["To:", "To", "Mr.", "Mrs.", "Ms.", "Dr.", "Prof."]
FIRST_NAMES = [
    "Aarav", "Sunita", "Rajesh", "Priya", "Vikram", "Ananya", "Rohan", "Sneha",
    "Arjun", "Kavita", "Deepak", "Pooja", "Sanjay", "Meera", "Amit", "Neha",
    "John", "Sarah", "David", "Emily", "Michael", "Emma", "Robert", "Olivia"
]
LAST_NAMES = [
    "Sharma", "Patel", "Verma", "Rao", "Nair", "Iyer", "Gupta", "Singh",
    "Reddy", "Deshmukh", "Chatterjee", "Kulkarni", "Mehta", "Bose", "Joshi",
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis"
]

PREMISES = [
    "Flat No. 102", "Flat No. 304", "House # 12", "Villa 7", "Plot No. 45",
    "Block B-201", "Door No. 14/2", "Building 5", "Apt 4B", "Suite 210",
    "House 88", "Flat 502, Wing A", "Quarter No. 16", "Room 303"
]

BUILDINGS = [
    "Sunshine Heights", "Green Meadows", "Orchid Residency", "Shanti Niwas",
    "Silver Oak Apts", "Royal Palms", "Galaxy Enclave", "Lotus Tower",
    "Park View Apts", "Maple Woods", "Gokul Vihar", "Cyber City Towers"
]

STREETS = [
    "MG Road", "Brigade Road", "Park Avenue", "Gandhi Marg", "Station Road",
    "Nehru Street", "Link Road", "Ring Road", "Airport Road", "Church Street",
    "Indira Nagar 1st Cross", "Hill Road", "Lake View Drive", "Commercial Street"
]

LOCALITIES = [
    "Near City Metro Station", "Opp. Central Bank", "Behind Town Hall",
    "Sector 14", "Phase 2", "Koramangala 5th Block", "Andheri West",
    "Jayanagar 4th Block", "Banjara Hills", "Connaught Place", "Alwarpet"
]

CITIES_STATES_PINS = [
    ("Bengaluru", "Karnataka", "560001"),
    ("Bengaluru", "Karnataka", "560034"),
    ("Mumbai", "Maharashtra", "400001"),
    ("Mumbai", "Maharashtra", "400053"),
    ("New Delhi", "Delhi", "110001"),
    ("New Delhi", "Delhi", "110025"),
    ("Hyderabad", "Telangana", "500001"),
    ("Hyderabad", "Telangana", "500081"),
    ("Chennai", "Tamil Nadu", "600001"),
    ("Chennai", "Tamil Nadu", "600028"),
    ("Kolkata", "West Bengal", "700001"),
    ("Kolkata", "West Bengal", "700091"),
    ("Pune", "Maharashtra", "411001"),
    ("Ahmedabad", "Gujarat", "380001"),
    ("Jaipur", "Rajasthan", "302001"),
    ("Lucknow", "Uttar Pradesh", "226001")
]

# Common Windows fonts to search for
FONT_CANDIDATES = [
    "arial.ttf", "arialbd.ttf", "calibri.ttf", "calibrib.ttf",
    "times.ttf", "timesbd.ttf", "tahoma.ttf", "tahomabd.ttf",
    "verdana.ttf", "consola.ttf", "georgia.ttf", "comic.ttf"
]


def get_available_fonts() -> List[str]:
    """Find valid TrueType font files from Windows font directory."""
    font_dir = r"C:\Windows\Fonts"
    available = []
    if os.path.exists(font_dir):
        for candidate in FONT_CANDIDATES:
            path = os.path.join(font_dir, candidate)
            if os.path.exists(path):
                available.append(path)
    return available if available else ["arial.ttf"]


def load_font(font_path: Optional[str] = None, size: int = 20) -> ImageFont.ImageFont:
    """Load a font with fallback to default PIL font."""
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass

    # Try available system fonts
    for f in get_available_fonts():
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            continue

    return ImageFont.load_default()


def generate_random_address_record() -> Dict[str, str]:
    """Generate a structured address record dictionary."""
    recipient = f"{random.choice(TITLES)} {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    line2 = f"{random.choice(PREMISES)}, {random.choice(BUILDINGS)}"
    line3 = f"{random.choice(STREETS)}, {random.choice(LOCALITIES)}"

    city, state, pin = random.choice(CITIES_STATES_PINS)
    format_choice = random.choice([
        f"{city}, {state} - {pin}",
        f"{city}, {state} {pin}",
        f"{city} - {pin}, {state}",
        f"{city}, PIN: {pin}"
    ])

    return {
        "recipient": recipient,
        "line2": line2,
        "line3": line3,
        "city_state_pin": format_choice,
        "city": city,
        "state": state,
        "pin": pin
    }


def render_text_line_image(
    text: str,
    font_path: Optional[str] = None,
    font_size: int = 22,
    target_height: int = 32,
    noise_level: float = 0.05,
    add_slant: bool = False
) -> Tuple[Image.Image, str]:
    """
    Renders a single address text line into an image normalized to target_height (e.g., 32px).
    Returns (PIL.Image in Grayscale 'L', ground_truth_text).
    """
    font = load_font(font_path, font_size)

    # Estimate text dimensions
    dummy_img = Image.new("RGB", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = max(10, bbox[2] - bbox[0])
    text_h = max(10, bbox[3] - bbox[1])

    # Canvas with padding
    pad_x = random.randint(8, 16)
    pad_y = random.randint(4, 8)
    canvas_w = text_w + pad_x * 2
    canvas_h = text_h + pad_y * 2

    # Background color (slight paper variation: 235 - 255)
    bg_val = random.randint(235, 255)
    img = Image.new("L", (canvas_w, canvas_h), color=bg_val)
    draw = ImageDraw.Draw(img)

    # Ink color (dark grey to black: 0 - 45)
    ink_val = random.randint(0, 45)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, fill=ink_val, font=font)

    # Optional slant (affine transform)
    if add_slant and random.random() < 0.35:
        shear_factor = random.uniform(-0.15, 0.15)
        img = img.transform(
            img.size,
            Image.AFFINE,
            (1, shear_factor, 0, 0, 1, 0),
            fillcolor=bg_val
        )

    # Add subtle gaussian blur or noise
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))

    # Add salt and pepper / sensor noise
    arr = np.array(img, dtype=np.float32)
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * 255, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    img = Image.fromarray(arr)

    # Resize preserving aspect ratio so height = target_height
    curr_w, curr_h = img.size
    aspect = curr_w / float(curr_h)
    new_w = max(16, int(round(target_height * aspect)))
    img = img.resize((new_w, target_height), Image.Resampling.BILINEAR)

    return img, text


def render_postal_envelope(
    address_record: Optional[Dict[str, str]] = None,
    width: int = 700,
    height: int = 440,
    font_path: Optional[str] = None
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Renders an authentic, realistic postal mail envelope with:
    - Envelope background texture (ivory/kraft paper tone)
    - Postage stamp in upper right corner with serrated border
    - Circular rubber postmark cancellation mark
    - Return address block in upper left
    - Recipient address block centered / lower right
    - Text lines with exact bounding box annotations
    """
    if address_record is None:
        address_record = generate_random_address_record()

    # Base paper background color (warm ivory / cream tone)
    bg_r = random.randint(242, 250)
    bg_g = random.randint(238, 246)
    bg_b = random.randint(228, 238)
    img = Image.new("RGB", (width, height), color=(bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)

    # 1. Subtle Paper Texture & Envelope Border
    # Draw faint border
    border_color = (max(0, bg_r - 25), max(0, bg_g - 25), max(0, bg_b - 25))
    draw.rectangle([10, 10, width - 10, height - 10], outline=border_color, width=1)

    # 2. Postage Stamp (Top-Right)
    stamp_w, stamp_h = 75, 95
    stamp_x = width - stamp_w - 30
    stamp_y = 25

    # Stamp serrated white base
    stamp_bg = (random.randint(180, 230), random.randint(70, 120), random.randint(70, 120))  # Postage red/crimson
    draw.rectangle([stamp_x, stamp_y, stamp_x + stamp_w, stamp_y + stamp_h], fill=(255, 255, 255))
    draw.rectangle([stamp_x + 3, stamp_y + 3, stamp_x + stamp_w - 3, stamp_y + stamp_h - 3], fill=stamp_bg)

    # Inner stamp text
    stamp_font = load_font(font_path, 11)
    draw.text((stamp_x + 8, stamp_y + 8), "POSTAGE", fill=(255, 255, 255), font=stamp_font)
    draw.text((stamp_x + 15, stamp_y + 40), "MAIL", fill=(255, 255, 255), font=stamp_font)
    draw.text((stamp_x + 12, stamp_y + 70), "Rs. 25.00", fill=(255, 255, 255), font=stamp_font)

    # 3. Postmark Cancellation Stamp (Black circular rubber stamp over edge of postage stamp)
    pm_cx = stamp_x + 15
    pm_cy = stamp_y + stamp_h // 2
    pm_r = 38
    pm_color = (60, 60, 60)
    draw.ellipse([pm_cx - pm_r, pm_cy - pm_r, pm_cx + pm_r, pm_cy + pm_r], outline=pm_color, width=2)
    draw.ellipse([pm_cx - pm_r + 6, pm_cy - pm_r + 6, pm_cx + pm_r - 6, pm_cy + pm_r - 6], outline=pm_color, width=1)
    pm_font = load_font(font_path, 9)
    draw.text((pm_cx - 28, pm_cy - 6), "GPO SORTING", fill=pm_color, font=pm_font)
    # Cancellation wavy lines
    for offset_y in [-12, -4, 4, 12]:
        draw.line([pm_cx + pm_r - 5, pm_cy + offset_y, pm_cx + pm_r + 45, pm_cy + offset_y], fill=pm_color, width=1)

    # 4. Return Address (Top-Left corner)
    ret_font = load_font(font_path, 12)
    ret_x, ret_y = 35, 25
    draw.text((ret_x, ret_y), "From:", fill=(80, 80, 80), font=ret_font)
    draw.text((ret_x, ret_y + 16), "Dept of Logistics, Hub 4", fill=(80, 80, 80), font=ret_font)
    draw.text((ret_x, ret_y + 32), "Central Postal Terminal", fill=(80, 80, 80), font=ret_font)

    # 5. Destination Address (Center to Lower-Right area)
    # 4 Address lines
    lines = [
        address_record["recipient"],
        address_record["line2"],
        address_record["line3"],
        address_record["city_state_pin"]
    ]

    addr_font_size = random.randint(18, 22)
    addr_font = load_font(font_path, addr_font_size)

    start_x = random.randint(int(width * 0.32), int(width * 0.40))
    start_y = random.randint(int(height * 0.38), int(height * 0.44))
    line_spacing = int(addr_font_size * 1.6)

    line_annotations = []
    current_y = start_y

    for idx, text in enumerate(lines):
        bbox = draw.textbbox((start_x, current_y), text, font=addr_font)
        # Add slight padding to line bbox
        pad = 4
        line_bbox = (
            max(0, bbox[0] - pad),
            max(0, bbox[1] - pad),
            min(width, bbox[2] + pad),
            min(height, bbox[3] + pad)
        )

        # Ink text rendering with natural contrast
        ink = (random.randint(10, 35), random.randint(10, 35), random.randint(10, 45))
        draw.text((start_x, current_y), text, fill=ink, font=addr_font)

        line_annotations.append({
            "line_index": idx,
            "text": text,
            "bbox": line_bbox  # (x1, y1, x2, y2)
        })

        current_y += line_spacing

    # Add realistic envelope effects: noise, subtle blur
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.35))

    metadata = {
        "record": address_record,
        "lines": line_annotations,
        "envelope_size": (width, height),
        "address_block_bbox": (
            min(l["bbox"][0] for l in line_annotations),
            min(l["bbox"][1] for l in line_annotations),
            max(l["bbox"][2] for l in line_annotations),
            max(l["bbox"][3] for l in line_annotations)
        )
    }

    return img, metadata
