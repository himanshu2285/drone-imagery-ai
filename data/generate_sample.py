"""
generate_sample.py
------------------
Creates a realistic synthetic Drone_SAMPLE.tiff (1024×1024 RGB, saved as GeoTIFF-style TIFF).
Regions:
  - Vegetation (parks, trees)         → deep greens
  - Water body (river/pond)           → blue tones
  - Soil / bare ground                → brown / tan
  - Built structures (rooftops)       → grey / red-brown
  - Roads (grid)                      → asphalt grey with lane markings
"""

import numpy as np
from PIL import Image
import os

np.random.seed(2024)
SIZE = 1024
img  = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

def noise(arr, sigma=8):
    n = np.random.normal(0, sigma, arr.shape).astype(np.int16)
    return np.clip(arr.astype(np.int16) + n, 0, 255).astype(np.uint8)

def fill(img, y1, x1, y2, x2, rgb, sigma=10):
    region = np.full((y2-y1, x2-x1, 3), rgb, dtype=np.uint8)
    img[y1:y2, x1:x2] = noise(region, sigma)

# ── Background: soil / bare ground ────────────────────────────────────
img[:] = noise(np.full((SIZE, SIZE, 3), [160, 120, 70], np.uint8), 12)

# ── Large vegetation zones ─────────────────────────────────────────────
fill(img,   0,   0, 420, 420, [34, 120, 34],  15)   # NW park
fill(img, 600, 620, 900, 980, [50, 140, 45],  15)   # SE park
fill(img, 700,   0, 950, 250, [40, 130, 40],  12)   # SW treeline
fill(img,   0, 700, 300, 980, [60, 150, 55],  14)   # NE park

# ── Tree clusters (darker green circles) ──────────────────────────────
import cv2
for (cy, cx, r) in [(80,80,50),(200,150,70),(100,300,45),(320,80,55),
                    (150,370,40),(650,700,60),(750,800,80),(820,660,50),
                    (750,120,55),(870,180,45)]:
    cv2.circle(img, (cx, cy), r, (22, 90, 22), -1)
    cv2.circle(img, (cx, cy), r, (18, 75, 18),  3)

# ── Water body (river diagonal + pond) ────────────────────────────────
pts = np.array([[470,0],[530,0],[620,200],[640,400],[600,600],[560,800],[580,1024],[520,1024],[480,800],[500,600],[460,400],[440,200]], np.int32)
cv2.fillPoly(img, [pts], (30, 100, 200))
# Pond
cv2.ellipse(img, (850, 150), (95, 65), 20, 0, 360, (25, 90, 185), -1)
# Water shimmer
for _ in range(400):
    wy = np.random.randint(0, SIZE); wx = np.random.randint(0, SIZE)
    if img[wy, wx, 2] > 150:  # blue pixel
        cv2.line(img, (wx, wy), (wx+np.random.randint(3,12), wy), (180, 210, 240), 1)

# ── Road grid ──────────────────────────────────────────────────────────
road_color = [80, 80, 80]
# Horizontal roads
for y, w in [(420,22),(700,20),(850,18),(130,16),(280,15)]:
    fill(img, y, 0, y+w, SIZE, road_color, 5)
    cv2.line(img, (0, y+w//2), (SIZE, y+w//2), (230,230,180), 1)  # centre line
# Vertical roads
for x, w in [(420,22),(660,20),(200,18),(790,16),(50,15)]:
    fill(img, 0, x, SIZE, x+w, road_color, 5)
    cv2.line(img, (x+w//2, 0), (x+w//2, SIZE), (230,230,180), 1)

# ── Built structures / rooftops ────────────────────────────────────────
building_specs = [
    # (y1,x1,y2,x2, roof_rgb)
    (440, 440, 530, 530, [180, 80,  80]),
    (440, 550, 530, 640, [160, 90,  70]),
    (440, 660, 530, 790, [140, 100, 60]),
    (540, 440, 630, 530, [170, 75,  75]),
    (540, 550, 630, 640, [155, 85,  65]),
    (540, 660, 630, 790, [145, 95,  55]),
    (440, 440, 470, 470, [200, 200, 200]),  # flat white roof
    (450,  60, 540, 180, [120, 100, 180]),  # purple warehouse
    (450, 230, 540, 390, [180, 160, 100]),  # tan factory
    (720, 270, 820, 390, [160,  70,  70]),
    (720,  60, 810, 250, [130, 110,  90]),
    (320, 450, 400, 600, [190, 170, 130]),
    (320, 620, 400, 790, [170, 150, 110]),
    ( 30, 450, 110, 610, [200, 180, 160]),
    ( 30, 640, 110, 800, [180, 160, 140]),
]
for (y1,x1,y2,x2,rgb) in building_specs:
    fill(img, y1, x1, y2, x2, rgb, 8)
    cv2.rectangle(img, (x1,y1), (x2,y2), (40,40,40), 2)  # edge
    # Roof details
    mid_y, mid_x = (y1+y2)//2, (x1+x2)//2
    cv2.rectangle(img, (x1+5,y1+5), (x2-5,y2-5), tuple(max(0,c-20) for c in rgb), 1)

# ── Parking lots ───────────────────────────────────────────────────────
fill(img, 630, 440, 700, 800, [100, 100, 95], 6)
for i in range(440, 800, 22):
    cv2.line(img, (i, 630), (i, 700), (140,140,140), 1)

# ── Final global noise pass ────────────────────────────────────────────
noise_layer = np.random.normal(0, 4, img.shape).astype(np.int16)
img = np.clip(img.astype(np.int16) + noise_layer, 0, 255).astype(np.uint8)

# ── Save as TIFF ───────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), 'Drone_SAMPLE.tiff')
Image.fromarray(img, 'RGB').save(out_path, format='TIFF', compression='raw')
print(f"Saved: {out_path}  shape={img.shape}")