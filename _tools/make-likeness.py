"""Regenerate the drawn likeness on the Observation Reading page.

Kept because the drawing still needs the practitioner's sign-off: if she wants a
different crop, a lighter hand or a plainer background, this reruns in a second
rather than being rebuilt from memory.

  python3 _tools/make-likeness.py

Source is _tools/likeness-source-crop.png, already cropped so that the patient
in the original working photograph is outside the frame entirely.
"""
import numpy as np
from PIL import Image, ImageFilter, ImageOps

W, H = 1000, 1250
SRC = '_tools/likeness-source-crop.png'
OUT = 'images/practitioner/likeness-drawn.jpg'

g = np.asarray(Image.open(SRC).convert('L').resize((W, H), Image.LANCZOS)).astype(np.float32)

def blur(a, r):
    return np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
                      .filter(ImageFilter.GaussianBlur(r))).astype(np.float32)

# Flatten first, so the drawing follows shapes rather than skin texture or the print
# on the fabric, then take the gradient for the contour.
sm = blur(np.asarray(Image.fromarray(g.astype(np.uint8))
                     .filter(ImageFilter.MedianFilter(9))).astype(np.float32), 1.5)
mag = blur(np.hypot(np.gradient(sm, axis=1), np.gradient(sm, axis=0)) * 6, 1.0)
line = 255 - np.clip((mag - 18) * 6, 0, 255)

# Four flat tones, the way a drawing lays in shadow.
flat = (np.array([0.10, 0.52, 0.80, 1.0]) * 255)[np.digitize(sm, [88, 145, 198])]

# Fade the room out. The window sill and the sofa run hard against the left and
# right margins, so those sides fade wider than the top and bottom.
yy, xx = np.mgrid[0:H, 0:W]
d = np.sqrt(((xx - W * 0.46) / (W * 0.58)) ** 2 + ((yy - H * 0.48) / (H * 0.60)) ** 2)
fade = np.clip(np.maximum(
    np.clip((d - 0.74) / 0.36, 0, 1),
    np.maximum.reduce([np.clip((230 - xx) / 230, 0, 1) ** 1.6,
                       np.clip((xx - (W - 150)) / 150, 0, 1) ** 1.6,
                       np.clip((70 - yy) / 70, 0, 1),
                       np.clip((yy - (H - 90)) / 90, 0, 1)])), 0, 1)

art = np.minimum(line, flat * 0.55 + 255 * 0.45)
art = art + (255 - art) * fade
L = Image.fromarray(np.clip(art, 0, 255).astype(np.uint8), 'L')
ImageOps.colorize(L, black=(20, 20, 20), white=(250, 250, 248)).convert('RGB').save(
    OUT, quality=88, optimize=True)
print('wrote', OUT)
