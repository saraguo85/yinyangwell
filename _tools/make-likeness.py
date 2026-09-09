"""Regenerate the drawn likeness on the Observation Reading page.

Kept because the drawing still needs the practitioner's sign-off: if she wants a
different crop, a lighter hand or a plainer background, this reruns in a second
rather than being rebuilt from memory.

  python3 _tools/make-likeness.py

Source is _tools/likeness-source-crop.png, already cropped so that the patient
in the original working photograph is outside the frame entirely.

The drawing is built from hatching rather than from a tonal filter, deliberately.
The page calls this a drawn likeness and not a photograph, and a hatched drawing
reads as drawn at any size, where a smooth tonal treatment can still read as a
photograph someone has run a filter over.
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

# Flatten first, so the strokes follow shapes rather than skin texture or the print
# on the fabric.
sm = blur(np.asarray(Image.fromarray(g.astype(np.uint8))
                     .filter(ImageFilter.MedianFilter(7))).astype(np.float32), 1.2)

yy, xx = np.mgrid[0:H, 0:W]
rng = np.random.default_rng(11)
jitter = blur(rng.random((H, W)) * 255, 4) / 255.0 - 0.5   # strokes are not mechanical
breakup = blur(rng.random((H, W)) * 255, 9) / 255.0        # strokes break and restart

def hatch(angle, freq, phase=0.0):
    a = np.deg2rad(angle)
    v = (np.sin((xx * np.cos(a) + yy * np.sin(a)) * freq + jitter * 4 + phase) + 1) / 2
    return np.clip(v + (breakup - 0.5) * 0.55, 0, 1)

# Tone curve. She is sitting in shade, so a faithful mapping puts her face at a
# mid grey: 208 becomes paper, 45 becomes the darkest note, and the gamma lifts
# the face clear of the hair.
dark = np.clip((208 - sm) / (208 - 45), 0, 1) ** 2.2

# Three passes, the way a tone is actually built: a base layer, a crossing layer in
# the mid darks, a third only in the deepest.
ink = np.clip((dark - hatch(38, 0.60)) * 2.2, 0, 1)
ink = np.maximum(ink, np.clip(((dark - 0.42) - hatch(-30, 0.68, 1.7)) * 2.2, 0, 1))
ink = np.maximum(ink, np.clip(((dark - 0.70) - hatch(82, 0.54, 0.4)) * 2.2, 0, 1))

mag = blur(np.hypot(np.gradient(sm, axis=1), np.gradient(sm, axis=0)) * 6, 1.1)
ink = np.maximum(ink, np.clip((mag - 20) * 5 / 255.0, 0, 1) * 0.55)

# Fade the room out. The window sill and the sofa run hard against the left and
# right margins, so those sides fade wider than the top and bottom.
d = np.sqrt(((xx - W * 0.46) / (W * 0.58)) ** 2 + ((yy - H * 0.48) / (H * 0.60)) ** 2)
fade = np.clip(np.maximum(
    np.clip((d - 0.74) / 0.36, 0, 1),
    np.maximum.reduce([np.clip((230 - xx) / 230, 0, 1) ** 1.6,
                       np.clip((xx - (W - 150)) / 150, 0, 1) ** 1.6,
                       np.clip((70 - yy) / 70, 0, 1),
                       np.clip((yy - (H - 90)) / 90, 0, 1)])), 0, 1)
ink = ink * (1 - fade)

grain = (blur(rng.random((H, W)) * 255, 0.6) / 255.0 - 0.5) * 0.04
art = np.clip(1 - ink * 0.92 + grain, 0, 1) * 255
ImageOps.colorize(Image.fromarray(art.astype(np.uint8), 'L'),
                  black=(32, 30, 28), white=(250, 250, 248)).convert('RGB').save(
    OUT, quality=90, optimize=True)
print('wrote', OUT)
