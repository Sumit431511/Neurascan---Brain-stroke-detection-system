"""
core/preprocessing.py — Image preprocessing pipeline
======================================================
MUST exactly match the preprocessing used during training.

New model (3-class) uses rescale=1./255
Old model (binary) used ResNet50 preprocess_input (mean subtraction)
These are DIFFERENT — using the wrong one gives wrong predictions.
"""

import io
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
from config import IMG_SIZE, SCANS_DIR


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image bytes for 3-class CNN inference.

    Matches training notebook exactly:
      ImageDataGenerator(rescale=1./255, ...)
      val_test_datagen = ImageDataGenerator(rescale=1./255)

    Steps:
      1. Open + convert to RGB
      2. Resize to 224×224 (LANCZOS)
      3. Cast to float32
      4. Add batch dimension → shape (1, 224, 224, 3)
      5. Divide by 255.0  ← matches rescale=1./255 in notebook
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)   # explicit float32 cast
    arr = np.expand_dims(arr, axis=0)        # add batch dimension
    arr = arr / 255.0                        # rescale — matches training
    return arr


def save_scan(image_bytes: bytes, original_filename: str) -> str:
    """Save the original scan image to disk. Returns the saved path."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(original_filename).stem
    path = SCANS_DIR / f"{stem}_{ts}.jpg"

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.save(str(path), format="JPEG", quality=90)
    return str(path)