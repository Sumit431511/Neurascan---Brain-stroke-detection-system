"""
gradcam.py v6 — Grad-CAM for NeuraScan
Fixes: 
1. Dynamic classifier head (no hardcoded layer names).
2. Dynamic class index (fixes blank heatmaps on Ischemic/Normal).
3. Logarithmic loss to prevent Vanishing Gradients on Confident Softmax.
"""

import io
import uuid
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

logger = logging.getLogger(__name__)

from config import HEATMAP_DIR


def generate_gradcam(
    model,
    img_array: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> tuple:
    import tensorflow as tf
    import tf_keras
    import matplotlib.cm as cm

    # ── Find nested ResNet50 ──────────────────────────────────────────────────
    resnet_layer = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and 'resnet' in layer.name.lower():
            resnet_layer = layer
            break

    if resnet_layer is None:
        raise ValueError("Could not find nested ResNet50 in model.")

    # ── Find target conv layer ────────────────────────────────────────────────
    TARGET_NAMES = [
        "conv5_block3_out",
        "conv5_block3_3_conv",
        "conv5_block2_out",
        "conv4_block6_out",
    ]
    target_layer = None
    for name in TARGET_NAMES:
        try:
            target_layer = resnet_layer.get_layer(name)
            logger.info("✅ Grad-CAM target layer: %s", name)
            break
        except ValueError:
            continue

    if target_layer is None:
        conv_layers = [l for l in resnet_layer.layers
                       if isinstance(l, tf_keras.layers.Conv2D)]
        if not conv_layers:
            raise ValueError("No Conv2D found inside ResNet50.")
        target_layer = conv_layers[-1]
        logger.info("Fallback Grad-CAM layer: %s", target_layer.name)

    # ── Build sub-model: resnet input → [conv_out, resnet output] ─────────────
    resnet_conv_model = tf_keras.models.Model(
        inputs=resnet_layer.input,
        outputs=[target_layer.output, resnet_layer.output],
        name="resnet_conv_model"
    )

    # ── Build dynamic classifier head: resnet output → final prediction ───────
    # This dynamically extracts all layers AFTER the ResNet50 layer.
    classifier_input = tf_keras.Input(shape=resnet_layer.output_shape[1:])
    x = classifier_input
    start_idx = model.layers.index(resnet_layer) + 1
    for layer in model.layers[start_idx:]:
        x = layer(x)
    classifier_model = tf_keras.Model(classifier_input, x, name="classifier_head")

    # ── Run forward pass in two parts, watching conv_output ───────────────────
    inputs_tf = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        # Part 1: get conv output from resnet sub-model
        conv_outputs, resnet_out = resnet_conv_model(inputs_tf, training=False)

        # Watch conv_outputs (not inputs) — works with numpy-loaded weights
        tape.watch(conv_outputs)

        # Part 2: run classifier head dynamically
        predictions = classifier_model(resnet_out, training=False)

        # THE FIX: Get the dynamically predicted class (0, 1, or 2)
        class_idx = tf.argmax(predictions[0])
        
        # THE FIX: Use log to prevent gradient vanishing on 99.9% confident Softmax
        loss = tf.math.log(predictions[:, class_idx] + 1e-8)

    # Gradient of loss w.r.t. conv_outputs
    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        logger.warning("Gradients are None. Falling back to raw neural activations.")
        heatmap = tf.reduce_mean(conv_outputs, axis=-1)
        heatmap = tf.squeeze(heatmap).numpy()
    else:
        # ── Pool + weight heatmap ─────────────────────────────────────────────
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)
        conv_out     = conv_outputs[0]                         # (H, W, C)
        heatmap      = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap).numpy()
        
        # ReLU (Only keep positive influences)
        heatmap = np.maximum(heatmap, 0)
        
        # Fallback if heatmap is totally blank
        if heatmap.max() == 0:
            logger.warning("Empty heatmap detected. Falling back to raw neural activations.")
            heatmap = tf.reduce_mean(conv_outputs[0], axis=-1).numpy()

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap_raw = heatmap.astype(np.float32)

    # ── Resize heatmap ────────────────────────────────────────────────────────
    h, w = img_array.shape[1], img_array.shape[2]
    heatmap_resized = np.array(
        Image.fromarray((heatmap_raw * 255).astype(np.uint8)).resize(
            (w, h), Image.LANCZOS
        )
    ).astype(np.float32) / 255.0

    # ── Colorise ──────────────────────────────────────────────────────────────
    import cv2
    heatmap_colored = (
        cm.get_cmap(colormap)(heatmap_resized)[:, :, :3] * 255
    ).astype(np.uint8)

    # ── Recover original image ────────────────────────────────────────────────
    orig  = img_array[0].copy()
    orig  = orig * 255.0  # reverse rescale=1./255 used in new model
    orig  = orig[:, :, ::-1]
    orig  = np.clip(orig, 0, 255).astype(np.uint8)

    # ── Blend ─────────────────────────────────────────────────────────────────
    overlay = (
        (1 - alpha) * orig.astype(np.float32) +
        alpha * heatmap_colored.astype(np.float32)
    ).astype(np.uint8)

    logger.info("✅ Grad-CAM heatmap generated successfully.")
    return heatmap_raw, overlay


def save_heatmap(overlay_rgb: np.ndarray, patient_id: str = None) -> Path:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid  = patient_id or str(uuid.uuid4())[:8]
    path = HEATMAP_DIR / f"heatmap_{pid}_{ts}.png"
    Image.fromarray(overlay_rgb).save(str(path), format="PNG")
    logger.info("Heatmap saved: %s", path)
    return path


def heatmap_to_base64(overlay_rgb: np.ndarray) -> str:
    import base64
    buf = io.BytesIO()
    Image.fromarray(overlay_rgb).save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")