"""
core/model.py — Model loading and 3-class inference
=====================================================
Architecture matches new notebook exactly:
  ResNet50 (last 100 layers trainable)
  → GlobalAveragePooling2D
  → Dense(256, relu)
  → BatchNormalization
  → Dropout(0.3)
  → Dense(128, relu)      ← new layer vs old binary model
  → Dropout(0.3)          ← new layer vs old binary model
  → Dense(3, softmax)     ← 3-class output
"""

import logging
import numpy as np
from config import MODEL_WEIGHTS_PATH, CLASS_NAMES

logger = logging.getLogger(__name__)

# Global model instance — loaded once at startup
_model = None


def load_model():
    """Load the 3-class ResNet50 model with trained weights."""
    global _model
    import tf_keras
    from tf_keras.applications import ResNet50
    from tf_keras.layers import (
        Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
    )
    from tf_keras.models import Model

    logger.info("Loading model from %s …", MODEL_WEIGHTS_PATH)

    if not MODEL_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_WEIGHTS_PATH}. "
            "Place model_weights.npy in the models/ directory."
        )

    # Rebuild exact architecture from notebook cell 6
    inputs  = tf_keras.Input(shape=(224, 224, 3))
    base    = ResNet50(weights=None, include_top=False)(inputs)
    x       = GlobalAveragePooling2D()(base)
    x       = Dense(256, activation="relu")(x)
    x       = BatchNormalization()(x)
    x       = Dropout(0.3)(x)
    x       = Dense(128, activation="relu")(x)   # new vs old model
    x       = Dropout(0.3)(x)                    # new vs old model
    outputs = Dense(3, activation="softmax")(x)  # 3-class softmax
    model   = Model(inputs, outputs)

    weights = np.load(str(MODEL_WEIGHTS_PATH), allow_pickle=True)
    model.set_weights(weights)

    _model = model
    logger.info("✅ Model loaded successfully (3-class: Hemorrhagic / Ischemic / Normal).")
    return model


def get_model():
    """Return the loaded model. Raises if not loaded."""
    if _model is None:
        raise RuntimeError("Model is not loaded. Check server startup logs.")
    return _model


def release_model():
    """Release model from memory on shutdown."""
    global _model
    _model = None
    logger.info("Model released.")


def run_inference(img_array: np.ndarray) -> dict:
    """
    Run 3-class prediction on a preprocessed image array.

    Args:
        img_array: shape (1, 224, 224, 3), already scaled to [0, 1]

    Returns:
        dict with all prediction results
    """
    model = get_model()
    probs = model.predict(img_array, verbose=0)[0]  # shape (3,)

    # Class indices: 0=Hemorrhagic, 1=Ischemic, 2=Normal
    hemorrhagic_prob = float(probs[0])
    ischemic_prob    = float(probs[1])
    normal_prob      = float(probs[2])

    predicted_idx  = int(np.argmax(probs))
    predicted_label = CLASS_NAMES[predicted_idx]
    confidence      = float(probs[predicted_idx])

    # Stroke type — any non-Normal result
    is_stroke   = predicted_idx != 2   # 0=Hemorrhagic or 1=Ischemic → stroke
    stroke_type = predicted_label if is_stroke else None

    # Risk level based on confidence of the predicted class
    # and whether it's a stroke
    if not is_stroke:
        risk_level = "Low"
    elif confidence > 0.80:
        risk_level = "High"
    elif confidence > 0.55:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "prediction":       predicted_label,
        "stroke_type":      stroke_type,
        "is_stroke":        is_stroke,
        "confidence":       round(confidence * 100, 2),
        "hemorrhagic_prob": round(hemorrhagic_prob * 100, 2),
        "ischemic_prob":    round(ischemic_prob * 100, 2),
        "normal_prob":      round(normal_prob * 100, 2),
        "risk_level":       risk_level,
        "predicted_class":  predicted_idx,
    }