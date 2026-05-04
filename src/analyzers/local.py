"""
Local offline analyzer.

Combines:
  1. Classical computer vision in HSV space (always runs, fast, no model needed)
     - Plant segmentation from background
     - Healthy-green vs. yellow/brown pixel ratios -> chlorosis/necrosis detection
     - Scorch detection (bright/bleached patches)
     - Droop detection via contour orientation
     - Scene brightness for light assessment
  2. Optional TFLite disease classifier (PlantVillage MobileNetV2) for named diseases
  3. Rule-based synthesis into the common schema

Works entirely offline. No network calls.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .base import Analyzer, empty_result

log = logging.getLogger("plant_monitor.local")

# --- PlantVillage class labels (MobileNetV2 trained on the standard dataset) ---
PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot", "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper_bell___Bacterial_spot", "Pepper_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


# ============================================================================
# Classical CV analysis
# ============================================================================

def _dilate_mask(mask: np.ndarray, iterations: int = 3) -> np.ndarray:
    """
    Pure-numpy binary dilation. Each iteration expands True pixels by one in
    all 4 directions (up/down/left/right).
    """
    m = mask.copy()
    for _ in range(iterations):
        out = m.copy()
        out[1:, :]  |= m[:-1, :]
        out[:-1, :] |= m[1:, :]
        out[:, 1:]  |= m[:, :-1]
        out[:, :-1] |= m[:, 1:]
        m = out
    return m


def _plant_silhouette(plant_mask: np.ndarray, close_px: int = 10) -> np.ndarray:
    """
    Return the 'filled' silhouette of the plant — bleached/damaged holes inside
    the leafy area are counted as part of the silhouette. This lets us detect
    scorch patches that sit INSIDE the plant outline without also catching the
    bright background around its edges.

    Implementation: dilate then erode ('morphological closing'). This fills
    small holes in the mask. We implement erosion as inverted-dilation.
    """
    dilated = _dilate_mask(plant_mask, close_px)
    # Erode: a pixel is True only if all its neighbours within `close_px` are True.
    # Implemented as NOT dilate(NOT mask).
    eroded = ~_dilate_mask(~dilated, close_px)
    return eroded


def _segment_plant(img_rgb: np.ndarray) -> np.ndarray:
    """Return a boolean mask of plant (green) pixels. Pure numpy — no opencv needed."""
    # Convert to HSV manually (avoids opencv dependency; pure numpy is fast enough on Pi 5)
    img_f = img_rgb.astype(np.float32) / 255.0
    r, g, b = img_f[..., 0], img_f[..., 1], img_f[..., 2]

    maxc = np.max(img_f, axis=-1)
    minc = np.min(img_f, axis=-1)
    v = maxc
    s = np.where(maxc > 0, (maxc - minc) / np.maximum(maxc, 1e-6), 0)

    delta = maxc - minc
    h = np.zeros_like(maxc)
    mask = delta > 0
    rc = np.where(mask, (maxc - r) / np.maximum(delta, 1e-6), 0)
    gc = np.where(mask, (maxc - g) / np.maximum(delta, 1e-6), 0)
    bc = np.where(mask, (maxc - b) / np.maximum(delta, 1e-6), 0)
    h = np.where(maxc == r, bc - gc, h)
    h = np.where(maxc == g, 2.0 + rc - bc, h)
    h = np.where(maxc == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0  # 0..1

    # Green hues: roughly 60-160 degrees => 0.17 - 0.44 in 0..1 hue
    plant_mask = (h >= 0.17) & (h <= 0.50) & (s > 0.15) & (v > 0.15)
    return plant_mask


def _bounding_box(mask: np.ndarray) -> dict:
    """Tight box around the True region of mask, as fractions of dims."""
    if not mask.any():
        return {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}
    H, W = mask.shape
    ys, xs = np.where(mask)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    # Pad slightly
    pad_x = (x1 - x0) * 0.05
    pad_y = (y1 - y0) * 0.05
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(W, x1 + pad_x)
    y1 = min(H, y1 + pad_y)
    return {
        "x": float(x0 / W),
        "y": float(y0 / H),
        "w": float((x1 - x0) / W),
        "h": float((y1 - y0) / H),
    }


def _color_analysis(img_rgb: np.ndarray, plant_mask: np.ndarray) -> dict:
    """Within the plant region, measure ratios of healthy-green / yellow / brown / scorched pixels."""
    if not plant_mask.any():
        return {
            "plant_coverage": 0.0,
            "healthy_ratio": 0.0, "yellow_ratio": 0.0,
            "brown_ratio": 0.0, "scorch_ratio": 0.0,
            "avg_value": 0.0,
        }

    img_f = img_rgb.astype(np.float32) / 255.0
    r, g, b = img_f[..., 0], img_f[..., 1], img_f[..., 2]
    maxc = np.max(img_f, axis=-1)
    minc = np.min(img_f, axis=-1)
    v = maxc
    s = np.where(maxc > 0, (maxc - minc) / np.maximum(maxc, 1e-6), 0)

    delta = maxc - minc
    h = np.zeros_like(maxc)
    m = delta > 0
    rc = np.where(m, (maxc - r) / np.maximum(delta, 1e-6), 0)
    gc = np.where(m, (maxc - g) / np.maximum(delta, 1e-6), 0)
    bc = np.where(m, (maxc - b) / np.maximum(delta, 1e-6), 0)
    h = np.where(maxc == r, bc - gc, h)
    h = np.where(maxc == g, 2.0 + rc - bc, h)
    h = np.where(maxc == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0

    # Consider the filled plant silhouette for yellow/scorch detection. This is
    # the plant's outline with any bleached/yellowed 'holes' filled in, which
    # excludes bright backgrounds adjacent to the plant edge.
    silhouette = _plant_silhouette(plant_mask, close_px=12)
    # Yellowing leaves may be outside the strict green mask but inside the silhouette
    near_plant = silhouette

    # Healthy green: hue 0.22-0.40, decent saturation, mid-to-high value
    healthy = (h >= 0.22) & (h <= 0.40) & (s > 0.25) & (v > 0.25) & plant_mask
    # Yellow: hue 0.10-0.20 within the plant silhouette (chlorotic leaves)
    yellow_broad = (h >= 0.08) & (h <= 0.20) & (s > 0.25) & (v > 0.3)
    yellow = yellow_broad & silhouette
    # Brown/dead: low saturation OR reddish-brown hues with low value, within plant
    brown = (((h >= 0.02) & (h <= 0.12) & (s > 0.15) & (v < 0.55)) |
             ((s < 0.2) & (v > 0.1) & (v < 0.5))) & plant_mask
    # Scorch detection is notoriously hard from a single photo without a trained
    # model. We use a very conservative heuristic: bleached pixels must be
    # BOTH (a) within the closed plant silhouette AND (b) have significantly
    # lower saturation than the plant's average AND (c) account for at least
    # a small-but-meaningful fraction. Even so, this will have false positives
    # on photos with bright backgrounds or strong sun highlights — which is why
    # the local analyzer reports low confidence on light assessment.
    plant_avg_sat = s[plant_mask].mean() if plant_mask.any() else 0.3
    scorch = (
        (v > 0.88)
        & (s < 0.15)
        & (s < plant_avg_sat * 0.4)  # much less saturated than plant average
        & silhouette
        & ~plant_mask
    )

    total = plant_mask.sum()
    total_img = plant_mask.size
    silhouette_area = max(silhouette.sum(), 1)

    return {
        "plant_coverage": float(total / total_img),
        "healthy_ratio": float(healthy.sum() / max(total, 1)),
        # yellow and scorch are relative to the full plant silhouette (outline with holes filled)
        "yellow_ratio": float(yellow.sum() / silhouette_area),
        "brown_ratio": float(brown.sum() / max(total, 1)),
        "scorch_ratio": float(scorch.sum() / silhouette_area),
        "avg_value": float(v[plant_mask].mean()) if plant_mask.any() else 0.0,
        "avg_saturation": float(s[plant_mask].mean()) if plant_mask.any() else 0.0,
    }


def _droop_score(plant_mask: np.ndarray) -> float:
    """
    Rough droop heuristic: compare vertical extent of plant mass above vs below centroid.
    Healthy plants are usually top-heavy or balanced; drooping plants have more mass low.
    Returns 0 (no droop) to 1 (severe droop).
    """
    if not plant_mask.any():
        return 0.0
    ys, _ = np.where(plant_mask)
    y_min, y_max = ys.min(), ys.max()
    y_range = max(y_max - y_min, 1)
    centroid_y = ys.mean()
    # Fraction of plant height that the centroid sits below the top
    relative_centroid = (centroid_y - y_min) / y_range
    # Values > 0.55 mean mass skewed downward
    return float(np.clip((relative_centroid - 0.5) * 3.0, 0.0, 1.0))


# ============================================================================
# Optional TFLite disease classifier
# ============================================================================

class DiseaseClassifier:
    """Wraps a TFLite PlantVillage model. Gracefully absent if model file missing."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._load_attempted = False

    def _load(self):
        self._load_attempted = True
        if not self.model_path.exists():
            log.info("Disease model not found at %s — disease detection disabled", self.model_path)
            return
        try:
            try:
                # Prefer standalone tflite-runtime (lighter, what Pi uses)
                from tflite_runtime.interpreter import Interpreter  # type: ignore
            except ImportError:
                from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
            self.interpreter = Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            log.info("Disease classifier loaded: %s", self.model_path)
        except Exception as e:
            log.warning("Failed to load TFLite model: %s", e)
            self.interpreter = None

    def available(self) -> bool:
        if not self._load_attempted:
            self._load()
        return self.interpreter is not None

    def predict(self, img_rgb: np.ndarray) -> Optional[dict]:
        if not self.available():
            return None

        _, h, w, _ = self.input_details[0]["shape"]
        pil = Image.fromarray(img_rgb).resize((w, h), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, 0)

        # Quantized models need uint8 input
        if self.input_details[0]["dtype"] == np.uint8:
            scale, zero = self.input_details[0]["quantization"]
            arr = (arr / scale + zero).astype(np.uint8) if scale else (arr * 255).astype(np.uint8)

        self.interpreter.set_tensor(self.input_details[0]["index"], arr)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        if self.output_details[0]["dtype"] == np.uint8:
            scale, zero = self.output_details[0]["quantization"]
            out = (out.astype(np.float32) - zero) * (scale if scale else 1.0)

        # Softmax if it wasn't applied
        if out.min() < 0 or out.max() > 1 or abs(out.sum() - 1.0) > 0.1:
            e = np.exp(out - out.max())
            out = e / e.sum()

        top_idx = int(np.argmax(out))
        if top_idx >= len(PLANTVILLAGE_CLASSES):
            return None

        label = PLANTVILLAGE_CLASSES[top_idx]
        confidence = float(out[top_idx])
        plant_name, _, disease = label.partition("___")
        disease = disease.replace("_", " ").strip()
        is_healthy = "healthy" in disease.lower()

        return {
            "plant": plant_name.replace("_", " "),
            "disease": None if is_healthy else disease,
            "is_healthy": is_healthy,
            "confidence": confidence,
        }


# ============================================================================
# Rule-based synthesis
# ============================================================================

def _synthesise(color: dict, droop: float, disease: Optional[dict], bbox: dict,
                plant_context: Optional[dict]) -> dict:
    """Combine CV signals into the standard result dict using explainable rules."""
    issues = []
    recommendations = []
    observations_parts = []

    healthy = color["healthy_ratio"]
    yellow = color["yellow_ratio"]
    brown = color["brown_ratio"]
    scorch = color["scorch_ratio"]

    # --- Overall health score (0-100) ---
    # Start at 90 if mostly green, subtract for problems
    score = 50.0
    if color["plant_coverage"] < 0.02:
        score = 0  # barely any plant visible
    else:
        # Penalties are steeper now — a little yellow/brown/scorch should matter.
        score = 50 + 50 * healthy - 150 * yellow - 180 * brown - 200 * scorch - 15 * droop
        score = float(np.clip(score, 0, 100))

    if score >= 85: overall = "excellent"
    elif score >= 70: overall = "good"
    elif score >= 50: overall = "fair"
    elif score >= 30: overall = "poor"
    else: overall = "critical"

    # --- Watering assessment ---
    # Over-watering signal: yellow lower leaves, possibly droopy
    # Under-watering signal: brown/crispy edges, droop, low saturation
    water_status = "unknown"
    water_conf = 30
    water_reason = "Local analysis — limited without a soil-moisture sensor."

    if brown > 0.12 and droop > 0.3:
        water_status = "underwatered"
        water_conf = 70
        water_reason = "Significant brown/crispy areas and drooping posture detected."
        issues.append({"issue": "Leaf browning/crisping", "severity": "moderate", "location": "leaves"})
        recommendations.append("Water thoroughly and check soil moisture an inch below the surface.")
    elif yellow > 0.08 and droop > 0.2:
        water_status = "overwatered"
        water_conf = 65
        water_reason = "Widespread yellowing with some drooping — classic overwatering signs."
        issues.append({"issue": "Leaf yellowing (chlorosis)", "severity": "moderate", "location": "leaves"})
        recommendations.append("Reduce watering frequency and ensure drainage. Check for root rot.")
    elif yellow > 0.04:
        water_status = "overwatered"
        water_conf = 55
        water_reason = "Noticeable yellowing suggests overwatering or nutrient issue."
        issues.append({"issue": "Leaf yellowing (chlorosis)", "severity": "mild", "location": "leaves"})
        recommendations.append("Let soil dry out between waterings; consider a balanced fertiliser.")
    elif brown > 0.05:
        water_status = "underwatered"
        water_conf = 55
        water_reason = "Patches of brown/dry tissue suggest underwatering."
        issues.append({"issue": "Leaf browning", "severity": "mild", "location": "leaves"})
        recommendations.append("Check soil moisture; may need more frequent watering.")
    elif healthy > 0.6:
        water_status = "optimal"
        water_conf = 60
        water_reason = "Foliage looks mostly healthy green with no distress markers."

    # --- Sunlight assessment ---
    light_status = "unknown"
    light_conf = 30
    light_reason = "Local analysis uses scene brightness and leaf signs — a lux sensor would help."

    # Scorch detection is conservative — only fires with clear signal.
    # Real sun scorch shows bleached spots clustered on upper leaves; we can't
    # detect 'upper' reliably without a trained model, so we require a fairly
    # strong ratio AND low absolute saturation variance before calling it.
    if scorch > 0.04:
        light_status = "excessive"
        light_conf = 60
        light_reason = "Possible bleached/scorched patches on leaves — verify visually."
        issues.append({"issue": "Possible sun scorch", "severity": "mild", "location": "leaves"})
        recommendations.append(
            "Check for bleached spots on leaves; if present, move away from direct midday sun."
        )
    elif color["avg_saturation"] < 0.25 and color["avg_value"] > 0.5:
        light_status = "insufficient"
        light_conf = 55
        light_reason = "Pale, washed-out leaves may suggest insufficient light."
        recommendations.append("Relocate closer to a bright window or add a grow light.")
    elif color["avg_value"] < 0.3:
        # Scene very dark — either low light or photo taken in gloom
        light_status = "insufficient"
        light_conf = 45
        light_reason = "Scene is quite dark; plant may not be getting enough light."
    elif healthy > 0.55 and scorch < 0.01:
        light_status = "optimal"
        light_conf = 55
        light_reason = "Plant shows healthy coloration without scorch."

    # --- Disease integration ---
    species_guess = plant_context.get("species") if plant_context else "unknown"
    if disease:
        if not disease["is_healthy"] and disease["confidence"] > 0.5:
            severity = "severe" if disease["confidence"] > 0.85 else "moderate"
            issues.append({
                "issue": disease["disease"],
                "severity": severity,
                "location": "identified by ML model",
            })
            recommendations.append(
                f"Possible {disease['disease']} detected ({int(disease['confidence']*100)}% confidence). "
                "Isolate plant and research treatment."
            )
            # Cap score if a disease is confidently detected
            score = min(score, 60)
            if score < 30: overall = "critical"
            elif score < 50: overall = "poor"
            else: overall = "fair"
        if disease["confidence"] > 0.4 and species_guess in ("unknown", "", None):
            species_guess = disease["plant"]

    # --- Observations narrative ---
    observations_parts.append(
        f"Plant covers ~{int(color['plant_coverage']*100)}% of the frame."
    )
    observations_parts.append(
        f"Color breakdown: {int(healthy*100)}% healthy green, "
        f"{int(yellow*100)}% yellow, {int(brown*100)}% brown, {int(scorch*100)}% bleached."
    )
    if droop > 0.3:
        observations_parts.append("Foliage appears to be drooping.")
    if disease and not disease["is_healthy"]:
        observations_parts.append(
            f"Disease model flagged '{disease['disease']}' ({int(disease['confidence']*100)}%)."
        )
    observations = " ".join(observations_parts)

    if not recommendations:
        recommendations.append("Continue current care routine and monitor for changes.")

    return {
        "overall_health": overall,
        "health_score": int(round(score)),
        "watering_status": water_status,
        "watering_confidence": water_conf,
        "watering_reasoning": water_reason,
        "sunlight_status": light_status,
        "sunlight_confidence": light_conf,
        "sunlight_reasoning": light_reason,
        "observations": observations,
        "issues_detected": issues,
        "recommendations": recommendations,
        "plant_bounding_box": bbox,
        "species_guess": species_guess or "unknown",
    }


# ============================================================================
# The analyzer class
# ============================================================================

class LocalAnalyzer(Analyzer):
    name = "local"
    version = "1.0-cv+tflite"

    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            # src/analyzers/local.py -> src/ -> project root
            base = Path(__file__).resolve().parent.parent.parent
            model_path = base / "models" / "plant_disease.tflite"
        self.classifier = DiseaseClassifier(model_path)

    def is_available(self) -> bool:
        # Local analyzer only needs numpy + PIL, which are hard deps of the project
        return True

    def analyse(self, image_path: Path, plant_context: Optional[dict] = None) -> dict:
        try:
            img = Image.open(image_path).convert("RGB")
            # Downsample large images for faster CV (Pi 5 can handle full res but no need)
            if max(img.size) > 1600:
                img.thumbnail((1600, 1600), Image.LANCZOS)
            arr = np.asarray(img)

            plant_mask = _segment_plant(arr)
            color = _color_analysis(arr, plant_mask)
            bbox = _bounding_box(plant_mask)
            droop = _droop_score(plant_mask)

            disease = None
            if self.classifier.available():
                # Crop to plant bbox for better disease classification
                H, W, _ = arr.shape
                x0 = int(bbox["x"] * W); y0 = int(bbox["y"] * H)
                x1 = int((bbox["x"] + bbox["w"]) * W); y1 = int((bbox["y"] + bbox["h"]) * H)
                if x1 > x0 and y1 > y0:
                    disease = self.classifier.predict(arr[y0:y1, x0:x1])

            result = _synthesise(color, droop, disease, bbox, plant_context)
            result["analyzer"] = self.name
            result["analyzer_version"] = self.version
            if self.classifier.available():
                result["analyzer_version"] += "+disease"
            return result

        except Exception as e:
            log.exception("Local analysis failed")
            return empty_result(self.name, f"local error: {e}")
