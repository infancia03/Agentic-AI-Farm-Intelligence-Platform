"""
AgriFarm — Disease Detection Agent
ResNet18 vision + OpenRouter LLM treatment plan.

Two modes:
  1. Fine-tuned (data/plantvillage_resnet18.pt exists) — accurate 38-class detection
  2. Pretrained only — confidence threshold + LLM-based advisory from crop type
"""

from __future__ import annotations
import io
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from loguru import logger

from app.llm_client import detailed_ask

NUM_CLASSES          = 38
CONFIDENCE_THRESHOLD = 0.45

PLANTVILLAGE_CLASSES = [
    "Apple__Apple_scab", "Apple__Black_rot", "Apple__Cedar_apple_rust", "Apple__healthy",
    "Blueberry__healthy", "Cherry__Powdery_mildew", "Cherry__healthy",
    "Corn__Cercospora_leaf_spot", "Corn__Common_rust", "Corn__Northern_Leaf_Blight", "Corn__healthy",
    "Grape__Black_rot", "Grape__Esca", "Grape__Leaf_blight", "Grape__healthy",
    "Orange__Haunglongbing",
    "Peach__Bacterial_spot", "Peach__healthy",
    "Pepper__Bacterial_spot", "Pepper__healthy",
    "Potato__Early_blight", "Potato__Late_blight", "Potato__healthy",
    "Raspberry__healthy", "Soybean__healthy", "Squash__Powdery_mildew",
    "Strawberry__Leaf_scorch", "Strawberry__healthy",
    "Tomato__Bacterial_spot", "Tomato__Early_blight", "Tomato__Late_blight",
    "Tomato__Leaf_Mold", "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites", "Tomato__Target_Spot",
    "Tomato__Yellow_Leaf_Curl_Virus", "Tomato__mosaic_virus", "Tomato__healthy",
]

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model:        Optional[nn.Module] = None
_is_finetuned: bool                = False


def _load_model() -> nn.Module:
    global _model, _is_finetuned
    if _model is not None:
        return _model
    logger.info("Loading ResNet18 ...")
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, NUM_CLASSES))
    ckpt = Path("./data/plantvillage_resnet18.pt")
    if ckpt.exists():
        logger.info(f"Fine-tuned checkpoint found: {ckpt}")
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        _is_finetuned = True
    else:
        logger.warning("No fine-tuned checkpoint. Confidence threshold active at 45%.")
        _is_finetuned = False
    m.eval()
    _model = m
    return m


def _parse(raw: str) -> tuple[str, str]:
    parts = raw.split("__", 1)
    crop    = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    return crop, disease


def classify_image(image_bytes: bytes) -> dict:
    model = _load_model()
    img   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    t     = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0]
    top3_idx  = probs.topk(3).indices.tolist()
    top3_val  = probs.topk(3).values.tolist()
    best_conf = float(top3_val[0])
    raw       = PLANTVILLAGE_CLASSES[top3_idx[0]]
    crop, disease = _parse(raw)
    top3 = [
        {
            "class":      PLANTVILLAGE_CLASSES[i],
            "crop":       _parse(PLANTVILLAGE_CLASSES[i])[0],
            "disease":    _parse(PLANTVILLAGE_CLASSES[i])[1],
            "confidence": round(float(v), 4),
        }
        for i, v in zip(top3_idx, top3_val)
    ]
    return {
        "crop":           crop,
        "disease":        disease,
        "confidence":     round(best_conf, 4),
        "is_healthy":     "healthy" in disease.lower(),
        "raw_class":      raw,
        "top3":           top3,
        "is_finetuned":   _is_finetuned,
        "low_confidence": best_conf < CONFIDENCE_THRESHOLD,
    }


def get_treatment_advice(vision: dict, farm_context: dict | None = None) -> str:
    crop      = vision["crop"]
    disease   = vision["disease"]
    conf      = vision["confidence"]
    low_conf  = vision.get("low_confidence", False)
    finetuned = vision.get("is_finetuned", False)

    # Use farmer's crop hint if provided — more reliable than auto-detect
    crop_hint = crop
    if farm_context:
        farmer_crop = farm_context.get("crop", "")
        if farmer_crop and farmer_crop.lower() not in ("unknown", "auto-detect", ""):
            crop_hint = farmer_crop

    location = (farm_context or {}).get("location", "India")

    # ── LOW CONFIDENCE + no fine-tuned model ─────────────────
    # Don't trust the vision result — give LLM full control
    if low_conf and not finetuned:
        prompt = (
            f"A farmer in {location} has uploaded a leaf image showing visible symptoms.\n"
            f"The vision model confidence is low ({conf:.0%}) so crop identification "
            f"from the image is unreliable.\n"
            f"The farmer's crop: {crop_hint}\n\n"
            "Based on the image showing orange/brown spots, lesions, or discolouration "
            "(describe what you can infer from common disease patterns), provide:\n\n"
            "1. **Most likely disease** for this crop in Indian conditions with confidence\n"
            "2. **Visual symptoms to confirm** (what to look for more carefully)\n"
            "3. **IMMEDIATE action** (what to do today)\n"
            "4. **Treatment** — specific Indian product names (e.g. Dithane M-45, Blitox-50, "
            "Indofil M-45) with dosage and cost estimate in INR\n"
            "5. **Organic option** if available\n"
            "6. **When to call extension officer** (KVK helpline: 1800-180-1551)\n\n"
            "Be practical and specific. Max 300 words."
        )
    # ── HIGH CONFIDENCE or FINE-TUNED ────────────────────────
    elif vision["is_healthy"]:
        prompt = (
            f"A {crop_hint} leaf in {location} appears healthy (model confidence: {conf:.0%}).\n\n"
            "Provide:\n"
            "1. **Health confirmation** — what healthy leaves of this crop should look like\n"
            "2. **Preventive care** this week (3 specific tips)\n"
            "3. **Early warning signs** to watch for in next 2-4 weeks\n"
            "4. **Quick yield tip** for this crop\n\n"
            "Max 200 words."
        )
    else:
        prompt = (
            f"Disease detected: **{disease}** in **{crop_hint}** — {location}\n"
            f"Model confidence: {conf:.0%}\n\n"
            "Provide a structured treatment plan:\n"
            "1. **SEVERITY** — Critical / High / Medium and why\n"
            "2. **IMMEDIATE ACTIONS** in next 24-48 hours\n"
            "3. **CHEMICAL TREATMENT** — Indian brand names, exact dosage, application method, "
            "estimated cost per acre in INR\n"
            "4. **ORGANIC ALTERNATIVE** — neem oil, trichoderma, bio-agents with dosage\n"
            "5. **SPREAD PREVENTION** — isolate, remove, quarantine steps\n"
            "6. **NEXT SEASON** — resistant variety names available in India\n"
            "7. **YIELD LOSS** — estimated % if untreated for 1 week / 2 weeks\n\n"
            "Max 350 words. Be specific to Indian farming conditions."
        )

    return detailed_ask(
        prompt,
        system=(
            "You are a senior plant pathologist with 20 years field experience across Tamil Nadu, "
            "Karnataka, Maharashtra and Andhra Pradesh. You know Indian agri-input product names, "
            "KVK resources, and local farming practices. Give advice that a small farmer can act on "
            "immediately with products available at their local agri-input shop."
        ),
    )


def run(image_bytes: bytes, farm_context: dict | None = None) -> dict:
    vision = classify_image(image_bytes)
    advice = get_treatment_advice(vision, farm_context)

    warning = None
    if not vision["is_finetuned"]:
        if vision["low_confidence"]:
            warning = (
                f"⚠️ Image confidence too low ({vision['confidence']:.0%}) for reliable detection. "
                "Treatment advice is based on your selected crop and common disease patterns — "
                "not solely the image. For accurate detection, run: python data/finetune_model.py"
            )
        else:
            warning = (
                f"⚠️ Model uses ImageNet weights (not fine-tuned on PlantVillage). "
                f"Confidence: {vision['confidence']:.0%}. "
                "Run `python data/finetune_model.py` for accurate disease detection."
            )

    return {
        "vision_result":  vision,
        "crop":           vision["crop"],
        "disease":        vision["disease"],
        "confidence":     vision["confidence"],
        "is_healthy":     vision["is_healthy"],
        "low_confidence": vision["low_confidence"],
        "is_finetuned":   vision["is_finetuned"],
        "top3":           vision["top3"],
        "llm_advice":     advice,
        "warning":        warning,
        "agent":          "disease_agent",
    }
