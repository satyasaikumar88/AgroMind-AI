"""
services/inference.py  [UPGRADED v2]

UPGRADES over v1:
  1. Multi-label disease detection — top-K predictions (K=5)
  2. Separate crop identification from disease detection
  3. Confidence gate returns structured uncertain response
  4. Unknown disease safe fallback
  5. Region-aware safety layer (blocks incomplete/wrong-region treatments)
  6. Safety check: blocks any treatment missing dosage or source
  7. Multi-disease structured output — each disease treated separately
  8. All confidence values from real model logits (never invented)

All v1 functionality preserved. Architecture unchanged.
"""

import io
import os
import json
import time
import base64
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ─── PATHS ───────────────────────────────────────────────────────────
VALIDATOR_PATH  = "models/plant_validator.pth"
CLASSIFIER_PATH = "models/plant_classifier.pth"
CLASSES_PATH    = "models/plant_classes.json"
METRICS_PATH    = "models/plant_metrics.json"

# ─── THRESHOLDS ──────────────────────────────────────────────────────
PLANT_THRESHOLD       = 0.60
CONFIDENCE_THRESHOLD  = 0.45
SECONDARY_THRESHOLD   = 0.20   # NEW: include as secondary disease if above this
BLUR_THRESHOLD        = 80.0
TOP_K_PREDICTIONS     = 5      # NEW: always return top-5 from model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ─── BLUR DETECTION (unchanged from v1) ──────────────────────────────
def compute_blur_score(image_bytes: bytes) -> float:
    """Laplacian variance. Pertuz et al. 2013. Higher=sharper."""
    try:
        img    = Image.open(io.BytesIO(image_bytes)).convert("L")
        arr    = np.array(img, dtype=np.float64)
        kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float64)
        h, w   = arr.shape
        padded = np.pad(arr, 1, mode='reflect')
        lap    = np.zeros_like(arr)
        for i in range(3):
            for j in range(3):
                lap += kernel[i,j] * padded[i:i+h, j:j+w]
        return float(np.var(lap))
    except Exception:
        return 0.0


def preprocess_image(image_bytes: bytes) -> Optional[torch.Tensor]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return INFERENCE_TRANSFORM(img).unsqueeze(0)
    except Exception:
        return None


# ─── SYMPTOM MAP (deterministic rule-mapped, not generated) ──────────
SYMPTOM_MAP: Dict[str, Dict] = {
    "Apple___Apple_scab": {"symptoms":["olive-green corky spots","dark scabby lesions on fruit"],"severity":"moderate"},
    "Apple___Black_rot": {"symptoms":["purple to black circular lesions with frog-eye pattern","mummified fruit"],"severity":"severe"},
    "Apple___Cedar_apple_rust": {"symptoms":["bright orange-yellow spots on upper leaf","tubular structures below"],"severity":"moderate"},
    "Apple___healthy": {"symptoms":[],"severity":"none"},
    "Blueberry___healthy": {"symptoms":[],"severity":"none"},
    "Cherry_(including_sour)___Powdery_mildew": {"symptoms":["white powdery coating on leaves","distorted young leaves"],"severity":"mild"},
    "Cherry_(including_sour)___healthy": {"symptoms":[],"severity":"none"},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {"symptoms":["rectangular grey lesions bounded by veins","tan to grey colour"],"severity":"moderate"},
    "Corn_(maize)___Common_rust_": {"symptoms":["small circular brown pustules on both surfaces","powdery rust spores"],"severity":"moderate"},
    "Corn_(maize)___Northern_Leaf_Blight": {"symptoms":["long cigar-shaped greyish-tan lesions 5-15cm","parallel to veins"],"severity":"severe"},
    "Corn_(maize)___healthy": {"symptoms":[],"severity":"none"},
    "Grape___Black_rot": {"symptoms":["circular tan lesions with dark borders","mummified berries"],"severity":"severe"},
    "Grape___Esca_(Black_Measles)": {"symptoms":["tiger-striped chlorosis","dark spots in wood cross-section"],"severity":"severe"},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {"symptoms":["dark brown angular lesions on leaf edges","premature defoliation"],"severity":"moderate"},
    "Grape___healthy": {"symptoms":[],"severity":"none"},
    "Orange___Haunglongbing_(Citrus_greening)": {"symptoms":["asymmetric yellow blotch mottle","small bitter misshapen fruit"],"severity":"severe"},
    "Peach___Bacterial_spot": {"symptoms":["angular water-soaked spots","shot-hole appearance as lesions drop out"],"severity":"moderate"},
    "Peach___healthy": {"symptoms":[],"severity":"none"},
    "Pepper,_bell___Bacterial_spot": {"symptoms":["water-soaked circular spots becoming necrotic","yellowing around lesions"],"severity":"moderate"},
    "Pepper,_bell___healthy": {"symptoms":[],"severity":"none"},
    "Potato___Early_blight": {"symptoms":["concentric ring target-board pattern","dark brown spots on older leaves first"],"severity":"moderate"},
    "Potato___Late_blight": {"symptoms":["dark water-soaked lesions","white sporulation on leaf underside","rapid collapse"],"severity":"severe"},
    "Potato___healthy": {"symptoms":[],"severity":"none"},
    "Raspberry___healthy": {"symptoms":[],"severity":"none"},
    "Soybean___healthy": {"symptoms":[],"severity":"none"},
    "Squash___Powdery_mildew": {"symptoms":["white powdery patches on leaf surface","yellowing and drying"],"severity":"mild"},
    "Strawberry___Leaf_scorch": {"symptoms":["small purple to red spots becoming tan in centre","scorch from leaf tips"],"severity":"moderate"},
    "Strawberry___healthy": {"symptoms":[],"severity":"none"},
    "Tomato___Bacterial_spot": {"symptoms":["small water-soaked spots with yellow halo","shot-hole appearance"],"severity":"moderate"},
    "Tomato___Early_blight": {"symptoms":["concentric target pattern on older leaves","dark brown spots with yellow halo"],"severity":"moderate"},
    "Tomato___Late_blight": {"symptoms":["dark brown water-soaked lesions","white mould on underside","rapid collapse"],"severity":"severe"},
    "Tomato___Leaf_Mold": {"symptoms":["pale green to yellow patches upper leaf","olive-green mould below"],"severity":"moderate"},
    "Tomato___Septoria_leaf_spot": {"symptoms":["circular spots dark border light grey center","tiny dark pycnidia in center"],"severity":"moderate"},
    "Tomato___Spider_mites Two-spotted_spider_mite": {"symptoms":["stippling and bronzing of leaves","fine webbing on undersides"],"severity":"moderate"},
    "Tomato___Target_Spot": {"symptoms":["brown target-like concentric ring spots","yellowing surrounding lesions"],"severity":"moderate"},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"symptoms":["upward curling of leaves","yellowing leaf margins","stunted plant"],"severity":"severe"},
    "Tomato___Tomato_mosaic_virus": {"symptoms":["mosaic pattern light and dark green","distorted leaves","reduced fruit"],"severity":"severe"},
    "Tomato___healthy": {"symptoms":[],"severity":"none"},
}


# ─── CROP/DISEASE EXTRACTORS ─────────────────────────────────────────
def extract_crop(class_name: str) -> str:
    parts = class_name.split("___")
    raw = parts[0].replace("_", " ").strip()
    if "(" in raw:
        raw = raw.split("(")[0].strip()
    return raw

def extract_disease(class_name: str) -> str:
    parts = class_name.split("___")
    if len(parts) < 2:
        return "Unknown"
    return parts[1].replace("_", " ").strip()


# ─── SAFETY LAYER (NEW) ──────────────────────────────────────────────
def safety_check_treatment(treatment: Dict, region: Optional[str] = None) -> Dict:
    """
    Blocks treatment output if critical safety fields are missing.
    Rule: if missing dosage OR source → BLOCK.
    Rule: if region mismatch → FILTER with warning.
    """
    status = treatment.get("status", "")
    if status in ("no_treatment_found", "healthy_no_treatment"):
        return {"safe": True, "reason": status, "blocked_fields": []}

    blocked = []

    # Dosage check — every chemical entry must contain "@" or "per" (dosage indicator)
    for chem in treatment.get("chemical", []):
        if chem.strip().upper().startswith("NOTE:"):
            continue  # informational notes are not chemical treatments
        if "@" not in chem and " per " not in chem.lower():
            blocked.append(f"missing_dosage_in: {chem[:50]}")

    if not treatment.get("source"):
        blocked.append("missing_source_citation")

    if not treatment.get("safety_notes"):
        blocked.append("missing_safety_notes")

    if blocked:
        return {
            "safe": False,
            "reason": "Treatment record incomplete — output blocked for farmer safety",
            "blocked_fields": blocked,
            "action": "consult_local_agricultural_extension_officer",
        }

    # Region filter
    if region and treatment.get("region"):
        approved = treatment["region"].lower()
        if region.lower() not in approved and "global" not in approved and "worldwide" not in approved:
            return {
                "safe": False,
                "reason": f"Treatment not approved for region '{region}'. Approved: {treatment['region']}",
                "blocked_fields": ["region_mismatch"],
                "filtered_treatment": True,
                "action": "consult_local_extension_officer_for_region_specific_advice",
            }

    return {"safe": True, "reason": "all_safety_checks_passed", "blocked_fields": []}


# ─── MODEL REGISTRY ──────────────────────────────────────────────────
class ModelRegistry:
    def __init__(self):
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"
        self._validator     = None
        self._classifier    = None
        self._classes       = None
        self._validator_ok  = False
        self._classifier_ok = False

    def load_all(self) -> Dict[str, bool]:
        return {"validator": self._load_validator(), "classifier": self._load_classifier()}

    def _load_validator(self) -> bool:
        if not Path(VALIDATOR_PATH).exists():
            print(f"[INFERENCE] Validator not found — train first")
            return False
        try:
            import timm
            model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=0)
            in_f  = model.num_features
            model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f,128), nn.Hardswish(), nn.Dropout(0.15), nn.Linear(128,2))
            ckpt  = torch.load(VALIDATOR_PATH, map_location=self.device)
            model.load_state_dict(ckpt["model_state"])
            model.to(self.device).eval()
            self._validator    = model
            self._validator_ok = True
            print("[INFERENCE] Validator loaded")
            return True
        except Exception as e:
            print(f"[INFERENCE] Validator load failed: {e}")
            return False

    def _load_classifier(self) -> bool:
        if not Path(CLASSIFIER_PATH).exists():
            print("[INFERENCE] Classifier not found — train first")
            return False
        try:
            import timm
            with open(CLASSES_PATH) as f:
                cdata = json.load(f)
            self._classes = cdata["classes"]
            n = len(self._classes)
            model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
            in_f  = model.num_features
            model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f,512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.15), nn.Linear(512,n))
            ckpt  = torch.load(CLASSIFIER_PATH, map_location=self.device)
            model.load_state_dict(ckpt["model_state"])
            model.to(self.device).eval()
            self._classifier    = model
            self._classifier_ok = True
            print(f"[INFERENCE] Classifier loaded: {n} classes")
            return True
        except Exception as e:
            print(f"[INFERENCE] Classifier load failed: {e}")
            return False

    @property
    def validator_ready(self):
        return self._validator_ok
    @property
    def classifier_ready(self):
        return self._classifier_ok


# ─── INFERENCE ENGINE (UPGRADED) ─────────────────────────────────────
class InferenceEngine:

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.device   = registry.device

    @torch.no_grad()
    def validate_image(self, image_bytes: bytes) -> Dict:
        """Unchanged from v1 — blur + dimension + ML plant check."""
        t0 = time.time()
        blur = compute_blur_score(image_bytes)
        if blur < BLUR_THRESHOLD:
            return {"valid": False, "rejection_reason": f"Image too blurry (score:{blur:.1f} threshold:{BLUR_THRESHOLD})", "rejection_type": "blurry", "blur_score": blur, "provenance": {"method":"Laplacian_variance","value":blur,"threshold":BLUR_THRESHOLD}}

        try:
            img = Image.open(io.BytesIO(image_bytes))
            w,h = img.size
            if w < 50 or h < 50:
                return {"valid": False, "rejection_reason": f"Too small ({w}x{h})", "rejection_type": "too_small"}
        except Exception:
            return {"valid": False, "rejection_reason": "Cannot parse image", "rejection_type": "parse_error"}

        plant_prob = None
        vsrc = "heuristic_fallback"
        if self.registry.validator_ready:
            t = preprocess_image(image_bytes)
            if t is not None:
                probs = F.softmax(self.registry._validator(t.to(self.device)), dim=1)
                plant_prob = float(probs[0,1].cpu())
                vsrc = "MobileNetV3-Small"
                if plant_prob < PLANT_THRESHOLD:
                    return {"valid": False, "rejection_reason": f"No plant detected (prob:{plant_prob:.3f} threshold:{PLANT_THRESHOLD})", "rejection_type": "not_plant", "plant_probability": plant_prob, "model": "MobileNetV3-Small", "provenance": {"source":vsrc,"value":plant_prob,"threshold":PLANT_THRESHOLD}}
        else:
            arr = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.float32)
            plant_prob = float(min(arr[:,:,1].mean()/(arr.sum(axis=2).mean()+1e-6)*2.5, 1.0))
            vsrc = "green_channel_heuristic_FALLBACK"

        return {"valid": True, "blur_score": blur, "plant_probability": plant_prob, "validation_source": vsrc, "latency_ms": int((time.time()-t0)*1000)}

    @torch.no_grad()
    def classify_multi_disease(self, image_bytes: bytes) -> Dict:
        """
        UPGRADED: Multi-label disease detection with top-K output.

        Output format:
          {
            "predictions": [
              {"disease": "...", "confidence": 0.87, "type": "primary|secondary"},
              ...
            ]
          }

        Confidence source: EfficientNet-B0 softmax logits.
        All values are real model outputs — never invented.
        """
        t0 = time.time()

        if not self.registry.classifier_ready:
            return {"status":"model_unavailable","action":"python training/train_plant_classifier.py","fallback":"plant_id_api"}

        t = preprocess_image(image_bytes)
        if t is None:
            return {"status":"error","message":"Cannot preprocess image"}

        logits = self.registry._classifier(t.to(self.device))
        probs  = F.softmax(logits, dim=1)[0]

        k = min(TOP_K_PREDICTIONS, len(self.registry._classes))
        top_probs, top_idx = torch.topk(probs, k)

        all_preds = []
        for prob, idx in zip(top_probs.cpu().tolist(), top_idx.cpu().tolist()):
            cn = self.registry._classes[idx]
            all_preds.append({"class_name":cn, "crop":extract_crop(cn), "disease":extract_disease(cn), "confidence":round(float(prob),4), "is_healthy":"healthy" in cn.lower()})

        top1     = all_preds[0]
        top1_prob = top1["confidence"]

        # Confidence gate
        if top1_prob < CONFIDENCE_THRESHOLD:
            return {
                "status":    "uncertain",
                "message":   f"Disease not confidently identified. Confidence: {top1_prob:.1%} (threshold: {CONFIDENCE_THRESHOLD:.0%}). Retake image or consult expert.",
                "action":    "retake_image_or_consult_expert",
                "confidence": top1_prob,
                "threshold":  CONFIDENCE_THRESHOLD,
                "top5":       all_preds,
                "confidence_source": "EfficientNet-B0 softmax logits",
            }

        # Unknown disease fallback
        if top1["class_name"] not in SYMPTOM_MAP:
            return {
                "status":  "unknown_disease",
                "message": f"Disease '{top1['disease']}' not in knowledge base. Consult agricultural expert.",
                "action":  "consult_agricultural_expert",
                "confidence": top1_prob,
            }

        is_healthy = top1["is_healthy"]

        # Build multi-disease predictions list
        diseases: List[Dict] = []

        # Primary
        diseases.append({
            "disease":    top1["disease"],
            "confidence": top1_prob,
            "confidence_pct": round(top1_prob*100, 1),
            "type":       "primary",
            "class_name": top1["class_name"],
            "confidence_source": "EfficientNet-B0 softmax logits (PlantVillage 38 classes)",
        })

        # Secondary: same crop, different disease, above threshold
        for pred in all_preds[1:]:
            if (pred["confidence"] >= SECONDARY_THRESHOLD and
                pred["crop"].lower() == top1["crop"].lower() and
                pred["disease"].lower() != top1["disease"].lower() and
                not pred["is_healthy"]):
                diseases.append({
                    "disease":    pred["disease"],
                    "confidence": pred["confidence"],
                    "confidence_pct": round(pred["confidence"]*100, 1),
                    "type":       "secondary",
                    "class_name": pred["class_name"],
                    "confidence_source": "EfficientNet-B0 softmax logits (PlantVillage 38 classes)",
                })

        symp = SYMPTOM_MAP.get(top1["class_name"], {"symptoms":[],"severity":"unknown"})

        return {
            "status":     "success",
            "crop":       top1["crop"],
            "crop_source": "parsed from EfficientNet-B0 PlantVillage class name",
            "is_healthy": is_healthy,
            "predictions": diseases,           # ← structured multi-disease output
            "top5_raw":   all_preds,
            "symptoms":   symp.get("symptoms",[]),
            "severity":   symp.get("severity","none") if not is_healthy else "none",
            "model":      "EfficientNet-B0",
            "dataset":    "PlantVillage",
            "latency_ms": int((time.time()-t0)*1000),
            "provenance": {
                "model":           "EfficientNet-B0 fine-tuned PlantVillage",
                "confidence_method": "softmax over 38 logits",
                "primary_confidence": top1_prob,
                "secondary_threshold": SECONDARY_THRESHOLD,
                "num_diseases_found": len(diseases),
            },
        }

    def get_multi_disease_treatment(self, predictions: List[Dict], crop: str, region: Optional[str] = None) -> Dict:
        """
        NEW: Separate treatment retrieval for each detected disease.
        Never merges treatments. Safety-checks each one individually.
        """
        from database.treatment_db import treatment_db
        results = []
        for pred in predictions:
            disease   = pred["disease"]
            treatment = treatment_db.lookup(crop, disease)
            safety    = safety_check_treatment(treatment, region)
            results.append({
                "disease":          disease,
                "disease_type":     pred.get("type","primary"),
                "disease_confidence": pred.get("confidence"),
                "treatment":        treatment if safety["safe"] else None,
                "treatment_blocked": not safety["safe"],
                "safety_check":     safety,
                "source":           treatment.get("source","unknown"),
            })
        return {
            "multi_disease_treatments": results,
            "treatment_count":          len(results),
            "all_safe":                 all(r["safety_check"]["safe"] for r in results),
            "warning":                  "Treat each disease separately. Do not mix chemicals without expert guidance.",
        }


# ─── PLANT.ID FALLBACK (unchanged) ───────────────────────────────────
async def plant_id_fallback(image_bytes: bytes, api_key: str) -> Dict:
    import aiohttp, asyncio
    b64      = base64.b64encode(image_bytes).decode()
    details  = "common_names,description,taxonomy,edible_parts,watering"
    url_post = f"https://plant.id/api/v3/identification?details={details}&language=en&async=true"
    headers  = {"Api-Key": api_key, "Content-Type": "application/json"}
    payload  = {"images": [b64], "similar_images": True, "health": "all"}

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        async with session.post(url_post, headers=headers, json=payload) as resp:
            if resp.status not in (200,201):
                return {"status":"api_error","http_status":resp.status}
            init = await resp.json()
        token = init.get("access_token")
        if not token:
            return {"status":"no_token"}
        poll = f"https://plant.id/api/v3/identification/{token}?details={details}&language=en"
        result = None
        for _ in range(20):
            await asyncio.sleep(2.5)
            async with session.get(poll, headers=headers) as pr:
                result = await pr.json()
                if result.get("status") == "COMPLETED":
                    break

    if not result or result.get("status") != "COMPLETED":
        return {"status":"timeout"}

    plant       = (result.get("result",{}).get("classification",{}).get("suggestions",[{}]) or [{}])[0]
    disease_sug = result.get("result",{}).get("disease",{}).get("suggestions",[])
    predictions = []
    for i, dis in enumerate(disease_sug[:3]):
        p = dis.get("probability",0.0)
        if p >= SECONDARY_THRESHOLD:
            predictions.append({"disease":dis.get("name","Unknown"),"confidence":round(p,4),"confidence_pct":round(p*100,1),"type":"primary" if i==0 else "secondary","confidence_source":"Plant.ID API v3"})

    return {
        "status":    "success",
        "source":    "plant_id_api_v3_FALLBACK",
        "crop":      (plant.get("details",{}).get("common_names") or ["Unknown"])[0],
        "predictions": predictions,
        "is_healthy": len(predictions)==0,
        "provenance": {"source":"Plant.ID API v3","note":"external fallback — local model preferred"},
    }


# ─── SINGLETONS ──────────────────────────────────────────────────────
model_registry   = ModelRegistry()
inference_engine = InferenceEngine(model_registry)
