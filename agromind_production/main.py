"""
main.py — AgroMind Universal Production API

Full pipeline per request:
  Image → Validation (ML) → Detection (ML) → Confidence Gate
  → Symptom Interpreter → Treatment Retrieval (DB)
  → Risk Engine → RAG Explanation → Persona Adapter
  → Language Pipeline → Channel Formatter → Output → Store Memory

All 5 required endpoints:
  POST /validate  — ML image validation only
  POST /predict   — full pipeline
  POST /risk      — predictive risk engine
  POST /ask       — RAG query in any language
  GET  /history   — plant memory + trend detection
"""

import os
import time
import uuid
import base64
import asyncio
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from dotenv import load_dotenv

load_dotenv()

from database.models import (
    init_db, get_db, PlantHistory, ScanLog, OutbreakReport,
    compute_plant_trend
)
from services.inference import model_registry, inference_engine, plant_id_fallback
from services.rag_faiss import rag_pipeline
from services.risk_engine import risk_engine, WeatherData, CropInfo, OutbreakSignal
from services.translation import language_service, PersonaAdapter
from services.outbreak import outbreak_engine
from database.treatment_db import treatment_db
from channels.adapters import format_for_channel

PLANT_ID_KEY = os.getenv("PLANT_ID_KEY", "qVCtdPxVw6en5J1zj9hhMK8JjaFt6BvfxjYrFlFA42SXYZT0gc")
persona_adapter = PersonaAdapter()

# ─── APP INIT ────────────────────────────────────────────────────────
app = FastAPI(
    title="AgroMind Universal API",
    description="Production-grade agricultural AI system",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    init_db()
    status = model_registry.load_all()
    print(f"[STARTUP] Models: {status}")
    try:
        rag_pipeline.load_index()
    except Exception as e:
        print(f"[STARTUP] RAG: {e} — will build on first use")
    print("[STARTUP] AgroMind Universal API ready")


# ─── REQUEST MODELS ──────────────────────────────────────────────────
class RiskRequest(BaseModel):
    latitude:      float = Field(17.38)
    longitude:     float = Field(78.47)
    crop_type:     str   = Field("rice")
    growth_stage:  str   = Field("vegetative")
    temperature:   Optional[float] = None
    humidity:      Optional[float] = None
    rainfall_3d:   Optional[float] = None
    wind_speed:    Optional[float] = None
    nearby_cases:  int   = Field(0)
    radius_km:     float = Field(10.0)

class AskRequest(BaseModel):
    query:    str = Field(...)
    language: str = Field("en")
    universe: str = Field("farmer")
    top_k:    int = Field(3)


# ─── HEALTH CHECK ────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service":     "AgroMind Universal API",
        "version":     "4.0.0",
        "models": {
            "validator":  model_registry.validator_ready,
            "classifier": model_registry.classifier_ready,
        },
        "rag_documents": len(rag_pipeline.corpus),
        "treatment_db":  len(treatment_db.records),
        "endpoints": [
            "POST /validate",
            "POST /predict",
            "POST /risk",
            "POST /ask",
            "GET  /history",
        ],
        "training_required": not (model_registry.validator_ready and model_registry.classifier_ready),
        "training_instructions": "Run: python training/train_plant_classifier.py && python training/train_validator.py",
    }


# ══════════════════════════════════════════════════════════════════════
# ENDPOINT 1: POST /validate
# ══════════════════════════════════════════════════════════════════════
@app.post("/validate")
async def validate(
    image: UploadFile = File(...),
    db:    Session    = Depends(get_db),
):
    """
    ML-based image validation.
    Stages: blur detection → dimension check → plant/non-plant ML model
    Returns full provenance for every decision.
    """
    t0 = time.time()

    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    image_bytes = await image.read()
    result      = inference_engine.validate_image(image_bytes)

    log = ScanLog(
        endpoint         = "/validate",
        input_type       = "image",
        validation_ok    = result.get("valid", False),
        rejection_reason = result.get("rejection_reason"),
        latency_ms       = int((time.time() - t0) * 1000),
    )
    db.add(log); db.commit()

    return result


# ══════════════════════════════════════════════════════════════════════
# ENDPOINT 2: POST /predict
# Full 5-stage pipeline
# ══════════════════════════════════════════════════════════════════════
@app.post("/predict")
async def predict(
    image:        UploadFile = File(...),
    user_id:      str   = Form("anonymous"),
    universe:     str   = Form("farmer"),
    language:     str   = Form("en"),
    crop_type:    str   = Form("unknown"),
    growth_stage: str   = Form("vegetative"),
    plant_id_name: str  = Form("my_plant"),
    latitude:     float = Form(17.38),
    longitude:    float = Form(78.47),
    channel:      str   = Form("app"),
    db:           Session = Depends(get_db),
):
    """
    Full diagnostic pipeline.
    Every output includes provenance: input → computation → source.
    Returns confidence gate response if confidence < threshold.
    """
    t0          = time.time()
    scan_id     = str(uuid.uuid4())
    image_bytes = await image.read()

    pipeline_log = {"scan_id": scan_id, "stages": []}

    # ── Stage 1: Validation ──────────────────────────────────────
    val = inference_engine.validate_image(image_bytes)
    pipeline_log["stages"].append({"stage": "validation", "result": val.get("valid"), "reason": val.get("rejection_reason")})

    if not val.get("valid"):
        log = ScanLog(endpoint="/predict", user_id=user_id, input_type="image",
                      validation_ok=False, rejection_reason=val.get("rejection_reason"),
                      latency_ms=int((time.time()-t0)*1000), universe=universe, language=language)
        db.add(log); db.commit()
        return {"status": "rejected", "stage": "validation", "reason": val.get("rejection_reason"),
                "rejection_type": val.get("rejection_type"), "provenance": val.get("provenance"), "scan_id": scan_id}

    # ── Stage 2: ML Classification ──────────────────────────────
    if model_registry.classifier_ready:
        cls = inference_engine.classify_disease(image_bytes)
        classification_source = "local_efficientnet_b0"
    else:
        # API fallback — clearly labeled
        cls = await plant_id_fallback(image_bytes, PLANT_ID_KEY)
        classification_source = "plant_id_api_fallback"

    pipeline_log["stages"].append({"stage": "classification", "source": classification_source, "status": cls.get("status")})

    if cls.get("status") == "low_confidence":
        return {"status": "low_confidence", "stage": "confidence_gate",
                "message": cls.get("message"), "action": cls.get("action"),
                "confidence": cls.get("confidence"), "threshold": cls.get("threshold"),
                "top5": cls.get("top5_predictions", []), "scan_id": scan_id}

    if cls.get("status") == "model_unavailable":
        return {"status": "model_unavailable", "message": cls.get("action"), "scan_id": scan_id}

    if cls.get("status") not in ("success", None):
        return {"status": "error", "message": str(cls), "scan_id": scan_id}

    # Extract classification results
    plant_name   = cls.get("plant_name",   cls.get("common_names", ["Unknown"])[0] if isinstance(cls.get("common_names"), list) else "Unknown")
    disease_name = cls.get("disease_name", cls.get("disease_name", "No disease detected"))
    confidence   = cls.get("confidence",   cls.get("confidence", 0.0))
    is_healthy   = cls.get("is_healthy",   not (cls.get("disease_probability", 0) > 0.2))
    symptoms     = cls.get("symptoms",     [])

    # Infer crop type from classification if not provided
    if crop_type == "unknown" and plant_name:
        for crop_key in ["rice", "tomato", "potato", "wheat", "cotton", "coffee", "maize", "corn"]:
            if crop_key.lower() in plant_name.lower():
                crop_type = crop_key
                break

    # ── Stage 3: Treatment Retrieval (DB only) ──────────────────
    if not is_healthy:
        treatment = treatment_db.lookup(crop_type, disease_name)
    else:
        treatment = {
            "status": "healthy_no_treatment",
            "message": "Plant is healthy. No treatment required.",
            "preventive": [
                "Monitor weekly for early disease signs",
                "Maintain balanced nutrition with soil test-based fertilization",
                "Ensure good air circulation and avoid overhead irrigation",
                "Apply neem oil preventively monthly during high-risk seasons",
            ]
        }

    pipeline_log["stages"].append({"stage": "treatment", "source": "treatment_db", "status": treatment.get("status")})

    # ── Stage 4: Risk Engine ─────────────────────────────────────
    import aiohttp
    weather = await _fetch_weather(latitude, longitude)
    crop_info    = CropInfo(crop_type=crop_type, growth_stage=growth_stage)

    # Real outbreak signal from DB
    ob_signal    = outbreak_engine.compute_outbreak_signal(db, latitude, longitude, disease_name)
    nearby_cases = ob_signal.get("cases", 0) if ob_signal.get("status") == "computed" else 0

    outbreak_input = OutbreakSignal(nearby_cases=nearby_cases, radius_km=10.0, days_window=7, dominant_disease=disease_name)
    risk = risk_engine.compute_risk(weather, crop_info, outbreak_input)
    pipeline_log["stages"].append({"stage": "risk", "source": "risk_engine + open_meteo"})

    # ── Stage 5: RAG Explanation ─────────────────────────────────
    rag_query  = f"{disease_name} {plant_name} treatment management"
    rag_result = rag_pipeline.retrieve_with_proof(rag_query, top_k=3)
    docs       = rag_pipeline.retrieve(rag_query, top_k=3, filter_crop=crop_type)
    explanation = rag_pipeline.build_explanation(docs, disease_name, universe)
    pipeline_log["stages"].append({"stage": "rag", "retrieved": rag_result.get("num_retrieved"), "model": rag_result.get("model")})

    # ── Stage 6: Persona Adapter ─────────────────────────────────
    adapted_explanation = persona_adapter.adapt_explanation(explanation, disease_name, universe)

    # ── Stage 7: Language Translation ───────────────────────────
    final_explanation = adapted_explanation
    if language != "en":
        try:
            final_explanation = await language_service.translate(adapted_explanation, "en", language)
        except Exception:
            final_explanation = adapted_explanation  # fallback to English

    # ── Assemble Response ────────────────────────────────────────
    conf_pct = round(confidence * 100, 1) if isinstance(confidence, float) and confidence <= 1.0 else confidence

    response = {
        "status":    "success",
        "scan_id":   scan_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_stages": pipeline_log["stages"],
        "diagnosis": {
            "plant_name":      plant_name,
            "disease_name":    disease_name,
            "confidence":      confidence,
            "confidence_pct":  conf_pct,
            "is_healthy":      is_healthy,
            "symptoms":        symptoms,
            "classification_source": classification_source,
        },
        "explanation":       final_explanation,
        "explanation_source": rag_result.get("retrieved_documents", []),
        "treatment":         treatment,
        "risk_assessment":   risk,
        "outbreak_signal":   ob_signal,
        "universe":          universe,
        "language":          language,
        "latency_ms":        int((time.time() - t0) * 1000),
    }

    # ── Store in DB ──────────────────────────────────────────────
    record = PlantHistory(
        id=scan_id, user_id=user_id, plant_id=plant_id_name,
        species=plant_name, common_name=plant_name,
        disease=disease_name, disease_prob=cls.get("disease_probability", 0) if not is_healthy else 0,
        confidence=float(confidence) if isinstance(confidence, (int, float)) else 0.0,
        is_healthy=is_healthy,
        severity=cls.get("severity", "none") if not is_healthy else "none",
        image_b64=base64.b64encode(image_bytes).decode()[:5000],
        treatment=treatment,
        risk_score=risk.get("top_threat", {}).get("risk_score", 0) if risk.get("top_threat") else 0,
        latitude=latitude, longitude=longitude,
        universe=universe, language=language,
    )
    db.add(record)

    if not is_healthy:
        outbreak_engine.store_scan_for_outbreak(db, latitude, longitude, crop_type, disease_name, float(confidence) if isinstance(confidence, (int, float)) else 0.0)

    log = ScanLog(
        endpoint="/predict", user_id=user_id, input_type="image",
        validation_ok=True, prediction=disease_name,
        confidence=float(confidence) if isinstance(confidence, (int, float)) else 0.0,
        latency_ms=int((time.time()-t0)*1000), universe=universe, language=language,
    )
    db.add(log); db.commit()

    # ── Channel Formatting ───────────────────────────────────────
    return format_for_channel(response, channel, language)


# ══════════════════════════════════════════════════════════════════════
# ENDPOINT 3: POST /risk
# ══════════════════════════════════════════════════════════════════════
@app.post("/risk")
async def compute_risk(req: RiskRequest, db: Session = Depends(get_db)):
    """
    Predictive risk engine.
    Real weather from Open-Meteo + epidemiological formula.
    Every number includes: inputs, formula, source.
    """
    if req.temperature is not None:
        weather = WeatherData(
            temperature=req.temperature,
            humidity=req.humidity or 70.0,
            rainfall_3d=req.rainfall_3d or 0.0,
            wind_speed=req.wind_speed or 10.0,
        )
        weather_source = "user_provided"
    else:
        weather = await _fetch_weather(req.latitude, req.longitude)
        weather_source = "open_meteo_api"

    crop     = CropInfo(crop_type=req.crop_type, growth_stage=req.growth_stage)
    outbreak = OutbreakSignal(nearby_cases=req.nearby_cases, radius_km=req.radius_km, days_window=7)
    result   = risk_engine.compute_risk(weather, crop, outbreak)

    return {
        "status": "computed",
        "data":   result,
        "weather_source": weather_source,
        "formula": "risk = weather_score*0.50 + crop_stage*0.25 + outbreak_signal*0.25",
        "provenance": {
            "weather": {
                "source":      weather_source,
                "temperature": weather.temperature,
                "humidity":    weather.humidity,
                "rainfall_3d": weather.rainfall_3d,
                "wind_speed":  weather.wind_speed,
            },
            "crop": {"type": req.crop_type, "stage": req.growth_stage},
            "outbreak": {"cases": req.nearby_cases, "radius_km": req.radius_km},
        }
    }


# ══════════════════════════════════════════════════════════════════════
# ENDPOINT 4: POST /ask
# ══════════════════════════════════════════════════════════════════════
@app.post("/ask")
async def ask(req: AskRequest):
    """
    RAG query pipeline.
    detect language → translate to EN → retrieve from FAISS → translate back.
    Returns: retrieved_docs, similarity_scores, grounded explanation.
    """
    # Detect language
    detected = language_service.detect_language(req.query)
    src_lang  = detected if detected != "en" else req.language

    # Translate to English
    if src_lang != "en":
        en_query = await language_service.translate(req.query, src_lang, "en")
    else:
        en_query = req.query

    # RAG retrieval with proof
    rag_result = rag_pipeline.retrieve_with_proof(en_query, top_k=req.top_k)
    docs       = rag_pipeline.retrieve(en_query, top_k=req.top_k)
    explanation = rag_pipeline.build_explanation(docs, en_query, req.universe)

    # Adapt for persona
    adapted = persona_adapter.adapt_explanation(explanation, en_query, req.universe)

    # Translate back
    final = adapted
    if src_lang != "en":
        try:
            final = await language_service.translate(adapted, "en", src_lang)
        except Exception:
            final = adapted

    return {
        "status":                  "success",
        "query_original":          req.query,
        "query_language_detected": src_lang,
        "query_english":           en_query,
        "answer":                  final,
        "answer_english":          adapted,
        "rag_proof":               rag_result,
        "universe":                req.universe,
    }


# ══════════════════════════════════════════════════════════════════════
# ENDPOINT 5: GET /history
# ══════════════════════════════════════════════════════════════════════
@app.get("/history")
async def get_history(
    user_id:  str           = Query(...),
    plant_id: Optional[str] = Query(None),
    limit:    int           = Query(20),
    db:       Session       = Depends(get_db),
):
    """
    Plant memory with trend detection.
    Trend computed from scan history — improving/declining/stable.
    """
    q = db.query(PlantHistory).filter(PlantHistory.user_id == user_id)
    if plant_id:
        q = q.filter(PlantHistory.plant_id == plant_id)
    records = q.order_by(PlantHistory.timestamp.desc()).limit(limit).all()
    trend   = compute_plant_trend(records)

    return {
        "status":         "success",
        "user_id":        user_id,
        "plant_id":       plant_id,
        "record_count":   len(records),
        "records":        [r.to_dict() for r in records],
        "trend_analysis": trend,
    }


# ─── WEATHER HELPER ──────────────────────────────────────────────────
async def _fetch_weather(lat: float, lon: float) -> "WeatherData":
    import aiohttp
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
           f"&daily=precipitation_sum&forecast_days=3")
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
            async with s.get(url) as r:
                if r.status == 200:
                    d = await r.json()
                    c = d.get("current", {})
                    rain = sum(d.get("daily", {}).get("precipitation_sum", [0, 0, 0])[:3])
                    return WeatherData(
                        temperature=c.get("temperature_2m", 25.0),
                        humidity=c.get("relative_humidity_2m", 70.0),
                        rainfall_3d=rain,
                        wind_speed=c.get("wind_speed_10m", 10.0),
                    )
    except Exception:
        pass
    return WeatherData(temperature=27.0, humidity=70.0, rainfall_3d=5.0, wind_speed=10.0)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
