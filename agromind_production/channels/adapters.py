"""
channels/adapters.py

Multi-Channel Output Formatters

Same backend result → 5 different channel formats:
  App     → full JSON with all fields
  WhatsApp → concise text (max 500 chars)
  IVR     → spoken text, no jargon, short sentences
  SMS     → 160-character limit, critical info only
  Kiosk   → guided structured format with steps

Each adapter is a pure function: Dict → Dict
"""

from typing import Dict, Optional
from datetime import datetime


# ─── APP ADAPTER (Full) ──────────────────────────────────────────────
def format_app(result: Dict) -> Dict:
    """
    Full structured response for mobile/web app.
    Includes all fields, provenance, timing, sources.
    """
    return {
        "channel":  "app",
        "format":   "full_json",
        "data":     result,
        "metadata": {
            "timestamp":    datetime.utcnow().isoformat(),
            "channel":      "app",
            "content_type": "application/json",
        }
    }


# ─── WHATSAPP ADAPTER ────────────────────────────────────────────────
def format_whatsapp(result: Dict, language: str = "en") -> Dict:
    """
    WhatsApp-optimised format.
    - Maximum 500 characters for text message
    - Bold using *text* (WhatsApp markdown)
    - Use emojis for visual scanning
    - Critical info only: disease, confidence, top treatment step
    """
    diag = result.get("diagnosis", result.get("result", {}))
    tx   = result.get("treatment", {})

    # Extract key fields
    plant      = diag.get("plant_name", diag.get("species", "Unknown plant"))
    disease    = diag.get("disease_name", diag.get("disease", {}).get("name", "Unknown"))
    confidence = diag.get("confidence", diag.get("confidence_pct", 0))
    if isinstance(confidence, float) and confidence <= 1.0:
        confidence = round(confidence * 100)
    is_healthy = diag.get("is_healthy", True)
    top_step   = (tx.get("immediate") or ["Consult local extension officer"])[0]

    if is_healthy:
        text = (
            f"🌿 *AgroMind Diagnosis*\n"
            f"Plant: {plant}\n"
            f"Status: ✅ Healthy ({confidence}% confidence)\n"
            f"Action: Continue regular care. Monitor weekly."
        )
    else:
        text = (
            f"🌿 *AgroMind Diagnosis*\n"
            f"Plant: {plant}\n"
            f"Disease: ⚠️ {disease} ({confidence}% confidence)\n"
            f"Action: {top_step[:150]}\n"
            f"Powered by AgroMind Universal"
        )

    # Truncate to 500 chars
    if len(text) > 500:
        text = text[:497] + "..."

    return {
        "channel":    "whatsapp",
        "format":     "text",
        "text":       text,
        "char_count": len(text),
        "max_chars":  500,
        "metadata": {
            "timestamp": datetime.utcnow().isoformat(),
            "language":  language,
        }
    }


# ─── IVR ADAPTER (Voice) ─────────────────────────────────────────────
def format_ivr(result: Dict, language: str = "en") -> Dict:
    """
    IVR (Interactive Voice Response) format.
    - Short sentences (max 15 words each)
    - No jargon, no chemical names (hard to understand over phone)
    - Designed for text-to-speech synthesis
    - Includes DTMF prompt for next action
    """
    diag = result.get("diagnosis", result.get("result", {}))
    tx   = result.get("treatment", {})

    plant    = diag.get("plant_name", diag.get("species", "your plant"))
    disease  = diag.get("disease_name", diag.get("disease", {}).get("name", "a condition"))
    conf     = diag.get("confidence", 0)
    if isinstance(conf, float) and conf <= 1.0:
        conf = round(conf * 100)
    healthy  = diag.get("is_healthy", True)
    step1    = (tx.get("immediate") or ["consult your local agricultural officer"])[0]
    # Simplify step1 for voice
    step1_voice = step1[:100].split('.')[0] if step1 else "consult your local agricultural officer"

    if healthy:
        speech_text = (
            f"Your {plant} is healthy. "
            f"Confidence level is {conf} percent. "
            f"Continue regular watering and care. "
            f"Monitor your plant every week. "
            f"Press 1 to hear this again. "
            f"Press 2 to scan another plant. "
            f"Press 3 to speak with an expert."
        )
    else:
        severity = diag.get("disease", {}).get("severity", "moderate") if isinstance(diag.get("disease"), dict) else "moderate"
        speech_text = (
            f"Warning. Your {plant} has a disease. "
            f"The disease is {disease}. "
            f"Confidence level is {conf} percent. "
            f"Severity is {severity}. "
            f"First action: {step1_voice}. "
            f"Press 1 to hear again. "
            f"Press 2 to hear full treatment. "
            f"Press 3 to speak with an expert. "
            f"Press 4 to send this to WhatsApp."
        )

    return {
        "channel":    "ivr",
        "format":     "speech",
        "speech_text": speech_text,
        "word_count":  len(speech_text.split()),
        "dtmf_prompts": {
            "1": "repeat",
            "2": "full_treatment",
            "3": "connect_expert",
            "4": "send_whatsapp",
        },
        "metadata": {
            "timestamp": datetime.utcnow().isoformat(),
            "language":  language,
            "tts_note":  "Apply regional language TTS model before output",
        }
    }


# ─── SMS ADAPTER ─────────────────────────────────────────────────────
def format_sms(result: Dict) -> Dict:
    """
    SMS format: 160 GSM characters maximum.
    Includes only: disease name, confidence, single most critical action.
    Uses abbreviations for space saving.
    """
    diag     = result.get("diagnosis", result.get("result", {}))
    tx       = result.get("treatment", {})
    plant    = (diag.get("plant_name", diag.get("species", "Plant")) or "Plant")[:15]
    disease  = (diag.get("disease_name") or (diag.get("disease", {}) or {}).get("name") or "Unknown")[:25]
    conf     = diag.get("confidence", 0)
    if isinstance(conf, float) and conf <= 1.0:
        conf = round(conf * 100)
    healthy  = diag.get("is_healthy", True)
    step     = (tx.get("immediate") or ["See AgroMind app"])[0][:60]

    if healthy:
        msg = f"AgroMind:{plant} HEALTHY({conf}%). Keep care routine. -agromind.ai"
    else:
        msg = f"AgroMind:{plant} {disease}({conf}%). {step} -agromind.ai"

    # Truncate to 160 chars
    if len(msg) > 160:
        msg = msg[:157] + "..."

    return {
        "channel":    "sms",
        "format":     "text",
        "text":       msg,
        "char_count": len(msg),
        "max_chars":  160,
        "gsm_compliant": all(ord(c) < 128 for c in msg),
        "metadata": {"timestamp": datetime.utcnow().isoformat()}
    }


# ─── KIOSK ADAPTER ──────────────────────────────────────────────────
def format_kiosk(result: Dict, language: str = "en") -> Dict:
    """
    Village kiosk format.
    Designed for: thermal printer output, guided step-by-step display.
    Large text, numbered steps, no abbreviations.
    """
    diag     = result.get("diagnosis", result.get("result", {}))
    tx       = result.get("treatment", {})
    rag      = result.get("knowledge_sources", [])

    plant    = diag.get("plant_name", diag.get("species", "Unknown"))
    disease  = diag.get("disease_name") or (diag.get("disease", {}) or {}).get("name") or "None detected"
    conf     = diag.get("confidence", 0)
    if isinstance(conf, float) and conf <= 1.0:
        conf = round(conf * 100)
    healthy  = diag.get("is_healthy", True)
    symptoms = diag.get("symptoms", [])
    source   = rag[0].get("source", "AgroMind") if rag else "AgroMind Universal"

    immediate = tx.get("immediate", ["Consult local extension officer"])
    schedule  = tx.get("schedule", [])

    kiosk_output = {
        "channel":    "kiosk",
        "format":     "printable_structured",
        "sections": {
            "header": {
                "title":       "AgroMind Universal — Crop Health Report",
                "date":        datetime.utcnow().strftime("%d/%m/%Y %H:%M"),
                "print_large": True,
            },
            "diagnosis": {
                "title":       "DIAGNOSIS RESULT",
                "plant":       plant,
                "status":      "HEALTHY ✓" if healthy else f"DISEASE DETECTED: {disease}",
                "confidence":  f"{conf}%",
                "confidence_note": "Higher % = more certain",
            },
            "symptoms": {
                "title":    "VISIBLE SYMPTOMS",
                "items":    symptoms or ["No visible symptoms" if healthy else "See extension officer for symptom identification"],
            },
            "treatment_steps": {
                "title":  "TREATMENT STEPS (IN ORDER)",
                "steps":  [{"step": i+1, "action": step} for i, step in enumerate(immediate)],
                "source": source,
            },
            "spray_schedule": {
                "title":    "SPRAY SCHEDULE",
                "schedule": schedule or [{"day": "As needed", "action": "Follow extension officer advice"}],
            },
            "emergency": {
                "title":   "NEED HELP?",
                "contacts": [
                    "Kisan Call Centre: 1800-180-1551 (FREE, 24/7)",
                    "Local KVK: Visit your nearest Krishi Vigyan Kendra",
                    "AgroMind App: agromind.ai",
                ]
            }
        },
        "print_settings": {
            "paper":        "80mm thermal",
            "font_size":    "large",
            "language":     language,
            "copies":       1,
        },
        "metadata": {"timestamp": datetime.utcnow().isoformat()},
    }

    return kiosk_output


# ─── CHANNEL ROUTER ──────────────────────────────────────────────────
def format_for_channel(result: Dict, channel: str, language: str = "en") -> Dict:
    """
    Route to appropriate formatter based on channel.
    Valid channels: app, whatsapp, ivr, sms, kiosk
    """
    formatters = {
        "app":       lambda r: format_app(r),
        "whatsapp":  lambda r: format_whatsapp(r, language),
        "ivr":       lambda r: format_ivr(r, language),
        "sms":       lambda r: format_sms(r),
        "kiosk":     lambda r: format_kiosk(r, language),
    }

    formatter = formatters.get(channel.lower())
    if not formatter:
        return {
            "error":    f"Unknown channel: {channel}",
            "valid_channels": list(formatters.keys()),
        }

    formatted = formatter(result)
    formatted["source_result_id"] = result.get("scan_id", result.get("id", "unknown"))
    return formatted
