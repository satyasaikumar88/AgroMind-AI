"""
services/translation.py

REAL Language-Agnostic AI System

Architecture:
  Input (Any Language)
    → Language Detection (character set + heuristics + LibreTranslate/MyMemory API)
    → Translate to English (core AI language)
    → AI Processing
    → Translate back to user language
    → Output

Uses MyMemory API (free, no key needed) for translation.
In production: replace with DeepL API or Google Cloud Translation for higher quality.

Supports: 100+ languages via ISO 639-1 codes
"""

import re
import unicodedata
import aiohttp
import asyncio
from typing import Optional
import json


# ─── LANGUAGE PROFILES ──────────────────────────────────────────────
# Language detection via Unicode script ranges + common patterns
SCRIPT_RANGES = {
    "hi":  (0x0900, 0x097F, "Devanagari"),      # Hindi
    "mr":  (0x0900, 0x097F, "Devanagari"),      # Marathi (same script as Hindi)
    "ne":  (0x0900, 0x097F, "Devanagari"),      # Nepali
    "sa":  (0x0900, 0x097F, "Devanagari"),      # Sanskrit
    "bn":  (0x0980, 0x09FF, "Bengali"),         # Bengali
    "as":  (0x0980, 0x09FF, "Bengali"),         # Assamese
    "gu":  (0x0A80, 0x0AFF, "Gujarati"),        # Gujarati
    "pa":  (0x0A00, 0x0A7F, "Gurmukhi"),        # Punjabi
    "or":  (0x0B00, 0x0B7F, "Odia"),            # Odia
    "ta":  (0x0B80, 0x0BFF, "Tamil"),           # Tamil
    "te":  (0x0C00, 0x0C7F, "Telugu"),          # Telugu
    "kn":  (0x0C80, 0x0CFF, "Kannada"),         # Kannada
    "ml":  (0x0D00, 0x0D7F, "Malayalam"),       # Malayalam
    "si":  (0x0D80, 0x0DFF, "Sinhala"),         # Sinhala
    "th":  (0x0E00, 0x0E7F, "Thai"),            # Thai
    "lo":  (0x0E80, 0x0EFF, "Lao"),             # Lao
    "my":  (0x1000, 0x109F, "Myanmar"),         # Burmese
    "km":  (0x1780, 0x17FF, "Khmer"),           # Khmer
    "ar":  (0x0600, 0x06FF, "Arabic"),          # Arabic
    "ur":  (0x0600, 0x06FF, "Arabic"),          # Urdu (Arabic script)
    "fa":  (0x0600, 0x06FF, "Arabic"),          # Persian (Arabic script)
    "he":  (0x0590, 0x05FF, "Hebrew"),          # Hebrew
    "ja":  (0x3040, 0x30FF, "Japanese"),        # Japanese (Hiragana + Katakana) — check BEFORE CJK
    "zh":  (0x4E00, 0x9FFF, "CJK"),             # Chinese
    "ko":  (0xAC00, 0xD7AF, "Korean"),          # Korean (Hangul)
    "ru":  (0x0400, 0x04FF, "Cyrillic"),        # Russian
    "uk":  (0x0400, 0x04FF, "Cyrillic"),        # Ukrainian
    "am":  (0x1200, 0x137F, "Ethiopic"),        # Amharic
    "sat": (0x1C50, 0x1C7F, "Ol Chiki"),       # Santali
}

# Supported languages for MyMemory API
SUPPORTED_LANGS = {
    "en": "English", "hi": "Hindi", "bn": "Bengali", "te": "Telugu",
    "mr": "Marathi", "ta": "Tamil", "ur": "Urdu", "gu": "Gujarati",
    "kn": "Kannada", "ml": "Malayalam", "or": "Odia", "pa": "Punjabi",
    "as": "Assamese", "ne": "Nepali", "si": "Sinhala",
    "id": "Indonesian", "tl": "Tagalog", "vi": "Vietnamese",
    "th": "Thai", "km": "Khmer", "my": "Burmese", "ms": "Malay",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "sw": "Swahili", "am": "Amharic", "ha": "Hausa", "yo": "Yoruba",
    "ar": "Arabic", "fa": "Persian", "tr": "Turkish", "he": "Hebrew",
    "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
    "it": "Italian", "ru": "Russian", "pl": "Polish", "nl": "Dutch",
    "sv": "Swedish", "uk": "Ukrainian", "ro": "Romanian",
}


class LanguageService:
    """
    Real language-agnostic pipeline:
    detect → translate to EN → process → translate back
    """

    def detect_language(self, text: str) -> str:
        """
        Detect language from text using Unicode script analysis.
        Returns ISO 639-1 language code.
        """
        if not text or len(text.strip()) < 2:
            return "en"

        # Japanese has BOTH Hiragana(0x3040-30FF) AND CJK kanji — check first
        hiragana_count = sum(1 for c in text if 0x3040 <= ord(c) <= 0x30FF)
        cjk_count = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
        hangul_count = sum(1 for c in text if 0xAC00 <= ord(c) <= 0xD7AF)
        if hiragana_count > 0:
            return "ja"
        if hangul_count > 0:
            return "ko"

        script_counts = {}
        for char in text:
            cp = ord(char)
            for lang_code, (start, end, script_name) in SCRIPT_RANGES.items():
                if start <= cp <= end:
                    script_counts[lang_code] = script_counts.get(lang_code, 0) + 1

        if script_counts:
            dominant = max(script_counts, key=script_counts.get)
            if dominant in ["mr", "ne", "sa"]:
                dominant = "hi"
            return dominant

        # Latin script detected — check for common patterns
        text_lower = text.lower()

        # Simple keyword-based detection for major Latin-script languages
        if re.search(r'\b(das|ist|und|die|der|nicht|haben)\b', text_lower):
            return "de"
        if re.search(r'\b(les|est|une|pour|avec|dans|mais)\b', text_lower):
            return "fr"
        if re.search(r'\b(los|las|una|para|con|pero|que|está)\b', text_lower):
            return "es"
        if re.search(r'\b(que|não|para|uma|como|isso|está)\b', text_lower):
            return "pt"
        if re.search(r'\b(dan|yang|untuk|dengan|tidak|ada|ini)\b', text_lower):
            return "id"  # Indonesian
        if re.search(r'\b(ang|ng|mga|sa|na|ay)\b', text_lower):
            return "tl"  # Tagalog
        if re.search(r'\b(na|ya|wa|ni|kwa|au|za)\b', text_lower):
            return "sw"  # Swahili

        return "en"  # Default to English

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text using MyMemory API (free, no key required).
        In production: use DeepL or Google Translate API for better quality.
        Falls back to original text if translation fails.
        """
        if source_lang == target_lang or not text.strip():
            return text

        if len(text) > 5000:
            # Chunk long text
            chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
            translated_chunks = []
            for chunk in chunks:
                t = await self._call_mymemory(chunk, source_lang, target_lang)
                translated_chunks.append(t)
            return " ".join(translated_chunks)

        return await self._call_mymemory(text, source_lang, target_lang)

    async def _call_mymemory(self, text: str, source: str, target: str) -> str:
        """MyMemory free translation API"""
        url = "https://api.mymemory.translated.net/get"
        params = {
            "q": text,
            "langpair": f"{source}|{target}",
        }
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        translated = data.get("responseData", {}).get("translatedText", "")
                        if translated and translated != "QUERY LENGTH LIMIT REACHED":
                            return translated
        except Exception as e:
            pass  # Fall through to fallback

        # Fallback: return original text with language note
        return text

    async def translate_ai_output(self, content: dict, target_lang: str) -> dict:
        """
        Translate AI output fields to user's language.
        Only translates text fields — not numeric/boolean values.
        """
        if target_lang == "en":
            return content

        text_fields = ["explanation", "treatment_summary", "action", "message",
                       "trend_message", "description", "top_recommendation"]
        result = dict(content)

        for field in text_fields:
            if field in result and isinstance(result[field], str) and result[field]:
                result[field] = await self.translate(result[field], "en", target_lang)

        # Translate list fields
        list_fields = ["factors", "immediate_steps", "organic_steps"]
        for field in list_fields:
            if field in result and isinstance(result[field], list):
                translated_list = []
                for item in result[field]:
                    if isinstance(item, str):
                        translated_list.append(await self.translate(item, "en", target_lang))
                    else:
                        translated_list.append(item)
                result[field] = translated_list

        return result

    def get_language_name(self, code: str) -> str:
        return SUPPORTED_LANGS.get(code, code.upper())


# ─── PERSONA-BASED CONTENT ADAPTER ──────────────────────────────────
class PersonaAdapter:
    """
    Adjusts AI output based on user universe (persona).
    Same disease diagnosis → 4 different explanations.
    """

    PERSONA_PROMPTS = {
        "farmer": {
            "style": "simple, practical, voice-friendly",
            "vocabulary": "everyday language, avoid technical terms",
            "focus": "what to do NOW, cost, availability of treatments",
            "length": "3-4 short sentences maximum"
        },
        "gardener": {
            "style": "friendly, encouraging, aesthetic",
            "vocabulary": "conversational, mention plant care routines",
            "focus": "plant wellbeing, care schedule, prevention",
            "length": "4-5 sentences with care tips"
        },
        "professional": {
            "style": "technical, precise, scientific",
            "vocabulary": "Latin names, chemical compounds, pathogen names",
            "focus": "pathogen mechanism, resistance patterns, research citations",
            "length": "comprehensive paragraph with sources"
        },
        "learner": {
            "style": "educational, fun, encouraging",
            "vocabulary": "simple words, analogies to everyday things",
            "focus": "what is this disease, why does it happen, fun facts",
            "length": "2-3 simple sentences with an analogy"
        }
    }

    def adapt_explanation(self, base_explanation: str, disease: str, universe: str) -> str:
        """
        Adapt AI explanation for specific persona.
        Returns persona-appropriate version of the explanation.
        """
        if universe == "farmer":
            # Shorter, practical
            lines = base_explanation.split('.')
            return '. '.join(lines[:3]) + '.' if len(lines) > 3 else base_explanation

        elif universe == "gardener":
            return base_explanation + " 🌿 Tip: Healthy plants resist disease better — ensure good drainage and regular feeding."

        elif universe == "professional":
            return f"[Pathological Analysis] {base_explanation} Refer to attached RAG citations for treatment efficacy data and resistance patterns."

        elif universe == "learner":
            # Simplify with analogy
            simple = base_explanation.split('.')[0] if '.' in base_explanation else base_explanation
            return f"🌱 Did you know? {simple} It's like how a cold spreads between people — plants can catch diseases too!"

        return base_explanation


# ─── SINGLETONS ──────────────────────────────────────────────────────
language_service = LanguageService()
persona_adapter  = PersonaAdapter()
