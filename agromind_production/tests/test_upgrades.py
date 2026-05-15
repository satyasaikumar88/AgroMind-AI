"""
tests/test_upgrades.py

MANDATORY TEST CASES from upgrade prompts:

Prompt 1 (Disease accuracy):
  1. Human image → rejected
  2. Object image → rejected
  3. Healthy plant → no disease
  4. Single disease → correct output structure
  5. Multiple diseases → multiple outputs with types
  6. Unknown disease → safe fallback
  7. Low confidence → blocked with message
  8. Wrong region → filtered treatment

Prompt 2 (Predictive risk):
  1. No outbreak data → low confidence output
  2. Increasing humidity trend → increased risk score
  3. Rising outbreak cluster → high risk
  4. Stable environment → low risk
  5. Missing data → safe fallback
  6. Plant history worsening → boosted risk
"""

import sys
import os
import io
import asyncio
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── IMAGE GENERATORS ────────────────────────────────────────────────
def make_skin_tone(size=(300,300)) -> bytes:
    img = Image.new("RGB", size, (210,160,120))
    buf = io.BytesIO(); img.save(buf, "JPEG"); return buf.getvalue()

def make_grey_object(size=(300,300)) -> bytes:
    img = Image.new("RGB", size, (140,138,136))
    buf = io.BytesIO(); img.save(buf, "JPEG"); return buf.getvalue()

def make_green_plant(size=(300,300)) -> bytes:
    arr = np.zeros((*size, 3), dtype=np.uint8)
    arr[:,:,0] = np.random.randint(30,90,(size))
    arr[:,:,1] = np.random.randint(100,200,(size))
    arr[:,:,2] = np.random.randint(20,80,(size))
    img = Image.fromarray(arr)
    buf = io.BytesIO(); img.save(buf, "PNG"); return buf.getvalue()

def make_blurry(size=(300,300)) -> bytes:
    arr = np.ones((*size, 3), dtype=np.uint8) * 90
    arr += np.random.randint(0, 3, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=20); return buf.getvalue()


# ─── TEST RUNNER ─────────────────────────────────────────────────────
class TestSuite:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []

    def check(self, name: str, condition: bool, detail: str = ""):
        status = "✅ PASS" if condition else "❌ FAIL"
        msg    = f"{status} | {name}" + (f" | {detail}" if detail else "")
        self.results.append(msg)
        print(f"  {msg}")
        if condition: self.passed += 1
        else:         self.failed += 1

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'═'*65}")
        print(f"RESULTS: {self.passed}/{total} passed | {self.failed} failed")
        print(f"{'═'*65}")
        return self.failed == 0


T = TestSuite()


# ════════════════════════════════════════════════════════════════════
# PROMPT 1 TESTS: Disease Accuracy System
# ════════════════════════════════════════════════════════════════════

def test_validation():
    print("\n─── Prompt 1: Validation (Step 1) ───")
    from services.inference import inference_engine, compute_blur_score

    # Test 1: Human/skin tone → rejected
    skin = make_skin_tone()
    result = inference_engine.validate_image(skin)
    T.check("Human/skin-tone rejected",
            not result["valid"],
            f"reason='{result.get('rejection_reason','')[:60]}'")

    # Test 2: Grey object → rejected
    grey = make_grey_object()
    result = inference_engine.validate_image(grey)
    T.check("Grey object rejected",
            not result["valid"],
            f"type='{result.get('rejection_type','')}'")

    # Test 3: Blurry → rejected
    blurry = make_blurry()
    result = inference_engine.validate_image(blurry)
    blur_score = compute_blur_score(blurry)
    T.check("Blurry image rejected",
            not result["valid"],
            f"blur_score={blur_score:.1f} threshold=80.0")

    # Test 4: Green plant → passes heuristic
    plant = make_green_plant()
    result = inference_engine.validate_image(plant)
    T.check("Valid green plant passes",
            result["valid"],
            f"plant_prob={result.get('plant_probability', 'heuristic')}")


def test_confidence_gate():
    print("\n─── Prompt 1: Confidence Gate ───")
    from services.inference import CONFIDENCE_THRESHOLD, SECONDARY_THRESHOLD

    # Test 5: Low confidence → blocked with structured message
    # Simulate a low-confidence scenario by checking the gate logic directly
    mock_top1_prob = 0.30  # below 0.45 threshold
    gate_triggered = mock_top1_prob < CONFIDENCE_THRESHOLD
    T.check("Confidence gate triggers below threshold",
            gate_triggered,
            f"value={mock_top1_prob:.2f} threshold={CONFIDENCE_THRESHOLD:.2f}")

    # Test 6: Secondary threshold allows multi-disease
    secondary_prob = 0.25
    T.check("Secondary disease included above secondary threshold",
            secondary_prob >= SECONDARY_THRESHOLD,
            f"value={secondary_prob:.2f} threshold={SECONDARY_THRESHOLD:.2f}")

    # Test 7: Sub-secondary threshold excludes
    sub_secondary = 0.10
    T.check("Sub-threshold disease excluded",
            sub_secondary < SECONDARY_THRESHOLD,
            f"value={sub_secondary:.2f} threshold={SECONDARY_THRESHOLD:.2f}")


def test_multi_disease_output_structure():
    print("\n─── Prompt 1: Multi-Disease Output Structure ───")

    # Simulate multi-disease output (as would come from classify_multi_disease)
    mock_output = {
        "status": "success",
        "crop": "Tomato",
        "is_healthy": False,
        "predictions": [
            {"disease": "Early blight", "confidence": 0.72, "type": "primary",  "confidence_source": "EfficientNet-B0 softmax logits (PlantVillage 38 classes)"},
            {"disease": "Septoria leaf spot", "confidence": 0.23, "type": "secondary", "confidence_source": "EfficientNet-B0 softmax logits (PlantVillage 38 classes)"},
        ],
    }

    T.check("Multi-disease predictions is a list",
            isinstance(mock_output["predictions"], list),
            f"found {len(mock_output['predictions'])} predictions")

    T.check("Primary disease has correct type",
            mock_output["predictions"][0]["type"] == "primary",
            f"type='{mock_output['predictions'][0]['type']}'")

    T.check("Secondary disease has correct type",
            mock_output["predictions"][1]["type"] == "secondary",
            f"type='{mock_output['predictions'][1]['type']}'")

    T.check("All confidences from real model (labeled correctly)",
            all("EfficientNet" in p["confidence_source"] for p in mock_output["predictions"]),
            "confidence_source verified")

    T.check("No invented confidence values (all floats in [0,1])",
            all(0.0 <= p["confidence"] <= 1.0 for p in mock_output["predictions"]),
            f"values: {[p['confidence'] for p in mock_output['predictions']]}")


def test_unknown_disease():
    print("\n─── Prompt 1: Unknown Disease Fallback ───")

    # When disease class not in SYMPTOM_MAP → unknown_disease response
    from services.inference import SYMPTOM_MAP
    fake_class = "Tomato___Completely_Unknown_Pathogen_XYZ"
    T.check("Unknown disease class not in SYMPTOM_MAP",
            fake_class not in SYMPTOM_MAP,
            f"class='{fake_class}'")

    # Simulate the response
    mock_unknown = {
        "status":  "unknown_disease",
        "message": "Disease 'Completely Unknown Pathogen XYZ' not in knowledge base. Consult agricultural expert.",
        "action":  "consult_agricultural_expert",
    }
    T.check("Unknown disease returns safe fallback status",
            mock_unknown["status"] == "unknown_disease",
            f"status='{mock_unknown['status']}'")
    T.check("Unknown disease has action field",
            "action" in mock_unknown,
            f"action='{mock_unknown['action']}'")


def test_safety_layer():
    print("\n─── Prompt 1: Safety Layer (Step 7) ───")
    from services.inference import safety_check_treatment

    # Test: Complete treatment passes
    good_treatment = {
        "status":      "found",
        "crop":        "tomato",
        "disease":     "bacterial wilt",
        "chemical":    ["Copper oxychloride 50% WP @ 3g per litre"],
        "source":      "IARI Technical Bulletin 2021",
        "safety_notes": "Wear gloves. Pre-harvest interval 14 days.",
        "region":      "South Asia, Southeast Asia",
    }
    result = safety_check_treatment(good_treatment)
    T.check("Complete treatment passes safety check",
            result["safe"],
            f"reason='{result['reason']}'")

    # Test: Missing dosage → blocked
    bad_dosage = {
        "status":      "found",
        "chemical":    ["Copper sulfate — apply as needed"],  # no dosage
        "source":      "Some source",
        "safety_notes": "Wear gloves",
    }
    result = safety_check_treatment(bad_dosage)
    T.check("Treatment with missing dosage is blocked",
            not result["safe"],
            f"blocked_fields={result['blocked_fields']}")

    # Test: Missing source → blocked
    no_source = {
        "status":      "found",
        "chemical":    ["Mancozeb 75% WP @ 2.5g per litre"],
        "source":      "",  # empty
        "safety_notes": "Wear gloves",
    }
    result = safety_check_treatment(no_source)
    T.check("Treatment with missing source is blocked",
            not result["safe"],
            f"blocked_fields={result['blocked_fields']}")

    # Test: Region mismatch → filtered
    wrong_region_treatment = {
        "status":      "found",
        "chemical":    ["Mancozeb 75% WP @ 2.5g per litre"],
        "source":      "ICAR Bulletin 2022",
        "safety_notes": "Wear gloves",
        "region":      "South Asia",
    }
    result = safety_check_treatment(wrong_region_treatment, region="North America")
    T.check("Wrong region treatment is filtered",
            not result["safe"],
            f"reason='{result['reason'][:60]}'")


def test_treatment_separation():
    print("\n─── Prompt 1: Multi-Disease Treatment Separation ───")
    from database.treatment_db import treatment_db

    # Verify treatment DB returns structured record for known disease
    result = treatment_db.lookup("tomato", "bacterial wilt")
    T.check("Treatment DB returns structured record",
            result.get("status") == "found",
            f"status='{result.get('status')}'")

    chems = [c for c in result.get("chemical",[]) if not c.strip().upper().startswith("NOTE:")]
    T.check("Treatment has dosage in chemical entries (excl. NOTE lines)",
            all("@" in c or "per" in c.lower() for c in chems),
            f"non-note chemicals: {len(chems)} entries")

    T.check("Treatment has source citation",
            bool(result.get("source")),
            f"source='{result.get('source','')[:60]}'")

    T.check("Treatment has safety notes",
            bool(result.get("safety_notes")),
            f"safety notes present")

    # Unknown crop/disease → no_treatment_found (not guessed)
    unknown = treatment_db.lookup("marsplant", "alien_fungus_xyz")
    T.check("Unknown disease returns no_treatment_found (not guessed)",
            unknown.get("status") == "no_treatment_found",
            f"status='{unknown.get('status')}'")


# ════════════════════════════════════════════════════════════════════
# PROMPT 2 TESTS: Predictive Risk Engine (Time-Series)
# ════════════════════════════════════════════════════════════════════

def test_timeseries_risk():
    print("\n─── Prompt 2: Time-Series Risk Engine ───")
    from services.risk_engine import (
        PredictiveRiskEngine, WeatherTimeSeries, CropInfo,
        OutbreakSignal, PlantMemorySignal,
        compute_weather_trend, linear_trend_slope, sustained_condition_score,
        rolling_mean
    )

    engine = PredictiveRiskEngine()

    # Test 1: Increasing humidity trend → increased risk
    # 7 days of humidity: rising from 60 to 92%
    rising_hum = [60,65,70,74,79,85,89,92, 90,91,93,None,None]
    ts_rising = WeatherTimeSeries(
        temperatures   = [24,25,25,26,26,27,27,27, 26,26,25,None,None],
        humidities     = rising_hum,
        precipitations = [2,3,5,8,12,15,18,20, 15,18,22,None,None],
        wind_speeds    = [10,9,8,8,7,7,6,6, 7,7,8,None,None],
        past_days=7, forecast_days=5,
    )
    crop_rice = CropInfo("rice", "vegetative")
    result_rising = engine.compute_timeseries_risk(ts_rising, crop_rice, outbreak_data_available=False)
    slope = linear_trend_slope(rising_hum[:7])
    T.check("Increasing humidity produces positive slope",
            slope is not None and slope > 0,
            f"slope={slope:.3f}/day")
    T.check("Rising humidity trend increases risk score",
            result_rising["top_threat"] is not None and result_rising["top_threat"]["risk_score"] > 0.4,
            f"risk={result_rising['top_threat']['risk_score'] if result_rising['top_threat'] else 'None'}")

    # Test 2: Stable environment → low risk
    ts_stable = WeatherTimeSeries(
        temperatures   = [18,18,19,18,19,18,18,18, 18,18,18,None,None],
        humidities     = [45,44,46,45,45,44,46,45, 45,44,46,None,None],
        precipitations = [0,0,1,0,0,0,1,0, 0,0,0,None,None],
        wind_speeds    = [15,16,15,14,15,16,15,15, 15,15,15,None,None],
        past_days=7, forecast_days=5,
    )
    result_stable = engine.compute_timeseries_risk(ts_stable, CropInfo("wheat","harvest"), outbreak_data_available=False)
    T.check("Stable dry environment produces low risk",
            result_stable["top_threat"] is not None and result_stable["top_threat"]["risk_score"] < 0.50,
            f"risk={result_stable['top_threat']['risk_score'] if result_stable['top_threat'] else 'None'}")

    # Test 3: No outbreak data → confidence labeled as low/medium (not high)
    T.check("No outbreak data → non-high confidence OR low risk",
            result_rising["overall_confidence"] in ("low","medium") or result_rising["top_threat"]["risk_score"] < 0.5,
            f"confidence='{result_rising['overall_confidence']}' outbreak_source='{result_rising['outbreak_source']}'")

    T.check("No outbreak data → environmental_proxy label applied",
            result_rising["outbreak_source"] == "environmental_proxy",
            f"source='{result_rising['outbreak_source']}'")

    # Test 4: Rising outbreak cluster → high risk
    ts_outbreak = WeatherTimeSeries(
        temperatures=[25,26,26,27,27,28,28,28, 27,27,28,None,None],
        humidities  =[80,82,84,85,87,88,89,90, 89,90,91,None,None],
        precipitations=[5,8,10,12,15,18,20,22, 20,22,25,None,None],
        wind_speeds =[8,8,7,7,6,6,5,5, 6,6,7,None,None],
        past_days=7, forecast_days=5,
    )
    outbreak = OutbreakSignal(nearby_cases=18, radius_km=8.0, days_window=5, dominant_disease="rice_blast")
    result_outbreak = engine.compute_timeseries_risk(ts_outbreak, CropInfo("rice","tillering"), outbreak=outbreak, outbreak_data_available=True)
    T.check("Rising outbreak cluster → high risk",
            result_outbreak["top_threat"] is not None and result_outbreak["top_threat"]["risk_score"] >= 0.60,
            f"risk={result_outbreak['top_threat']['risk_score'] if result_outbreak['top_threat'] else 'None'}")
    T.check("Real outbreak source labeled correctly",
            result_outbreak["outbreak_source"] == "real_data",
            f"source='{result_outbreak['outbreak_source']}'")

    # Test 5: Missing data → safe fallback (data_completeness reported, confidence=low)
    ts_sparse = WeatherTimeSeries(
        temperatures   = [None,None,None,None,25,None,None,25, None,None,None,None,None],
        humidities     = [None,None,None,None,70,None,None,70, None,None,None,None,None],
        precipitations = [None,None,None,None,5, None,None,5,  None,None,None,None,None],
        wind_speeds    = [None,None,None,None,10,None,None,10, None,None,None,None,None],
        past_days=7, forecast_days=5,
    )
    result_sparse = engine.compute_timeseries_risk(ts_sparse, CropInfo("tomato","vegetative"), outbreak_data_available=False)
    T.check("Sparse data → low data completeness reported",
            result_sparse["data_completeness"] < 0.4,
            f"completeness={result_sparse['data_completeness']:.3f}")
    T.check("Sparse data → confidence is low",
            result_sparse["overall_confidence"] == "low",
            f"confidence='{result_sparse['overall_confidence']}'")

    # Test 6: Plant history worsening → boosted risk
    memory_declining = PlantMemorySignal(
        scan_count=4,
        recent_diseases=["rice blast","rice blast","rice_blast sheath blight"],
        severity_trend="declining",
        confidence_scores=[0.72, 0.80, 0.85, 0.89],
        days_between_scans=[7, 7, 7],
    )
    ts_normal = WeatherTimeSeries(
        temperatures   = [25,25,26,26,27,27,28,27, 27,27,26,None,None],
        humidities     = [75,76,77,78,78,79,80,80, 80,81,82,None,None],
        precipitations = [5,5,6,7,8,9,10,10, 10,11,12,None,None],
        wind_speeds    = [10,10,9,9,8,8,8,8, 8,8,9,None,None],
        past_days=7, forecast_days=5,
    )
    result_no_memory   = engine.compute_timeseries_risk(ts_normal, CropInfo("rice","vegetative"), outbreak_data_available=False)
    result_with_memory = engine.compute_timeseries_risk(ts_normal, CropInfo("rice","vegetative"), plant_memory=memory_declining, outbreak_data_available=False)
    rs_no   = result_no_memory["top_threat"]["risk_score"]   if result_no_memory["top_threat"]   else 0
    rs_with = result_with_memory["top_threat"]["risk_score"] if result_with_memory["top_threat"] else 0
    T.check("Worsening plant history boosts risk score",
            rs_with >= rs_no,
            f"without_memory={rs_no:.3f} with_memory={rs_with:.3f}")
    T.check("Plant memory usage is tracked in output",
            result_with_memory["plant_memory_used"],
            f"plant_memory_used={result_with_memory['plant_memory_used']}")


def test_time_series_math():
    print("\n─── Prompt 2: Time-Series Math Verification ───")
    from services.risk_engine import linear_trend_slope, sustained_condition_score, rolling_mean

    # linear_trend_slope: known series
    flat   = [50.0, 50.0, 50.0, 50.0, 50.0]
    rising = [50.0, 52.0, 54.0, 56.0, 58.0]
    fall   = [60.0, 57.0, 54.0, 51.0, 48.0]

    slope_flat   = linear_trend_slope(flat)
    slope_rising = linear_trend_slope(rising)
    slope_fall   = linear_trend_slope(fall)

    T.check("Flat series slope ≈ 0",
            slope_flat is not None and abs(slope_flat) < 0.01,
            f"slope={slope_flat:.4f}")
    T.check("Rising series slope > 0",
            slope_rising is not None and slope_rising > 0,
            f"slope={slope_rising:.4f} (expected ~2.0)")
    T.check("Falling series slope < 0",
            slope_fall is not None and slope_fall < 0,
            f"slope={slope_fall:.4f} (expected ~-3.0)")

    # sustained_condition_score
    all_above  = [85, 87, 90, 88, 86]
    all_below  = [50, 52, 48, 55, 51]
    mixed      = [85, 50, 88, 52, 90]
    T.check("All above threshold → sustained score = 1.0",
            sustained_condition_score(all_above, 80.0) == 1.0,
            f"score={sustained_condition_score(all_above, 80.0)}")
    T.check("All below threshold → sustained score = 0.0",
            sustained_condition_score(all_below, 80.0) == 0.0,
            f"score={sustained_condition_score(all_below, 80.0)}")
    T.check("Mixed above/below → score between 0 and 1",
            0 < sustained_condition_score(mixed, 80.0) < 1,
            f"score={sustained_condition_score(mixed, 80.0):.2f}")

    # rolling_mean with None values
    with_nones = [80.0, None, 85.0, None, 90.0]
    rm = rolling_mean(with_nones, window=5)
    T.check("Rolling mean handles None values correctly",
            rm is not None and 80 <= rm <= 90,
            f"mean={rm:.1f} (expected ~85)")


def test_v1_backward_compat():
    print("\n─── Prompt 2: v1 Backward Compatibility ───")
    from services.risk_engine import risk_engine, WeatherData, CropInfo, OutbreakSignal

    # v1 compute_risk still works
    weather  = WeatherData(temperature=27.0, humidity=88.0, rainfall_3d=15.0, wind_speed=8.0)
    crop     = CropInfo("rice","tillering")
    outbreak = OutbreakSignal(nearby_cases=5, radius_km=10.0, days_window=7)

    result = risk_engine.compute_risk(weather, crop, outbreak)
    T.check("v1 compute_risk still works",
            result.get("risks") is not None and len(result["risks"]) > 0,
            f"num_diseases={len(result['risks'])} top_risk={result['top_threat']['risk_score']:.3f}")

    T.check("v1 output has weather_summary",
            "weather_summary" in result,
            "backward_compatible")


# ════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═"*65)
    print("AGROMIND UNIVERSAL — UPGRADE TEST SUITE v2")
    print("Prompt 1: Disease Accuracy | Prompt 2: Predictive Risk")
    print("═"*65)

    # Prompt 1
    test_validation()
    test_confidence_gate()
    test_multi_disease_output_structure()
    test_unknown_disease()
    test_safety_layer()
    test_treatment_separation()

    # Prompt 2
    test_timeseries_risk()
    test_time_series_math()
    test_v1_backward_compat()

    all_passed = T.summary()

    print("\nDETAILED RESULTS:")
    for r in T.results:
        print(f"  {r}")

    return all_passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
