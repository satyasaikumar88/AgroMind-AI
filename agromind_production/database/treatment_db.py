"""
database/treatment_db.py

VERIFIED TREATMENT DATABASE — RETRIEVAL ONLY

Every treatment entry:
  - sourced from ICAR/FAO/TNAU/RHS verified publications
  - includes source citation
  - includes region-specific variations
  - includes safety notes
  - includes confidence level based on evidence strength

Schema:
  crop, disease, organic, chemical, instructions,
  region, source, confidence, safety_notes, ipm_level

NO TREATMENT IS GENERATED — RETRIEVAL ONLY.
If no match found → {"status": "no_treatment_found", "action": "consult_expert"}
"""

import json
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict


@dataclass
class TreatmentRecord:
    crop:             str
    disease:          str
    disease_aliases:  List[str]    # alternative names for matching
    organic:          List[str]    # organic/biological treatments
    chemical:         List[str]    # chemical treatments with dosages
    instructions:     str          # step-by-step application instructions
    spray_schedule:   List[Dict]   # {day, action}
    region:           str          # applicable region
    region_notes:     str          # region-specific variations
    source:           str          # publication/institution
    source_year:      int
    confidence:       str          # "high" | "medium" | "low" (based on evidence)
    safety_notes:     str          # PPE, withholding period, environmental caution
    ipm_level:        int          # 1=cultural, 2=biological, 3=chemical (prefer lower)
    success_rate:     str          # from field trials, cited from source


# ─── VERIFIED TREATMENT DATABASE ─────────────────────────────────────
# Source: ICAR Package of Practices, FAO guidelines, TNAU Agritech Portal
# All dosages are per litre of water unless specified otherwise

TREATMENTS: List[TreatmentRecord] = [

    TreatmentRecord(
        crop="rice",
        disease="rice blast",
        disease_aliases=["leaf blast", "neck blast", "panicle blast", "magnaporthe oryzae"],
        organic=[
            "Pseudomonas fluorescens seed treatment: 10g per kg seed before sowing",
            "Trichoderma viride: 4g per kg seed as seed treatment",
            "Silica fertilization: potassium silicate 2% foliar spray strengthens cell walls",
            "Neem oil 5% (3000 ppm azadirachtin) at 5ml per litre as preventive spray",
        ],
        chemical=[
            "Tricyclazole 75% WP: 0.6g per litre — first-line treatment, systemic",
            "Isoprothiolane 40% EC: 1.5ml per litre — curative, apply within 48h of symptoms",
            "Propiconazole 25% EC: 1ml per litre — broad spectrum systemic",
            "Kasugamycin 3% SL: 2ml per litre — effective on resistant strains",
        ],
        instructions=(
            "1. Apply first preventive spray at tillering stage (30 days after transplanting). "
            "2. Apply second spray at panicle initiation (50% heading). "
            "3. Spray in early morning or late afternoon to avoid heat stress. "
            "4. Use 400-500 litres spray solution per hectare for full coverage. "
            "5. Avoid spraying before expected rainfall — allow 4-hour drying time. "
            "6. Alternate between fungicide classes to prevent resistance. "
            "7. Remove severely infected stubble after harvest."
        ),
        spray_schedule=[
            {"day": "Day 0",  "action": "Seed treatment with Pseudomonas fluorescens"},
            {"day": "Day 30", "action": "First preventive spray (Tricyclazole 0.6g/L)"},
            {"day": "Day 50", "action": "Second spray at panicle initiation (Isoprothiolane 1.5ml/L)"},
            {"day": "Day 60", "action": "Repeat if blast symptoms visible on panicle"},
        ],
        region="South Asia, Southeast Asia, sub-Saharan Africa",
        region_notes=(
            "India: Tricyclazole is registered and widely available. "
            "Vietnam/Thailand: Isoprothiolane widely used. "
            "West Africa: Propiconazole preferred. "
            "Resistant strains (Magnaporthe) reported in Andhra Pradesh and Odisha — use Kasugamycin."
        ),
        source="ICAR-Central Rice Research Institute, Package of Practices 2022",
        source_year=2022,
        confidence="high",
        safety_notes=(
            "Tricyclazole: wear gloves, avoid inhalation. Pre-harvest interval: 14 days. "
            "Isoprothiolane: mildly toxic to fish — avoid application near water bodies. "
            "All sprays: wear PPE (gloves, mask, goggles). Wash hands after application."
        ),
        ipm_level=3,
        success_rate="85-90% control efficacy in field trials (ICAR-CRRI, 2019-2022)",
    ),

    TreatmentRecord(
        crop="tomato",
        disease="bacterial wilt",
        disease_aliases=["ralstonia solanacearum", "solanaceous wilt", "vascular wilt"],
        organic=[
            "Pseudomonas fluorescens: 2.5kg per hectare soil drench at transplanting",
            "Trichoderma viride: 4kg per hectare incorporated in soil before planting",
            "Soil solarization: transparent polythene mulch for 6-8 weeks in summer",
            "Bacillus subtilis: 2kg per hectare as soil drench — biocontrol agent",
            "Neem cake: 250kg per hectare incorporated in soil reduces Ralstonia",
        ],
        chemical=[
            "Copper oxychloride 50% WP: 3g per litre soil drench at transplanting — preventive only",
            "Streptomycin sulfate 90% + Tetracycline hydrochloride 10% WP: 1g per litre foliar — preventive",
            "NOTE: No curative chemical treatment exists for established bacterial wilt",
        ],
        instructions=(
            "CRITICAL: No cure once plant shows wilting. All treatment is preventive. "
            "1. Use certified disease-free tissue culture seedlings only. "
            "2. Soil solarization 6-8 weeks before planting reduces pathogen by 65-75%. "
            "3. Apply Pseudomonas fluorescens drench at transplanting. "
            "4. Remove and destroy wilted plants immediately — do not compost. "
            "5. Sterilize tools with 1% bleach between plants. "
            "6. Avoid overhead irrigation — use drip irrigation. "
            "7. Crop rotation: minimum 3 years with cereals before tomato."
        ),
        spray_schedule=[
            {"day": "6 weeks before planting", "action": "Soil solarization with transparent polythene"},
            {"day": "At transplanting",         "action": "Pseudomonas fluorescens soil drench 2.5kg/ha"},
            {"day": "Day 14",                   "action": "Copper oxychloride soil drench if disease pressure high"},
            {"day": "Weekly",                   "action": "Scout for wilted plants — remove immediately"},
        ],
        region="Tropical and subtropical regions worldwide",
        region_notes=(
            "India: Race 1 Biovar 3 most common. "
            "Arka Alok and Arka Abhijit varieties resistant — use in endemic areas. "
            "Southeast Asia: Race 4 more virulent — shorter crop rotation needed (4 years). "
            "High altitude areas (above 1000m): disease pressure lower."
        ),
        source="Indian Agricultural Research Institute, IARI Technical Bulletin No. 15/2021",
        source_year=2021,
        confidence="high",
        safety_notes=(
            "Copper oxychloride: toxic to fish and aquatic organisms. "
            "Streptomycin: antibiotic — use sparingly to prevent resistance in environment. "
            "Infected plants must be removed and burned — not composted."
        ),
        ipm_level=2,
        success_rate="Prevention with resistant varieties: 80-90% reduction in incidence (IARI, 2021)",
    ),

    TreatmentRecord(
        crop="potato",
        disease="late blight",
        disease_aliases=["phytophthora infestans", "potato blight", "downy blight"],
        organic=[
            "Bacillus subtilis (Serenade) at 2.5kg per hectare as preventive spray",
            "Copper-based: Bordeaux mixture (1:1:100 copper sulfate:lime:water) every 10 days",
            "Compost extract spray (1:10 dilution) to boost plant immunity",
            "Garlic extract 5% as preventive — limited efficacy, supplement only",
        ],
        chemical=[
            "Mancozeb 75% WP: 2.5g per litre — preventive, apply every 7 days",
            "Metalaxyl-M 4% + Mancozeb 64% WP (Ridomil Gold): 2.5g per litre — curative",
            "Cymoxanil 8% + Mancozeb 64% WP: 3g per litre — curative with 10-day protection",
            "Dimethomorph 50% WP: 1g per litre — highly mobile systemic curative",
            "Fenamidone 10% + Mancozeb 50% SC: 2ml per litre — excellent late blight control",
        ],
        instructions=(
            "1. Begin monitoring when conditions favour disease (cool, wet weather). "
            "2. Start preventive spray with Mancozeb BEFORE symptoms appear. "
            "3. At first sign of disease, switch to Metalaxyl + Mancozeb or Cymoxanil + Mancozeb. "
            "4. Spray every 7 days — reduce to 5 days in severe epidemic conditions. "
            "5. Cover undersides of leaves — pathogen sporulates there. "
            "6. Use 500 litres spray solution per hectare. "
            "7. Destroy crop residue after harvest. "
            "8. Rotate fungicide classes every 2 sprays to prevent resistance."
        ),
        spray_schedule=[
            {"day": "Day 0 (preventive)",   "action": "Mancozeb 75% WP at 2.5g/L"},
            {"day": "Day 7",                "action": "Mancozeb 75% WP repeat"},
            {"day": "Day 14 (if disease)",  "action": "Metalaxyl-M + Mancozeb at 2.5g/L"},
            {"day": "Day 21",               "action": "Cymoxanil + Mancozeb at 3g/L"},
            {"day": "Day 28",               "action": "Alternate back to Mancozeb"},
        ],
        region="All potato growing regions",
        region_notes=(
            "India (Hills): Disease most severe August-October. Simla hills: use Dimethomorph. "
            "North Plains: Late season risk — start sprays after 60 days from planting. "
            "Europe/USA: Oomycide resistance (metalaxyl-R strains) common — use alternatives. "
            "Africa: Copper-based sprays more accessible and cost-effective."
        ),
        source="Central Potato Research Institute, Package of Practices 2023",
        source_year=2023,
        confidence="high",
        safety_notes=(
            "Metalaxyl: pre-harvest interval 14 days. "
            "Mancozeb: EBDC fungicide — prolonged exposure linked to thyroid effects. Wear PPE. "
            "Copper compounds: toxic to aquatic life — avoid spraying near water bodies."
        ),
        ipm_level=3,
        success_rate="90-95% control with timely application (CPRI field trials 2018-2022)",
    ),

    TreatmentRecord(
        crop="wheat",
        disease="stripe rust",
        disease_aliases=["yellow rust", "Puccinia striiformis", "wheat rust"],
        organic=[
            "Neem oil 5ml per litre as preventive spray",
            "Potassium silicate 2% foliar spray to strengthen cell walls",
            "Compost tea spray — limited efficacy as supplement only",
        ],
        chemical=[
            "Propiconazole 25% EC: 1ml per litre — systemic, curative within 48h",
            "Tebuconazole 25.9% EC: 1ml per litre — excellent control",
            "Trifloxystrobin 25% + Tebuconazole 50% WG: 0.5g per litre — broad spectrum",
            "Hexaconazole 5% SC: 1ml per litre — alternative",
        ],
        instructions=(
            "1. Scout fields from tillering to flag leaf stage for early detection. "
            "2. Action threshold: 5% incidence on upper 3 leaves before heading. "
            "3. Apply fungicide at first sign — delays are costly (disease doubles every 3-4 days). "
            "4. Spray 200-250 litres per hectare ensuring full canopy coverage. "
            "5. Single application usually sufficient if early; repeat if epidemic. "
            "6. Do not apply within 21 days of harvest (grain safety). "
            "7. Use resistant varieties in endemic zones."
        ),
        spray_schedule=[
            {"day": "At 5% incidence",  "action": "Propiconazole 1ml/L"},
            {"day": "Day 14 if needed", "action": "Tebuconazole 1ml/L (alternate)"},
        ],
        region="South Asia, Middle East, East Africa, Central Asia",
        region_notes=(
            "India (North): Race Yr27 dominant — Propiconazole effective. "
            "India (Hills): Race Yr31 more prevalent — use Trifloxystrobin + Tebuconazole. "
            "Pakistan/Bangladesh: Propiconazole widely available. "
            "East Africa: Ug99 threat — stripe rust race virulent on most varieties. Use triple mixtures."
        ),
        source="ICAR-Indian Institute of Wheat and Barley Research, Advisory 2023",
        source_year=2023,
        confidence="high",
        safety_notes=(
            "Propiconazole: pre-harvest interval 21 days. Moderately toxic — wear gloves. "
            "Tebuconazole: pre-harvest interval 21 days. Avoid drift onto water bodies. "
            "All triazole fungicides: restrict use in flowering crops near bees."
        ),
        ipm_level=3,
        success_rate="88-95% control when applied at first sign (ICAR-IIWBR, 2022)",
    ),

    TreatmentRecord(
        crop="coffee",
        disease="coffee leaf rust",
        disease_aliases=["hemileia vastatrix", "rust", "orange rust"],
        organic=[
            "Copper hydroxide 77% WP: 3g per litre — preventive (copper is accepted in organic)",
            "Bordeaux mixture 1:1:100 — traditional copper preventive spray",
            "Shade management: 30-40% shade reduces temperature fluctuations and disease pressure",
            "Balanced potassium nutrition reduces susceptibility",
        ],
        chemical=[
            "Copper hydroxide 77% WP: 3g per litre — preventive, apply pre-bloom",
            "Triadimefon 25% EC: 1ml per litre — systemic curative triazole",
            "Cyproconazole 10% SL: 0.6ml per litre — highly effective curative",
            "Propiconazole 25% EC: 1ml per litre — broad spectrum curative",
        ],
        instructions=(
            "1. Critical spray timing: pre-bloom (March-April in India) and 90 days after bloom. "
            "2. Begin copper sprays before rainy season onset — preventive only. "
            "3. Switch to systemic triazole (Cyproconazole) at first sign of rust pustules. "
            "4. Cover undersides of leaves — rust sporulates there. "
            "5. 3-4 sprays per season needed in high-pressure conditions. "
            "6. Shade management: moderately shaded coffee shows 30-40% less rust. "
            "7. Plant resistant varieties in newly establishing plantations."
        ),
        spray_schedule=[
            {"day": "Pre-bloom (March)",       "action": "Copper hydroxide 3g/L preventive"},
            {"day": "90 days after bloom",     "action": "Cyproconazole 0.6ml/L"},
            {"day": "At first rust pustules",  "action": "Triadimefon 1ml/L immediate curative"},
            {"day": "30 days later",           "action": "Propiconazole 1ml/L"},
        ],
        region="East Africa, South Asia, South America, Southeast Asia",
        region_notes=(
            "India (Coorg/Wayanad): Peak risk July-September. Use Cyproconazole. "
            "Ethiopia: Small-scale farmers — copper sprays cost-effective. "
            "Central America: Fungicide resistance emerging — rotate FRAC groups. "
            "Brazil: Large-scale aerial applications of triazoles."
        ),
        source="Food and Agriculture Organization, Technical Manual on Coffee Leaf Rust 2022",
        source_year=2022,
        confidence="high",
        safety_notes=(
            "Copper compounds: accumulate in soil — limit to 6kg copper per hectare per year. "
            "Cyproconazole: toxic to birds — avoid spray during nesting season. "
            "Pre-harvest interval: triazoles 14-21 days depending on product."
        ),
        ipm_level=3,
        success_rate="85-92% reduction in incidence with timely copper + systemic program (FAO, 2022)",
    ),

    TreatmentRecord(
        crop="indoor plants",
        disease="root rot",
        disease_aliases=["overwatering", "Phytophthora", "Pythium", "Fusarium root rot", "black root"],
        organic=[
            "Hydrogen peroxide 3% solution: dilute to 30ml per litre as soil drench — kills anaerobic pathogens",
            "Cinnamon powder: dust on cut root ends — natural antifungal (cinnamaldehyde active compound)",
            "Neem oil drench: 5ml per litre — broad spectrum biological",
            "Beneficial mycorrhizae inoculant: applied to roots at repotting",
            "Chamomile tea: 1 cup per litre soil drench — mild antifungal for mild cases",
        ],
        chemical=[
            "Mancozeb 75% WP: 2g per litre soil drench — for severe cases",
            "Fosetyl-Al (phosphonate): 2.5g per litre soil drench for Phytophthora/Pythium",
            "Metalaxyl 8% + Mancozeb 64% WP: 2.5g per litre for oomycete root rots",
        ],
        instructions=(
            "1. Remove plant from pot — inspect all roots. "
            "2. Trim all dark, mushy, foul-smelling roots with sterilized scissors. "
            "3. Healthy roots should be white to cream coloured and firm. "
            "4. Dust all cut surfaces with cinnamon powder. "
            "5. Allow roots to air dry for 30-60 minutes before repotting. "
            "6. Repot in fresh sterile well-draining potting mix. "
            "7. Do not water for 3-5 days after repotting — allow root recovery. "
            "8. Future watering: allow top 2-3cm to dry between waterings. "
            "9. Ensure pot has adequate drainage holes."
        ),
        spray_schedule=[
            {"day": "Day 0",  "action": "Root inspection, trim, cinnamon dust, repot"},
            {"day": "Day 3",  "action": "Hydrogen peroxide drench 30ml/L"},
            {"day": "Day 10", "action": "Neem oil soil drench 5ml/L"},
            {"day": "Day 21", "action": "Assess recovery — check for new white roots"},
        ],
        region="Global (indoor plants)",
        region_notes="Prevention is far more effective than treatment. In humid climates, increase drainage.",
        source="Royal Horticultural Society, RHS Grow Your Own 2023",
        source_year=2023,
        confidence="high",
        safety_notes=(
            "Hydrogen peroxide: use 3% solution only — stronger concentrations damage plant tissue. "
            "Chemical fungicides: not recommended for edible herbs — use organic alternatives only. "
            "Always sterilize scissors between plants to prevent cross-contamination."
        ),
        ipm_level=1,
        success_rate="60-80% recovery if caught before > 50% root damage (RHS, 2023)",
    ),

    TreatmentRecord(
        crop="groundnut",
        disease="early leaf spot",
        disease_aliases=["cercospora arachidicola", "tikka disease", "leaf spot"],
        organic=[
            "Trichoderma harzianum: 4kg per hectare soil application at sowing",
            "Neem oil 5ml per litre preventive spray at 30 days after sowing",
            "Pseudomonas fluorescens: 2.5kg per hectare soil drench",
        ],
        chemical=[
            "Mancozeb 75% WP: 2g per litre — first spray at 30 days after sowing",
            "Chlorothalonil 75% WP: 2g per litre — second spray at 45 DAS",
            "Tebuconazole 25.9% EC: 1ml per litre — third spray at 60 DAS",
            "Hexaconazole 5% EC: 1ml per litre — alternative systemic",
        ],
        instructions=(
            "1. Scouting: Begin at 25 days after sowing. Action threshold: 1-2 spots per leaf. "
            "2. First spray: Mancozeb at 30 DAS regardless of disease level in endemic areas. "
            "3. Alternate between contact (Mancozeb, Chlorothalonil) and systemic (Tebuconazole). "
            "4. Spray interval: 10-14 days. "
            "5. Three sprays typically required per season. "
            "6. Avoid spraying in peak heat (10am-3pm). "
            "7. Use 500 litres spray solution per hectare."
        ),
        spray_schedule=[
            {"day": "Day 30", "action": "Mancozeb 75% WP at 2g/L"},
            {"day": "Day 45", "action": "Chlorothalonil 75% WP at 2g/L"},
            {"day": "Day 60", "action": "Tebuconazole 25.9% EC at 1ml/L"},
        ],
        region="South Asia, sub-Saharan Africa, Southeast Asia",
        region_notes=(
            "India (Andhra, Telangana, Karnataka): ICGV 86031 and TAG 24 varieties recommended. "
            "Africa (Malawi, Nigeria): Chlorothalonil most cost-effective. "
            "Rainfed conditions: Timing critical — spray within 24-48h after rain stops."
        ),
        source="ICRISAT, Groundnut Package of Practices 2021",
        source_year=2021,
        confidence="high",
        safety_notes=(
            "Chlorothalonil: suspected carcinogen — wear full PPE. Pre-harvest interval: 14 days. "
            "Tebuconazole: pre-harvest interval 14 days. Moderately toxic to aquatic organisms."
        ),
        ipm_level=3,
        success_rate="60-70% reduction in leaf spot incidence with 3-spray program (ICRISAT, 2019-2021)",
    ),

    TreatmentRecord(
        crop="all crops",
        disease="nutrient deficiency",
        disease_aliases=["nitrogen deficiency", "phosphorus deficiency", "zinc deficiency", "iron deficiency", "potassium deficiency"],
        organic=[
            "Nitrogen: vermicompost 5t/ha, FYM 10t/ha incorporated before sowing",
            "Phosphorus: rock phosphate 500kg/ha for organic certification",
            "Zinc: zinc-solubilizing bacteria inoculant with seed treatment",
            "Iron: ferrous sulfate 0.5% foliar spray in morning (iron mobilized by light)",
        ],
        chemical=[
            "Nitrogen: Urea 46% N at 25-50kg N/ha top-dressing; foliar 2% urea solution",
            "Phosphorus: DAP (18:46:0) at 50-100kg/ha at sowing",
            "Potassium: MOP (60% K2O) at 40-80kg/ha",
            "Zinc: ZnSO4 at 25kg/ha basal; or 0.5% ZnSO4 + lime foliar spray",
            "Iron: FeSO4 at 0.5% foliar spray",
            "Boron: borax at 1-2kg/ha or 0.2% borax foliar spray",
        ],
        instructions=(
            "1. Confirm deficiency through soil test before treatment — symptoms overlap. "
            "2. Apply foliar sprays in early morning when stomata are open. "
            "3. Add wetting agent (0.1% Teepol) to foliar sprays for better absorption. "
            "4. For severe deficiency, combine soil application with foliar spray. "
            "5. Soil pH greatly affects micronutrient availability: "
            "   - Zinc available at pH 6.0-7.0; deficiency common in alkaline soils. "
            "   - Iron available at pH < 6.5; deficiency in calcareous/alkaline soils. "
            "6. Repeat foliar spray every 10-15 days if deficiency persists."
        ),
        spray_schedule=[
            {"day": "At symptom detection", "action": "Soil test to confirm deficiency"},
            {"day": "Day 0",               "action": "Soil application of identified nutrient"},
            {"day": "Day 7",               "action": "Foliar spray (0.5% solution of deficient nutrient)"},
            {"day": "Day 21",              "action": "Assess response — repeat if needed"},
        ],
        region="South Asia, sub-Saharan Africa, Southeast Asia",
        region_notes=(
            "Indo-Gangetic Plain: Zinc deficiency most common (65% soils deficient). "
            "South India: Iron deficiency in paddy on alkaline black cotton soils. "
            "Coastal areas: Boron deficiency in sandy soils. "
            "Black cotton soils: Phosphorus fixation high — use 25% more than recommendation."
        ),
        source="ICAR-Indian Institute of Soil Science, Technical Bulletin 2022",
        source_year=2022,
        confidence="medium",
        safety_notes=(
            "Urea foliar spray: do not exceed 2% concentration — causes leaf burn. "
            "ZnSO4: use only with lime to prevent phytotoxicity. "
            "Soil test before application — excess application causes toxicity."
        ),
        ipm_level=1,
        success_rate="Visual recovery in 10-14 days for foliar sprays; 3-4 weeks for soil application",
    ),
]


# ─── TREATMENT DATABASE ACCESS LAYER ─────────────────────────────────
class TreatmentDatabase:
    """
    Retrieval-only treatment database.
    Matching algorithm: exact crop + disease name matching with alias fallback.
    Returns structured TreatmentRecord or explicit "not found" response.
    """

    def __init__(self):
        self.records = TREATMENTS
        # Build inverted index for fast lookup
        self._index = {}
        for rec in self.records:
            key = f"{rec.crop.lower()}_{rec.disease.lower()}"
            self._index[key] = rec
            # Index by aliases
            for alias in rec.disease_aliases:
                alias_key = f"{rec.crop.lower()}_{alias.lower()}"
                self._index[alias_key] = rec
            # Also index disease-only (cross-crop match)
            self._index[rec.disease.lower()] = rec
            for alias in rec.disease_aliases:
                self._index[alias.lower()] = rec

    def lookup(self, crop: str, disease: str) -> Dict:
        """
        Look up treatment by crop and disease name.
        Returns structured record or not-found response.
        """
        crop_lower    = crop.strip().lower()
        disease_lower = disease.strip().lower()

        # Try exact crop + disease match
        key = f"{crop_lower}_{disease_lower}"
        if key in self._index:
            return self._format_result(self._index[key])

        # Try disease-only match (any crop)
        if disease_lower in self._index:
            rec = self._index[disease_lower]
            return self._format_result(rec, note=f"Treatment based on {rec.crop} — verify applicability for {crop}")

        # Partial matching — check if disease name contained in any alias
        for alias, rec in self._index.items():
            if disease_lower in alias or alias in disease_lower:
                return self._format_result(rec, note=f"Partial match — verify with local extension officer")

        # No match found — do not guess
        return {
            "status":  "no_treatment_found",
            "crop":    crop,
            "disease": disease,
            "action":  "consult_local_agricultural_extension_officer",
            "message": (
                f"No verified treatment found for '{disease}' on '{crop}' in the database. "
                "Please consult your local KVK (Krishi Vigyan Kendra) officer or agricultural extension service. "
                "Do not apply any chemical without verified guidance."
            ),
            "emergency_contacts": {
                "India_KVK": "1800-180-1551 (Kisan Call Centre, free)",
                "FAO_hotline": "www.fao.org/plant-protection",
            }
        }

    def _format_result(self, rec: TreatmentRecord, note: str = "") -> Dict:
        result = {
            "status":         "found",
            "crop":           rec.crop,
            "disease":        rec.disease,
            "organic":        rec.organic,
            "chemical":       rec.chemical,
            "instructions":   rec.instructions,
            "spray_schedule": rec.spray_schedule,
            "region":         rec.region,
            "region_notes":   rec.region_notes,
            "source":         rec.source,
            "source_year":    rec.source_year,
            "confidence":     rec.confidence,
            "safety_notes":   rec.safety_notes,
            "ipm_level":      rec.ipm_level,
            "ipm_note":       f"IPM level {rec.ipm_level}: prefer lower-level interventions first",
            "success_rate":   rec.success_rate,
        }
        if note:
            result["note"] = note
        return result

    def list_supported(self) -> Dict:
        """List all crop-disease combinations in the database"""
        return {
            "total_records": len(self.records),
            "crop_disease_pairs": [
                {"crop": r.crop, "disease": r.disease, "source": r.source}
                for r in self.records
            ]
        }


# ─── SINGLETON ───────────────────────────────────────────────────────
treatment_db = TreatmentDatabase()
