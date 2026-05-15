"""
services/risk_engine.py  [UPGRADED v2]

UPGRADES over v1:
  1. Time-series weather analysis: rolling averages, trend slopes, sustained conditions
  2. Forecast integration: next 3-5 days weather affects risk score
  3. Dynamic weight fusion: outbreak weight 0-20% based on data availability
  4. Spatio-temporal fusion: weather trend + outbreak growth + crop stage combined
  5. Plant memory integration: worsening scan history boosts risk_score
  6. Confidence model: output confidence reflects data completeness
  7. Environmental proxy for cold-start (no fake data — labeled clearly)
  8. Future disease forecast output with prediction_window

v1 WeatherData, CropInfo, OutbreakSignal, DISEASE_WEATHER_PARAMS preserved.
compute_risk() signature unchanged for backward compatibility.
New method: compute_timeseries_risk() for upgraded pipeline.
"""

import math
from typing import Optional, List, Dict
from dataclasses import dataclass, field


@dataclass
class WeatherData:
    temperature: float
    humidity:    float
    rainfall_3d: float
    wind_speed:  float
    dew_point:   Optional[float] = None

@dataclass
class CropInfo:
    crop_type:           str
    growth_stage:        str
    days_since_planting: int = 0

@dataclass
class OutbreakSignal:
    nearby_cases:     int
    radius_km:        float
    days_window:      int
    dominant_disease: str = ""

@dataclass
class WeatherTimeSeries:
    """7-day history + 5-day forecast. None = missing value (handled safely)."""
    temperatures:   List[Optional[float]]
    humidities:     List[Optional[float]]
    precipitations: List[Optional[float]]
    wind_speeds:    List[Optional[float]]
    past_days:      int = 7
    forecast_days:  int = 5

    def today(self) -> WeatherData:
        idx = self.past_days
        return WeatherData(
            temperature = self.temperatures[idx] if idx < len(self.temperatures) and self.temperatures[idx] is not None else 25.0,
            humidity    = self.humidities[idx]    if idx < len(self.humidities)    and self.humidities[idx]    is not None else 70.0,
            rainfall_3d = sum(p for p in self.precipitations[max(0,idx-3):idx+1] if p is not None),
            wind_speed  = self.wind_speeds[idx]   if idx < len(self.wind_speeds)   and self.wind_speeds[idx]   is not None else 10.0,
        )

    def valid_count(self, series: List[Optional[float]]) -> int:
        return sum(1 for v in series if v is not None)

@dataclass
class PlantMemorySignal:
    scan_count:        int
    recent_diseases:   List[str]
    severity_trend:    str
    confidence_scores: List[float]
    days_between_scans: List[int]


DISEASE_WEATHER_PARAMS = {
    "rice_blast":        {"temp_min":20,"temp_max":30,"humidity_min":85,"rain_sensitivity":0.9,"name":"Rice Blast"},
    "rice_sheath_blight":{"temp_min":28,"temp_max":35,"humidity_min":80,"rain_sensitivity":0.7,"name":"Sheath Blight"},
    "late_blight":       {"temp_min":10,"temp_max":22,"humidity_min":90,"rain_sensitivity":0.95,"name":"Late Blight"},
    "powdery_mildew":    {"temp_min":15,"temp_max":28,"humidity_min":40,"rain_sensitivity":-0.3,"name":"Powdery Mildew"},
    "leaf_rust":         {"temp_min":15,"temp_max":25,"humidity_min":70,"rain_sensitivity":0.8,"name":"Leaf Rust"},
    "bacterial_wilt":    {"temp_min":25,"temp_max":35,"humidity_min":75,"rain_sensitivity":0.6,"name":"Bacterial Wilt"},
    "early_blight":      {"temp_min":24,"temp_max":30,"humidity_min":60,"rain_sensitivity":0.7,"name":"Early Blight"},
    "coffee_rust":       {"temp_min":21,"temp_max":25,"humidity_min":80,"rain_sensitivity":0.85,"name":"Coffee Leaf Rust"},
}

CROP_DISEASE_MAP = {
    "rice":["rice_blast","rice_sheath_blight"],"wheat":["leaf_rust"],
    "tomato":["bacterial_wilt","early_blight","late_blight"],"potato":["late_blight","early_blight"],
    "cotton":["bacterial_wilt"],"coffee":["coffee_rust"],"cucumber":["powdery_mildew"],
    "default":["early_blight","powdery_mildew"],
}

STAGE_MULTIPLIERS = {
    "seedling":1.4,"vegetative":1.2,"flowering":1.3,
    "grain_fill":1.1,"harvest":0.8,"unknown":1.0,
}


def rolling_mean(series, window=3):
    valid = [v for v in series[-window:] if v is not None]
    return sum(valid)/len(valid) if len(valid) >= 2 else None

def linear_trend_slope(series):
    pairs = [(i,v) for i,v in enumerate(series) if v is not None]
    n = len(pairs)
    if n < 3:
        return None
    xs,ys = [p[0] for p in pairs],[p[1] for p in pairs]
    sx,sy,sxy,sx2 = sum(xs),sum(ys),sum(x*y for x,y in zip(xs,ys)),sum(x*x for x in xs)
    denom = n*sx2 - sx*sx
    return (n*sxy - sx*sy)/denom if abs(denom) > 1e-10 else 0.0

def sustained_condition_score(series, threshold, window=5):
    recent = [v for v in series[-window:] if v is not None]
    return sum(1 for v in recent if v > threshold)/len(recent) if recent else 0.0

def compute_weather_trend(ts):
    past_h = ts.humidities[:ts.past_days]
    past_t = ts.temperatures[:ts.past_days]
    past_p = ts.precipitations[:ts.past_days]
    fore_h = ts.humidities[ts.past_days:]
    fore_p = ts.precipitations[ts.past_days:]
    fhv = [v for v in fore_h[:3] if v is not None]
    fpv = [v for v in fore_p[:3] if v is not None]
    total_exp = (ts.past_days+ts.forecast_days)*4
    total_pres = ts.valid_count(ts.humidities)+ts.valid_count(ts.temperatures)+ts.valid_count(ts.precipitations)+ts.valid_count(ts.wind_speeds)
    return {
        "humidity_slope":          round(linear_trend_slope(past_h),4) if linear_trend_slope(past_h) is not None else None,
        "temperature_slope":       round(linear_trend_slope(past_t),4) if linear_trend_slope(past_t) is not None else None,
        "sustained_high_humidity": round(sustained_condition_score(past_h,80.0),3),
        "sustained_rainfall":      round(sustained_condition_score(past_p,5.0),3),
        "forecast_avg_humidity":   round(sum(fhv)/len(fhv),1) if fhv else None,
        "forecast_rain_3d_mm":     round(sum(fpv),1) if fpv else None,
        "data_completeness":       round(total_pres/max(total_exp,1),3),
        "past_days_available":     ts.valid_count(past_h),
        "forecast_days_available": ts.valid_count(fore_h),
    }


class PredictiveRiskEngine:

    def compute_risk(self, weather: WeatherData, crop: CropInfo, outbreak: OutbreakSignal = None) -> dict:
        """v1 method — preserved exactly for backward compatibility."""
        diseases = CROP_DISEASE_MAP.get(crop.crop_type.lower(), CROP_DISEASE_MAP["default"])
        results  = []
        for disease_key in diseases:
            params = DISEASE_WEATHER_PARAMS.get(disease_key)
            if not params: continue
            ws = self._compute_weather_score(weather, params)
            sf = STAGE_MULTIPLIERS.get(crop.growth_stage.lower(),1.0)
            cs = min(sf*0.8,1.0)
            os_ = self._compute_outbreak_signal(outbreak, disease_key) if outbreak and outbreak.nearby_cases > 0 else 0.0
            rs = min(max(ws*0.50+cs*0.25+os_*0.25,0.0),1.0)
            rl,tw,ac = ("high","3-5 days","⛔ URGENT: Apply preventive treatment immediately") if rs>=0.70 else (("moderate","5-8 days","⚠️ WARNING: Increase monitoring to daily") if rs>=0.45 else ("low","8-14 days","✅ LOW RISK: Weekly monitoring sufficient"))
            results.append({"disease":params["name"],"disease_key":disease_key,"risk_score":round(rs,3),"risk_level":rl,"time_window":tw,"action":ac,"factors":self._explain_factors(weather,params,crop,outbreak),"component_scores":{"weather":round(ws,3),"crop_stage":round(cs,3),"outbreak":round(os_,3)}})
        results.sort(key=lambda x:x["risk_score"],reverse=True)
        return {"crop":crop.crop_type,"stage":crop.growth_stage,"risks":results,"top_threat":results[0] if results else None,"weather_summary":{"temperature":weather.temperature,"humidity":weather.humidity,"rainfall_3d":weather.rainfall_3d,"wind_speed":weather.wind_speed}}

    def compute_timeseries_risk(
        self,
        ts:            WeatherTimeSeries,
        crop:          CropInfo,
        outbreak:      Optional[OutbreakSignal] = None,
        plant_memory:  Optional[PlantMemorySignal] = None,
        outbreak_data_available: bool = False,
    ) -> Dict:
        """
        UPGRADED: Time-series predictive risk engine.
        
        Output:
          {
            "risk_score": 0.78,
            "prediction_window": "3-7 days",
            "confidence": "high|medium|low",
            "weather_trend": "...",
            "factors": [...],
            "component_scores": {...},
          }

        Dynamic weights: outbreak weight 0-20% based on data availability.
        Environmental proxy used for cold-start (labeled clearly, confidence=low).
        All numbers from real computation.
        """
        today        = ts.today()
        trend        = compute_weather_trend(ts)
        completeness = trend["data_completeness"]
        diseases     = CROP_DISEASE_MAP.get(crop.crop_type.lower(), CROP_DISEASE_MAP["default"])
        results      = []

        for disease_key in diseases:
            params = DISEASE_WEATHER_PARAMS.get(disease_key)
            if not params: continue

            w_current  = self._compute_weather_score(today, params)

            # Trend boost from 7-day history
            w_trend = 0.0
            hs = trend["humidity_slope"]
            if hs is not None:
                if hs > 0.5 and today.humidity > params["humidity_min"]*0.85:
                    w_trend += min(hs/5.0, 0.3)
                elif hs < -0.5:
                    w_trend -= 0.1
            w_trend += trend["sustained_high_humidity"]*0.2
            w_trend += trend["sustained_rainfall"]*0.15*max(params.get("rain_sensitivity",0.5),0)
            w_trend  = max(min(w_trend,0.5),-0.3)
            weather_combined = min(w_current+w_trend,1.0)

            # Forecast component
            forecast_score = 0.0
            if trend["forecast_avg_humidity"] is not None:
                fh = trend["forecast_avg_humidity"]
                if fh >= params["humidity_min"]:
                    forecast_score = min((fh-params["humidity_min"])/(100-params["humidity_min"]),1.0)*0.8
            if trend["forecast_rain_3d_mm"] is not None and params["rain_sensitivity"]>0:
                forecast_score += min(trend["forecast_rain_3d_mm"]/30.0,1.0)*params["rain_sensitivity"]*0.5
            forecast_score = min(forecast_score,1.0)

            # Crop stage
            crop_score = min(STAGE_MULTIPLIERS.get(crop.growth_stage.lower(),1.0)*0.8,1.0)

            # Outbreak: dynamic weight
            if outbreak_data_available and outbreak and outbreak.nearby_cases>0:
                ob_score,ob_weight,ob_src = self._compute_outbreak_signal(outbreak,disease_key),0.20,"real_geo_temporal_data"
            elif not outbreak_data_available:
                ob_score,ob_weight,ob_src = weather_combined*0.4,0.08,"environmental_proxy_low_confidence"
            else:
                ob_score,ob_weight,ob_src = 0.0,0.0,"no_outbreak_data"

            # Plant memory
            memory_boost = 0.0
            if plant_memory and plant_memory.scan_count >= 2:
                if plant_memory.severity_trend == "declining": memory_boost = 0.15
                elif plant_memory.severity_trend == "improving": memory_boost = -0.05
                if disease_key.replace("_"," ") in " ".join(plant_memory.recent_diseases).lower(): memory_boost += 0.10

            # Weighted fusion
            fixed_sum     = 0.30+0.25+0.15+0.10+ob_weight
            mem_weight    = max(0.0,1.0-fixed_sum-0.05)
            raw_score     = weather_combined*0.30 + min(max(w_trend,0),1.0)*0.25 + forecast_score*0.15 + crop_score*0.10 + ob_score*ob_weight + memory_boost*mem_weight
            risk_score    = min(max(raw_score,0.0),1.0)

            # Confidence
            conf_score    = completeness*0.50 + (0.30 if outbreak_data_available else 0.10) + (0.20 if plant_memory and plant_memory.scan_count>=3 else 0.05)
            conf_label    = "high" if conf_score>=0.75 else ("medium" if conf_score>=0.45 else "low")
            pred_window   = "3-5 days" if forecast_score>0.6 else ("5-7 days" if forecast_score>0.3 else "7-14 days")
            rl,ac         = ("high","⛔ URGENT: Apply preventive treatment immediately") if risk_score>=0.70 else (("moderate","⚠️ WARNING: Increase monitoring to daily") if risk_score>=0.45 else ("low","✅ LOW RISK: Weekly monitoring sufficient"))

            results.append({
                "disease":params["name"],"disease_key":disease_key,
                "risk_score":round(risk_score,3),"risk_level":rl,
                "prediction_window":pred_window,"confidence":conf_label,
                "confidence_score":round(conf_score,3),"action":ac,
                "factors":self._timeseries_factors(today,params,trend,crop,outbreak,plant_memory,ob_src),
                "component_scores":{"weather_current":round(weather_combined,3),"weather_trend":round(float(w_trend),3),"forecast":round(forecast_score,3),"crop_stage":round(crop_score,3),"outbreak":{"score":round(ob_score,3),"weight":ob_weight,"source":ob_src},"plant_memory":round(memory_boost,3)},
                "computation":{"formula":"risk=weather*0.30+trend*0.25+forecast*0.15+crop*0.10+outbreak*(0-0.20)+memory(residual)","dynamic_weights":True,"data_completeness":round(completeness,3)},
            })

        results.sort(key=lambda x:x["risk_score"],reverse=True)
        hs = trend.get("humidity_slope")
        wt = "insufficient_data" if hs is None else ("rapidly_increasing_humidity" if hs>1.0 else ("gradually_increasing_humidity" if hs>0.3 else ("decreasing_humidity" if hs<-0.5 else "stable_humidity")))
        return {"crop":crop.crop_type,"stage":crop.growth_stage,"risks":results,"top_threat":results[0] if results else None,"weather_trend":wt,"weather_analysis":trend,"outbreak_source":"real_data" if outbreak_data_available else "environmental_proxy","plant_memory_used":plant_memory is not None and plant_memory.scan_count>=2,"overall_confidence":results[0]["confidence"] if results else "low","data_completeness":round(completeness,3),"model_version":"v2_timeseries"}

    def _compute_weather_score(self, weather, params):
        score=0.0
        t_min,t_max=params["temp_min"],params["temp_max"]
        t_mid=(t_min+t_max)/2; t_range=(t_max-t_min)/2
        t_score=max(0,1-abs(weather.temperature-t_mid)/t_range) if t_range>0 else (1.0 if t_min<=weather.temperature<=t_max else 0.0)
        score+=t_score*0.35
        h_thresh=params["humidity_min"]
        h_score=min((weather.humidity-h_thresh)/(100-h_thresh),1.0) if weather.humidity>=h_thresh else max(0,weather.humidity/h_thresh-0.3)
        score+=h_score*0.40
        rs=params["rain_sensitivity"]
        r_score=min(weather.rainfall_3d/30.0,1.0)*rs if rs>0 else max(0,1-weather.rainfall_3d/10.0)*abs(rs)
        score+=r_score*0.25
        return min(score,1.0)

    def _compute_outbreak_signal(self, outbreak, disease_key):
        if outbreak.nearby_cases==0: return 0.0
        df=math.exp(-0.5*(1.0/max(outbreak.radius_km,1.0)))
        cf=min(math.log(outbreak.nearby_cases+1)/math.log(50),1.0)
        rf=max(0,1-(outbreak.days_window-7)/30.0) if outbreak.days_window>7 else 1.0
        return min(df*cf*rf,1.0)

    def _explain_factors(self, weather, params, crop, outbreak):
        f=[]
        if weather.humidity>=params["humidity_min"]: f.append(f"High humidity ({weather.humidity:.0f}%) — ideal for pathogen spread")
        if weather.rainfall_3d>10 and params["rain_sensitivity"]>0: f.append(f"Recent rainfall ({weather.rainfall_3d:.0f}mm/3d)")
        if params["temp_min"]<=weather.temperature<=params["temp_max"]: f.append(f"Temperature in optimal range ({weather.temperature:.0f}°C)")
        if crop.growth_stage.lower() in ["seedling","flowering"]: f.append(f"Crop at vulnerable {crop.growth_stage} stage")
        if outbreak and outbreak.nearby_cases>0: f.append(f"{outbreak.nearby_cases} cases within {outbreak.radius_km:.0f}km")
        return f or ["Low environmental stress — conditions not ideal for disease"]

    def _timeseries_factors(self, today, params, trend, crop, outbreak, plant_memory, ob_src):
        f=[]
        hs=trend.get("humidity_slope")
        if hs is not None:
            if hs>0.5: f.append(f"Humidity rising {hs:.1f}%/day over 7 days — approaching threshold")
            elif hs<-0.5: f.append(f"Humidity falling {abs(hs):.1f}%/day — reducing risk")
        sh=trend.get("sustained_high_humidity",0)
        if sh>0.6: f.append(f"High humidity sustained {sh:.0%} of past 5 days")
        fh=trend.get("forecast_avg_humidity")
        if fh and fh>=params["humidity_min"]: f.append(f"3-day forecast: {fh:.0f}% humidity — favorable for spread")
        fp=trend.get("forecast_rain_3d_mm")
        if fp and fp>10: f.append(f"Rain forecast: {fp:.0f}mm in 3 days")
        if today.humidity>=params["humidity_min"]: f.append(f"Current humidity {today.humidity:.0f}% above {params['humidity_min']}% threshold")
        if params["temp_min"]<=today.temperature<=params["temp_max"]: f.append(f"Temperature {today.temperature:.0f}°C in optimal disease range")
        if crop.growth_stage.lower() in ["seedling","flowering"]: f.append(f"Crop at vulnerable {crop.growth_stage} stage")
        if outbreak and outbreak.nearby_cases>0 and ob_src=="real_geo_temporal_data": f.append(f"{outbreak.nearby_cases} confirmed cases within {outbreak.radius_km:.0f}km")
        elif ob_src=="environmental_proxy_low_confidence": f.append("Outbreak risk estimated from weather (no local scan data — low confidence)")
        if plant_memory and plant_memory.scan_count>=2:
            if plant_memory.severity_trend=="declining": f.append(f"Plant health declining over {plant_memory.scan_count} scans")
            if plant_memory.recent_diseases: f.append(f"Previous infections: {', '.join(plant_memory.recent_diseases[:2])}")
        return f or ["Low environmental stress — conditions not ideal for disease"]


risk_engine = PredictiveRiskEngine()
