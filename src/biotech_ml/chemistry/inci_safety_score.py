"""INCI Safety Scorer - Ensemble combining toxicity + allergen detection + regulatory rules."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

from biotech_ml.base import BaseModel
from biotech_ml.chemistry.allergen_detector import CosmeticAllergenDetector
from biotech_ml.chemistry.toxicity_scorer import ToxicityScorer

logger = logging.getLogger(__name__)

REGULATORY_DB_DEFAULT: dict[str, dict[str, Any]] = {
    "WATER": {"status": "approved", "max_concentration": 100.0, "function": "solvent", "notes": ""},
    "GLYCERIN": {"status": "approved", "max_concentration": 100.0, "function": "humectant", "notes": ""},
    "DIMETHICONE": {"status": "approved", "max_concentration": 100.0, "function": "emollient", "notes": ""},
    "CETEARYL ALCOHOL": {"status": "approved", "max_concentration": 100.0, "function": "emollient", "notes": ""},
    "SODIUM LAURYL SULFATE": {"status": "restricted", "max_concentration": 1.0, "function": "surfactant",
                               "notes": "Known irritant at high concentrations"},
    "SODIUM LAURETH SULFATE": {"status": "approved", "max_concentration": 50.0, "function": "surfactant", "notes": ""},
    "TOCOPHERYL ACETATE": {"status": "approved", "max_concentration": 100.0, "function": "antioxidant", "notes": ""},
    "RETINOL": {"status": "restricted", "max_concentration": 0.3, "function": "anti-aging",
                "notes": "Restricted in leave-on products"},
    "HYDROQUINONE": {"status": "banned", "max_concentration": 0.0, "function": "skin lightening",
                     "notes": "Banned in EU cosmetics (Annex II)"},
    "FORMALDEHYDE": {"status": "banned", "max_concentration": 0.0, "function": "preservative",
                     "notes": "Banned as cosmetic ingredient in EU"},
    "TRICLOSAN": {"status": "restricted", "max_concentration": 0.3, "function": "preservative",
                  "notes": "Max 0.3% in toothpaste only"},
    "METHYLISOTHIAZOLINONE": {"status": "restricted", "max_concentration": 0.0,
                               "function": "preservative", "notes": "Banned in leave-on products"},
    "PARABENS": {"status": "restricted", "max_concentration": 0.4, "function": "preservative",
                 "notes": "Max 0.4% individual, 0.8% mixtures"},
    "METHYLPARABEN": {"status": "restricted", "max_concentration": 0.4, "function": "preservative",
                      "notes": "Max 0.4%"},
    "PROPYLPARABEN": {"status": "restricted", "max_concentration": 0.14, "function": "preservative",
                      "notes": "Max 0.14%"},
    "BUTYLPARABEN": {"status": "restricted", "max_concentration": 0.14, "function": "preservative",
                     "notes": "Max 0.14%"},
    "PHENOXYETHANOL": {"status": "approved", "max_concentration": 1.0, "function": "preservative",
                       "notes": "Max 1.0%"},
    "SALICYLIC ACID": {"status": "restricted", "max_concentration": 2.0, "function": "exfoliant",
                       "notes": "Max 2.0% (3.0% in shampoo)"},
    "BENZOYL PEROXIDE": {"status": "restricted", "max_concentration": 5.0, "function": "anti-acne",
                         "notes": "OTC drug in US, restricted in EU cosmetics"},
    "COAL TAR": {"status": "restricted", "max_concentration": 5.0, "function": "anti-dandruff",
                 "notes": "Restricted; banned in EU cosmetics"},
    "LEAD ACETATE": {"status": "banned", "max_concentration": 0.0, "function": "colorant",
                     "notes": "Banned in cosmetics"},
    "MERCURY": {"status": "banned", "max_concentration": 0.0, "function": "preservative",
                "notes": "Banned in cosmetics (trace <1ppm tolerated)"},
    "NIACINAMIDE": {"status": "approved", "max_concentration": 100.0, "function": "skin conditioning", "notes": ""},
    "HYALURONIC ACID": {"status": "approved", "max_concentration": 100.0, "function": "humectant", "notes": ""},
    "PANTHENOL": {"status": "approved", "max_concentration": 100.0, "function": "hair conditioning", "notes": ""},
    "ALLANTOIN": {"status": "approved", "max_concentration": 100.0, "function": "skin protectant", "notes": ""},
    "ALOE BARBADENSIS LEAF JUICE": {"status": "approved", "max_concentration": 100.0,
                                     "function": "skin conditioning", "notes": ""},
    "TITANIUM DIOXIDE": {"status": "restricted", "max_concentration": 25.0, "function": "UV filter/colorant",
                         "notes": "Max 25% as UV filter; CI 77891"},
    "ZINC OXIDE": {"status": "approved", "max_concentration": 25.0, "function": "UV filter",
                   "notes": "Max 25% as UV filter"},
}


def _allergen_risk_level(allergen_count: int) -> Literal["none", "low", "moderate", "high"]:
    if allergen_count == 0:
        return "none"
    if allergen_count <= 2:
        return "low"
    if allergen_count <= 5:
        return "moderate"
    return "high"


def _allergen_risk_penalty(risk: str) -> float:
    return {"none": 0.0, "low": 0.5, "moderate": 1.5, "high": 3.0}.get(risk, 0.0)


class INCISafetyScorer(BaseModel):
    """Composite safety scorer: combines toxicity + allergen detection + regulatory database."""

    def __init__(self) -> None:
        super().__init__()
        self._toxicity_scorer: ToxicityScorer | None = None
        self._allergen_detector: CosmeticAllergenDetector | None = None
        self._regulatory_db: dict[str, dict[str, Any]] = {}
        self._version = "1.0.0"

    @property
    def model_id(self) -> str:
        return "chemistry.inci_safety_scorer"

    def load(self, artifact_path: Path) -> None:
        logger.info("Loading INCISafetyScorer from %s", artifact_path)

        self._toxicity_scorer = ToxicityScorer()
        tox_path = artifact_path.parent / "chemistry.toxicity_scorer"
        if tox_path.exists():
            self._toxicity_scorer.load(tox_path)
        else:
            logger.warning("Toxicity scorer artifacts not found at %s; sub-model not loaded", tox_path)

        self._allergen_detector = CosmeticAllergenDetector()
        allergen_path = artifact_path.parent / "chemistry.allergen_detector"
        if allergen_path.exists():
            self._allergen_detector.load(allergen_path)
        else:
            logger.warning("Allergen detector artifacts not found at %s; loading built-in DB", allergen_path)
            self._allergen_detector.load(artifact_path)

        reg_file = artifact_path / "regulatory_db.json"
        if reg_file.exists():
            with open(reg_file) as f:
                self._regulatory_db = json.load(f)
            logger.info("Loaded regulatory DB from file: %d entries", len(self._regulatory_db))
        else:
            self._regulatory_db = REGULATORY_DB_DEFAULT
            logger.info("Using built-in regulatory DB: %d entries", len(self._regulatory_db))

        self._loaded = True
        logger.info("INCISafetyScorer loaded")

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        start = time.perf_counter()

        inci_name: str = input_data["inci_name"]
        inci_upper = inci_name.strip().upper()

        tox_result = self._get_toxicity_score(inci_upper)
        tox_score = tox_result["score"]
        allergen_result = self._get_allergen_risk(inci_upper)
        reg_info = self._get_regulatory_info(inci_upper)

        allergen_risk = allergen_result["risk"]
        allergen_penalty = _allergen_risk_penalty(allergen_risk) if allergen_risk != "unknown" else 0.0

        base_score = 10.0 * (1.0 - tox_score)

        reg_status = reg_info.get("status", "not_in_database")
        if reg_status == "banned":
            base_score = 0.0
        elif reg_status == "restricted":
            base_score = min(base_score, 6.0)

        safety_score = max(0.0, min(10.0, base_score - allergen_penalty))
        latency = (time.perf_counter() - start) * 1000

        degradation_reasons: list[str] = []
        if tox_result.get("degraded"):
            degradation_reasons.append("toxicity_scorer_unavailable")
        if allergen_result.get("degraded"):
            degradation_reasons.append("allergen_detector_unavailable")
        if reg_info.get("status") == "not_in_database":
            degradation_reasons.append("ingredient_not_in_regulatory_db")

        status = "degraded" if degradation_reasons else "ok"

        return {
            "screening_index": round(safety_score, 2),
            "score_type": "heuristic_composite",
            "status": status,
            "details": {
                "toxicity_score": round(tox_score, 4),
                "regulatory_status": reg_status,
                "allergen_risk": allergen_risk,
                "function": reg_info.get("function"),
            },
            "meta": {
                "model_id": self.model_id,
                "version": self._version,
                "latency_ms": round(latency, 2),
                "degradation_reasons": degradation_reasons if degradation_reasons else None,
            },
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self._version,
            "regulatory_db_size": len(self._regulatory_db),
            "sub_models": {
                "toxicity": self._toxicity_scorer.model_id if self._toxicity_scorer else None,
                "allergen": self._allergen_detector.model_id if self._allergen_detector else None,
            },
            "regulatory_db_note": "Built-in regulatory DB is a demo/fallback layer. Always validate against official CosIng/SCCS sources.",
        }

    def _get_toxicity_score(self, inci_upper: str) -> dict[str, Any]:
        if self._toxicity_scorer is None or not self._toxicity_scorer.is_loaded:
            return {"score": 0.3, "degraded": True}

        try:
            result = self._toxicity_scorer.predict({"inci_name": inci_upper})
            return {"score": result.get("overall_score", 0.5), "degraded": False}
        except Exception:
            logger.warning("Toxicity prediction failed for '%s'", inci_upper)
            return {"score": 0.5, "degraded": True}

    def _get_allergen_risk(self, inci_upper: str) -> dict[str, Any]:
        if self._allergen_detector is None or not self._allergen_detector.is_loaded:
            return {"risk": "unknown", "count": 0, "degraded": True}

        try:
            result = self._allergen_detector.predict({"ingredients": [inci_upper]})
            count = len(result.get("allergens", []))
            return {"risk": _allergen_risk_level(count), "count": count, "degraded": False}
        except Exception:
            logger.warning("Allergen detection failed for '%s'", inci_upper)
            return {"risk": "unknown", "count": 0, "degraded": True}

    def _get_regulatory_info(self, inci_upper: str) -> dict[str, Any]:
        if inci_upper in self._regulatory_db:
            return self._regulatory_db[inci_upper]

        logger.debug("INCI name '%s' not found in regulatory database", inci_upper)
        return {"status": "not_in_database", "max_concentration": None, "function": None, "notes": "Not in database"}
