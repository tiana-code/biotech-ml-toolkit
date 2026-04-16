"""Additive risk scorer using Isolation Forest."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

ADDITIVE_RISK_DB: dict[str, dict[str, str]] = {
    "E102": {"name": "Tartrazine", "risk": "medium", "reason": "Azo dye, linked to hyperactivity"},
    "E110": {"name": "Sunset Yellow", "risk": "medium", "reason": "Azo dye, potential allergen"},
    "E120": {"name": "Cochineal", "risk": "low", "reason": "Natural colorant, rare allergic reactions"},
    "E122": {"name": "Carmoisine", "risk": "medium", "reason": "Azo dye, linked to hyperactivity"},
    "E124": {"name": "Ponceau 4R", "risk": "medium", "reason": "Azo dye, banned in some countries"},
    "E129": {"name": "Allura Red", "risk": "medium", "reason": "Azo dye, linked to hyperactivity"},
    "E150d": {"name": "Caramel colour", "risk": "low", "reason": "Contains 4-MEI in some processes"},
    "E171": {"name": "Titanium dioxide", "risk": "high", "reason": "Banned in EU since 2022, nanoparticle concerns"},
    "E211": {"name": "Sodium benzoate", "risk": "medium", "reason": "Forms benzene with ascorbic acid"},
    "E249": {"name": "Potassium nitrite", "risk": "high", "reason": "Nitrosamine formation risk"},
    "E250": {"name": "Sodium nitrite", "risk": "high", "reason": "Nitrosamine formation risk"},
    "E320": {"name": "BHA", "risk": "high", "reason": "Possible carcinogen (IARC Group 2B)"},
    "E321": {"name": "BHT", "risk": "medium", "reason": "Endocrine disruptor concerns"},
    "E330": {"name": "Citric acid", "risk": "safe", "reason": "Generally recognized as safe"},
    "E338": {"name": "Phosphoric acid", "risk": "low", "reason": "Bone health concerns at high intake"},
    "E412": {"name": "Guar gum", "risk": "safe", "reason": "Natural thickener, generally safe"},
    "E415": {"name": "Xanthan gum", "risk": "safe", "reason": "Generally recognized as safe"},
    "E420": {"name": "Sorbitol", "risk": "low", "reason": "Laxative effect at high doses"},
    "E450": {"name": "Diphosphates", "risk": "low", "reason": "Phosphate intake concerns"},
    "E451": {"name": "Triphosphates", "risk": "low", "reason": "Phosphate intake concerns"},
    "E466": {"name": "Carboxymethyl cellulose", "risk": "low", "reason": "Gut microbiome concerns"},
    "E471": {"name": "Mono/diglycerides", "risk": "safe", "reason": "Generally recognized as safe"},
    "E500": {"name": "Sodium carbonates", "risk": "safe", "reason": "Baking soda, safe"},
    "E621": {"name": "MSG", "risk": "low", "reason": "Generally safe, sensitivity in some individuals"},
    "E951": {"name": "Aspartame", "risk": "medium", "reason": "IARC Group 2B (possibly carcinogenic)"},
    "E955": {"name": "Sucralose", "risk": "low", "reason": "Generally safe, gut microbiome concerns"},
}

_RISK_TO_SCORE: dict[str, float] = {
    "safe": 0.0,
    "low": 0.25,
    "medium": 0.5,
    "high": 0.85,
}


class AdditiveRiskScorer(BaseModel):
    """Scores additive combinations for anomaly / risk using Isolation Forest."""

    def __init__(self) -> None:
        super().__init__()
        self._if_model: Any = None
        self._known_additives: list[str] = []
        self._version: str = "0.1.0"

    @property
    def model_id(self) -> str:
        return "food.additive_risk"

    def load(self, artifact_path: Path) -> None:
        model_file = artifact_path / "additive_risk_if.joblib"
        additives_file = artifact_path / "known_additives.joblib"

        if model_file.exists() and additives_file.exists():
            self._if_model = joblib.load(model_file)
            self._known_additives = joblib.load(additives_file)
            logger.info("Loaded Isolation Forest additive risk model from %s", model_file)
        else:
            logger.warning("No IF model at %s, using knowledge-base fallback", model_file)
            self._if_model = None
            self._known_additives = sorted(ADDITIVE_RISK_DB.keys())

        self._loaded = True

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        additives: list[str] = input_data.get("additives", [])
        if not additives:
            return {"risk_score": 0.0, "anomalies": []}

        normalized = [additive.strip().upper() for additive in additives]

        anomalies: list[dict[str, Any]] = []
        kb_scores: list[float] = []

        for additive in normalized:
            info = ADDITIVE_RISK_DB.get(additive)
            if info is not None:
                risk_level = info["risk"]
                reason = info["reason"]
                score = _RISK_TO_SCORE.get(risk_level, 0.3)
            else:
                risk_level = "unknown"
                reason = "Additive not in knowledge base"
                score = 0.3

            kb_scores.append(score)
            if risk_level not in ("safe",):
                anomalies.append({
                    "additive": additive,
                    "risk_level": risk_level,
                    "reason": reason,
                })

        if_anomaly_score = 0.0
        if self._if_model is not None and self._known_additives:
            feature_vec = np.zeros((1, len(self._known_additives)))
            for additive in normalized:
                if additive in self._known_additives:
                    idx = self._known_additives.index(additive)
                    feature_vec[0, idx] = 1.0
            raw_score = self._if_model.decision_function(feature_vec)[0]
            if_anomaly_score = float(max(0.0, min(1.0, 0.5 - raw_score)))

        kb_avg = float(np.mean(kb_scores)) if kb_scores else 0.0
        risk_score = max(kb_avg, if_anomaly_score)
        risk_score = round(min(1.0, risk_score), 4)

        return {
            "risk_score": risk_score,
            "anomalies": anomalies,
            "score_type": "heuristic_composite",
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self._version,
            "description": "Additive risk scoring (Isolation Forest + knowledge base)",
            "known_additives_count": len(self._known_additives),
            "backend": "isolation_forest" if self._if_model is not None else "knowledge_base",
        }
