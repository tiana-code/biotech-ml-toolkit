"""HACCP text classifier using TF-IDF + LightGBM."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

HAZARD_CATEGORIES = [
    "biological",
    "chemical",
    "physical",
    "no_hazard_detected",
]

_BIOLOGICAL_KEYWORDS = [
    "salmonella", "listeria", "e.coli", "e. coli", "campylobacter", "clostridium",
    "staphylococcus", "norovirus", "hepatitis", "parasite", "mold", "mould",
    "yeast", "bacteria", "pathogen", "contamination", "spoilage", "ferment",
    "cross-contamination", "temperature abuse", "undercooking",
]

_CHEMICAL_KEYWORDS = [
    "pesticide", "herbicide", "fungicide", "allergen", "toxin", "mycotoxin",
    "aflatoxin", "heavy metal", "lead", "mercury", "cadmium", "arsenic",
    "cleaning agent", "sanitizer", "residue", "antibiotic", "hormone",
    "additive", "preservative", "sulfite", "nitrite", "nitrate",
]

_PHYSICAL_KEYWORDS = [
    "glass", "metal", "bone", "stone", "plastic", "wood", "foreign body",
    "foreign object", "fragment", "shard", "splinter", "hair", "insect",
    "rodent", "pest", "debris",
]


def _classify_by_keywords(text: str) -> tuple[bool, str, float]:
    text_lower = text.lower()

    scores: dict[str, int] = {"biological": 0, "chemical": 0, "physical": 0}

    for keyword in _BIOLOGICAL_KEYWORDS:
        if keyword in text_lower:
            scores["biological"] += 1

    for keyword in _CHEMICAL_KEYWORDS:
        if keyword in text_lower:
            scores["chemical"] += 1

    for keyword in _PHYSICAL_KEYWORDS:
        if keyword in text_lower:
            scores["physical"] += 1

    total_hits = sum(scores.values())
    if total_hits == 0:
        return False, "no_hazard_detected", 0.6

    best_category = max(scores, key=lambda k: scores[k])
    confidence = min(0.95, 0.5 + 0.1 * scores[best_category])
    return True, best_category, confidence


class HACCPClassifier(BaseModel):
    """Classifies text for HACCP hazard categories using TF-IDF + LightGBM.

    Falls back to keyword-based classification when model artifact
    is unavailable.
    """

    def __init__(self) -> None:
        super().__init__()
        self._vectorizer: Any = None
        self._classifier: Any = None
        self._version: str = "0.1.0"

    @property
    def model_id(self) -> str:
        return "food.haccp_classifier"

    def load(self, artifact_path: Path) -> None:
        vec_file = artifact_path / "haccp_tfidf.joblib"
        clf_file = artifact_path / "haccp_lgbm.joblib"

        if vec_file.exists() and clf_file.exists():
            self._vectorizer = joblib.load(vec_file)
            self._classifier = joblib.load(clf_file)
            logger.info("Loaded HACCP TF-IDF + LightGBM from %s", artifact_path)
        else:
            logger.warning("No HACCP model at %s, using keyword-based fallback", artifact_path)
            self._vectorizer = None
            self._classifier = None

        self._loaded = True

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        text: str = input_data.get("text", "")
        if not text.strip():
            return {"hazard_detected": False, "model_probability": 0.0, "category": "no_hazard_detected"}

        if self._vectorizer is not None and self._classifier is not None:
            return self._predict_model(text)
        return self._predict_keywords(text)

    def _predict_model(self, text: str) -> dict[str, Any]:
        features = self._vectorizer.transform([text])
        pred_class = int(self._classifier.predict(features)[0])
        proba = self._classifier.predict_proba(features)[0]
        confidence = float(np.max(proba))
        category = HAZARD_CATEGORIES[pred_class] if pred_class < len(HAZARD_CATEGORIES) else "no_hazard_detected"

        return {
            "hazard_detected": category != "no_hazard_detected",
            "model_probability": round(confidence, 4),
            "category": category,
        }

    def _predict_keywords(self, text: str) -> dict[str, Any]:
        hazard_detected, category, confidence = _classify_by_keywords(text)
        return {
            "hazard_detected": hazard_detected,
            "heuristic_score": round(confidence, 4),
            "category": category,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self._version,
            "description": "HACCP hazard text classifier (TF-IDF + LightGBM / keyword fallback)",
            "categories": HAZARD_CATEGORIES,
            "backend": "lightgbm" if self._classifier is not None else "keyword_rules",
        }
