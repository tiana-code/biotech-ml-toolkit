"""Nutri-Score predictor using XGBoost classifier."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

_GRADE_MAP: dict[int, str] = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
_GRADE_TO_INT: dict[str, int] = {grade: idx for idx, grade in _GRADE_MAP.items()}

_NEGATIVE_THRESHOLDS = {
    "energy_kcal": [335, 670, 1005, 1340, 1675, 2010, 2345, 2680, 3015, 3350],
    "sugar": [4.5, 9, 13.5, 18, 22.5, 27, 31, 36, 40, 45],
    "saturated_fat": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "salt": [0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1, 9.0],
}

_POSITIVE_THRESHOLDS = {
    "fiber": [0.9, 1.9, 2.8, 3.7, 4.7],
    "protein": [1.6, 3.2, 4.8, 6.4, 8.0],
    "fruits_vegetables_percent": [40, 60, 80, 80, 80],
}

_FEATURE_NAMES = ["energy_kcal", "fat", "saturated_fat", "sugar", "salt", "protein", "fiber"]


def _count_points(value: float, thresholds: list[float]) -> int:
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            return i
    return len(thresholds)


def _compute_nutriscore_points(nutrients: dict[str, float]) -> dict[str, Any]:
    negative = 0
    for key in ("energy_kcal", "sugar", "saturated_fat", "salt"):
        pts = _count_points(nutrients.get(key, 0.0), _NEGATIVE_THRESHOLDS[key])
        negative += pts

    positive = 0
    for key in ("fiber", "protein"):
        pts = _count_points(nutrients.get(key, 0.0), _POSITIVE_THRESHOLDS[key])
        positive += pts

    fvp = nutrients.get("fruits_vegetables_percent", 0.0) or 0.0
    positive += _count_points(fvp, _POSITIVE_THRESHOLDS["fruits_vegetables_percent"])

    final_score = negative - positive
    return {"negative_points": negative, "positive_points": positive, "final_score": final_score}


def _score_to_grade(score: int) -> str:
    if score <= -1:
        return "A"
    if score <= 2:
        return "B"
    if score <= 10:
        return "C"
    if score <= 18:
        return "D"
    return "E"


class NutriScorePredictor(BaseModel):
    """Predicts Nutri-Score grade (A-E) using an XGBoost classifier.

    Falls back to deterministic rule-based scoring when model artifact is
    not available.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._version: str = "0.1.0"

    @property
    def model_id(self) -> str:
        return "food.nutriscore_predictor"

    def load(self, artifact_path: Path) -> None:
        model_file = artifact_path / "xgb_nutriscore.joblib"
        if model_file.exists():
            self._model = joblib.load(model_file)
            logger.info("Loaded XGBoost Nutri-Score model from %s", model_file)
        else:
            logger.warning("No XGBoost artifact at %s, using rule-based fallback", model_file)
            self._model = None
        self._loaded = True

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        nutrients = {feat: float(input_data.get(feat, 0.0)) for feat in _FEATURE_NAMES}
        nutrients["fruits_vegetables_percent"] = float(input_data.get("fruits_vegetables_percent", 0.0) or 0.0)

        details = _compute_nutriscore_points(nutrients)

        if self._model is not None:
            features = np.array([[nutrients[feat] for feat in _FEATURE_NAMES]])
            pred_class = int(self._model.predict(features)[0])
            grade = _GRADE_MAP.get(pred_class, "C")
            score = details["final_score"]
            return {
                "grade": grade,
                "score": score,
                "details": details,
                "score_type": "model_prediction",
            }
        else:
            score = details["final_score"]
            grade = _score_to_grade(score)
            return {
                "grade": grade,
                "score": score,
                "details": details,
                "score_type": "rule_based",
            }

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self._version,
            "description": "Nutri-Score A-E predictor (XGBoost / rule-based fallback)",
            "features": _FEATURE_NAMES,
        }
