"""Nutritional anomaly detection using Isolation Forest."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

NUTRIENT_REFERENCE_RANGES: dict[str, tuple[float, float, str]] = {
    "energy": (50.0, 900.0, "kcal"),
    "protein": (0.0, 40.0, "g"),
    "fat": (0.0, 50.0, "g"),
    "saturated_fat": (0.0, 25.0, "g"),
    "carbohydrates": (0.0, 80.0, "g"),
    "sugar": (0.0, 50.0, "g"),
    "fiber": (0.0, 20.0, "g"),
    "sodium": (0.0, 2000.0, "mg"),
    "salt": (0.0, 5.0, "g"),
    "calcium": (0.0, 1200.0, "mg"),
    "iron": (0.0, 20.0, "mg"),
    "vitamin_a": (0.0, 3000.0, "ug"),
    "vitamin_c": (0.0, 500.0, "mg"),
    "vitamin_d": (0.0, 100.0, "ug"),
    "vitamin_e": (0.0, 50.0, "mg"),
    "vitamin_b12": (0.0, 100.0, "ug"),
    "potassium": (0.0, 3500.0, "mg"),
    "magnesium": (0.0, 500.0, "mg"),
    "zinc": (0.0, 30.0, "mg"),
    "phosphorus": (0.0, 1500.0, "mg"),
}

_IF_FEATURES = ["energy", "protein", "fat", "carbohydrates", "sugar", "fiber", "sodium"]


class NutritionalAnomalyDetector(BaseModel):
    """Detects anomalies in nutrient test results.

    Uses an Isolation Forest model on standardized nutrient profiles, with
    reference-range fallback when the IF artifact is unavailable.
    """

    def __init__(self) -> None:
        super().__init__()
        self._if_model: Any = None
        self._scaler_params: dict[str, Any] | None = None
        self._version: str = "0.1.0"

    @property
    def model_id(self) -> str:
        return "food.nutritional_anomaly"

    def load(self, artifact_path: Path) -> None:
        model_file = artifact_path / "nutritional_anomaly_if.joblib"
        scaler_file = artifact_path / "nutritional_anomaly_scaler.joblib"

        if model_file.exists():
            self._if_model = joblib.load(model_file)
            logger.info("Loaded Isolation Forest nutritional anomaly model from %s", model_file)
        else:
            logger.warning("No IF model at %s, using reference-range fallback", model_file)
            self._if_model = None

        if scaler_file.exists():
            self._scaler_params = joblib.load(scaler_file)
        else:
            self._scaler_params = None

        self._loaded = True

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        test_results: list[dict[str, Any]] = input_data.get("test_results", [])
        if not test_results:
            return {"anomalies": []}

        anomalies: list[dict[str, Any]] = []

        nutrient_map: dict[str, tuple[float, str]] = {}
        for result in test_results:
            name = result.get("nutrient", "").lower().strip()
            value = float(result.get("value", 0.0))
            unit = result.get("unit", "")
            nutrient_map[name] = (value, unit)

        for name, (value, unit) in nutrient_map.items():
            ref = NUTRIENT_REFERENCE_RANGES.get(name)
            if ref is None:
                continue
            low, high, _ref_unit = ref
            if value < low or value > high:
                distance = max(0.0, low - value, value - high)
                range_span = high - low if high > low else 1.0
                anomaly_score = min(1.0, distance / range_span)
                anomalies.append({
                    "nutrient": name,
                    "value": value,
                    "expected_range": (low, high),
                    "anomaly_score": round(anomaly_score, 4),
                })

        if self._if_model is not None:
            feature_vec = np.zeros((1, len(_IF_FEATURES)))
            for idx, feature_name in enumerate(_IF_FEATURES):
                if feature_name in nutrient_map:
                    feature_vec[0, idx] = nutrient_map[feature_name][0]

            if self._scaler_params is not None:
                means = self._scaler_params.get("means", np.zeros(len(_IF_FEATURES)))
                stds = self._scaler_params.get("stds", np.ones(len(_IF_FEATURES)))
                stds[stds == 0] = 1.0
                feature_vec = (feature_vec - means) / stds

            raw_score = self._if_model.decision_function(feature_vec)[0]
            if raw_score < -0.1:
                for entry in anomalies:
                    entry["anomaly_score"] = min(1.0, entry["anomaly_score"] + 0.2)

                for feature_name in _IF_FEATURES:
                    already_flagged = any(entry["nutrient"] == feature_name for entry in anomalies)
                    if not already_flagged and feature_name in nutrient_map:
                        ref = NUTRIENT_REFERENCE_RANGES.get(feature_name)
                        if ref:
                            anomalies.append({
                                "nutrient": feature_name,
                                "value": nutrient_map[feature_name][0],
                                "expected_range": (ref[0], ref[1]),
                                "anomaly_score": round(float(max(0.0, -raw_score)), 4),
                            })

        return {"anomalies": anomalies}

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self._version,
            "description": "Nutritional anomaly detector (Isolation Forest + reference ranges)",
            "reference_nutrients": list(NUTRIENT_REFERENCE_RANGES.keys()),
            "backend": "isolation_forest" if self._if_model is not None else "reference_ranges",
        }
