"""Differential Diagnosis Suggester - XGBoost multiclass classifier."""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

_MODEL_VERSION = "0.1.0"


class DDxSuggester(BaseModel):
    """Suggests differential diagnoses from symptoms + lab results using XGBoost."""

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._label_encoder: Any = None
        self._feature_names: list[str] = []
        self._icd_map: dict[str, str] = {}

    @property
    def model_id(self) -> str:
        return "medical.ddx_suggester"

    def load(self, artifact_path: Path) -> None:
        artifact_file = artifact_path / "model.joblib"
        if not artifact_file.exists():
            logger.warning(
                "No trained artifact at %s - model will return empty results until trained",
                artifact_file,
            )
            self._loaded = True
            return

        data = joblib.load(artifact_file)
        self._model = data["model"]
        self._label_encoder = data["label_encoder"]
        self._feature_names = data["feature_names"]
        self._icd_map = data.get("icd_map", {})
        self._loaded = True
        logger.info(
            "Loaded DDx model from %s (%d features, %d diagnoses)",
            artifact_file,
            len(self._feature_names),
            len(self._label_encoder.classes_) if self._label_encoder else 0,
        )

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        if self._model is None:
            return {"diagnoses": [], "status": "model_not_trained", "score_type": "none"}

        symptoms: list[str] = input_data.get("symptoms", [])
        lab_results: list[dict] = input_data.get("lab_results", [])

        feature_vec = self._encode_features(symptoms, lab_results)
        probas = self._model.predict_proba(feature_vec.reshape(1, -1))[0]

        top_indices = np.argsort(probas)[::-1][:10]
        diagnoses = []
        for idx in top_indices:
            prob = float(probas[idx])
            if prob < 0.01:
                continue
            name = self._label_encoder.inverse_transform([idx])[0]
            diagnoses.append(
                {
                    "name": name,
                    "model_probability": round(prob, 4),
                    "icd_code": self._icd_map.get(name),
                }
            )

        unknown_symptoms = [symptom for symptom in symptoms if f"sym_{symptom.lower().replace(' ', '_')}" not in set(self._feature_names)]
        unknown_tests = [lab_result.get("test_code", "") for lab_result in lab_results if f"lab_{lab_result.get('test_code', '').upper()}" not in set(self._feature_names)]

        return {
            "diagnoses": diagnoses,
            "status": "ok",
            "score_type": "raw_model_probability",
            "unknown_symptoms": unknown_symptoms,
            "unknown_tests": unknown_tests,
        }

    def metadata(self) -> dict[str, Any]:
        n_classes = len(self._label_encoder.classes_) if self._label_encoder else 0
        return {
            "model_id": self.model_id,
            "version": _MODEL_VERSION,
            "description": "XGBoost multiclass DDx from symptoms + lab results",
            "n_features": len(self._feature_names),
            "n_diagnoses": n_classes,
            "has_trained_model": self._model is not None,
        }

    def _encode_features(self, symptoms: list[str], lab_results: list[dict]) -> np.ndarray:
        feature_map: dict[str, float] = {}

        for symptom in symptoms:
            key = f"sym_{symptom.lower().replace(' ', '_')}"
            feature_map[key] = 1.0

        for lab_result in lab_results:
            code = lab_result.get("test_code", "").upper()
            value = float(lab_result.get("value", 0.0))
            feature_map[f"lab_{code}"] = value

        vec = np.zeros(len(self._feature_names))
        for idx, feature_name in enumerate(self._feature_names):
            if feature_name in feature_map:
                vec[idx] = feature_map[feature_name]

        return vec
