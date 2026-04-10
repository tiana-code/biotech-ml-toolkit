"""Phenotype predictor from genomic features.

Uses XGBoost multi-label classifier to predict phenotypic traits
(resistance markers, virulence, metabolism) from genomic feature vectors.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

PHENOTYPE_TRAITS: list[str] = [
    "motility",
    "biofilm_formation",
    "hemolysis",
    "catalase",
    "oxidase",
    "coagulase",
    "beta_lactamase",
    "esbl",
    "carbapenemase",
    "methicillin_resistance",
    "vancomycin_resistance",
]

TRAIT_CATEGORIES: dict[str, str] = {
    "motility": "virulence",
    "biofilm_formation": "virulence",
    "hemolysis": "virulence",
    "catalase": "metabolism",
    "oxidase": "metabolism",
    "coagulase": "virulence",
    "beta_lactamase": "resistance",
    "esbl": "resistance",
    "carbapenemase": "resistance",
    "methicillin_resistance": "resistance",
    "vancomycin_resistance": "resistance",
}

PROBABILITY_THRESHOLD: float = 0.3


class PhenotypePredictor(BaseModel):
    """XGBoost multi-label classifier for phenotypic trait prediction."""

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._trait_names: list[str] = []
        self._feature_names: list[str] = []
        self._version: str = "0.1.0"

    @property
    def model_id(self) -> str:
        return "microbiology.phenotype_predictor"

    def load(self, artifact_path: Path) -> None:
        try:
            data = joblib.load(artifact_path / "phenotype_predictor.joblib")
            self._model = data["model"]
            self._trait_names = data["trait_names"]
            self._feature_names = data["feature_names"]
            self._version = data.get("version", self._version)
            self._loaded = True
            logger.info(
                "Loaded phenotype predictor: %d traits, %d features",
                len(self._trait_names),
                len(self._feature_names),
            )
        except Exception:
            logger.exception("Failed to load phenotype predictor from %s", artifact_path)
            raise

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        genomic_features: dict[str, Any] = input_data["genomic_features"]
        features = self._encode_features(genomic_features)
        probas = self._model.predict_proba(features)

        if isinstance(probas, list):
            proba_vector = np.array([arr[0, 1] if arr.ndim == 2 else float(arr[0]) for arr in probas])
        else:
            proba_vector = probas[0] if probas.ndim == 2 else probas

        traits: list[dict[str, Any]] = []
        for i, trait_name in enumerate(self._trait_names):
            prob = float(proba_vector[i]) if i < len(proba_vector) else 0.0
            if prob >= PROBABILITY_THRESHOLD:
                traits.append({
                    "name": trait_name,
                    "probability": round(prob, 4),
                    "category": TRAIT_CATEGORIES.get(trait_name, "unknown"),
                })

        traits.sort(key=lambda t: t["probability"], reverse=True)

        return {"traits": traits}

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self._version,
            "trait_names": self._trait_names,
            "feature_count": len(self._feature_names),
            "threshold": PROBABILITY_THRESHOLD,
        }

    def _encode_features(self, genomic_features: dict[str, Any]) -> np.ndarray:
        vector = [
            float(genomic_features.get(fname, 0.0))
            for fname in self._feature_names
        ]
        return np.array([vector], dtype=np.float32)
