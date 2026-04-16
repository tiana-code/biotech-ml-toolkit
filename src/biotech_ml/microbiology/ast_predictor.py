"""AST predictor - XGBoost classifiers per organism-antibiotic pair, predicts R/S/I."""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

COMMON_ORGANISMS: list[str] = [
    "escherichia_coli",
    "staphylococcus_aureus",
    "klebsiella_pneumoniae",
    "pseudomonas_aeruginosa",
    "enterococcus_faecalis",
    "streptococcus_pneumoniae",
    "acinetobacter_baumannii",
    "proteus_mirabilis",
]

COMMON_ANTIBIOTICS: list[str] = [
    "amoxicillin",
    "ampicillin",
    "ciprofloxacin",
    "gentamicin",
    "meropenem",
    "vancomycin",
    "ceftriaxone",
    "trimethoprim_sulfamethoxazole",
]

_RSI_LABELS: list[str] = ["R", "S", "I"]

_BREAKPOINT_SOURCES: dict[tuple[str, str], str] = {
    ("escherichia_coli", "ciprofloxacin"): "EUCAST",
    ("staphylococcus_aureus", "vancomycin"): "EUCAST",
    ("klebsiella_pneumoniae", "meropenem"): "EUCAST",
    ("pseudomonas_aeruginosa", "meropenem"): "EUCAST",
    ("enterococcus_faecalis", "vancomycin"): "CLSI",
    ("streptococcus_pneumoniae", "ceftriaxone"): "CLSI",
    ("acinetobacter_baumannii", "meropenem"): "EUCAST",
    ("proteus_mirabilis", "ampicillin"): "CLSI",
}


class ASTPredictor(BaseModel):
    """XGBoost-based AST predictor (R/S/I) per organism-antibiotic pair."""

    def __init__(self) -> None:
        super().__init__()
        self._pair_models: dict[str, Any] = {}
        self._general_model: Any = None
        self._organism_encoder: dict[str, int] = {}
        self._antibiotic_encoder: dict[str, int] = {}
        self._version: str = "0.1.0"

    @property
    def model_id(self) -> str:
        return "microbiology.ast_predictor"

    def load(self, artifact_path: Path) -> None:
        try:
            data = joblib.load(artifact_path / "ast_predictor.joblib")
            self._pair_models = data.get("pair_models", {})
            self._general_model = data["general_model"]
            self._organism_encoder = data["organism_encoder"]
            self._antibiotic_encoder = data["antibiotic_encoder"]
            self._version = data.get("version", self._version)
            self._loaded = True
            logger.info(
                "Loaded AST predictor: %d pair models, general model ready",
                len(self._pair_models),
            )
        except Exception:
            logger.exception("Failed to load AST predictor from %s", artifact_path)
            raise

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        organism_id: str = input_data["organism_id"]
        antibiotic_id: str = input_data["antibiotic_id"]
        extra_features: dict[str, Any] | None = input_data.get("features")

        pair_key = f"{organism_id}__{antibiotic_id}"
        features = self._encode_features(organism_id, antibiotic_id, extra_features)

        if self._organism_encoder.get(organism_id) is None:
            logger.warning("Unknown organism_id '%s', encoded as -1", organism_id)
        if self._antibiotic_encoder.get(antibiotic_id) is None:
            logger.warning("Unknown antibiotic_id '%s', encoded as -1", antibiotic_id)

        model = self._pair_models.get(pair_key, self._general_model)
        if model is None:
            raise RuntimeError("No trained model available for prediction")

        probas = model.predict_proba(features)
        pred_idx = int(np.argmax(probas, axis=1)[0])
        confidence = float(probas[0, pred_idx])
        prediction = _RSI_LABELS[pred_idx]

        breakpoint_source = _BREAKPOINT_SOURCES.get(
            (organism_id, antibiotic_id), "EUCAST"
        )

        return {
            "prediction": prediction,
            "model_probability": round(confidence, 4),
            "score_type": "raw_model_probability",
            "breakpoint_source": breakpoint_source,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self._version,
            "pair_models_count": len(self._pair_models),
            "organisms": COMMON_ORGANISMS,
            "antibiotics": COMMON_ANTIBIOTICS,
        }

    def _encode_features(
        self,
        organism_id: str,
        antibiotic_id: str,
        extra: dict[str, Any] | None,
    ) -> np.ndarray:
        org_idx = self._organism_encoder.get(organism_id, -1)
        abx_idx = self._antibiotic_encoder.get(antibiotic_id, -1)
        base = [org_idx, abx_idx]

        if extra:
            for key in sorted(extra.keys()):
                base.append(float(extra[key]))

        return np.array([base], dtype=np.float32)
