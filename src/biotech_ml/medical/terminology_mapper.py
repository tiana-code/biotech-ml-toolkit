"""Terminology Mapper - maps local lab names to SNOMED/LOINC codes via TF-IDF cosine similarity."""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

_MODEL_VERSION = "0.1.0"


class TerminologyMapper(BaseModel):
    """Maps local/proprietary lab terms to standard SNOMED CT or LOINC codes."""

    def __init__(self) -> None:
        super().__init__()
        self._vectorizer: Any = None
        self._tfidf_matrix: Any = None
        self._terminology: list[dict[str, str]] = []

    @property
    def model_id(self) -> str:
        return "medical.terminology_mapper"

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
        self._vectorizer = data["vectorizer"]
        self._tfidf_matrix = data["tfidf_matrix"]
        self._terminology = data["terminology"]
        self._loaded = True
        logger.info(
            "Loaded terminology mapper from %s (%d terms)",
            artifact_file,
            len(self._terminology),
        )

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        if self._vectorizer is None or self._tfidf_matrix is None:
            return {"mappings": []}

        local_name = input_data.get("local_name", "")
        target_system = input_data.get("target_system", "snomed").lower()

        query_vec = self._vectorizer.transform([local_name.lower()])
        similarities = cosine_similarity(query_vec, self._tfidf_matrix).ravel()

        top_indices = np.argsort(similarities)[::-1][:20]

        mappings = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < 0.05:
                break

            term = self._terminology[idx]
            if target_system != "all" and term.get("system", "").lower() != target_system:
                continue

            mappings.append(
                {
                    "code": term["code"],
                    "display": term["display"],
                    "system": term["system"],
                    "similarity_score": round(score, 4),
                }
            )

            if len(mappings) >= 10:
                break

        return {"mappings": mappings}

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": _MODEL_VERSION,
            "description": "TF-IDF cosine similarity terminology mapper (SNOMED/LOINC)",
            "terminology_size": len(self._terminology),
            "has_trained_model": self._vectorizer is not None,
        }
