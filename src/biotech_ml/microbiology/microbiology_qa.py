"""Microbiology QA using BM25 retrieval over a pre-built corpus."""

import logging
from pathlib import Path
from typing import Any

import joblib

from biotech_ml.features.text import BM25Index
from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

QA_CATEGORIES: list[str] = [
    "antimicrobial_therapy",
    "organism_identification",
    "laboratory_methods",
    "resistance_mechanisms",
    "infection_control",
]

DEFAULT_TOP_K: int = 5


class MicrobiologyQA(BaseModel):
    """BM25-based microbiology question-answering model."""

    def __init__(self) -> None:
        super().__init__()
        self._index: BM25Index = BM25Index()
        self._version: str = "0.1.0"

    @property
    def model_id(self) -> str:
        return "microbiology.microbiology_qa"

    def load(self, artifact_path: Path) -> None:
        try:
            index_path = artifact_path / "microbiology_qa_index.joblib"
            self._index.load(index_path)

            meta_path = artifact_path / "microbiology_qa_meta.joblib"
            if meta_path.exists():
                meta = joblib.load(meta_path)
                self._version = meta.get("version", self._version)

            self._loaded = True
            logger.info("Loaded microbiology QA index with %d documents", self._index.document_count)
        except Exception:
            logger.exception("Failed to load microbiology QA from %s", artifact_path)
            raise

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        question: str = input_data["question"]
        top_k: int = input_data.get("top_k", DEFAULT_TOP_K)

        results = self._index.search(question, top_k=top_k)

        answers: list[dict[str, Any]] = []
        for result in results:
            meta = result.get("metadata", {})
            answers.append({
                "text": result["text"],
                "score": round(result["score"], 4),
                "source": meta.get("source", "microbiology_corpus"),
                "category": meta.get("category"),
            })

        return {"answers": answers}

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self._version,
            "document_count": self._index.document_count,
            "categories": QA_CATEGORIES,
        }
