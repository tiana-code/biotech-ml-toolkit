"""Clinical QA - retrieval-based question answering using BM25."""

import logging
from pathlib import Path
from typing import Any

from biotech_ml.features.text import BM25Index
from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

_MODEL_VERSION = "0.1.0"


class ClinicalQA(BaseModel):
    """Retrieval-based clinical Q&A using BM25 index over a knowledge corpus."""

    def __init__(self) -> None:
        super().__init__()
        self._index: BM25Index | None = None

    @property
    def model_id(self) -> str:
        return "medical.clinical_qa"

    def load(self, artifact_path: Path) -> None:
        index_file = artifact_path / "bm25_index.joblib"
        if not index_file.exists():
            logger.warning(
                "No BM25 index at %s - model will return empty results until trained",
                index_file,
            )
            self._index = BM25Index()
            self._loaded = True
            return

        self._index = BM25Index()
        self._index.load(index_file)
        self._loaded = True
        logger.info(
            "Loaded clinical QA BM25 index from %s (%d documents)",
            index_file,
            self._index.document_count,
        )

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        question = input_data.get("question", "")
        top_k = input_data.get("top_k", 5)

        if self._index is None or self._index.document_count == 0:
            return {"answers": [], "status": "no_index"}

        raw_results = self._index.search(question, top_k=top_k)

        answers = []
        for result in raw_results:
            answers.append(
                {
                    "text": result["text"],
                    "score": round(result["score"], 4),
                    "source": result.get("metadata", {}).get("source", "knowledge_base"),
                }
            )

        return {"answers": answers, "status": "ok", "score_type": "bm25_relevance"}

    def metadata(self) -> dict[str, Any]:
        corpus_size = self._index.document_count if self._index else 0
        return {
            "model_id": self.model_id,
            "version": _MODEL_VERSION,
            "description": "BM25-based clinical question answering",
            "corpus_size": corpus_size,
        }
