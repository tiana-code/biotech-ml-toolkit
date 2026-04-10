"""Toxicity Scorer - XGBoost classifiers on Morgan fingerprints for Tox21 12 endpoints."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

TOX21_TARGETS: list[str] = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

TOX21_WEIGHTS: dict[str, float] = {
    "NR-AR": 0.08,
    "NR-AR-LBD": 0.08,
    "NR-AhR": 0.10,
    "NR-Aromatase": 0.08,
    "NR-ER": 0.09,
    "NR-ER-LBD": 0.08,
    "NR-PPAR-gamma": 0.07,
    "SR-ARE": 0.09,
    "SR-ATAD5": 0.08,
    "SR-HSE": 0.08,
    "SR-MMP": 0.09,
    "SR-p53": 0.08,
}


class ToxicityScorer(BaseModel):
    """Predicts compound toxicity across 12 Tox21 assay endpoints using XGBoost."""

    def __init__(self) -> None:
        super().__init__()
        self._models: dict[str, Any] = {}
        self._inci_to_smiles: dict[str, str] = {}
        self._version = "1.0.0"

    @property
    def model_id(self) -> str:
        return "chemistry.toxicity_scorer"

    def load(self, artifact_path: Path) -> None:
        import xgboost as xgb

        logger.info("Loading ToxicityScorer from %s", artifact_path)

        for target in TOX21_TARGETS:
            model_file = artifact_path / f"tox21_{target.lower().replace('-', '_')}.json"
            if model_file.exists():
                model = xgb.XGBClassifier()
                model.load_model(str(model_file))
                self._models[target] = model
                logger.debug("Loaded Tox21 model for %s", target)
            else:
                logger.warning("Model file missing for target %s: %s", target, model_file)

        lookup_file = artifact_path / "inci_to_smiles.json"
        if lookup_file.exists():
            with open(lookup_file) as f:
                self._inci_to_smiles = json.load(f)
            logger.info("Loaded %d INCI-to-SMILES mappings", len(self._inci_to_smiles))

        self._loaded = True
        logger.info("ToxicityScorer loaded: %d/%d targets", len(self._models), len(TOX21_TARGETS))

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        from biotech_ml.features.molecular import smiles_to_morgan

        self._ensure_loaded()
        start = time.perf_counter()

        smiles = input_data.get("smiles")
        inci_name = input_data.get("inci_name")

        if not smiles and inci_name:
            smiles = self._inci_to_smiles.get(inci_name.strip().upper())
            if not smiles:
                return self._fallback_result(inci_name, time.perf_counter() - start)

        if not smiles:
            raise ValueError("No SMILES resolved - provide 'smiles' or a known 'inci_name'")

        fingerprint = smiles_to_morgan(smiles, radius=2, n_bits=2048).reshape(1, -1)

        missing_targets: list[str] = []
        tox21_scores: dict[str, float] = {}
        for target in TOX21_TARGETS:
            model = self._models.get(target)
            if model is not None:
                proba = model.predict_proba(fingerprint)
                toxic_prob = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
                tox21_scores[target] = round(toxic_prob, 4)
            else:
                tox21_scores[target] = 0.0
                missing_targets.append(target)

        overall = self._weighted_mean(tox21_scores)
        latency = (time.perf_counter() - start) * 1000

        status = "partial" if missing_targets else "ok"

        result: dict[str, Any] = {
            "overall_score": round(overall, 4),
            "tox21_scores": tox21_scores,
            "status": status,
            "score_type": "raw_model_probability",
            "meta": {
                "model_id": self.model_id,
                "version": self._version,
                "latency_ms": round(latency, 2),
            },
        }
        if missing_targets:
            result["meta"]["missing_targets"] = missing_targets

        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self._version,
            "targets": TOX21_TARGETS,
            "loaded_targets": list(self._models.keys()),
            "inci_lookup_size": len(self._inci_to_smiles),
            "weight_note": "TOX21_WEIGHTS are heuristic and not derived from regulatory guidance",
        }

    def _weighted_mean(self, scores: dict[str, float]) -> float:
        total_weight = 0.0
        weighted_sum = 0.0
        for target, score in scores.items():
            weight = TOX21_WEIGHTS.get(target, 1.0 / len(TOX21_TARGETS))
            weighted_sum += score * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _fallback_result(self, inci_name: str | None, elapsed: float) -> dict[str, Any]:
        logger.warning("INCI name '%s' not found in lookup", inci_name)
        return {
            "overall_score": None,
            "tox21_scores": {target: None for target in TOX21_TARGETS},
            "status": "inci_not_resolved",
            "score_type": "none",
            "meta": {
                "model_id": self.model_id,
                "version": self._version,
                "latency_ms": round(elapsed * 1000, 2),
            },
        }
