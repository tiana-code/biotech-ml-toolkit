"""Solubility Predictor - XGBoost regression on Morgan fingerprints for aqueous solubility."""

import logging
import time
from pathlib import Path
from typing import Any, Literal

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

_SOLUBILITY_THRESHOLDS: list[tuple[float, str]] = [
    (100.0, "highly_soluble"),
    (1.0, "soluble"),
    (0.01, "slightly_soluble"),
]


def _categorize_solubility(mg_l: float) -> Literal["highly_soluble", "soluble", "slightly_soluble", "insoluble"]:
    for threshold, category in _SOLUBILITY_THRESHOLDS:
        if mg_l > threshold:
            return category  # type: ignore[return-value]
    return "insoluble"


def _log_s_to_mg_l(log_s: float, molecular_weight: float) -> float | None:
    if molecular_weight <= 0:
        return None
    solubility_mol_l = 10**log_s
    return solubility_mol_l * molecular_weight * 1000


class SolubilityPredictor(BaseModel):
    """Predicts aqueous solubility (log S) from SMILES using XGBoost regression."""

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._version = "1.0.0"

    @property
    def model_id(self) -> str:
        return "chemistry.solubility_predictor"

    def load(self, artifact_path: Path) -> None:
        import xgboost as xgb

        logger.info("Loading SolubilityPredictor from %s", artifact_path)
        model_file = artifact_path / "solubility_model.json"
        if not model_file.exists():
            raise FileNotFoundError(f"Solubility model not found: {model_file}")

        self._model = xgb.XGBRegressor()
        self._model.load_model(str(model_file))
        self._loaded = True
        logger.info("SolubilityPredictor loaded successfully")

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        from biotech_ml.features.molecular import smiles_to_morgan, smiles_to_descriptors

        self._ensure_loaded()
        start = time.perf_counter()

        smiles: str = input_data["smiles"]
        fingerprint = smiles_to_morgan(smiles, radius=2, n_bits=2048).reshape(1, -1)

        log_s = float(self._model.predict(fingerprint)[0])

        descriptors = smiles_to_descriptors(smiles)
        molecular_weight = descriptors.get("molecular_weight", 0.0)
        mg_l = _log_s_to_mg_l(log_s, molecular_weight)

        category = _categorize_solubility(mg_l) if mg_l is not None else "insoluble"
        latency = (time.perf_counter() - start) * 1000

        return {
            "log_s": round(log_s, 4),
            "solubility_mg_l": round(mg_l, 4) if mg_l is not None else None,
            "category": category,
            "meta": {
                "model_id": self.model_id,
                "version": self._version,
                "latency_ms": round(latency, 2),
            },
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self._version,
            "categories": ["highly_soluble", "soluble", "slightly_soluble", "insoluble"],
        }
