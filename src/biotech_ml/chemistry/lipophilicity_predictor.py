"""Lipophilicity Predictor - XGBoost regression on Morgan fingerprints for logD prediction."""

import logging
import time
from pathlib import Path
from typing import Any, Literal

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)


def _assess_skin_penetration(log_d: float) -> Literal["high", "moderate", "low"]:
    if 1.0 <= log_d <= 3.0:
        return "high"
    if -1.0 <= log_d < 1.0 or 3.0 < log_d <= 5.0:
        return "moderate"
    return "low"


def _lipophilicity_category(log_d: float) -> str:
    if log_d < -1.0:
        return "very_hydrophilic"
    if log_d < 1.0:
        return "hydrophilic"
    if log_d < 3.0:
        return "moderately_lipophilic"
    if log_d < 5.0:
        return "lipophilic"
    return "very_lipophilic"


class LipophilicityPredictor(BaseModel):
    """Predicts lipophilicity (logD at pH 7.4) from SMILES and assesses skin penetration potential."""

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._version = "1.0.0"

    @property
    def model_id(self) -> str:
        return "chemistry.lipophilicity_predictor"

    def load(self, artifact_path: Path) -> None:
        import xgboost as xgb

        logger.info("Loading LipophilicityPredictor from %s", artifact_path)
        model_file = artifact_path / "lipophilicity_model.json"
        if not model_file.exists():
            raise FileNotFoundError(f"Lipophilicity model not found: {model_file}")

        self._model = xgb.XGBRegressor()
        self._model.load_model(str(model_file))
        self._loaded = True
        logger.info("LipophilicityPredictor loaded successfully")

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        from biotech_ml.features.molecular import smiles_to_morgan

        self._ensure_loaded()
        start = time.perf_counter()

        smiles: str = input_data["smiles"]
        fingerprint = smiles_to_morgan(smiles, radius=2, n_bits=2048).reshape(1, -1)

        log_d = float(self._model.predict(fingerprint)[0])
        penetration = _assess_skin_penetration(log_d)
        category = _lipophilicity_category(log_d)
        latency = (time.perf_counter() - start) * 1000

        return {
            "log_d": round(log_d, 4),
            "skin_penetration": penetration,
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
            "penetration_levels": ["high", "moderate", "low"],
            "categories": [
                "very_hydrophilic",
                "hydrophilic",
                "moderately_lipophilic",
                "lipophilic",
                "very_lipophilic",
            ],
        }
