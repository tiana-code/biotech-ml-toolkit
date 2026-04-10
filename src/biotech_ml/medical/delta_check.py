"""Delta Check - detects suspicious changes between consecutive lab results."""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

_MODEL_VERSION = "0.1.0"

DELTA_THRESHOLDS: dict[str, dict[str, float]] = {
    "GLU": {"mean_delta": 5.0, "std_delta": 15.0, "max_percent": 50.0},
    "WBC": {"mean_delta": 0.2, "std_delta": 1.5, "max_percent": 60.0},
    "HGB": {"mean_delta": 0.1, "std_delta": 0.8, "max_percent": 15.0},
    "PLT": {"mean_delta": 5.0, "std_delta": 30.0, "max_percent": 50.0},
    "CRE": {"mean_delta": 0.02, "std_delta": 0.15, "max_percent": 30.0},
    "ALT": {"mean_delta": 2.0, "std_delta": 10.0, "max_percent": 50.0},
    "AST": {"mean_delta": 1.5, "std_delta": 8.0, "max_percent": 50.0},
    "K": {"mean_delta": 0.1, "std_delta": 0.3, "max_percent": 20.0},
    "Na": {"mean_delta": 0.5, "std_delta": 2.0, "max_percent": 5.0},
    "Ca": {"mean_delta": 0.1, "std_delta": 0.4, "max_percent": 10.0},
    "BUN": {"mean_delta": 1.0, "std_delta": 3.0, "max_percent": 40.0},
    "TSH": {"mean_delta": 0.2, "std_delta": 0.6, "max_percent": 50.0},
    "CHOL": {"mean_delta": 3.0, "std_delta": 12.0, "max_percent": 20.0},
    "TG": {"mean_delta": 5.0, "std_delta": 20.0, "max_percent": 40.0},
    "TBIL": {"mean_delta": 0.05, "std_delta": 0.2, "max_percent": 50.0},
}


class DeltaChecker(BaseModel):
    """Combines rule-based z-score delta checks with Isolation Forest for anomaly detection."""

    def __init__(self) -> None:
        super().__init__()
        self._thresholds: dict[str, dict[str, float]] = {}
        self._iso_model: Any = None

    @property
    def model_id(self) -> str:
        return "medical.delta_check"

    def load(self, artifact_path: Path) -> None:
        artifact_file = artifact_path / "model.joblib"
        if artifact_file.exists():
            data = joblib.load(artifact_file)
            self._thresholds = data.get("thresholds", DELTA_THRESHOLDS)
            self._iso_model = data.get("iso_model")
            logger.info("Loaded delta check model from %s", artifact_file)
        else:
            logger.warning(
                "No trained artifact at %s - using default thresholds",
                artifact_file,
            )
            self._thresholds = DELTA_THRESHOLDS.copy()

        self._loaded = True

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        test_code = input_data.get("test_code", "").upper()
        current = float(input_data.get("current_value", 0.0))
        previous = float(input_data.get("previous_value", 0.0))
        hours_between = input_data.get("hours_between")

        delta = current - previous
        delta_percent = (delta / previous * 100.0) if previous != 0.0 else 0.0

        thresholds = self._thresholds.get(test_code, {"mean_delta": 0.0, "std_delta": 1.0, "max_percent": 50.0})
        mean_d = thresholds["mean_delta"]
        std_d = thresholds["std_delta"]
        max_pct = thresholds["max_percent"]

        z_score = (abs(delta) - mean_d) / std_d if std_d != 0 else 0.0

        rule_flag = abs(delta_percent) > max_pct or z_score > 2.0

        iso_flag = False
        if self._iso_model is not None:
            features = np.array([[abs(delta), abs(delta_percent), z_score]])
            iso_score = -self._iso_model.score_samples(features)[0]
            iso_flag = iso_score > 0.5

        delta_flag = rule_flag or iso_flag

        if z_score > 3.0 or abs(delta_percent) > max_pct * 2:
            severity = "critical"
        elif z_score > 2.0 or abs(delta_percent) > max_pct:
            severity = "high"
        elif z_score > 1.5 or abs(delta_percent) > max_pct * 0.7:
            severity = "medium"
        else:
            severity = "low"

        if hours_between is not None and hours_between < 1.0 and delta_flag:
            severity = "critical" if severity in ("high", "critical") else "high"

        return {
            "delta_flag": delta_flag,
            "z_score": round(z_score, 4),
            "delta_percent": round(delta_percent, 4),
            "severity": severity,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": _MODEL_VERSION,
            "description": "Hybrid rule-based + Isolation Forest delta check",
            "supported_tests": list(self._thresholds.keys()),
            "has_iso_model": self._iso_model is not None,
        }
