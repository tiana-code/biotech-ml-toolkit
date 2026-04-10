"""Result Anomaly Detector - flags lab results outside expected ranges using Isolation Forest."""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

REFERENCE_RANGES: dict[str, tuple[float, float]] = {
    "GLU": (70.0, 110.0),
    "WBC": (4.0, 11.0),
    "HGB": (12.0, 17.5),
    "PLT": (150.0, 400.0),
    "CRE": (0.6, 1.2),
    "ALT": (7.0, 56.0),
    "AST": (10.0, 40.0),
    "K": (3.5, 5.1),
    "Na": (136.0, 145.0),
    "Ca": (8.5, 10.5),
    "BUN": (7.0, 20.0),
    "TSH": (0.4, 4.0),
    "CHOL": (0.0, 200.0),
    "TG": (0.0, 150.0),
    "TBIL": (0.1, 1.2),
    "ALP": (44.0, 147.0),
    "RBC": (4.2, 5.9),
    "HCT": (36.0, 50.0),
    "MCV": (80.0, 100.0),
    "MCH": (27.0, 33.0),
}

_MODEL_VERSION = "0.1.0"


class ResultAnomalyDetector(BaseModel):
    """Detects anomalous lab results using Isolation Forest + reference ranges."""

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._reference_ranges: dict[str, tuple[float, float]] = {}
        self._feature_names: list[str] = []

    @property
    def model_id(self) -> str:
        return "medical.anomaly_detector"

    def load(self, artifact_path: Path) -> None:
        artifact_file = artifact_path / "model.joblib"
        if not artifact_file.exists():
            logger.warning(
                "No trained artifact at %s - using reference-range-only mode",
                artifact_file,
            )
            self._reference_ranges = REFERENCE_RANGES.copy()
            self._loaded = True
            return

        data = joblib.load(artifact_file)
        self._model = data["model"]
        self._reference_ranges = data.get("reference_ranges", REFERENCE_RANGES)
        self._feature_names = data.get("feature_names", [])
        self._loaded = True
        logger.info("Loaded anomaly detector from %s", artifact_file)

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        results: list[dict] = input_data.get("results", [])
        if not results:
            return {"anomaly_score": 0.0, "flags": []}

        flags: list[dict] = []
        anomaly_scores: list[float] = []
        unsupported_tests: list[str] = []

        for lab_result in results:
            test_code = lab_result.get("test_code", "").upper()
            value = float(lab_result.get("value", 0.0))

            ref = self._reference_ranges.get(test_code)
            if ref is None:
                unsupported_tests.append(test_code)
                continue

            low, high = ref
            mid = (low + high) / 2.0
            span = (high - low) / 2.0 if high != low else 1.0

            deviation = abs(value - mid) / span
            anomaly_scores.append(min(deviation / 3.0, 1.0))

            if value < low or value > high:
                if deviation > 3.0:
                    severity = "critical"
                elif deviation > 2.0:
                    severity = "high"
                elif deviation > 1.5:
                    severity = "medium"
                else:
                    severity = "low"

                direction = "above" if value > high else "below"
                flags.append(
                    {
                        "test_code": test_code,
                        "severity": severity,
                        "message": f"{test_code}={value} is {direction} reference range [{low}-{high}]",
                        "expected_range": (low, high),
                    }
                )

        if self._model is not None and results:
            feature_vec = self._build_feature_vector(results)
            if feature_vec is not None:
                iso_score = -self._model.score_samples(feature_vec.reshape(1, -1))[0]
                iso_normalized = min(max(iso_score, 0.0), 1.0)
                anomaly_scores.append(iso_normalized)

        overall = float(np.mean(anomaly_scores)) if anomaly_scores else 0.0

        return {
            "anomaly_score": round(overall, 4),
            "score_type": "heuristic_deviation",
            "flags": flags,
            "unsupported_tests": unsupported_tests,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": _MODEL_VERSION,
            "description": "Detects anomalous lab results via Isolation Forest + reference ranges",
            "supported_tests": list(self._reference_ranges.keys()),
            "has_trained_model": self._model is not None,
            "reference_note": "Built-in ranges are approximate adult unisex defaults. Always validate against laboratory-specific reference intervals.",
        }

    def _build_feature_vector(self, results: list[dict]) -> np.ndarray | None:
        if not self._feature_names:
            return None
        feature_map = {lr.get("test_code", "").upper(): float(lr.get("value", 0.0)) for lr in results}
        vec = []
        for fname in self._feature_names:
            if fname in feature_map:
                ref = self._reference_ranges.get(fname, (0.0, 1.0))
                mid = (ref[0] + ref[1]) / 2.0
                span = (ref[1] - ref[0]) / 2.0 if ref[1] != ref[0] else 1.0
                vec.append((feature_map[fname] - mid) / span)
            else:
                vec.append(0.0)
        return np.array(vec)
