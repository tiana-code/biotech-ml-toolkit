"""Minimum Inhibitory Concentration (MIC) regressor.

Uses LightGBM regression to predict MIC values on a log2 scale, then
interprets the result using EUCAST/CLSI breakpoint tables.
"""

import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

MIC_DILUTIONS: list[float] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]

DEFAULT_BREAKPOINTS: dict[tuple[str, str], dict[str, Any]] = {
    ("escherichia_coli", "ampicillin"): {"S": 8.0, "R": 8.0, "source": "EUCAST"},
    ("escherichia_coli", "ciprofloxacin"): {"S": 0.25, "R": 0.5, "source": "EUCAST"},
    ("escherichia_coli", "ceftriaxone"): {"S": 1.0, "R": 2.0, "source": "EUCAST"},
    ("escherichia_coli", "gentamicin"): {"S": 2.0, "R": 4.0, "source": "EUCAST"},
    ("escherichia_coli", "meropenem"): {"S": 2.0, "R": 8.0, "source": "EUCAST"},
    ("escherichia_coli", "trimethoprim_sulfamethoxazole"): {"S": 2.0, "R": 4.0, "source": "EUCAST"},
    ("staphylococcus_aureus", "vancomycin"): {"S": 2.0, "R": 2.0, "source": "EUCAST"},
    ("staphylococcus_aureus", "gentamicin"): {"S": 1.0, "R": 1.0, "source": "EUCAST"},
    ("staphylococcus_aureus", "ciprofloxacin"): {"S": 1.0, "R": 1.0, "source": "EUCAST"},
    ("klebsiella_pneumoniae", "meropenem"): {"S": 2.0, "R": 8.0, "source": "EUCAST"},
    ("klebsiella_pneumoniae", "ceftriaxone"): {"S": 1.0, "R": 2.0, "source": "EUCAST"},
    ("klebsiella_pneumoniae", "ciprofloxacin"): {"S": 0.25, "R": 0.5, "source": "EUCAST"},
    ("pseudomonas_aeruginosa", "meropenem"): {"S": 2.0, "R": 8.0, "source": "EUCAST"},
    ("pseudomonas_aeruginosa", "ciprofloxacin"): {"S": 0.5, "R": 1.0, "source": "EUCAST"},
    ("pseudomonas_aeruginosa", "gentamicin"): {"S": 4.0, "R": 4.0, "source": "EUCAST"},
    ("enterococcus_faecalis", "ampicillin"): {"S": 4.0, "R": 8.0, "source": "EUCAST"},
    ("enterococcus_faecalis", "vancomycin"): {"S": 4.0, "R": 4.0, "source": "EUCAST"},
    ("streptococcus_pneumoniae", "ceftriaxone"): {"S": 0.5, "R": 2.0, "source": "CLSI"},
    ("streptococcus_pneumoniae", "meropenem"): {"S": 0.25, "R": 1.0, "source": "EUCAST"},
    ("acinetobacter_baumannii", "meropenem"): {"S": 2.0, "R": 8.0, "source": "EUCAST"},
    ("proteus_mirabilis", "ampicillin"): {"S": 8.0, "R": 8.0, "source": "CLSI"},
}


def _snap_to_dilution(mic: float) -> float:
    if mic <= MIC_DILUTIONS[0]:
        return MIC_DILUTIONS[0]
    if mic >= MIC_DILUTIONS[-1]:
        return MIC_DILUTIONS[-1]
    log_mic = math.log2(mic)
    best = min(MIC_DILUTIONS, key=lambda d: abs(math.log2(d) - log_mic))
    return best


def _format_dilution(mic: float) -> str:
    if mic <= MIC_DILUTIONS[0]:
        return f"<={MIC_DILUTIONS[0]:g}"
    if mic >= MIC_DILUTIONS[-1]:
        return f">={MIC_DILUTIONS[-1]:g}"
    return f"{mic:g}"


def _interpret_mic(
    mic: float,
    organism_id: str,
    antibiotic_id: str,
    breakpoints: dict[tuple[str, str], dict[str, Any]],
) -> tuple[str, str | None]:
    breakpoint_entry = breakpoints.get((organism_id, antibiotic_id))
    if breakpoint_entry is None:
        return "S", None

    source = breakpoint_entry["source"]
    if mic <= breakpoint_entry["S"]:
        return "S", source
    if mic > breakpoint_entry["R"]:
        return "R", source
    return "I", source


class MICRegressor(BaseModel):
    """LightGBM regressor for MIC value prediction."""

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._organism_encoder: dict[str, int] = {}
        self._antibiotic_encoder: dict[str, int] = {}
        self._breakpoints: dict[tuple[str, str], dict[str, Any]] = {}
        self._version: str = "0.1.0"

    @property
    def model_id(self) -> str:
        return "microbiology.mic_regressor"

    def load(self, artifact_path: Path) -> None:
        try:
            data = joblib.load(artifact_path / "mic_regressor.joblib")
            self._model = data["model"]
            self._organism_encoder = data["organism_encoder"]
            self._antibiotic_encoder = data["antibiotic_encoder"]
            self._breakpoints = data.get("breakpoints", DEFAULT_BREAKPOINTS)
            self._version = data.get("version", self._version)
            self._loaded = True
            logger.info("Loaded MIC regressor with %d breakpoint entries", len(self._breakpoints))
        except Exception:
            logger.exception("Failed to load MIC regressor from %s", artifact_path)
            raise

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        organism_id: str = input_data["organism_id"]
        antibiotic_id: str = input_data["antibiotic_id"]
        extra_features: dict[str, Any] | None = input_data.get("features")

        features = self._encode_features(organism_id, antibiotic_id, extra_features)
        raw_log2 = float(self._model.predict(features)[0])
        raw_mic = 2.0 ** raw_log2
        mic = _snap_to_dilution(raw_mic)
        dilution_range = _format_dilution(mic)
        interpretation, source = _interpret_mic(
            mic, organism_id, antibiotic_id, self._breakpoints
        )

        return {
            "mic": round(mic, 4),
            "dilution_range": dilution_range,
            "interpretation": interpretation,
            "breakpoint_source": source,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self._version,
            "dilutions": MIC_DILUTIONS,
            "breakpoint_count": len(self._breakpoints),
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
