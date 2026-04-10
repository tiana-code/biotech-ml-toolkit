"""GHS Classifier - XGBoost multi-label on Morgan fingerprints for GHS hazard classification."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

H_CODE_DESCRIPTIONS: dict[str, str] = {
    "H300": "Fatal if swallowed",
    "H301": "Toxic if swallowed",
    "H302": "Harmful if swallowed",
    "H303": "May be harmful if swallowed",
    "H304": "May be fatal if swallowed and enters airways",
    "H310": "Fatal in contact with skin",
    "H311": "Toxic in contact with skin",
    "H312": "Harmful in contact with skin",
    "H313": "May be harmful in contact with skin",
    "H314": "Causes severe skin burns and eye damage",
    "H315": "Causes skin irritation",
    "H316": "Causes mild skin irritation",
    "H317": "May cause an allergic skin reaction",
    "H318": "Causes serious eye damage",
    "H319": "Causes serious eye irritation",
    "H330": "Fatal if inhaled",
    "H331": "Toxic if inhaled",
    "H332": "Harmful if inhaled",
    "H333": "May be harmful if inhaled",
    "H334": "May cause allergy or asthma symptoms or breathing difficulties if inhaled",
    "H335": "May cause respiratory irritation",
    "H336": "May cause drowsiness or dizziness",
    "H340": "May cause genetic defects",
    "H341": "Suspected of causing genetic defects",
    "H350": "May cause cancer",
    "H351": "Suspected of causing cancer",
    "H360": "May damage fertility or the unborn child",
    "H361": "Suspected of damaging fertility or the unborn child",
    "H370": "Causes damage to organs",
    "H371": "May cause damage to organs",
    "H372": "Causes damage to organs through prolonged or repeated exposure",
    "H373": "May cause damage to organs through prolonged or repeated exposure",
    "H400": "Very toxic to aquatic life",
    "H410": "Very toxic to aquatic life with long lasting effects",
    "H411": "Toxic to aquatic life with long lasting effects",
    "H412": "Harmful to aquatic life with long lasting effects",
    "H413": "May cause long lasting harmful effects to aquatic life",
    "H420": "Harms public health and the environment by destroying ozone in the upper atmosphere",
}

H_CODE_TO_PICTOGRAM: dict[str, list[str]] = {
    "H300": ["GHS06"], "H301": ["GHS06"], "H302": ["GHS07"], "H303": ["GHS07"],
    "H304": ["GHS08"], "H310": ["GHS06"], "H311": ["GHS06"], "H312": ["GHS07"],
    "H313": ["GHS07"], "H314": ["GHS05"], "H315": ["GHS07"], "H317": ["GHS07"],
    "H318": ["GHS05"], "H319": ["GHS07"],
    "H330": ["GHS06"], "H331": ["GHS06"], "H332": ["GHS07"], "H334": ["GHS08"],
    "H335": ["GHS07"], "H336": ["GHS07"],
    "H340": ["GHS08"], "H341": ["GHS08"], "H350": ["GHS08"], "H351": ["GHS08"],
    "H360": ["GHS08"], "H361": ["GHS08"], "H370": ["GHS08"], "H371": ["GHS08"],
    "H372": ["GHS08"], "H373": ["GHS08"],
    "H400": ["GHS09"], "H410": ["GHS09"], "H411": ["GHS09"], "H412": ["GHS09"],
    "H420": ["GHS07"],
}

DANGER_H_CODES: set[str] = {
    "H300", "H310", "H330", "H304", "H314", "H318", "H330",
    "H340", "H350", "H360", "H370", "H372", "H400", "H410",
}

ALL_H_CODES: list[str] = sorted(H_CODE_DESCRIPTIONS.keys())


class GHSClassifier(BaseModel):
    """Predicts GHS hazard classification (H-codes, pictograms, signal word) from SMILES."""

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._h_codes: list[str] = ALL_H_CODES
        self._version = "1.0.0"

    @property
    def model_id(self) -> str:
        return "chemistry.ghs_classifier"

    def load(self, artifact_path: Path) -> None:
        import xgboost as xgb

        logger.info("Loading GHSClassifier from %s", artifact_path)
        model_file = artifact_path / "ghs_model.json"

        if not model_file.exists():
            raise FileNotFoundError(f"GHS model not found: {model_file}")

        self._model = xgb.XGBClassifier()
        self._model.load_model(str(model_file))

        h_codes_file = artifact_path / "ghs_h_codes.json"
        if h_codes_file.exists():
            with open(h_codes_file) as f:
                self._h_codes = json.load(f)

        self._loaded = True
        logger.info("GHSClassifier loaded: %d H-code labels", len(self._h_codes))

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        from biotech_ml.features.molecular import smiles_to_morgan

        self._ensure_loaded()
        start = time.perf_counter()

        smiles: str = input_data["smiles"]
        fingerprint = smiles_to_morgan(smiles, radius=2, n_bits=2048).reshape(1, -1)

        raw_proba = self._model.predict_proba(fingerprint)

        if isinstance(raw_proba, list):
            probas = np.array([arr[0][1] if arr.shape[1] > 1 else arr[0][0] for arr in raw_proba])
        elif raw_proba.ndim == 2 and raw_proba.shape[1] == len(self._h_codes):
            probas = raw_proba[0]
        elif raw_proba.ndim == 2 and raw_proba.shape[1] == 2:
            probas = raw_proba[:, 1] if raw_proba.shape[0] == len(self._h_codes) else raw_proba[0]
        else:
            probas = raw_proba.flatten()

        threshold = 0.5
        predicted_h_codes: list[str] = []
        for i, h_code in enumerate(self._h_codes):
            if i < len(probas) and probas[i] > threshold:
                predicted_h_codes.append(h_code)

        pictograms = self._h_codes_to_pictograms(predicted_h_codes)
        signal_word = self._determine_signal_word(predicted_h_codes)
        latency = (time.perf_counter() - start) * 1000

        return {
            "h_codes": predicted_h_codes,
            "pictograms": pictograms,
            "signal_word": signal_word,
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
            "h_code_count": len(self._h_codes),
            "pictograms": sorted({pic for pics in H_CODE_TO_PICTOGRAM.values() for pic in pics}),
        }

    @staticmethod
    def _h_codes_to_pictograms(h_codes: list[str]) -> list[str]:
        pictograms: set[str] = set()
        for h_code in h_codes:
            pics = H_CODE_TO_PICTOGRAM.get(h_code, [])
            pictograms.update(pics)
        return sorted(pictograms)

    @staticmethod
    def _determine_signal_word(h_codes: list[str]) -> Literal["Danger", "Warning"] | None:
        if not h_codes:
            return None
        for h_code in h_codes:
            if h_code in DANGER_H_CODES:
                return "Danger"
        return "Warning"
