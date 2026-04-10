"""Cosmetic Allergen Detector - Rule-based + cosine similarity matching for EU 26 allergens."""

import json
import logging
import time
from pathlib import Path
from typing import Any

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

EU_26_ALLERGENS: list[dict[str, Any]] = [
    {"name": "Amyl cinnamal", "inci": "AMYL CINNAMAL", "eu_number": 1,
     "synonyms": ["ALPHA-AMYL CINNAMALDEHYDE", "2-BENZYLIDENE HEPTANAL", "ALPHA-AMYLCINNAMALDEHYDE"]},
    {"name": "Amylcinnamyl alcohol", "inci": "AMYLCINNAMYL ALCOHOL", "eu_number": 2,
     "synonyms": ["ALPHA-AMYL CINNAMYL ALCOHOL", "2-PENTYL-3-PHENYL-2-PROPEN-1-OL"]},
    {"name": "Anise alcohol", "inci": "ANISE ALCOHOL", "eu_number": 3,
     "synonyms": ["4-METHOXYBENZYL ALCOHOL", "ANISYL ALCOHOL", "P-METHOXYBENZYL ALCOHOL"]},
    {"name": "Benzyl alcohol", "inci": "BENZYL ALCOHOL", "eu_number": 4,
     "synonyms": ["PHENYLMETHANOL", "BENZENEMETHANOL", "ALPHA-HYDROXYTOLUENE"]},
    {"name": "Benzyl benzoate", "inci": "BENZYL BENZOATE", "eu_number": 5,
     "synonyms": ["BENZOIC ACID BENZYL ESTER", "PHENYLMETHYL BENZOATE"]},
    {"name": "Benzyl cinnamate", "inci": "BENZYL CINNAMATE", "eu_number": 6,
     "synonyms": ["CINNAMIC ACID BENZYL ESTER", "3-PHENYL-2-PROPENOIC ACID PHENYLMETHYL ESTER"]},
    {"name": "Benzyl salicylate", "inci": "BENZYL SALICYLATE", "eu_number": 7,
     "synonyms": ["SALICYLIC ACID BENZYL ESTER", "2-HYDROXYBENZOIC ACID PHENYLMETHYL ESTER"]},
    {"name": "Cinnamal", "inci": "CINNAMAL", "eu_number": 8,
     "synonyms": ["CINNAMALDEHYDE", "TRANS-CINNAMALDEHYDE", "3-PHENYL-2-PROPENAL",
                   "CINNAMIC ALDEHYDE", "BETA-PHENYLACROLEIN"]},
    {"name": "Cinnamyl alcohol", "inci": "CINNAMYL ALCOHOL", "eu_number": 9,
     "synonyms": ["3-PHENYL-2-PROPEN-1-OL", "CINNAMIC ALCOHOL", "GAMMA-PHENYLALLYL ALCOHOL"]},
    {"name": "Citral", "inci": "CITRAL", "eu_number": 10,
     "synonyms": ["GERANIAL", "NERAL", "3,7-DIMETHYL-2,6-OCTADIENAL", "LEMONAL"]},
    {"name": "Citronellol", "inci": "CITRONELLOL", "eu_number": 11,
     "synonyms": ["3,7-DIMETHYL-6-OCTEN-1-OL", "BETA-CITRONELLOL", "DIHYDROGERANIOL"]},
    {"name": "Coumarin", "inci": "COUMARIN", "eu_number": 12,
     "synonyms": ["2H-1-BENZOPYRAN-2-ONE", "1,2-BENZOPYRONE", "CIS-O-COUMARINIC ACID LACTONE"]},
    {"name": "Eugenol", "inci": "EUGENOL", "eu_number": 13,
     "synonyms": ["4-ALLYL-2-METHOXYPHENOL", "2-METHOXY-4-(2-PROPENYL)PHENOL",
                   "4-ALLYLGUAIACOL", "EUGENIC ACID"]},
    {"name": "Farnesol", "inci": "FARNESOL", "eu_number": 14,
     "synonyms": ["3,7,11-TRIMETHYL-2,6,10-DODECATRIEN-1-OL", "TRANS,TRANS-FARNESOL"]},
    {"name": "Geraniol", "inci": "GERANIOL", "eu_number": 15,
     "synonyms": ["TRANS-3,7-DIMETHYL-2,6-OCTADIEN-1-OL", "2,6-DIMETHYL-TRANS-2,6-OCTADIEN-8-OL"]},
    {"name": "Hexyl cinnamal", "inci": "HEXYL CINNAMAL", "eu_number": 16,
     "synonyms": ["HEXYL CINNAMALDEHYDE", "ALPHA-HEXYL CINNAMALDEHYDE",
                   "2-(PHENYLMETHYLENE)OCTANAL", "ALPHA-HEXYLCINNAMALDEHYDE"]},
    {"name": "Hydroxycitronellal", "inci": "HYDROXYCITRONELLAL", "eu_number": 17,
     "synonyms": ["7-HYDROXYCITRONELLAL", "3,7-DIMETHYL-7-HYDROXYOCTANAL"]},
    {"name": "Hydroxyisohexyl 3-cyclohexene carboxaldehyde", "inci": "HYDROXYISOHEXYL 3-CYCLOHEXENE CARBOXALDEHYDE",
     "eu_number": 18,
     "synonyms": ["HICC", "LYRAL", "4-(4-HYDROXY-4-METHYLPENTYL)-3-CYCLOHEXENE-1-CARBOXALDEHYDE"]},
    {"name": "Isoeugenol", "inci": "ISOEUGENOL", "eu_number": 19,
     "synonyms": ["2-METHOXY-4-PROPENYLPHENOL", "4-PROPENYLGUAIACOL", "TRANS-ISOEUGENOL"]},
    {"name": "Lilial", "inci": "BUTYLPHENYL METHYLPROPIONAL", "eu_number": 20,
     "synonyms": ["LILIAL", "2-(4-TERT-BUTYLBENZYL)PROPIONALDEHYDE",
                   "P-TERT-BUTYL-ALPHA-METHYLHYDROCINNAMIC ALDEHYDE", "BMHCA"]},
    {"name": "d-Limonene", "inci": "LIMONENE", "eu_number": 21,
     "synonyms": ["D-LIMONENE", "(R)-LIMONENE", "(+)-LIMONENE",
                   "4-ISOPROPENYL-1-METHYLCYCLOHEXENE", "(R)-4-ISOPROPENYL-1-METHYLCYCLOHEXENE"]},
    {"name": "Linalool", "inci": "LINALOOL", "eu_number": 22,
     "synonyms": ["3,7-DIMETHYL-1,6-OCTADIEN-3-OL", "LINALYL ALCOHOL", "DL-LINALOOL"]},
    {"name": "Methyl 2-octynoate", "inci": "METHYL 2-OCTYNOATE", "eu_number": 23,
     "synonyms": ["METHYL HEPTIN CARBONATE", "METHYL HEPTYNE CARBONATE"]},
    {"name": "Alpha-isomethyl ionone", "inci": "ALPHA-ISOMETHYL IONONE", "eu_number": 24,
     "synonyms": ["3-METHYL-4-(2,6,6-TRIMETHYL-2-CYCLOHEXEN-1-YL)-3-BUTEN-2-ONE",
                   "ISOMETHYL IONONE", "METHYL IONONE ALPHA"]},
    {"name": "Evernia prunastri extract", "inci": "EVERNIA PRUNASTRI EXTRACT", "eu_number": 25,
     "synonyms": ["OAK MOSS EXTRACT", "OAKMOSS EXTRACT", "OAKMOSS ABSOLUTE",
                   "TREEMOSS EXTRACT", "OAK MOSS ABSOLUTE"]},
    {"name": "Evernia furfuracea extract", "inci": "EVERNIA FURFURACEA EXTRACT", "eu_number": 26,
     "synonyms": ["TREEMOSS EXTRACT", "TREE MOSS EXTRACT", "TREE MOSS ABSOLUTE",
                   "TREEMOSS ABSOLUTE", "FURFURACEOUS EVERNIA EXTRACT"]},
]


def _normalize(name: str) -> str:
    return name.strip().upper().replace("-", " ").replace("  ", " ")


class CosmeticAllergenDetector(BaseModel):
    """Detects EU 26 fragrance allergens in cosmetic ingredient lists using rule-based matching."""

    def __init__(self) -> None:
        super().__init__()
        self._allergen_db: list[dict[str, Any]] = []
        self._index: dict[str, dict[str, Any]] = {}
        self._version = "1.0.0"

    @property
    def model_id(self) -> str:
        return "chemistry.allergen_detector"

    def load(self, artifact_path: Path) -> None:
        logger.info("Loading CosmeticAllergenDetector from %s", artifact_path)

        db_file = artifact_path / "eu26_allergens.json"
        if db_file.exists():
            with open(db_file) as f:
                self._allergen_db = json.load(f)
            logger.info("Loaded allergen DB from file: %d entries", len(self._allergen_db))
        else:
            self._allergen_db = EU_26_ALLERGENS
            logger.info("Using built-in EU 26 allergen database")

        self._build_index()
        self._loaded = True
        logger.info("CosmeticAllergenDetector loaded: %d index entries", len(self._index))

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        start = time.perf_counter()

        ingredients: list[str] = input_data["ingredients"]
        found_allergens: list[dict[str, Any]] = []
        seen: set[int] = set()

        for ingredient in ingredients:
            normalized = _normalize(ingredient)
            matches = self._match_ingredient(normalized)
            for allergen in matches:
                eu_num = allergen["eu_number"]
                if eu_num not in seen:
                    seen.add(eu_num)
                    found_allergens.append({
                        "name": allergen["name"],
                        "inci_name": allergen["inci"],
                        "eu_number": allergen["eu_number"],
                        "regulation": "EU 1223/2009 Annex III",
                        "source_ingredient": ingredient,
                    })

        latency = (time.perf_counter() - start) * 1000

        return {
            "allergens": found_allergens,
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
            "allergen_count": len(self._allergen_db),
            "index_size": len(self._index),
        }

    def _build_index(self) -> None:
        self._index = {}
        for allergen in self._allergen_db:
            inci = _normalize(allergen["inci"])
            self._index[inci] = allergen

            name_norm = _normalize(allergen["name"])
            self._index[name_norm] = allergen

            for synonym in allergen.get("synonyms", []):
                syn_norm = _normalize(synonym)
                self._index[syn_norm] = allergen

    def _match_ingredient(self, normalized_ingredient: str) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []

        if normalized_ingredient in self._index:
            matches.append(self._index[normalized_ingredient])
            return matches

        for key, allergen in self._index.items():
            if key in normalized_ingredient or normalized_ingredient in key:
                if allergen not in matches:
                    matches.append(allergen)

        if not matches:
            matches = self._fuzzy_match(normalized_ingredient)

        return matches

    def _fuzzy_match(self, normalized_ingredient: str, threshold: float = 0.7) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        ingredient_tokens = set(normalized_ingredient.split())

        if not ingredient_tokens:
            return matches

        for key, allergen in self._index.items():
            key_tokens = set(key.split())
            if not key_tokens:
                continue

            intersection = ingredient_tokens & key_tokens
            union = ingredient_tokens | key_tokens
            similarity = len(intersection) / len(union)

            if similarity >= threshold and allergen not in matches:
                matches.append(allergen)

        return matches
