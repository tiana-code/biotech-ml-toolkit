"""Drug-Lab Interaction - identifies known drug interferences on lab tests."""

import logging
from pathlib import Path
from typing import Any

import joblib

from biotech_ml.features.text import TFIDFIndex
from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

_MODEL_VERSION = "0.1.0"

INTERACTION_DB: dict[str, list[dict[str, str]]] = {
    "biotin": [
        {"test": "TROP", "effect": "Falsely decreased troponin (streptavidin-biotin assays)", "severity": "severe", "mechanism": "Biotin competes with biotinylated antibodies in immunoassays"},
        {"test": "TSH", "effect": "Falsely decreased TSH", "severity": "severe", "mechanism": "Biotin interference in sandwich immunoassays"},
        {"test": "FT4", "effect": "Falsely elevated free T4", "severity": "severe", "mechanism": "Biotin interference in competitive immunoassays"},
        {"test": "BHCG", "effect": "Falsely decreased beta-hCG", "severity": "moderate", "mechanism": "Biotin competes in streptavidin-based assays"},
        {"test": "FERR", "effect": "Falsely decreased ferritin", "severity": "moderate", "mechanism": "Biotin interference in sandwich immunoassays"},
    ],
    "heparin": [
        {"test": "PTT", "effect": "Prolonged aPTT", "severity": "severe", "mechanism": "Direct anticoagulant effect on intrinsic pathway"},
        {"test": "TT", "effect": "Prolonged thrombin time", "severity": "severe", "mechanism": "Direct thrombin inhibition"},
        {"test": "K", "effect": "Falsely elevated potassium (if heparin tube contamination)", "severity": "moderate", "mechanism": "Potassium salt in heparin tubes"},
        {"test": "TRIG", "effect": "Decreased triglycerides", "severity": "low", "mechanism": "Heparin activates lipoprotein lipase"},
    ],
    "warfarin": [
        {"test": "PT", "effect": "Prolonged prothrombin time / elevated INR", "severity": "severe", "mechanism": "Vitamin K antagonist - reduces factors II, VII, IX, X"},
        {"test": "INR", "effect": "Elevated INR (therapeutic goal)", "severity": "severe", "mechanism": "Vitamin K antagonist"},
    ],
    "acetaminophen": [
        {"test": "ALT", "effect": "Elevated ALT (hepatotoxicity at high doses)", "severity": "moderate", "mechanism": "NAPQI metabolite causes hepatocellular damage"},
        {"test": "AST", "effect": "Elevated AST", "severity": "moderate", "mechanism": "Hepatocellular damage from toxic metabolite"},
        {"test": "TBIL", "effect": "Elevated bilirubin in overdose", "severity": "moderate", "mechanism": "Hepatic dysfunction"},
    ],
    "metformin": [
        {"test": "B12", "effect": "Decreased vitamin B12", "severity": "moderate", "mechanism": "Impaired ileal absorption of B12-intrinsic factor complex"},
        {"test": "LAC", "effect": "Elevated lactate (rare lactic acidosis)", "severity": "severe", "mechanism": "Inhibition of mitochondrial complex I"},
        {"test": "CRE", "effect": "Monitor - contraindicated in renal impairment", "severity": "low", "mechanism": "Renal clearance required"},
    ],
    "lisinopril": [
        {"test": "K", "effect": "Elevated potassium", "severity": "moderate", "mechanism": "Reduced aldosterone secretion via ACE inhibition"},
        {"test": "CRE", "effect": "Mildly elevated creatinine (initial)", "severity": "low", "mechanism": "Reduced glomerular filtration pressure"},
    ],
    "atorvastatin": [
        {"test": "ALT", "effect": "Elevated ALT (hepatotoxicity)", "severity": "moderate", "mechanism": "Statin hepatotoxicity"},
        {"test": "CK", "effect": "Elevated CK (rhabdomyolysis risk)", "severity": "severe", "mechanism": "Statin myopathy - muscle fiber damage"},
    ],
    "prednisone": [
        {"test": "GLU", "effect": "Elevated glucose", "severity": "moderate", "mechanism": "Glucocorticoid-induced gluconeogenesis and insulin resistance"},
        {"test": "WBC", "effect": "Elevated WBC (leukocytosis)", "severity": "low", "mechanism": "Demargination of neutrophils"},
        {"test": "Ca", "effect": "Decreased calcium (chronic use)", "severity": "low", "mechanism": "Reduced intestinal calcium absorption"},
        {"test": "K", "effect": "Decreased potassium", "severity": "moderate", "mechanism": "Mineralocorticoid effect - renal potassium wasting"},
    ],
    "furosemide": [
        {"test": "K", "effect": "Decreased potassium (hypokalemia)", "severity": "severe", "mechanism": "Loop diuretic - increased renal potassium excretion"},
        {"test": "Na", "effect": "Decreased sodium (hyponatremia)", "severity": "moderate", "mechanism": "Increased renal sodium excretion"},
        {"test": "Ca", "effect": "Decreased calcium (hypocalcemia)", "severity": "moderate", "mechanism": "Increased renal calcium excretion"},
        {"test": "GLU", "effect": "Elevated glucose", "severity": "low", "mechanism": "Impaired glucose tolerance"},
        {"test": "UA", "effect": "Elevated uric acid", "severity": "low", "mechanism": "Reduced renal urate clearance"},
    ],
    "levothyroxine": [
        {"test": "TSH", "effect": "Decreased TSH (therapeutic effect)", "severity": "low", "mechanism": "Negative feedback on pituitary TSH secretion"},
        {"test": "FT4", "effect": "Elevated free T4", "severity": "low", "mechanism": "Exogenous thyroid hormone"},
    ],
    "amiodarone": [
        {"test": "TSH", "effect": "Altered TSH (hypo- or hyperthyroidism)", "severity": "severe", "mechanism": "Iodine load + inhibition of T4-to-T3 conversion"},
        {"test": "FT4", "effect": "Elevated free T4", "severity": "moderate", "mechanism": "Inhibits peripheral T4 deiodination"},
        {"test": "ALT", "effect": "Elevated ALT (hepatotoxicity)", "severity": "moderate", "mechanism": "Direct hepatotoxicity"},
    ],
    "lipemia": [
        {"test": "HGB", "effect": "Falsely elevated hemoglobin (spectrophotometric)", "severity": "moderate", "mechanism": "Lipid particles scatter light, increasing absorbance"},
        {"test": "TBIL", "effect": "Falsely elevated bilirubin", "severity": "moderate", "mechanism": "Turbidity interference"},
        {"test": "GLU", "effect": "Falsely decreased glucose", "severity": "low", "mechanism": "Volume displacement by lipids"},
        {"test": "Na", "effect": "Falsely decreased sodium (pseudohyponatremia)", "severity": "moderate", "mechanism": "Volume displacement in indirect ISE methods"},
    ],
    "hemolysis": [
        {"test": "K", "effect": "Falsely elevated potassium", "severity": "severe", "mechanism": "Intracellular potassium release from lysed RBCs"},
        {"test": "LDH", "effect": "Falsely elevated LDH", "severity": "moderate", "mechanism": "LDH released from lysed erythrocytes"},
        {"test": "AST", "effect": "Falsely elevated AST", "severity": "moderate", "mechanism": "AST released from lysed erythrocytes"},
        {"test": "TBIL", "effect": "Falsely elevated bilirubin", "severity": "low", "mechanism": "Hemoglobin spectral interference"},
    ],
}


class DrugLabInteraction(BaseModel):
    """Identifies known drug-lab interferences using a knowledge base + TF-IDF search."""

    def __init__(self) -> None:
        super().__init__()
        self._interaction_db: dict[str, list[dict[str, str]]] = {}
        self._tfidf_index: TFIDFIndex | None = None
        self._drug_names: list[str] = []

    @property
    def model_id(self) -> str:
        return "medical.drug_lab_interaction"

    def load(self, artifact_path: Path) -> None:
        artifact_file = artifact_path / "model.joblib"
        if artifact_file.exists():
            data = joblib.load(artifact_file)
            self._interaction_db = data.get("interaction_db", INTERACTION_DB)
            self._drug_names = data.get("drug_names", [])
            tfidf_path = artifact_path / "tfidf_index.joblib"
            if tfidf_path.exists():
                self._tfidf_index = TFIDFIndex()
                self._tfidf_index.load(tfidf_path)
            logger.info("Loaded drug-lab interaction model from %s", artifact_file)
        else:
            logger.warning(
                "No trained artifact at %s - using built-in knowledge base",
                artifact_file,
            )
            self._interaction_db = INTERACTION_DB.copy()
            self._drug_names = list(INTERACTION_DB.keys())
            self._build_tfidf_index()

        self._loaded = True

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        medications: list[str] = input_data.get("medications", [])
        test_code = input_data.get("test_code", "").upper()

        interactions: list[dict] = []

        for med_name in medications:
            med_lower = med_name.lower().strip()

            direct_matches = self._interaction_db.get(med_lower, [])
            for entry in direct_matches:
                if entry["test"].upper() == test_code:
                    interactions.append(
                        {
                            "drug": med_name,
                            "effect": entry["effect"],
                            "severity": entry["severity"],
                            "mechanism": entry.get("mechanism"),
                        }
                    )

            if not direct_matches and self._tfidf_index is not None:
                fuzzy_results = self._tfidf_index.search(med_lower, top_k=3)
                for match in fuzzy_results:
                    idx, score = match["index"], match["score"]
                    if score < 0.3:
                        continue
                    matched_drug = self._drug_names[idx]
                    for entry in self._interaction_db.get(matched_drug, []):
                        if entry["test"].upper() == test_code:
                            interactions.append(
                                {
                                    "drug": med_name,
                                    "effect": f"[Fuzzy match: {matched_drug}] {entry['effect']}",
                                    "severity": entry["severity"],
                                    "mechanism": entry.get("mechanism"),
                                }
                            )

        return {"interactions": interactions}

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": _MODEL_VERSION,
            "description": "Drug-lab interaction detection via knowledge base + TF-IDF fuzzy matching",
            "known_drugs": len(self._interaction_db),
            "total_interactions": sum(len(entries) for entries in self._interaction_db.values()),
        }

    def _build_tfidf_index(self) -> None:
        if not self._drug_names:
            return
        self._tfidf_index = TFIDFIndex(max_features=5000, ngram_range=(1, 2))
        self._tfidf_index.fit(self._drug_names)
