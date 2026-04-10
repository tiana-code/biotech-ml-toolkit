"""Organism Named Entity Recognition (NER).

Uses spaCy EntityRuler (rule-based) for extracting organism names from
clinical microbiology text. Supports full binomial names, abbreviations,
and common acronyms (MRSA, VRE, etc.).
"""

import logging
from pathlib import Path
from typing import Any

import joblib

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

TAXONOMY: dict[str, dict[str, Any]] = {
    "Escherichia": {"gram_type": "negative", "species": ["coli"]},
    "Staphylococcus": {"gram_type": "positive", "species": ["aureus", "epidermidis", "saprophyticus", "haemolyticus", "lugdunensis"]},
    "Streptococcus": {"gram_type": "positive", "species": ["pneumoniae", "pyogenes", "agalactiae", "mutans", "viridans"]},
    "Enterococcus": {"gram_type": "positive", "species": ["faecalis", "faecium"]},
    "Klebsiella": {"gram_type": "negative", "species": ["pneumoniae", "oxytoca", "aerogenes"]},
    "Pseudomonas": {"gram_type": "negative", "species": ["aeruginosa", "fluorescens", "putida"]},
    "Acinetobacter": {"gram_type": "negative", "species": ["baumannii", "lwoffii", "calcoaceticus"]},
    "Proteus": {"gram_type": "negative", "species": ["mirabilis", "vulgaris"]},
    "Salmonella": {"gram_type": "negative", "species": ["enterica", "typhimurium", "typhi"]},
    "Shigella": {"gram_type": "negative", "species": ["dysenteriae", "flexneri", "sonnei", "boydii"]},
    "Enterobacter": {"gram_type": "negative", "species": ["cloacae", "aerogenes", "hormaechei"]},
    "Serratia": {"gram_type": "negative", "species": ["marcescens"]},
    "Citrobacter": {"gram_type": "negative", "species": ["freundii", "koseri"]},
    "Morganella": {"gram_type": "negative", "species": ["morganii"]},
    "Providencia": {"gram_type": "negative", "species": ["stuartii", "rettgeri"]},
    "Haemophilus": {"gram_type": "negative", "species": ["influenzae", "parainfluenzae"]},
    "Neisseria": {"gram_type": "negative", "species": ["meningitidis", "gonorrhoeae"]},
    "Moraxella": {"gram_type": "negative", "species": ["catarrhalis"]},
    "Helicobacter": {"gram_type": "negative", "species": ["pylori"]},
    "Campylobacter": {"gram_type": "negative", "species": ["jejuni", "coli"]},
    "Vibrio": {"gram_type": "negative", "species": ["cholerae", "parahaemolyticus", "vulnificus"]},
    "Legionella": {"gram_type": "negative", "species": ["pneumophila"]},
    "Bordetella": {"gram_type": "negative", "species": ["pertussis"]},
    "Brucella": {"gram_type": "negative", "species": ["melitensis", "abortus"]},
    "Bacteroides": {"gram_type": "negative", "species": ["fragilis"]},
    "Clostridium": {"gram_type": "positive", "species": ["difficile", "perfringens", "botulinum", "tetani"]},
    "Clostridioides": {"gram_type": "positive", "species": ["difficile"]},
    "Bacillus": {"gram_type": "positive", "species": ["cereus", "subtilis", "anthracis"]},
    "Listeria": {"gram_type": "positive", "species": ["monocytogenes"]},
    "Corynebacterium": {"gram_type": "positive", "species": ["diphtheriae", "striatum", "jeikeium"]},
    "Mycobacterium": {"gram_type": "positive", "species": ["tuberculosis", "avium", "abscessus", "leprae"]},
    "Nocardia": {"gram_type": "positive", "species": ["asteroides", "brasiliensis"]},
    "Actinomyces": {"gram_type": "positive", "species": ["israelii"]},
    "Candida": {"gram_type": "fungus", "species": ["albicans", "glabrata", "tropicalis", "parapsilosis", "auris", "krusei"]},
    "Aspergillus": {"gram_type": "fungus", "species": ["fumigatus", "niger", "flavus", "terreus"]},
    "Cryptococcus": {"gram_type": "fungus", "species": ["neoformans", "gattii"]},
    "Mucor": {"gram_type": "fungus", "species": ["racemosus"]},
    "Pneumocystis": {"gram_type": "fungus", "species": ["jirovecii"]},
    "Trichophyton": {"gram_type": "fungus", "species": ["rubrum", "mentagrophytes"]},
    "Stenotrophomonas": {"gram_type": "negative", "species": ["maltophilia"]},
    "Burkholderia": {"gram_type": "negative", "species": ["cepacia", "pseudomallei"]},
}

ACRONYM_MAP: dict[str, str] = {
    "MRSA": "Staphylococcus aureus",
    "MSSA": "Staphylococcus aureus",
    "VISA": "Staphylococcus aureus",
    "VRSA": "Staphylococcus aureus",
    "VRE": "Enterococcus faecium",
    "ESBL": None,
    "CRE": None,
    "CoNS": "Staphylococcus epidermidis",
    "GAS": "Streptococcus pyogenes",
    "GBS": "Streptococcus agalactiae",
    "MRAB": "Acinetobacter baumannii",
    "MDR-TB": "Mycobacterium tuberculosis",
    "XDR-TB": "Mycobacterium tuberculosis",
}


def _build_spacy_patterns() -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []

    for genus, info in TAXONOMY.items():
        patterns.append({
            "label": "ORGANISM",
            "pattern": genus,
            "id": genus,
        })

        for species in info["species"]:
            patterns.append({
                "label": "ORGANISM",
                "pattern": [{"LOWER": genus.lower()}, {"LOWER": species.lower()}],
                "id": f"{genus} {species}",
            })

            abbr = f"{genus[0]}."
            patterns.append({
                "label": "ORGANISM",
                "pattern": [{"TEXT": abbr}, {"LOWER": species.lower()}],
                "id": f"{genus} {species}",
            })

            patterns.append({
                "label": "ORGANISM",
                "pattern": [{"TEXT": genus[0]}, {"TEXT": "."}, {"LOWER": species.lower()}],
                "id": f"{genus} {species}",
            })

    for acronym, canonical in ACRONYM_MAP.items():
        if canonical is not None:
            patterns.append({
                "label": "ORGANISM",
                "pattern": acronym,
                "id": canonical,
            })

    compound_patterns = [
        ("ESBL-producing", "Klebsiella pneumoniae", [{"LOWER": "esbl-producing"}, {"LOWER": "k"}, {"TEXT": "."}, {"LOWER": "pneumoniae"}]),
        ("ESBL-producing", "Escherichia coli", [{"LOWER": "esbl-producing"}, {"LOWER": "e"}, {"TEXT": "."}, {"LOWER": "coli"}]),
        ("Methicillin-resistant", "Staphylococcus aureus", [{"LOWER": "methicillin-resistant"}, {"LOWER": "staphylococcus"}, {"LOWER": "aureus"}]),
        ("Vancomycin-resistant", "Enterococcus", [{"LOWER": "vancomycin-resistant"}, {"LOWER": "enterococcus"}]),
    ]
    for _, canonical, token_pattern in compound_patterns:
        patterns.append({
            "label": "ORGANISM",
            "pattern": token_pattern,
            "id": canonical,
        })

    return patterns


def _lookup_taxonomy(organism_name: str) -> dict[str, Any] | None:
    parts = organism_name.split()
    if not parts:
        return None

    genus = parts[0]
    species = parts[1] if len(parts) > 1 else None

    info = TAXONOMY.get(genus)
    if info is None:
        for genus_name, data in TAXONOMY.items():
            if genus_name.lower() == genus.lower():
                info = data
                genus = genus_name
                break

    if info is None:
        return None

    result: dict[str, Any] = {
        "genus": genus,
        "gram_type": info["gram_type"],
    }
    if species:
        result["species"] = species
    return result


class OrganismNER(BaseModel):
    """spaCy EntityRuler-based organism NER."""

    def __init__(self) -> None:
        super().__init__()
        self._nlp: Any = None
        self._patterns: list[dict[str, Any]] = []
        self._version: str = "0.1.0"

    @property
    def model_id(self) -> str:
        return "microbiology.organism_ner"

    def load(self, artifact_path: Path) -> None:
        try:
            import spacy

            nlp_path = artifact_path / "organism_ner_nlp"
            if nlp_path.exists():
                self._nlp = spacy.load(str(nlp_path))
                logger.info("Loaded spaCy model from %s", nlp_path)
            else:
                self._nlp = spacy.blank("en")
                ruler = self._nlp.add_pipe("entity_ruler")
                patterns = _build_spacy_patterns()
                ruler.add_patterns(patterns)
                self._patterns = patterns
                logger.info(
                    "Built organism NER with %d patterns (no saved model found)", len(patterns)
                )

            meta_path = artifact_path / "organism_ner_meta.joblib"
            if meta_path.exists():
                meta = joblib.load(meta_path)
                self._version = meta.get("version", self._version)

            self._loaded = True
        except Exception:
            logger.exception("Failed to load organism NER from %s", artifact_path)
            raise

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        text: str = input_data["text"]
        doc = self._nlp(text)

        organisms: list[dict[str, Any]] = []
        for entity in doc.ents:
            if entity.label_ != "ORGANISM":
                continue

            canonical = entity.ent_id_ if entity.ent_id_ else entity.text
            taxonomy = _lookup_taxonomy(canonical)

            organisms.append({
                "name": entity.text,
                "span_start": entity.start_char,
                "span_end": entity.end_char,
                "match_type": "rule_based",
                "taxonomy": taxonomy,
            })

        return {"organisms": organisms}

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self._version,
            "genera_count": len(TAXONOMY),
            "acronym_count": len([acr for acr, org in ACRONYM_MAP.items() if org is not None]),
        }
