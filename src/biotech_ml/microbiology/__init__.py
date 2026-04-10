"""Microbiology domain models - AST, MIC, phenotype prediction, organism NER, microbiology QA."""

__all__ = [
    "ASTPredictor",
    "MICRegressor",
    "PhenotypePredictor",
    "OrganismNER",
    "MicrobiologyQA",
]


def __getattr__(name: str):
    """Lazy imports to avoid pulling in rdkit/spacy at import time."""
    _map = {
        "ASTPredictor": "biotech_ml.microbiology.ast_predictor",
        "MICRegressor": "biotech_ml.microbiology.mic_regressor",
        "MicrobiologyQA": "biotech_ml.microbiology.microbiology_qa",
        "OrganismNER": "biotech_ml.microbiology.organism_ner",
        "PhenotypePredictor": "biotech_ml.microbiology.phenotype_predictor",
    }
    if name in _map:
        import importlib
        mod = importlib.import_module(_map[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
