"""Tests for microbiology domain models."""

import pytest


class TestASTPredictor:
    def test_model_id(self):
        from biotech_ml.microbiology.ast_predictor import ASTPredictor
        model = ASTPredictor()
        assert model.model_id == "microbiology.ast_predictor"

    def test_metadata(self):
        from biotech_ml.microbiology.ast_predictor import ASTPredictor
        model = ASTPredictor()
        meta = model.metadata()
        assert "organisms" in meta
        assert "antibiotics" in meta
        assert len(meta["organisms"]) == 8


class TestMICRegressor:
    def test_snap_to_dilution(self):
        from biotech_ml.microbiology.mic_regressor import _snap_to_dilution
        assert _snap_to_dilution(1.5) == 1.0 or _snap_to_dilution(1.5) == 2.0
        assert _snap_to_dilution(0.1) == 0.25
        assert _snap_to_dilution(300.0) == 256.0

    def test_format_dilution(self):
        from biotech_ml.microbiology.mic_regressor import _format_dilution
        assert _format_dilution(0.1) == "<=0.25"
        assert _format_dilution(300.0) == ">=256"
        assert _format_dilution(4.0) == "4"

    def test_interpret_mic_susceptible(self):
        from biotech_ml.microbiology.mic_regressor import _interpret_mic, DEFAULT_BREAKPOINTS
        result, source = _interpret_mic(0.25, "escherichia_coli", "ciprofloxacin", DEFAULT_BREAKPOINTS)
        assert result == "S"
        assert source == "EUCAST"

    def test_interpret_mic_resistant(self):
        from biotech_ml.microbiology.mic_regressor import _interpret_mic, DEFAULT_BREAKPOINTS
        result, source = _interpret_mic(16.0, "escherichia_coli", "meropenem", DEFAULT_BREAKPOINTS)
        assert result == "R"


class TestPhenotypePredictor:
    def test_model_id(self):
        from biotech_ml.microbiology.phenotype_predictor import PhenotypePredictor
        model = PhenotypePredictor()
        assert model.model_id == "microbiology.phenotype_predictor"

    def test_trait_categories(self):
        from biotech_ml.microbiology.phenotype_predictor import TRAIT_CATEGORIES
        assert TRAIT_CATEGORIES["esbl"] == "resistance"
        assert TRAIT_CATEGORIES["catalase"] == "metabolism"
        assert TRAIT_CATEGORIES["motility"] == "virulence"


class TestOrganismNER:
    def test_taxonomy_lookup(self):
        from biotech_ml.microbiology.organism_ner import _lookup_taxonomy
        result = _lookup_taxonomy("Escherichia coli")
        assert result is not None
        assert result["genus"] == "Escherichia"
        assert result["gram_type"] == "negative"
        assert result["species"] == "coli"

    def test_taxonomy_lookup_unknown(self):
        from biotech_ml.microbiology.organism_ner import _lookup_taxonomy
        result = _lookup_taxonomy("Unknown species")
        assert result is None

    def test_acronym_map(self):
        from biotech_ml.microbiology.organism_ner import ACRONYM_MAP
        assert ACRONYM_MAP["MRSA"] == "Staphylococcus aureus"
        assert ACRONYM_MAP["VRE"] == "Enterococcus faecium"


class TestMicrobiologyQA:
    def test_model_id(self):
        from biotech_ml.microbiology.microbiology_qa import MicrobiologyQA
        model = MicrobiologyQA()
        assert model.model_id == "microbiology.microbiology_qa"
