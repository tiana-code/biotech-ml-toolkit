"""Tests for medical domain models."""

import pytest
from pathlib import Path

from biotech_ml.base import ModelNotLoadedError


class TestResultAnomalyDetector:
    def test_normal_results_low_score(self, tmp_path):
        from biotech_ml.medical.anomaly_detector import ResultAnomalyDetector
        model = ResultAnomalyDetector()
        model.load(tmp_path)
        result = model.predict({"results": [{"test_code": "GLU", "value": 90.0}]})
        assert result["anomaly_score"] < 0.5
        assert result["flags"] == []

    def test_high_result_flagged(self, tmp_path):
        from biotech_ml.medical.anomaly_detector import ResultAnomalyDetector
        model = ResultAnomalyDetector()
        model.load(tmp_path)
        result = model.predict({"results": [{"test_code": "GLU", "value": 300.0}]})
        assert len(result["flags"]) == 1
        assert result["flags"][0]["severity"] in ("critical", "high", "medium", "low")

    def test_empty_results(self, tmp_path):
        from biotech_ml.medical.anomaly_detector import ResultAnomalyDetector
        model = ResultAnomalyDetector()
        model.load(tmp_path)
        result = model.predict({"results": []})
        assert result["anomaly_score"] == 0.0

    def test_predict_before_load_raises(self):
        from biotech_ml.medical.anomaly_detector import ResultAnomalyDetector
        model = ResultAnomalyDetector()
        with pytest.raises(ModelNotLoadedError):
            model.predict({"results": [{"test_code": "GLU", "value": 90.0}]})


class TestDeltaChecker:
    def test_no_flag_small_change(self, tmp_path):
        from biotech_ml.medical.delta_check import DeltaChecker
        model = DeltaChecker()
        model.load(tmp_path)
        result = model.predict({
            "test_code": "GLU",
            "current_value": 92.0,
            "previous_value": 90.0,
        })
        assert not result["delta_flag"]
        assert result["severity"] == "low"

    def test_flag_large_change(self, tmp_path):
        from biotech_ml.medical.delta_check import DeltaChecker
        model = DeltaChecker()
        model.load(tmp_path)
        result = model.predict({
            "test_code": "GLU",
            "current_value": 200.0,
            "previous_value": 90.0,
        })
        assert result["delta_flag"]


class TestDrugLabInteraction:
    def test_known_interaction(self, tmp_path):
        from biotech_ml.medical.drug_lab_interaction import DrugLabInteraction
        model = DrugLabInteraction()
        model.load(tmp_path)
        result = model.predict({
            "medications": ["biotin"],
            "test_code": "TSH",
        })
        assert len(result["interactions"]) >= 1
        assert result["interactions"][0]["severity"] == "severe"

    def test_no_interaction(self, tmp_path):
        from biotech_ml.medical.drug_lab_interaction import DrugLabInteraction
        model = DrugLabInteraction()
        model.load(tmp_path)
        result = model.predict({
            "medications": ["aspirin"],
            "test_code": "GLU",
        })
        assert result["interactions"] == []


class TestClinicalQA:
    def test_empty_index_returns_no_index_status(self, tmp_path):
        from biotech_ml.medical.clinical_qa import ClinicalQA
        model = ClinicalQA()
        model.load(tmp_path)
        result = model.predict({"question": "What is hemoglobin?"})
        assert result["answers"] == []
        assert result["status"] == "no_index"


class TestDDxSuggester:
    def test_no_model_returns_status(self, tmp_path):
        from biotech_ml.medical.ddx_suggester import DDxSuggester
        model = DDxSuggester()
        model.load(tmp_path)
        result = model.predict({"symptoms": ["fever"]})
        assert result["diagnoses"] == []
        assert result["status"] == "model_not_trained"


class TestTerminologyMapper:
    def test_no_model_returns_empty(self, tmp_path):
        from biotech_ml.medical.terminology_mapper import TerminologyMapper
        model = TerminologyMapper()
        model.load(tmp_path)
        result = model.predict({"local_name": "blood sugar"})
        assert result["mappings"] == []
