"""Typed schemas for biotech-ml-toolkit inputs, outputs, and artifacts."""

from typing import Any, Literal, TypedDict


class LabResult(TypedDict, total=False):
    test_code: str
    value: float
    unit: str
    specimen: str
    reference_source: str


class PatientContext(TypedDict, total=False):
    age_years: int
    sex: Literal["male", "female", "unknown"]
    pregnant: bool


class ArtifactManifest(TypedDict, total=False):
    model_id: str
    version: str
    created_at: str
    feature_names: list[str]
    package_versions: dict[str, str]
    training_config: dict[str, Any]
    data_snapshot_id: str
    schema_version: str
