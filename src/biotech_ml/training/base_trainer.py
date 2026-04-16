"""Abstract base class for all ML model trainers."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingBundle:
    train_features: Any
    train_targets: Any
    validation_features: Any | None = None
    validation_targets: Any | None = None
    test_features: Any | None = None
    test_targets: Any | None = None
    feature_names: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTrainer(ABC):
    """Base trainer providing a standard train-save pipeline."""

    def __init__(self, data_dir: Path, artifact_dir: Path) -> None:
        self.data_dir = data_dir
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def prepare_data(self) -> TrainingBundle:
        ...

    @abstractmethod
    def train(self, bundle: TrainingBundle) -> dict[str, Any]:
        ...

    @abstractmethod
    def save_artifacts(self) -> None:
        ...

    def run(self) -> dict[str, Any]:
        logger.info("[%s] Preparing data...", self.__class__.__name__)
        bundle = self.prepare_data()

        logger.info("[%s] Training...", self.__class__.__name__)
        metrics = self.train(bundle)

        logger.info("[%s] Saving artifacts to %s...", self.__class__.__name__, self.artifact_dir)
        self.save_artifacts()

        logger.info("[%s] Done. Metrics: %s", self.__class__.__name__, metrics)
        return metrics
