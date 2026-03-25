from __future__ import annotations
from abc import ABC, abstractmethod
from src.infrastructure.training_context import TrainingContext
from src.experiments.utils import get_model_density


class NPLHStoppingPolicy(ABC):
    @abstractmethod
    def stop_experiment(self, ctx: TrainingContext) -> bool:
        pass


class NPLHDensityLimitStoppingPolicy(NPLHStoppingPolicy):
    # uses: ctx.model  (reads current mask density)
    def __init__(self, density_threshold: float):
        self.density_threshold = density_threshold

    def stop_experiment(self, ctx: TrainingContext) -> bool:
        return get_model_density(ctx.model) <= self.density_threshold

class NPLHAccuracyLimitStoppingPolicy(NPLHStoppingPolicy):
    # uses: ctx.evaluate
    def __init__(self, accuracy_threshold: float):
        self.accuracy_threshold = accuracy_threshold

    def stop_experiment(self, ctx: TrainingContext) -> bool:
        return ctx.evaluate() <= self.accuracy_threshold

class NPLHEpochLimitingStoppingPolicy(NPLHStoppingPolicy):
    # uses: ctx.epoch_count — reads cumulative train_one_epoch call count
    def __init__(self, max_epochs: int):
        self.max_epochs = max_epochs

    def stop_experiment(self, ctx: TrainingContext) -> bool:
        return ctx.epoch_count >= self.max_epochs
