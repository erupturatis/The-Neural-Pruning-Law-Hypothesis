from __future__ import annotations
from abc import ABC, abstractmethod

class NPLHStoppingPolicy(ABC):
    @abstractmethod
    def stop_experiment(self) -> bool:
        pass

class NPLHDensityLimitStoppingPolicy(NPLHStoppingPolicy):
    def __init__(self, density_threshold: float):
        self.density_threshold = density_threshold

    def stop_experiment(self) -> bool:
        raise Exception("NPLHDensityStoppingPolicy Not implemented yet.")

class NPLHEpochLimitingStoppingPolicy(NPLHStoppingPolicy):
    def __init__(self, max_epochs: int):
        self.max_epochs = max_epochs

    def stop_experiment(self) -> bool:
        raise Exception("NPLHEpochLimitingStoppingPolicy Not implemented yet.")

class NPLHAccuracyLimitStoppingPolicy(NPLHStoppingPolicy):
    def __init__(self, accuracy_threshold: float):
        self.accuracy_threshold = accuracy_threshold

    def stop_experiment(self) -> bool:
        raise Exception("NPLHAccuracyLimitStoppingPolicy Not implemented yet.")


