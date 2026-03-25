from __future__ import annotations
from abc import ABC, abstractmethod

class TrainingConvergencePolicy(ABC):
    @abstractmethod
    def train_until_convergence(self) -> None:
        pass

class FixedEpochsConvergencePolicy(TrainingConvergencePolicy):
    def __init__(self, epochs: int):
        self.epochs = epochs

    def train_until_convergence(self) -> None:
        raise Exception("FixedEpochsConvergencePolicy Not implemented yet.")

class UntilConvergencePolicy(TrainingConvergencePolicy):
    def __init__(self, window: int, tol: float, max_epochs: int):
        self.window = window
        self.tol = tol
        self.max_epochs = max_epochs

    def train_until_convergence(self) -> None:
        raise Exception("UntilConvergencePolicy Not implemented yet.")
