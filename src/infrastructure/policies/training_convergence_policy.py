from __future__ import annotations
from abc import ABC, abstractmethod
from src.infrastructure.training_context import TrainingContext


class TrainingConvergencePolicy(ABC):
    @abstractmethod
    def train_until_convergence(self, ctx: TrainingContext) -> float:
        pass


class FixedEpochsConvergencePolicy(TrainingConvergencePolicy):
    # uses: ctx.train_one_epoch, ctx.evaluate
    def __init__(self, epochs: int):
        self.epochs = epochs

    def train_until_convergence(self, ctx: TrainingContext) -> float:
        for epoch in range(1, self.epochs + 1):
            ctx.train_one_epoch()
            print(f"    epoch {epoch}/{self.epochs}")
        acc = ctx.evaluate()
        return acc

class UntilConvergencePolicy(TrainingConvergencePolicy):
    # uses: ctx.train_one_epoch, ctx.evaluate
    def __init__(self, window: int, tol: float, max_epochs: int):
        self.window = window
        self.tol = tol
        self.max_epochs = max_epochs

    def train_until_convergence(self, ctx: TrainingContext) -> float:
        acc_history = []
        for epoch in range(1, self.max_epochs + 1):
            ctx.train_one_epoch()
            acc = ctx.evaluate()
            acc_history.append(acc)
            print(f"    epoch {epoch}/{self.max_epochs}  acc={acc:.4f}")
            if len(acc_history) >= self.window:
                recent = acc_history[-self.window:]
                if max(recent) - min(recent) < self.tol:
                    return acc
        return acc_history[-1] if acc_history else 0.0
