from __future__ import annotations
from abc import ABC, abstractmethod

from src.infrastructure.layers import set_mask_apply_all, set_mask_training_all, set_weights_training_all
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
        set_mask_apply_all(ctx.model, True)
        set_mask_training_all(ctx.model, False)
        set_weights_training_all(ctx.model, True)
        for epoch in range(1, self.epochs + 1):
            ctx.train_one_epoch()
            print(f"    epoch {epoch}/{self.epochs}")
        acc, _ = ctx.evaluate()
        return acc

class UntilConvergencePolicy(TrainingConvergencePolicy):
    # uses: ctx.train_one_epoch, ctx.evaluate
    def __init__(self, window: int, tol: float, max_epochs: int):
        self.window = window        # Acts as "patience" (epochs to wait)
        self.tol = tol              # Minimum meaningful improvement
        self.max_epochs = max_epochs

    def train_until_convergence(self, ctx: TrainingContext) -> float:
        best_acc = 0.0
        epochs_without_improvement = 0
        last_acc = 0.0

        for epoch in range(1, self.max_epochs + 1):
            ctx.train_one_epoch()
            acc, _ = ctx.evaluate()
            last_acc = acc

            # Check if current accuracy beats the best recorded accuracy by at least the tolerance
            if acc >= best_acc + self.tol:
                best_acc = acc
                epochs_without_improvement = 0
                print(f"    epoch {epoch}/{self.max_epochs}  acc={acc:.4f}  (New Best!)")
            else:
                epochs_without_improvement += 1
                print(f"    epoch {epoch}/{self.max_epochs}  acc={acc:.4f}  (Patience: {epochs_without_improvement}/{self.window})")

            # Trigger convergence if the window of patience is exhausted
            if epochs_without_improvement >= self.window:
                print(f"    [Convergence] Reached plateau. No improvement > {self.tol}% for {self.window} epochs.")
                return best_acc

        print(f"    [Convergence] Hit maximum limit of {self.max_epochs} epochs.")
        return last_acc
