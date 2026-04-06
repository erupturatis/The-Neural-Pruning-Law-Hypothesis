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
    def __init__(self, epochs: int, lr: float | None = None):
        self.epochs = epochs
        self.lr = lr

    def train_until_convergence(self, ctx: TrainingContext) -> float:
        if self.lr is not None:
            for pg in ctx.optimizer.param_groups:
                pg['lr'] = self.lr
        set_mask_apply_all(ctx.model, True)
        set_mask_training_all(ctx.model, False)
        set_weights_training_all(ctx.model, True)
        for epoch in range(1, self.epochs + 1):
            ctx.train_one_epoch()
            print(f"    epoch {epoch}/{self.epochs}  lr={ctx.optimizer.param_groups[0]['lr']:.2e}")
        acc, _ = ctx.evaluate()
        return acc


class UntilConvergencePolicy(TrainingConvergencePolicy):
    """
    Multi-phase convergence with stepped LR decay.

    Each phase trains until training loss stops improving: if `window`
    consecutive epochs pass without the loss dropping below the best seen so
    far in that phase, the LR is divided by 10 and the next phase begins.
    After `max_lr_steps` step-downs the final phase runs to the same patience
    criterion and training stops — giving (max_lr_steps + 1) phases in total.

    Example with max_lr_steps=3, initial_lr=1e-3:
        Phase 1 — LR=1e-3  →  no improvement for `window` epochs  →  step
        Phase 2 — LR=1e-4  →  no improvement for `window` epochs  →  step
        Phase 3 — LR=1e-5  →  no improvement for `window` epochs  →  step
        Phase 4 — LR=1e-6  →  no improvement for `window` epochs  →  done

    The total budget across all phases is capped at max_epochs.
    Accuracy is evaluated and printed every epoch but does not drive stopping.

    If initial_lr is None, the optimizer LR is not touched (backward compatible
    with callers that manage LR externally).
    """

    def __init__(self, window: int, max_epochs: int,
                 initial_lr: float | None = None,
                 max_lr_steps: int = 3,
                 rel_tol: float = 1e-4):
        self.window       = window
        self.max_epochs   = max_epochs
        self.initial_lr   = initial_lr
        self.max_lr_steps = max_lr_steps
        self.rel_tol      = rel_tol

    def train_until_convergence(self, ctx: TrainingContext) -> float:
        # Reset LR to initial value so every call starts from the same point.
        if self.initial_lr is not None:
            for pg in ctx.optimizer.param_groups:
                pg['lr'] = self.initial_lr

        steps_remaining = self.max_lr_steps
        last_acc     = 0.0
        phase        = 1
        total_phases = self.max_lr_steps + 1
        best_loss    = float('inf')
        no_improve   = 0

        for epoch in range(1, self.max_epochs + 1):
            ctx.train_one_epoch()

            acc, _        = ctx.evaluate()
            _, train_loss = ctx.evaluate_train()
            last_acc   = acc
            current_lr = ctx.optimizer.param_groups[0]['lr']

            if train_loss < best_loss * (1 - self.rel_tol):
                best_loss  = train_loss
                no_improve = 0
            else:
                no_improve += 1

            print(f"    epoch {epoch}/{self.max_epochs}"
                  f"  phase={phase}/{total_phases}"
                  f"  acc={acc:.4f}  train_loss={train_loss:.10f}"
                  f"  best={best_loss:.10f}  no_improve={no_improve}/{self.window}"
                  f"  lr={current_lr:.2e}")

            if no_improve >= self.window:
                if steps_remaining > 0:
                    for pg in ctx.optimizer.param_groups:
                        pg['lr'] /= 10.0
                    steps_remaining -= 1
                    phase += 1
                    best_loss  = float('inf')
                    no_improve = 0
                    new_lr = ctx.optimizer.param_groups[0]['lr']
                    print(f"    [LR Step] no improvement for {self.window} epochs"
                          f" — LR {current_lr:.2e} → {new_lr:.2e}"
                          f"  (entering phase {phase}/{total_phases})")
                else:
                    print(f"    [Convergence] All {total_phases} phases complete"
                          f" — fully converged at LR={current_lr:.2e}.")
                    return last_acc

        print(f"    [Convergence] Hit maximum of {self.max_epochs} epochs.")
        return last_acc
