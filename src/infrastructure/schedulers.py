import numpy as np

from src.infrastructure.configs_general import VERBOSE_SCHEDULER
from src.infrastructure.constants import SCHEDULER_MESSAGE


# ─── Trajectory helpers ───────────────────────────────────────────────────────

def _sigmoid_pruning_factor(epoch, start, end, transition, late_aggressivity):
    s = 1 / (1 + np.exp(-transition * (epoch - late_aggressivity)))
    return start + (end - start) * s

def _cumulative_pruning(epochs_target, start, end, transition, late_aggressivity):
    """Cumulative remaining-params factor after `epochs_target` epochs (starts at 100)."""
    log_sum = sum(
        np.log(_sigmoid_pruning_factor(e, start, end, transition, late_aggressivity))
        for e in range(1, epochs_target + 1)
    )
    return np.exp(np.log(100) + log_sum)


class TrajectoryCalculator:
    """Fits a sigmoid-shaped pruning curve that hits `pruning_target` remaining params at `epochs_target`."""

    END = 0.999  # upper bound on per-epoch retention factor

    def __init__(self, pruning_target, epochs_target, late_aggresivity, aggresivity_transition):
        self.pruning_target = pruning_target
        self.epochs_target = epochs_target
        self.late_aggressivity = late_aggresivity
        self.aggressivity_transition = aggresivity_transition
        self.start = self._fit_start()

    def _fit_start(self):
        lo, hi, best_start, best_val = 0.0, 0.999, None, None
        for _ in range(100):
            mid = (lo + hi) / 2
            val = _cumulative_pruning(self.epochs_target, mid, self.END, self.aggressivity_transition, self.late_aggressivity)
            if best_val is None or abs(val - self.pruning_target) < abs(best_val - self.pruning_target):
                best_start, best_val = mid, val
            if abs(val - self.pruning_target) < 1e-6:
                break
            if val < self.pruning_target:
                lo = mid
            else:
                hi = mid
        return best_start

    def get_expected_pruning_at_epoch(self, epoch: int) -> float:
        return _cumulative_pruning(epoch, self.start, self.END, self.aggressivity_transition, self.late_aggressivity)


# ─── Scheduler 1: Trajectory-following pressure scheduler ────────────────────
#
#   Compares current sparsity against a precomputed sigmoid trajectory.
#   Increases gamma (pressure) when the network is behind the curve,
#   decreases it when ahead. Multiplier = gamma ** EXP.

class TrajectoryScheduler:
    def __init__(self, pressure_exponent: float, sparsity_target: float, epochs_target: int,
                 step_size: float = 0.3, aggressivity: int = 6):
        self.gamma = 0.0
        self.step_size = step_size
        self.epochs_target = epochs_target
        self.EXP = pressure_exponent
        self.inertia_pos = 0.0
        self.inertia_neg = 0.0

        self.trajectory = TrajectoryCalculator(
            pruning_target=100 - sparsity_target,
            epochs_target=epochs_target,
            late_aggresivity=epochs_target / 2,
            aggresivity_transition=aggressivity / epochs_target,
        )

    def step(self, epoch: int, current_remaining: float) -> None:
        if epoch >= self.epochs_target:
            return
        expected = self.trajectory.get_expected_pruning_at_epoch(epoch)
        if VERBOSE_SCHEDULER:
            print(SCHEDULER_MESSAGE + f"remaining={current_remaining:.2f}  expected={expected:.2f}")
        if current_remaining > expected:
            # behind curve — increase pressure
            self.gamma += self.step_size + self.inertia_pos
            self.inertia_pos += self.step_size / 4
            self.inertia_neg = 0.0
            if VERBOSE_SCHEDULER:
                print(SCHEDULER_MESSAGE + f"↑ pressure  gamma={self.gamma:.3f}")
        else:
            # ahead of curve — ease pressure
            self.gamma -= self.step_size + self.inertia_neg
            self.inertia_neg += self.step_size / 4
            self.inertia_pos = 0.0
            self.gamma = max(self.gamma, 0.0)
            if VERBOSE_SCHEDULER:
                print(SCHEDULER_MESSAGE + f"↓ pressure  gamma={self.gamma:.3f}")

    def get_multiplier(self) -> float:
        return self.gamma ** self.EXP


# ─── Scheduler 2: Upper-bound pressure scheduler ─────────────────────────────
#
#   At each step, fits a fresh trajectory from the current state to the target,
#   giving the expected per-epoch pruning rate. If the network is pruning slower
#   than that upper-bound rate, pressure increases; otherwise it eases.
#   Multiplier = baseline ** EXP.

class UpperBoundScheduler:
    def __init__(self, pressure_exponent: float, sparsity_target: float, epochs_target: int, step_size: float):
        self.baseline = 0.0
        self.EXP = pressure_exponent
        self.pruning_target = 100 - sparsity_target
        self.epochs_target = epochs_target
        self.step_size = step_size
        self.streak = 0.0
        self.history: list[float] = []

    def _actual_decrease_rate(self) -> float:
        """Rolling average of the per-epoch retention ratio (lower = faster pruning)."""
        h = self.history
        if len(h) < 2:
            return -1
        if len(h) == 2:
            return h[-1] / h[-2]
        return ((h[-1] / h[-2]) + (h[-2] / h[-3])) / 2

    def _upper_bound_rate(self) -> float:
        """Expected per-epoch retention rate from a fresh trajectory fit."""
        remaining = self.epochs_target - len(self.history)
        if remaining < 1:
            return 1.0
        current = self.history[-1]
        traj = TrajectoryCalculator(
            pruning_target=self.pruning_target / current * 100,
            epochs_target=remaining,
            late_aggresivity=remaining / 2,
            aggresivity_transition=6 / remaining,
        )
        return traj.get_expected_pruning_at_epoch(1) / 100

    def step(self, epoch: int, current_remaining: float) -> None:
        self.history.append(current_remaining)
        if self.epochs_target - len(self.history) <= 0:
            self.baseline = 0.0
            return
        actual = self._actual_decrease_rate()
        if actual == -1:
            return
        upper_bound = self._upper_bound_rate()
        if actual > upper_bound:
            # pruning too slowly — increase pressure
            self.baseline += self.step_size + self.streak
            self.streak += self.step_size * 0.1
        else:
            # within bound — ease pressure
            self.baseline -= self.step_size / 2
            self.streak = 0.0
            self.baseline = max(self.baseline, 0.0)

    def get_multiplier(self) -> float:
        return self.baseline ** self.EXP
