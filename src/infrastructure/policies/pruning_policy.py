from __future__ import annotations
from abc import ABC, abstractmethod

class PruningPolicy(ABC):
    @abstractmethod
    def apply_pruning(self) -> None:
        pass

# ---------------------------------------------------------------------------
# Policy 1: Magnitude (standard IMP)
# ---------------------------------------------------------------------------

class MagnitudePruningPolicy(PruningPolicy):
    """
    Prune the fraction of active weights with the smallest absolute magnitude.
    No data required.  Equivalent to ``prune_model_globally`` in layers.py.
    """

    def apply_pruning(
        self
    ) -> None:
        raise Exception("Magnitude pruning Not implemented yet.")

# ---------------------------------------------------------------------------
# Policy 2: Taylor expansion  (weight-level, unstructured)
# ---------------------------------------------------------------------------

class TaylorPruningPolicy(PruningPolicy):
    """
    First-order Taylor criterion.
        score(w_ij) = |w_ij · ∂L/∂w_ij|

    Estimates the absolute change in loss if weight w_ij were zeroed.
    Prune the lowest-scoring active weights.
    Requires one data batch and a loss criterion.
    """

    def apply_pruning(
        self
    ) -> None:
        raise Exception("Taylor pruning Not implemented yet.")

# ---------------------------------------------------------------------------
# Policy 3: Gradient magnitude  (weight-level, unstructured)
# ---------------------------------------------------------------------------

class GradientPruningPolicy(PruningPolicy):
    def apply_pruning(
        self,
    ) -> None:
        raise Exception("Gradient pruning Not implemented yet.")

# ---------------------------------------------------------------------------
# Policy 4: Hyperflux (L0 regularization based pruning)
# ---------------------------------------------------------------------------

class HyperfluxPruningPolicy(PruningPolicy):

    def apply_pruning(
        self
    ) -> None:
        raise Exception("Hyperflux pruning Not implemented yet.")

# ---------------------------------------------------------------------------
# Policy 5: Pure random pruning (weight-level, unstructured)
# ---------------------------------------------------------------------------

class RandomPruningPolicy(PruningPolicy):

    def apply_pruning(
        self,
    ) -> None:
        raise Exception("Random pruning Not implemented yet.")

# ---------------------------------------------------------------------------
# Policy 6: Hessian diagonal (weight-level, unstructured)
# ---------------------------------------------------------------------------

class HessianPruningPolicy(PruningPolicy):

    def apply_pruning(
        self
    ) -> None:
        raise Exception("Hessian pruning Not implemented yet.")