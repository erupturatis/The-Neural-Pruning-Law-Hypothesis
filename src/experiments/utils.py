import time
from contextlib import contextmanager

from src.infrastructure.layers import get_total_and_remaining_params, get_prunable_layers, get_layer_name
from src.infrastructure.constants import MASK_ATTR


def get_model_density(model) -> float:
    total, remaining = get_total_and_remaining_params(model)
    return round(remaining / total * 100, 4)


def log_layer_densities(model) -> None:
    """Print per-layer remaining weight counts and density after a pruning step."""
    print("  [LayerDensities]")
    for layer in get_prunable_layers(model):
        mask = getattr(layer, MASK_ATTR).data
        n_active = int((mask >= 0).sum().item())
        n_total = mask.numel()
        pct = n_active / n_total * 100
        name = get_layer_name(model, layer)
        layer_type = type(layer).__name__
        collapsed = "  *** COLLAPSED ***" if n_active == 0 else ""
        print(f"    {name:55s}  [{layer_type}]  {n_active:8d}/{n_total:8d}  ({pct:6.2f}%){collapsed}")


class _Elapsed:
    """Holds elapsed seconds after the with-block exits; formats as '1.2s'."""
    __slots__ = ("s",)
    def __init__(self): self.s = 0.0
    def __str__(self): return f"{self.s:.1f}s"


@contextmanager
def timed():
    """
    Measure wall-clock duration of a block.  Use as:

        with timed() as t:
            do_something()
        print(f"done  ({t})")   # prints e.g. "done  (3.7s)"
    """
    e = _Elapsed()
    start = time.perf_counter()
    yield e
    e.s = time.perf_counter() - start

