import time
from contextlib import contextmanager

from src.infrastructure.layers import get_total_and_remaining_params


def get_model_density(model) -> float:
    total, remaining = get_total_and_remaining_params(model)
    return round(remaining / total * 100, 4)


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

