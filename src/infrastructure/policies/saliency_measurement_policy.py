from __future__ import annotations
from abc import ABC, abstractmethod

class SaliencyMeasurementPolicy(ABC):
    @abstractmethod
    def measure_saliency(self) -> None:
        pass

class MagnitudeSaliencyMeasurementPolicy(SaliencyMeasurementPolicy):
    def measure_saliency(self) -> None:
        raise Exception("MagnitudeSaliencyMeasurementPolicy Not implemented yet.")

class TaylorSaliencyMeasurementPolicy(SaliencyMeasurementPolicy):
    def measure_saliency(self) -> None:
        raise Exception("TaylorSaliencyMeasurementPolicy Not implemented yet.")

class HessianSaliencyMeasurementPolicy(SaliencyMeasurementPolicy):
    def measure_saliency(self) -> None:
        raise Exception("HessianSaliencyMeasurementPolicy Not implemented yet.")


