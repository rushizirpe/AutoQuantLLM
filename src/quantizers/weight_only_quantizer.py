# src/quantizers/weight_only_quantizer.py

from .base_quantizer import BaseQuantizer

class WeightOnlyQuantizer(BaseQuantizer):
    def quantize(self):
        raise NotImplementedError("Quantize method must be implemented by subclasses.")
