# src/quantizers/dynamic_quantizer.py

from .base_quantizer import BaseQuantizer

class DynamicQuantizer(BaseQuantizer):
    def quantize(self):
        raise NotImplementedError("Quantize method must be implemented by subclasses.")
