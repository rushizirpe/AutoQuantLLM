# src/quantizers/static_quantizer.py

from .base_quantizer import BaseQuantizer

class StaticQuantizer(BaseQuantizer):
    def quantize(self):
        raise NotImplementedError("Quantize method must be implemented by subclasses.")
