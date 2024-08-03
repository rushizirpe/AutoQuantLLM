# src/quantizers/awq_quantizer.py

from .base_quantizer import BaseQuantizer

class AWQQuantizer(BaseQuantizer):
    def quantize(self):
        raise NotImplementedError("Quantize method must be implemented by subclasses.")