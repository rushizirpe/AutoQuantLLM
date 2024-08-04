# src/quantizers/base_quantizer.py

class BaseQuantizer:
    def __init__(self, model, bits, out_path, model_name):
        self.model = model
        self.bits = bits
        self.out_path = out_path
        self.model_name = model_name

    def quantize(self):
        raise NotImplementedError("Quantize method must be implemented by subclasses.")
