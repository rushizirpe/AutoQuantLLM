# src/quantizers/base_quantizer.py

class BaseQuantizer:
    def __init__(self, model, bit_width, out_path, model_name):
        self.model = model
        self.bit_width = bit_width
        self.out_path = out_path
        self.model_name = model_name

    def quantize(self):
        raise NotImplementedError("Quantize method must be implemented by subclasses.")
