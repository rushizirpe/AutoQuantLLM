# src/quantizers/base_quantizer.py

class BaseQuantizer:
    def __init__(self, model_name, bits, out_path, group_size=None, version=None, zero_point=None, device="cpu"):
        """
        Initialize the BaseQuantizer with common parameters.

        :param model_name: The name or path of the model to quantize.
        :param bits: Bit width for quantization.
        :param out_path: Output path where the quantized model will be saved.
        :param group_size: Group size for quantization (specific to some methods like AWQ).
        :param version: Version for quantization (specific to some methods like AWQ).
        :param zero_point: Whether to use zero point correction (specific to some methods like AWQ).
        """
        self.model_name = model_name
        self.bits = bits
        self.out_path = out_path
        self.group_size = group_size
        self.version = version
        self.zero_point = zero_point
        self.device = device

    def quantize(self):
        raise NotImplementedError("Quantize method must be implemented by subclasses.")
