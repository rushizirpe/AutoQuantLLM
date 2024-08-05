# src/quantizers/awq_quantizer.py

from .base_quantizer import BaseQuantizer
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch
import os

class AWQQuantizer(BaseQuantizer):
    def __init__(self, model, bit_width, out_path, model_name, quant_config=None):
        super().__init__(model, bit_width, out_path, model_name)
        self.quant_config = quant_config or {}

    def quantize(self):
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        model = AutoAWQForCausalLM.from_pretrained(self.model, **self.quant_config)

        # Prepare for quantization
        model.quantize(
            tokenizer,
            quant_bit=self.bit_width,
            **self.quant_config
        )

        # Save the quantized model
        out_path = os.path.join(self.out_path, f"{self.model_name}_awq_{self.bit_width}bit")
        model.save_quantized(out_path)
        tokenizer.save_pretrained(out_path)

        return out_path