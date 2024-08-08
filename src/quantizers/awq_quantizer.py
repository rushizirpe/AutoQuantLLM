# src/quantizers/awq_quantizer.py

from .base_quantizer import BaseQuantizer
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os

class AWQQuantizer(BaseQuantizer):
    def __init__(self, model_name, bit_width, group_size, version, zero_point, out_path):
        super().__init__(model_name, bit_width, out_path)
        self.group_size = group_size
        self.version = version
        self.zero_point = zero_point

    def quantize(self):
        # Define the quantization configuration
        quant_config = {
            "w_bit": self.bits,
            "q_group_size": self.group_size,
            "version": self.version,
            "zero_point": self.zero_point
        }
        
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoAWQForCausalLM.from_pretrained(self.model_name, safetensors=True, low_cpu_mem_usage=True)
        
        # Quantize the model
        model.quantize(tokenizer, quant_config=quant_config)
        
        # Save the quantized model and tokenizer
        save_folder = os.path.join(self.out_path, f"{self.model_name}-AWQ")
        model.save_quantized(save_folder)
        tokenizer.save_pretrained(save_folder)

        return save_folder
