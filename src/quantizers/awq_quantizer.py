# src/quantizers/awq_quantizer.py

import logging
from .base_quantizer import BaseQuantizer
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os
import torch

class AWQQuantizer(BaseQuantizer):
    def __init__(self, model_name, bit_width, group_size, version, zero_point, out_path, device="cpu"):
        super().__init__(model_name, bit_width, out_path, group_size, version, zero_point, device)

    
    def quantize(self):
        try:
            quant_config = {
                "w_bit": self.bits,
                "q_group_size": self.group_size,
                "version": self.version,
                "zero_point": self.zero_point
            }
            
            logging.info(f"Loading tokenizer and model: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoAWQForCausalLM.from_pretrained(self.model_name, safetensors=True, low_cpu_mem_usage=True)
            
            if self.device == "cuda":
                model = model.cuda()

            logging.info("Starting AWQ quantization")
            model.quantize(tokenizer, quant_config=quant_config)
            
            save_folder = os.path.join(self.out_path, f"{self.model_name}-AWQ")
            logging.info(f"Saving quantized model to: {save_folder}")
            model.save_quantized(save_folder)
            tokenizer.save_pretrained(save_folder)

            logging.info("AWQ quantization completed successfully")
            return save_folder
        except Exception as e:
            logging.error(f"Error during AWQ quantization: {str(e)}")
            raise