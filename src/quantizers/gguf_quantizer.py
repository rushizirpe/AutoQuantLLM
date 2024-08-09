# src/quantizers/gguf_quantizer.py

import logging
from .base_quantizer import BaseQuantizer
import os
import sys
from pathlib import Path
import torch

class GGUFQuantizer(BaseQuantizer):
    def __init__(self, model_name, bits, out_path, model_basename, device="cpu"):
        super().__init__(model_name, bits, out_path, device=device)
        self.model_basename = model_basename

    def _get_quantization_type(self):
        if self.bits == 8:
            return "q8_0"
        elif self.bits == 16:
            return "f16"
        elif self.bits == 32:
            return "f32"
        else:
            raise ValueError(f"Unsupported bit size: {self.bits}. Supported sizes are 8, 16, and 32.")

    def quantize(self):
        try:
            quantization_type = self._get_quantization_type()
            out_path = Path(f"{self.out_path}/{self.model_basename}")
            out_path.mkdir(parents=True, exist_ok=True)

            llama_cpp_path = Path("llama.cpp")
            if not llama_cpp_path.exists():
                self._setup_llama_cpp()

            sys.path.append(str(llama_cpp_path.resolve()))
            from convert_hf_to_gguf import main as convert_hf_to_gguf

            outfile = out_path / f"{self.model_basename}.{quantization_type}.gguf"
            
            original_argv = sys.argv
            sys.argv = [
                "convert_hf_to_gguf.py",
                "--outfile", str(outfile),
                "--outtype", quantization_type,
                self.model_name
            ]

            if self.device == "cuda":
                torch.cuda.set_device(0)  # Set CUDA device if available

            convert_hf_to_gguf()
            sys.argv = original_argv

            logging.info(f"GGUF quantization completed. Model saved to: {outfile}")
            return str(out_path)
        except Exception as e:
            logging.error(f"Error during GGUF quantization: {str(e)}")
            raise