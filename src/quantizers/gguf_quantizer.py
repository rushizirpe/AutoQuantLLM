# src/quantizers/gguf_quantizer.py

from .base_quantizer import BaseQuantizer
import os
import sys
from pathlib import Path

class GGUFQuantizer(BaseQuantizer):
    
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
        quantization_type = self._get_quantization_type()
        out_path = Path(f"{self.out_path}/{self.model_name}")
        out_path.mkdir(parents=True, exist_ok=True)

        # Ensure llama.cpp is available
        llama_cpp_path = Path("llama.cpp")
        if not llama_cpp_path.exists():
            self._setup_llama_cpp()

        sys.path.append(str(llama_cpp_path.resolve()))
        from convert_hf_to_gguf import main as convert_hf_to_gguf

        outfile = out_path / f"{self.model_name}.{quantization_type}.gguf"
        
        original_argv = sys.argv
        sys.argv = [
            "convert_hf_to_gguf.py",
            "--outfile", str(outfile),
            "--outtype", quantization_type,
            self.model
        ]

        # Run the conversion and restore the original sys.argv
        try:
            convert_hf_to_gguf()
        finally:
            sys.argv = original_argv

        return str(out_path)

    def _setup_llama_cpp(self):
        import git
        import pip
        
        git.Repo.clone_from("https://github.com/ggerganov/llama.cpp.git", "llama.cpp")
        requirements_path = Path("llama.cpp/requirements.txt")
        if requirements_path.exists():
            pip.main(["install", "-r", str(requirements_path)])