# src/quantizers/gguf_quantizer.py

from .base_quantizer import BaseQuantizer
import os
import sys
import git
from pathlib import Path

class GGUFQuantizer(BaseQuantizer):
    def quantize(self):
        out_path = None
        llama_cpp_path = Path("llama.cpp")

        if not llama_cpp_path.exists():
            git.Repo.clone_from("https://github.com/ggerganov/llama.cpp.git", llama_cpp_path, quiet=True)
            
            # Install requirements
            requirements_path = llama_cpp_path / "requirements.txt"
            if requirements_path.exists():
                import pip
                pip.main(["install", "-r", str(requirements_path)])

        # Add llama.cpp to sys.path
        sys.path.append(str(llama_cpp_path))

        # Import the convert function
        from convert import convert_hf_to_gguf

        out_path = Path(self.out_path) / self.model_name
        out_path.mkdir(parents=True, exist_ok=True)

        output_file = out_path / f"{self.model_name}.bf16.gguf"

        # Call the convert function
        convert_hf_to_gguf(
            model_path=self.model,
            output_path=str(output_file),
            output_type="bf16"
        )

        return str(out_path)