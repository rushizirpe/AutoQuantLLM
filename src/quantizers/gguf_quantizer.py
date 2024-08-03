# src/quantizers/gguf_quantizer.py

from .base_quantizer import BaseQuantizer
import subprocess
import os

class GGUFQuantizer(BaseQuantizer):
    def quantize(self):
        out_path = None
        if not os.path.exists("llama.cpp/"):
            subprocess.run(['git', "clone", "-q", "https://github.com/ggerganov/llama.cpp.git"])
            subprocess.run(["pip", "install", "-r", "llama.cpp/requirements.txt"])
            
        out_path = f"{self.out_path}/{self.model_name}"
        
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
        # print("OUTPATH:", out_path)
        subprocess.run(['python', 'llama.cpp/convert_hf_to_gguf.py', self.model, '--outfile', out_path,'--outtype', 'bf16'])
        
        # quantized_model_path = self.model + ".bf16.gguf"
        
        # assert True
        return out_path
