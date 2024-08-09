# src/main.py

import argparse
from src.cli import process_command
from src.utils import get_device

def main():
    parser = argparse.ArgumentParser(description="AutoQuantLLM: Automated model quantization")
    parser.add_argument("--model", required=True, help="Model ID or path")
    parser.add_argument("--method", required=True, choices=["gguf", "awq", "dynamic", "static", "weight_only"], help="Quantization method")
    parser.add_argument("--bits", type=int, required=True, help="Bit width for quantization")
    parser.add_argument("--group_size", type=int, help="Group size for AWQ quantization")
    parser.add_argument("--version", choices=["GEMM", "GEMV"], help="Version for AWQ quantization")
    parser.add_argument("--zero_point", type=bool, help="Zero point for AWQ quantization")
    parser.add_argument("--output", required=True, help="Output path for quantized model")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="Device to use for quantization")
    parser.add_argument("--upload", action="store_true", help="Upload quantized model to Hugging Face")
    parser.add_argument("--hf_repo_name", help="Hugging Face repository name for upload")

    args = parser.parse_args()
    args.device = get_device(args.device)
    process_command(args)

if __name__ == "__main__":
    main()