import argparse
from src.cli import process_command

def main():
    parser = argparse.ArgumentParser(description="AutoQuantLLM: Automated model quantization")
    parser.add_argument("--model", required=True, help="Model ID or path")
    parser.add_argument("--method", required=True, choices=["gguf", "awq", "dynamic", "static", "weight_only"], help="Quantization method")
    parser.add_argument("--bits", type=int, required=True, help="Bit width for quantization")
    parser.add_argument("--group_size", type=int, help="Group size for AWQ quantization")  # For AWQ
    parser.add_argument("--version", choices=["GEMM", "GEMV"], help="Version for AWQ quantization")  # For AWQ
    parser.add_argument("--zero_point", type=bool, help="Zero point for AWQ quantization")  # For AWQ
    parser.add_argument("--output", required=True, help="Output path for quantized model")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    process_command(args)

if __name__ == "__main__":
    main()
