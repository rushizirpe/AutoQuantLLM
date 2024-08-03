# src/cli.py

import argparse
from .models.model_utils import load_model
from .quantizers.dynamic_quantizer import DynamicQuantizer
from .quantizers.static_quantizer import StaticQuantizer
from .quantizers.weight_only_quantizer import WeightOnlyQuantizer
from .quantizers.gguf_quantizer import GGUFQuantizer
from .quantizers.awq_quantizer import AWQQuantizer
from .hf_utils import download_model, upload_model
from .utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="AutoQuant Command Line Interface")
    parser.add_argument('--model', type=str, required=True, help='Hugging Face model identifier')
    parser.add_argument('--method', type=str, required=True, choices=['dynamic', 'static', 'weight-only', 'gguf', 'awq'], help='Quantization method')
    parser.add_argument('--bits', type=int, required=True, help='Bit-width for quantization')
    parser.add_argument('--output', type=str, required=True, help='Output path for the quantized model')
    parser.add_argument('--upload', action='store_true', help='Upload the quantized model to Hugging Face')
    parser.add_argument('--repo-name', type=str, help='Hugging Face repository name for upload')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging(args.verbose)

    model = download_model(args.model)

    quantizer_class = {
        'dynamic': DynamicQuantizer,
        'static': StaticQuantizer,
        'weight-only': WeightOnlyQuantizer,
        'gguf': GGUFQuantizer,
        'awq': AWQQuantizer
    }[args.method]
    
    quantizer = quantizer_class(model, args.bits, args.output, args.model)
    quantized_model = quantizer.quantize()
    if quantized_model:
        print("Model Saved Successfully to:", quantized_model)
    # quantized_model.save_pretrained(args.output)

    if args.upload:
        upload_model(args.model, args.repo_name, args.output)

if __name__ == "__main__":
    main()
