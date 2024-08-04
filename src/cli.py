from src.utils import setup_logging
from src.models.model_utils import load_model
from src.quantizers.gguf_quantizer import GGUFQuantizer
# Import other quantizers as needed

def process_command(args):
    setup_logging(args.verbose)
    
    model_path = load_model(args.model)
    
    if args.method == "gguf":
        quantizer = GGUFQuantizer(model_path, args.bits, args.output, args.model.split("/")[-1])
    # Add elif blocks for other quantization methods
    
    quantized_model_path = quantizer.quantize()
    print(f"Quantized model saved to: {quantized_model_path}")