from src.utils import setup_logging
from src.models.model_utils import load_model
from src.quantizers.gguf_quantizer import GGUFQuantizer
from src.quantizers.awq_quantizer import AWQQuantizer

def process_command(args):
    setup_logging(args.verbose)
    
    model_path = load_model(args.model)
    
    if args.method == "gguf":
        quantizer = GGUFQuantizer(
            model_path, 
            args.bits, 
            args.output, 
            args.model.split("/")[-1])
        
    elif args.method == "awq":
        quantizer = AWQQuantizer(
            model_name=args.model,
            bit_width=args.bits,
            group_size=args.group_size,
            version=args.version,
            zero_point=args.zero_point,
            out_path=args.output
        )
    else:
        raise ValueError("Unknown quantization method.")

    
    # Quantizer to be added
    
    quantized_model_path = quantizer.quantize()
    print(f"Quantized model saved to: {quantized_model_path}")