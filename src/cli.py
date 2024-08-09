# src/cli.py

import logging
import asyncio
from src.utils import setup_logging, get_api_key
from src.models.model_utils import load_model
from src.quantizers.gguf_quantizer import GGUFQuantizer
from src.quantizers.awq_quantizer import AWQQuantizer
from src.hf_utils import upload_model, check_repo_exists, create_repo_if_not_exists

async def ask_user(question, timeout=10):
    print(question)
    try:
        return await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(None, input), timeout)
    except asyncio.TimeoutError:
        return None

def validate_args(args):
    if args.upload and not args.hf_repo_name:
        raise ValueError("Please provide --hf_repo_name for uploading to Hugging Face")

    if args.upload:
        hf_token = get_api_key("HUGGINGFACE_TOKEN")
        repo_exists = check_repo_exists(args.hf_repo_name, hf_token)
        
        if not repo_exists:
            response = asyncio.get_event_loop().run_until_complete(
                ask_user(f"Repository {args.hf_repo_name} not found. Create it? (y/n): ", timeout=10)
            )
            
            if response is None or response.lower() != 'n':
                if not create_repo_if_not_exists(args.hf_repo_name, hf_token):
                    raise ValueError(f"Failed to create repository {args.hf_repo_name}.")
            else:
                raise ValueError(f"Repository {args.hf_repo_name} does not exist and user chose not to create it.")

    if args.method not in ["gguf", "awq"]:
        raise ValueError(f"Unknown quantization method: {args.method}")

    # You can add more validations here as needed

def process_command(args):
    try:
        setup_logging(args.verbose)
        
        # Perform all validations upfront
        validate_args(args)
        
        logging.info(f"Loading model: {args.model}")
        model_path = load_model(args.model)
        
        if args.method == "gguf":
            quantizer = GGUFQuantizer(
                model_path, 
                args.bits, 
                args.output, 
                args.model.split("/")[-1],
                device=args.device
            )
        elif args.method == "awq":
            quantizer = AWQQuantizer(
                model_name=args.model,
                bit_width=args.bits,
                group_size=args.group_size,
                version=args.version,
                zero_point=args.zero_point,
                out_path=args.output,
                device=args.device
            )

        logging.info(f"Starting quantization with method: {args.method}")
        quantized_model_path = quantizer.quantize()
        logging.info(f"Quantized model saved to: {quantized_model_path}")
        
        if args.upload:
            hf_token = get_api_key("HUGGINGFACE_TOKEN")
            logging.info(f"Uploading model to Hugging Face: {args.hf_repo_name}")
            upload_model(args.model, args.hf_repo_name, quantized_model_path, token=hf_token)
            logging.info(f"Model uploaded successfully to: {args.hf_repo_name}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise