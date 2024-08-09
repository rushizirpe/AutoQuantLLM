# src/models/model_utils.py

from src.utils import get_api_key
from huggingface_hub import create_repo, snapshot_download
from dotenv import load_dotenv
import os

load_dotenv()

HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')

print("HF Token Fetched:", HUGGING_FACE_TOKEN)
def load_model(model_id, token = None):
    MODEL_NAME = MODEL_ID = model_id 

    try:
        # Download model using huggingface_hub
        model_path = snapshot_download(
            repo_id=MODEL_ID,
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],  
            local_dir=MODEL_NAME
        )
    except Exception as e:
        print(f"Error downloading model: {e}")
        if "/" in MODEL_ID and MODEL_ID.split("/")[0] != "":
            hf_token = get_api_key("HUGGINGFACE_TOKEN")
            if hf_token:
                try:
                    # Retry downloading the model with the API key
                    model_path = snapshot_download(
                        repo_id=MODEL_ID,
                        token=hf_token,
                        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],  
                        local_dir=MODEL_NAME
                    )
                except Exception as retry_e:
                    print(f"Error retrying model download: {retry_e}")
            else:
                print("No API token found. Please set the HUGGINGFACE_TOKEN environment variable.")
        else:
            print("Invalid model identifier or model not found.")

    return model_path



