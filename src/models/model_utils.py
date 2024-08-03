# src/models/model_utils.py

from huggingface_hub import create_repo, snapshot_download
from dotenv import load_dotenv
import os

load_dotenv()

HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')

print("HF Token Fetched:", HUGGING_FACE_TOKEN)
def load_model(model_id):
    MODEL_NAME = MODEL_ID = model_id 

    # Download model using huggingface_hub
    model_path = snapshot_download(
        repo_id=MODEL_ID,
        token=HUGGING_FACE_TOKEN,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],  
        local_dir=MODEL_NAME
    )

    return model_path



