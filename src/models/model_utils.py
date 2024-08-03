# src/models/model_utils.py

from transformers import AutoModel
from huggingface_hub import create_repo, HfApi, ModelCard, snapshot_download
# from google.colab import userdata, runtime
import shutil
import fnmatch
import os
from dotenv import load_dotenv

load_dotenv()

HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')

print("HF Token Fetched:", HUGGING_FACE_TOKEN)
def load_model(model_id):
    MODEL_NAME = MODEL_ID = model_id 
    # api = HfApi()
    # HF_TOKEN = os.environ["<DUMMY>"]
    # HF_TOKEn = "<DUMMY>"

    # Download model using huggingface_hub
    model_path = snapshot_download(
        repo_id=MODEL_ID,
        token=HUGGING_FACE_TOKEN,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],  # Ignore certain file types
        local_dir=MODEL_NAME
    )

    return model_path
    return AutoModel.from_pretrained(model_id)



