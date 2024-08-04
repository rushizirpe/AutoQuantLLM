# src/hf_utils.py

from huggingface_hub import HfApi, snapshot_download, create_repo, upload_folder
import os

def download_model(model_id: str, token: str = None) :
    return snapshot_download(repo_id=model_id, token=token)

def upload_model(base_model_id: str, repo_name: str, save_folder: str):
    api = HfApi()
    repo_id = repo_name if repo_name else base_model_id
    create_repo(repo_id=repo_id, exist_ok=True)

    api.upload_folder(
        folder_path=save_folder,
        repo_id=repo_id,
        token=os.environ['HF_TOKEN']
    )
