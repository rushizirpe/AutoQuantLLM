# src/hf_utils.py

from huggingface_hub import HfApi, snapshot_download, create_repo, upload_folder
import os

def download_model(model_id: str, token: str = None):
    return snapshot_download(repo_id=model_id, token=token)

def upload_model(base_model_id: str, repo_name: str, save_folder: str, token: str):
    api = HfApi()
    repo_id = repo_name if repo_name else base_model_id
    create_repo(repo_id=repo_id, exist_ok=True, token=token)

    api.upload_folder(
        folder_path=save_folder,
        repo_id=repo_id,
        token=token
    )

def check_repo_exists(repo_name: str, token: str) -> bool:
    api = HfApi(token=token)
    try:
        api.repo_info(repo_id=repo_name)
        return True
    except Exception:
        return False

def create_repo_if_not_exists(repo_name: str, token: str) -> (bool, str):
    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo_name, exist_ok=False, private=False)
        return True, ""
    except Exception as e:
        if hasattr(e, 'response') and e.response is not None:
            if e.response.status_code == 403:
                return False, "You don't have permission to create a repository in this namespace. Please check your access rights or choose a different namespace."
            elif e.response.status_code == 409:
                return False, f"Repository '{repo_name}' already exists."
            else:
                return False, f"Failed to create repository: {str(e)}"
        else:
            return False, f"An unexpected error occurred: {str(e)}"