import os
import argparse
from huggingface_hub import HfApi

def upload_weight(file_path: str, repo_id: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到檔案: {file_path}")

    print(f"準備上傳 {file_path} 至 {repo_id}...")
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id=repo_id,
        repo_type="model",
    )
    print("上傳完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="上傳權重檔案至 Hugging Face Model Hub")
    parser.add_argument("--file", default="sam3.pt", help="要上傳的本機檔案路徑")
    parser.add_argument("--repo", required=True, help="目標 Hugging Face Repo ID (格式: username/repo-name)")
    args = parser.parse_args()

    upload_weight(args.file, args.repo)
