"""Deploy WasteWise to a Hugging Face Docker Space.

Uploads only what the Docker image needs (never .env, tests or .venv), using the
Hub API so binary files such as the .keras model need no Git LFS setup.

Usage:
    hf auth login                      # once; or set HF_TOKEN
    python scripts/deploy_hf.py <username>/wastewise [--set-secret] [--cors-origins URLS]
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# fnmatch-style: "*" also matches "/", so "app/*.py" covers every module under app/.
DEPLOY_PATTERNS = [
    "Dockerfile",
    "requirements.txt",
    "app/*.py",
    "models/*.keras",
    "knowledge/*.json",
    "config/*.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space_id", help="Space id, e.g. rifialdiif/wastewise")
    parser.add_argument(
        "--set-secret", action="store_true", help="copy GEMINI_API_KEY from local .env into the Space secrets"
    )
    parser.add_argument("--cors-origins", help="comma-separated website origins allowed to call the API")
    args = parser.parse_args()

    api = HfApi()
    print(f"Logged in as: {api.whoami()['name']}")

    api.create_repo(args.space_id, repo_type="space", space_sdk="docker", exist_ok=True)

    if args.set_secret:
        from app.config import settings

        if not settings.gemini_api_key:
            sys.exit("GEMINI_API_KEY is empty in .env; nothing to upload.")
        api.add_space_secret(args.space_id, "GEMINI_API_KEY", settings.gemini_api_key)
        print("Set secret GEMINI_API_KEY")

    if args.cors_origins is not None:
        api.add_space_variable(args.space_id, "CORS_ORIGINS", args.cors_origins)
        print(f"Set variable CORS_ORIGINS={args.cors_origins}")

    api.upload_folder(
        repo_id=args.space_id,
        repo_type="space",
        folder_path=ROOT,
        allow_patterns=DEPLOY_PATTERNS,
        delete_patterns=DEPLOY_PATTERNS,  # remove files that no longer exist locally
        commit_message="Deploy WasteWise",
    )
    api.upload_file(
        repo_id=args.space_id,
        repo_type="space",
        path_or_fileobj=ROOT / "deploy" / "huggingface" / "README.md",
        path_in_repo="README.md",
        commit_message="Update Space card",
    )

    owner, name = args.space_id.split("/")
    print(f"Uploaded. Build logs: https://huggingface.co/spaces/{args.space_id}")
    print(f"API (once running):  https://{owner}-{name}.hf.space/docs")


if __name__ == "__main__":
    main()
