import argparse
from huggingface_hub import snapshot_download
from pathlib import Path



BASE_DIR = Path("checkpoints")

def download_and_patch(model_id: str):
    model_name = model_id.split("/")[-1]
    target_dir = BASE_DIR / model_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Downloading model from: {model_id}")
    snapshot_download(
        repo_id=model_id,
        local_dir=target_dir,
    )
    print(f"✅ Model downloaded to: {target_dir.resolve()}")

    print("\n💡 You can now load the model from:")
    print(f"{target_dir.resolve()}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and patch a HuggingFace model locally.")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    args = parser.parse_args()

    download_and_patch(args.model_id)