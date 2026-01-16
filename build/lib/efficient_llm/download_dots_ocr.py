import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def find_configuration_dots(base_dir: Path) -> Path | None:
    """Find configuration_dots.py somewhere under base_dir."""
    for p in base_dir.rglob("configuration_dots.py"):
        return p
    return None


def patch_configuration_dots(local_dir: str) -> bool:
    """
    Patches configuration_dots.py to ensure compatibility with modern transformers.
    Returns True if a patch was applied, else False.
    """
    base_dir = Path(local_dir)
    config_path = find_configuration_dots(base_dir)

    if config_path is None:
        print("Warning: configuration_dots.py not found under:", base_dir)
        print("Skipping patch.")
        return False

    import re
    print(f"Patching {config_path} for transformers compatibility...")

    content = config_path.read_text(encoding="utf-8")
    original_content = content

    # 1. Patch attributes list
    # Look for: attributes = ["image_processor", "tokenizer"]
    if 'attributes = ["image_processor", "tokenizer", "video_processor"]' not in content:
        content = re.sub(
            r'attributes\s*=\s*\[\s*"image_processor"\s*,\s*"tokenizer"\s*\]',
            'attributes = ["image_processor", "tokenizer", "video_processor"]',
            content
        )

    # 2. Patch __init__ arguments
    # Add video_processor=None
    if "video_processor=None" not in content:
        content = re.sub(
            r'def __init__\s*\(\s*self\s*,\s*image_processor=None\s*,\s*tokenizer=None\s*,',
            'def __init__(\n        self,\n        image_processor=None,\n        tokenizer=None,\n        video_processor=None,',
            content
        )

    # 3. Patch super().__init__ call
    # Add video_processor=video_processor
    if "video_processor=video_processor" not in content:
        content = re.sub(
            r'super\(\)\.__init__\(\s*image_processor\s*,\s*tokenizer\s*,',
            'super().__init__(\n            image_processor,\n            tokenizer,\n            video_processor=video_processor,',
            content
        )

    if content != original_content:
        config_path.write_text(content, encoding="utf-8")
        print("✓ Successfully patched configuration_dots.py")
        
        print("\n" + "="*60)
        print("IMPORTANT: You must clear your Hugging Face cache for this model")
        print("to ensure the patched local version is used.")
        print("Run something like:")
        print("  rm -rf ~/.cache/huggingface/modules/transformers_modules/DotsOCR")
        print("="*60 + "\n")
        return True
    else:
        if "video_processor" in content:
            print("✓ File appears to be already patched.")
            return True
        else:
            print("! Failed to match patterns. Patch not applied.")
            print("  Please manually check configuration_dots.py and add video_processor.")
            return False


def download_dots_ocr(local_dir: str = "./weights/DotsOCR"):
    """
    Downloads the DoTS OCR model weights from Hugging Face and applies compatibility patches.
    """
    repo_id = "rednote-hilab/dots.ocr"
    print(f"Downloading DoTS OCR model weights from {repo_id}...")

    os.makedirs(local_dir, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )

    print(f"Successfully downloaded DoTS OCR weights to: {local_dir}")

    # Apply the patch (best-effort)
    patch_configuration_dots(local_dir)


def main():
    parser = argparse.ArgumentParser(description="Download and patch DoTS OCR model weights.")
    parser.add_argument("--dir", type=str, default="./weights/DotsOCR", help="Directory to save weights.")
    args = parser.parse_args()

    download_dots_ocr(args.dir)

if __name__ == "__main__":
    main()
