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

    print(f"Patching {config_path} for transformers compatibility...")

    content = config_path.read_text(encoding="utf-8")

    old_init = """    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(
            image_processor,
            tokenizer,
            chat_template=chat_template,
        )"""

    new_init = """    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(
            image_processor,
            tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
        )"""

    old_attr = '    attributes = ["image_processor", "tokenizer"]'
    new_attr = '    attributes = ["image_processor", "tokenizer", "video_processor"]'

    patched_any = False

    if old_init in content:
        content = content.replace(old_init, new_init)
        print("✓ Patched __init__ signature to include video_processor")
        patched_any = True
    else:
        print("! __init__ patch pattern not found (maybe already patched or upstream changed)")

    if old_attr in content:
        content = content.replace(old_attr, new_attr)
        print("✓ Patched attributes to include video_processor")
        patched_any = True
    else:
        print("! attributes patch pattern not found (maybe already patched or upstream changed)")

    if patched_any:
        config_path.write_text(content, encoding="utf-8")
        print("Patch finished.")
    else:
        print("No changes written (already compatible or patterns didn’t match).")

    return patched_any


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and patch DoTS OCR model weights.")
    parser.add_argument("--dir", type=str, default="./weights/DotsOCR", help="Directory to save weights.")
    args = parser.parse_args()

    download_dots_ocr(args.dir)
