import os
import hashlib
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "shilongz/FlashFace-SD1.5"
cache_dir = "cache"

# Create the cache directory if it doesn't exist
os.makedirs(cache_dir, exist_ok=True)

# Define the list of files to check
files_to_check = [
    "README.md",
    "bpe_simple_vocab_16e6.txt.gz",
    "flashface.ckpt",
    "openai-clip-vit-large-14.pth",
    "retinaface_resnet50.pth",
    "sd-v1-vae.pth",
]

def calculate_file_hash(file_path):
    """Calculate the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def compare_file_hashes(src_path, dst_path):
    """Compare the hashes of two files."""
    src_hash = calculate_file_hash(src_path)
    dst_hash = calculate_file_hash(dst_path)
    return src_hash == dst_hash

# Download the repository
local_dir_path = snapshot_download(
    repo_id=repo_id,
    cache_dir=cache_dir,
    local_dir_use_symlinks=False,
)
print(f"Repository downloaded to: {local_dir_path}")

# Find the latest snapshot directory
snapshot_dirs = sorted(Path(local_dir_path).parent.glob("*"))
latest_snapshot_dir = snapshot_dirs[-1]
print(f"Latest snapshot directory: {latest_snapshot_dir}")

# Copy the files from the latest snapshot directory to the cache directory
for file in files_to_check:
    src_path = latest_snapshot_dir / file
    dst_path = Path(cache_dir) / file

    # Check if the source file exists
    if src_path.exists():
        # Check if the destination file exists and compare hashes
        if dst_path.exists() and compare_file_hashes(src_path, dst_path):
            print(f"Skipped {file} (already exists and hashes match)")
        else:
            # Create the destination directory if it doesn't exist
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy the file from source to destination
            shutil.copy(str(src_path), str(dst_path))
            print(f"Copied {file} to {dst_path}")
    else:
        print(f"Skipped {file} (not found in the latest snapshot directory)")

print("Script execution completed.")