import os
import shutil
from pathlib import Path

# Source and destination directories
src_dir = 'inputs'
dest_dir = 'inputs_test'

# Ensure destination directory exists
Path(dest_dir).mkdir(parents=True, exist_ok=True)

# Get list of all files in source directory, sorted by name
files = sorted(Path(src_dir).iterdir(), key=lambda f: f.name)

# Move the last 4000 files
for file in files[-4000:]:
    if file.is_file():  # Ensure it's a file (not a directory)
        shutil.move(str(file), dest_dir)

print(f"Moved last 4000 files to {dest_dir}")
