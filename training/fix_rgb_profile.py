from PIL import Image
import os

# Some images seemed to have invalid sRGB profiles which
# caused tensorflow to complain so this script removes those
# invalid profiles
# - Fabian

dataset_path = "./data"

for root, _, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)
        if file_path.lower().endswith(".png"):
            try:
                img = Image.open(file_path)
                img = img.convert("RGB")  # Convert to RGB to strip color profile
                img.save(file_path, "PNG", icc_profile=None)  # Remove sRGB profile
                print(f"Fixed: {file_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
