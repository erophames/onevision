from PIL import Image
import os
import imghdr

# This script fixes some files that had wierd webp formats
# which is not supported by tensorflow
# - Fabian

dataset_path = "./data"

for root, _, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)
        if file_path.lower().endswith(".png") and imghdr.what(file_path) == "webp":
            try:
                img = Image.open(file_path).convert("RGB")
                new_file_path = file_path.replace(".png", ".jpg")
                img.save(new_file_path, "JPEG")
                os.remove(file_path)
                print(f"Converted: {file_path} -> {new_file_path}")
            except Exception as e:
                print(f"Failed to convert {file_path}: {e}")
