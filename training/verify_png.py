from PIL import Image
import os

dataset_path = "./data"

for root, _, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)
        if file_path.lower().endswith(".png"):
            try:
                img = Image.open(file_path)
                img.verify()
                print(f"Verified: {file_path}")
            except (IOError, SyntaxError) as e:
                print(f"Corrupted file: {file_path}")
                os.remove(file_path)
