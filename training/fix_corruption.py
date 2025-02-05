from PIL import Image
import os
import imghdr

dataset_path = "./data"

for root, _, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)
        if file_path.lower().endswith(".png"):
            # Check the actual file format
            image_format = imghdr.what(file_path)
            if image_format != "png":
                print(f"Invalid format for {file_path}, detected as {image_format}, converting to PNG")
                try:
                    img = Image.open(file_path)
                    # Convert to RGB to avoid color profile issues
                    img = img.convert("RGB")
                    new_file_path = file_path.replace(".png", ".jpg")  # Convert to .jpg if necessary
                    img.save(new_file_path, "JPEG")
                    os.remove(file_path)  # Remove the original invalid PNG file
                    print(f"Converted: {file_path} -> {new_file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")
