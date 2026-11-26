import os
from PIL import Image

train_dir = "data/phase1/Train"
test_dir = "data/phase1/Test"

def remove_invalid_images(directory):
    valid_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.lower().endswith(valid_extensions):
                print(f"Removing unsupported file: {file_path}")
                os.remove(file_path)
            else:
                try:
                    img = Image.open(file_path)
                    img.verify()  # Verify if the image is valid
                except (IOError, SyntaxError):
                    print(f"Removing corrupted file: {file_path}")
                    os.remove(file_path)

# Run the cleanup
remove_invalid_images(train_dir)
remove_invalid_images(test_dir)

