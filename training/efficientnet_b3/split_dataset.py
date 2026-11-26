import os
import shutil
import random

# Paths
original_dataset_dir = "data/raw"
output_base_dir = "data/dermo"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Validate ratios sum to 1
if round(train_ratio + val_ratio + test_ratio, 2) != 1.0:
    raise ValueError("Train, validation, and test ratios must sum to 1.0")

# Seed
random.seed(42)

# Clean output folders first
for split in ["Train", "Validation", "Test"]:
    split_path = os.path.join(output_base_dir, split)
    if os.path.exists(split_path):
        shutil.rmtree(split_path)
    os.makedirs(split_path)

# Process each class
for class_name in os.listdir(original_dataset_dir):
    class_dir = os.path.join(original_dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    # Collect image files only
    images = [f for f in os.listdir(class_dir)
              if os.path.isfile(os.path.join(class_dir, f)) and not f.startswith(".")]
    
    if len(images) == 0:
        continue  # Skip empty folders

    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    split_indices = {
        "Train": images[:train_end],
        "Validation": images[train_end:val_end],
        "Test": images[val_end:]
    }

    for split, image_list in split_indices.items():
        target_dir = os.path.join(output_base_dir, split, class_name)
        os.makedirs(target_dir, exist_ok=True)

        for img_name in image_list:
            src = os.path.join(class_dir, img_name)
            dst = os.path.join(target_dir, img_name)
            if os.path.exists(src):
                shutil.copy2(src, dst)

print("Dataset split completed without duplication.")
