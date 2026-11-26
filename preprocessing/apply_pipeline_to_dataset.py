import cv2
import numpy as np
import os
import time

input_dir = "data/dermo/Validation/Healthy"
output_dir = "data/dermo_dip/Validation/Healthy"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Supported image extensions
supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# List all valid image files
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_exts)]

for img_name in image_files:
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Skipped: {img_name} (Unreadable)")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (9, 9), 0)
    sharp = cv2.addWeighted(gray, 2, denoised, -0.6, 0)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(1, 1))
    enhanced = clahe.apply(sharp)

    _, otsu_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(otsu_thresh, 1300, 1300)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create mask from contours
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Apply mask to original image
    masked_result = cv2.bitwise_and(img, img, mask=mask)

    # Save the final masked output
    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, masked_result)

    print(f"Processed & saved: {img_name}")

print("Batch processing complete.")
