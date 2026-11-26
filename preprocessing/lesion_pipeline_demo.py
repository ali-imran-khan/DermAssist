import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Read image
img = cv2.imread("samples/example.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Denoise using Gaussian Blur
denoised = cv2.GaussianBlur(gray, (9, 9), 0)

# Sharpen image
sharp = cv2.addWeighted(gray, 2, denoised, -0.6, 0)

# Apply CLAHE (local contrast enhancement)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(1, 1))
enhanced = clahe.apply(sharp)

# Start processing timer
start_time = time.time()

# Otsu thresholding
_, otsu_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Canny edge detection
edges = cv2.Canny(otsu_thresh, 1300, 1300)

# Stronger dilation to expand boundaries
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
dilated = cv2.dilate(edges, kernel, iterations=2)

# Find contours from dilated edges
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# End processing timer
end_time = time.time()
processing_time = (end_time - start_time) * 1000  # in milliseconds

# Draw contours
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# Create mask from contours
mask = np.zeros_like(gray)
cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

# Apply mask to original image
masked_result = cv2.bitwise_and(img, img, mask=mask)

# Visualization
plt.figure(figsize=(18, 10))

plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(2, 4, 2)
plt.imshow(denoised, cmap='gray')
plt.title('Denoised')

plt.subplot(2, 4, 3)
plt.imshow(enhanced, cmap='gray')
plt.title('Enhanced (CLAHE)')

plt.subplot(2, 4, 4)
plt.imshow(otsu_thresh, cmap='gray')
plt.title('Otsu Threshold')

plt.subplot(2, 4, 5)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')

plt.subplot(2, 4, 6)
plt.imshow(dilated, cmap='gray')
plt.title('Dilated Edges (Stronger)')

plt.subplot(2, 4, 7)
plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
plt.title('Contours')

plt.subplot(2, 4, 8)
plt.imshow(cv2.cvtColor(masked_result, cv2.COLOR_BGR2RGB))
plt.title('Masked Region')

plt.tight_layout()
plt.show()

# Metrics
print("Evaluation Metrics:")
print(f"Detection Accuracy: (To be calculated manually based on ground truth)")
print(f"Processing Time   : {processing_time:.2f} ms")
