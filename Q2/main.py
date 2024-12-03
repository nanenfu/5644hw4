import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import urllib.request
import os
import matplotlib.colors as mcolors
from tqdm import tqdm

# Step 1: Load the image from the given dataset URL
image_url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/images/train/2092.jpg"
image_path = "/home/nanenfu/5644hw4/Q2/61060.jpg"

if not os.path.exists(image_path):
    urllib.request.urlretrieve(image_url, image_path)

# Read the image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Image loaded with shape:", image.shape)

# Step 2: Downsample the image to reduce computational needs
scale_percent = 20  # Resize to 50% of the original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
print("Image downsampled to shape:", image.shape)

# Step 3: Generate 5-dimensional feature vectors for each pixel
rows, cols, channels = image.shape
feature_vectors = np.zeros((rows * cols, 5))

for i in range(rows):
    for j in range(cols):
        r, g, b = image[i, j]
        feature_vectors[i * cols + j] = [i, j, r / 255.0, g / 255.0, b / 255.0]

# Normalize each feature entry individually to the interval [0, 1]
feature_vectors[:, 0] /= rows
feature_vectors[:, 1] /= cols
print("Feature vectors generated with shape:", feature_vectors.shape)

# Step 4: Fit a Gaussian Mixture Model (GMM) to the feature vectors
# Use 10-fold cross-validation to determine the optimal number of components
kf = KFold(n_splits=10, shuffle=True, random_state=42)
best_gmm = None
best_score = -np.inf
best_n_components = 0

# for n_components in tqdm(range(31, 81), desc="Selecting optimal number of components"):
# for n_components in range(1, 11):
# for n_components in tqdm([100, 200, 500, 100], desc="Selecting optimal number of components"):
for n_components in [100, 200, 500]:
    avg_score = 0
    for train_idx, val_idx in kf.split(feature_vectors):
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(feature_vectors[train_idx])
        score = gmm.score(feature_vectors[val_idx])
        avg_score += score
    avg_score /= 10
    print(f"Average score for {n_components} components: {avg_score}")
    
    if avg_score > best_score:
        best_score = avg_score
        best_n_components = n_components
        best_gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)

print(f"Best number of components based on cross-validation: {best_n_components}")

# Fit the best GMM on the entire dataset
best_gmm.fit(feature_vectors)

# Step 5: Assign each pixel to the most likely component label using MAP classification
labels = best_gmm.predict(feature_vectors)
labels_image = labels.reshape((rows, cols))
print("Labels assigned to all pixels.")

# Normalize labels to grayscale values between 0 and 255 for visualization
labels_image_normalized = (255 * (labels_image - labels_image.min()) / (labels_image.max() - labels_image.min())).astype(np.uint8)
print("Labels normalized for grayscale visualization.")

# Create a color map for the segmented image
unique_labels = np.unique(labels)
num_labels = len(unique_labels)
colors = plt.cm.get_cmap('hsv', num_labels)
colored_segmentation = np.zeros((rows, cols, 3), dtype=np.uint8)

for i in range(rows):
    for j in range(cols):
        colored_segmentation[i, j] = (np.array(colors(labels_image[i, j])[:3]) * 255).astype(np.uint8)

print("Color map for segmented image created.")

# Step 6: Display the original image, grayscale segmented image, and colored segmented image side by side
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('GMM-Based Segmentation (Grayscale)')
plt.imshow(labels_image_normalized, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('GMM-Based Segmentation (Color)')
plt.imshow(colored_segmentation)
plt.axis('off')

plt.tight_layout()
plt.show()
