"""
resize_and_flatten.py

Workflow:
1. Read each image from INPUT_DIR (grayscale).
2. Resize with cv2.resize to TARGET_SIZE.
3. Flatten with NumPy into a 1-D vector.
4. Stack all vectors into an (num_samples, num_features) matrix.
5. Save the matrix (and optional labels) to an .npy file.
"""

from pathlib import Path
import numpy as np
import cv2  

INPUT_DIR = Path("raw_images")       
TARGET_SIZE = (20, 10)               
OUTPUT_PATH = Path("data/resized.npy")


feature_list = []
label_list = []  


for image_path in sorted(INPUT_DIR.glob("*.png")): 
    
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    resized = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    vector = resized.reshape(-1).astype(np.float64)  
    
    feature_list.append(vector)


if not feature_list:
    raise RuntimeError("No images processed. Check INPUT_DIR and file pattern.")

X = np.vstack(feature_list)
print("Feature matrix shape:", X.shape)

payload = {"X": X}
if label_list:
    payload["y"] = np.array(label_list)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
np.save(OUTPUT_PATH, payload)
print(f"Saved to {OUTPUT_PATH.resolve()}")