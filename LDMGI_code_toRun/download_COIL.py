"""
Data loading and preprocessing for COIL-20 dataset
Assumes data is already in ./data/coil20/
"""

import numpy as np
import os
from PIL import Image
import glob


def load_coil20_images(data_dir='./data/coil20', 
                       use_pca=False, n_components=None,
                       normalization='global'):
    """
    Load COIL-20 images from ./data/coil20/
    
    Handles multiple naming conventions:
    - obj{id}__{angle}.png where angle can be:
      * 0, 5, 10, ..., 355 (72 views, 5-degree increments)
      * 0, 1, 2, ..., 71 (72 views, sequential)
    
    Parameters:
    -----------
    data_dir : str
        Directory containing COIL-20 PNG files (default: './data/coil20')
    use_pca : bool
        Whether to apply PCA dimensionality reduction
    n_components : int
        Number of PCA components (if use_pca=True)
    normalization : str
        'global': normalize each feature across all images (recommended)
        'local': normalize each image independently
        'none': no normalization
    
    Returns:
    --------
    X : numpy.ndarray, shape (n_images, d)
        Feature matrix
    labels : numpy.ndarray, shape (n_images,)
        Object labels (0-19)
    """
    
    print(f"Loading COIL-20 images from {data_dir}...")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Directory not found: {data_dir}\n"
            f"Please place COIL-20 images in ./data/coil20/"
        )
    
    # Find all PNG files
    all_files = glob.glob(os.path.join(data_dir, "*.png"))
    
    if len(all_files) == 0:
        raise FileNotFoundError(
            f"No PNG files found in {data_dir}\n"
            f"Please place COIL-20 images in ./data/coil20/"
        )
    
    print(f"Found {len(all_files)} PNG files")
    
    # Show sample filenames to detect format
    sample_files = sorted([os.path.basename(f) for f in all_files])[:10]
    print(f"Sample filenames: {sample_files}")
    
    images = []
    labels = []
    filenames = []
    
    # Load all images
    for filepath in sorted(all_files):
        filename = os.path.basename(filepath)
        
        # Parse filename: obj{id}__{angle}.png
        if "__" in filename and filename.startswith("obj") and filename.endswith(".png"):
            try:
                # Extract object ID and angle
                parts = filename.replace(".png", "").split("__")
                obj_str = parts[0].replace("obj", "")
                obj_id = int(obj_str)
                angle_or_view = int(parts[1])
                
                # Validate object ID (should be 1-20)
                if not (1 <= obj_id <= 20):
                    continue  # Skip invalid object IDs
                
                # Accept any angle/view number - don't validate
                # (handles both 0-355 by 5, and 0-71 sequential formats)
                
                # Load and preprocess image
                img = Image.open(filepath).convert('L')  # Grayscale
                img = img.resize((32, 32), Image.LANCZOS)  # Resize to 32x32
                img_array = np.array(img).flatten().astype(np.float64)
                img_array = img_array / 255.0  # Scale to [0, 1]
                
                images.append(img_array)
                labels.append(obj_id - 1)  # Convert to 0-indexed (0-19)
                filenames.append(filename)
                
            except (ValueError, IndexError) as e:
                # Silently skip files that don't match the expected format
                continue
        else:
            # Try alternative format: obj{id}.png (single image per object)
            if filename.startswith("obj") and filename.endswith(".png") and "__" not in filename:
                try:
                    obj_id = int(filename.replace("obj", "").replace(".png", ""))
                    
                    if not (1 <= obj_id <= 20):
                        continue
                    
                    # Load and preprocess image
                    img = Image.open(filepath).convert('L')
                    img = img.resize((32, 32), Image.LANCZOS)
                    img_array = np.array(img).flatten().astype(np.float64)
                    img_array = img_array / 255.0
                    
                    images.append(img_array)
                    labels.append(obj_id - 1)
                    filenames.append(filename)
                    
                except ValueError:
                    continue
    
    if len(images) == 0:
        raise ValueError(
            f"No valid images could be loaded from {data_dir}\n"
            f"Please check the image files are in the correct format"
        )
    
    # Convert to numpy arrays
    X = np.array(images)
    labels = np.array(labels)
    
    print(f"\n{'='*70}")
    print(f"Dataset loaded successfully:")
    print(f"  Total images: {X.shape[0]}")
    print(f"  Feature dimension: {X.shape[1]} (32×32 pixels)")
    print(f"  Number of objects: {len(np.unique(labels))}")
    
    # Check dataset completeness
    label_counts = np.bincount(labels)
    print(f"  Images per object: min={label_counts.min()}, max={label_counts.max()}, mean={label_counts.mean():.1f}")
    
    if X.shape[0] == 1440:
        print(f"  ✓ Full COIL-20 dataset (20 objects × 72 views = 1,440 images)")
    elif X.shape[0] == 1800:
        print(f"  ✓ Extended COIL-20 dataset (20 objects × 90 views = 1,800 images)")
    elif X.shape[0] == 300:
        print(f"  ⚠ WARNING: Subset detected (300 images)")
        print(f"     The paper uses the FULL dataset with 1,440 images!")
        print(f"     Your results will NOT match the paper's results!")
    else:
        print(f"  ℹ Dataset size: {X.shape[0]} images")
        print(f"     Paper uses 1,440 images (20 objects × 72 views)")
    print(f"{'='*70}\n")
    
    # Apply normalization
    if normalization == 'global':
        # Global normalization: each feature (pixel) has zero mean and unit variance
        # This is standard for spectral clustering methods
        print("Applying global normalization (zero mean, unit variance per feature)...")
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True)
        X_std[X_std == 0] = 1  # Avoid division by zero
        X = (X - X_mean) / X_std
        
    elif normalization == 'local':
        # Local normalization: each image has zero mean and unit variance
        print("Applying local normalization (zero mean, unit variance per image)...")
        X_mean = X.mean(axis=1, keepdims=True)
        X_std = X.std(axis=1, keepdims=True)
        X_std[X_std == 0] = 1
        X = (X - X_mean) / X_std
        
    elif normalization == 'none':
        print("No normalization applied.")
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    
    # Optional PCA dimensionality reduction
    if use_pca and n_components is not None:
        from sklearn.decomposition import PCA
        print(f"\nApplying PCA: {X.shape[1]} -> {n_components} dimensions")
        pca = PCA(n_components=n_components, random_state=42)
        X = pca.fit_transform(X)
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    print(f"\nFinal feature dimension: {X.shape[1]}\n")
    
    return X, labels