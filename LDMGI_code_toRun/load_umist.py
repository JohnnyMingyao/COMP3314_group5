import numpy as np


def load_umist_images(data_path='./data/umist.npy', normalization='global'):
    
    
    print(f"Loading Umist dataset from {data_path}...")
    
    # Load .npy file
    data_dict = np.load(data_path, allow_pickle=True).item()
    
    X = data_dict['data'].astype(np.float64)
    labels = data_dict['labels'].astype(np.int32)
    
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(labels))} classes")
    
    # Scale to [0, 1] if needed
    if X.max() > 1.0:
        X = X / 255.0
    
    # Normalize
    if normalization == 'global':
        print("Applying global normalization...")
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True)
        X_std[X_std == 0] = 1
        X = (X - X_mean) / X_std
    elif normalization == 'local':
        print("Applying local normalization...")
        X_mean = X.mean(axis=1, keepdims=True)
        X_std = X.std(axis=1, keepdims=True)
        X_std[X_std == 0] = 1
        X = (X - X_mean) / X_std
    
    return X, labels