import numpy as np


def load_mnistT_images(data_path='./data/mnistT.npy', normalization='global'):
    
    
    print(f"Loading mnist-t dataset from {data_path}...")
    
    # Load .npy file
    data_dict = np.load(data_path, allow_pickle=True).item()
    
    X = data_dict['data'].astype(np.float64)
    labels = data_dict['labels'].astype(np.int32)
    
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(labels))} classes")
    
    
    
    return X, labels