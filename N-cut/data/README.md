# Preprocessed Datasets

This folder contains preprocessed benchmark datasets in `.npy` format.

## Dataset Specifications

| Dataset            | Shape        | Features     | Classes | Description                                  |
| ------------------ | ------------ | ------------ | ------- | -------------------------------------------- |
| **coil-20.npy**    | (1440, 1024) | 32×32 pixels | 20      | COIL-20 objects                              |
| **usps.npy**       | (9298, 256)  | 16×16 pixels | 10      | USPS handwritten digits                      |
| **mnist-t.npy**    | (5000, 784)  | 28×28 pixels | 10      | MNIST training subset (first 5000)           |
| **mnist-s.npy**    | (6996, 784)  | 28×28 pixels | 10      | MNIST random subset (6996 samples)\*         |
| **jaffe.npy**      | (213, 676)   | 26×26 pixels | 10      | JAFFE facial expressions (10 persons)        |
| **yaleb.npy**      | (2452, 1024) | 32×32 pixels | 38      | Extended Yale Face Database B                |
| **pointing04.npy** | (2790, 1120) | 40×28 pixels | 15      | Pointing'04 hand gestures                    |
| **mpeg7.npy**      | (581, 200)   | 20×10 pixels | 30      | MPEG-7 shape silhouettes (30 classes subset) |
| **umist.npy**      | (1013, 644)  | 28×23 pixels | 20      | UMIST face database                          |

**\*MNIST-S Sampling Details:** 6996 samples were randomly selected from the combined MNIST training (60,000) and test (10,000) datasets using `numpy.random.seed(42)` for reproducibility, following the paper's experimental setup.

## File Format

Each `.npy` file contains a Python dictionary:

```python
{
    'data': numpy.ndarray,   # Shape: (n_samples, n_features), dtype: uint8
    'labels': numpy.ndarray  # Shape: (n_samples,), dtype: int32, 0-indexed
}
```

## Note on Missing Data

**USF HumanID** dataset is not included due to copyright restrictions. The original dataset contains biometric face recognition data that requires special licensing.

## Regenerating Datasets

Preprocessed data files (`.npy`) are **excluded from git** (see `.gitignore`) due to file size. To regenerate:

1. Place original dataset archives in project root
2. Run preprocessing scripts in `../prepared/` folder:
    ```bash
    python ../prepared/coil-20.py
    python ../prepared/usps.py
    python ../prepared/mnist.py
    # ... etc
    ```

All preprocessed files will be generated in this `data/` folder.
