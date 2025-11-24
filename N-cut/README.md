# N-Cut Spectral Clustering Implementation

This folder implements **Normalized Cut (N-Cut) spectral clustering** on multiple benchmark datasets.

## Structure

```
N-cut/
├── main.py              # Main script: runs N-Cut on all datasets
├── data/                # Preprocessed datasets (*.npy files)
│   ├── coil-20.npy      # (1440, 1024) - 20 classes
│   ├── usps.npy         # (9298, 256) - 10 classes
│   ├── mnist-t.npy      # (5000, 784) - 10 classes
│   ├── mnist-s.npy      # (6996, 784) - 10 classes (randomly sampled)*
│   ├── jaffe.npy        # (213, 676) - 10 classes
│   ├── yaleb.npy        # (2452, 1024) - 38 classes
│   ├── pointing04.npy   # (2790, 1120) - 15 classes
│   ├── mpeg7.npy        # (581, 200) - 30 classes
│   └── umist.npy        # (1013, 644) - 20 classes
└── result/              # Visualization outputs (*.png files)
```

**Note:** MNIST-S uses 6996 samples randomly selected from the full MNIST dataset (70,000 samples) following the paper's methodology with `random_state=42` for reproducibility.

## Usage

Simply run the main script to process all datasets:

```bash
python main.py
```

The script will:

1. Auto-discover all `.npy` files in `data/` folder
2. Run N-Cut with 9 sigma values: {10⁻⁸, 10⁻⁶, 10⁻⁴, 10⁻², 10⁰, 10², 10⁴, 10⁶, 10⁸}
3. Calculate ACC (Hungarian algorithm) and NMI for each sigma
4. Generate visualization with 4 subplots per dataset
5. Validate results against paper benchmarks

## Experimental Results

| Dataset    | ACC | Best Sigma | Paper Target | Status     |
| ---------- | -------- | ---------- | ------------ | ---------- |
| COIL-20    | 77.43%   | 1e+00      | 63.0-73.6%   | ✅ EXCEEDS |
| JAFFE      | 95.31%   | 1e+02      | 77.4-90.4%   | ✅ EXCEEDS |
| MPEG7      | 68.16%   | 1e+04      | 65.1-68.7%   | ✅ WITHIN  |
| UMIST      | 62.59%   | 1e+02      | 59.4-60.8%   | ✅ EXCEEDS |
| USPS       | 65.50%   | 1e+02      | 67.1-79.7%   | ⚠️ BELOW   |
| MNIST-T    | 50.90%   | 1e+02      | 62.8-69.6%   | ⚠️ BELOW   |
| MNIST-S    | 55.35%   | 1e+06      | 62.5-66.5%   | ⚠️ BELOW   |
| Pointing04 | 45.77%   | 1e+02      | 67.4-73.8%   | ⚠️ BELOW   |
| YaleB      | 34.22%   | 1e+04      | 44.7-47.7%   | ⚠️ BELOW   |

## Performance Analysis

**Good Performance (COIL-20, JAFFE, MPEG7, UMIST):**

-   These datasets have **distinct visual features** and **well-separated clusters**
-   JAFFE (facial expressions) benefits from clear facial structure differences
-   COIL-20 (objects) has high inter-class variance due to different object shapes

**Suboptimal Performance (Pointing04, YaleB, MNIST):**

-   **Pointing04 (45.77% vs 70.6% target)**: Hand gesture variations create **high intra-class variance**. Different hand poses from the same person can be more similar to other people's poses than to their own.
-   **YaleB (34.22% vs 46.2% target)**: Extreme lighting variations cause **feature space distortion**. Same person under different lighting can appear more different than different people under similar lighting.
-   **MNIST (50-55% vs 62-66% target)**: Handwritten digits have **overlapping feature distributions**. Different writing styles make "1" vs "7" or "4" vs "9" ambiguous even for spectral methods.

**N-Cut Limitations:**

1. **Assumes convex clusters**: N-Cut struggles with non-convex cluster shapes common in pose/lighting variations
2. **Sensitive to intra-class variance**: When within-class variance exceeds between-class variance, spectral embedding fails
3. **Linear separability**: RBF kernel may not capture complex manifold structures in high-dimensional spaces

## Missing Dataset

**USF HumanID** is excluded due to **copyright restrictions** and cannot be provided in this repository.

## Dependencies

-   numpy
-   scikit-learn
-   matplotlib
-   scipy

## Authors

COMP3314 Group 5
