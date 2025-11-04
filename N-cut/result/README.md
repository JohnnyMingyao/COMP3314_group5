# N-Cut Experimental Results

This folder contains visualization outputs from N-Cut spectral clustering experiments.

## Visualization Files

Each dataset generates a 4-subplot visualization showing:

1. **Accuracy vs Sigma** - Bar chart showing clustering accuracy across 9 sigma values
2. **NMI vs Sigma** - Normalized Mutual Information scores
3. **Confusion Matrix** - Cluster-to-class mapping (best sigma)
4. **Results Summary** - Best performance metrics and validation status

## Key Findings

-   **Strong performers**: COIL-20 (77.43%), JAFFE (95.31%), MPEG7 (68.16%), UMIST (62.59%)
-   **Underperformers**: Pointing04 (45.77%), YaleB (34.22%), MNIST-T/S (50-55%)
-   **Common issue**: Datasets with high intra-class variance (lighting/pose variations) struggle with spectral clustering

See main `README.md` for detailed performance analysis.
