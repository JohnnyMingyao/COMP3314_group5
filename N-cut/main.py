"""
N-Cut Clustering on Multiple Datasets
Automatically processes all .npy files in data/ folder
Implements Normalized Cut spectral clustering algorithm with auto-validation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import time

# ==============================================================================
#                 N-Cut Multi-Dataset Processor with Auto-Validation
# ==============================================================================

# Dataset Configurations - Based on Paper Table IV (N-Cut column)
# Key: core part of .npy filename (lowercase, flexible matching)
# n_clusters: number of classes
# target_acc/std: mean ± standard deviation reported in paper
DATASET_CONFIG = {
    'coil':        {'n_clusters': 20, 'target_acc': 68.3, 'target_acc_std': 5.3},
    'usps':        {'n_clusters': 10, 'target_acc': 73.4, 'target_acc_std': 6.3},
    'mnist-t':     {'n_clusters': 10, 'target_acc': 66.2, 'target_acc_std': 3.4},
    'mnist-s':     {'n_clusters': 10, 'target_acc': 64.5, 'target_acc_std': 2.0},
    'humanid':     {'n_clusters': 122,'target_acc': 42.6, 'target_acc_std': 0.4}, # the copyright issue -- will not be used
    'umist':       {'n_clusters': 20, 'target_acc': 60.1, 'target_acc_std': 0.7},
    'yale':        {'n_clusters': 38, 'target_acc': 46.2, 'target_acc_std': 1.5},
    'jaffe':       {'n_clusters': 10, 'target_acc': 83.9, 'target_acc_std': 6.5},
    'pointing':    {'n_clusters': 15, 'target_acc': 70.6, 'target_acc_std': 3.2},
    'mpeg':        {'n_clusters': 30, 'target_acc': 66.9, 'target_acc_std': 1.8},
}

# Path Setup
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data' 
RESULT_DIR = SCRIPT_DIR / 'result'
RESULT_DIR.mkdir(exist_ok=True)

# Grid search over sigma values
SIGMA_VALUES = [1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8]

def clustering_accuracy(y_true, y_pred):
    """Calculate clustering accuracy using Hungarian algorithm"""
    y_true, y_pred = y_true.astype(np.int64), y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size

# Main Script
print("="*80)
print("  N-Cut Clustering - Multi-Dataset Processor with Auto-Validation")
print("="*80)

npy_files = list(DATA_DIR.glob('*.npy'))
if not npy_files:
    print(f"ERROR: No .npy files found in {DATA_DIR}. Please run preprocessing scripts first.")
    exit(1)

print(f"Found {len(npy_files)} dataset(s) to process...")

for data_path in npy_files:
    # Match configuration
    dataset_name_key = next((key for key in DATASET_CONFIG if key in data_path.stem.lower()), None)
    if not dataset_name_key:
        print(f"\nWARNING: No configuration found for '{data_path.name}'. Skipping this file.")
        continue
    
    config = DATASET_CONFIG[dataset_name_key]
    dataset_name_clean = dataset_name_key.upper()
    
    print(f"\n{'='*25} Processing Dataset: {dataset_name_clean} {'='*25}")

    # Load data
    data = np.load(data_path, allow_pickle=True).item()
    X, y_true = data['data'], data['labels']
    if y_true.min() > 0:
        y_true = y_true - y_true.min()
    X_normalized = X / 255.0 if X.max() > 1.0 else X

    # Run N-Cut with 9 sigma values
    results, best_acc, best_nmi, best_sigma, best_y_pred = [], 0, 0, None, None
    print(f"Starting N-Cut experiment for {dataset_name_clean} with 9 sigma values...")
    
    for sigma in SIGMA_VALUES:
        gamma = 1 / (2 * sigma**2) if sigma != 0 else float('inf')
        
        print(f"  [{SIGMA_VALUES.index(sigma)+1}/{len(SIGMA_VALUES)}] Testing sigma = {sigma:.0e}")
        start_time = time.time()
        
        model = SpectralClustering(
            n_clusters=config['n_clusters'],
            affinity='rbf',
            gamma=gamma,
            n_jobs=-1,
            random_state=42,
            eigen_solver='arpack',  # Faster for large datasets
            n_init=10               # Default is 10, keeping it
        )
        y_pred = model.fit_predict(X_normalized)
        
        elapsed_time = time.time() - start_time
        
        acc = clustering_accuracy(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        
        print(f"    ACC: {acc*100:.2f}%, NMI: {nmi:.4f}, Time: {elapsed_time:.2f}s")
        
        results.append({'sigma': sigma, 'acc': acc, 'nmi': nmi})
        if acc > best_acc:
            best_acc, best_nmi, best_sigma, best_y_pred = acc, nmi, sigma, y_pred.copy()
        if nmi > best_nmi:
            best_nmi = nmi
    
    # Calculate target range and validation status
    acc_mean = config['target_acc']
    acc_std = config['target_acc_std']
    acc_min, acc_max = acc_mean - acc_std, acc_mean + acc_std
    is_success = acc_min <= best_acc * 100 <= acc_max

    # Print final report
    print("\n--- FINAL REPORT ---")
    print(f"  Dataset: {dataset_name_clean}")
    print(f"  Best Sigma Found: {best_sigma:.0e}")
    print(f"  Best ACC: {best_acc*100:.2f}%  |  Best NMI: {best_nmi*100:.2f}%")
    print(f"  Target ACC Range: {acc_min:.1f}% - {acc_max:.1f}%")
    if is_success:
        print(f"  Result: WITHIN TARGET RANGE")
    else:
        print(f"  Result: OUTSIDE TARGET RANGE")

    # Generate visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'N-Cut Clustering Results on {dataset_name_clean}', fontsize=16, fontweight='bold')
    
    # Plot 1: ACC vs Sigma
    ax1 = axes[0, 0]
    sigma_labels = [f"{s:.0e}" for s in SIGMA_VALUES]
    acc_values = [r['acc']*100 for r in results]
    colors = ['red' if s == best_sigma else 'steelblue' for s in SIGMA_VALUES]
    ax1.bar(sigma_labels, acc_values, color=colors, edgecolor='black')
    ax1.set_title('Accuracy vs Sigma')
    ax1.set_ylabel('Accuracy (%)')
    ax1.axhline(y=acc_min, color='green', linestyle='--', label=f'Target Min ({acc_min:.1f}%)')
    ax1.axhline(y=acc_max, color='orange', linestyle='--', label=f'Target Max ({acc_max:.1f}%)')
    ax1.legend()
    
    # Plot 2: NMI vs Sigma
    ax2 = axes[0, 1]
    nmi_values = [r['nmi'] for r in results]
    best_nmi_sigma = results[np.argmax([r['nmi'] for r in results])]['sigma']
    colors_nmi = ['red' if s == best_nmi_sigma else 'coral' for s in SIGMA_VALUES]
    ax2.bar(sigma_labels, nmi_values, color=colors_nmi, edgecolor='black')
    ax2.set_title('NMI vs Sigma')
    ax2.set_ylabel('NMI Score')

    # Plot 3: Confusion Matrix
    ax3 = axes[1, 0]
    confusion = np.zeros((config['n_clusters'], config['n_clusters']))
    for i in range(len(best_y_pred)):
        confusion[best_y_pred[i], y_true[i]] += 1
    im = ax3.imshow(confusion, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax3)
    ax3.set_title(f'Confusion Matrix (Best: σ={best_sigma:.0e})')
    ax3.set_xlabel('True Label')
    ax3.set_ylabel('Predicted Cluster')
    
    # Plot 4: Results Summary with Validation
    ax4 = axes[1, 1]
    ax4.axis('off')
    result_status = 'WITHIN TARGET' if is_success else 'OUTSIDE TARGET'
    metrics_text = f"""
BEST RESULTS SUMMARY
{'='*45}
Dataset:          {dataset_name_clean}
BEST ACC:         {best_acc*100:.2f}%
BEST NMI:         {best_nmi*100:.2f}%
Best Sigma:       {best_sigma:.0e}
Target ACC Range: {acc_min:.1f}% - {acc_max:.1f}%
Result:           {result_status}
"""
    ax4.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round',
                      facecolor='lightgreen' if is_success else 'wheat',
                      alpha=0.3))

    plt.tight_layout()
    fig_path = RESULT_DIR / f'{dataset_name_key}_visualization.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {fig_path}")

print("\n" + "="*80)
print("          ALL EXPERIMENTS FINISHED!")
print("="*80)