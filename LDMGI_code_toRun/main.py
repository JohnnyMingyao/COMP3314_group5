"""
Main script to run LDMGI clustering on COIL-20 dataset
Following the paper's experimental protocol exactly
"""

import os
import numpy as np
from sklearn.cluster import KMeans

from download_COIL import load_coil20_images
from LDMGI import LDMGI_complete, evaluate_clustering
from load_jaffe import load_jaffe_images
from load_mnistS import load_mnistS_images
from load_mnistT import load_mnistT_images
from load_mpeg7 import load_mpeg7_images
from load_pointing04 import load_pointing04_images
from load_umist import load_umist_images
from load_usps import load_usps_images
from load_yaleb import load_yaleb_images




def main():
    """
    Complete pipeline for COIL-20 image clustering with LDMGI
    Following the paper's experimental protocol:
    - 20 independent trials with random initializations
    - Test lambda in {10^-8, 10^-6, 10^-4, 10^-2, 10^0, 10^2, 10^4, 10^6, 10^8}
    - Report best results in terms of ACC and NMI
    """
    k = 5   #Number of neighbors (fixed as in paper)
    c =20  #class number to be changed
    print("="*70)
    print("LDMGI Image Clustering")
    print("Paper: Local Discriminant Models and Global Integration (2011)")
    print("="*70)
    
    
    
    try:
        X, labels_true = load_coil20_images()
        ##X, labels_true = load_jaffe_images()
        ##X, labels_true = load_mnistS_images()
        ##X, labels_true = load_mnistT_images()
        ##X, labels_true = load_mpeg7_images()
        ##X, labels_true = load_pointing04_images()
        ##X, labels_true = load_umist_images()
        ##X, labels_true = load_yaleb_images()
        ##X, labels_true = load_usps_images()
        setName = "coil_20"
    except FileNotFoundError as e:
        print(f"\n{'!'*70}")
        print(f"ERROR: {e}")
        print(f"{'!'*70}")
        return None, None
    
    #Check dataset size
    
    
    #Baseline K-means
    print("\n" + "="*70)
    print("Baseline: K-means ")
    print("="*70)
    
    n_trials = 1#may change later
    best_kmeans_acc = 0
    best_kmeans_nmi = 0
    
    for trial in range(n_trials):
        kmeans = KMeans(n_clusters=c, n_init=10, max_iter=300, random_state=trial)
        baseline_labels = kmeans.fit_predict(X)
        acc, nmi = evaluate_clustering(labels_true, baseline_labels)
        print("The base line lable is:",baseline_labels)
        print("The actual true label is:", list(labels_true))
        if acc > best_kmeans_acc:
            best_kmeans_acc = acc
            best_kmeans_nmi = nmi
        
        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial+1}/{n_trials}: Best so far - ACC={best_kmeans_acc*100:.2f}%, NMI={best_kmeans_nmi*100:.2f}%")
    
    print(f"\nBest K-means (from {n_trials} trials): ACC={best_kmeans_acc*100:.2f}%, NMI={best_kmeans_nmi*100:.2f}%")
    
    #Run LDMGI following paper's protocol
    print("\n" + "="*70)
    print("LDMGI Clustering - Following Paper's Protocol")
    print("="*70)
    print(f"Fixed parameters: k=5, c=20")
    print(f"Lambda values: {{10^-8, 10^-6, 10^-4, 10^-2, 10^0, 10^2, 10^4, 10^6, 10^8}}")
    print(f"Trials per lambda: {n_trials} independent runs")
    print(f"Total runs: {n_trials * 9} = 180 (as mentioned in the paper)")
    print("="*70)
    
    #Lambda values as specified in the paper
    lambda_values = [1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8]
    
    
    
    #Store best results across ALL trials
    global_best_acc = 0
    global_best_nmi = 0
    global_best_lambda = None
    global_best_labels = None
    
    #Store best results per lambda
    results_per_lambda = []
    
    total_runs = 0
    
    for lambda_param in lambda_values:
        print(f"\n{'='*70}")
        print(f"Lambda = {lambda_param:.0e}")
        print(f"{'='*70}")
        
        best_acc_for_lambda = 0
        best_nmi_for_lambda = 0
        best_labels_for_lambda = None
        
        #20 trails as the pape specifies 
        for trial in range(n_trials):
            total_runs += 1
            
            #the paper didn't sepcify rendom seed, I just use a random one.
            np.random.seed(trial)
            
            # Run LDMGI
            cluster_labels, G_star = LDMGI_complete(
                X, 
                k=k,
                c=c,
                lambda_param=lambda_param,
                random_state=trial  # Different seed for each trial
            )
            
            #Evaluate
            acc, nmi = evaluate_clustering(labels_true, cluster_labels)
            print("The label predicted is ",cluster_labels)
            
            #Track best for this lambda
            if acc > best_acc_for_lambda:
                best_acc_for_lambda = acc
                best_nmi_for_lambda = nmi
                best_labels_for_lambda = cluster_labels
            
            
            # get global best acc
            if acc > global_best_acc:
                global_best_acc = acc
                global_best_nmi = nmi
                global_best_lambda = lambda_param
                global_best_labels = cluster_labels
            
            
            # Print progress 
            if (trial + 1) % 5 == 0:
                print(f"  Trial {trial+1}/{n_trials}: Best for λ={lambda_param:.0e} - ACC={best_acc_for_lambda*100:.2f}%, NMI={best_nmi_for_lambda*100:.2f}%")
        
        print(f"\n  Best result for λ={lambda_param:.0e} (from {n_trials} trials):") #The print part from here is generated by AI
        print(f"    ACC: {best_acc_for_lambda*100:.2f}%")
        print(f"    NMI: {best_nmi_for_lambda*100:.2f}%")
        
        results_per_lambda.append({
            'lambda': lambda_param,
            'acc': best_acc_for_lambda,
            'nmi': best_nmi_for_lambda
        })
    
    # Report final results 
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\nTotal clustering runs performed: {total_runs}")
    
    print(f"\nBaseline K-means (best from {n_trials} trials):")
    print(f"  ACC: {best_kmeans_acc*100:.2f}%")
    print(f"  NMI: {best_kmeans_nmi*100:.2f}%")
    
    print(f"\nBest LDMGI (λ={global_best_lambda:.0e}, best from {n_trials} trials):")
    print(f"  ACC: {global_best_acc*100:.2f}%")
    print(f"  NMI: {global_best_nmi*100:.2f}%")
    
    print(f"\nPaper's reported results:")
    print(f"  ACC: 82.4%")
    print(f"  NMI: 88.7%")
    
    print(f"\nDifference from paper:")
    print(f"  ACC: {(global_best_acc - 0.824)*100:+.2f} percentage points")
    print(f"  NMI: {(global_best_nmi - 0.887)*100:+.2f} percentage points")
    
    # Results table
    print("\n" + "-"*70)
    print("Best Results per Lambda (each from 20 trials) ("+setName+"):")
    print("-"*70)
    print(f"{'Lambda':<15} {'ACC (%)':<15} {'NMI (%)':<15}")
    print("-" * 45)
    for r in results_per_lambda:
        marker = " ← BEST" if r['lambda'] == global_best_lambda else ""
        print(f"{r['lambda']:<15.0e} {r['acc']*100:<15.2f} {r['nmi']*100:<15.2f}{marker}")
    print("-"*70)
    
    # Visualize
    """
    print("\nGenerating visualizations...")
    visualize_results(X, labels_true, global_best_labels, n_samples=20, 
                     save_path='coil20_clustering_results.png')
    """
    
    return results_per_lambda, global_best_labels


if __name__ == "__main__":
    results, best_labels = main()