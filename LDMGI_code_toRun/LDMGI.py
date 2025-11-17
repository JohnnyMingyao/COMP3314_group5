"""
LDMGI (Local Discriminant Models and Global Integration) implementation
Following the paper's equations exactly
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh  # Add this import at the top





def LDMGI_complete(X, k=5, c=20, lambda_param=1e-4, random_state=None ):
   
    n, d = X.shape
    
    print(f"Running LDMGI: n={n}, d={d}, k={k}, c={c}, λ={lambda_param:.0e}")
    
    #Step 1: Build Global Laplacian Matrix L
    L = np.zeros((n, n))
    
    print("  Finding k-nearest neighbors...")
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    #For each data point, construct local discriminant model
    for i in range(n):

        N_i = indices[i] 
        m = len(N_i)  # m = k
        
        #Extract local data matrix X_i
        X_i = X[N_i, :]  # Shape: (m, d)
        H_k = np.eye(m) - (1.0 / m) * np.ones((m, m))
        X_tilde_i = X_i.T@H_k

        gram_matrix = X_tilde_i.T @ X_tilde_i + lambda_param * np.eye(m)
        
        try:
            gram_inv = np.linalg.inv(gram_matrix)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            gram_inv = np.linalg.pinv(gram_matrix)
        L_i = H_k @ gram_inv @ H_k
        
        #Integrate into global L
        #S_i is selection matrix, so this is just adding L_i to the submatrix
        for p in range(m):
            for q in range(m):
                L[N_i[p], N_i[q]] += L_i[p, q]
    
    # Symmetrize the Laplacian matrix
    print("  Symmetrizing Laplacian...")
    L = (L + L.T) / 2.0
    
    # Step 2: Eigenvalue Decomposition
    print("  Computing eigendecomposition...")
    
    # Use scipy's eigh for symmetric matrices (more stable)
    eigenvalues, eigenvectors = eigh(L)
    G_star = eigenvectors[:, 1:c+1]  #Shape: (n, c)
    
    print(f"  Selected eigenvalues: {eigenvalues[1:c+1]}")
    
   # Step 3: Normalization (Equation 23)


    diag_GGT = np.sum(G_star * G_star, axis=1)
    diag_GGT = np.maximum(diag_GGT, 1e-10)  # Avoid division by zero
    diag_inv_sqrt = 1.0 / np.sqrt(diag_GGT)
    Y_star = diag_inv_sqrt[:, np.newaxis] * G_star

    # Step 4: iteratively find best Y
    print("  Discretizing to binary cluster assignment...")
    best_Y = None
    best_obj = float('inf')

    if random_state is not None:
        np.random.seed(random_state)

    # Multiple random initializations
    for init_idx in range(10):
        R = np.eye(c)
        prev_labels = None
        #find the R
        for i in range(100):
            Y_star_R = Y_star @ R 
            new_labels = np.argmax(Y_star_R, axis=1)
            
            #Check convergence
            if prev_labels is not None and np.array_equal(new_labels, prev_labels):
                break
            
            prev_labels = new_labels
            
            #Update Y (binary indicator matrix)
            Y = np.zeros((n, c))
            Y[np.arange(n), new_labels] = 1
            
            #Verify constraint: Y 1_c = 1_n (each row sums to 1)
            assert np.allclose(Y.sum(axis=1), 1.0), "Constraint Y1_c = 1_n violated"

            U, _, Vt = np.linalg.svd(Y_star.T @ Y, full_matrices=False)
            R = U @ Vt  # (c, c)
            

        # Compute objective value
        obj = np.linalg.norm(Y - Y_star @ R, 'fro') ** 2
        
        if obj < best_obj:
            best_obj = obj
            best_Y = Y.copy()

    print(f"  Best objective value: {best_obj:.4f}")
    cluster_labels = np.argmax(best_Y, axis=1)

    return cluster_labels, G_star


    

def evaluate_clustering(labels_true, labels_pred):
    """
    Evaluate clustering using ACC and NMI
    
    Parameters:
    predicted label and the actual label
    
    Returns:
    --------
    acc : float
        Clustering accuracy (0 to 1)
    nmi : float
        Normalized mutual information (0 to 1)
    """
    
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    
    n = len(labels_true)
    assert len(labels_pred) == n
    
    # Compute NMI
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    
    # For ACC, we need to find the optimal mapping from predicted to true labels
    
    # Get the number of clusters
    n_clusters = max(labels_pred.max(), labels_true.max()) + 1
    
    # Build confusion/cost matrix
    # w[i, j] = number of samples with predicted label i and true label j
    w = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    
    for i in range(n):
        w[labels_pred[i], labels_true[i]] += 1
    
    # Find optimal mapping using Hungarian algorithm
    # We want to maximize the sum of w[i, map[i]]
    # linear_sum_assignment minimizes, so we use negative
    # OR use maximize=True if available
    from scipy import __version__ as scipy_version
    if tuple(map(int, scipy_version.split('.')[:2])) >= (1, 4):
        # scipy >= 1.4.0 supports maximize parameter
        pred_ind, true_ind = linear_sum_assignment(w, maximize=True)
    else:
        # older scipy versions
        pred_ind, true_ind = linear_sum_assignment(-w)
    
    # Create the mapping function: map[predicted_label] = true_label
    mapping = np.zeros(n_clusters, dtype=np.int64)
    mapping[pred_ind] = true_ind
    
    # Apply the mapping to predicted labels
    labels_pred_mapped = mapping[labels_pred]
    
    # Calculate accuracy: count how many match after mapping
    acc = np.sum(labels_pred_mapped == labels_true) / n
    
    return acc, nmi
