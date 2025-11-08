import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin
from typing import Optional, Tuple, Dict


def _compute_scatter_matrices(
    X: np.ndarray,
    labels: np.ndarray,
    reg: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples, n_features = X.shape
    classes = np.unique(labels)
    overall_mean = X.mean(axis=0, keepdims=True)

    S_B = np.zeros((n_features, n_features), dtype=float)
    S_W = np.zeros((n_features, n_features), dtype=float)

    for c in classes:
        X_c = X[labels == c]
        if X_c.size == 0:
            continue
        mean_c = X_c.mean(axis=0, keepdims=True)
        n_c = X_c.shape[0]

        diff_mean = (mean_c - overall_mean)
        S_B += n_c * diff_mean.T @ diff_mean

        diff_class = X_c - mean_c
        S_W += diff_class.T @ diff_class

    S_W += reg * np.eye(n_features)
    return S_B, S_W


def _solve_projection(
    S_B: np.ndarray,
    S_W: np.ndarray,
    proj_dim: int
) -> np.ndarray:
    
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S_W) @ S_B)
    
    sorted_idx = np.argsort(-eigvals.real)
    W = eigvecs[:, sorted_idx[:proj_dim]].real
    return W


def diskmeans(
    X: np.ndarray,
    n_clusters: int,
    proj_dim: Optional[int] = None,
    max_iter: int = 50,
    tol: float = 1e-4,
    reg: float = 1e-6,
    random_state: Optional[int] = 42,
    standardize: bool = True,
) -> Dict[str, np.ndarray]:
   
    rng = np.random.default_rng(random_state)
    X_proc = X.copy()

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X_proc)

    n_samples, n_features = X_proc.shape
    if proj_dim is None:
        proj_dim = min(n_clusters - 1, n_features)

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=300,
        random_state=random_state,
        algorithm="lloyd"
    )
    labels = kmeans.fit_predict(X_proc)
    prev_labels = labels.copy()

    history = []

    for iteration in range(max_iter):
        S_B, S_W = _compute_scatter_matrices(X_proc, labels, reg=reg) 
        W = _solve_projection(S_B, S_W, proj_dim=proj_dim)

        embedded = X_proc @ W

        kmeans_emb = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            max_iter=300,
            random_state=random_state,
            algorithm="lloyd"
        )
        labels = kmeans_emb.fit_predict(embedded)

        history.append({
            "iteration": iteration,
            "labels": labels.copy(),
            "projection": W.copy(),
            "embedded": embedded.copy()
        })

        diff_ratio = np.mean(labels != prev_labels)
        if diff_ratio < tol:
            break
        prev_labels = labels.copy()

    centers_orig = np.zeros((n_clusters, n_features))
    centers_emb = np.zeros((n_clusters, proj_dim))
    for c in range(n_clusters):
        mask = labels == c
        if np.any(mask):
            centers_orig[c] = X_proc[mask].mean(axis=0)
            centers_emb[c] = embedded[mask].mean(axis=0)

    result = {
        "labels": labels,
        "projection": W,
        "embedded": embedded,
        "centers_orig": centers_orig,
        "centers_emb": centers_emb,
        "history": history,
    }

    if scaler is not None:
        result["centers_orig_denorm"] = scaler.inverse_transform(centers_orig)

    return result


if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.metrics import normalized_mutual_info_score

    digits = load_digits()
    X = digits.data
    y = digits.target

    out = diskmeans(
        X,
        n_clusters=10,
        proj_dim=9,        
        max_iter=20,
        tol=1e-3,
        random_state=0
    )

    labels = out["labels"]
    nmi = normalized_mutual_info_score(y, labels)
    print(f"NMI = {nmi:.4f}")
    print(f"iterations:{len(out['history'])}")