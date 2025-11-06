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
    """计算类间散度 S_B 与类内散度 S_W。"""
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

    # 数值稳定性：在对角线上加一个小的正则
    S_W += reg * np.eye(n_features)
    return S_B, S_W


def _solve_projection(
    S_B: np.ndarray,
    S_W: np.ndarray,
    proj_dim: int
) -> np.ndarray:
    """解广义特征值问题 S_B v = λ S_W v，选取最大的 proj_dim 个特征向量。"""
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S_W) @ S_B)
    # 排序：按特征值从大到小
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
    """
    Discriminative K-means（DisKmeans）实现。

    参数
    ----
    X : (n_samples, n_features) 输入数据。
    n_clusters : 聚类数。
    proj_dim : 投影维度；若为 None，则默认为 min(n_clusters - 1, n_features)。
    max_iter : 最大迭代次数。
    tol : 相邻两次聚类标签变化比例低于 tol 时停止。
    reg : 类内散度矩阵的对角正则系数。
    random_state : 随机种子。
    standardize : 是否对数据做零均值单位方差标准化。

    返回
    ----
    dict，包含：
        labels       : 最终聚类标签。
        projection   : 判别投影矩阵 W。
        embedded     : 投影后的特征 Y = X W。
        centers_orig : 原空间中的簇中心（基于最终标签均值）。
        centers_emb  : 判别子空间中的簇中心。
        history      : 每次迭代的标签、投影等信息。
    """
    rng = np.random.default_rng(random_state)
    X_proc = X.copy()

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X_proc)

    n_samples, n_features = X_proc.shape
    if proj_dim is None:
        proj_dim = min(n_clusters - 1, n_features)

    # 初始化：直接用普通 KMeans
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
        # Step 1: 计算散度矩阵
        S_B, S_W = _compute_scatter_matrices(X_proc, labels, reg=reg)
        # Step 2: 求判别投影
        W = _solve_projection(S_B, S_W, proj_dim=proj_dim)

        # Step 3: 在子空间进行 KMeans
        embedded = X_proc @ W

        kmeans_emb = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            max_iter=300,
            random_state=random_state,
            algorithm="lloyd"
        )
        labels = kmeans_emb.fit_predict(embedded)

        # 记录历史
        history.append({
            "iteration": iteration,
            "labels": labels.copy(),
            "projection": W.copy(),
            "embedded": embedded.copy()
        })

        # 收敛判定
        diff_ratio = np.mean(labels != prev_labels)
        if diff_ratio < tol:
            break
        prev_labels = labels.copy()

    # 计算最终中心
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

    # 如果做了标准化，把中心还原回原始尺度（可选）
    if scaler is not None:
        result["centers_orig_denorm"] = scaler.inverse_transform(centers_orig)

    return result


# ------------------ 使用示例 ------------------
if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.metrics import normalized_mutual_info_score

    digits = load_digits()
    X = digits.data
    y = digits.target

    out = diskmeans(
        X,
        n_clusters=10,
        proj_dim=9,        # 经验上可设为 k-1
        max_iter=20,
        tol=1e-3,
        random_state=0
    )

    labels = out["labels"]
    nmi = normalized_mutual_info_score(y, labels)
    print(f"NMI = {nmi:.4f}")
    print(f"iterations:{len(out['history'])}")