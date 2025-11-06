import numpy as np
from scipy.optimize import linear_sum_assignment


def clustering_accuracy(y_true, y_pred):
    """使用 Hungarian 方法匹配簇标签，返回 ACC。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    assert y_true.shape == y_pred.shape

    labels = np.unique(y_true)
    clusters = np.unique(y_pred)
    cost = np.zeros((labels.size, clusters.size), dtype=int)

    for i, label in enumerate(labels):
        for j, cluster in enumerate(clusters):
            cost[i, j] = np.sum((y_true == label) & (y_pred == cluster))

    row_ind, col_ind = linear_sum_assignment(cost.max() - cost)
    correct = cost[row_ind, col_ind].sum()
    return correct / y_true.size