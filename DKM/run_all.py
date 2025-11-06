import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from diskmeans import diskmeans
from utils import clustering_accuracy


DATASET_CONFIG = {
    "coil-20":   {"k": 20, "proj_dim": 19},
    "jaffe":     {"k": 10, "proj_dim": 9},
    "mnist-s":   {"k": 10, "proj_dim": 9},
    "mnist-t":   {"k": 10, "proj_dim": 9},
    "mpeg7":     {"k": 70, "proj_dim": 69},
    "pointing04": {"k": 13, "proj_dim": 12},
    "umist":     {"k": 20, "proj_dim": 19},
    "usps":      {"k": 10, "proj_dim": 9},
    "yaleb":     {"k": 38, "proj_dim": 37},
}


def load_npy(path: Path):
    arr = np.load(path, allow_pickle=True)

    # .npy 中存的是 object 类型的标量（通常是 dict 或 tuple）
    if isinstance(arr, np.ndarray) and arr.ndim == 0 and arr.dtype == object:
        arr = arr.item()

    # dict 情况：尝试多种常见键名
    if isinstance(arr, dict):
        key_map_X = ["X", "data", "fea", "feature", "features"]
        key_map_y = ["y", "labels", "label", "gnd", "target"]

        def pick(d, candidates, required=True):
            for key in candidates:
                if key in d:
                    return d[key]
            if required:
                raise KeyError(
                    f"No available key name could be found in {path}.(candidate: {candidates})"
                )
            return None

        X = pick(arr, key_map_X)
        y = pick(arr, key_map_y, required=False)
        return X, y

    # tuple / list 情况：默认 (X, y)
    if isinstance(arr, (tuple, list)):
        if len(arr) == 2:
            return arr[0], arr[1]
        raise ValueError(f"{path} tuple/list length is {len(arr)},unable to parse.")

    raise TypeError(
        f"{path} the obtained type is {type(arr)},no support"
    )


def evaluate_dataset(data_dir: Path, dataset: str, cfg: dict, seed: int = 0):
    file_path = data_dir / f"{dataset}.npy"
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} inexistence")

    X, y = load_npy(file_path)

    result = diskmeans(
        X,
        n_clusters=cfg["k"],
        proj_dim=cfg.get("proj_dim"),
        max_iter=cfg.get("max_iter", 20),
        tol=cfg.get("tol", 1e-3),
        reg=cfg.get("reg", 1e-6),
        random_state=seed,
        standardize=True,
    )

    labels = result["labels"]
    metrics = {
        "dataset": dataset,
        "clusters": cfg["k"],
        "proj_dim": cfg.get("proj_dim"),
        "iterations": len(result["history"]),
    }

    if y is not None:
        metrics["ACC"] = clustering_accuracy(y, labels)
        metrics["NMI"] = normalized_mutual_info_score(y, labels)
        metrics["ARI"] = adjusted_rand_score(y, labels)
    else:
        metrics["ACC"] = np.nan
        metrics["NMI"] = np.nan
        metrics["ARI"] = np.nan

    return metrics


def main():
    data_dir = Path("data")
    output_csv = Path("results_diskmeans.csv")

    results = []
    for dataset, cfg in DATASET_CONFIG.items():
        print(f"Running DisKmeans on {dataset} ...")
        metrics = evaluate_dataset(data_dir, dataset, cfg)
        print(
            f"  ACC={metrics['ACC']*100:.2f}% "
            f"NMI={metrics['NMI']:.4f} "
            f"ARI={metrics['ARI']:.4f}"
        )
        results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n all the results have been saved to {output_csv.resolve()}")

    print("\n=== Summary ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()