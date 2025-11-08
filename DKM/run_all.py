import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler, normalize

from diskmeans import diskmeans
from utils import clustering_accuracy


DATASETS = [
    {
        "name": "coil-20",
        "display": "COIL-20",
        "config": {
            "k": 20,
            "proj_dim": 19,
            "runs": 20,
            "max_iter": 100,
            "tol": 1e-4,
            "reg": 1e-6,
            "standardize": True,
            "normalize": "l2",
            "diskmeans_standardize": False,
        },
    },
    {
        "name": "usps",
        "display": "USPS",
        "config": {
            "k": 10,
            "proj_dim": 9,
            "runs": 20,
            "max_iter": 100,
            "tol": 1e-4,
            "reg": 1e-6,
            "standardize": True,
            "normalize": "l2",
            "diskmeans_standardize": False,
        },
    },
    {
        "name": "mnist-t",
        "display": "MNIST-T",
        "config": {
            "k": 10,
            "proj_dim": 9,
            "runs": 10,
            "max_iter": 120,
            "tol": 5e-4,
            "reg": 1e-6,
            "standardize": True,
            "normalize": "l2",
            "diskmeans_standardize": False,
        },
    },
    {
        "name": "mnist-s",
        "display": "MNIST-S",
        "config": {
            "k": 10,
            "proj_dim": 9,
            "runs": 10,
            "max_iter": 120,
            "tol": 5e-4,
            "reg": 1e-6,
            "standardize": True,
            "normalize": "l2",
            "diskmeans_standardize": False,
        },
    },
    {
        "name": "usf-humanid",
        "display": "USF HumanID",
        "config": {
            "k": 12,  # adjust if your ground truth uses a different class count
            "proj_dim": 11,
            "runs": 20,
            "max_iter": 100,
            "tol": 1e-4,
            "reg": 1e-6,
            "standardize": True,
            "normalize": "l2",
            "diskmeans_standardize": False,
        },
    },
    {
        "name": "umist",
        "display": "UMIST",
        "config": {
            "k": 20,
            "proj_dim": 19,
            "runs": 20,
            "max_iter": 100,
            "tol": 1e-4,
            "reg": 1e-6,
            "standardize": True,
            "normalize": "l2",
            "diskmeans_standardize": False,
        },
    },
    {
        "name": "yaleb",
        "display": "YALE-B",
        "config": {
            "k": 38,
            "proj_dim": 37,
            "runs": 20,
            "max_iter": 150,
            "tol": 1e-4,
            "reg": 1e-6,
            "standardize": True,
            "normalize": "l2",
            "diskmeans_standardize": False,
        },
    },
    {
        "name": "jaffe",
        "display": "JAFFE",
        "config": {
            "k": 10,
            "proj_dim": 9,
            "runs": 50,
            "max_iter": 100,
            "tol": 1e-4,
            "reg": 1e-6,
            "standardize": True,
            "normalize": "l2",
            "diskmeans_standardize": False,
        },
    },
    {
        "name": "pointing04",
        "display": "Pointing04",
        "config": {
            "k": 15,              # corrected class count
            "proj_dim": 14,
            "runs": 20,
            "max_iter": 120,
            "tol": 1e-4,
            "reg": 1e-6,
            "standardize": True,
            "normalize": "l2",
            "diskmeans_standardize": False,
        },
    },
    {
        "name": "mpeg7",
        "display": "MPEG7",
        "config": {
            "k": 30,              # corrected class count for the 30-class subset
            "proj_dim": 29,
            "runs": 30,
            "max_iter": 150,
            "tol": 1e-4,
            "reg": 1e-6,
            "standardize": True,
            "normalize": "l2",
            "diskmeans_standardize": False,
        },
    },
]


def load_npy(path: Path):
    arr = np.load(path, allow_pickle=True)

    if isinstance(arr, np.ndarray) and arr.ndim == 0 and arr.dtype == object:
        arr = arr.item()

    if isinstance(arr, dict):
        key_map_X = ["X", "data", "fea", "feature", "features"]
        key_map_y = ["y", "labels", "label", "gnd", "target"]

        def pick(d, candidates, required=True):
            for key in candidates:
                if key in d:
                    return d[key]
            if required:
                raise KeyError(
                    f"None of the candidate keys {candidates} were found in {path}"
                )
            return None

        X = pick(arr, key_map_X)
        y = pick(arr, key_map_y, required=False)
        return X, y

    if isinstance(arr, (tuple, list)):
        if len(arr) == 2:
            return arr[0], arr[1]
        raise ValueError(
            f"{path} contains a tuple/list of length {len(arr)}, expected length 2."
        )

    raise TypeError(f"Unsupported data type {type(arr)} in {path}.")


def apply_preprocessing(X: np.ndarray, cfg: dict) -> np.ndarray:
    X_proc = np.asarray(X, dtype=np.float64)

    if cfg.get("standardize", False):
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X_proc)

    norm_type = cfg.get("normalize")
    if norm_type:
        X_proc = normalize(X_proc, norm=norm_type)

    return X_proc


def evaluate_dataset(
    data_dir: Path,
    dataset: str,
    cfg: dict,
    seed: int = 0,
    display_name: str | None = None,
):
    file_path = data_dir / f"{dataset}.npy"
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist.")

    cfg_local = dict(cfg) if cfg else {}
    X, y = load_npy(file_path)
    X_proc = apply_preprocessing(X, cfg_local)

    n_samples, n_features = X_proc.shape

    k = cfg_local.get("k")
    if k is None:
        if y is None:
            raise ValueError(
                f"`k` is missing for dataset '{dataset}', and no labels are available."
            )
        k = int(len(np.unique(y)))
    else:
        k = int(k)

    proj_dim = cfg_local.get("proj_dim")
    if proj_dim is None:
        proj_dim = max(1, min(k - 1, n_features))
    else:
        proj_dim = int(proj_dim)
        proj_dim = max(1, min(proj_dim, n_features))

    runs_cfg = int(cfg_local.get("runs", 10))
    if runs_cfg <= 0:
        raise ValueError("`runs` must be a positive integer.")

    seed_sequence = cfg_local.get("seeds")
    if seed_sequence is not None:
        seed_sequence = list(seed_sequence)
        if not seed_sequence:
            raise ValueError("`seeds` list must contain at least one seed.")
    else:
        base_seed = int(cfg_local.get("base_seed", seed))
        seed_sequence = [base_seed + idx for idx in range(runs_cfg)]

    acc_vals: list[float] = []
    nmi_vals: list[float] = []
    ari_vals: list[float] = []
    iterations_vals: list[float] = []
    objective_vals: list[float] = []

    projection_method = cfg_local.get("proj_method")
    diskmeans_standardize = cfg_local.get("diskmeans_standardize", False)
    max_iter = int(cfg_local.get("max_iter", 100))
    tol = float(cfg_local.get("tol", 1e-4))
    reg = float(cfg_local.get("reg", 1e-6))

    for run_idx, current_seed in enumerate(seed_sequence):
        diskmeans_kwargs = dict(
            n_clusters=k,
            proj_dim=proj_dim,
            max_iter=max_iter,
            tol=tol,
            reg=reg,
            random_state=current_seed,
            standardize=diskmeans_standardize,
        )
        if projection_method is not None:
            diskmeans_kwargs["projection"] = projection_method

        result = diskmeans(X_proc, **diskmeans_kwargs)

        labels = np.asarray(result["labels"])
        history = result.get("history")
        if history is None:
            iterations_vals.append(np.nan)
        elif isinstance(history, (list, tuple)):
            iterations_vals.append(len(history))
        else:
            try:
                iterations_vals.append(len(history))  # type: ignore[arg-type]
            except TypeError:
                iterations_vals.append(np.nan)

        objective_value = result.get("objective")
        if objective_value is None and history and isinstance(history, (list, tuple)):
            last_entry = history[-1]
            if isinstance(last_entry, dict) and "objective" in last_entry:
                objective_value = last_entry["objective"]
        objective_vals.append(objective_value if objective_value is not None else np.nan)

        if y is not None:
            acc_vals.append(float(clustering_accuracy(y, labels)))
            nmi_vals.append(float(normalized_mutual_info_score(y, labels)))
            ari_vals.append(float(adjusted_rand_score(y, labels)))

    acc_arr = np.array(acc_vals, dtype=float)
    nmi_arr = np.array(nmi_vals, dtype=float)
    ari_arr = np.array(ari_vals, dtype=float)
    iterations_arr = np.array(iterations_vals, dtype=float)
    objective_arr = np.array(objective_vals, dtype=float)

    best_run_idx: int | None
    if acc_arr.size:
        best_run_idx = int(np.nanargmax(acc_arr))
    elif objective_arr.size and not np.all(np.isnan(objective_arr)):
        best_run_idx = int(np.nanargmin(objective_arr))
    else:
        best_run_idx = 0

    best_seed_value = seed_sequence[best_run_idx] if seed_sequence else None
    best_iterations = (
        iterations_arr[best_run_idx] if iterations_arr.size else np.nan
    )
    best_objective = (
        objective_arr[best_run_idx]
        if objective_arr.size and not np.isnan(objective_arr[best_run_idx])
        else np.nan
    )

    def mean_or_nan(arr: np.ndarray) -> float:
        return float(np.nanmean(arr)) if arr.size else np.nan

    def std_or_nan(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        return float(np.nanstd(arr, ddof=1)) if valid.size > 1 else np.nan

    metrics = {
        "dataset": dataset,
        "display_name": display_name or dataset,
        "samples": int(n_samples),
        "features": int(n_features),
        "clusters": int(k),
        "proj_dim": int(proj_dim),
        "runs": len(seed_sequence),
        "ACC_mean": mean_or_nan(acc_arr),
        "ACC_std": std_or_nan(acc_arr),
        "ACC_best": float(acc_arr[best_run_idx]) if acc_arr.size else np.nan,
        "NMI_mean": mean_or_nan(nmi_arr),
        "NMI_std": std_or_nan(nmi_arr),
        "NMI_best": float(nmi_arr[best_run_idx]) if nmi_arr.size else np.nan,
        "ARI_mean": mean_or_nan(ari_arr),
        "ARI_std": std_or_nan(ari_arr),
        "ARI_best": float(ari_arr[best_run_idx]) if ari_arr.size else np.nan,
        "iterations_mean": mean_or_nan(iterations_arr),
        "iterations_std": std_or_nan(iterations_arr),
        "iterations_best": float(best_iterations),
        "best_seed": int(best_seed_value) if best_seed_value is not None else np.nan,
        "best_objective": float(best_objective) if not np.isnan(best_objective) else np.nan,
        "objective_mean": mean_or_nan(objective_arr),
        "objective_std": std_or_nan(objective_arr),
    }

    return metrics


def prepare_display_table(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()

    percent_prefixes = ("ACC", "NMI", "ARI")
    percent_cols = [
        col for col in display_df.columns if col.startswith(percent_prefixes)
    ]
    for col in percent_cols:
        display_df[col] = display_df[col] * 100.0

    rename_map = {
        "display_name": "Dataset",
        "samples": "Samples",
        "features": "Features",
        "clusters": "k",
        "proj_dim": "Proj dim",
        "runs": "Runs",
        "ACC_mean": "ACC mean (%)",
        "ACC_std": "ACC std (%)",
        "ACC_best": "ACC best (%)",
        "NMI_mean": "NMI mean (%)",
        "NMI_std": "NMI std (%)",
        "NMI_best": "NMI best (%)",
        "ARI_mean": "ARI mean (%)",
        "ARI_std": "ARI std (%)",
        "ARI_best": "ARI best (%)",
        "iterations_mean": "Iter mean",
        "iterations_std": "Iter std",
        "iterations_best": "Iter best",
        "best_seed": "Best seed",
        "best_objective": "Best objective",
        "objective_mean": "Objective mean",
        "objective_std": "Objective std",
    }
    display_df = display_df.rename(columns=rename_map)

    int_columns = ["Samples", "Features", "k", "Proj dim", "Runs", "Best seed"]
    for col in int_columns:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").astype("Int64")

    column_order = [
        "Dataset",
        "Samples",
        "Features",
        "k",
        "Proj dim",
        "Runs",
        "ACC best (%)",
        "ACC mean (%)",
        "ACC std (%)",
        "NMI best (%)",
        "NMI mean (%)",
        "NMI std (%)",
        "ARI best (%)",
        "ARI mean (%)",
        "ARI std (%)",
        "Best seed",
        "Iter best",
        "Iter mean",
        "Iter std",
        "Best objective",
        "Objective mean",
        "Objective std",
    ]
    columns_present = [col for col in column_order if col in display_df.columns]
    return display_df[columns_present]


def save_table_image(df: pd.DataFrame, path: Path, dpi: int = 200, title: str | None = None):
    df_to_plot = df.copy()
    float_cols = df_to_plot.select_dtypes(include="float").columns
    for col in float_cols:
        df_to_plot[col] = df_to_plot[col].map(
            lambda x: f"{x:.2f}" if pd.notnull(x) else "NaN"
        )

    fig_height = max(2.5, 0.45 * len(df_to_plot.index) + 1.5)
    fig_width = max(8.0, 1.2 * len(df_to_plot.columns))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=14, pad=12, weight="bold")

    table = ax.table(
        cellText=df_to_plot.values,
        colLabels=df_to_plot.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#1f77b4")
            cell.set_text_props(weight="bold", color="white")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def format_percent(x: float | None) -> str:
    if x is None or np.isnan(x):
        return "NaN"
    return f"{x * 100:.2f}%"


def main():
    data_dir = Path("data")
    output_csv = Path("results_diskmeans.csv")
    output_png = Path("results_diskmeans.png")

    results: list[dict] = []

    for entry in DATASETS:
        dataset = entry["name"]
        display_name = entry.get("display", dataset)
        cfg = entry.get("config", {})

        print(f"\nRunning DisKmeans on {display_name} (key='{dataset}') ...")
        try:
            metrics = evaluate_dataset(
                data_dir=data_dir,
                dataset=dataset,
                cfg=cfg,
                seed=0,
                display_name=display_name,
            )
        except FileNotFoundError as err:
            print(f"  Skipping {display_name}: {err}")
            continue

        acc_mean = metrics.get("ACC_mean")
        acc_std = metrics.get("ACC_std")
        acc_best = metrics.get("ACC_best")
        best_seed = metrics.get("best_seed")
        print(
            "  ACC best = {best}  |  ACC mean = {mean}"
            "{std}".format(
                best=format_percent(acc_best),
                mean=format_percent(acc_mean),
                std=f" ± {format_percent(acc_std)}" if acc_std is not None and not np.isnan(acc_std) else "",
            )
        )

        results.append(metrics)

    if not results:
        print("No datasets were processed. Please check that your data files are available.")
        return

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv.resolve()}")

    display_df = prepare_display_table(results_df)
    save_table_image(
        display_df,
        output_png,
        title="DisKmeans Clustering Metrics (percent scale)",
    )
    print(f"Table image saved to {output_png.resolve()}")

    pd.set_option("display.max_columns", None)
    with pd.option_context("display.float_format", lambda v: f"{v:0.2f}"):
        print("\n=== Summary (percent scale) ===")
        print(display_df.to_string(index=False))


if __name__ == "__main__":
    main()