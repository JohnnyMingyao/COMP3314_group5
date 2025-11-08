import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from diskmeans import diskmeans  
from utils import clustering_accuracy 


def load_dataset(path: Path):
    data = np.load(path)
    X = data["X"]
    y = data.get("y", None)  
    return X, y


def main(args):
    np.random.seed(args.seed)

    data_path = Path(args.data_dir) / f"{args.dataset}.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} inexistence")

    X, y = load_dataset(data_path)

    result = diskmeans(
        X,
        n_clusters=args.k,
        proj_dim=args.proj_dim,
        max_iter=args.max_iter,
        tol=args.tol,
        reg=args.reg,
        random_state=args.seed,
        standardize=True,
    )

    y_pred = result["labels"]

    print(f"[{args.dataset}] iterations: {len(result['history'])}")
    if y is not None:
        acc = clustering_accuracy(y, y_pred)
        nmi = normalized_mutual_info_score(y, y_pred)
        ari = adjusted_rand_score(y, y_pred)
        print(f"  ACC = {acc * 100:.2f}%")
        print(f"  NMI = {nmi:.4f}")
        print(f"  ARI = {ari:.4f}")
    else:
        print("  No real labels were provided, so it is impossible to calculate ACC/NMI/ARI.")

    if args.save_embedded:
        out_path = Path(args.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path / f"{args.dataset}_diskmeans_output.npz",
            labels=y_pred,
            embedded=result["embedded"],
            projection=result["projection"],
        )
        print(f"  The projection result has been saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Data folder")
    parser.add_argument("--dataset", required=True, help="Data file name (without extension)")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters")
    parser.add_argument("--proj-dim", type=int, default=None, help="Projection dimension")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-embedded", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    main(args)