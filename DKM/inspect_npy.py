import argparse
import numpy as np
from pathlib import Path

def inspect(path: Path):
    arr = np.load(path, allow_pickle=True)
    print(f"\n=== {path.name} ===")
    print(f"type: {type(arr)}")

    if isinstance(arr, np.ndarray):
        print(f"ndim: {arr.ndim}, dtype: {arr.dtype}, shape: {arr.shape}")
        if arr.ndim == 0 and arr.dtype == object:
            obj = arr.item()
            print(f" -> item type: {type(obj)}")
            if isinstance(obj, dict):
                print(f" -> dict keys: {list(obj.keys())}")
            elif isinstance(obj, (tuple, list)):
                print(f" -> length: {len(obj)}")
    else:
        print("value:", arr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help=".npy file path")
    args = parser.parse_args()

    for file in args.files:
        inspect(Path(file))