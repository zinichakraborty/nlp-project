# probe_evaluate.py

import argparse
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained probe on embeddings; report accuracy + confusion matrix."
    )
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--probe_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="probe_results")
    return parser.parse_args()


def main():
    args = parse_args()
    emb_path = Path(args.embeddings_path)
    probe_path = Path(args.probe_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(emb_path)
    X = data["X"]
    y = data["y"]

    print(f"Loaded embeddings: {X.shape}, labels: {y.shape}")

    clf = joblib.load(probe_path)
    print(f"Loaded probe from {probe_path}")

    y_pred = clf.predict(X)

    acc = accuracy_score(y, y_pred)
    print(f"\nOverall accuracy (all examples): {acc:.4f}")

    cm = confusion_matrix(y, y_pred, labels=[0, 1, 2, 3, 4])

    print("\nConfusion matrix (rows = true position, cols = predicted position)")
    print("Positions are 0..4 internally; interpret as 1..5 in your narrative.\n")
    print(cm)

    print("\nConfusion matrix with 1-based labels:")
    header = "     " + "  ".join([f"pred_{i}" for i in range(1, 6)])
    print(header)
    for i, row in enumerate(cm, start=1):
        row_str = " ".join([f"{v:5d}" for v in row])
        print(f"true_{i} {row_str}")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.colorbar()
    plt.xticks(range(5), [f"Pred {i}" for i in range(1, 6)])
    plt.yticks(range(5), [f"True {i}" for i in range(1, 6)])
    plt.xlabel("Predicted Position")
    plt.ylabel("True Position")
    plt.title("Sentence Position Probe Confusion Matrix")

    for i in range(5):
        for j in range(5):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    probe_name = Path(args.probe_path).stem
    cm_path = output_dir / f"{probe_name}.confusion_matrix.png"
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved confusion matrix plot to: {cm_path}")

    print("\nJob finished successfully.")


if __name__ == "__main__":
    main()