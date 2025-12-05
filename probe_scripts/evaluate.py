import argparse
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()
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

    clf = joblib.load(probe_path)

    y_pred = clf.predict(X)

    acc = accuracy_score(y, y_pred)
    print(f"\nOverall accuracy: {acc:.4f}")

    cm = confusion_matrix(y, y_pred, labels=[0, 1, 2, 3, 4])

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
    cm_path = output_dir / f"{probe_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()