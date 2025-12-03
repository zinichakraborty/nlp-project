import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Train position probe on sentence embeddings.")
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    emb_path = Path(args.embeddings_path)

    print(f"Loading embeddings from {emb_path}...")
    data = np.load(emb_path)
    X = data["X"]
    y = data["y"]
    print(f"Loaded embeddings: {X.shape}, labels: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=y,
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    test_split_path = Path(args.output_model_path).with_suffix(".test_split.npz")
    np.savez(test_split_path, X=X_test, y=y_test)
    print(f"Saved test split embeddings to {test_split_path}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=300,
        )),
    ])

    param_grid = {
        "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    print("Starting grid search over C...")
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    best_model = grid.best_estimator_

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1, 2, 3, 4])
    print("\nConfusion matrix (test set, rows=true, cols=pred):")
    print(cm)

    out_path = Path(args.output_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out_path)
    print(f"Saved probe model to {out_path}")


if __name__ == "__main__":
    main()
