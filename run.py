"""Orchestrator: load data, train models, evaluate, and save the best model."""

import pathlib

import joblib

from src.data import load_and_split
from src.evaluate import evaluate_all, print_comparison_table
from src.train import train_all

MODELS_DIR = pathlib.Path("models")


def main() -> None:
    # ── Data ────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = load_and_split()

    # ── Train ───────────────────────────────────────────────────────────
    results = train_all(X_train, y_train)

    # ── Evaluate ────────────────────────────────────────────────────────
    comparison = evaluate_all(results, X_test, y_test)
    print_comparison_table(comparison)

    # ── Save best model ─────────────────────────────────────────────────
    best_name = comparison.iloc[0]["Model"]
    best_model = results[best_name].best_estimator_
    MODELS_DIR.mkdir(exist_ok=True)
    save_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, save_path)
    print(f"\nBest model ({best_name}) saved to {save_path}")


if __name__ == "__main__":
    main()
