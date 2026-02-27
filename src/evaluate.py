"""Model evaluation: metrics computation and comparison table."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Compute all metrics for a single fitted model (or GridSearchCV)."""
    estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Confusion Matrix": cm,
    }


def evaluate_all(results: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate every model and return a comparison DataFrame sorted by ROC-AUC."""
    rows = []
    for name, model in results.items():
        metrics = evaluate_model(name, model, X_test, y_test)
        rows.append(metrics)

        print(f"\n{'='*40}")
        print(f" {name}")
        print(f"{'='*40}")
        print(f"  Accuracy  : {metrics['Accuracy']:.4f}")
        print(f"  Precision : {metrics['Precision']:.4f}")
        print(f"  Recall    : {metrics['Recall']:.4f}")
        print(f"  F1        : {metrics['F1']:.4f}")
        print(f"  ROC-AUC   : {metrics['ROC-AUC']:.4f}")
        print(f"  Confusion Matrix:\n{metrics['Confusion Matrix']}")

    df = pd.DataFrame(rows)
    df = df.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return df


def print_comparison_table(df: pd.DataFrame) -> None:
    """Pretty-print the comparison table (excludes confusion matrix column)."""
    display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    print("\n" + "=" * 70)
    print(" MODEL COMPARISON (sorted by ROC-AUC)")
    print("=" * 70)
    print(df[display_cols].to_string(index=False, float_format="{:.4f}".format))
    print("=" * 70)
