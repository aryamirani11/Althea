"""Leakage audit: why a naive classifier on this dataset scores 0.94 ROC-AUC.

The obvious approach is to train a classifier on the raw columns to
separate label 2 ("Crohn's") from label 0 ("control"). Three things
explain its score, and none of them is biology:

1. Label 0 means "status unknown", and every label-0 row is a parent.
   The task was really "child or parent?".
2. Children's alleles are stored in inheritance order (father, mother);
   parents' alleles are stored sorted. A single feature — how often a
   person's alleles appear in non-sorted order — recovers who is a child.
3. Re-encode the exact same genotypes in an order-free way and the
   classifier drops to chance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import AFFECTED, ID_COLS, MISSING, N_LOCI, alleles, allele_coding, dosage, is_founder

SEED = 42


def _cv() -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)


def _auc(model, X, y) -> dict:
    scores = cross_val_score(model, X, y, cv=_cv(), scoring="roc_auc")
    return {"auc_mean": round(float(scores.mean()), 3), "auc_sd": round(float(scores.std()), 3)}


def _logreg():
    return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=5000))


def naive_labels(df: pd.DataFrame) -> np.ndarray:
    return (df["crohn"] == AFFECTED).astype(int).to_numpy()


def order_flip_rate(df: pd.DataFrame) -> np.ndarray:
    """Share of each person's heterozygous SNPs stored in non-canonical order.

    "Canonical" is the order parents' heterozygous genotypes are stored in
    at that SNP. Parents score ~0 by construction; phased children score ~0.5.
    """
    a1, a2 = alleles(df)
    founders = is_founder(df)
    het = (a1 != a2) & (a1 != MISSING) & (a2 != MISSING)
    flipped = np.zeros_like(het)
    for j in range(N_LOCI):
        f_het = het[founders, j]
        if not f_het.any():
            continue
        pairs = pd.Series(list(zip(a1[founders, j][f_het], a2[founders, j][f_het])))
        first, _ = pairs.value_counts().index[0]
        flipped[:, j] = het[:, j] & (a1[:, j] != first)
    return flipped.sum(1) / np.maximum(het.sum(1), 1)


def run_audit(df: pd.DataFrame) -> tuple[dict, np.ndarray]:
    y = naive_labels(df)
    founders = is_founder(df)
    raw = df.drop(columns=ID_COLS + ["crohn"])  # what a naive pipeline would use
    a1, a2 = alleles(df)
    order_free = dosage(a1, a2, allele_coding(df))
    flip = order_flip_rate(df)

    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    return {
        "label_breakdown": {
            "label_2_children": int(((y == 1) & ~founders).sum()),
            "label_2_parents": int(((y == 1) & founders).sum()),
            "label_0_parents": int(((y == 0) & founders).sum()),
            "label_0_children": int(((y == 0) & ~founders).sum()),
        },
        "order_flip_rate": {
            "parents_mean": round(float(flip[founders].mean()), 3),
            "children_mean": round(float(flip[~founders].mean()), 3),
        },
        "auc": {
            "naive_raw_columns_logreg": _auc(_logreg(), raw, y),
            "naive_raw_columns_random_forest": _auc(rf, raw, y),
            "order_flip_rate_only": _auc(_logreg(), flip.reshape(-1, 1), y),
            "same_genotypes_order_removed_logreg": _auc(_logreg(), order_free, y),
            "same_genotypes_order_removed_random_forest": _auc(rf, order_free, y),
        },
    }, flip
