"""An honest predictive baseline: case vs pseudo-control.

Every affected child is paired with a pseudo-control built from the alleles
their parents did not pass on. Cross-validation keeps both members of a
family in the same fold, so the model never sees a family it is tested on.
The question it answers is narrow and fair: from 103 SNPs in one region,
how well can you tell inherited genotypes from non-inherited ones?
"""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_REPEATS = 10
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def _model() -> GridSearchCV:
    pipe = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        LogisticRegression(max_iter=5000),
    )
    return GridSearchCV(
        pipe,
        {"logisticregression__C": C_GRID},
        cv=GroupKFold(n_splits=5),
        scoring="roc_auc",
    )


def evaluate(X, y, groups) -> dict:
    """Repeated, family-grouped 5-fold CV with nested tuning of C.

    Returns fold AUCs plus pooled out-of-fold predictions from the first
    repeat (used for the ROC figure).
    """
    fold_aucs, chosen_C = [], []
    oof = np.full(len(y), np.nan)
    for r in range(N_REPEATS):
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + r)
        for train, test in outer.split(X, y, groups):
            m = _model().fit(X.iloc[train], y[train], groups=groups[train])
            p = m.predict_proba(X.iloc[test])[:, 1]
            fold_aucs.append(roc_auc_score(y[test], p))
            chosen_C.append(m.best_params_["logisticregression__C"])
            if r == 0:
                oof[test] = p

    fold_aucs = np.array(fold_aucs)
    fpr, tpr, _ = roc_curve(y, oof)
    lo, hi = np.percentile(fold_aucs, [2.5, 97.5])
    return {
        "n_cases": int(y.sum()),
        "n_pseudocontrols": int((y == 0).sum()),
        "n_families": int(len(set(groups))),
        "auc_mean": round(float(fold_aucs.mean()), 3),
        "auc_sd": round(float(fold_aucs.std()), 3),
        "auc_fold_range_95": [round(float(lo), 3), round(float(hi), 3)],
        "pooled_oof_auc_repeat1": round(float(roc_auc_score(y, oof)), 3),
        "most_chosen_C": float(max(set(chosen_C), key=chosen_C.count)),
        "_roc": (fpr, tpr),
    }
