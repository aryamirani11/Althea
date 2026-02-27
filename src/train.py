"""Model definitions, hyperparameter grids, and training via GridSearchCV."""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = "roc_auc"


def _build_configs() -> list[tuple[str, Pipeline, dict]]:
    """Return a list of (name, pipeline, param_grid) tuples."""
    return [
        (
            "LogisticRegression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
            ]),
            {
                "clf__C": [0.01, 0.1, 1, 10],
                "clf__l1_ratio": [0],
            },
        ),
        (
            "RandomForest",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
            ]),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [5, 10, None],
                "clf__min_samples_split": [2, 5],
            },
        ),
        (
            "GradientBoosting",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [3, 5],
            },
        ),
        (
            "SVM",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(random_state=RANDOM_STATE, probability=True)),
            ]),
            {
                "clf__C": [0.1, 1, 10],
                "clf__kernel": ["rbf", "linear"],
            },
        ),
    ]


def train_all(
    X_train,
    y_train,
    cv: int = CV_FOLDS,
    scoring: str = SCORING,
) -> dict:
    """Train and tune every model. Returns {name: GridSearchCV} dict."""
    results: dict = {}
    configs = _build_configs()

    for name, pipeline, param_grid in configs:
        print(f"\n--- Training {name} ---")
        gs = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X_train, y_train)
        print(f"Best params : {gs.best_params_}")
        print(f"Best CV {scoring}: {gs.best_score_:.4f}")
        results[name] = gs

    return results
