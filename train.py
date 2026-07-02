"""
Standalone training script for the Loan Default Risk model.
Runs headless (no Drive mount, no Colab) — for local use or CI.

Trains Logistic Regression + SMOTE, evaluates on a held-out test set,
saves the model + metrics, and fails (non-zero exit code) if quality
drops below the thresholds below. That failure is what GitHub Actions
uses to block a bad deployment.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = Path("data/Loan_default.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Default"

# Minimum acceptable quality — matches roughly what the notebook achieved.
# If a future retrain scores below this, the CI build fails on purpose.
MIN_F1 = 0.30
MIN_RECALL = 0.60

STRATEGIES = {
    "Conservative": {"approve_threshold": 0.20, "reject_threshold": 0.50},
}


def decision(p, a, r):
    if p < a:
        return "Approve"
    if p <= r:
        return "Manual Review"
    return "Reject"


def main():
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    id_cols = [c for c in df.columns if c.lower() in {"loanid", "loan_id", "id"}]
    df = df.drop(columns=id_cols, errors="ignore")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features),
    ])

    model = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1_score": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
    }
    print("Metrics:", json.dumps(metrics, indent=2))

    # Apply the Conservative strategy (final production decision)
    results_df = X_test.copy()
    results_df["Actual_Default"] = y_test.values
    results_df["Predicted_Default_Probability"] = prob
    s = STRATEGIES["Conservative"]
    results_df["Conservative_Decision"] = [
        decision(p, s["approve_threshold"], s["reject_threshold"]) for p in prob
    ]
    results_df.to_csv(OUTPUT_DIR / "strategy_predictions.csv", index=False)

    joblib.dump(model, OUTPUT_DIR / "selected_loan_default_model.joblib")
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining complete. Outputs saved to {OUTPUT_DIR}/")

    # Quality gate — this is what makes it CI/CD, not just a training script.
    if metrics["f1_score"] < MIN_F1 or metrics["recall"] < MIN_RECALL:
        print(
            f"\n QUALITY GATE FAILED: f1={metrics['f1_score']:.3f} "
            f"(min {MIN_F1}), recall={metrics['recall']:.3f} (min {MIN_RECALL})"
        )
        sys.exit(1)

    print("\n Quality gate passed.")


if __name__ == "__main__":
    main()
