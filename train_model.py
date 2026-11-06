"""Train an EduPredict teaching demo with a train/validation/test split.

Synthetic outcomes are for demonstrating ML workflows only. They are not
proof of real student failure prediction performance.
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "demo_students.csv"
OUTPUT_PATH = ROOT / "models" / "risk_model.joblib"
FEATURES = ["attendance", "assignment", "internal"]
SEED = 42


def evaluate(y_true, y_pred):
    """Class 1 means Fail; the matrix rows/columns are Pass, Fail."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fail_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "fail_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fail_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fail_f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + ["final_result"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("Dataset contains missing input values or outcomes")
    if not df["final_result"].isin(["Pass", "Fail"]).all():
        raise ValueError("final_result must contain only Pass or Fail")
    if not df[FEATURES].apply(pd.to_numeric, errors="coerce").notna().all().all():
        raise ValueError("Academic indicators must be numeric")
    if not ((df[FEATURES] >= 0) & (df[FEATURES] <= 100)).all().all():
        raise ValueError("Academic indicators must be between 0 and 100")

    x = df[FEATURES]
    y = (df["final_result"] == "Fail").astype(int)
    if y.nunique() != 2 or min(y.value_counts()) < 5:
        raise ValueError("Need at least five records of each outcome for a stratified split")

    # The test data is put aside before any model comparison.
    x_develop, x_test, y_develop, y_test = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=SEED
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_develop, y_develop, test_size=0.25, stratify=y_develop, random_state=SEED
    )
    # Effective proportions: 60% train, 20% validation, 20% test.
    candidates = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)
        ),
        "Balanced Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=SEED
            )
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=160, min_samples_leaf=3, random_state=SEED
        ),
        "Balanced Random Forest Weights": RandomForestClassifier(
            n_estimators=160, min_samples_leaf=3,
            class_weight="balanced", random_state=SEED
        ),
    }

    valid_results = []
    fitted = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        stats = evaluate(y_valid, model.predict(x_valid))
        valid_results.append({"model": name, **stats})
        fitted[name] = model

    # Pick using validation F2, which values recall more than precision,
    # while still penalizing an approach that flags everyone as failing.
    chosen = max(
        valid_results,
        key=lambda result: (
            result["fail_f2"], result["fail_precision"], result["fail_recall"]
        ),
    )
    selected_model = fitted[chosen["model"]]
    test_metrics = evaluate(y_test, selected_model.predict(x_test))
    result = {
        "model": chosen["model"],
        **test_metrics,
        "validation_f2": chosen["fail_f2"],
        "split_sizes": {
            "train": len(x_train), "validation": len(x_valid), "test": len(x_test)
        },
    }

    OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(
        {
            "model": selected_model,
            "metrics": result,
            "features": FEATURES,
            "dataset_kind": "synthetic",
            "validation_comparison": valid_results,
        },
        OUTPUT_PATH,
    )

    print("Validation comparison (selection used F2, not test performance):")
    for candidate in valid_results:
        print(
            f"  {candidate['model']}: recall={candidate['fail_recall']:.1%}, "
            f"precision={candidate['fail_precision']:.1%}, "
            f"F2={candidate['fail_f2']:.1%}"
        )
    print("\nFinal untouched test-set evaluation:")
    print(json.dumps(result, indent=2))
    print(f"Saved: {OUTPUT_PATH}")
    print("CAUTION: Synthetic demonstration results are not real-world accuracy.")


if __name__ == "__main__":
    main()
