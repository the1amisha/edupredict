"""Train and evaluate ML baselines on SYNTHETIC demonstration records.

Not suitable for deployment or decision-making about real students.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'demo_students.csv'
OUTPUT = ROOT / 'models' / 'risk_model.joblib'
FEATURES = ['attendance','assignment','internal']


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = (df['final_result'] == 'Fail').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )
    candidates = {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        'Random Forest': RandomForestClassifier(n_estimators=160, min_samples_leaf=3, random_state=42),
    }
    evaluated = []
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        scores = {
            'model': name,
            'accuracy': float(accuracy_score(y_test, pred)),
            'fail_precision': float(precision_score(y_test, pred, zero_division=0)),
            'fail_recall': float(recall_score(y_test, pred, zero_division=0)),
            'fail_f1': float(f1_score(y_test, pred, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        }
        evaluated.append((scores, clf))
    # For early warnings, prioritise recall of students who may fail.
    best_metrics, best_clf = max(evaluated, key=lambda v: (v[0]['fail_recall'], v[0]['fail_f1']))
    OUTPUT.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump({'model': best_clf, 'metrics': best_metrics, 'features': FEATURES, 'dataset_kind': 'synthetic'}, OUTPUT)
    print('Selected teaching demo:', best_metrics['model'])
    print(json.dumps(best_metrics, indent=2))
    print(f'Saved: {OUTPUT}')
    print('WARNING: Results evaluate a SYNTHETIC data generator, not accuracy on real students.')

if __name__ == '__main__':
    main()
