# EduPredict — MCA Semester 3 Teaching Demo

A charcoal, cream, and gold **Streamlit** dashboard inspired by the user's design reference. It demonstrates **subject-wise 75% attendance eligibility** and a separate **machine-learning early-warning classifier** using fictional student records.

## What's included

- 34 **fictional** students with generated demo IDs (no real student directory or email addresses)
- 8 actual MCA Semester 3 subject titles/codes, 21 total credits, and faculty details supplied for the assignment; unspecified faculty are explicitly labeled
- 7 subjects with a 75% attendance requirement; Mini Project is **attendance-exempt**
- 272 synthetic student–subject records with attendance counts, derived percentages, assignment scores, and internal assessment scores
- A Subjects page, a detailed Students page, attendance shortage alerts, and two downloadable demo CSV reports
- A separate **280-row synthetic ML-training example** with a fabricated Pass/Fail label, plus a training/validation/test training script and Analytics page

## Install and run (Windows, CMD)

In the existing `D:\EduPredict` folder, keep your working `.venv` Python environment.

```cmd
cd /d D:\EduPredict
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python generate_mca_demo.py
python train_model.py
python -m streamlit run app.py
```

Open <http://localhost:8501>. The existing saved `models/risk_model.joblib`, if available, can also be reused without retraining.

## Files

| File | Purpose |
|---|---|
| `data/mca_students.csv` | 34 fictional student aggregate profiles for dashboard display |
| `data/mca_subjects.csv` | 8 MCA subjects and faculty / credits / attendance requirements |
| `data/mca_subject_records.csv` | 272 fictional student-subject records |
| `generate_mca_demo.py` | Deterministic synthetic class-data generator |
| `data/demo_students.csv` | Separate synthetic example data with simulated final outcomes for ML training |
| `generate_demo_data.py` | Reproduce the 280-row synthetic ML data |
| `train_model.py` | Select classifiers on validation data and evaluate on an untouched synthetic test set |
| `app.py` | Streamlit UI with class, subject, prediction, and analytics views |
| `tests/test_mca_demo.py` | Data integrity and attendance threshold tests |

To run unit tests:

```cmd
python -m unittest discover -s tests -v
```

## Attendance rules

Subject status is determined **using exact counts**, not rounded percentages:

- Meets requirement when `classes_attended / classes_conducted >= 0.75`.
- Below requirement otherwise.
- Mini Project is `Exempt` and does not have attendance counts or percentages.
- The **entire student** has an attendance shortfall if they are below 75% in *at least one* attendance-required subject (demo assumption).
- The `Classes to reach 75%` figure is the number of consecutive **future attended classes** required if those classes actually occur: `max(0, 3*conducted - 4*attended)`.
- An overall percentage is calculated across all seven attendance-required subjects for display; **it cannot override an individual subject shortfall**.

The minimum is based on the user's stated rule for this assignment; actual university exam regulations may have exceptions or additional conditions.

## ML model (teaching-only)

The `Predict` page estimates failure-risk using three aggregate features: attendance, assignment scores, and internal assessment scores. The `High` risk label follows the selected classifier's default **50% decision threshold**, `Medium` is a caution band below that threshold, and `Low` is below the caution band. Risk is **not** equivalent to attendance-based exam eligibility. The model scores are not calibrated probabilities.

The ML training data (`demo_students.csv`) has 280 **simulated** outcomes. The displayed fictional MCA class (`mca_students.csv`) has **no actual final results**, and it is not included in the model evaluation. The evaluation measures how well a model approximates a synthetic data-generation process, **not real-world accuracy**. This is appropriate for an academic prototype, not production decisions about actual students.

## Data privacy and GitHub

Do not commit any real student directory, names, email addresses, roll numbers, or private attendance/marks. No data from an actual class directory is used in the synthetic files. `.gitignore` excludes `.venv`, model binaries, private local files, and common secrets.

## Academic-indicator badges (teaching-only)

The Overview, Students, Profiles, and Predict views show all three inputs
beside an ML risk label: overall/subject attendance, assignment percentage,
and internal assessment percentage. Coral chips flag subject attendance below
75% or assignments/internals below **60%**, a *demonstration support benchmark*.
Green chips indicate the benchmark is not triggered.

The **75% requirement applies separately to each non-exempt subject**; an
overall attendance average cannot determine eligibility. The 60% marks
benchmarks are **not SBU passing thresholds**. The chips are descriptive
rule-based checks, **not causal model explanations or feature importance**.
All academic records are synthetic. Private class rosters remain local.
