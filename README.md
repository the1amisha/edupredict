# EduPredict · Smart Student Early Warning System

Behance-inspired teacher dashboard built using **Python + Streamlit**, with a demonstration of early academic-risk screening and a baseline ML-training pipeline.

## Features

- Cream / charcoal / muted-gold dashboard with collapsible concept-style navigation
- Real navigation between Overview, Students, Predict, Analytics, Reports, and About
- Search and risk-category filtering
- Class-level indicators, distribution ring, attendance chart, academic calendar, and alerts
- Interactive individual risk checking
- CSV report download
- Logistic Regression and Random Forest comparison (optional step)

## Setup (Windows Command Prompt)

Your existing `.venv` folder does not need to be replaced.

```cmd
cd /d D:\EduPredict
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open http://localhost:8501 in your browser. If Streamlit is already running, stop it with **Ctrl+C** and start it again after replacing the files.

## Train the ML demo

```cmd
python train_model.py
python -m streamlit run app.py
```

The training script fits a Logistic Regression and a Random Forest on synthetic data; compares held-out failure recall and F1; and saves the chosen demonstration model to `models/risk_model.joblib` (excluded from Git). You can view its synthetic-test performance on the **Analytics** page.

To regenerate the demo CSV (optional):

```cmd
python generate_demo_data.py
```

## Important limitations

**All 280 student records and all final outcomes are artificially generated.** Before a model exists, the UI runs a **rule-based placeholder**. After training, the UI runs an **ML demonstration trained and evaluated on simulated data**. Neither method is validated for real students and neither should be described as proven to predict actual examination failures. A real deployment requires permissioned, anonymised, representative data with *actual, correctly timed* final outcomes, out-of-sample validation, fair evaluation, and human review.

The right-side academic schedule is **illustrative**, not a connected school calendar.

## GitHub

Do not commit `.venv`, secret tokens, model binary files or identifiable student data. The included `.gitignore` excludes common generated and private files.

```cmd
git init
git add .
git commit -m "feat: initial EduPredict dashboard"
git branch -M main
```

Create an empty repository named `edupredict` on GitHub, then set its origin URL and push.
