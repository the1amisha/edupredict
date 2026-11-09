"""Privacy-safe, pure helpers for EduPredict's MCA student subject profiles.

Data and outcomes are synthetic teaching examples. The baseline model was
trained on aggregate synthetic scores, NOT on actual subject-exam results.
Subject-level probabilities are exploratory only, not validated forecasts.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

MIN_ATTENDANCE = 75
FEATURES = ['attendance', 'assignment', 'internal']


def consecutive_classes_to_75(attended: int, conducted: int) -> int:
    """Minimum future classes attended in a row to get to >= 75%.

    (a+x)/(c+x) >= 0.75  -> x >= 3c - 4a. Cannot assess an exempt
    subject with zero sessions; the caller must exclude those records.
    """
    if conducted <= 0 or attended < 0 or attended > conducted:
        raise ValueError('Expected 0 <= attended <= conducted and conducted > 0')
    return max(0, 3 * conducted - 4 * attended)


def demo_risk_band(p: float, threshold: float = 0.50) -> str:
    """Model demo bands; High follows the classifier's decision threshold."""
    if not (0 <= p <= 1):
        raise ValueError('A risk score must be between 0 and 1')
    if p >= threshold:
        return 'High'
    if p >= max(0.0, threshold - 0.15):
        return 'Medium'
    return 'Low'


def subject_profile_rows(
    subject_records: pd.DataFrame,
    subject_catalog: pd.DataFrame,
    student_id: str,
    model_bundle=None,
) -> pd.DataFrame:
    """Join faculty details, re-evaluate attendance, and (optionally) score subjects.

    Model scores reuse the existing demonstration model on each subject's
    metrics. They are *not* validated subject-level exam predictions.
    Mini Project must be excluded because it is attendance-exempt and has
    no subject exam outcome to predict.
    """
    match = subject_records.loc[subject_records['student_id'].astype(str) == str(student_id)].copy()
    if match.empty:
        raise ValueError('Student ID has no subject records')
    if match['subject_code'].duplicated().any():
        raise ValueError('Duplicate subject code for student')

    metadata = subject_catalog[['subject_code', 'professor', 'credits', 'attendance_required']].copy()
    merged = match.merge(metadata, on='subject_code', how='left', validate='many_to_one', sort=False)
    if merged['attendance_required'].isna().any():
        raise ValueError('One or more subject codes are missing from the catalog')
    merged['attendance_required'] = merged['attendance_required'].astype(bool)
    if len(merged) != len(metadata):
        raise ValueError('Expected one record for every MCA subject')

    merged['subject_risk'] = 'Not applicable'
    merged['risk_score_pct'] = np.nan
    merged['verified_attendance_status'] = 'Exempt'
    merged['verified_classes_to_75'] = pd.Series(pd.NA, index=merged.index, dtype='Int64')

    target_rows = merged.index[merged['attendance_required']].tolist()
    for idx in target_rows:
        row = merged.loc[idx]
        a, c = int(row['classes_attended']), int(row['classes_conducted'])
        eligible = 4 * a >= 3 * c
        merged.at[idx, 'verified_attendance_status'] = 'Meets 75%' if eligible else 'Below 75%'
        merged.at[idx, 'verified_classes_to_75'] = consecutive_classes_to_75(a, c)
        if not math.isclose(float(row['attendance_pct']), 100 * a / c, abs_tol=0.02):
            raise ValueError('Stored attendance percentage does not match attendance counts')

    if model_bundle is not None and target_rows:
        model = model_bundle['model']
        fail_index = list(model.classes_).index(1)
        features = pd.DataFrame({
            'attendance': merged.loc[target_rows, 'attendance_pct'].astype(float).to_numpy(),
            'assignment': merged.loc[target_rows, 'assignment'].astype(float).to_numpy(),
            'internal': merged.loc[target_rows, 'internal'].astype(float).to_numpy(),
        }, index=target_rows)
        probs = model.predict_proba(features[FEATURES])[:, fail_index]
        threshold = float(model_bundle.get('decision_threshold', 0.50))
        merged.loc[target_rows, 'risk_score_pct'] = np.round(probs * 100, 1)
        merged.loc[target_rows, 'subject_risk'] = [demo_risk_band(float(p), threshold) for p in probs]
    else:
        merged.loc[target_rows, 'subject_risk'] = 'Model not trained'

    return merged


def academic_support(row) -> list[str]:
    """Descriptive support suggestions from observed (synthetic) indicators.

    These are not causal explanations of the ML model.
    """
    if not bool(row['attendance_required']):
        return ['Mini Project is attendance-exempt; no exam-attendance risk is calculated.']
    notes = []
    if row['verified_attendance_status'] == 'Below 75%':
        n = int(row['verified_classes_to_75'])
        notes.append(f'Attend at least {n} consecutive additional classes (if held) to reach 75%; discuss the attendance shortage with the lecturer.')
    if float(row['assignment']) < 60:
        notes.append('Review assignment mistakes and plan focused practice exercises.')
    if float(row['internal']) < 60:
        notes.append('Schedule revision of weak topics before the internal/final assessment.')
    if not notes:
        notes.append('Maintain consistent attendance and regular academic practice.')
    return notes

# Descriptive early-warning benchmarks. These are not ML model explanations.
ASSIGNMENT_REVIEW_BENCHMARK = 60.0
INTERNAL_REVIEW_BENCHMARK = 60.0


def academic_indicators(attendance: float, assignment: float, internal: float, *, subject_level: bool = False) -> list[dict[str, object]]:
    """Return three non-causal academic review signals independent of ML risk."""
    values = (float(attendance), float(assignment), float(internal))
    if not all(math.isfinite(v) and 0 <= v <= 100 for v in values):
        raise ValueError('Academic indicators must be between 0 and 100')
    att, assn, intern = values
    attendance_note = (
        'Subject attendance below 75%; review exam eligibility'
        if subject_level and att < MIN_ATTENDANCE else
        'Subject attendance meets the 75% demo requirement'
        if subject_level else
        'Overall attendance below 75%; check individual subjects'
        if att < MIN_ATTENDANCE else
        'Overall average only; subject attendance is checked separately'
    )
    return [
        {'short': 'Att', 'label': 'Attendance', 'value': att, 'flag': att < MIN_ATTENDANCE,
         'note': attendance_note,
         'alert': 'Subject attendance below 75%' if subject_level else 'Overall attendance below 75% (check subjects)'},
        {'short': 'Assign', 'label': 'Assignments', 'value': assn,
         'flag': assn < ASSIGNMENT_REVIEW_BENCHMARK,
         'note': 'Below 60% demo support benchmark' if assn < 60 else 'At or above 60% demo support benchmark',
         'alert': 'Assignments below 60% (demo benchmark)'},
        {'short': 'Internal', 'label': 'Internal assessment', 'value': intern,
         'flag': intern < INTERNAL_REVIEW_BENCHMARK,
         'note': 'Below 60% demo support benchmark' if intern < 60 else 'At or above 60% demo support benchmark',
         'alert': 'Internal below 60% (demo benchmark)'},
    ]


def indicator_review_summary(attendance: float, assignment: float, internal: float, *, subject_level: bool = False) -> str:
    alerts = [item['alert'] for item in academic_indicators(
        attendance, assignment, internal, subject_level=subject_level) if item['flag']]
    return '; '.join(alerts) if alerts else 'No indicators below demo benchmarks'
