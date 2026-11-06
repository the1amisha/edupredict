"""Generate 34 *fictional* MCA Semester 3 student profiles and subject records.

This is an assignment-only demonstration: no real university data is read.
The university's 75% attendance rule is implemented per attendance-required
subject. Mini Project is attendance-exempt.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
SEED = 2026
MINIMUM_ATTENDANCE = 75.0

SUBJECTS = [
    ('MCA-EL-2301', 'Object-Oriented Modeling and Design', 'Dr. Prakash Kumar', 3, 38, True),
    ('MCA-EL-2302', 'Object-Oriented Modeling and Design Lab', 'Mr. Rahul Das', 2, 30, True),
    ('MCA-EL-2303', 'Introduction to Machine Learning', 'Dr. Sandip Kumar Singh Modak', 3, 36, True),
    ('MCA-SEC-2302', 'Machine Learning Lab using Python', 'Dr. Ragini Shukla', 3, 30, True),
    ('MCA-SEC-2301', 'Cyber Law and Information Security', 'Dr. Priyanka Srivastava', 3, 36, True),
    ('MCA-PCC-2302', 'Operations Research', 'Faculty not specified', 4, 38, True),
    ('MCA-AEC-2301', 'Personality Development Program-II', 'Mr. C. S. Maheta', 1, 22, True),
    ('MCA-P2301', 'Mini Project', 'Faculty not specified', 2, 0, False),
]

# Completely fabricated identities. These are NOT copied from a class directory.
NAMES = [
    'Aditi Mehra', 'Arjun Bhat', 'Ishita Sen', 'Raghav Jain', 'Anaya Kapoor',
    'Kunal Roy', 'Meera Das', 'Vivaan Khanna', 'Nisha Gupta', 'Kabir Malhotra',
    'Sneha Saxena', 'Riya Bose', 'Dev Sinha', 'Diya Nair', 'Aditya Mishra',
    'Sanvi Rao', 'Aman Pandey', 'Nandini Ghosh', 'Yash Anand', 'Kirti Sahu',
    'Pooja Prasad', 'Rohan Verma', 'Avni Joshi', 'Tanya Singh', 'Neha Shah',
    'Siddharth Kumar', 'Simran Dutta', 'Varun Patel', 'Priya Banerjee',
    'Aryan Sethi', 'Ayesha Ali', 'Manav Saxena', 'Ritu Das', 'Harsh Sharma',
]


def classes_needed_to_reach_75(attended: int, conducted: int) -> int:
    """Minimum consecutive future attended classes needed to reach >=75%.

    For (a+x)/(t+x) >= 3/4, the minimum integer x is max(0, 3*t - 4*a).
    It is an estimate, conditional on future classes actually being held.
    """
    if conducted <= 0 or not 0 <= attended <= conducted:
        raise ValueError('Invalid attendance counts')
    return max(0, 3 * conducted - 4 * attended)


def build():
    rng = np.random.default_rng(SEED)
    subjects = pd.DataFrame(SUBJECTS, columns=[
        'subject_code', 'subject', 'professor', 'credits',
        'classes_conducted', 'attendance_required',
    ])
    subjects['minimum_attendance_pct'] = subjects['attendance_required'].map(
        {True: MINIMUM_ATTENDANCE, False: np.nan}
    )
    records = []
    for i, name in enumerate(NAMES, start=1):
        sid = f'DEMO-MCA-{i:03d}'
        student_engagement = np.clip(rng.normal(0.87, 0.085), 0.44, 0.98)
        student_ability = rng.normal(0.0, 1.0)
        for code, subject, faculty, credits, sessions, requires_att in SUBJECTS:
            if requires_att:
                subject_rate = float(np.clip(student_engagement + rng.normal(0, 0.055), .28, 1.0))
                attended = int(rng.binomial(sessions, subject_rate))
                pct = 100.0 * attended / sessions
                status = 'Meets 75%' if 4 * attended >= 3 * sessions else 'Below 75%'
                needed = classes_needed_to_reach_75(attended, sessions)
                assignment = int(np.clip(round(66 + 11 * student_ability + .32 * (pct - 75) + rng.normal(0, 11)), 8, 99))
                internal = int(np.clip(round(63 + 13 * student_ability + .27 * (pct - 75) + rng.normal(0, 11)), 5, 99))
            else:
                attended = None
                pct = np.nan
                status = 'Exempt'
                needed = None
                assignment = None
                internal = None
            records.append({
                'student_id': sid,
                'name': name,
                'subject_code': code,
                'subject': subject,
                'classes_attended': attended,
                'classes_conducted': sessions if requires_att else None,
                'attendance_pct': round(pct, 2) if requires_att else np.nan,
                'attendance_status': status,
                'classes_needed_to_75': needed,
                'assignment': assignment,
                'internal': internal,
            })

    detail = pd.DataFrame(records)
    for col in ['classes_attended', 'classes_conducted', 'classes_needed_to_75', 'assignment', 'internal']:
        detail[col] = detail[col].astype('Int64')
    student_rows = []
    for sid, g in detail.groupby('student_id', sort=False):
        valid = g[g['attendance_status'] != 'Exempt']
        a = int(valid['classes_attended'].sum())
        t = int(valid['classes_conducted'].sum())
        missing = int((valid['attendance_status'] == 'Below 75%').sum())
        student_rows.append({
            'student_id': sid,
            'name': str(g['name'].iloc[0]),
            'attendance': round(100*a/t),
            'assignment': round(valid['assignment'].mean()),
            'internal': round(valid['internal'].mean()),
            'attendance_eligibility': 'Below 75% in subject(s)' if missing else 'Meets attendance rule',
            'subjects_below_75': missing,
        })
    students = pd.DataFrame(student_rows)
    DATA.mkdir(parents=True, exist_ok=True)
    subjects.to_csv(DATA / 'mca_subjects.csv', index=False)
    detail.to_csv(DATA / 'mca_subject_records.csv', index=False)
    students.to_csv(DATA / 'mca_students.csv', index=False)
    print(f'Generated {len(students)} fictional students, {len(subjects)} subjects, {len(detail)} subject records')
    print(f'Students with attendance shortfall in >=1 subject: {(students.subjects_below_75 > 0).sum()}')
    print('Mini Project: attendance exempt for every student')
    print('Data generated for demonstration only: NOT university records')
    return subjects, detail, students


if __name__ == '__main__':
    build()
