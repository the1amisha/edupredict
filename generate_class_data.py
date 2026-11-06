"""Map synthetic MCA Semester 3 academic records to the LOCAL class roster.

Personal names and supplied placeholder SBU IDs remain under data/private/, which
is ignored by Git. Emails are neither read nor needed. No actual grades or
attendance are used: academic data remain explicitly synthetic.
"""
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
PRIVATE = DATA / 'private'
ROSTER_FILE = PRIVATE / 'class_roster.csv'
STUDENT_DEMO_FILE = DATA / 'mca_students.csv'
SUBJECT_DEMO_FILE = DATA / 'mca_subject_records.csv'


def build():
    if not ROSTER_FILE.is_file():
        raise FileNotFoundError(f'Private roster not found: {ROSTER_FILE}')
    roster = pd.read_csv(ROSTER_FILE, dtype=str, keep_default_na=False)
    expected_columns = ['student_id', 'name']
    if roster.columns.tolist() != expected_columns:
        raise ValueError(f'Roster needs exactly these columns: {expected_columns}')
    roster = roster.apply(lambda col: col.str.strip())
    if len(roster) != 34 or roster.isna().any().any() or (roster == '').any().any():
        raise ValueError('Expected 34 complete student names and IDs')
    if roster.student_id.nunique() != 34 or roster.name.nunique() != 34:
        raise ValueError('Student IDs and names must each be unique')
    if not roster.student_id.map(lambda value: bool(re.fullmatch(r'SBU25\d{5}', value))).all():
        raise ValueError('Each SBU ID must match SBU25 followed by five digits')

    students = pd.read_csv(STUDENT_DEMO_FILE, dtype={'student_id': str})
    records = pd.read_csv(SUBJECT_DEMO_FILE, dtype={'student_id': str})
    if len(students) != 34 or students.student_id.nunique() != 34:
        raise ValueError('The included public demo must contain 34 unique students')
    if len(records) != 34 * 8 or set(records.student_id) != set(students.student_id):
        raise ValueError('The public demo subject rows are incomplete or inconsistent')

    # Preserve the *existing* synthetic academic data. Only replace identities.
    # Student order is taken from the public demo, matching the 34 roster rows.
    old_ids = students['student_id'].tolist()
    replacements = {
        old_id: (str(new_id), str(new_name))
        for old_id, (new_id, new_name) in zip(old_ids, roster[['student_id', 'name']].itertuples(index=False, name=None))
    }
    for df in (students, records):
        df['name'] = df['student_id'].map(lambda sid: replacements[sid][1])
        df['student_id'] = df['student_id'].map(lambda sid: replacements[sid][0])

    PRIVATE.mkdir(parents=True, exist_ok=True)
    students.to_csv(PRIVATE / 'mca_students.csv', index=False)
    records.to_csv(PRIVATE / 'mca_subject_records.csv', index=False)

    assert students[['student_id','name']].reset_index(drop=True).equals(roster)
    assert len(records) == 272
    assert set(records.student_id) == set(roster.student_id)
    print('Local roster applied: 34 exact names and supplied SBU IDs')
    print('272 subject records retained with synthetic attendance/marks unchanged')
    print('Saved to data/private/ (ignored by Git; no email addresses included)')
    return students, records


if __name__ == '__main__':
    build()
