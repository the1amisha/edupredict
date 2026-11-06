"""Safe, generic local-roster integrity checks; contains no personal data."""
import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
PRIVATE = DATA / 'private'


@unittest.skipUnless((PRIVATE / 'class_roster.csv').exists(), 'No local roster available')
class RosterMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roster = pd.read_csv(PRIVATE / 'class_roster.csv', dtype=str)
        cls.students = pd.read_csv(PRIVATE / 'mca_students.csv', dtype={'student_id': str})
        cls.subjects = pd.read_csv(PRIVATE / 'mca_subject_records.csv', dtype={'student_id': str})
        cls.public_students = pd.read_csv(DATA / 'mca_students.csv')
        cls.public_subjects = pd.read_csv(DATA / 'mca_subject_records.csv')

    def test_exact_roster_identities_and_no_emails(self):
        self.assertEqual(len(self.roster), 34)
        self.assertEqual(self.students[['student_id','name']].values.tolist(), self.roster[['student_id','name']].values.tolist())
        self.assertEqual(self.students.student_id.nunique(), 34)
        self.assertEqual(self.subjects.student_id.nunique(), 34)
        self.assertEqual(len(self.subjects), 272)
        self.assertNotIn('email', ' '.join(self.students.columns).lower())

    def test_academic_values_are_unchanged(self):
        for col in ['attendance', 'assignment', 'internal', 'attendance_eligibility', 'subjects_below_75']:
            self.assertEqual(self.students[col].tolist(), self.public_students[col].tolist())
        for col in ['classes_attended', 'classes_conducted', 'attendance_pct', 'attendance_status',
                    'classes_needed_to_75', 'assignment', 'internal', 'subject_code']:
            self.assertEqual(self.subjects[col].fillna('MISSING').tolist(), self.public_subjects[col].fillna('MISSING').tolist())

    def test_mini_project_still_exempt(self):
        mini = self.subjects.query("subject_code == 'MCA-P2301'")
        self.assertEqual(len(mini), 34)
        self.assertEqual(set(mini['attendance_status']), {'Exempt'})
        self.assertTrue(mini['attendance_pct'].isna().all())


if __name__ == '__main__':
    unittest.main()
