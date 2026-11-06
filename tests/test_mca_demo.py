"""Data-integrity checks for fictional MCA Semester 3 records."""
import unittest
import pandas as pd

from generate_mca_demo import (
    NAMES, SUBJECTS, classes_needed_to_reach_75,
)
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data'


class MCADemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.students = pd.read_csv(DATA / 'mca_students.csv')
        cls.subjects = pd.read_csv(DATA / 'mca_subjects.csv')
        cls.records = pd.read_csv(DATA / 'mca_subject_records.csv')

    def test_record_counts_and_identities(self):
        self.assertEqual(len(self.students), 34)
        self.assertEqual(len(self.subjects), 8)
        self.assertEqual(len(self.records), 272)
        self.assertEqual(self.students.student_id.nunique(), 34)
        self.assertEqual(set(self.students.name), set(NAMES))
        self.assertTrue(self.students.student_id.str.startswith('DEMO-MCA-').all())
        self.assertEqual(self.records.groupby('student_id').size().unique().tolist(), [8])

    def test_subject_rules(self):
        self.assertEqual(self.subjects.credits.sum(), 21)
        self.assertEqual(int(self.subjects.attendance_required.sum()), 7)
        mini = self.records[self.records.subject_code == 'MCA-P2301']
        self.assertEqual(len(mini), 34)
        self.assertTrue(mini.attendance_pct.isna().all())
        self.assertTrue(mini.classes_attended.isna().all())
        self.assertTrue(mini.classes_needed_to_75.isna().all())
        self.assertEqual(set(mini.attendance_status), {'Exempt'})

    def test_attendance_eligibility_exact_75_threshold(self):
        non_exempt = self.records[self.records.attendance_status != 'Exempt']
        self.assertEqual(len(non_exempt), 34*7)
        for r in non_exempt.itertuples():
            attended, conducted = int(r.classes_attended), int(r.classes_conducted)
            expected = 'Meets 75%' if 4*attended >= 3*conducted else 'Below 75%'
            self.assertEqual(r.attendance_status, expected)
            self.assertEqual(int(r.classes_needed_to_75), classes_needed_to_reach_75(attended, conducted))
            self.assertAlmostEqual(r.attendance_pct, round(100*attended/conducted, 2))
            self.assertTrue(0 <= r.assignment <= 100)
            self.assertTrue(0 <= r.internal <= 100)

    def test_student_aggregates(self):
        for r in self.students.itertuples():
            detail = self.records[self.records.student_id == r.student_id]
            included = detail[detail.attendance_status != 'Exempt']
            expected = round(100*included.classes_attended.sum()/included.classes_conducted.sum())
            self.assertEqual(int(r.attendance), expected)
            missing = int((included.attendance_status == 'Below 75%').sum())
            self.assertEqual(int(r.subjects_below_75), missing)
            self.assertEqual(r.attendance_eligibility == 'Meets attendance rule', missing == 0)

    def test_recovery_math(self):
        self.assertEqual(classes_needed_to_reach_75(21,30), 6)  # 70% now; 27/36 = 75%
        self.assertEqual(classes_needed_to_reach_75(22, 30), 2) # 73.33% now; 24/32 = 75%
        self.assertEqual(classes_needed_to_reach_75(23, 30), 0)
        with self.assertRaises(ValueError):
            classes_needed_to_reach_75(32, 30)


if __name__ == '__main__':
    unittest.main()
