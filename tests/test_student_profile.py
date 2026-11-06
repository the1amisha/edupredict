"""Tests for the public fictional demo and optional local-roster data.

No class names, placeholder IDs or emails are stored in this test file.
"""
import unittest
from pathlib import Path

import joblib
import pandas as pd

from student_profile import (
    academic_support,
    consecutive_classes_to_75,
    demo_risk_band,
    subject_profile_rows,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
PRIVATE = DATA / 'private'


class AttendanceMathTests(unittest.TestCase):
    def test_minimum_recovery_calculation(self):
        self.assertEqual(consecutive_classes_to_75(28, 38), 2)
        self.assertEqual(consecutive_classes_to_75(27, 38), 6)
        self.assertEqual(consecutive_classes_to_75(30, 40), 0)
        self.assertEqual(consecutive_classes_to_75(38, 38), 0)
        with self.assertRaises(ValueError):
            consecutive_classes_to_75(4, 0)

    def test_decision_bands(self):
        self.assertEqual(demo_risk_band(.50), 'High')
        self.assertEqual(demo_risk_band(.49), 'Medium')
        self.assertEqual(demo_risk_band(.25), 'Low')


@unittest.skipUnless((DATA/'mca_students.csv').exists(), 'Run in installed EduPredict project')
class ProfileDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.subjects = pd.read_csv(DATA/'mca_subjects.csv')
        cls.public_students = pd.read_csv(DATA/'mca_students.csv',dtype={'student_id':str})
        cls.public_records = pd.read_csv(DATA/'mca_subject_records.csv',dtype={'student_id':str})
        model_path = ROOT/'models'/'risk_model.joblib'
        cls.bundle = joblib.load(model_path) if model_path.exists() else None

    def test_all_students_have_eight_subjects(self):
        self.assertEqual(len(self.public_students), 34)
        self.assertEqual(len(self.public_records), 34*8)
        self.assertEqual(self.public_records.groupby('student_id').size().unique().tolist(), [8])
        self.assertEqual(len(self.subjects),8)

    def test_subject_risk_and_exemption(self):
        for sid in self.public_students['student_id']:
            p = subject_profile_rows(self.public_records,self.subjects,sid,self.bundle)
            self.assertEqual(len(p),8)
            self.assertEqual(int(p['attendance_required'].sum()),7)
            mini = p.loc[p['subject_code']=='MCA-P2301'].iloc[0]
            self.assertFalse(bool(mini['attendance_required']))
            self.assertEqual(mini['verified_attendance_status'],'Exempt')
            self.assertEqual(mini['subject_risk'],'Not applicable')
            self.assertTrue(pd.isna(mini['risk_score_pct']))
            real = p.loc[p['attendance_required']]
            self.assertTrue((real['verified_classes_to_75'].astype(int)>=0).all())
            for _,row in real.iterrows():
                self.assertEqual(row['verified_attendance_status'], 'Meets 75%' if 4*row['classes_attended']>=3*row['classes_conducted'] else 'Below 75%')
                self.assertEqual(int(row['verified_classes_to_75']), consecutive_classes_to_75(int(row['classes_attended']),int(row['classes_conducted'])))
                if self.bundle is not None:
                    self.assertIn(row['subject_risk'],{'High','Medium','Low'})
                    self.assertGreaterEqual(row['risk_score_pct'],0)
                    self.assertLessEqual(row['risk_score_pct'],100)
                else:
                    self.assertEqual(row['subject_risk'],'Model not trained')

    def test_fallback_without_trained_model(self):
        sid = self.public_students['student_id'].iloc[0]
        p = subject_profile_rows(self.public_records, self.subjects, sid, None)
        self.assertEqual(set(p.loc[p.attendance_required, 'subject_risk']), {'Model not trained'})
        self.assertTrue(p['risk_score_pct'].isna().all())

    def test_unknown_student_rejected(self):
        with self.assertRaises(ValueError):
            subject_profile_rows(self.public_records, self.subjects, 'not-in-roster', self.bundle)

    def test_suggestions_are_descriptive(self):
        p=subject_profile_rows(self.public_records,self.subjects,self.public_students['student_id'].iloc[0],self.bundle)
        self.assertIn('exempt', ' '.join(academic_support(p.iloc[-1])).lower())

    @unittest.skipUnless((PRIVATE/'class_roster.csv').exists(), 'Private roster not installed')
    def test_local_roster_same_academic_data(self):
        local_students=pd.read_csv(PRIVATE/'mca_students.csv', dtype={'student_id':str})
        local_records=pd.read_csv(PRIVATE/'mca_subject_records.csv', dtype={'student_id':str})
        roster=pd.read_csv(PRIVATE/'class_roster.csv', dtype=str)
        self.assertEqual(local_students[['student_id','name']].values.tolist(),roster[['student_id','name']].values.tolist())
        self.assertEqual(local_students.student_id.nunique(),34)
        self.assertEqual(len(local_records),272)
        self.assertEqual(local_students['attendance'].tolist(),self.public_students['attendance'].tolist())
        for sid in local_students['student_id']:
            p=subject_profile_rows(local_records,self.subjects,sid,self.bundle)
            self.assertEqual(len(p),8)
            self.assertEqual(int(p.attendance_required.sum()),7)


if __name__=='__main__':
    unittest.main()
