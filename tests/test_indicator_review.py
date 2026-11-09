"""Tests for descriptive risk indicators; no class roster needed."""
import unittest
from student_profile import academic_indicators, indicator_review_summary

class AcademicIndicatorTests(unittest.TestCase):
    def test_all_three_visible(self):
        items = academic_indicators(85, 70, 65)
        self.assertEqual([item['label'] for item in items], [
            'Attendance', 'Assignments', 'Internal assessment'])
        self.assertFalse(any(item['flag'] for item in items))

    def test_assignment_internal_below_sixty(self):
        flags = academic_indicators(80, 59, 58)
        self.assertEqual([i['flag'] for i in flags], [False, True, True])
        self.assertIn('Assignments below 60%', indicator_review_summary(80, 59, 58))
        self.assertIn('Internal below 60%', indicator_review_summary(80, 59, 58))

    def test_exact_thresholds_not_flagged(self):
        items = academic_indicators(75, 60, 60, subject_level=True)
        self.assertFalse(any(i['flag'] for i in items))

    def test_overall_not_claimed_eligible(self):
        self.assertIn('subject attendance is checked separately', academic_indicators(86, 75, 85)[0]['note'])
        self.assertIn('Overall attendance below 75%', indicator_review_summary(74, 75, 85))

    def test_invalid_values(self):
        for values in [(-1, 50, 50), (75, 105, 70), (75, 65, float('nan'))]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                academic_indicators(*values)

if __name__ == '__main__':
    unittest.main()
