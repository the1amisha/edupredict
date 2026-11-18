"""Source-level smoke checks for the student search and sticky topbar update."""
import ast
from pathlib import Path
import unittest

PROJECT = Path(__file__).resolve().parents[1]


class HeaderSearchSmokeTests(unittest.TestCase):
    def test_header_selectbox_and_callback_wiring(self):
        source = (PROJECT / 'app.py').read_text(encoding='utf-8')
        ast.parse(source)
        self.assertIn("key='header_student_picker'", source)
        self.assertIn('on_change=open_student_from_header_search', source)
        self.assertIn("st.session_state['profile_student_id'] = str(student_id)", source)
        self.assertIn("st.session_state['page'] = 'Profiles'", source)

    def test_header_sticky_css_present(self):
        source = (PROJECT / 'app.py').read_text(encoding='utf-8')
        self.assertIn('EDUPREDICT-STUDENT-SEARCH-STICKY-2026-11-V2', source)
        self.assertIn('position: sticky !important', source)
        self.assertIn('.st-key-topbar', source)


if __name__ == '__main__':
    unittest.main()
