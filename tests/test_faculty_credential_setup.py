"""Tests that never read or change the real Streamlit secrets file."""
import tempfile
import tomllib
import unittest
from pathlib import Path

from create_faculty_credentials import FACULTY, create_secrets
from faculty_auth import authenticate


class FacultyCredentialSetupTests(unittest.TestCase):
    def test_named_faculty_has_aditya(self):
        self.assertIn(('aditya', 'Prof. Aditya Swaroop'), FACULTY)
        self.assertEqual(len(FACULTY), 7)
        self.assertEqual(len({username for username, _ in FACULTY}), 7)

    def test_one_shared_password_is_salted_per_account(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'secrets.toml'
            self.assertEqual(create_secrets(path, shared_password='test-only-password'), [])
            contents = path.read_text(encoding='utf-8')
            self.assertNotIn('test-only-password', contents)
            accounts = tomllib.loads(contents)['faculty_users']
            self.assertEqual(len(accounts), 7)
            for username, display in FACULTY:
                result = authenticate(accounts, username, 'test-only-password')
                self.assertEqual(result['name'], display)
                self.assertIsNone(authenticate(accounts, username, 'not-the-password'))
            self.assertEqual(len({x['salt'] for x in accounts.values()}), 7)
            self.assertEqual(len({x['password_hash'] for x in accounts.values()}), 7)

    def test_does_not_overwrite_existing_secrets_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'secrets.toml'
            create_secrets(path, shared_password='first')
            with self.assertRaises(FileExistsError):
                create_secrets(path, shared_password='second')


if __name__ == '__main__':
    unittest.main()
