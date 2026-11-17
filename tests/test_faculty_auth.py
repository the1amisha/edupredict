import unittest
from faculty_auth import authenticate, password_digest, session_is_valid


class FacultyAuthenticationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accounts = {'prakash': {
            'display_name': 'Dr. Prakash Kumar',
            'salt': '1234567890abcdef' * 2,
            'password_hash': password_digest('correct-secret', bytes.fromhex('1234567890abcdef'*2)),
        }}

    def test_accept_correct_password_and_case_insensitive_username(self):
        result = authenticate(self.accounts, '  PRAKASH ', 'correct-secret')
        self.assertEqual(result['name'], 'Dr. Prakash Kumar')
        self.assertNotIn('password', result)
        self.assertTrue(session_is_valid(result, self.accounts))

    def test_refuse_wrong_password_and_unknown_users(self):
        self.assertIsNone(authenticate(self.accounts, 'prakash', 'wrong'))
        self.assertIsNone(authenticate(self.accounts, 'unknown', 'correct-secret'))

    def test_password_rotation_invalidates_existing_session(self):
        session = authenticate(self.accounts, 'prakash', 'correct-secret')
        accounts2 = {'prakash': dict(self.accounts['prakash'])}
        accounts2['prakash']['password_hash'] = password_digest('new-pass', bytes.fromhex('1234567890abcdef'*2))
        self.assertFalse(session_is_valid(session, accounts2))
        self.assertFalse(session_is_valid(session, {}))

    def test_broken_secret_record_fails_closed(self):
        self.assertIsNone(authenticate({'other': {'salt': 'INVALID', 'password_hash': 'abc'}}, 'other', 'x'))
        self.assertFalse(session_is_valid({}, self.accounts))


if __name__ == '__main__':
    unittest.main()
