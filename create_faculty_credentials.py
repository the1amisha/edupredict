"""Create local Streamlit secrets for named MCA faculty (never commit the file).

Run once: python create_faculty_credentials.py
For cloud: copy .streamlit/secrets.toml into the app's Streamlit Secrets editor.
Plaintext passwords print to your terminal once and are NOT written to files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import secrets
from faculty_auth import password_digest

FACULTY = [
    ('prakash', 'Dr. Prakash Kumar'),
    ('rahul', 'Mr. Rahul Das'),
    ('sandip', 'Dr. Sandip Kumar Singh Modak'),
    ('ragini', 'Dr. Ragini Shukla'),
    ('priyanka', 'Dr. Priyanka Srivastava'),
    ('csmaheta', 'Mr. C. S. Maheta'),
]


def create_secrets(destination: Path, *, extra=(), overwrite: bool=False):
    if destination.exists() and not overwrite:
        raise FileExistsError(f'{destination} exists. Refusing to replace it; use --overwrite only when rotating ALL passwords.')
    records = list(FACULTY) + list(extra)
    if len(set(user for user, _ in records)) != len(records):
        raise ValueError('Usernames must be distinct')
    for username, name in records:
        if not re.fullmatch(r'[a-z][a-z0-9_]{1,31}', username):
            raise ValueError(f'Invalid username: {username!r}')
        if not name.strip():
            raise ValueError('Display names cannot be empty')
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = ['# PRIVATE FACULTY CREDENTIALS - NEVER COMMIT TO GITHUB', '']
    one_time = []
    for username, display in records:
        password = secrets.token_urlsafe(12)
        salt = secrets.token_bytes(16)
        digest = password_digest(password, salt)
        text.extend([f'[faculty_users.{username}]',
                     f'display_name = {json.dumps(display)}',
                     f'salt = "{salt.hex()}"',
                     f'password_hash = "{digest}"', ''])
        one_time.append((username, display, password))
    destination.write_text('\n'.join(text), encoding='utf-8')
    try:
        destination.chmod(0o600)
    except OSError:
        pass
    return one_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--add', action='append', nargs=2, metavar=('USERNAME', 'DISPLAY_NAME'),
                        default=[], help='Additional faculty account (e.g. for Operations Research)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace existing secrets and rotate all faculty passwords')
    args = parser.parse_args()
    try:
        credentials = create_secrets(Path('.streamlit/secrets.toml'),
                                     extra=args.add, overwrite=args.overwrite)
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    print('\nCreated .streamlit/secrets.toml with hashed passwords.\n')
    print('PRIVATE: Write down these passwords and share each one only with its respective faculty member.')
    print('DO NOT upload/screenshot this terminal output or commit the secrets file.\n')
    for username, display, password in credentials:
        print(f'{display}: username={username}  password={password}')
    print('\nCloud: paste the ENTIRE secrets.toml contents into Streamlit Cloud → Manage app → Settings → Secrets.')
