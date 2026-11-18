"""Create faculty login records for EduPredict using Streamlit secrets.

Default (recommended for real accounts): generates a different random password
for each faculty. --shared-password asks once for a classroom-only shared
password without putting it in shell history, source files, or GitHub.

Usage for classroom demonstration:
    python create_faculty_credentials.py --overwrite --shared-password

This REPLACES all current faculty credentials; copy the generated secrets
file into Streamlit Cloud's Secrets settings as well.
"""
from __future__ import annotations

import argparse
import getpass
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
    ('aditya', 'Prof. Aditya Swaroop'),
    ('csmaheta', 'Mr. C. S. Maheta'),
]


def create_secrets(
    destination: Path,
    *,
    extra=(),
    overwrite: bool = False,
    shared_password: str | None = None,
):
    """Write salted credential hashes; return credentials only for random mode.

    When shared_password is supplied, only hashed values reach the file, and
    no plaintext password is returned or printed. This mode is for demos only.
    """
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f'{destination} exists. Refusing to replace it; '
            'use --overwrite only when rotating ALL passwords.'
        )
    if shared_password is not None and not shared_password:
        raise ValueError('Shared password must not be empty')

    records = list(FACULTY) + list(extra)
    if len(set(username for username, _ in records)) != len(records):
        raise ValueError('Usernames must be distinct')
    for username, name in records:
        if not re.fullmatch(r'[a-z][a-z0-9_]{1,31}', username):
            raise ValueError(f'Invalid username: {username!r}')
        if not name.strip():
            raise ValueError('Display names cannot be empty')

    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# PRIVATE FACULTY CREDENTIALS - NEVER COMMIT TO GITHUB', '']
    one_time = []
    for username, display in records:
        password = shared_password if shared_password is not None else secrets.token_urlsafe(12)
        salt = secrets.token_bytes(16)
        digest = password_digest(password, salt)
        lines.extend([
            f'[faculty_users.{username}]',
            f'display_name = {json.dumps(display)}',
            f'salt = "{salt.hex()}"',
            f'password_hash = "{digest}"',
            '',
        ])
        if shared_password is None:
            one_time.append((username, display, password))

    destination.write_text('\n'.join(lines), encoding='utf-8')
    try:
        destination.chmod(0o600)
    except OSError:
        pass
    return one_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--add', action='append', nargs=2, metavar=('USERNAME', 'DISPLAY_NAME'),
        default=[], help='Extra account beyond the named MCA faculty',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Replace existing secrets; invalidates all previous faculty passwords',
    )
    parser.add_argument(
        '--shared-password', action='store_true',
        help='Prompt privately for one password for ALL accounts (demo use only)',
    )
    args = parser.parse_args()

    shared = None
    if args.shared_password:
        print('Demo only: a shared short password is not safe for real faculty accounts.')
        shared = getpass.getpass('Enter shared demo password: ')
        confirmed = getpass.getpass('Confirm shared demo password: ')
        if not shared or shared != confirmed:
            parser.error('Passwords did not match or were empty; no secrets changed')

    try:
        credentials = create_secrets(
            Path('.streamlit/secrets.toml'),
            extra=args.add,
            overwrite=args.overwrite,
            shared_password=shared,
        )
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))

    print('\nUpdated .streamlit/secrets.toml with separately salted password hashes.')
    if args.shared_password:
        print('All faculty accounts now use the shared password you entered.')
        print('No plaintext password was saved or printed.')
        for username, name in FACULTY + args.add:
            print(f'  {username} — {name}')
    else:
        print('PRIVATE: Save each generated password in a password manager.')
        for username, name, password in credentials:
            print(f'{name}: username={username}  password={password}')

    print('\nCloud: copy the ENTIRE .streamlit/secrets.toml file into')
    print('Streamlit Community Cloud → Manage app → Settings → Secrets.')
    print('Never push secrets.toml to GitHub. Existing sessions may be invalidated.')


if __name__ == '__main__':
    main()
