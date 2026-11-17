"""Lightweight faculty-only sign-in for an EduPredict classroom demonstration.

Credential hashes and salts come from Streamlit secrets, never from GitHub.
Sessions are per Streamlit browser connection, not durable login cookies.
For use with fictional academic outcomes; not production account management.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import Mapping

PBKDF2_ITERATIONS = 600_000
MAX_ATTEMPTS = 5
LOCK_SECONDS = 30


def password_digest(password: str, salt: bytes) -> str:
    """Return a salted PBKDF2-HMAC-SHA256 hex digest."""
    if not isinstance(password, str) or not isinstance(salt, bytes) or len(salt) < 16:
        raise ValueError('Invalid password or salt')
    return hashlib.pbkdf2_hmac(
        'sha256', password.encode('utf-8'), salt, PBKDF2_ITERATIONS
    ).hex()


def authenticate(accounts: Mapping, username: str, password: str) -> dict | None:
    """Verify a credential against a non-public mapping of username -> hash record.

    Return only the username, displayed name, and digest for session validation.
    Passwords are never persisted in Streamlit Session State.
    """
    key = (username or '').strip().casefold()
    record = accounts.get(key)
    if not isinstance(record, Mapping):
        return None
    try:
        salt = bytes.fromhex(str(record['salt']))
        expected = str(record['password_hash']).lower()
        if len(expected) != 64:
            return None
        actual = password_digest(password, salt)
        if not hmac.compare_digest(actual, expected):
            return None
        name = str(record.get('display_name', key)).strip() or key
        return {'username': key, 'name': name, 'password_hash': expected}
    except (ValueError, KeyError, TypeError):
        return None


def session_is_valid(session: object, accounts: Mapping) -> bool:
    """Invalidate a session when a user's account is deleted or hash rotated."""
    if not isinstance(session, dict):
        return False
    record = accounts.get(session.get('username'))
    return (isinstance(record, Mapping)
            and hmac.compare_digest(str(session.get('password_hash', '')),
                                    str(record.get('password_hash', ''))))


def require_faculty_login() -> str:
    """Block all dashboard work until a faculty credential is accepted."""
    import streamlit as st

    try:
        accounts = st.secrets.get('faculty_users', {})
    except FileNotFoundError:
        accounts = {}
    current = st.session_state.get('_edupredict_faculty')
    if session_is_valid(current, accounts):
        return str(current['name'])
    st.session_state.pop('_edupredict_faculty', None)

    st.html('''<style>
      [data-testid="stAppViewContainer"]{background:#e8e3da!important;}
      .block-container{max-width:550px!important;padding:8vh 22px 50px!important;}
      .login-brand{text-align:center;color:#a0823d;font:700 34px Georgia,serif;margin-bottom:5px;}
      .login-kicker{text-align:center;font:600 11px sans-serif;letter-spacing:2px;color:#827762;margin-bottom:24px;}
      .login-title{text-align:center;color:#3f423d;font:700 25px Georgia,serif;margin:4px 0 8px;}
      .login-sub{text-align:center;color:#7b7368;font-size:13px;margin-bottom:22px;}
      [data-testid="stForm"]{background:#faf8f3!important;border:1px solid #e0d9cc!important;
         border-radius:22px!important;padding:20px 26px!important;box-shadow:0 14px 40px #47403014;}
      [data-testid="stFormSubmitButton"] button{background:#a78b49!important;color:white!important;
         border:0!important;width:100%!important;border-radius:10px!important;}
    </style>''')
    st.markdown('<div class="login-brand">EduPredict.</div><div class="login-kicker">SMART EDUCATION · FACULTY ACCESS</div>'
                '<div class="login-title">Welcome back</div><div class="login-sub">Sign in to open the MCA Semester 3 dashboard</div>',
                unsafe_allow_html=True)
    if not accounts:
        st.error('Faculty accounts are not configured yet. An administrator must add faculty_users in Streamlit secrets.')
        st.stop()

    count = int(st.session_state.get('_edupredict_login_failures', 0))
    locked_until = float(st.session_state.get('_edupredict_login_locked_until', 0))
    remaining = int(max(0, locked_until - time.monotonic())) + 1 if locked_until > time.monotonic() else 0
    if remaining:
        st.warning(f'Too many unsuccessful attempts. Try again in about {remaining} seconds.')
    with st.form('faculty_sign_in', clear_on_submit=False):
        username = st.text_input('Username', placeholder='Faculty username', autocomplete='username')
        password = st.text_input('Password', type='password', placeholder='Password', autocomplete='current-password')
        submitted = st.form_submit_button('Sign in', disabled=bool(remaining))
    if submitted and not remaining:
        result = authenticate(accounts, username, password)
        if result:
            st.session_state['_edupredict_faculty'] = result
            st.session_state['_edupredict_login_failures'] = 0
            st.session_state['_edupredict_login_locked_until'] = 0
            st.session_state['page'] = 'Overview'
            st.rerun()
        count += 1
        st.session_state['_edupredict_login_failures'] = count
        if count >= MAX_ATTEMPTS:
            st.session_state['_edupredict_login_failures'] = 0
            st.session_state['_edupredict_login_locked_until'] = time.monotonic() + LOCK_SECONDS
            st.warning('Too many attempts. Sign-in is temporarily paused in this browser session.')
        else:
            st.error('Incorrect username or password.')
    st.caption('Authorized faculty accounts only · Same dashboard for every faculty member')
    st.stop()


def logout_faculty() -> None:
    """End the current browser session's sign-in immediately."""
    import streamlit as st
    st.session_state.pop('_edupredict_faculty', None)
    st.session_state['page'] = 'Overview'
    st.rerun()
