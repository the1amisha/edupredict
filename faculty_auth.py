"""Lightweight faculty-only sign-in for an EduPredict classroom demonstration.

Credential hashes and salts come from Streamlit secrets, never from GitHub.
A signed browser cookie permits refresh for up to 8 hours (demo only).
For use with fictional academic outcomes; not production account management.
"""
from __future__ import annotations

import base64
import hashlib
import json
import hmac
import time
from collections.abc import Mapping

PBKDF2_ITERATIONS = 600_000
MAX_ATTEMPTS = 5
LOCK_SECONDS = 30
COOKIE_NAME = 'edupredict_faculty_session_v1'
SESSION_TTL_SECONDS = 8 * 60 * 60  # Eight hours; refresh does not extend it.


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




def make_session_token(session: dict, accounts: Mapping,
                       *, now: int | None = None, ttl: int = SESSION_TTL_SECONDS) -> str:
    """Sign an expiry-limited browser token using the *private* account digest.

    Never put passwords, hashes, or raw session dicts into a cookie. Changing an
    account's password hash invalidates all of that account's old tokens.
    """
    if not session_is_valid(session, accounts):
        raise ValueError('Cannot sign an invalid faculty session')
    if ttl < 1 or ttl > SESSION_TTL_SECONDS:
        raise ValueError('Invalid session duration')
    username = session['username']
    expiry = int(time.time() if now is None else now) + ttl
    raw = json.dumps({'username': username, 'expiry': expiry},
                     sort_keys=True, separators=(',', ':')).encode('utf-8')
    payload = base64.urlsafe_b64encode(raw).rstrip(b'=').decode('ascii')
    secret = bytes.fromhex(str(accounts[username]['password_hash']))
    sig = hmac.new(secret, ('edupredict-v1.' + payload).encode('ascii'),
                   hashlib.sha256).hexdigest()
    return 'v1.' + payload + '.' + sig


def restore_session_token(token: str | None, accounts: Mapping,
                          *, now: int | None = None) -> dict | None:
    """Restore only a signed, unexpired faculty identity from the browser."""
    if not isinstance(token, str) or len(token) > 1000:
        return None
    try:
        version, payload, signature = token.split('.')
        if version != 'v1' or len(signature) != 64 or len(payload) > 500:
            return None
        decoded = base64.urlsafe_b64decode(payload + '=' * (-len(payload) % 4))
        body = json.loads(decoded)
        username = body['username']
        expiry = body['expiry']
        instant = int(time.time() if now is None else now)
        if (not isinstance(username, str) or not isinstance(expiry, int)
                or expiry <= instant or expiry > instant + SESSION_TTL_SECONDS):
            return None
        record = accounts.get(username)
        if not isinstance(record, Mapping):
            return None
        expected_hash = str(record.get('password_hash', ''))
        if len(expected_hash) != 64:
            return None
        secret = bytes.fromhex(expected_hash)
        correct = hmac.new(secret, ('edupredict-v1.' + payload).encode('ascii'),
                           hashlib.sha256).hexdigest()
        if not hmac.compare_digest(correct, signature):
            return None
        return {'username': username,
                'name': str(record.get('display_name', username)),
                'password_hash': expected_hash}
    except (ValueError, KeyError, TypeError, UnicodeError, AttributeError,
            json.JSONDecodeError):
        return None


def browser_cookie_value(st, controller) -> str | None:
    """Prefer cookies attached to the browser request; fall back to component."""
    try:
        value = st.context.cookies.get(COOKIE_NAME)
        if value:
            return value
    except (AttributeError, TypeError, KeyError):
        pass
    return controller.get(COOKIE_NAME)

def require_faculty_login() -> str:
    """Block all dashboard work until a faculty credential is accepted."""
    import streamlit as st
    from streamlit_cookies_controller import CookieController

    # A browser refresh creates a NEW WebSocket/Streamlit session. A signed
    # cookie lets the new session recover the login without saving a password.
    controller = CookieController(key='edupredict_faculty_cookie_widget')
    try:
        accounts = st.secrets.get('faculty_users', {})
    except FileNotFoundError:
        accounts = {}

    logout_requested = st.session_state.pop('_edupredict_logout_requested', False)
    if logout_requested:
        controller.remove(COOKIE_NAME)
        st.session_state.pop('_edupredict_faculty', None)
    else:
        current = st.session_state.get('_edupredict_faculty')
        if session_is_valid(current, accounts):
            # A Streamlit component writes cookies in the browser. Calling
            # st.rerun() immediately after controller.set() may cancel that
            # browser-side write. Do it during the *next full page render*.
            if st.session_state.pop('_edupredict_cookie_write_pending', False):
                controller.set(
                    COOKIE_NAME,
                    make_session_token(current, accounts),
                    max_age=SESSION_TTL_SECONDS,
                    same_site='lax',
                )
            return str(current['name'])
        # Cookie is signed by a server-only derived key and expires after 8h.
        restored = restore_session_token(browser_cookie_value(st, controller), accounts)
        if restored:
            st.session_state['_edupredict_faculty'] = restored
            return str(restored['name'])
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
            # Persist only a signed, time-limited identity; never store a password.
            # Schedule persistence; the next run must finish rendering so
            # the browser receives the write before any page refresh.
            st.session_state['_edupredict_cookie_write_pending'] = True
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
    st.session_state['_edupredict_logout_requested'] = True
    st.session_state['page'] = 'Overview'
    st.rerun()
