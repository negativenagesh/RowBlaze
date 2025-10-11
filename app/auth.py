import os
import bcrypt
import streamlit as st


def load_credentials():
    """
    Reads ROWBLAZE_CREDENTIALS env var:
    Format: user:hashed,user2:hashed2
    """
    raw = os.getenv("ROWBLAZE_CREDENTIALS", "")
    creds = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        user, hashed = pair.split(":", 1)
        creds[user.strip()] = hashed.strip()
    return creds


def verify_user(username: str, password: str) -> bool:
    creds = load_credentials()
    if username not in creds:
        return False
    try:
        return bcrypt.checkpw(password.encode(), creds[username].encode())
    except Exception:
        return False


def require_auth():
    if not st.session_state.get("authenticated", False):
        st.info("Please login from the Landing page.")
        st.stop()
