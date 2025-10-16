from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt
import streamlit as st


def is_authenticated() -> bool:
    """Check if user is authenticated."""
    return st.session_state.get("auth_token") is not None


def get_current_user() -> Optional[Dict[str, Any]]:
    """Get current user information from session state."""
    return st.session_state.get("user")


def logout_user():
    """Logout the current user by clearing session state."""
    # Clear authentication data
    if "auth_token" in st.session_state:
        del st.session_state.auth_token
    if "user" in st.session_state:
        del st.session_state.user

    # Clear other session data
    keys_to_clear = [
        "messages",
        "active_session_id",
        "chat_sessions",
        "indexed_files",
        "processing",
        "current_query",
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token and return payload."""
    import os

    # Use the same secret as the API
    jwt_secret = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")

    try:
        payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
    except Exception:
        return None


def create_local_token(user_data: Dict[str, Any]) -> str:
    """Create a local JWT token for development."""
    import os

    # Use the same secret as the API
    jwt_secret = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")

    token_data = {
        "sub": user_data.get("id", "local_user"),
        "username": user_data.get("username"),
        "email": user_data.get("email"),
        "exp": (datetime.utcnow() + timedelta(hours=24)).timestamp(),
    }

    return jwt.encode(token_data, jwt_secret, algorithm="HS256")
