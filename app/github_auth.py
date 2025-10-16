import logging
import os
import uuid
from urllib.parse import urlencode

import httpx
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GitHub OAuth Configuration
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
# IMPORTANT: default to nginx front door, NOT :8501
GITHUB_REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI", "http://localhost/")

API_URL = os.getenv("ROWBLAZE_API_URL", "http://localhost:8000/api")


async def start_github_auth():
    """Initiates GitHub OAuth flow."""
    if not GITHUB_CLIENT_ID:
        raise ValueError("GITHUB_CLIENT_ID environment variable is not set")

    # Generate and store state for CSRF protection
    state = str(uuid.uuid4())
    st.session_state.oauth_state = state

    # Prepare GitHub OAuth parameters
    params = {
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": GITHUB_REDIRECT_URI,
        "scope": "read:user user:email",
        "state": state,
    }

    auth_url = f"https://github.com/login/oauth/authorize?{urlencode(params)}"
    logger.info(f"Generated GitHub auth URL with redirect_uri: {GITHUB_REDIRECT_URI}")
    return auth_url


async def handle_github_callback(code, state):
    """Handles the OAuth callback from GitHub."""
    logger.info("Processing GitHub callback")

    # Verify state to prevent CSRF attacks
    if state != st.session_state.get("oauth_state"):
        logger.error(
            f"State mismatch: {state} vs {st.session_state.get('oauth_state')}"
        )
        return None

    if not (GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET):
        logger.error("GitHub OAuth credentials not configured")
        raise ValueError("GitHub OAuth credentials not configured")

    try:
        # Exchange code for access token
        logger.info("Exchanging code for access token")
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1: Get GitHub access token
            token_response = await client.post(
                "https://github.com/login/oauth/access_token",
                data={
                    "client_id": GITHUB_CLIENT_ID,
                    "client_secret": GITHUB_CLIENT_SECRET,
                    "code": code,
                    "redirect_uri": GITHUB_REDIRECT_URI,
                },
                headers={"Accept": "application/json"},
            )

            token_data = token_response.json()
            logger.info(f"Token response status: {token_response.status_code}")

            if "access_token" not in token_data:
                logger.error(f"Token exchange failed: {token_data}")
                return None

            # Step 2: Get GitHub user data
            github_token = token_data["access_token"]
            user_response = await client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"token {github_token}",
                    "Accept": "application/json",
                },
            )
            user_data = user_response.json()

            # Step 3: Get GitHub email data
            email_response = await client.get(
                "https://api.github.com/user/emails",
                headers={
                    "Authorization": f"token {github_token}",
                    "Accept": "application/json",
                },
            )
            emails = email_response.json()
            primary_email = next((e["email"] for e in emails if e.get("primary")), None)

            # Step 4: Call our API to register/login the user
            logger.info(f"Calling API at {API_URL}/auth/oauth/github")
            auth_response = await client.post(
                f"{API_URL}/auth/oauth/github",
                json={
                    "github_id": str(user_data["id"]),
                    "username": user_data["login"],
                    "email": primary_email or user_data.get("email", ""),
                    "name": user_data.get("name", ""),
                    "avatar_url": user_data.get("avatar_url", ""),
                },
                timeout=30.0,
            )

            if auth_response.status_code == 200:
                logger.info("GitHub authentication successful")
                return auth_response.json()

            logger.error(
                f"Backend auth failed: {auth_response.status_code} {auth_response.text}"
            )
            return None

    except Exception as e:
        logger.exception(f"GitHub authentication error: {e}")
        return None
