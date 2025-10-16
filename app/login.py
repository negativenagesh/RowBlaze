import asyncio
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import jwt
import streamlit as st

LOGO_PATH = Path(__file__).parent / "assets" / "image.png"


# Smart API URL detection based on environment
def get_api_url():
    # First check environment variable
    env_url = os.getenv("ROWBLAZE_API_URL")
    if env_url:
        return env_url

    # If running in Docker container, use internal service name
    if os.path.exists("/.dockerenv"):
        return "http://api:8000/api"

    # For local development, try nginx proxy first, then direct API
    return "http://localhost/api"


API_URL = get_api_url()


async def test_api_connection():
    """Test if the API is reachable with fallback URLs."""
    global API_URL

    # List of possible API URLs to try
    possible_urls = [
        API_URL,  # Primary URL from environment or detection
        "http://localhost/api",  # Nginx proxy
        "http://localhost:8000/api",  # Direct API (if port is exposed)
        "http://api:8000/api",  # Docker internal (if running in container)
    ]

    # Remove duplicates while preserving order
    urls_to_try = []
    for url in possible_urls:
        if url not in urls_to_try:
            urls_to_try.append(url)

    for url in urls_to_try:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{url}/health")
                if response.status_code == 200:
                    # Update the API_URL if we found a working one
                    if url != API_URL:
                        API_URL = url
                    return True
        except Exception:
            continue

    return False


async def login_user_api(email: str, password: str):
    """Login user with email/password via API."""
    try:
        # First test API connectivity (this will update API_URL if needed)
        api_reachable = await test_api_connection()
        if not api_reachable:
            return {
                "success": False,
                "error": "Unable to connect to authentication service. Please try again later.",
            }

        async with httpx.AsyncClient(timeout=15.0) as client:
            # Use JSON payload for the new login endpoint
            payload = {"email": email, "password": password}
            response = await client.post(
                f"{API_URL}/auth/login",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "access_token": result["access_token"],
                    "user": result["user"],
                }
            elif response.status_code == 401:
                return {"success": False, "error": "Invalid email or password"}
            else:
                error_detail = "Login failed"
                try:
                    error_response = response.json()
                    error_detail = error_response.get("detail", error_detail)
                except:
                    pass
                return {"success": False, "error": error_detail}

    except Exception as e:
        print(f"Login error: {e}")
        return {
            "success": False,
            "error": "An unexpected error occurred. Please try again.",
        }


async def register_user_api(email: str, password: str):
    """Register a new user via API."""
    try:
        # Test API connectivity (this will update API_URL if needed)
        api_reachable = await test_api_connection()
        if not api_reachable:
            return {
                "success": False,
                "error": "Unable to connect to registration service. Please try again later.",
            }

        async with httpx.AsyncClient(timeout=15.0) as client:
            payload = {"email": email, "password": password}
            response = await client.post(
                f"{API_URL}/auth/register",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "access_token": result["access_token"],
                    "user": result["user"],
                }
            else:
                # Handle error responses
                error_detail = "Registration failed"
                try:
                    error_response = response.json()
                    error_detail = error_response.get("detail", error_detail)
                except:
                    pass

                return {"success": False, "error": error_detail}

    except Exception as e:
        print(f"Registration error: {e}")
        return {
            "success": False,
            "error": "An unexpected error occurred during registration. Please try again.",
        }


async def start_github_auth():
    """Start GitHub OAuth flow (placeholder)."""
    # GitHub OAuth not implemented yet
    return None


async def handle_github_callback(code: str, state: str):
    """Handle GitHub OAuth callback (placeholder)."""
    # GitHub OAuth not implemented yet
    return None


def apply_login_css():
    st.markdown(
        """
        <style>
            /* Hide Streamlit elements */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display: none;}
            section[data-testid="stSidebar"] {display: none;}

            /* App background matching main app */
            .stApp {
                background: #2e2b2b !important;
                color: #fafafa !important;
            }

            /* Container styling */
            .block-container {
                padding: 2rem 1rem !important;
                max-width: 500px !important;
                margin: 0 auto !important;
            }

            /* Button styling */
            .stButton > button {
                border-radius: 8px !important;
                padding: 0.75rem 1rem !important;
                font-weight: 500 !important;
                background-color: #3a7be0 !important;
                color: white !important;
                border: none !important;
                transition: all 0.3s ease !important;
            }

            .stButton > button:hover {
                background-color: #2a6bd0 !important;
                box-shadow: 0 0 8px rgba(74, 139, 245, 0.5) !important;
            }

            /* Input styling */
            .stTextInput > div > div > input {
                border-radius: 8px !important;
                border: 1px solid #3a7be0 !important;
                padding: 0.75rem !important;
                background-color: #1e2130 !important;
                color: white !important;
            }

            /* Tab styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 2px !important;
                background-color: transparent !important;
                justify-content: center !important;
            }

            .stTabs [data-baseweb="tab"] {
                background-color: #1a1a1a !important;
                color: #fafafa !important;
                border-radius: 8px 8px 0 0 !important;
                padding: 0.75rem 1.5rem !important;
                font-weight: 500 !important;
            }

            .stTabs [aria-selected="true"] {
                background-color: #3a7be0 !important;
                color: white !important;
            }

            /* Center title */
            h1 {
                text-align: center !important;
                margin-bottom: 0.5rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def login_page():
    """Display the login/signup page."""
    # Apply minimal CSS
    apply_login_css()

    # Display logo if available
    logo_path = Path(__file__).parent / "assets" / "image.png"
    if logo_path.exists():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(str(logo_path), width=300)
    else:
        st.markdown(
            "<div style='text-align: center; font-size: 48px; margin-bottom: 20px;'>🗿</div>",
            unsafe_allow_html=True,
        )

    # st.title("RowBlaze")
    # st.markdown("<p style='text-align: center; color: #888; margin-bottom: 2rem;'>Document Intelligence Platform</p>", unsafe_allow_html=True)

    # Tab selection for Login/Signup
    tab1, tab2 = st.tabs(["🔐 Sign In", "📝 Sign Up"])

    with tab1:
        st.subheader("Sign In")

        email = st.text_input(
            "Email",
            value="",
            placeholder="Enter your email address",
            key="login_email",
            help="Use the email address you registered with",
        )
        password = st.text_input(
            "Password",
            type="password",
            value="",
            placeholder="Enter your password",
            key="login_password",
        )

        if st.button("Sign In", use_container_width=True):
            if not email or not password:
                st.error("⚠️ Please enter both email and password")
            elif "@" not in email or "." not in email:
                st.error("⚠️ Please enter a valid email address")
            else:
                with st.spinner("🔐 Signing in..."):
                    try:
                        auth_result = asyncio.run(
                            login_user_api(email.strip().lower(), password)
                        )

                        if auth_result and auth_result.get("success"):
                            st.session_state.auth_token = auth_result["access_token"]
                            st.session_state.user = auth_result["user"]
                            st.success("✅ Login successful! Redirecting...")
                            time.sleep(1)
                            st.rerun()
                        else:
                            error_msg = (
                                auth_result.get("error", "Login failed")
                                if auth_result
                                else "Authentication service unavailable"
                            )
                            st.error(f"❌ {error_msg}")
                    except Exception as e:
                        st.error(f"❌ Login error: {str(e)}")

        # Help text
        st.info("💡 **New user?** Switch to the Sign Up tab to create an account")

        # Debug info (only in development)
        if os.getenv("ENVIRONMENT") == "development" and st.checkbox("Show debug info"):
            st.write("API URL:", API_URL)
            api_status = asyncio.run(test_api_connection())
            st.write("API Status:", "✅ Connected" if api_status else "❌ Offline")

    with tab2:
        st.subheader("Create Account")

        new_email = st.text_input(
            "Email",
            value="",
            placeholder="Enter your email address",
            key="signup_email",
            help="We'll use this email for your account",
        )
        new_password = st.text_input(
            "Password",
            type="password",
            value="",
            placeholder="Create a strong password",
            key="signup_password",
            help="Must be at least 8 characters with letters and numbers",
        )
        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            value="",
            placeholder="Confirm your password",
            key="signup_confirm_password",
        )

        # Password strength indicator
        if new_password:
            strength_issues = []
            if len(new_password) < 8:
                strength_issues.append("At least 8 characters")
            if not re.search(r"[A-Za-z]", new_password):
                strength_issues.append("At least one letter")
            if not re.search(r"\d", new_password):
                strength_issues.append("At least one number")

            if strength_issues:
                st.warning(f"Password needs: {', '.join(strength_issues)}")
            else:
                st.success("✅ Password strength: Good")

        if st.button("Create Account", use_container_width=True):
            # Validation
            if not new_email or not new_password or not confirm_password:
                st.error("⚠️ Please fill in all fields")
            elif "@" not in new_email or "." not in new_email:
                st.error("⚠️ Please enter a valid email address")
            elif len(new_password) < 8:
                st.error("⚠️ Password must be at least 8 characters long")
            elif not re.search(r"[A-Za-z]", new_password):
                st.error("⚠️ Password must contain at least one letter")
            elif not re.search(r"\d", new_password):
                st.error("⚠️ Password must contain at least one number")
            elif new_password != confirm_password:
                st.error("⚠️ Passwords do not match")
            else:
                with st.spinner("📝 Creating account..."):
                    try:
                        auth_result = asyncio.run(
                            register_user_api(new_email.strip().lower(), new_password)
                        )

                        if auth_result and auth_result.get("success"):
                            st.session_state.auth_token = auth_result["access_token"]
                            st.session_state.user = auth_result["user"]
                            st.success(
                                "✅ Account created successfully! Redirecting..."
                            )
                            time.sleep(1)
                            st.rerun()
                        else:
                            error_msg = (
                                auth_result.get("error", "Failed to create account")
                                if auth_result
                                else "Registration service unavailable"
                            )
                            st.error(f"❌ {error_msg}")
                    except Exception as e:
                        st.error(f"❌ Registration error: {str(e)}")

        st.info("💡 **Already have an account?** Switch to the Sign In tab to log in")
