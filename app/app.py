import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import httpx
import requests
import streamlit as st

# CHANGE: safe import for cookie component
try:
    import extra_streamlit_components as stx
except Exception:
    stx = None

from auth import get_current_user, is_authenticated, logout_user
from login import login_page


# Add this function at the top of your file to create a global cookie manager
def get_cookie_manager():
    """Get or create a global cookie manager instance."""
    try:
        if stx is None:
            # Fallback when extra_streamlit_components isn't available
            class DummyCookieManager:
                def get(self, key):
                    return None

                def set(self, key, value, **kwargs):
                    pass

            return DummyCookieManager()
        if "cookie_manager" not in st.session_state:
            # Ensure component is created with a stable key
            st.session_state.cookie_manager = stx.CookieManager(
                key="global_cookie_manager"
            )
        return st.session_state.cookie_manager
    except Exception as e:
        print(f"Error initializing cookie manager: {e}")

        class DummyCookieManager:
            def get(self, key):
                return None

            def set(self, key, value, **kwargs):
                pass

        return DummyCookieManager()


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


# Test API connection with fallback (similar to login.py)
async def test_api_connection_with_fallback():
    """Test API connection with multiple URL fallbacks."""
    global API_URL

    possible_urls = [
        API_URL,
        "http://localhost/api",
        "http://localhost:8000/api",
        "http://api:8000/api",
    ]

    # Remove duplicates
    urls_to_try = []
    for url in possible_urls:
        if url not in urls_to_try:
            urls_to_try.append(url)

    for url in urls_to_try:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{url}/health")
                if response.status_code == 200:
                    if url != API_URL:
                        API_URL = url
                    return True
        except Exception:
            continue

    return False


st.set_page_config(
    page_title="RowBlaze",
    page_icon="🗿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Update get_session_id to use the global cookie manager
def get_session_id():
    """Get a unique session ID from cookie or generate a new one."""
    cookie_manager = get_cookie_manager()

    if "session_id" not in st.session_state:
        # Check if we have a cookie
        session_id = cookie_manager.get("rowblaze_session")
        if session_id:
            st.session_state.session_id = session_id
        else:
            # Generate a new UUID
            st.session_state.session_id = str(uuid.uuid4())
            # Set cookie for next time
            cookie_manager.set(
                "rowblaze_session",
                st.session_state.session_id,
                expires_at=datetime.now() + timedelta(days=30),
            )

    return st.session_state.session_id


async def save_chat_history(session_id, messages):
    """Save chat history to the API."""
    try:
        # Convert messages for API
        api_messages = [
            {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": datetime.now().isoformat(),
            }
            for msg in messages
        ]

        # Create payload
        payload = {
            "session_id": session_id,
            "messages": api_messages,
            "timestamp": datetime.now().isoformat(),
        }

        # Call API
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(f"{API_URL}/chat/save", json=payload)
            response.raise_for_status()
            return True

    except Exception as e:
        print(f"Error saving chat history: {e}")
        return False


async def load_chat_history(session_id):
    """Load chat history from the API."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{API_URL}/chat/{session_id}")
            response.raise_for_status()
            result = response.json()

            if result.get("success"):
                # Convert API messages to app format
                return [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in result.get("messages", [])
                ]

            return []

    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []


# Add these functions to your app.py


async def load_chat_sessions():
    """Load all chat sessions from the API."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{API_URL}/chat/list/sessions")
            response.raise_for_status()
            result = response.json()

            if result.get("success"):
                return result.get("sessions", [])

            return []

    except Exception as e:
        print(f"Error loading chat sessions: {e}")
        return []


async def delete_chat_session(session_id):
    """Delete a chat session from the API."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.delete(f"{API_URL}/chat/{session_id}")
            response.raise_for_status()
            result = response.json()

            return result.get("success", False)

    except Exception as e:
        print(f"Error deleting chat session: {e}")
        return False


# Update initialize_session_state to include user-specific indices
def initialize_session_state():
    """Initialize session state variables for chat history and settings."""
    # Get or generate session ID first
    session_id = get_session_id()

    # Get current user for user-specific indices
    current_user = get_current_user()
    # Handle both old and new user structure
    user_id = (
        current_user.get("user_id") or current_user.get("id")
        if current_user
        else "anonymous"
    )

    # --- One-time initialization block ---
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        # User-specific index name
        st.session_state.index_name = f"rowblaze-{user_id}"
        st.session_state.selected_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        st.session_state.processing = False
        st.session_state.is_processing_file = False
        st.session_state.upload_message = ""
        st.session_state.upload_success = False
        st.session_state.indexed_files = []
        st.session_state.files_last_fetched = 0
        st.session_state.chat_sessions = []
        st.session_state.sessions_last_fetched = 0
        st.session_state.active_session_id = session_id
        st.session_state.messages = []
        st.session_state.load_history_flag = True  # Load history on first run
        st.session_state.file_uploader_key = "file_uploader_0"
        st.session_state.user_input = ""
        st.session_state.show_chunks_modal = False
        st.session_state.show_kg_modal = False
        st.session_state.rag_mode = "Normal RAG"  # Default to Normal RAG

        # Initial health check with fallback
        try:
            st.session_state.index_checked = asyncio.run(
                test_api_connection_with_fallback()
            )
        except Exception as e:
            print(f"Error checking API health: {e}")
            st.session_state.index_checked = False

    # Update chat sessions for this user
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = []
    if "sessions_last_fetched" not in st.session_state:
        st.session_state.sessions_last_fetched = 0
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = session_id
    if "show_chat_history" not in st.session_state:
        st.session_state.show_chat_history = False

    # Initialize messages WITHOUT auto-loading history (critical change)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Don't automatically set load_history_flag = True

    if "load_history_flag" not in st.session_state:
        st.session_state.load_history_flag = False

    # Added flag to initialize the app
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    # Create index if it doesn't exist and fetch files periodically
    if "index_checked" not in st.session_state:
        try:
            st.session_state.index_checked = asyncio.run(
                test_api_connection_with_fallback()
            )
        except Exception as e:
            print(f"Error checking API health: {e}")
            st.session_state.index_checked = False

    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = "uploaded_file_0"

    # Initialize the user input state
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""


# Add your existing CSS styling function here
def apply_custom_css():
    """Apply custom CSS for Grok-inspired UI."""
    st.markdown(
        """
    <style>
        /* Main theme - updated background with gradient */
        .stApp {
            background: #2e2b2b;
            background: linear-gradient(90deg, rgba(46, 43, 43, 1) 100%, rgba(0, 212, 255, 1) 0%);
            color: #fafafa;
            font-family: 'Montserrat', 'Roboto', sans-serif;
        }

        /* Force all main content area elements to use the same background */
        .main, [data-testid="stAppViewContainer"],
        .st-emotion-cache-1wrcr25, .st-emotion-cache-6qob1r,
        .st-emotion-cache-uf99v8, .st-emotion-cache-16txtl3,
        .st-emotion-cache-18ni7ap, .st-emotion-cache-1kyxreq {
            background-color: #2e2b2b !important;
            background: linear-gradient(90deg, rgba(46, 43, 43, 1) 100%, rgba(0, 212, 255, 1) 0%) !important;
        }

        /* File uploader help text - make it wrap properly */
        .st-emotion-cache-16idsys p,
        .stFileUploadDropzone p,
        [data-testid="stFileUploader"] p {
            white-space: normal !important;
            overflow-wrap: break-word !important;
            word-wrap: break-word !important;
            max-width: 100% !important;
            line-height: 1.4 !important;
        }

        /* File uploader overall container formatting */
        [data-testid="stFileUploader"] {
            width: 100% !important;
            max-width: 100% !important;
        }

        /* Add an appropriate margin below the help text */
        .st-emotion-cache-16idsys,
        .stFileUploadDropzone {
            margin-bottom: 12px !important;
        }

        /* Main content containers */
        div[data-testid="stVerticalBlock"] {
            background-color: transparent !important;
        }

        /* Chat container background */
        .chat-container {
            background-color: transparent !important;
        }

        /* Enhanced chat input styling to match content area */
        .stChatInputContainer,
        [data-testid="stChatInput"],
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] input,
        [data-testid="stChatInput"] textarea {
            background-color: #2e2b2b !important;
            color: #fafafa !important;
            border-color: #3a7be0 !important;
        }

        /* Specifically target the chat input background elements */
        .st-emotion-cache-1aumxhk {
            background-color: #2e2b2b !important;
        }

        /* Sidebar specific styling with black gradient - keep as is */
        .css-1d391kg, .css-1lcbmhc, .css-17eq0hr, .css-1y4p8pa, .css-12oz5g7,
        section[data-testid="stSidebar"], .css-ng1t4o, .css-1cypcdb, .css-18e3th9 {
            background: #000000 !important;
            background: linear-gradient(90deg, rgba(0, 0, 0, 1) 100%, rgba(0, 212, 255, 1) 0%) !important;
        }

        /* Sidebar content area */
        .css-1lcbmhc .css-17eq0hr {
            background: #000000 !important;
            background: linear-gradient(90deg, rgba(0, 0, 0, 1) 100%, rgba(0, 212, 255, 1) 0%) !important;
        }

        /* Alternative sidebar selectors for different Streamlit versions */
        [data-testid="stSidebar"] > div:first-child {
            background: #000000 !important;
            background: linear-gradient(90deg, rgba(0, 0, 0, 1) 100%, rgba(0, 212, 255, 1) 0%) !important;
        }

        /* Sidebar text color to ensure visibility on black background */
        .css-1d391kg, .css-1lcbmhc, .css-17eq0hr, .css-1y4p8pa, .css-12oz5g7,
        section[data-testid="stSidebar"] * {
            color: #fafafa !important;
        }

        /* Sidebar headers */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #ffffff !important;
        }

        /* Sidebar text areas and inputs */
        section[data-testid="stSidebar"] .stTextArea textarea {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
            border: 1px solid #3a7be0 !important;
        }

        /* File uploader in sidebar */
        section[data-testid="stSidebar"] .stFileUploader {
            background-color: transparent !important;
        }

        section[data-testid="stSidebar"] .stFileUploader > div {
            background-color: #1a1a1a !important;
            border: 1px solid #3a7be0 !important;
        }

        /* Improved typography */
        body {
            font-family: 'Montserrat', 'Roboto', sans-serif;
            font-weight: 300;
            letter-spacing: 0.3px;
        }

        /* Chat message styling - UPDATED */
        .user-message {
            background-color: #2d2d2d; /* Changed from #3a7be0 to match bot message color */
            color: #fafafa; /* Changed from white to match bot text color */
            border-radius: 18px 18px 0 18px;
            padding: 12px 18px;
            margin: 10px 0;
            margin-left: auto;
            max-width: 80%;
            align-self: flex-end;
            word-wrap: break-word;
            display: block;
            width: fit-content;
            font-family: 'Montserrat', 'Roboto', sans-serif;
            font-weight: 400;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }

        .bot-message {
            background-color: #2d2d2d;
            color: #fafafa;
            border-radius: 18px 18px 18px 0;
            padding: 12px 18px;
            margin: 10px 0;
            margin-right: auto;
            max-width: 85%;
            align-self: flex-start;
            word-wrap: break-word;
            display: block;
            width: fit-content;
            font-family: 'Montserrat', 'Roboto', sans-serif;
            font-weight: 300;
            line-height: 1.5;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }

        /* Input box styling */
        .stTextInput > div > div > input {
            border-radius: 20px;
            border: 1px solid #3a7be0;
            padding: 10px 15px;
            background-color: #1e2130;
            color: white;
            font-size: 16px;
            font-family: 'Montserrat', 'Roboto', sans-serif;
            font-weight: 300;
        }

        .stTextInput > div > div > input:focus {
            box-shadow: 0 0 10px rgba(74, 139, 245, 0.5);
        }

        /* Button styling */
        .stButton > button {
            border-radius: 20px;
            background-color: #3a7be0;
            color: white;
            border: none;
            padding: 8px 18px;
            font-weight: 500;
            transition: all 0.3s ease;
            font-family: 'Montserrat', 'Roboto', sans-serif;
            letter-spacing: 0.5px;
        }

        .stButton > button:hover {
            background-color: #2a6bd0;
            box-shadow: 0 0 8px rgba(74, 139, 245, 0.5);
        }

        /* Header styling */
        h1, h2, h3 {
            color: white;
            font-family: 'Montserrat', 'Roboto', sans-serif;
            font-weight: 500;
            letter-spacing: 0.5px;
        }

        h1 {
            font-size: 2.2rem;
            font-weight: 600;
            letter-spacing: 1px;
        }

        /* File uploader styling */
        .stFileUploader > div > button {
            background-color: #3a7be0;
            color: white;
            font-family: 'Montserrat', 'Roboto', sans-serif;
        }

        /* Progress bar styling */
        .stProgress > div > div {
            background-color: #3a7be0;
        }

        /* Spinner color */
        .stSpinner > div > div {
            border-color: #3a7be0 transparent transparent transparent;
        }

        /* Chat container - scrollable area */
        .chat-container {
            max-height: 65vh;
            overflow-y: auto;
            padding-right: 10px;
            margin-bottom: 20px;
            border-radius: 10px;
        }

        /* Logo styling */
        .logo-container {
            text-align: center;
            margin-bottom: 15px;
        }

        /* Make sure code blocks look good */
        pre {
            background-color: #1e1e1e;
            border-radius: 5px;
            padding: 10px;
            overflow-x: auto;
        }

        code {
            color: #f8f8f2;
        }

        /* File item styling for sidebar */
        .file-item {
            padding: 8px 12px;
            margin: 4px 0;
            background-color: #1a1a1a !important;
            border-radius: 6px;
            border-left: 3px solid #3a7be0;
            font-size: 0.9em;
            color: #ffffff !important;
        }

        .file-count {
            color: #cccccc !important;
            font-size: 0.8em;
            margin-top: 5px;
        }

        /* Add custom font imports */
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');

        /* Chat history sidebar styles */
        .chat-session {
            padding: 8px 12px;
            margin: 4px 0;
            background-color: #1a1a1a;
            border-radius: 6px;
            border-left: 3px solid #3a7be0;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .chat-session:hover {
            background-color: #252525;
        }

        .chat-session.active {
            background-color: #2a2a2a;
            border-left: 3px solid #5a9bff;
        }

        .chat-session-title {
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .chat-session-meta {
            font-size: 0.8em;
            color: #888;
        }

        /* Delete button */
        .delete-btn {
            opacity: 0.6;
            transition: all 0.2s ease;
        }

        .delete-btn:hover {
            opacity: 1;
            color: #ff5555;
        }

        /* Tab styling for sidebar - UPDATED */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px !important;  /* Increased from 1px to 10px for more spacing */
            background-color: #000000 !important;
            padding: 5px 5px 0 5px !important;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #2a2a2a !important;  /* Changed to better match UI background */
            border-radius: 8px 8px 0 0 !important;
            padding: 10px 15px !important;
            font-weight: 500 !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: #3a7be0 !important;
        }

        /* Modal styling */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.8);
            z-index: 9999;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .modal-content {
            background-color: #2e2b2b;
            border-radius: 15px;
            padding: 25px;
            width: 90%;
            max-width: 1200px;
            max-height: 85vh;
            overflow-y: auto;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            border: 1px solid #3a7be0;
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            border-bottom: 2px solid #3a7be0;
            padding-bottom: 15px;
        }

        .modal-title {
            color: #ffffff;
            font-size: 1.5rem;
            font-weight: 600;
            margin: 0;
        }

        .modal-close {
            background: none;
            border: none;
            color: #ffffff;
            font-size: 1.5rem;
            cursor: pointer;
            padding: 5px 10px;
            border-radius: 5px;
            transition: background-color 0.2s;
        }

        .modal-close:hover {
            background-color: #ff5555;
        }

        .chunk-item {
            background-color: #1a1a1a;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #3a7be0;
        }

        .chunk-meta {
            color: #888;
            font-size: 0.9rem;
            margin-bottom: 10px;
            font-weight: 500;
        }

        .chunk-text {
            color: #fafafa;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .kg-section {
            margin-bottom: 25px;
        }

        .kg-section-title {
            color: #3a7be0;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 15px;
            border-bottom: 1px solid #3a7be0;
            padding-bottom: 5px;
        }

        .entity-item, .relationship-item, .hierarchy-item {
            background-color: #1a1a1a;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 10px;
            border-left: 3px solid #4CAF50;
        }

        .relationship-item {
            border-left-color: #FF9800;
        }

        .hierarchy-item {
            border-left-color: #9C27B0;
        }

        .entity-name, .relationship-source {
            color: #ffffff;
            font-weight: 600;
            margin-bottom: 5px;
        }

        .entity-type, .entity-description {
            color: #cccccc;
            font-size: 0.9rem;
            margin-bottom: 3px;
        }

        .file-selector {
            margin-bottom: 20px;
        }

        .view-buttons {
            display: flex;
            gap: 15px;
            margin: 20px 0;
            justify-content: center;
            padding: 15px;
            background: linear-gradient(135deg, rgba(58, 123, 224, 0.1), rgba(58, 123, 224, 0.05));
            border-radius: 12px;
            border: 1px solid rgba(58, 123, 224, 0.2);
        }

        .view-button {
            background-color: #3a7be0;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 1rem;
        }

        .view-button:hover {
            background-color: #2a6bd0;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(58, 123, 224, 0.3);
        }

        /* Enhanced button styling for Streamlit buttons */
        div[data-testid="stButton"] > button {
            background: linear-gradient(135deg, #3a7be0, #2a6bd0) !important;
            border: none !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 12px 24px !important;
            border-radius: 10px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 8px rgba(58, 123, 224, 0.2) !important;
        }

        div[data-testid="stButton"] > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 16px rgba(58, 123, 224, 0.4) !important;
            background: linear-gradient(135deg, #2a6bd0, #1a5bc0) !important;
        }

        /* RAG Mode Radio Button Styling */
        .stRadio > div {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border-radius: 10px !important;
            padding: 10px !important;
            border: 1px solid rgba(58, 123, 224, 0.3) !important;
        }

        .stRadio > div > label {
            color: #ffffff !important;
            font-weight: 500 !important;
        }

        /* Mode indicator styling */
        .rag-mode-indicator {
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }

        /* Sidebar button styling */
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            font-size: 0.9rem !important;
            padding: 8px 16px !important;
            margin: 2px 0 !important;
        }

        /* Data view section styling */
        section[data-testid="stSidebar"] h3 {
            margin-top: 20px !important;
            margin-bottom: 10px !important;
            color: #ffffff !important;
            font-size: 1.1rem !important;
        }

        /* Info message styling */
        .stInfo {
            background-color: rgba(58, 123, 224, 0.1) !important;
            border-left: 4px solid #3a7be0 !important;
            border-radius: 8px !important;
        }

        .scrollable-content {
            max-height: 60vh;
            overflow-y: auto;
            padding-right: 10px;
        }

        /* Custom scrollbar for modal content */
        .scrollable-content::-webkit-scrollbar {
            width: 8px;
        }

        .scrollable-content::-webkit-scrollbar-track {
            background: #1a1a1a;
            border-radius: 4px;
        }

        .scrollable-content::-webkit-scrollbar-thumb {
            background: #3a7be0;
            border-radius: 4px;
        }

        .scrollable-content::-webkit-scrollbar-thumb:hover {
            background: #2a6bd0;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def display_header():
    """Display header with only the image logo on left side, medium size."""
    col1, col2 = st.columns([1, 3])  # 1:3 ratio gives logo ~25% width on left

    with col1:  # Left column
        logo_path = Path(__file__).parent / "assets" / "image.png"
        if logo_path.exists():
            # Display logo with medium size (width=150)
            st.image(str(logo_path), width=350)
        else:
            st.markdown("<div style='font-size:42px'>🗿</div>", unsafe_allow_html=True)


def display_chat_messages():
    """Display chat messages from history."""
    # If no messages, show welcome message
    # if not st.session_state.messages:
    #     st.markdown("""
    #     <div class="bot-message">
    #         <p>👋 Welcome to RowBlaze! I'm here to help you work with your documents.</p>
    #         <p>To get started:</p>
    #         <ul>
    #             <li>Upload documents in the sidebar under "Files" tab</li>
    #             <li>Ask me questions about your documents</li>
    #             <li>Try saying "What can you help me with?"</li>
    #         </ul>
    #     </div>
    #     """, unsafe_allow_html=True)
    #     return

    # Display existing messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-message">{message["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bot-message">{message["content"]}</div>',
                unsafe_allow_html=True,
            )


async def process_query(query: str):
    """Process a user query and get response from RAG system using the API."""
    try:
        # Get current user's authentication token
        auth_token = st.session_state.get("auth_token")
        if not auth_token:
            return "Authentication required. Please log in again."

        # Get RAG mode selection
        rag_mode = st.session_state.get("rag_mode", "Normal RAG")
        print(f"🔍 Processing query with RAG mode: {rag_mode}")

        # Create base payload
        base_payload = {
            "question": query,
            "index_name": st.session_state.index_name,
            "top_k_chunks": st.session_state.get("top_k_chunks", 5),
            "enable_references_citations": st.session_state.get(
                "enable_citations", True
            ),
            "deep_research": st.session_state.get("deep_research", False),
            "auto_chunk_sizing": st.session_state.get("auto_chunk_sizing", True),
            "model": st.session_state.get("selected_model", "gpt-4o-mini"),
            "max_tokens": MODEL_TOKEN_LIMITS.get(
                st.session_state.get("selected_model", "gpt-4o-mini"), 16384
            ),
        }

        # Add authentication headers
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }

        # Choose endpoint based on RAG mode
        if rag_mode == "Normal RAG":
            # Use normal RAG endpoint
            endpoint = f"{API_URL}/query-rag"
            payload = base_payload
        else:
            # Use Agentic RAG endpoint
            endpoint = f"{API_URL}/agent-query"
            payload = {
                **base_payload,
                "max_iterations": 2,  # Fixed to 2 iterations as requested
                "use_agent": True,
                "query_complexity_analysis": True,
            }

        # Call the appropriate API endpoint
        print(f"🌐 Calling endpoint: {endpoint}")
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            # Log metadata for agentic mode
            if rag_mode == "Agentic RAG" and "metadata" in result:
                metadata = result["metadata"]
                print(
                    f"🧠 Agent completed: {metadata.get('iterations_completed', 0)} iterations"
                )
                print(f"🔧 Tools used: {metadata.get('tools_used', [])}")

            return result.get("answer", "No answer generated")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            # Clear invalid token and redirect to login
            st.session_state.auth_token = None
            return "Session expired. Please refresh the page to log in again."
        return f"Error: API returned {e.response.status_code}: {e.response.text}"
    except httpx.RequestError as e:
        return f"Error: Request to API failed: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


async def process_file_upload(uploaded_file):
    """Process an uploaded file using the API."""
    try:
        # Get current user's authentication token
        auth_token = st.session_state.get("auth_token")
        if not auth_token:
            return False, "Authentication required. Please log in again."

        # Prepare form data
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        data = {
            "index_name": st.session_state.index_name,
            "description": st.session_state.get(
                "doc_description", f"Uploaded document: {uploaded_file.name}"
            ),
            "is_ocr_pdf": str(st.session_state.get("is_ocr_pdf", False)).lower(),
            "is_structured_pdf": str(
                st.session_state.get("is_structured_pdf", False)
            ).lower(),
            "model": st.session_state.get("selected_model", "gpt-4o-mini"),
            "max_tokens": str(
                MODEL_TOKEN_LIMITS.get(
                    st.session_state.get("selected_model", "gpt-4o-mini"), 16384
                )
            ),
        }

        # Add authentication headers
        headers = {"Authorization": f"Bearer {auth_token}"}

        # Make API call with authentication (increased timeout for large files)
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{API_URL}/ingest",
                files=files,
                data=data,
                headers=headers,  # Add headers here
            )

            if response.status_code == 200:
                result = response.json()
                return True, result.get("message", "File processed successfully")
            elif response.status_code == 401:
                st.session_state.auth_token = None
                return (
                    False,
                    "Session expired. Please refresh the page to log in again.",
                )
            else:
                return False, f"API Error: {response.status_code} - {response.text}"

    except Exception as e:
        return False, f"Processing failed: {str(e)}"


def handle_chat_input():
    """Handle user chat input submission."""
    # Get the input but don't try to clear it directly
    user_query = st.session_state.user_input

    if user_query.strip():
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Set processing flag to display spinner
        st.session_state.processing = True

        # Store query for processing
        st.session_state.current_query = user_query

        # Flag to save history
        st.session_state.save_history = True


def handle_file_upload():
    """Handle file upload from the sidebar."""
    uploaded_file = st.session_state.get("uploaded_file")

    if uploaded_file:
        st.session_state.upload_status = "Processing..."

        with st.spinner("Processing document..."):
            success, message = asyncio.run(process_file_upload(uploaded_file))

        if success:
            st.session_state.upload_status = "✅ " + message
        else:
            st.session_state.upload_status = "🔄" + message


def clear_chat():
    """Clear the chat history."""
    st.session_state.messages = []


def display_chunks_modal(chunks_data, selected_file=None):
    """Display chunks in a modal format."""
    if not chunks_data.get("success"):
        st.error(f"Error loading chunks: {chunks_data.get('error', 'Unknown error')}")
        st.info(
            "This might be due to index mapping issues. Try uploading a new document to refresh the index structure."
        )
        return

    chunks = chunks_data.get("chunks", [])

    # Filter chunks by selected file if specified
    if selected_file and selected_file != "All Files":
        chunks = [chunk for chunk in chunks if chunk.get("file_name") == selected_file]

    if not chunks:
        st.info("No chunks found for the selected file.")
        if selected_file != "All Files":
            st.info("Try selecting 'All Files' to see if there are chunks available.")
        return

    # Add search functionality
    search_term = st.text_input(
        "🔍 Search in chunks:", key="chunk_search", placeholder="Enter search term..."
    )

    # Filter chunks by search term if provided
    if search_term:
        filtered_chunks = []
        for chunk in chunks:
            chunk_text = chunk.get("text", "").lower()
            if search_term.lower() in chunk_text:
                filtered_chunks.append(chunk)
        chunks = filtered_chunks
        st.info(f"Found {len(chunks)} chunks containing '{search_term}'")

    # Group chunks by file
    chunks_by_file = {}
    for chunk in chunks:
        file_name = chunk.get("file_name", "Unknown")
        if file_name not in chunks_by_file:
            chunks_by_file[file_name] = []
        chunks_by_file[file_name].append(chunk)

    # Sort chunks by page number within each file
    for file_name in chunks_by_file:
        chunks_by_file[file_name].sort(
            key=lambda x: (x.get("page_number", 0), x.get("chunk_index", 0))
        )

    # Display summary
    total_chunks = sum(len(file_chunks) for file_chunks in chunks_by_file.values())
    st.markdown(f"**Total Chunks:** {total_chunks} across {len(chunks_by_file)} files")

    # Display chunks
    for file_name, file_chunks in chunks_by_file.items():
        st.markdown(f"### 📄 {file_name}")
        st.markdown(f"*{len(file_chunks)} chunks*")

        for i, chunk in enumerate(file_chunks):
            chunk_preview = (
                chunk.get("text", "No text available")[:150] + "..."
                if len(chunk.get("text", "")) > 150
                else chunk.get("text", "No text available")
            )

            with st.expander(
                f"Chunk {i+1} - Page {chunk.get('page_number', 'N/A')} | {chunk_preview}",
                expanded=False,
            ):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown("**Chunk Text:**")
                    chunk_text = chunk.get("text", "No text available")

                    # Highlight search term if present
                    if search_term and search_term.lower() in chunk_text.lower():
                        # Simple highlighting (case-insensitive)
                        highlighted_text = (
                            chunk_text.replace(search_term, f"**{search_term}**")
                            .replace(search_term.lower(), f"**{search_term.lower()}**")
                            .replace(search_term.upper(), f"**{search_term.upper()}**")
                        )
                        st.markdown(highlighted_text)
                    else:
                        st.text_area(
                            "",
                            value=chunk_text,
                            height=200,
                            key=f"chunk_text_{file_name}_{i}",
                            disabled=True,
                        )

                with col2:
                    st.markdown("**Metadata:**")
                    st.markdown(f"**Page:** {chunk.get('page_number', 'N/A')}")
                    st.markdown(f"**Chunk Index:** {chunk.get('chunk_index', 'N/A')}")
                    st.markdown(f"**Characters:** {len(chunk_text)}")

                    if chunk.get("document_summary"):
                        st.markdown("**Document Summary:**")
                        with st.expander("View Summary", expanded=False):
                            st.markdown(chunk.get("document_summary"))


def display_knowledge_graph_modal(kg_data, selected_file=None):
    """Display knowledge graph data in a modal format."""
    if not kg_data.get("success"):
        st.error(
            f"Error loading knowledge graph: {kg_data.get('error', 'Unknown error')}"
        )
        st.info(
            "This might be due to index mapping issues. Try uploading a new document to refresh the index structure."
        )
        return

    kg = kg_data.get("knowledge_graph", {})
    entities = kg.get("entities", [])
    relationships = kg.get("relationships", [])
    hierarchies = kg.get("hierarchies", [])

    # Filter by selected file if specified
    if selected_file and selected_file != "All Files":
        entities = [e for e in entities if e.get("file_name") == selected_file]
        relationships = [
            r for r in relationships if r.get("file_name") == selected_file
        ]
        hierarchies = [h for h in hierarchies if h.get("file_name") == selected_file]

    # Add search functionality
    search_term = st.text_input(
        "🔍 Search in knowledge graph:",
        key="kg_search",
        placeholder="Search entities, relationships...",
    )

    # Filter by search term if provided
    if search_term:
        search_lower = search_term.lower()
        entities = [
            e
            for e in entities
            if search_lower in e.get("name", "").lower()
            or search_lower in e.get("description", "").lower()
            or search_lower in e.get("type", "").lower()
        ]
        relationships = [
            r
            for r in relationships
            if search_lower in r.get("source_entity", "").lower()
            or search_lower in r.get("target_entity", "").lower()
            or search_lower in r.get("relation", "").lower()
        ]
        hierarchies = [
            h
            for h in hierarchies
            if search_lower in h.get("name", "").lower()
            or search_lower in h.get("description", "").lower()
        ]

    # Display summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Entities", len(entities))
    with col2:
        st.metric("Relationships", len(relationships))
    with col3:
        st.metric("Hierarchies", len(hierarchies))

    # Tabs for different KG components
    tab1, tab2, tab3 = st.tabs(["🏷️ Entities", "🔗 Relationships", "🌳 Hierarchies"])

    with tab1:
        if entities:
            # Group entities by file
            entities_by_file = {}
            for entity in entities:
                file_name = entity.get("file_name", "Unknown")
                if file_name not in entities_by_file:
                    entities_by_file[file_name] = []
                entities_by_file[file_name].append(entity)

            for file_name, file_entities in entities_by_file.items():
                st.markdown(f"#### 📄 {file_name}")

                # Group by entity type
                entities_by_type = {}
                for entity in file_entities:
                    entity_type = entity.get("type", "Unknown")
                    if entity_type not in entities_by_type:
                        entities_by_type[entity_type] = []
                    entities_by_type[entity_type].append(entity)

                for entity_type, type_entities in entities_by_type.items():
                    with st.expander(
                        f"{entity_type} ({len(type_entities)} entities)", expanded=False
                    ):
                        for entity in type_entities:
                            st.markdown(f"**{entity.get('name', 'Unnamed')}**")
                            if entity.get("description"):
                                st.markdown(f"*{entity.get('description')}*")
                            st.markdown(f"📍 Page {entity.get('page_number', 'N/A')}")
                            st.divider()
        else:
            st.info("No entities found.")
            if selected_file != "All Files":
                st.info(
                    "Try selecting 'All Files' or upload documents with more structured content to see entities."
                )

    with tab2:
        if relationships:
            # Group relationships by file
            relationships_by_file = {}
            for rel in relationships:
                file_name = rel.get("file_name", "Unknown")
                if file_name not in relationships_by_file:
                    relationships_by_file[file_name] = []
                relationships_by_file[file_name].append(rel)

            for file_name, file_relationships in relationships_by_file.items():
                st.markdown(f"#### 📄 {file_name}")

                for i, rel in enumerate(file_relationships):
                    with st.expander(
                        f"Relationship {i+1}: {rel.get('source_entity', 'Unknown')} → {rel.get('target_entity', 'Unknown')}",
                        expanded=False,
                    ):
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.markdown(
                                f"**Source:** {rel.get('source_entity', 'Unknown')}"
                            )
                            st.markdown(
                                f"**Relation:** {rel.get('relation', 'Unknown')}"
                            )
                            st.markdown(
                                f"**Target:** {rel.get('target_entity', 'Unknown')}"
                            )
                            if rel.get("relationship_description"):
                                st.markdown(
                                    f"**Description:** {rel.get('relationship_description')}"
                                )

                        with col2:
                            st.markdown(f"**Page:** {rel.get('page_number', 'N/A')}")
                            if rel.get("relationship_weight"):
                                st.markdown(
                                    f"**Weight:** {rel.get('relationship_weight')}"
                                )
        else:
            st.info("No relationships found.")

    with tab3:
        if hierarchies:
            # Group hierarchies by file
            hierarchies_by_file = {}
            for hierarchy in hierarchies:
                file_name = hierarchy.get("file_name", "Unknown")
                if file_name not in hierarchies_by_file:
                    hierarchies_by_file[file_name] = []
                hierarchies_by_file[file_name].append(hierarchy)

            for file_name, file_hierarchies in hierarchies_by_file.items():
                st.markdown(f"#### 📄 {file_name}")

                for i, hierarchy in enumerate(file_hierarchies):
                    with st.expander(
                        f"Hierarchy {i+1}: {hierarchy.get('name', 'Unnamed')}",
                        expanded=False,
                    ):
                        st.markdown(f"**Name:** {hierarchy.get('name', 'Unnamed')}")
                        st.markdown(
                            f"**Description:** {hierarchy.get('description', 'No description')}"
                        )
                        st.markdown(
                            f"**Root Type:** {hierarchy.get('root_type', 'Unknown')}"
                        )
                        st.markdown(f"**Page:** {hierarchy.get('page_number', 'N/A')}")

                        # Display levels with nodes if available
                        levels = hierarchy.get("levels", [])
                        if levels:
                            st.markdown("**Hierarchy Structure:**")
                            for level in levels:
                                level_name = level.get(
                                    "name", f"Level {level.get('id', 'Unknown')}"
                                )
                                st.markdown(
                                    f"**{level_name}** (ID: {level.get('id', 'N/A')})"
                                )

                                # Display level description if available
                                level_desc = level.get("description", "")
                                if level_desc:
                                    st.markdown(f"  *{level_desc}*")

                                # Display nodes in this level
                                nodes = level.get("nodes", [])
                                if nodes:
                                    for node in nodes:
                                        node_name = node.get("name", "Unnamed Node")
                                        node_id = node.get("id", "N/A")
                                        st.markdown(f"  • {node_name} (ID: {node_id})")

                                        # Display children if available
                                        children = node.get("children", [])
                                        if children:
                                            st.markdown("    **Children:**")
                                            for child in children:
                                                child_level = child.get("level", "N/A")
                                                child_node_id = child.get(
                                                    "node_id", "N/A"
                                                )
                                                st.markdown(
                                                    f"      → Level {child_level}, Node {child_node_id}"
                                                )

                                        # Display data sources if available
                                        data_sources = node.get("data_sources", "")
                                        if data_sources:
                                            st.markdown(
                                                f"    **Sources:** {data_sources}"
                                            )
                                else:
                                    st.markdown("  *No nodes defined for this level*")

                        # Display hierarchy relationships if available
                        hierarchy_relationships = hierarchy.get("relationships", [])
                        if hierarchy_relationships:
                            st.markdown("**Hierarchy Relationships:**")
                            for rel in hierarchy_relationships:
                                rel_type = rel.get("type", "Unknown")
                                source = rel.get("source", {})
                                target = rel.get("target", {})

                                source_info = f"Level {source.get('level', 'N/A')}, Node {source.get('node_id', 'N/A')}"
                                target_info = f"Level {target.get('level', 'N/A')}, Node {target.get('node_id', 'N/A')}"

                                st.markdown(
                                    f"  • **{rel_type}:** {source_info} → {target_info}"
                                )

                                # Display relationship description if available
                                rel_desc = rel.get("description", "")
                                if rel_desc:
                                    st.markdown(f"    *{rel_desc}*")

                                # Display data sources for relationship if available
                                rel_data_sources = rel.get("data_sources", "")
                                if rel_data_sources:
                                    st.markdown(f"    **Sources:** {rel_data_sources}")
        else:
            st.info("No hierarchies found.")


MODEL_TOKEN_LIMITS = {
    "gpt-4o-mini-2024-07-18": 16384,
    "gpt-5-mini-2025-08-07": 128000,
    "gpt-oss-120b": 131072,
    "gpt-5-nano-2025-08-07": 128000,
    "gpt-4.1-mini-2025-04-14": 32768,
    "gpt-4.1-nano-2025-04-14": 32768,
}


def create_new_chat():
    """Create a new chat session."""
    # Generate a new session ID
    st.session_state.active_session_id = str(uuid.uuid4())

    # Mark as a new session so we don't try to load history
    st.session_state.new_session = True

    # Clear messages
    st.session_state.messages = []
    st.session_state.load_history_flag = False

    # Refresh the page
    st.rerun()


def main():
    # Debug: Check authentication status
    auth_status = is_authenticated()
    print(f"Authentication status: {auth_status}")

    # Check if user is authenticated
    if not auth_status:
        print("User not authenticated, showing login page")
        try:
            login_page()
        except Exception as e:
            print(f"Error in login_page: {e}")
            # Fallback simple login
            st.title("Login Required")
            st.error(f"Login page error: {e}")
            st.write("Please check the console for more details.")
        return

    print("User authenticated, showing main app")

    # --- 1. INITIALIZATION ---
    initialize_session_state()
    apply_custom_css()

    # Get current user for UI display
    current_user = get_current_user()

    # Render cookie manager once in the sidebar
    with st.sidebar:
        _ = get_cookie_manager()

        # Display user info and logout button
        user_email = current_user.get("email", "Unknown User")
        st.markdown(f"**Logged in as:** {user_email}")
        if st.button("Logout"):
            logout_user()
            st.rerun()

    # --- 2. ASYNC DATA LOADING (RUNS ONCE OR WHEN NEEDED) ---
    # Load chat history if the flag is set (e.g., on first load or session switch)
    if st.session_state.load_history_flag:
        try:
            with st.spinner("Loading conversation..."):
                messages = asyncio.run(
                    load_chat_history(st.session_state.active_session_id)
                )
                st.session_state.messages = messages
                print(
                    f"Loaded {len(messages)} messages for session {st.session_state.active_session_id}"
                )
        except Exception as e:
            print(f"Error loading chat history: {e}")
            st.session_state.messages = []  # Default to empty on error
        finally:
            st.session_state.load_history_flag = False  # Always reset the flag

    # Refresh chat sessions list periodically
    if time.time() - st.session_state.sessions_last_fetched > 60:
        try:
            st.session_state.chat_sessions = asyncio.run(load_chat_sessions())
            st.session_state.sessions_last_fetched = time.time()
        except Exception as e:
            print(f"Error refreshing chat sessions: {e}")

    with st.sidebar:
        tab1, tab2 = st.tabs(["📁 Files", "💬 Chat History"])

        with tab1:
            st.title("File Upload")

            # Display status message for uploads
            if st.session_state.upload_message:
                icon = "✅" if st.session_state.upload_success else "❌"
                st.markdown(
                    f"""
                    <div style="padding: 10px; border-radius: 5px;
                        background-color: {'#1a3d1a' if st.session_state.upload_success else '#5c1c1c'};
                        border-left: 3px solid {'#4CAF50' if st.session_state.upload_success else '#f44336'};
                        margin-bottom: 15px;">
                        {icon} {st.session_state.upload_message}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # RAG Mode Selection (NEW)
            st.markdown("### 🤖 RAG Mode Selection")
            rag_mode = st.radio(
                "Choose RAG Mode:",
                options=["Normal RAG", "Agentic RAG"],
                index=(
                    0
                    if st.session_state.get("rag_mode", "Normal RAG") == "Normal RAG"
                    else 1
                ),
                key="rag_mode",
                help="Normal RAG: Fast, direct retrieval. Agentic RAG: Multi-step reasoning with tool selection.",
            )

            # Display mode description
            if rag_mode == "Normal RAG":
                st.info(
                    "🚀 **Normal RAG**: Direct document retrieval and answer generation. Fast and efficient for straightforward queries."
                )
            else:
                st.info(
                    "🧠 **Agentic RAG**: Multi-step reasoning with intelligent tool selection. Better for complex analysis and multi-faceted queries."
                )

            st.divider()

            # Model selection dropdown (keep at top as it's important)
            available_models = list(MODEL_TOKEN_LIMITS.keys())
            current_model = st.session_state.get("selected_model", "gpt-4o-mini")
            if current_model not in available_models:
                current_model = available_models[0]

            selected_model = st.selectbox(
                "Select LLM",
                options=available_models,
                index=(
                    available_models.index(current_model)
                    if current_model in available_models
                    else 0
                ),
                key="selected_model",
            )

            # MOVED UP: File upload section for better visibility
            st.markdown("### Upload Document")

            # List supported file types before the uploader
            supported_types = [
                "pdf",
                "doc",
                "docx",
                "txt",
                "odt",
                "xlsx",
                "csv",
                "jpg",
                "jpeg",
                "png",
                "gif",
                "bmp",
                "webp",
                "heic",
                "tiff",
                "tif",
            ]

            st.markdown(
                f"<div style='margin-bottom: 10px; color: #aaaaaa;'>Supported file types: {', '.join(supported_types)}</div>",
                unsafe_allow_html=True,
            )

            # File uploader with better positioning
            uploaded_file = st.file_uploader(
                "Upload Files",
                type=supported_types,
                key=st.session_state.file_uploader_key,
                help="Upload your documents to extract information",
            )

            # Document options (moved after uploader)
            with st.expander("Document Options", expanded=False):
                st.checkbox("Structured PDF", key="is_structured_pdf")
                st.checkbox("Scanned PDF (OCR)", key="is_ocr_pdf")
                st.text_area(
                    "Document Description",
                    key="doc_description",
                    placeholder="Optional: Describe the document content",
                )

            # Advanced settings in collapsed expander
            with st.expander("Advanced Settings", expanded=False):
                st.slider(
                    "Number of chunks to retrieve (Top-K)",
                    min_value=3,
                    max_value=100,
                    value=20,
                    step=1,
                    help="Higher values may improve comprehensiveness but increase processing time",
                    key="top_k_chunks",
                )
                st.checkbox(
                    "Enable citations/references", value=True, key="enable_citations"
                )
                st.checkbox(
                    "Auto-adjust chunk sizing", value=True, key="auto_chunk_sizing"
                )

                # Agent-specific settings (only show when Agentic RAG is selected)
                if st.session_state.get("rag_mode") == "Agentic RAG":
                    st.markdown("**🧠 Agentic RAG Settings**")
                    st.info(
                        "Agentic RAG uses 2 iterations and automatically selects tools based on query complexity."
                    )

                    # Show tool selection info
                    st.markdown("**Available Tools:**")
                    st.markdown("• 🔍 **Vector Search**: Semantic similarity search")
                    st.markdown("• 🔤 **Keyword Search**: Exact phrase matching")
                    st.markdown(
                        "• 📚 **File Knowledge Search**: Comprehensive document analysis"
                    )
                    st.markdown("• 🕸️ **Graph Traversal**: Entity relationship analysis")

                    st.markdown(
                        "*Tools are automatically selected based on query complexity and type.*"
                    )

            # File upload handling (no changes)
            if uploaded_file:
                st.session_state.is_processing_file = True
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Check if file already exists
                        if uploaded_file.name in st.session_state.indexed_files:
                            raise ValueError(
                                f"File '{uploaded_file.name}' already exists."
                            )

                        # Process the file
                        success, message = asyncio.run(
                            process_file_upload(uploaded_file)
                        )
                        st.session_state.upload_success = success
                        st.session_state.upload_message = message

                        if not success:
                            raise Exception(message)

                        # On success, refresh file list immediately
                        st.session_state.indexed_files = asyncio.run(fetch_files())
                        st.session_state.files_last_fetched = time.time()

                    except Exception as e:
                        st.session_state.upload_success = False
                        st.session_state.upload_message = f"Error: {str(e)}"

                # Reset the uploader and processing state, then rerun ONCE
                st.session_state.file_uploader_key = f"file_uploader_{int(time.time())}"
                st.session_state.is_processing_file = False
                st.rerun()

            # --- REFACTORED FILE LIST & REFRESH ---
            st.markdown("### Files in Index")
            if st.button(
                "🔄 Refresh List", disabled=st.session_state.is_processing_file
            ):
                with st.spinner("Refreshing..."):
                    try:
                        st.session_state.indexed_files = asyncio.run(fetch_files())
                        st.session_state.files_last_fetched = time.time()
                    except Exception as e:
                        st.error(f"Failed to refresh: {e}")

            if st.session_state.indexed_files:
                for file_name in st.session_state.indexed_files:
                    st.markdown(
                        f'<div class="file-item">📄 {file_name}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div class="file-count">Total files: {len(st.session_state.indexed_files)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("*No files indexed yet*")

            # Add view buttons in sidebar when files are available
            if st.session_state.indexed_files:
                st.markdown("### 📊 Data Views")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "📄 Chunks",
                        key="view_chunks_btn",
                        help="View document chunks",
                        use_container_width=True,
                    ):
                        st.session_state.show_chunks_modal = True
                        st.rerun()

                with col2:
                    if st.button(
                        "🕸️ Graph",
                        key="view_kg_btn",
                        help="View knowledge graph",
                        use_container_width=True,
                    ):
                        st.session_state.show_kg_modal = True
                        st.rerun()

        # --- CHAT HISTORY TAB ---
        with tab2:
            st.title("Chat History")
            if st.button("➕ New Chat", use_container_width=True):
                create_new_chat()
            st.divider()
            # (Your existing chat history display logic is fine here)
            if st.session_state.chat_sessions:
                for session in st.session_state.chat_sessions:
                    # ... your button logic for switching/deleting sessions
                    is_active = (
                        session["session_id"] == st.session_state.active_session_id
                    )
                    button_style = "primary" if is_active else "secondary"
                    if st.button(
                        f"{session.get('title', 'Chat')[:25]}...",
                        key=f"session_{session['session_id']}",
                        type=button_style,
                        use_container_width=True,
                    ):
                        if not is_active:
                            st.session_state.active_session_id = session["session_id"]
                            st.session_state.load_history_flag = True
                            st.rerun()
            else:
                st.info("No chat history yet.")

    # --- 4. MAIN CHAT INTERFACE ---
    display_header()

    # Display current RAG mode indicator
    rag_mode = st.session_state.get("rag_mode", "Normal RAG")
    mode_color = "#4CAF50" if rag_mode == "Normal RAG" else "#FF9800"
    mode_icon = "🚀" if rag_mode == "Normal RAG" else "🧠"

    st.markdown(
        f"""
        <div style="text-align: center; margin: 10px 0; padding: 8px;
                    background-color: rgba(255,255,255,0.1); border-radius: 20px;
                    border: 1px solid {mode_color};">
            <span style="color: {mode_color}; font-weight: 600;">
                {mode_icon} Current Mode: {rag_mode}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Handle modals - Display in main content area
    if st.session_state.get("show_chunks_modal", False):
        # Create a prominent header with close button
        col1, col2 = st.columns([6, 1])
        with col1:
            st.markdown("## 📄 Document Chunks")
        with col2:
            if st.button("✕ Close", key="close_chunks_modal", type="secondary"):
                st.session_state.show_chunks_modal = False
                st.rerun()

        # File selector
        if st.session_state.indexed_files:
            file_options = ["All Files"] + st.session_state.indexed_files
            selected_file = st.selectbox(
                "Select File:", options=file_options, key="chunks_file_selector"
            )

            # Load and display chunks
            try:
                with st.spinner("Loading chunks..."):
                    file_name = selected_file if selected_file != "All Files" else None
                    chunks_data = asyncio.run(fetch_chunks(file_name))
                    display_chunks_modal(chunks_data, selected_file)
            except Exception as e:
                st.error(f"Failed to load chunks: {str(e)}")
                st.info(
                    "Please try refreshing the page or contact support if the issue persists."
                )
        else:
            st.info("No files available to display chunks.")

    elif st.session_state.get("show_kg_modal", False):
        # Create a prominent header with close button
        col1, col2 = st.columns([6, 1])
        with col1:
            st.markdown("## 🕸️ Knowledge Graph")
        with col2:
            if st.button("✕ Close", key="close_kg_modal", type="secondary"):
                st.session_state.show_kg_modal = False
                st.rerun()

        # File selector
        if st.session_state.indexed_files:
            file_options = ["All Files"] + st.session_state.indexed_files
            selected_file = st.selectbox(
                "Select File:", options=file_options, key="kg_file_selector"
            )

            # Load and display knowledge graph
            try:
                with st.spinner("Loading knowledge graph..."):
                    file_name = selected_file if selected_file != "All Files" else None
                    kg_data = asyncio.run(fetch_knowledge_graph(file_name))
                    display_knowledge_graph_modal(kg_data, selected_file)
            except Exception as e:
                st.error(f"Failed to load knowledge graph: {str(e)}")
                st.info(
                    "Please try refreshing the page or contact support if the issue persists."
                )
        else:
            st.info("No files available to display knowledge graph.")

    else:

        # Display chat messages
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        display_chat_messages()
        st.markdown("</div>", unsafe_allow_html=True)

        # Handle response generation if processing
        if st.session_state.processing:
            with st.spinner("Thinking..."):
                query = st.session_state.current_query
                response = asyncio.run(process_query(query))
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                # Save history
                asyncio.run(
                    save_chat_history(
                        st.session_state.active_session_id, st.session_state.messages
                    )
                )
                st.session_state.processing = False
                st.rerun()  # Rerun once to display the new message

        # User input form
        if prompt := st.chat_input("Ask anything about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.current_query = prompt
            st.session_state.processing = True
            st.rerun()


# Helper function for fetching files (to avoid code duplication)
async def fetch_files():
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{API_URL}/files/{st.session_state.index_name}")
        response.raise_for_status()
        return response.json()


async def fetch_chunks(file_name=None):
    """Fetch chunks for a specific file or all files."""
    try:
        auth_token = st.session_state.get("auth_token")
        if not auth_token:
            return {"success": False, "error": "Authentication required"}

        headers = {"Authorization": f"Bearer {auth_token}"}
        params = {"file_name": file_name} if file_name else {}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{API_URL}/chunks/{st.session_state.index_name}",
                params=params,
                headers=headers,
            )

            if response.status_code == 500:
                # Try to get more specific error from response
                try:
                    error_detail = response.json().get(
                        "detail", "Internal server error"
                    )
                except:
                    error_detail = (
                        "Internal server error - possibly due to index mapping issues"
                    )
                return {"success": False, "error": f"Server error: {error_detail}"}

            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return {
            "success": False,
            "error": f"HTTP {e.response.status_code}: {e.response.text}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def fetch_knowledge_graph(file_name=None):
    """Fetch knowledge graph data for a specific file or all files."""
    try:
        auth_token = st.session_state.get("auth_token")
        if not auth_token:
            return {"success": False, "error": "Authentication required"}

        headers = {"Authorization": f"Bearer {auth_token}"}
        params = {"file_name": file_name} if file_name else {}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{API_URL}/knowledge-graph/{st.session_state.index_name}",
                params=params,
                headers=headers,
            )

            if response.status_code == 500:
                # Try to get more specific error from response
                try:
                    error_detail = response.json().get(
                        "detail", "Internal server error"
                    )
                except:
                    error_detail = (
                        "Internal server error - possibly due to index mapping issues"
                    )
                return {"success": False, "error": f"Server error: {error_detail}"}

            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return {
            "success": False,
            "error": f"HTTP {e.response.status_code}: {e.response.text}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    main()
