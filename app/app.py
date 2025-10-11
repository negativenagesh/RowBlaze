import os
import requests
import streamlit as st
from pathlib import Path
import json
import time
import sys
from typing import List, Tuple
import httpx
import asyncio
import uuid
from datetime import datetime, timedelta
# CHANGE: safe import for cookie component
try:
    import extra_streamlit_components as stx
except Exception:
    stx = None

# Add this function at the top of your file to create a global cookie manager
def get_cookie_manager():
    """Get or create a global cookie manager instance."""
    try:
        if stx is None:
            # Fallback when extra_streamlit_components isn't available
            class DummyCookieManager:
                def get(self, key): return None
                def set(self, key, value, **kwargs): pass
            return DummyCookieManager()
        if "cookie_manager" not in st.session_state:
            # Ensure component is created with a stable key
            st.session_state.cookie_manager = stx.CookieManager(key="global_cookie_manager")
        return st.session_state.cookie_manager
    except Exception as e:
        print(f"Error initializing cookie manager: {e}")
        class DummyCookieManager:
            def get(self, key): return None
            def set(self, key, value, **kwargs): pass
        return DummyCookieManager()

API_URL = os.getenv("ROWBLAZE_API_URL", "http://localhost:8000/api")

st.set_page_config(
    page_title="RowBlaze",
    page_icon="🗿",
    layout="wide",
    initial_sidebar_state="expanded"
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
            cookie_manager.set("rowblaze_session", st.session_state.session_id, expires_at=datetime.now() + timedelta(days=30))
    
    return st.session_state.session_id

async def save_chat_history(session_id, messages):
    """Save chat history to the API."""
    try:
        # Convert messages for API
        api_messages = [
            {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": datetime.now().isoformat()
            }
            for msg in messages
        ]
        
        # Create payload
        payload = {
            "session_id": session_id,
            "messages": api_messages,
            "timestamp": datetime.now().isoformat()
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

def initialize_session_state():
    """Initialize session state variables for chat history and settings."""
    # Get or generate session ID first
    session_id = get_session_id()

    # --- One-time initialization block ---
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.index_name = "rowblaze"
        st.session_state.selected_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")
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
        st.session_state.load_history_flag = True # Load history on first run
        st.session_state.file_uploader_key = "file_uploader_0"
        st.session_state.user_input = ""

        # Initial health check
        try:
            response = requests.get(f"{API_URL}/health")
            st.session_state.index_checked = (response.status_code == 200)
        except Exception as e:
            print(f"Error checking API health: {e}")
            st.session_state.index_checked = False
    
    # Add chat sessions state
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = []
    if "sessions_last_fetched" not in st.session_state:
        st.session_state.sessions_last_fetched = 0
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = session_id  # Default to current session
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
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                st.session_state.index_checked = True
            else:
                st.session_state.index_checked = False
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
    st.markdown("""
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
    </style>
    """, unsafe_allow_html=True)

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
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)

async def process_query(query: str):
    """Process a user query and get response from RAG system using the API."""
    try:
        # Create payload for API call
        payload = {
            "question": query,
            "index_name": st.session_state.index_name,
            "top_k_chunks": st.session_state.get("top_k_chunks", 5),
            "enable_references_citations": st.session_state.get("enable_citations", True),
            "deep_research": st.session_state.get("deep_research", False),
            "auto_chunk_sizing": st.session_state.get("auto_chunk_sizing", True),
            "model": st.session_state.selected_model,
            "max_tokens": MODEL_TOKEN_LIMITS[st.session_state.selected_model]
        }
        
        # Call the retrieval API
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f"{API_URL}/query", json=payload)
            response.raise_for_status()
            result = response.json()
            
            return result.get("answer", "No answer generated")
            
    except httpx.RequestError as e:
        return f"Error: Request to API failed: {str(e)}"
    except httpx.HTTPStatusError as e:
        return f"Error: API returned {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

async def process_file_upload(uploaded_file):
    """Process an uploaded file using the API."""
    try:
        # Prepare form data
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        data = {
            "index_name": st.session_state.index_name,
            "description": st.session_state.get("doc_description", f"Uploaded document: {uploaded_file.name}"),
            "is_ocr_pdf": str(st.session_state.get("is_ocr_pdf", False)).lower(),
            "is_structured_pdf": str(st.session_state.get("is_structured_pdf", False)).lower(),
            "model": st.session_state.selected_model,
            "max_tokens": str(MODEL_TOKEN_LIMITS[st.session_state.selected_model])
        }
        
        # Make API call
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{API_URL}/ingest", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Refresh the indexed files list after successful upload
                files_response = await client.get(f"{API_URL}/files/{st.session_state.index_name}")
                if files_response.status_code == 200:
                    st.session_state.indexed_files = files_response.json()
                    st.session_state.files_last_fetched = time.time()
                
                return True, result.get("message", "Successfully processed and indexed")
            else:
                return False, f"Error: API returned status {response.status_code}: {response.text}"
                
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

MODEL_TOKEN_LIMITS = {
    "gpt-4o-mini-2024-07-18": 16384,
    "gpt-4.1-nano-2025-04-14": 32768,
    "gpt-5-nano-2025-08-07": 128000,
    "gpt-oss-120b": 131072
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
    # --- 1. INITIALIZATION ---
    initialize_session_state()
    apply_custom_css()
    # Render cookie manager once in the sidebar
    with st.sidebar:
        _ = get_cookie_manager()

    # --- 2. ASYNC DATA LOADING (RUNS ONCE OR WHEN NEEDED) ---
    # Load chat history if the flag is set (e.g., on first load or session switch)
    if st.session_state.load_history_flag:
        try:
            with st.spinner("Loading conversation..."):
                messages = asyncio.run(load_chat_history(st.session_state.active_session_id))
                st.session_state.messages = messages
                print(f"Loaded {len(messages)} messages for session {st.session_state.active_session_id}")
        except Exception as e:
            print(f"Error loading chat history: {e}")
            st.session_state.messages = [] # Default to empty on error
        finally:
            st.session_state.load_history_flag = False # Always reset the flag

    # Refresh chat sessions list periodically
    if time.time() - st.session_state.sessions_last_fetched > 60:
        try:
            st.session_state.chat_sessions = asyncio.run(load_chat_sessions())
            st.session_state.sessions_last_fetched = time.time()
        except Exception as e:
            print(f"Error refreshing chat sessions: {e}")

    # --- 3. SIDEBAR UI ---
    with st.sidebar:
        tab1, tab2 = st.tabs(["📁 Files", "💬 Chat History"])

        # --- FILES TAB ---
        with tab1:
            st.title("File Upload")

            # Display status message for uploads
            if st.session_state.upload_message:
                icon = "✅" if st.session_state.upload_success else "❌"
                st.markdown(f"""
                <div style="padding: 10px; border-radius: 5px;
                    background-color: {'#1a3d1a' if st.session_state.upload_success else '#5c1c1c'};
                    border-left: 3px solid {'#4CAF50' if st.session_state.upload_success else '#f44336'};
                    margin-bottom: 15px;">
                    {icon} {st.session_state.upload_message}
                </div>
                """, unsafe_allow_html=True)

            # Model selection and other options...
            # (Your existing code for model selection, checkboxes, etc. is fine here)
            available_models = list(MODEL_TOKEN_LIMITS.keys())
            st.selectbox("Select LLM", options=available_models, key="selected_model")
            st.checkbox("Structured PDF", key="is_structured_pdf")
            st.checkbox("Scanned PDF (OCR)", key="is_ocr_pdf")
            st.text_area("Document Description", key="doc_description")

            # --- REFACTORED FILE UPLOADER ---
            uploaded_file = st.file_uploader(
                "Upload a file",
                type=["pdf", "doc", "docx", "txt", "odt", "xlsx", "csv", "jpg", "jpeg", "png", "gif", "bmp", "webp", "heic", "tiff", "tif"],
                key=st.session_state.file_uploader_key,
                label_visibility="collapsed"
            )

            if uploaded_file:
                st.session_state.is_processing_file = True
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Check if file already exists
                        if uploaded_file.name in st.session_state.indexed_files:
                            raise ValueError(f"File '{uploaded_file.name}' already exists.")

                        # Process the file
                        success, message = asyncio.run(process_file_upload(uploaded_file))
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
            if st.button("🔄 Refresh List", disabled=st.session_state.is_processing_file):
                with st.spinner("Refreshing..."):
                    try:
                        st.session_state.indexed_files = asyncio.run(fetch_files())
                        st.session_state.files_last_fetched = time.time()
                    except Exception as e:
                        st.error(f"Failed to refresh: {e}")

            if st.session_state.indexed_files:
                for file_name in st.session_state.indexed_files:
                    st.markdown(f'<div class="file-item">📄 {file_name}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="file-count">Total files: {len(st.session_state.indexed_files)}</div>', unsafe_allow_html=True)
            else:
                st.markdown("*No files indexed yet*")

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
                    is_active = session["session_id"] == st.session_state.active_session_id
                    button_style = "primary" if is_active else "secondary"
                    if st.button(f"{session.get('title', 'Chat')[:25]}...", key=f"session_{session['session_id']}", type=button_style, use_container_width=True):
                        if not is_active:
                            st.session_state.active_session_id = session["session_id"]
                            st.session_state.load_history_flag = True
                            st.rerun()
            else:
                st.info("No chat history yet.")


    # --- 4. MAIN CHAT INTERFACE ---
    display_header()

    # Display chat messages
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    display_chat_messages()
    st.markdown("</div>", unsafe_allow_html=True)

    # Handle response generation if processing
    if st.session_state.processing:
        with st.spinner("Thinking..."):
            query = st.session_state.current_query
            response = asyncio.run(process_query(query))
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Save history
            asyncio.run(save_chat_history(st.session_state.active_session_id, st.session_state.messages))
            st.session_state.processing = False
            st.rerun() # Rerun once to display the new message

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

if __name__ == "__main__":
    main()