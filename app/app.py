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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# API settings
API_URL = os.getenv("ROWBLAZE_API_URL", "http://localhost:8000/api")

st.set_page_config(
    page_title="RowBlaze",
    page_icon="🗿",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables for chat history and settings."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "index_name" not in st.session_state:
        st.session_state.index_name = "rowblaze"  # Default index
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "is_processing_file" not in st.session_state:
        st.session_state.is_processing_file = False  # Flag for file processing
    if "upload_message" not in st.session_state:
        st.session_state.upload_message = ""  # Upload status message
    if "upload_success" not in st.session_state:
        st.session_state.upload_success = False  # Success state for styling
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []  # Store fetched file names
    if "files_last_fetched" not in st.session_state:
        st.session_state.files_last_fetched = 0
    if "last_refresh_time" not in st.session_state:
        st.session_state.last_refresh_time = time.time()

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
    
    # Only fetch files once at startup or when explicitly requested
    if not st.session_state.indexed_files:  # Only fetch if we have no files
        try:
            print("Initial file list fetch...")
            response = requests.get(f"{API_URL}/files/{st.session_state.index_name}")
            if response.status_code == 200:
                st.session_state.indexed_files = response.json()
                st.session_state.files_last_fetched = time.time()
                print(f"Fetched {len(st.session_state.indexed_files)} files")
        except Exception as e:
            print(f"Error fetching files: {e}")
    
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = "uploaded_file_0"

    # Initialize the user input state
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

# Add your existing CSS styling function here
def apply_custom_css():
    """Apply custom CSS for Grok-inspired UI."""
    # Your existing CSS styling code from app.py
    st.markdown("""
    <style>
        /* Main theme - updated background with new gradient, light text */
        .stApp {
            background: #2e2b2b;
            background: linear-gradient(90deg, rgba(46, 43, 43, 1) 100%, rgba(0, 212, 255, 1) 0%);
            color: #fafafa;
            font-family: 'Montserrat', 'Roboto', sans-serif;
        }
        
        /* Sidebar specific styling with black gradient */
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
        
        /* Chat message styling */
        .user-message {
            background-color: #3a7be0;
            color: white;
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
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display header with only the image logo on left side, medium size."""
    col1, col2 = st.columns([1, 3])  # 1:3 ratio gives logo ~25% width on left
    
    with col1:  # Left column
        logo_path = Path(__file__).parent / "assets" / "cover.png"
        if logo_path.exists():
            # Display logo with medium size (width=150)
            st.image(str(logo_path), width=250)
        else:
            st.markdown("<div style='font-size:42px'>🗿</div>", unsafe_allow_html=True)
    

def display_chat_messages():
    """Display chat messages from history."""
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

def main():
    initialize_session_state()
    apply_custom_css()
    
    # Sidebar - document upload
    with st.sidebar:
        st.title("Document Upload")
        
        # Show file processing status
        if st.session_state.upload_message:
            # MODIFY THIS SECTION
            if st.session_state.is_processing_file:
                # Special case for processing state - use the loading icon
                icon = "🔄"
                message_style = "processing"
                background_color = "#1f3b6a"  # Blue-ish background for processing
                border_color = "#3a7be0"
            else:
                # Success or error state
                message_style = "success" if st.session_state.upload_success else "error"
                icon = "✅" if st.session_state.upload_success else "❌"
                background_color = "#1a3d1a" if st.session_state.upload_success else "#5c1c1c"
                border_color = "#4CAF50" if st.session_state.upload_success else "#f44336"
            
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; 
                background-color: {background_color}; 
                border-left: 3px solid {border_color};
                margin-bottom: 15px;">
                {icon} {st.session_state.upload_message}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Available OpenAI models")
        available_models = [
            "gpt-4o-mini-2024-07-18",
            "gpt-4.1-nano-2025-04-14",
            "gpt-5-nano-2025-08-07",
            "gpt-oss-120b"
        ]

        model_options = [f"{model} ({MODEL_TOKEN_LIMITS[model]:,} tokens)" for model in available_models]
        
        selected_option = st.selectbox(
            "Select LLM Model",
            options=model_options,
            index=model_options.index(f"{st.session_state.selected_model} ({MODEL_TOKEN_LIMITS[st.session_state.selected_model]:,} tokens)") 
                if st.session_state.selected_model in available_models else 0,
            key="model_selector_display",
            help="Choose the OpenAI model to use for processing queries"
        )
        
        selected_model = selected_option.split(" (")[0]
        
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.success(f"Model changed to: {selected_model} with {MODEL_TOKEN_LIMITS[selected_model]:,} max output tokens")
        
        is_structured_pdf = st.checkbox(
            "Structured PDF",
            value=False,
            help="Use only if your pdf contains tables(structured data) like csv"
        )
        st.session_state["is_structured_pdf"] = is_structured_pdf
        
        is_ocr_pdf = st.checkbox(
            "Scanned PDF (OCR)",
            value=False,
            help="Use OCR (Optical Character Recognition) for scanned PDFs that contain images of text."
        )
        st.session_state["is_ocr_pdf"] = is_ocr_pdf
        
        st.text_area("Document Description", 
                    placeholder="Enter a description of the document", 
                    key="doc_description",
                    help="This description helps the system understand the document's content",
                    disabled=st.session_state.is_processing_file)
        
        # Set a clean, custom placeholder for the file uploader
        custom_css = """
        <style>
        .uploadedFile {display: none}
        .stFileUploader > label {display: none}
        .stFileUploader [data-testid="stFileUploadDropzone"] {
            padding-top: 4px;
            padding-bottom: 4px;
        }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)
        
        # File uploader with appropriate message based on processing state
        if st.session_state.is_processing_file:
            st.markdown("""
            <div style="text-align: center; color: #3a7be0; font-size: 0.9em; padding: 5px 0;
                        animation: pulse 1.5s infinite; background-color: #1f3b6a; 
                        border-radius: 5px; border: 1px solid #3a7be0;">
                <span style="font-size: 16px;">🔄</span> Processing file...
            </div>
            <style>
            @keyframes pulse {
                0% { opacity: 0.7; }
                50% { opacity: 1; }
                100% { opacity: 0.7; }
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Show spinner during processing
            with st.spinner("Processing document..."):
                pass  # Visual indicator only
        else:
            # Updated file uploader with dynamic key
            uploaded_file = st.file_uploader(
                "Upload Document", 
                type=["pdf", "xlsx", "csv"], 
                key=st.session_state.file_uploader_key,  # Use dynamic key from session state
                label_visibility="hidden"
            )
            
            # Only display our custom message, without the limit text
            if not uploaded_file:
                st.markdown("""
                <div style="text-align: center; color: #999; font-size: 0.9em; padding: 5px 0;">
                Drag and drop file here<br>
                PDF, DOCX, DOC, TXT, ODT, XLSX, CSV
                </div>
                """, unsafe_allow_html=True)
            
            # Process file when uploaded
            if uploaded_file:
                # Check if file already exists in the index
                if uploaded_file.name in st.session_state.indexed_files:
                    st.session_state.upload_success = False
                    st.session_state.upload_message = f"File '{uploaded_file.name}' already exists in the index."
                    st.rerun()
                else:
                    st.session_state.is_processing_file = True
                    st.session_state.upload_message = f"Processing {uploaded_file.name}..."
                    st.session_state.upload_success = False
                    st.rerun()  # Rerun to show processing state
        
        # Handle file processing after UI shows processing state
        if st.session_state.is_processing_file and st.session_state.file_uploader_key in st.session_state:
            # Get the file from session state using the current key
            uploaded_file = st.session_state[st.session_state.file_uploader_key]
            
            # Process the file
            if uploaded_file:
                try:
                    success, message = asyncio.run(process_file_upload(uploaded_file))
                    
                    # Update status based on result
                    st.session_state.upload_success = success
                    st.session_state.upload_message = f"{message}"
                    
                    # Reset file processing state
                    st.session_state.is_processing_file = False
                    
                    # Only generate a new key for the file uploader to reset it
                    st.session_state.file_uploader_key = f"uploaded_file_{int(time.time())}"
                    
                    # Refresh file list after processing
                    try:
                        response = requests.get(f"{API_URL}/files/{st.session_state.index_name}")
                        if response.status_code == 200:
                            st.session_state.indexed_files = response.json()
                            st.session_state.files_last_fetched = time.time()
                    except Exception as e:
                        print(f"Error refreshing file list: {e}")
                    
                    st.rerun()  # Show success/error message
                except Exception as e:
                    st.session_state.is_processing_file = False
                    st.session_state.upload_success = False
                    st.session_state.upload_message = f"Error: {str(e)}"
                    st.rerun()
        
        # Add a refresh button for files
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### Files in Index")
        with col2:
            if st.button("🔄", help="Refresh file list", disabled=st.session_state.is_processing_file):
                with st.spinner("Refreshing..."):
                    try:
                        response = requests.get(f"{API_URL}/files/{st.session_state.index_name}")
                        if response.status_code == 200:
                            st.session_state.indexed_files = response.json()
                            st.session_state.files_last_fetched = time.time()
                            st.session_state.upload_message = "File list refreshed"
                            st.session_state.upload_success = True
                    except Exception as e:
                        st.session_state.upload_message = f"Error refreshing file list: {str(e)}"
                        st.session_state.upload_success = False
                    st.rerun()

        # Update file display section in the sidebar - simplified without delete buttons
        if st.session_state.indexed_files:
            for file_name in st.session_state.indexed_files:
                st.markdown(f"""
                <div class="file-item">
                    📄 {file_name}
                </div>
                """, unsafe_allow_html=True)
            
            # Show total count
            st.markdown(f"""
            <div class="file-count">
                Total files: {len(st.session_state.indexed_files)}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("*No files indexed yet*")
            st.markdown("Upload a document to get started!")
    
    # Main area
    display_header()
    
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    display_chat_messages()
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.session_state.processing:
        with st.spinner("Thinking..."):
            # Use the stored query
            query = st.session_state.get("current_query", "")
            if query:
                response = asyncio.run(process_query(query))
                
                # Add bot response to chat
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Clear the current query and reset input
                st.session_state.current_query = ""
                
            st.session_state.processing = False
            st.rerun()  # Important: Update UI to show the new message

    # User input box with placeholder text
    user_input_container = st.container()
    with user_input_container:
        col1, col2 = st.columns([8, 1])

        # Initialize a key for the input box if it doesn't exist
        if "input_key" not in st.session_state:
            st.session_state.input_key = "user_input_0"

        # Single callback used by both Enter and Send button
        def send_callback():
            user_query = st.session_state.get(st.session_state.input_key, "").strip()
            if user_query:
                st.session_state.messages.append({"role": "user", "content": user_query})
                st.session_state.processing = True
                st.session_state.current_query = user_query
                # Clear input by rotating the key
                st.session_state.input_key = f"user_input_{int(time.time())}"

        with col1:
            st.text_input(
                "Type your message...",
                key=st.session_state.input_key,
                placeholder="Ask anything about the document...",
                label_visibility="collapsed",
                on_change=send_callback  # Enter triggers this
            )

        with col2:
            st.button("Send", on_click=send_callback)

# Run the application
if __name__ == "__main__":
    main()