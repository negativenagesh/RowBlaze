# rag_chatbot.py - Streamlit RAG chatbot with Grok-inspired UI
#
# To run: streamlit run rag_chatbot.py

import os
import asyncio
import streamlit as st
from pathlib import Path
import tempfile
import json
import time
import sys
from typing import List, Tuple

# Add project root to path to import modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary modules
from src.core.retrieval.rag_retrieval import handle_request as retrieval_handle_request
from src.core.ingestion.rag_ingestion import handle_request as ingestion_handle_request
from src.core.retrieval.rag_retrieval import check_async_elasticsearch_connection
from sdk.message import Message
from sdk.response import FunctionResponse, Messages

# Set page configuration
st.set_page_config(
    page_title="RowBlaze",
    page_icon="🗿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define helper functions for UI rendering and chat logic
def initialize_session_state():
    """Initialize session state variables for chat history and settings."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "index_name" not in st.session_state:
        st.session_state.index_name = "rowblaze"  # Default index
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
            asyncio.run(ensure_index_exists(st.session_state.index_name))
            st.session_state.index_checked = True
        except Exception as e:
            print(f"Error creating index: {e}")
            st.session_state.index_checked = False
    
    # Only fetch files once at startup or when explicitly requested
    if not st.session_state.indexed_files:  # Only fetch if we have no files
        try:
            print("Initial file list fetch...")
            st.session_state.indexed_files = asyncio.run(fetch_unique_files_from_es(st.session_state.index_name))
            st.session_state.files_last_fetched = time.time()
            print(f"Fetched {len(st.session_state.indexed_files)} files")
        except Exception as e:
            print(f"Error fetching files: {e}")
    
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = "uploaded_file_0"

    # Initialize the user input state
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

def apply_custom_css():
    """Apply custom CSS for Grok-inspired UI."""
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
    """Display the header with logo."""
    col1, col2 = st.columns([1, 15])
    with col1:
        st.markdown("""
        <div class="logo-container">
            <span style="font-size: 40px; color: #3a7be0; display: flex; align-items: center; justify-content: center; margin-top: 8px;">🗿</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.title("RowBlaze")
        st.markdown("<p style='color: #999; margin-top: -10px;'>RAG for Structured data</p>", unsafe_allow_html=True)

def display_chat_messages():
    """Display chat messages from history."""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)

async def process_query(query: str):
    """Process a user query and get response from RAG system."""
    try:
        # Create message object for retrieval
        message = Message(
            params={
                "question": query,
                "top_k_chunks": st.session_state.get("top_k_chunks", 5),
                "enable_references_citations": st.session_state.get("enable_citations", True),
                "deep_research": st.session_state.get("deep_research", False),
                "auto_chunk_sizing": st.session_state.get("auto_chunk_sizing", True),
            },
            config={
                "index_name": st.session_state.index_name,
            }
        )
        
        # Call the retrieval function
        response = await retrieval_handle_request(message)
        
        if response.failed:
            return f"Error: {response.message}"
        
        # Extract the actual message content without the Message wrapper
        if hasattr(response.message, 'message'):
            if isinstance(response.message.message, dict) and 'final_answer' in response.message.message:
                return response.message.message['final_answer']
            elif isinstance(response.message.message, str):
                return response.message.message
        
        context = str(response.message)
        if "Original Query:" in context and "Vector Search Results" in context:
            # This is raw context - we need to generate a final answer
            final_answer_message = Message(
                params={
                    "question": query,
                    "context": context,
                    "generate_final_answer": True,
                    "enable_references_citations": True,
                },
                config={
                    "index_name": st.session_state.index_name,
                }
            )
            
            # Request final answer generation
            final_response = await retrieval_handle_request(final_answer_message)
            if not final_response.failed and final_response.message:
                if hasattr(final_response.message, 'message'):
                    if isinstance(final_response.message.message, dict) and 'final_answer' in final_response.message.message:
                        return final_response.message.message['final_answer']
                    else:
                        return str(final_response.message.message)
                return str(final_response.message)
                
        # Fallback: just return whatever we got
        return str(response.message)
    except Exception as e:
        return f"An error occurred: {str(e)}"

async def process_file_upload(uploaded_file):
    """Process an uploaded file using the ingestion module."""
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as temp:
            temp.write(uploaded_file.getvalue())
            temp_path = temp.name
        
        # Create message object for ingestion
        params = {
            "index_name": st.session_state.index_name,
            "file_name": uploaded_file.name,
            "file_path": temp_path,
            "description": st.session_state.get("doc_description", f"Uploaded document: {uploaded_file.name}"),
            "is_ocr_pdf": st.session_state.get("is_ocr", False),
            "chunk_size": 1024,
            "chunk_overlap": 512,
        }
        
        message = Message(
            params=params,
            config={
                "api_key": os.getenv("OPEN_AI_KEY"),
            }
        )
        
        # Call the ingestion function
        response = await ingestion_handle_request(message)
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        
        if response.failed:
            return False, f"Error: {response.message}"
        
        # Refresh the indexed files list after successful upload
        print("File uploaded successfully, refreshing file list...")
        # Add a small delay to ensure the document is indexed
        await asyncio.sleep(2)
        st.session_state.indexed_files = await fetch_unique_files_from_es(st.session_state.index_name)
        st.session_state.files_last_fetched = time.time()
        
        return True, "Successfully processed and indexed"
    except Exception as e:
        # Clean up temp file on error
        try:
            if 'temp_path' in locals():
                os.unlink(temp_path)
        except OSError:
            pass
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
            st.session_state.upload_status = "❌ " + message

def clear_chat():
    """Clear the chat history."""
    st.session_state.messages = []

async def ensure_index_exists(index_name: str):
    """Ensure the Elasticsearch index exists, creating it if necessary."""
    try:
        from elasticsearch import AsyncElasticsearch
        import os
        import json
        from src.core.ingestion.rag_ingestion import CHUNKED_PDF_MAPPINGS

        # Get credentials from environment
        ELASTICSEARCH_URL = os.getenv("RAG_UPLOAD_ELASTIC_URL")
        ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")
        
        es_client = AsyncElasticsearch(
            ELASTICSEARCH_URL,
            api_key=ELASTICSEARCH_API_KEY,
            request_timeout=60,
            retry_on_timeout=True
        )
        
        if not await es_client.indices.exists(index=index_name):
            print(f"Creating missing index '{index_name}'...")
            await es_client.indices.create(index=index_name, body=CHUNKED_PDF_MAPPINGS)
            print(f"✅ Index '{index_name}' created successfully.")
        else:
            print(f"✅ Index '{index_name}' already exists.")
            
        await es_client.close()
        return True
        
    except Exception as e:
        print(f"❌ Error ensuring index exists: {e}")
        import traceback
        traceback.print_exc()
        return False

# Add this function to fetch unique files from Elasticsearch

async def fetch_unique_files_from_es(index_name: str) -> List[str]:
    """Fetch all unique file names from the Elasticsearch index using the correct field mapping."""
    try:
        # Import the connection function from the same module used in retrieval
        from src.core.retrieval.rag_retrieval import check_async_elasticsearch_connection
        
        es_client = await check_async_elasticsearch_connection()
        if not es_client:
            print("Could not connect to Elasticsearch to fetch files.")
            return []
            
        try:
            # First check if the index exists
            if not await es_client.indices.exists(index=index_name):
                print(f"Index '{index_name}' does not exist")
                return []
            
            # Get aggregation of unique file names using the exact field mapping from ingestion
            # FIXED: Remove the 'size' parameter and use only 'body'
            response = await es_client.search(
                index=index_name,
                body={
                    "size": 0,  # Moved inside body
                    "aggs": {
                        "unique_files": {
                            "terms": {
                                "field": "metadata.file_name",  # Use the exact field from CHUNKED_PDF_MAPPINGS
                                "size": 1000  # Get up to 1000 unique file names
                            }
                        }
                    }
                }
            )
            
            # Extract file names from the aggregation buckets
            unique_files = []
            if "aggregations" in response and "unique_files" in response["aggregations"]:
                buckets = response["aggregations"]["unique_files"]["buckets"]
                print(f"Found {len(buckets)} unique files in aggregation")
                for bucket in buckets:
                    file_name = bucket["key"]
                    doc_count = bucket["doc_count"]
                    unique_files.append(file_name)
                    print(f"File: {file_name} (chunks: {doc_count})")
            else:
                print("No aggregations found in response")
                # Fallback: try to get some sample documents to see the structure
                sample_response = await es_client.search(
                    index=index_name,
                    body={
                        "size": 5,
                        "_source": ["metadata.file_name"],
                        "query": {"match_all": {}}
                    }
                )
                
                if "hits" in sample_response and "hits" in sample_response["hits"]:
                    print("Sample documents found:")
                    for hit in sample_response["hits"]["hits"]:
                        print(f"Document structure: {hit.get('_source', {})}")
                        if "metadata" in hit["_source"] and "file_name" in hit["_source"]["metadata"]:
                            file_name = hit["_source"]["metadata"]["file_name"]
                            if file_name not in unique_files:
                                unique_files.append(file_name)
                    
            print(f"Total unique files found: {len(unique_files)}")
            return sorted(unique_files) if unique_files else []
            
        finally:
            if es_client and hasattr(es_client, 'close'):
                await es_client.close()
                print("Elasticsearch client closed.")
            
    except Exception as e:
        print(f"Error fetching unique files: {e}")
        import traceback
        traceback.print_exc()
        return []

# Main application function
def main():
    initialize_session_state()
    apply_custom_css()
    
    # Sidebar - document upload
    with st.sidebar:
        st.title("Document Upload")
        
        # Show file processing status
        if st.session_state.upload_message:
            message_style = "success" if st.session_state.upload_success else "error"
            icon = "✅" if st.session_state.upload_success else "❌"
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; 
                background-color: {'#1a3d1a' if st.session_state.upload_success else '#5c1c1c'}; 
                border-left: 3px solid {'#4CAF50' if st.session_state.upload_success else '#f44336'};
                margin-bottom: 15px;">
                {icon} {st.session_state.upload_message}
            </div>
            """, unsafe_allow_html=True)
        
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
                PDF, XLSX, CSV
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
                    
                    # Refresh file list only after successful upload
                    if success:
                        st.session_state.indexed_files = asyncio.run(
                            fetch_unique_files_from_es(st.session_state.index_name))
                        st.session_state.files_last_fetched = time.time()
        
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
                    # Only refresh files when button is clicked
                    st.session_state.indexed_files = asyncio.run(fetch_unique_files_from_es(st.session_state.index_name))
                    st.session_state.files_last_fetched = time.time()
                    st.session_state.upload_message = "File list refreshed"
                    st.session_state.upload_success = True
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
            
        with col1:
            st.text_input("Type your message...", 
                          key=st.session_state.input_key,
                          placeholder="Ask anything about the document...",
                          label_visibility="collapsed")
        
        # Create a button with a callback function
        def send_callback():
            user_query = st.session_state[st.session_state.input_key]
            if user_query.strip():
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": user_query})
                # Set processing flag to display spinner
                st.session_state.processing = True
                # Store query for processing
                st.session_state.current_query = user_query
                # Generate a new key for the input box to clear it
                st.session_state.input_key = f"user_input_{int(time.time())}"
        
        with col2:
            st.button("Send", on_click=send_callback)

# Run the application
if __name__ == "__main__":
    main()