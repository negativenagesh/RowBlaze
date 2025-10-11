import asyncio
import os
import time
from typing import Any, Callable, Dict, List

import streamlit as st


def stream_response(response_text):
    """
    Function to stream text responses character by character
    for a more realistic chat experience
    """
    message_placeholder = st.empty()
    full_response = ""

    # Simulate streaming
    for chunk in response_text.split():
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")

    # Replace with the final response
    message_placeholder.markdown(full_response)

    return full_response


def display_file_info(file_info: Dict[str, Any]) -> None:
    """
    Display information about an uploaded file
    """
    st.markdown(f"**{file_info['name']}**")

    if file_info.get("description"):
        st.markdown(f"*{file_info['description']}*")

    if file_info.get("timestamp"):
        from datetime import datetime

        timestamp = datetime.fromtimestamp(file_info["timestamp"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        st.caption(f"Uploaded on {timestamp}")


def create_temp_directory():
    """
    Create a temporary directory for file uploads if it doesn't exist
    """
    if not os.path.exists("/tmp"):
        os.makedirs("/tmp")
    return "/tmp"


def display_json(json_data):
    """
    Format and display JSON data
    """
    try:
        if isinstance(json_data, str):
            import json

            data = json.loads(json_data)
        else:
            data = json_data

        import pandas as pd

        if isinstance(data, list):
            # Convert list to dataframe for display
            df = pd.DataFrame(data)
            st.dataframe(df)
        elif isinstance(data, dict):
            # Display dictionary as JSON
            st.json(data)
        else:
            st.write(data)
    except Exception as e:
        st.error(f"Error displaying JSON: {str(e)}")
        st.code(json_data)
