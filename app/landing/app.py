import streamlit as st
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from components.navbar import render_navbar
from components.hero_section import render_hero_section

# Set page config
st.set_page_config(
    page_title="RowBlaze - Intelligent RAG",
    page_icon="🗿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def render_landing_page():
    """Renders the full RowBlaze landing page."""

    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
            [data-testid="stHeader"] {
                display: none;
            }
            /* Smooth scrolling for anchor links */
            html {
                scroll-behavior: smooth;
            }
            /* Apply the main app background gradient to the entire page */
            .stApp {
                background: linear-gradient(90deg, rgba(46, 43, 43, 1) 100%, rgba(0, 212, 255, 1) 0%);
                color: #fafafa;
            }
            body {
                font-family: 'Montserrat', 'Roboto', sans-serif;
                margin: 0;
                padding: 0;
            }
        </style>
        <script>
            // Disable the browser's automatic scroll restoration and scroll to top
            if (history.scrollRestoration) {
                history.scrollRestoration = 'manual';
            }
            window.scrollTo(0, 0);
        </script>
    """,
        unsafe_allow_html=True,
    )

    # --- Render the custom navbar ---
    render_navbar()

    # The hero_section component returns HTML, which we render here.
    st.markdown(
        f'<div id="hero-section">{render_hero_section()}</div>', unsafe_allow_html=True
    )

    # These are placeholders. You can create new components for them.
    st.markdown(
        '<div id="about" style="height: 500px; padding: 50px; color: white;"><h2>About RowBlaze</h2><p>Details about the project...</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div id="features" style="height: 500px; padding: 50px; background-color: #111; color: white;"><h2>Features</h2><p>List of features...</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div id="how-it-works" style="height: 500px; padding: 50px; color: white;"><h2>How It Works</h2><p>Explanation of the workflow...</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div id="benefits" style="height: 500px; padding: 50px; background-color: #111; color: white;"><h2>Benefits</h2><p>Key benefits...</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div id="faq" style="height: 500px; padding: 50px; color: white;"><h2>FAQ</h2><p>Frequently asked questions...</p></div>',
        unsafe_allow_html=True,
    )

    # --- Footer ---
    st.markdown(
        """
        <div style="text-align: center; padding: 40px; color: #888;">
            <p>© 2025 RowBlaze. All Rights Reserved.</p>
        </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render_landing_page()
