# components/navbar.py
import streamlit as st
import os
import base64
from pathlib import Path


def render_navbar():
    # --- Use Pathlib for robust path handling ---
    css_path = Path(__file__).parent.parent / "styles" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.error("styles.css not found!")

    # Path to the logo file relative to this script
    logo_path = Path(__file__).parent.parent.parent / "assets" / "cover.png"

    # Navbar HTML with updated links for RowBlaze
    navbar_html = f"""
    <style>
    .navbar {{
        position: relative;
        top: -20px; /* Move navbar up by 20px */
    }}
    </style>
    <div class="navbar">
        <div class="navbar-logo">
            <img src="data:image/png;base64,{get_image_as_base64(logo_path)}" height="130">
        </div>
        <div class="navbar-links">
            <a href="#hero-section" style="font-size: 20px;">Home</a>
            <a href="#about" style="font-size: 20px;">About</a>
            <a href="#features" style="font-size: 20px;">Features</a>
            <a href="#how-it-works" style="font-size: 20px;">How It Works</a>
            <a href="#benefits" style="font-size: 20px;">Benefits</a>
            <a href="#faq" style="font-size: 20px;">FAQ</a>
            <a href="/app" target="_self" style="font-size: 20px;">Login to App</a>
            <a href="https://github.com/negativenagesh/RowBlaze" target="_blank">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="width: 30px; height: 30px; margin-left: 10px;">
            </a>
        </div>
    </div>
    """
    st.markdown(navbar_html, unsafe_allow_html=True)


def get_image_as_base64(path: Path) -> str:
    """Convert an image file to a base64 string"""
    if not path.exists():
        print(f"Warning: Image not found at {path}")
        return ""
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
