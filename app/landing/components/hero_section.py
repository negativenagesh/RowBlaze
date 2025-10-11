import streamlit as st


def render_hero_section():
    """Renders the hero section HTML for the RowBlaze landing page."""

    # Hero section HTML with updated content and a direct link to the app
    hero_html = """
    <div id="hero-section" class="hero-container">
        <div class="hero-text">
            <h2>Intelligent RAG for Enterprise Data</h2>
            <p>A production-grade, open-source RAG platform designed for speed, structured data enrichment, and full observability.</p>
            <div class="hero-buttons">
                <a href="#features" class="hero-button explore-btn">Explore Features</a>
                <a href="#how-it-works" class="hero-button how-it-works-btn">How It Works</a>
                <a href="/app" target="_self" class="hero-button explore-btn" style="margin-top: 10px;">Launch App</a>
            </div>
            <div class="hero-tags">
                <span class="hero-tag">Async Pipelines</span>
                <span class="hero-tag">Open Source</span>
                <span class="hero-tag">Multi-Model</span>
            </div>
        </div>
        <div class="resume-preview">
            <div class="resume-header">
                <div class="resume-logo">R</div>
                <div class="resume-score">92% Match</div>
            </div>
            <div class="resume-content">
                <div class="resume-bar" style="width: 90%;"></div>
                <div class="resume-placeholder" style="width: 70%;"></div>
                <div class="resume-placeholder" style="width: 80%;"></div>
                <div class="resume-bar" style="width: 60%;"></div>
                <div class="resume-placeholder" style="width: 75%;"></div>
            </div>
        </div>
    </div>
    """

    return hero_html
