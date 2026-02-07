import os
import json
import logging
import streamlit as st
from pathlib import Path

# Initialize logger
logger = logging.getLogger(__name__)


class ThemeManager:
    """Manages theme preferences and persistence for the application."""

    def __init__(self):
        self.config_dir = Path.home() / ".creator_catalyst"
        self.config_file = self.config_dir / "theme_config.json"
        self._ensure_config_dir()
        self._init_session_state()

    def _ensure_config_dir(self):
        """Create config directory if it doesn't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _init_session_state(self):
        """Initialize theme in session state."""
        if "theme" not in st.session_state:
            st.session_state.theme = self.load_theme()

    def load_theme(self) -> str:
        """Load theme preference from local file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                    return config.get("theme", "light")
        except Exception as e:
            logger.error(f"Error loading theme config: {e}")
        return "light"

    def save_theme(self, theme: str) -> bool:
        """Save theme preference to local file."""
        try:
            config = {"theme": theme}
            with open(self.config_file, "w") as f:
                json.dump(config, f)
            st.session_state.theme = theme
            return True
        except Exception as e:
            logger.error(f"Error saving theme config: {e}")
            return False

    def get_theme_css(self, theme: str) -> str:
        """Get CSS styling for the specified theme."""
        if theme == "dark":
            return """
            <style>
            :root {
                --primary-color: #1f77b4;
                --background-color: #0e1117;
                --secondary-background-color: #161b22;
                --text-color: #e6edf3;
                --text-secondary: #8b949e;
                --border-color: #30363d;
            }
            
            body, .main {
                color: var(--text-color);
                background-color: var(--background-color);
            }
            
            .stMarkdown, .stText {
                color: var(--text-color);
            }
            
            .stContainer, .stExpander {
                background-color: var(--secondary-background-color);
                border-color: var(--border-color);
            }
            
            /* Streamlit specific dark mode adjustments */
            .stButton > button {
                color: white;
                border-color: var(--border-color);
            }
            
            .stSelectbox, .stTextInput, .stTextArea {
                background-color: var(--secondary-background-color);
                color: var(--text-color);
            }
            
            .stTabs [data-baseweb="tab-list"] {
                background-color: var(--secondary-background-color);
            }
            
            .stMetric {
                background-color: var(--secondary-background-color);
                border: 1px solid var(--border-color);
                border-radius: 0.5rem;
            }
            </style>
            """
        else:  # light theme
            return """
            <style>
            :root {
                --primary-color: #1f77b4;
                --background-color: #ffffff;
                --secondary-background-color: #f6f8fa;
                --text-color: #24292e;
                --text-secondary: #586069;
                --border-color: #d0d7de;
            }
            
            body, .main {
                color: var(--text-color);
                background-color: var(--background-color);
            }
            
            .stMarkdown, .stText {
                color: var(--text-color);
            }
            
            h1, h2, h3, h4, h5, h6 {
                color: var(--text-color) !important;
            }
            
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
            .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
                color: var(--text-color) !important;
            }
            
            [data-testid="stHeader"] {
                color: var(--text-color) !important;
            }
            
            [data-testid="stSubheader"] {
                color: var(--text-color) !important;
            }
            
            .stCaption {
                color: var(--text-secondary) !important;
            }
            
            a {
                color: var(--primary-color) !important;
                text-decoration: underline;
            }
            
            a:visited {
                color: #6f42c1 !important;
            }
            
            .stContainer, .stExpander {
                background-color: var(--secondary-background-color);
                border: 1px solid var(--border-color) !important;
                border-radius: 0.5rem;
            }
            
            /* Streamlit specific light mode adjustments */
            .stButton > button {
                color: var(--text-color) !important;
                background-color: white !important;
                border: 1.5px solid var(--border-color) !important;
            }
            
            .stButton > button:hover {
                background-color: var(--secondary-background-color) !important;
                border-color: var(--primary-color) !important;
            }
            
            .stSelectbox, .stTextInput, .stTextArea {
                background-color: white !important;
                color: var(--text-color) !important;
            }
            
            .stSelectbox [data-baseweb="select"] {
                background-color: white !important;
                border: 1.5px solid var(--border-color) !important;
            }
            
            .stTextInput input, .stTextArea textarea {
                background-color: white !important;
                color: var(--text-color) !important;
                border: 1.5px solid var(--border-color) !important;
            }
            
            .stTabs [data-baseweb="tab-list"] {
                background-color: var(--secondary-background-color);
                border-bottom: 2px solid var(--border-color);
            }
            
            .stTabs [aria-selected="true"] {
                border-bottom: 3px solid var(--primary-color) !important;
            }
            
            .stMetric {
                background-color: white;
                border: 1.5px solid var(--border-color);
                border-radius: 0.5rem;
                padding: 1rem;
            }
            
            .stDivider {
                border-color: var(--border-color) !important;
            }
            
            .stExpander {
                border: 1px solid var(--border-color) !important;
            }
            
            /* Alert and notification styles */
            [data-testid="stAlert"], 
            [data-testid="stInfo"],
            [data-testid="stSuccess"],
            [data-testid="stWarning"],
            [data-testid="stError"] {
                border-width: 2px !important;
                padding: 1rem !important;
                border-radius: 0.5rem !important;
            }
            
            [data-testid="stAlert"] {
                background-color: #fff9e6 !important;
                border-color: #ffc53d !important;
                color: #ad6800 !important;
            }
            
            [data-testid="stAlert"] p,
            [data-testid="stAlert"] span,
            [data-testid="stAlert"] div {
                color: #ad6800 !important;
            }
            
            [data-testid="stInfo"] {
                background-color: #e6f7ff !important;
                border-color: #1890ff !important;
                color: #0050b3 !important;
            }
            
            [data-testid="stInfo"] p,
            [data-testid="stInfo"] span,
            [data-testid="stInfo"] div {
                color: #0050b3 !important;
            }
            
            [data-testid="stSuccess"] {
                background-color: #f6ffed !important;
                border-color: #52c41a !important;
                color: #274e0f !important;
            }
            
            [data-testid="stSuccess"] p,
            [data-testid="stSuccess"] span,
            [data-testid="stSuccess"] div {
                color: #274e0f !important;
            }
            
            [data-testid="stWarning"] {
                background-color: #fffbe6 !important;
                border-color: #faad14 !important;
                color: #ad6800 !important;
            }
            
            [data-testid="stWarning"] p,
            [data-testid="stWarning"] span,
            [data-testid="stWarning"] div {
                color: #ad6800 !important;
            }
            
            [data-testid="stError"] {
                background-color: #fff2f0 !important;
                border-color: #ff4d4f !important;
                color: #820000 !important;
            }
            
            [data-testid="stError"] p,
            [data-testid="stError"] span,
            [data-testid="stError"] div {
                color: #820000 !important;
            }
            
            /* Fallback for any alert containers */
            .element-container div[class*="alert"],
            .element-container div[class*="info"],
            .element-container div[class*="success"],
            .element-container div[class*="warning"],
            .element-container div[class*="error"] {
                border: 2px solid currentColor !important;
                border-radius: 0.5rem !important;
            }
            </style>
            """

    def apply_theme(self):
        """Apply the current theme via CSS injection."""
        current_theme = st.session_state.get("theme", "light")
        css = self.get_theme_css(current_theme)
        st.markdown(css, unsafe_allow_html=True)

    def get_current_theme(self) -> str:
        """Get the current theme from session state."""
        return st.session_state.get("theme", "light")

    def toggle_theme(self):
        """Toggle between light and dark themes."""
        current = self.get_current_theme()
        new_theme = "light" if current == "dark" else "dark"
        self.save_theme(new_theme)
        st.rerun()


def get_theme_manager() -> ThemeManager:
    """Get or create a theme manager instance."""
    if "theme_manager" not in st.session_state:
        st.session_state.theme_manager = ThemeManager()
    return st.session_state.theme_manager
