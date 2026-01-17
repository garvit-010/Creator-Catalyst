"""
Keyword UI Components for Creator Catalyst
Displays extracted keywords in the Streamlit interface.
"""

import streamlit as st
from typing import List, Dict
from src.core.keyword_extractor import get_keyword_extractor


def render_keywords_badge(keywords: List[str]):
    """Render keywords as colorful badges."""
    if not keywords:
        st.info("No keywords extracted.")
        return

    # Color palette for badges
    colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#FFA07A",
        "#98D8C8",
        "#F7B731",
        "#5F27CD",
        "#00D2D3",
    ]

    keyword_html = ""
    for i, keyword in enumerate(keywords):
        color = colors[i % len(colors)]
        keyword_html += f"""
        <span style="
            display: inline-block;
            background-color: {color};
            color: white;
            padding: 6px 12px;
            border-radius: 16px;
            margin: 4px 4px;
            font-weight: 500;
            font-size: 0.85em;
        ">{keyword}</span>
        """

    st.markdown(keyword_html, unsafe_allow_html=True)


def display_keywords_section(
    keywords: List[str], title: str = "🔍 Keywords for SEO", show_copy: bool = True
):
    """Display a complete keywords section with options."""
    if not keywords:
        return

    with st.container(border=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"### {title}")
            st.caption(f"**{len(keywords)}** keywords")

        with col2:
            if show_copy:
                keywords_str = ", ".join(keywords)
                st.download_button(
                    label="📋",
                    data=keywords_str,
                    file_name="keywords.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        st.markdown("---")
        render_keywords_badge(keywords)

        st.caption(
            "💡 **Tip:** Use these keywords in titles, meta descriptions, and tags for better SEO"
        )


def extract_and_show_keywords(
    content: str,
    content_type: str = "general",
    title: str = "🔍 Keywords for SEO",
    num_keywords: int = 8,
) -> List[str]:
    """
    Extract keywords and display them immediately.

    Args:
        content: Text to extract from
        content_type: Type of content
        title: Section title
        num_keywords: Number of keywords

    Returns:
        List of extracted keywords
    """
    if not content or len(content.strip()) < 20:
        st.warning("Content too short to extract keywords.")
        return []

    extractor = get_keyword_extractor()
    keywords = extractor.extract_keywords(content, num_keywords, content_type)

    display_keywords_section(keywords, title)
    return keywords


def show_keywords_grid(keywords_dict: Dict[str, List[str]]):
    """Display keywords from multiple contents in columns."""
    if not keywords_dict:
        return

    cols = st.columns(len(keywords_dict))

    labels = {"blog": "📝 Blog", "social": "📱 Social", "shorts": "⚡ Shorts"}

    for col_idx, (content_type, keywords) in enumerate(keywords_dict.items()):
        with cols[col_idx]:
            label = labels.get(content_type, content_type.title())
            st.subheader(label)
            if keywords:
                st.markdown(", ".join([f"`{kw}`" for kw in keywords]))
            else:
                st.caption("—")
