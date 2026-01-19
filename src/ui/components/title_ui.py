"""
Title Generation UI Components for Creator Catalyst
Displays AI-generated title suggestions with selection and editing capabilities.
"""

import streamlit as st
from typing import List, Dict, Optional, Callable
from src.core.title_generator import (
    TitleSuggestion, 
    TitleGenerationResult, 
    get_title_generator
)


def render_title_suggestions(
    titles: List[TitleSuggestion],
    title_type: str = "video",
    item_key: str = "main",
    on_select: Optional[Callable[[str], None]] = None,
    show_metadata: bool = True
) -> Optional[str]:
    """
    Render a list of title suggestions with selection and editing.
    
    Args:
        titles: List of TitleSuggestion objects
        title_type: "video" or "short" - affects styling
        item_key: Unique key for session state management
        on_select: Callback when a title is selected
        show_metadata: Whether to show CTR estimates and style info
        
    Returns:
        Selected or edited title string, or None if nothing selected
    """
    if not titles:
        st.info("No title suggestions available.")
        return None
    
    # Session state keys
    selected_key = f"selected_title_{item_key}"
    custom_key = f"custom_title_{item_key}"
    editing_key = f"editing_title_{item_key}"
    
    # Initialize session state
    if selected_key not in st.session_state:
        st.session_state[selected_key] = None
    if custom_key not in st.session_state:
        st.session_state[custom_key] = ""
    if editing_key not in st.session_state:
        st.session_state[editing_key] = False
    
    st.markdown("#### 🎬 Suggested Titles")
    st.caption("Click a title to select it, or write your own below")
    
    # Display each title as a selectable card
    for i, suggestion in enumerate(titles):
        is_selected = st.session_state[selected_key] == suggestion.title
        
        with st.container(border=True):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                # Title display with selection highlighting
                if is_selected:
                    st.markdown(f"✅ **{suggestion.title}**")
                else:
                    st.markdown(f"**{suggestion.title}**")
                
                # Metadata badges
                if show_metadata:
                    badge_col1, badge_col2, badge_col3 = st.columns(3)
                    
                    with badge_col1:
                        style_emoji = _get_style_emoji(suggestion.style)
                        st.caption(f"{style_emoji} {suggestion.style.replace('_', ' ').title()}")
                    
                    with badge_col2:
                        hook_emoji = _get_hook_emoji(suggestion.hook_type)
                        st.caption(f"{hook_emoji} {suggestion.hook_type.replace('_', ' ').title()}")
                    
                    with badge_col3:
                        ctr_display = _get_ctr_display(suggestion.estimated_ctr)
                        st.caption(ctr_display)
            
            with col2:
                if st.button(
                    "✓" if is_selected else "Select",
                    key=f"select_title_{item_key}_{i}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary"
                ):
                    st.session_state[selected_key] = suggestion.title
                    st.session_state[editing_key] = False
                    if on_select:
                        on_select(suggestion.title)
                    st.rerun()
    
    # Custom title option
    st.markdown("---")
    st.markdown("#### ✏️ Or Write Your Own")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        custom_title = st.text_input(
            "Custom title",
            value=st.session_state[custom_key],
            placeholder="Enter your own catchy title...",
            key=f"custom_input_{item_key}",
            label_visibility="collapsed"
        )
    
    with col2:
        if st.button("Use This", key=f"use_custom_{item_key}", use_container_width=True):
            if custom_title.strip():
                st.session_state[selected_key] = custom_title.strip()
                st.session_state[custom_key] = custom_title.strip()
                if on_select:
                    on_select(custom_title.strip())
                st.rerun()
    
    # Display currently selected title
    if st.session_state[selected_key]:
        st.success(f"📌 **Selected:** {st.session_state[selected_key]}")
        return st.session_state[selected_key]
    
    return None


def render_video_titles_section(
    video_summary: str,
    platform: str = "YouTube",
    llm_wrapper = None
) -> Optional[str]:
    """
    Render the complete title suggestions section for the main video.
    
    Args:
        video_summary: Summary text to generate titles from
        platform: Target platform
        llm_wrapper: Optional LLM wrapper for AI-powered generation
        
    Returns:
        Selected title or None
    """
    st.markdown("### 🎬 Video Title Suggestions")
    st.caption(f"AI-generated catchy titles optimized for {platform}")
    
    # Session state for titles
    titles_key = "video_title_suggestions"
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if titles_key in st.session_state and st.session_state[titles_key]:
            st.caption(f"✅ {len(st.session_state[titles_key])} titles generated")
    
    with col2:
        generate_btn = st.button(
            "🔄 Generate Titles",
            key="generate_video_titles",
            use_container_width=True
        )
    
    if generate_btn:
        with st.spinner("✨ Generating catchy titles..."):
            generator = get_title_generator(llm_wrapper)
            titles = generator.generate_titles_for_video(
                video_summary=video_summary,
                platform=platform,
                num_titles=3
            )
            st.session_state[titles_key] = titles
    
    # Display titles if available
    if titles_key in st.session_state and st.session_state[titles_key]:
        return render_title_suggestions(
            titles=st.session_state[titles_key],
            title_type="video",
            item_key="main_video",
            show_metadata=True
        )
    
    return None


def render_short_titles_section(
    short_index: int,
    short_topic: str,
    short_summary: str,
    platform: str = "YouTube",
    llm_wrapper = None
) -> Optional[str]:
    """
    Render title suggestions for a specific short.
    
    Args:
        short_index: Index of the short (0-based)
        short_topic: Topic/title of the short
        short_summary: Summary of the short content
        platform: Target platform
        llm_wrapper: Optional LLM wrapper
        
    Returns:
        Selected title or None
    """
    # Session state key for this short's titles
    titles_key = f"short_{short_index}_titles"
    
    with st.expander(f"🎬 Title Suggestions for Short #{short_index + 1}", expanded=False):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if titles_key in st.session_state and st.session_state[titles_key]:
                st.caption(f"✅ {len(st.session_state[titles_key])} titles available")
            else:
                st.caption("Click generate to get AI-powered title suggestions")
        
        with col2:
            if st.button(
                "✨ Generate",
                key=f"gen_short_titles_{short_index}",
                use_container_width=True
            ):
                with st.spinner("Generating..."):
                    generator = get_title_generator(llm_wrapper)
                    titles = generator.generate_titles_for_short(
                        short_summary=short_summary,
                        short_topic=short_topic,
                        platform=platform,
                        num_titles=3
                    )
                    st.session_state[titles_key] = titles
        
        if titles_key in st.session_state and st.session_state[titles_key]:
            return render_title_suggestions(
                titles=st.session_state[titles_key],
                title_type="short",
                item_key=f"short_{short_index}",
                show_metadata=False  # Compact view for shorts
            )
    
    return None


def render_all_titles_dashboard(
    video_summary: str,
    shorts_ideas: List[Dict],
    platform: str = "YouTube",
    llm_wrapper = None
) -> Dict:
    """
    Render a complete dashboard for all title suggestions.
    
    Args:
        video_summary: Main video summary
        shorts_ideas: List of short idea dictionaries
        platform: Target platform
        llm_wrapper: LLM wrapper instance
        
    Returns:
        Dictionary with selected titles for video and each short
    """
    st.markdown("## 🎬 Title Generator")
    st.markdown("Generate catchy, click-worthy titles for your video and shorts")
    
    selected_titles = {
        'video': None,
        'shorts': {}
    }
    
    # Generate All button
    if st.button("🚀 Generate All Titles", type="primary", use_container_width=True):
        with st.spinner("✨ Generating titles for video and all shorts..."):
            generator = get_title_generator(llm_wrapper)
            result = generator.generate_all_titles(
                video_summary=video_summary,
                shorts_ideas=shorts_ideas,
                platform=platform
            )
            
            # Store in session state
            st.session_state['video_title_suggestions'] = result.original_titles
            for idx, titles in result.shorts_titles.items():
                st.session_state[f'short_{idx}_titles'] = titles
            
            st.success("✅ All titles generated!")
    
    st.divider()
    
    # Main Video Section
    st.markdown("### 📹 Main Video Title")
    video_title = render_video_titles_section(
        video_summary=video_summary,
        platform=platform,
        llm_wrapper=llm_wrapper
    )
    selected_titles['video'] = video_title
    
    st.divider()
    
    # Shorts Section
    if shorts_ideas:
        st.markdown("### ✂️ Shorts Titles")
        st.caption(f"Generate individual titles for {len(shorts_ideas)} shorts")
        
        for i, short in enumerate(shorts_ideas):
            topic = short.get('topic', f'Short {i+1}')
            summary = short.get('summary', '')
            
            short_title = render_short_titles_section(
                short_index=i,
                short_topic=topic,
                short_summary=summary,
                platform=platform,
                llm_wrapper=llm_wrapper
            )
            selected_titles['shorts'][i] = short_title
    
    return selected_titles


def _get_style_emoji(style: str) -> str:
    """Get emoji for title style."""
    emojis = {
        'curiosity': '🤔',
        'listicle': '📋',
        'how_to': '📖',
        'secret': '🤫',
        'urgency': '⚡',
        'results': '📈'
    }
    return emojis.get(style.lower(), '💡')


def _get_hook_emoji(hook_type: str) -> str:
    """Get emoji for hook type."""
    emojis = {
        'question': '❓',
        'number': '🔢',
        'power_word': '💪',
        'mystery': '🔮'
    }
    return emojis.get(hook_type.lower(), '🎯')


def _get_ctr_display(ctr: str) -> str:
    """Get display string for CTR estimate."""
    displays = {
        'high': '🔥 High CTR',
        'medium': '📊 Medium CTR',
        'low': '📉 Low CTR'
    }
    return displays.get(ctr.lower(), '📊 CTR Unknown')


def get_title_tips(platform: str = "YouTube") -> str:
    """Get platform-specific title tips."""
    tips = {
        "YouTube": """
**YouTube Title Tips:**
- Keep it under 60 characters
- Front-load keywords
- Use numbers when possible
- Add brackets for context [2024]
- Create curiosity gaps
        """,
        "TikTok": """
**TikTok Title Tips:**
- Be casual and trendy
- Use trending phrases
- Hook viewers in first 3 words
- Emojis work great
- Keep it relatable
        """,
        "Instagram": """
**Instagram Title Tips:**
- Make it aesthetic
- Use emojis strategically
- Keep it aspirational
- Focus on visual appeal
        """
    }
    return tips.get(platform, tips["YouTube"])
