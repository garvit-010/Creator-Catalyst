"""
Engagement Scoring UI Components for Creator Catalyst
Displays engagement predictions and recommendations in the Streamlit interface.
"""

import streamlit as st
from typing import Dict, Optional
from engagement_scorer import EngagementScore, get_engagement_scorer


def render_engagement_score_card(score: EngagementScore, show_details: bool = True):
    """
    Render a beautiful engagement score card with all details.
    
    Args:
        score: EngagementScore object with analysis results
        show_details: Whether to show detailed breakdown
    """
    # Main score display
    with st.container(border=True):
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Score visualization
            score_color = _get_score_color(score.overall_score)
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <h1 style="font-size: 4em; margin: 0; color: {score_color};">
                    {score.overall_score}
                </h1>
                <p style="font-size: 1.2em; color: gray; margin: 5px 0;">
                    out of 100
                </p>
                <p style="font-size: 1em; color: {score_color}; font-weight: bold;">
                    {_get_score_label(score.overall_score)}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Platform recommendation
            st.markdown("### 🎯 Best Platform")
            
            # Platform icon and name
            platform_icon = _get_platform_icon(score.recommended_platform)
            st.markdown(f"""
            <div style="padding: 10px; background-color: #f0f2f6; border-radius: 10px; text-align: center;">
                <p style="font-size: 2em; margin: 0;">{platform_icon}</p>
                <p style="font-size: 1.3em; font-weight: bold; margin: 5px 0;">
                    {score.recommended_platform}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Posting time
            if score.optimal_posting_time:
                st.caption(f"⏰ {score.optimal_posting_time}")
    
    # Detailed breakdown
    if show_details:
        st.divider()
        
        # Tabs for different views
        tabs = st.tabs(["💪 Strengths", "📈 Improvements", "📊 Platform Comparison", "🔍 Deep Dive"])
        
        # Strengths tab
        with tabs[0]:
            st.markdown("### What's Working")
            
            if score.strengths:
                for i, strength in enumerate(score.strengths, 1):
                    st.success(f"**{i}.** {strength}", icon="✅")
            else:
                st.info("No major strengths detected. Consider revising your content.")
        
        # Improvements tab
        with tabs[1]:
            st.markdown("### Areas to Improve")
            
            if score.improvements:
                for i, improvement in enumerate(score.improvements, 1):
                    st.warning(f"**{i}.** {improvement}", icon="💡")
            else:
                st.success("Great job! No major improvements needed.", icon="🎉")
        
        # Platform comparison tab
        with tabs[2]:
            st.markdown("### Platform-Specific Scores")
            
            # Sort platforms by score
            sorted_platforms = sorted(
                score.platform_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for platform, platform_score in sorted_platforms:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    icon = _get_platform_icon(platform)
                    is_recommended = platform == score.recommended_platform
                    badge = " 🏆 **Recommended**" if is_recommended else ""
                    
                    st.markdown(f"{icon} **{platform}**{badge}")
                
                with col2:
                    st.metric("", f"{platform_score}/100", label_visibility="collapsed")
                
                # Progress bar
                st.progress(platform_score / 100)
                st.markdown("")  # Spacing
        
        # Deep dive tab
        with tabs[3]:
            st.markdown("### Engagement Factors Analysis")
            
            # Display all factors with visual indicators
            factors_sorted = sorted(
                score.engagement_factors.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for factor, value in factors_sorted:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    factor_display = factor.replace('_', ' ').title()
                    st.markdown(f"**{factor_display}**")
                
                with col2:
                    percentage = int(value * 100)
                    st.metric("", f"{percentage}%", label_visibility="collapsed")
                
                with col3:
                    if value >= 0.8:
                        st.success("Excellent", icon="🔥")
                    elif value >= 0.6:
                        st.info("Good", icon="👍")
                    elif value >= 0.4:
                        st.warning("Fair", icon="⚠️")
                    else:
                        st.error("Needs Work", icon="❌")
                
                st.progress(value)
                st.markdown("")  # Spacing


def render_compact_score(score: EngagementScore):
    """
    Render a compact version of the engagement score.
    Useful for inline display in content lists.
    """
    score_color = _get_score_color(score.overall_score)
    platform_icon = _get_platform_icon(score.recommended_platform)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center;">
            <h2 style="color: {score_color}; margin: 0;">{score.overall_score}/100</h2>
            <p style="color: gray; font-size: 0.8em; margin: 0;">Engagement Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"**Recommended:** {platform_icon} {score.recommended_platform}")
        
        if score.strengths:
            st.caption(f"✅ {score.strengths[0]}")


def analyze_and_display_score(
    content: str, 
    content_type: str = "social_post",
    target_platform: Optional[str] = None,
    show_compact: bool = False
):
    """
    Analyze content and display engagement score.
    
    Args:
        content: Content to analyze
        content_type: Type of content
        target_platform: Optional target platform
        show_compact: Whether to show compact view
    """
    if not content or len(content.strip()) < 10:
        st.warning("Content too short to analyze. Add more text for accurate scoring.")
        return None
    
    # Get scorer and analyze
    scorer = get_engagement_scorer()
    
    with st.spinner("🔍 Analyzing engagement potential..."):
        score = scorer.score_content(content, content_type, target_platform)
    
    # Display results
    if show_compact:
        render_compact_score(score)
    else:
        render_engagement_score_card(score, show_details=True)
    
    return score


def render_score_comparison(scores: Dict[str, EngagementScore]):
    """
    Compare multiple scores side by side.
    
    Args:
        scores: Dictionary mapping content names to scores
    """
    st.markdown("### 📊 Score Comparison")
    
    cols = st.columns(len(scores))
    
    for (name, score), col in zip(scores.items(), cols):
        with col:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                
                score_color = _get_score_color(score.overall_score)
                st.markdown(f"""
                <div style="text-align: center;">
                    <h2 style="color: {score_color};">{score.overall_score}</h2>
                    <p style="font-size: 0.9em; color: gray;">
                        {_get_platform_icon(score.recommended_platform)} 
                        {score.recommended_platform}
                    </p>
                </div>
                """, unsafe_allow_html=True)


def _get_score_color(score: int) -> str:
    """Get color based on score value."""
    if score >= 80:
        return "#10b981"  # Green
    elif score >= 60:
        return "#3b82f6"  # Blue
    elif score >= 40:
        return "#f59e0b"  # Orange
    else:
        return "#ef4444"  # Red


def _get_score_label(score: int) -> str:
    """Get label based on score value."""
    if score >= 85:
        return "🔥 Viral Potential"
    elif score >= 70:
        return "🚀 High Engagement"
    elif score >= 55:
        return "👍 Good Performance"
    elif score >= 40:
        return "⚠️ Moderate Engagement"
    else:
        return "❌ Needs Improvement"


def _get_platform_icon(platform: str) -> str:
    """Get emoji icon for platform."""
    icons = {
        'Instagram': '📸',
        'Twitter/X': '🐦',
        'LinkedIn': '💼',
        'TikTok': '🎵',
        'YouTube': '📺',
        'General': '🌐'
    }
    return icons.get(platform, '📱')


def add_engagement_scoring_section(content: str, platform: str = "General"):
    """
    Add engagement scoring section to any page.
    
    Args:
        content: Content to score
        platform: Target platform
    """
    with st.expander("📊 Engagement Score Analysis", expanded=False):
        st.markdown("""
        Get AI-powered predictions for how well this content will perform.
        Our algorithm analyzes multiple factors to predict engagement potential.
        """)
        
        if st.button("🔍 Analyze Engagement", key=f"analyze_{hash(content)}"):
            analyze_and_display_score(content, "social_post", platform)