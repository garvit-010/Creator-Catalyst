"""
History browsing page for Creator Catalyst.
Allows users to view, search, and manage past video analyses.
"""

import streamlit as st
from datetime import datetime
import json

from src.database.storage_manager import get_storage_manager
from src.database.database import Database
from src.database.report_generator import get_report_generator
from src.core.strategy_advisor import get_strategy_advisor


def render_history_page():
    """Main history browsing page."""
    st.title("📚 Content History")
    st.markdown("Browse, search, and manage all your processed videos and generated content.")
    
    # Initialize storage manager
    storage = get_storage_manager()
    
    # Get statistics
    stats = storage.get_statistics()
    
    # Display stats at top
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Videos", stats['total_videos'])
    
    with col2:
        st.metric("Content Pieces", stats['total_contents'])
    
    with col3:
        avg_grounding = stats['average_grounding_rates']
        overall_avg = (avg_grounding['blog'] + avg_grounding['social'] + avg_grounding['shorts']) / 3
        st.metric("Avg Grounding", f"{overall_avg:.0%}")
    
    with col4:
        content_by_type = stats['content_by_type']
        st.metric("Blog Posts", content_by_type.get('blog_post', 0))
    
    st.divider()
    
    # Search and filter section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "🔍 Search videos by filename",
            placeholder="Enter filename to search...",
            key="video_search"
        )
    
    with col2:
        view_mode = st.selectbox(
            "View Mode",
            ["Grid View", "List View", "Recent Activity"],
            key="view_mode"
        )
    
    st.divider()
    
    # Display content based on view mode
    if view_mode == "Recent Activity":
        render_recent_activity(storage)
    elif search_query:
        render_search_results(storage, search_query)
    else:
        if view_mode == "Grid View":
            render_grid_view(storage)
        else:
            render_list_view(storage)


def render_recent_activity(storage):
    """Display recent content generation activity."""
    st.subheader("🕐 Recent Activity")
    
    activities = storage.get_recent_activity(limit=20)
    
    if not activities:
        st.info("No recent activity found.")
        return
    
    for activity in activities:
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.markdown(f"**{activity['filename']}**")
                st.caption(f"Video ID: {activity['video_id']}")
            
            with col2:
                content_type_display = activity['content_type'].replace('_', ' ').title()
                st.markdown(f"📄 {content_type_display}")
            
            with col3:
                st.caption(f"Platform: {activity['platform']}")
                created = datetime.fromisoformat(activity['created_at'])
                st.caption(f"Created: {created.strftime('%Y-%m-%d %H:%M')}")
            
            with col4:
                if st.button("👁️ View", key=f"view_activity_{activity['content_id']}"):
                    st.session_state.selected_video_id = activity['video_id']
                    st.rerun()


def render_search_results(storage, query):
    """Display search results."""
    st.subheader(f"🔍 Search Results for '{query}'")
    
    videos = storage.db.search_videos(query)
    
    if not videos:
        st.warning(f"No videos found matching '{query}'")
        return
    
    st.info(f"Found {len(videos)} video(s)")
    
    for video in videos:
        render_video_card(storage, video, expanded=True)


def render_grid_view(storage):
    """Display videos in grid layout."""
    st.subheader("📊 All Videos")
    
    videos = storage.get_all_videos_summary(limit=100)
    
    if not videos:
        st.info("No videos processed yet. Upload a video in the Creator Tool to get started!")
        return
    
    # Display in rows of 3
    for i in range(0, len(videos), 3):
        cols = st.columns(3)
        
        for j, col in enumerate(cols):
            if i + j < len(videos):
                with col:
                    render_video_card_compact(storage, videos[i + j])


def render_list_view(storage):
    """Display videos in detailed list."""
    st.subheader("📋 All Videos (Detailed)")
    
    videos = storage.get_all_videos_summary(limit=100)
    
    if not videos:
        st.info("No videos processed yet. Upload a video in the Creator Tool to get started!")
        return
    
    for video in videos:
        render_video_card(storage, video, expanded=False)


def render_video_card_compact(storage, video_dict):
    """Render compact video card for grid view."""
    with st.container(border=True):
        st.markdown(f"### 🎥 {video_dict['filename'][:25]}...")
        
        # Status badge
        status = video_dict['processing_status']
        if status == 'completed':
            st.success("✅ Completed", icon="✅")
        elif status == 'processing':
            st.warning("⏳ Processing", icon="⏳")
        else:
            st.error("❌ Failed", icon="❌")
        
        # Stats
        st.metric("Content Pieces", video_dict['total_content'])
        st.caption(f"Platform: {video_dict['platform']}")
        st.caption(f"Size: {video_dict['file_size_mb']:.1f} MB")
        
        uploaded = datetime.fromisoformat(video_dict['uploaded_at'])
        st.caption(f"📅 {uploaded.strftime('%Y-%m-%d')}")
        
        # Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👁️ View", key=f"view_compact_{video_dict['id']}", use_container_width=True):
                st.session_state.selected_video_id = video_dict['id']
                st.rerun()
        
        with col2:
            if st.button("🗑️", key=f"del_compact_{video_dict['id']}", use_container_width=True):
                if st.session_state.get(f"confirm_delete_{video_dict['id']}"):
                    storage.delete_video_and_content(video_dict['id'])
                    st.success("Deleted!")
                    st.rerun()
                else:
                    st.session_state[f"confirm_delete_{video_dict['id']}"] = True
                    st.warning("Click again to confirm")


def render_video_card(storage, video_dict, expanded=False):
    """Render detailed video card for list view."""
    with st.expander(f"🎥 {video_dict['filename']}", expanded=expanded):
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.markdown(f"**Video ID:** {video_dict['id']}")
            st.markdown(f"**Platform:** {video_dict['platform']}")
            st.markdown(f"**File Size:** {video_dict['file_size_mb']:.2f} MB")
            
            uploaded = datetime.fromisoformat(video_dict['uploaded_at'])
            st.markdown(f"**Uploaded:** {uploaded.strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col2:
            st.markdown(f"**Status:** {video_dict['processing_status']}")
            st.markdown(f"**Grounding:** {'✅ Enabled' if video_dict['grounding_enabled'] else '❌ Disabled'}")
            
            # Content counts
            st.markdown("**Content Generated:**")
            content_counts = video_dict.get('content_counts', {})
            for content_type, count in content_counts.items():
                display_type = content_type.replace('_', ' ').title()
                st.markdown(f"- {display_type}: {count}")
        
        with col3:
            if st.button("👁️ View Details", key=f"view_detail_{video_dict['id']}", use_container_width=True):
                st.session_state.selected_video_id = video_dict['id']
                st.rerun()
            
            if st.button("� Download Toolkit", key=f"toolkit_{video_dict['id']}", use_container_width=True):
                try:
                    zip_path = storage.export_video_toolkit_zip(video_dict['id'])
                    with open(zip_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download ZIP",
                            data=f.read(),
                            file_name=f"toolkit_video_{video_dict['id']}.zip",
                            mime="application/zip",
                            key=f"dl_toolkit_{video_dict['id']}",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
            
            if st.button("�📥 Export JSON", key=f"export_{video_dict['id']}", use_container_width=True):
                export_path = f"export_video_{video_dict['id']}.json"
                storage.export_video_results(video_dict['id'], export_path)
                st.success(f"Exported to {export_path}")
            
            if st.button("🗑️ Delete", key=f"delete_{video_dict['id']}", use_container_width=True):
                if st.session_state.get(f"confirm_delete_detail_{video_dict['id']}"):
                    storage.delete_video_and_content(video_dict['id'])
                    st.success("Video and all content deleted!")
                    st.rerun()
                else:
                    st.session_state[f"confirm_delete_detail_{video_dict['id']}"] = True
                    st.warning("⚠️ Click again to confirm deletion")


def render_video_details(storage, video_id):
    """Render detailed view of a single video's content."""
    st.title("📄 Video Details")
    
    if st.button("← Back to History"):
        del st.session_state.selected_video_id
        st.rerun()
    
    st.divider()
    
    # Load results
    results = storage.load_video_results(video_id)
    
    if not results:
        st.error("Video not found!")
        return
    
    # Video info
    video = results['video']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Video ID", video['id'])
    
    with col2:
        st.metric("Platform", video['platform'])
    
    with col3:
        st.metric("File Size", f"{video['file_size_mb']:.1f} MB")
    
    with col4:
        uploaded = datetime.fromisoformat(video['uploaded_at'])
        st.metric("Uploaded", uploaded.strftime('%Y-%m-%d'))
    
    st.markdown(f"**Filename:** `{video['filename']}`")
    st.markdown(f"**Status:** {video['processing_status']}")
    st.markdown(f"**Grounding:** {'✅ Enabled' if video['grounding_enabled'] else '❌ Disabled'}")
    
    st.divider()
    
    # Download Toolkit Button - Prominent placement
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        # Campaign Report Generator
        try:
            report_gen = get_report_generator()
            pdf_path = report_gen.generate_report(video_id)
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()
                st.download_button(
                    label="📄 Campaign Summary (PDF)",
                    data=pdf_data,
                    file_name=f"Campaign_Summary_{video['filename'].rsplit('.', 1)[0]}.pdf",
                    mime="application/pdf",
                    key=f"report_download_{video_id}",
                    use_container_width=True,
                    help="Download a professional PDF summary of this video campaign"
                )
        except Exception as e:
            st.error(f"Failed to generate report: {str(e)}")

    with col2:
        # Create ZIP and provide download button
        try:
            zip_path = storage.export_video_toolkit_zip(video_id)
            with open(zip_path, 'rb') as f:
                zip_data = f.read()
                st.download_button(
                    label="📦 Download Toolkit (.zip)",
                    data=zip_data,
                    file_name=f"toolkit_{video['filename'].rsplit('.', 1)[0]}.zip",
                    mime="application/zip",
                    key=f"toolkit_download_{video_id}",
                    use_container_width=True,
                    help="Download all generated assets (captions, blog, social, shorts, thumbnails)"
                )
        except Exception as e:
            st.error(f"Failed to create toolkit: {str(e)}")
    
    st.divider()
    
    # Grounding Report
    if results.get('grounding_report'):
        with st.expander("📊 Grounding Report", expanded=False):
            report = results['grounding_report']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Blog Grounding", f"{report['blog_grounding_rate']:.0%}")
            
            with col2:
                st.metric("Social Grounding", f"{report['social_grounding_rate']:.0%}")
            
            with col3:
                st.metric("Shorts Verification", f"{report['shorts_verification_rate']:.0%}")
            
            st.markdown(f"**Total Claims:** {report['total_claims']}")
            st.markdown(f"**Verified:** {report['verified_claims']} ✅")
            st.markdown(f"**Unverified:** {report['unverified_claims']} ❌")
    
    # AI Strategy Tips
    with st.expander("💡 AI Strategy Tips & Next Steps", expanded=True):
        try:
            advisor = get_strategy_advisor()
            with st.spinner("Generating strategy tips..."):
                tips = advisor.generate_next_steps(video_id)
                st.markdown(tips)
        except Exception as e:
            st.info("Strategy tips will be available once content is fully analyzed.")
    
    # Content Tabs
    tabs = st.tabs(["📝 Captions", "📰 Blog Post", "📱 Social Post", "✂️ Shorts Ideas", "🎨 Thumbnails"])
    
    with tabs[0]:
        st.subheader("Video Captions (SRT)")
        if results.get('captions'):
            st.text_area("Transcript", results['captions'], height=400, key="captions_view")
            st.download_button(
                "📥 Download SRT",
                data=results['captions'],
                file_name=f"captions_video_{video_id}.srt",
                mime="text/plain"
            )
        else:
            st.info("No captions available")
    
    with tabs[1]:
        st.subheader("Blog Post")
        
        if results.get('blog_post'):
            # Show comparison if original exists
            if results.get('blog_post_original'):
                view_tab1, view_tab2 = st.tabs(["✅ Grounded Version", "⚠️ Original (Unfiltered)"])
                
                with view_tab1:
                    st.markdown(results['blog_post'])
                
                with view_tab2:
                    st.warning("This version may contain unverified claims")
                    st.markdown(results['blog_post_original'])
            else:
                st.markdown(results['blog_post'])
            
            st.download_button(
                "📥 Download Markdown",
                data=results['blog_post'],
                file_name=f"blog_post_video_{video_id}.md",
                mime="text/markdown"
            )
        else:
            st.info("No blog post available")
    
    with tabs[2]:
        st.subheader("Social Media Post")
        
        if results.get('social_post'):
            st.markdown(f"> {results['social_post']}")
            
            st.download_button(
                "📥 Copy to Clipboard",
                data=results['social_post'],
                file_name=f"social_post_video_{video_id}.txt",
                mime="text/plain"
            )
        else:
            st.info("No social post available")
    
    with tabs[3]:
        st.subheader("Short Clip Ideas")
        
        shorts = results.get('shorts_ideas', [])
        
        if shorts:
            for i, short in enumerate(shorts):
                with st.container(border=True):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"### {i+1}. {short.get('topic', 'N/A')}")
                    
                    with col2:
                        if short.get('validation_badge'):
                            st.markdown(f"**{short['validation_badge']}**")
                    
                    st.markdown(f"**Timestamps:** `{short.get('start_time', 'N/A')} - {short.get('end_time', 'N/A')}`")
                    st.markdown(f"**Summary:** {short.get('summary', 'N/A')}")
                    
                    if short.get('supporting_text'):
                        with st.expander("📝 Transcript Evidence"):
                            st.caption(short['supporting_text'])
        else:
            st.info("No shorts ideas available")
    
    with tabs[4]:
        st.subheader("Thumbnail Ideas")
        
        thumbnails = results.get('thumbnail_ideas', [])
        
        if thumbnails:
            for i, thumb in enumerate(thumbnails):
                with st.container(border=True):
                    st.markdown(f"### Idea {i+1}")
                    
                    idea_text = thumb['idea'] if isinstance(thumb, dict) else thumb
                    st.markdown(f"*{idea_text}*")
        else:
            st.info("No thumbnail ideas available")


def main():
    """Main entry point for history page."""
    # Check if viewing specific video
    if 'selected_video_id' in st.session_state:
        storage = get_storage_manager()
        render_video_details(storage, st.session_state.selected_video_id)
    else:
        render_history_page()


if __name__ == "__main__":
    main()