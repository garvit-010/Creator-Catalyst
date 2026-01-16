"""
AI Request Logs Dashboard for Creator Catalyst
Admin interface for viewing AI usage logs, analytics, and rate limits.
"""

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from ai_request_logger import get_ai_logger


def render_ai_logs_dashboard():
    """Main AI logs dashboard page."""
    st.title("🔍 AI Request Logs & Analytics")
    st.markdown("Monitor AI usage, costs, and performance metrics")
    
    # Initialize logger
    logger = get_ai_logger()
    
    # User selector (for multi-user support in future)
    user_id = st.sidebar.text_input("User ID", value="default_user")
    
    # Time range selector
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"],
        index=2
    )
    
    # Calculate date range
    if time_range == "Last Hour":
        days = 1
        start_date = datetime.now() - timedelta(hours=1)
    elif time_range == "Last 24 Hours":
        days = 1
        start_date = datetime.now() - timedelta(days=1)
    elif time_range == "Last 7 Days":
        days = 7
        start_date = datetime.now() - timedelta(days=7)
    elif time_range == "Last 30 Days":
        days = 30
        start_date = datetime.now() - timedelta(days=30)
    else:
        days = 365
        start_date = datetime.now() - timedelta(days=365)
    
    st.divider()
    
    # Get analytics
    analytics = logger.get_usage_analytics(user_id=user_id, days=days)
    
    # ========== KEY METRICS ==========
    st.markdown(f"### 📊 Overview ({time_range})")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Requests",
            f"{analytics['total_requests']:,}",
            help="Total AI API calls made"
        )
    
    with col2:
        st.metric(
            "Total Tokens",
            f"{analytics['total_tokens']:,}",
            help="Total tokens consumed across all requests"
        )
    
    with col3:
        st.metric(
            "Credits Spent",
            f"{analytics['total_credits']:.1f}",
            help="Total application credits spent"
        )
    
    with col4:
        st.metric(
            "Est. USD Cost",
            f"${analytics['total_usd']:.2f}",
            help="Estimated cost in USD"
        )
    
    with col5:
        success_rate = analytics['success_rate']
        delta_color = "normal" if success_rate >= 95 else "inverse"
        st.metric(
            "Success Rate",
            f"{success_rate:.1f}%",
            delta=f"{analytics['failed_requests']} failed",
            delta_color=delta_color,
            help="Percentage of successful requests"
        )
    
    st.divider()
    
    # ========== RATE LIMIT STATUS ==========
    st.markdown("### ⏱️ Current Rate Limit Status")
    
    is_allowed, rate_stats = logger.check_rate_limit(
        user_id=user_id,
        max_requests_per_hour=100,
        max_tokens_per_hour=1_000_000
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        requests_pct = (rate_stats['requests_used'] / 100) * 100
        st.metric(
            "Requests This Hour",
            f"{rate_stats['requests_used']}/100",
            delta=f"{rate_stats['requests_remaining']} remaining"
        )
        st.progress(requests_pct / 100)
    
    with col2:
        tokens_pct = (rate_stats['tokens_used'] / 1_000_000) * 100
        st.metric(
            "Tokens This Hour",
            f"{rate_stats['tokens_used']:,}/1M",
            delta=f"{rate_stats['tokens_remaining']:,} remaining"
        )
        st.progress(tokens_pct / 100)
    
    with col3:
        st.metric(
            "Credits This Hour",
            f"{rate_stats['credits_spent']:.1f}"
        )
        
        if is_allowed:
            st.success("✅ Within Limits", icon="✅")
        else:
            st.error("⚠️ Rate Limit Exceeded", icon="⚠️")
    
    if rate_stats.get('window_end'):
        window_end = datetime.fromisoformat(rate_stats['window_end'])
        time_until_reset = window_end - datetime.now()
        minutes_left = int(time_until_reset.total_seconds() / 60)
        st.caption(f"🕐 Window resets in {minutes_left} minutes")
    
    st.divider()
    
    # ========== TABS FOR DETAILED VIEWS ==========
    tabs = st.tabs([
        "📈 Analytics",
        "📜 Request History",
        "🔧 By Provider",
        "⚙️ By Operation",
        "📅 Daily Breakdown"
    ])
    
    # ========== TAB 1: ANALYTICS ==========
    with tabs[0]:
        st.markdown("### 📈 Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚡ Average Response Time")
            avg_time = analytics['avg_response_time_ms']
            
            if avg_time < 1000:
                st.success(f"{avg_time:.0f} ms - Excellent")
            elif avg_time < 3000:
                st.info(f"{avg_time:.0f} ms - Good")
            else:
                st.warning(f"{avg_time:.0f} ms - Slow")
        
        with col2:
            st.markdown("#### 💰 Cost Efficiency")
            if analytics['total_requests'] > 0:
                cost_per_request = analytics['total_credits'] / analytics['total_requests']
                st.metric(
                    "Avg Credits/Request",
                    f"{cost_per_request:.2f}",
                    help="Average credits spent per request"
                )
        
        st.divider()
        
        # Provider breakdown chart
        st.markdown("#### 🤖 Usage by Provider")
        
        if analytics['by_provider']:
            provider_data = []
            for provider, stats in analytics['by_provider'].items():
                provider_data.append({
                    'Provider': provider.upper(),
                    'Requests': stats['request_count'],
                    'Tokens': stats['tokens'],
                    'Credits': stats['credits'],
                    'USD': f"${stats['usd']:.2f}"
                })
            
            df_providers = pd.DataFrame(provider_data)
            st.dataframe(df_providers, use_container_width=True, hide_index=True)
        else:
            st.info("No provider data available")
    
    # ========== TAB 2: REQUEST HISTORY ==========
    with tabs[1]:
        st.markdown("### 📜 Request History")
        
        # Filters
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            provider_filter = st.selectbox(
                "Filter by Provider",
                ["All"] + list(analytics['by_provider'].keys()) if analytics['by_provider'] else ["All"],
                key="provider_filter"
            )
        
        with col2:
            limit = st.number_input("Show Last N Requests", min_value=10, max_value=500, value=50, step=10)
        
        with col3:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        # Get history
        history = logger.get_request_history(
            user_id=user_id,
            limit=limit,
            provider=None if provider_filter == "All" else provider_filter,
            start_date=start_date
        )
        
        if not history:
            st.info("No requests found in this time range")
        else:
            st.markdown(f"**Showing {len(history)} requests**")
            
            # Display as expandable cards
            for req in history:
                with st.expander(
                    f"{'✅' if req['success'] else '❌'} {req['endpoint']} - "
                    f"{req['provider'].upper()} - "
                    f"{datetime.fromisoformat(req['created_at']).strftime('%Y-%m-%d %H:%M:%S')}"
                ):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"**Request ID:** {req['id']}")
                        st.markdown(f"**Endpoint:** `{req['endpoint']}`")
                        st.markdown(f"**Provider:** {req['provider'].upper()}")
                        st.markdown(f"**Operation:** {req['operation_type']}")
                    
                    with col2:
                        st.markdown(f"**Tokens Used:** {req['tokens_used']:,}")
                        st.markdown(f"**Cost (Credits):** {req['cost_credits']:.2f}")
                        st.markdown(f"**Cost (USD):** ${req['cost_usd']:.4f}")
                        st.markdown(f"**Response Time:** {req['response_time_ms']} ms")
                    
                    with col3:
                        st.markdown(f"**Success:** {'✅ Yes' if req['success'] else '❌ No'}")
                        st.markdown(f"**Timestamp:** {req['created_at']}")
                        
                        if not req['success'] and req['error_message']:
                            st.error(f"Error: {req['error_message']}")
                    
                    if req['metadata']:
                        with st.expander("📋 Request Metadata"):
                            st.json(req['metadata'])
    
    # ========== TAB 3: BY PROVIDER ==========
    with tabs[2]:
        st.markdown("### 🔧 Usage by Provider")
        
        if not analytics['by_provider']:
            st.info("No provider data available")
        else:
            for provider, stats in analytics['by_provider'].items():
                with st.container(border=True):
                    st.markdown(f"### {provider.upper()}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Requests", f"{stats['request_count']:,}")
                    
                    with col2:
                        st.metric("Tokens", f"{stats['tokens']:,}")
                    
                    with col3:
                        st.metric("Credits", f"{stats['credits']:.1f}")
                    
                    with col4:
                        st.metric("USD", f"${stats['usd']:.2f}")
    
    # ========== TAB 4: BY OPERATION ==========
    with tabs[3]:
        st.markdown("### ⚙️ Usage by Operation Type")
        
        if not analytics['by_operation']:
            st.info("No operation data available")
        else:
            for operation, stats in analytics['by_operation'].items():
                with st.container(border=True):
                    st.markdown(f"### {operation.replace('_', ' ').title()}")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Requests", f"{stats['request_count']:,}")
                    
                    with col2:
                        st.metric("Total Tokens", f"{stats['tokens']:,}")
                    
                    with col3:
                        avg_time = stats['avg_response_time']
                        st.metric("Avg Response", f"{avg_time:.0f} ms")
    
    # ========== TAB 5: DAILY BREAKDOWN ==========
    with tabs[4]:
        st.markdown("### 📅 Daily Usage Breakdown")
        
        if not analytics['daily_breakdown']:
            st.info("No daily data available")
        else:
            # Create DataFrame for chart
            df_daily = pd.DataFrame(analytics['daily_breakdown'])
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            
            st.markdown("#### 📊 Daily Request Volume")
            st.line_chart(df_daily.set_index('date')['requests'])
            
            st.markdown("#### 💰 Daily Credits Spent")
            st.area_chart(df_daily.set_index('date')['credits'])
            
            st.markdown("#### 📋 Detailed Daily Data")
            st.dataframe(
                df_daily.sort_values('date', ascending=False),
                use_container_width=True,
                hide_index=True
            )
    
    st.divider()
    
    # ========== ADMIN ACTIONS ==========
    st.markdown("### 🛠️ Admin Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Cleanup Old Logs", help="Delete logs older than 90 days"):
            with st.spinner("Cleaning up old logs..."):
                deleted = logger.cleanup_old_logs(days_to_keep=90)
                st.success(f"✅ Deleted {deleted} old log entries")
    
    with col2:
        if st.button("📊 Export CSV", help="Export current view to CSV"):
            history = logger.get_request_history(user_id=user_id, limit=1000, start_date=start_date)
            
            if history:
                df = pd.DataFrame(history)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"ai_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col3:
        st.caption("Logs are stored in SQLite database")
        st.caption(f"Database: `creator_catalyst.db`")


def main():
    """Main entry point for AI logs dashboard."""
    st.set_page_config(
        page_title="AI Logs Dashboard",
        page_icon="🔍",
        layout="wide"
    )
    
    render_ai_logs_dashboard()


if __name__ == "__main__":
    main()