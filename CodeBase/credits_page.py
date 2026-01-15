"""
Credits page for Creator Catalyst.
Displays credit balance, usage history, and purchase options.
"""

import streamlit as st
from datetime import datetime
from credits_manager import get_credits_manager


def render_credits_page():
    """Main credits management page."""
    st.title("💳 Credits Management")
    st.markdown("Manage your credits and view usage history")
    
    # Initialize credits manager
    credits = get_credits_manager()
    
    # Get user stats
    stats = credits.get_user_stats()
    
    # Display current balance prominently
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Balance", 
            f"{stats['current_balance']} credits",
            delta=None,
            help="Your available credits for operations"
        )
    
    with col2:
        st.metric(
            "Total Earned", 
            f"{stats['total_earned']} credits",
            help="Total credits you've received"
        )
    
    with col3:
        st.metric(
            "Total Spent", 
            f"{stats['total_spent']} credits",
            help="Total credits you've used"
        )
    
    with col4:
        if stats['current_balance'] < 10:
            st.error("⚠️ Low Balance")
        elif stats['current_balance'] < 25:
            st.warning("⚠️ Running Low")
        else:
            st.success("✅ Good Balance")
    
    st.divider()
    
    # Tabs for different views
    tabs = st.tabs(["💰 Purchase Credits", "📊 Usage Statistics", "📜 Transaction History"])
    
    # ========== PURCHASE CREDITS TAB ==========
    with tabs[0]:
        st.subheader("💰 Purchase Credits")
        st.markdown("Select a credit package to purchase:")
        
        # Credit packages
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container(border=True):
                st.markdown("### 🥉 Starter")
                st.markdown("**50 Credits**")
                st.markdown("$9.99")
                st.caption("Perfect for trying out the platform")
                st.caption("• ~10 video uploads")
                st.caption("• ~25 blog posts")
                st.caption("• ~50 social posts")
                
                if st.button("Purchase Starter", key="buy_starter", use_container_width=True):
                    # Simulate purchase (in production, integrate payment gateway)
                    new_balance = credits.add_credits(50, description="Purchased Starter Package ($9.99)")
                    st.success(f"✅ Added 50 credits! New balance: {new_balance}")
                    st.rerun()
        
        with col2:
            with st.container(border=True):
                st.markdown("### 🥈 Pro")
                st.markdown("**150 Credits**")
                st.markdown("~~$29.99~~ **$24.99**")
                st.success("💎 Best Value - Save 17%")
                st.caption("• ~30 video uploads")
                st.caption("• ~75 blog posts")
                st.caption("• ~150 social posts")
                
                if st.button("Purchase Pro", key="buy_pro", use_container_width=True, type="primary"):
                    new_balance = credits.add_credits(150, description="Purchased Pro Package ($24.99)")
                    st.success(f"✅ Added 150 credits! New balance: {new_balance}")
                    st.balloons()
                    st.rerun()
        
        with col3:
            with st.container(border=True):
                st.markdown("### 🥇 Business")
                st.markdown("**500 Credits**")
                st.markdown("~~$99.99~~ **$79.99**")
                st.success("🚀 Maximum Savings - Save 20%")
                st.caption("• ~100 video uploads")
                st.caption("• ~250 blog posts")
                st.caption("• ~500 social posts")
                
                if st.button("Purchase Business", key="buy_business", use_container_width=True):
                    new_balance = credits.add_credits(500, description="Purchased Business Package ($79.99)")
                    st.success(f"✅ Added 500 credits! New balance: {new_balance}")
                    st.balloons()
                    st.rerun()
        
        st.divider()
        
        # Custom amount (for testing/admin)
        with st.expander("🔧 Admin: Add Custom Credits"):
            st.caption("For testing purposes only")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                custom_amount = st.number_input(
                    "Amount to add",
                    min_value=1,
                    max_value=10000,
                    value=10,
                    step=1
                )
            
            with col2:
                if st.button("Add Credits", use_container_width=True):
                    new_balance = credits.add_credits(
                        custom_amount,
                        description=f"Admin added {custom_amount} credits"
                    )
                    st.success(f"✅ Added {custom_amount} credits!")
                    st.rerun()
    
    # ========== USAGE STATISTICS TAB ==========
    with tabs[1]:
        st.subheader("📊 Usage Statistics")
        
        operation_counts = stats.get('operation_counts', {})
        
        if not operation_counts:
            st.info("No usage data yet. Start creating content to see statistics!")
        else:
            # Display usage by operation type
            st.markdown("### Operations Breakdown")
            
            for operation, data in operation_counts.items():
                with st.container(border=True):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        # Format operation name nicely
                        op_name = operation.replace('_', ' ').title()
                        st.markdown(f"**{op_name}**")
                    
                    with col2:
                        st.metric("Times Used", data['count'])
                    
                    with col3:
                        st.metric("Credits Spent", data['total_cost'])
            
            st.divider()
            
            # Calculate efficiency metrics
            st.markdown("### Efficiency Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if stats['total_spent'] > 0:
                    video_uploads = operation_counts.get('video_upload', {}).get('count', 0)
                    if video_uploads > 0:
                        avg_per_video = stats['total_spent'] / video_uploads
                        st.metric(
                            "Avg Credits per Video",
                            f"{avg_per_video:.1f}",
                            help="Average total credits spent per video uploaded"
                        )
            
            with col2:
                total_operations = sum(data['count'] for data in operation_counts.values())
                if total_operations > 0:
                    avg_per_operation = stats['total_spent'] / total_operations
                    st.metric(
                        "Avg Credits per Operation",
                        f"{avg_per_operation:.1f}",
                        help="Average credits per individual operation"
                    )
    
    # ========== TRANSACTION HISTORY TAB ==========
    with tabs[2]:
        st.subheader("📜 Transaction History")
        
        # Get transaction history
        transactions = credits.get_transaction_history(limit=100)
        
        if not transactions:
            st.info("No transactions yet.")
        else:
            st.markdown(f"**Last {len(transactions)} transactions**")
            
            # Display transactions in a nice format
            for txn in transactions:
                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    
                    with col1:
                        created = datetime.fromisoformat(txn['created_at'])
                        st.caption(created.strftime('%Y-%m-%d %H:%M:%S'))
                    
                    with col2:
                        if txn['type'] == 'credit':
                            st.markdown(f"**➕ +{txn['amount']} credits**")
                            st.success(txn['description'], icon="✅")
                        else:
                            st.markdown(f"**➖ -{txn['amount']} credits**")
                            st.caption(txn['description'])
                    
                    with col3:
                        if txn['operation']:
                            op_name = txn['operation'].replace('_', ' ').title()
                            st.caption(f"Operation: {op_name}")
                    
                    with col4:
                        st.metric(
                            "Balance",
                            txn['balance_after'],
                            label_visibility="collapsed"
                        )
    
    st.divider()
    
    # Credit costs reference
    st.markdown("### 💡 Credit Cost Reference")
    
    cost_data = [
        ("📹 Video Upload", "5 credits", "Full video analysis with captions, blog, social post, shorts, and thumbnails"),
        ("📝 Blog Generation", "2 credits", "Individual blog post generation"),
        ("📱 Social Post", "1 credit", "Enhanced social media post"),
        ("✂️ Shorts Clip", "1 credit", "Video clip preparation and export"),
        ("🎨 Thumbnail Generation", "1 credit", "AI-generated thumbnail image"),
        ("✨ Tweet Enhancement", "1 credit", "Enhanced social post with AI optimization")
    ]
    
    for operation, cost, description in cost_data:
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{operation}**")
                st.caption(description)
            with col2:
                st.markdown(f"**{cost}**")


def main():
    """Main entry point for credits page."""
    render_credits_page()


if __name__ == "__main__":
    main()