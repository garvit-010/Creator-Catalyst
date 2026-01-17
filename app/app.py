import streamlit as st
import os
import time
import io
import re
import tempfile
import sys
import subprocess
from PIL import Image
from huggingface_hub import InferenceClient
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Import the wrapper and storage
from src.core.llm_wrapper import LLMWrapper
from src.database.storage_manager import get_storage_manager
from src.database.credits_manager import get_credits_manager
# NEW: Import engagement scoring modules
from src.core.engagement_scorer import get_engagement_scorer
from src.ui.components.engagement_ui import (
    render_engagement_score_card,
    analyze_and_display_score,
    add_engagement_scoring_section
)


def render_credits_page(credits_manager):
    """Main credits management page."""
    st.title("💳 Credits Management")
    st.markdown("Manage your credits and view usage history")
    
    # Use the passed credits manager
    credits = credits_manager
    
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

# --- PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Creator Catalyst",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Keys & Config ---
env_path = Path(__file__).parent.parent / '.env.local'
load_dotenv(env_path)

# Initialize Wrapper, Storage Manager, and Credits Manager
llm_client = LLMWrapper()
storage_manager = get_storage_manager()
credits_manager = get_credits_manager()

HF_TOKEN = os.getenv('HF_TOKEN')

# --- Configure HF API ---
try:
    hf_client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=HF_TOKEN)
except Exception as e:
    st.error(f"Failed to initialize Hugging Face API: {e}")
    hf_client = None

# --- Platform-Specific Tone Configurations ---
PLATFORM_TONES = {
    "General": {
        "description": "Balanced tone suitable for all platforms",
        "style_guide": "Use a clear, engaging tone that works across multiple platforms. Be informative and accessible."
    },
    "YouTube": {
        "description": "Storytelling and engaging narrative style",
        "style_guide": "Use a conversational, storytelling approach. Build narrative arcs, create anticipation, and maintain viewer engagement. Use phrases like 'let me show you', 'here's the thing', and create emotional connections."
    },
    "LinkedIn": {
        "description": "Professional and thought-leadership focused",
        "style_guide": "Use a professional, authoritative tone. Focus on insights, industry trends, and actionable takeaways. Use business-appropriate language and emphasize value propositions. Avoid casual slang."
    },
    "Twitter/X": {
        "description": "Punchy, viral-worthy with high energy",
        "style_guide": "Use short, punchy sentences. Create hype and urgency. Use power words, trending terminology, and emojis strategically. Be bold and attention-grabbing. Focus on hooks and viral potential."
    },
    "Instagram": {
        "description": "Visual-first, lifestyle-oriented narrative",
        "style_guide": "Use descriptive, visual language. Focus on aesthetics, lifestyle elements, and emotional appeal. Be inspirational and aspirational. Use casual, friendly tone with strategic emoji use."
    },
    "TikTok": {
        "description": "Fast-paced, trend-aware, Gen-Z friendly",
        "style_guide": "Use very short, snappy language. Be extremely casual and relatable. Reference trends, use internet slang appropriately, and create immediate hooks. Focus on entertainment value and quick pacing."
    }
}

# --- Helper Functions ---

def get_platform_prompt_modifier(platform):
    """Returns the style guide for the selected platform."""
    return PLATFORM_TONES.get(platform, PLATFORM_TONES["General"])["style_guide"]

def time_str_to_seconds(time_str):
    """Converts MM:SS or HH:MM:SS string to seconds."""
    if not time_str: 
        return 0
    numbers = re.findall(r'\d+', time_str)
    parts = [int(n) for n in numbers]
    if len(parts) == 2: 
        return parts[0] * 60 + parts[1]
    if len(parts) == 3: 
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 1: 
        return parts[0]
    return 0

def clip_video_ffmpeg(video_path, start_time_str, end_time_str):
    """Clips the video using ffmpeg for robustness."""
    try:
        start_seconds = time_str_to_seconds(start_time_str)
        end_seconds = time_str_to_seconds(end_time_str)
        duration = end_seconds - start_seconds
        
        if duration <= 0:
            st.error("Invalid timestamps: End time must be after start time.")
            return None

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_clip_file:
            output_path = temp_clip_file.name

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_seconds),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg clipping failed. Make sure ffmpeg is installed. Error: {e.stderr.decode()}")
        return None
    except Exception as e:
        st.error(f"Failed to clip video: {e}")
        return None

def process_video_with_progress(uploaded_file, video_path, target_platform="General", enable_grounding=True):
    """
    Uploads and analyzes video with progress tracking.
    Shows progress bar during upload and processing stages.
    """
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Stage 1: File Upload (0-20%)
            status_text.text("📤 Uploading video file...")
            progress_bar.progress(10)
            
            file_size = uploaded_file.size
            file_size_mb = file_size / (1024 * 1024)
            
            chunk_size = 1024 * 1024
            bytes_written = 0
            
            with open(video_path, "wb") as f:
                while True:
                    chunk = uploaded_file.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_written += len(chunk)
                    
                    upload_progress = min(20, int((bytes_written / file_size) * 20))
                    progress_bar.progress(upload_progress)
            
            status_text.text(f"✅ Upload complete ({file_size_mb:.1f} MB)")
            progress_bar.progress(20)
            time.sleep(0.5)
            
            # Stage 2: Preprocessing (20-30%)
            status_text.text("🔄 Preprocessing video...")
            progress_bar.progress(25)
            time.sleep(0.3)
            
            if not os.path.exists(video_path):
                raise Exception("Video file not found after upload")
            
            progress_bar.progress(30)
            status_text.text("✅ Preprocessing complete")
            time.sleep(0.3)
            
            # Stage 3: AI Upload to Gemini (30-50%)
            status_text.text("☁️ Uploading to AI service...")
            progress_bar.progress(35)
            
            video_file, provider = llm_client.upload_video_file(video_path)
            
            if not video_file:
                progress_bar.progress(50)
                status_text.text("⚠️ Using fallback mode (AI upload failed)")
                time.sleep(1)
            else:
                progress_bar.progress(50)
                status_text.text(f"✅ Uploaded to {provider.upper()}")
                time.sleep(0.3)
            
            # Stage 4: AI Analysis (50-90%)
            style_modifier = get_platform_prompt_modifier(target_platform)
            
            analysis_prompt = f"""
You are an expert video analyst and content strategist. Your primary and most important task is to provide a complete and accurate transcription of the provided video in SRT format.
After the transcription, you will also provide a creative content plan.

IMPORTANT STYLE INSTRUCTION:
The content you generate should be optimized for: {target_platform}
Style Guide: {style_modifier}

CRITICAL FACT-GROUNDING RULES:
1. ALL claims must come directly from the video - no speculation or assumptions
2. Cite timestamps for every factual claim: [Source: MM:SS]
3. If you cannot verify a claim from the transcript, DO NOT include it
4. Do not embellish or extrapolate beyond what is explicitly said
5. Statistics, quotes, and details must be word-for-word accurate

Structure your entire response using the following markdown format, and do not include any other text or explanations.

### Captions
```srt
(Your generated SRT captions here)
```

### Shorts Ideas
(Provide at least 5 ideas here, each formatted exactly as follows. Ensure the tone and topics align with {target_platform} style.)
1. Topic: (A short, catchy title appropriate for {target_platform})
   Start Time: MM:SS
   End Time: MM:SS
   Summary: (A one-sentence summary that captures the {target_platform} vibe - MUST be grounded in transcript)

2. Topic: ...

### Blog Post
(Your full, well-structured blog post between 300 and 400 words with markdown formatting here. Write in a style appropriate for {target_platform}. ALL factual claims must be verifiable in the transcript.)

### Social Media Post
(Your single, short, and engaging social media post specifically optimized for {target_platform}. Follow the style guide closely. Only include verified information.)

### Thumbnail Ideas
(Provide 3 distinct ideas here, each as a numbered list item, keeping {target_platform} aesthetics in mind and based on actual video content)
1. (A detailed, visually descriptive prompt for an AI image generator)
2. (Another detailed prompt)
3. (Another detailed prompt)
"""
            
            status_text.text(f"🤖 Analyzing video with {provider.upper() if video_file else 'fallback'}...")
            progress_bar.progress(60)
            
            for i in range(60, 90, 5):
                time.sleep(0.5)
                progress_bar.progress(i)
                if i == 70:
                    status_text.text("🤖 Generating captions...")
                elif i == 75:
                    status_text.text("🤖 Creating shorts ideas...")
                elif i == 80:
                    status_text.text("🤖 Writing blog post...")
                elif i == 85:
                    status_text.text("🤖 Crafting social media content...")
            
            results = llm_client.analyze_video(video_file, analysis_prompt, enable_grounding=enable_grounding)
            
            progress_bar.progress(90)
            status_text.text("✅ Analysis complete")
            time.sleep(0.3)
            
            # Stage 5: Fact-Grounding Validation (90-95%)
            if enable_grounding and results.get('captions'):
                status_text.text("🔍 Validating claims against transcript...")
                progress_bar.progress(93)
                time.sleep(0.5)
            
            progress_bar.progress(95)
            
            # Stage 6: Saving Results (95-100%)
            status_text.text("💾 Saving results to database...")
            progress_bar.progress(97)
            
            if results and "error" not in results:
                try:
                    video_id = storage_manager.save_analysis_results(
                        video_path=video_path,
                        results=results,
                        platform=target_platform,
                        grounding_enabled=enable_grounding
                    )
                    st.session_state.current_video_id = video_id
                except Exception as e:
                    st.warning(f"⚠️ Results generated but not saved to database: {e}")
            
            progress_bar.progress(100)
            status_text.text("✅ All processing complete!")
            time.sleep(1)
            
            progress_bar.empty()
            status_text.empty()
            
            return results
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Processing failed: {e}")
            return {"error": str(e)}

def enhance_tweet_with_llm(tweet_text, target_platform="Twitter/X"):
    """Enhances a tweet using the LLM wrapper with platform-specific style."""
    style_modifier = get_platform_prompt_modifier(target_platform)
    
    prompt = f"""You are a social media expert specializing in {target_platform}. 
Enhance the following post to make it more engaging for {target_platform}.

Style Guide: {style_modifier}

IMPORTANT: Only use information already present in the original post. Do not add new claims or facts.

Original post: '{tweet_text}'

Enhanced post (keep it under 280 characters if for Twitter/X):"""
    
    return llm_client.generate_text(prompt)

def generate_thumbnail_hf(prompt, reference_image=None):
    """Generates a thumbnail using Hugging Face SDXL."""
    if not hf_client:
        return None
        
    enhanced_prompt = f"Ultra-detailed, cinematic, exaggerated, vibrant YouTube thumbnail. Clickable, dramatic lighting, no text. Concept: {prompt}"
    try:
        if reference_image:
            return hf_client.image_to_image(
                image=reference_image, 
                prompt=enhanced_prompt, 
                guidance_scale=8.0, 
                num_inference_steps=30
            )
        else:
            return hf_client.text_to_image(
                prompt=enhanced_prompt, 
                guidance_scale=8.0, 
                num_inference_steps=30, 
                width=1024, 
                height=576
            )
    except Exception as e:
        st.error(f"Thumbnail generation failed: {e}")
        return None

def pil_to_bytes(image):
    """Converts PIL image to bytes for download."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

# --- Page Definitions ---

def home_page():
    """Renders the landing/description page for the app."""
    st.title("🚀 Welcome to Creator Catalyst")
    st.header("Your AI-Powered Content Repurposing Co-Pilot")
    st.markdown("""
    Tired of the content grind? **Creator Catalyst** is your secret weapon. 
    Turn a single long-form video into a full-blown marketing campaign, instantly.
    """)
    st.divider()
    st.subheader("What It Does")
    st.markdown("""
    - **✅ Accurate SRT Captions**
    - **💡 Viral Shorts Ideas**
    - **✏️ Full-Length Blog Post**
    - **📱 Engaging Social Post**
    - **🎨 Clickable Thumbnail Ideas**
    - **🎯 Platform-Specific Tone Optimization**
    - **🔍 Fact-Grounding Verification**
    - **💾 Persistent Storage & History Browsing**
    - **💳 Credits-Based Usage System**
    - **📊 Engagement Score Analytics** ← NEW!
    """)
    
    st.info("🔍 **New Feature**: All generated content is now verified against the video transcript to prevent AI hallucinations!")
    st.success("💾 **Persistent Storage**: All your videos and content are automatically saved to a local database for easy browsing and reuse!")
    st.success("📊 **Engagement Scoring**: Predict content performance before publishing!")
    
    st.divider()
    
    provider = llm_client.get_current_provider()
    if provider != "none":
        st.success(f"✅ AI Provider Active: **{provider.upper()}**")
    else:
        st.warning("⚠️ No AI providers configured. Please add API keys to `.env.local`")

def creator_tool_page():
    """Renders the main tool for video analysis and content generation."""
    st.title("🛠️ Creator Catalyst Tool")
    
    # Initialize session state
    if 'results' not in st.session_state: 
        st.session_state.results = {}
    if 'video_path' not in st.session_state: 
        st.session_state.video_path = None
    if 'enhanced_tweet' not in st.session_state: 
        st.session_state.enhanced_tweet = ""
    if 'selected_platform' not in st.session_state:
        st.session_state.selected_platform = "General"
    if 'enable_grounding' not in st.session_state:
        st.session_state.enable_grounding = True

    provider = llm_client.get_current_provider()
    credits_balance = credits_manager.get_balance()
    
    col1, col2 = st.columns(2)
    with col1:
        if provider != "none":
            st.info(f"🤖 Current AI Provider: **{provider.upper()}**")
    with col2:
        if credits_balance < 10:
            st.warning(f"⚠️ Low Credits: **{credits_balance}** remaining")
        else:
            st.success(f"💳 Credits: **{credits_balance}**")

    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        selected_platform = st.selectbox(
            "🎯 Target Platform:",
            options=list(PLATFORM_TONES.keys()),
            index=list(PLATFORM_TONES.keys()).index(st.session_state.selected_platform),
            help="This will adjust the tone and style of all generated content"
        )
        st.session_state.selected_platform = selected_platform
    
    with col2:
        platform_info = PLATFORM_TONES[selected_platform]
        st.info(f"**{selected_platform}**: {platform_info['description']}")
    
    with col3:
        enable_grounding = st.checkbox(
            "🔍 Fact-Grounding",
            value=st.session_state.enable_grounding,
            help="Verify all claims against transcript to prevent hallucinations"
        )
        st.session_state.enable_grounding = enable_grounding

    st.divider()

    uploaded_file = st.file_uploader("Upload your video file", type=['mp4','mov','webm','mkv'])

    if uploaded_file:
        temp_dir = tempfile.gettempdir()
        video_path = os.path.join(temp_dir, uploaded_file.name)
        
        st.session_state.video_path = video_path
        
        st.video(uploaded_file)

        if st.button("🚀 Analyze Video & Generate All Content", type="primary", use_container_width=True):
            has_credits, balance, cost = credits_manager.has_sufficient_credits('video_upload')
            
            if not has_credits:
                st.error(f"❌ Insufficient credits! You need {cost} credits but only have {balance}.")
                st.info("💡 Purchase more credits from the Credits page to continue.")
                return
            
            st.info(f"💳 This operation will cost **{cost} credits**. Current balance: {balance}")
            
            success, new_balance = credits_manager.deduct_credits(
                'video_upload',
                description=f"Video analysis: {uploaded_file.name}"
            )
            
            if not success:
                st.error("❌ Failed to deduct credits. Please try again.")
                return
            
            st.success(f"✅ Credits deducted! New balance: **{new_balance}**")
            
            st.session_state.results = process_video_with_progress(
                uploaded_file=uploaded_file,
                video_path=video_path,
                target_platform=st.session_state.selected_platform,
                enable_grounding=st.session_state.enable_grounding
            )
            
            if st.session_state.results and "error" not in st.session_state.results:
                st.success(f"✅ Full analysis complete for {st.session_state.selected_platform}!")

                if 'grounding_metadata' in st.session_state.results:
                    metadata = st.session_state.results['grounding_metadata']
                    if metadata.get('enabled'):
                        st.metric("🔍 Fact-Grounding Active", "Enabled")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Blog Verification",
                                f"{metadata.get('blog_grounding_rate', 0):.0%}"
                            )
                        with col2:
                            st.metric(
                                "Social Verification",
                                f"{metadata.get('social_grounding_rate', 0):.0%}"
                            )
                        with col3:
                            st.metric(
                                "Shorts Verification",
                                f"{metadata.get('shorts_verification_rate', 0):.0%}"
                            )
        # --- Display Results --- (Continuation from Part 1)
    if st.session_state.results and st.session_state.results.get('captions'):
        results = st.session_state.results

        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.success(f"📊 Content optimized for: **{st.session_state.selected_platform}**")
        with status_col2:
            if results.get('grounding_metadata', {}).get('enabled'):
                st.success("🔍 **Fact-Grounding**: Active")
            else:
                st.info("🔍 **Fact-Grounding**: Disabled")

        # UPDATED: Added new Engagement Analytics tab
        tabs = st.tabs([
            "🎧 Captions",
            "✂️ Shorts Ideas",
            "📝 Blog Post",
            "📱 Social Media",
            "🎨 Thumbnails",
            "📊 Grounding Report",
            "📈 Engagement Analytics"  # NEW TAB
        ])

        # TAB 0: Captions
        with tabs[0]:
            st.header("Captions (SRT)")
            captions_text = results.get('captions', "No captions generated.")
            st.text_area("Transcript", captions_text, height=400)

            if captions_text and captions_text != "No captions generated.":
                st.download_button(
                    label="📥 Download SRT File",
                    data=captions_text,
                    file_name="captions.srt",
                    mime="text/plain"
                )

        # TAB 1: Shorts Ideas (ENHANCED with engagement scoring)
        with tabs[1]:
            st.header("Short Clip Ideas")
            st.caption(f"Optimized for {st.session_state.selected_platform}")
            shorts = results.get('shorts_ideas', [])

            if not shorts:
                st.info("No shorts ideas generated.")

            for i, short in enumerate(shorts):
                with st.container(border=True):
                    title_col, badge_col = st.columns([4, 1])
                    with title_col:
                        st.subheader(f"Idea {i+1}: {short.get('topic', 'N/A')}")
                    with badge_col:
                        if 'validation_badge' in short:
                            st.markdown(f"**{short['validation_badge']}**")

                    st.markdown(f"**Timestamps:** `{short.get('start_time', 'N/A')} - {short.get('end_time', 'N/A')}`")
                    st.markdown(f"**Summary:** {short.get('summary', 'N/A')}")

                    # NEW: Quick engagement score for each short
                    if st.button(f"📊 Score This Idea", key=f"score_short_{i}"):
                        scorer = get_engagement_scorer()
                        summary = short.get('summary', '')
                        if summary:
                            score = scorer.score_content(
                                summary,
                                'social_post',
                                st.session_state.selected_platform
                            )

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Engagement Score", f"{score.overall_score}/100")
                            with col2:
                                st.metric("Best Platform", score.recommended_platform)

                    if 'supporting_text' in short:
                        with st.expander("🔍 Transcript Evidence"):
                            st.caption(short['supporting_text'])

                    if st.button(f"✂️ Prepare Clip {i+1}", key=f"clip_{i}"):
                        has_credits, balance, cost = credits_manager.has_sufficient_credits('shorts_clip')

                        if not has_credits:
                            st.error(f"❌ Need {cost} credits (have {balance})")
                        elif st.session_state.video_path:
                            with st.spinner("Clipping with ffmpeg..."):
                                clip_path = clip_video_ffmpeg(
                                    st.session_state.video_path,
                                    short.get('start_time'),
                                    short.get('end_time')
                                )
                                if clip_path:
                                    credits_manager.deduct_credits(
                                        'shorts_clip',
                                        description=f"Shorts clip {i+1}"
                                    )
                                    st.session_state[f"clip_path_{i}"] = clip_path
                                    st.success(f"✅ Clip {i+1} ready! (1 credit used)")

                    if f"clip_path_{i}" in st.session_state:
                        with open(st.session_state[f"clip_path_{i}"], "rb") as file:
                            st.download_button(
                                label=f"📥 Download Clip {i+1}",
                                data=file,
                                file_name=f"clip_{i+1}.mp4",
                                mime="video/mp4",
                                key=f"download_{i}"
                            )

        # TAB 2: Blog Post (ENHANCED with engagement scoring)
        with tabs[2]:
            st.header("Blog Post")
            st.caption(f"Written in {st.session_state.selected_platform} style")
            blog_content = results.get('blog_post', 'No blog post generated.')

            if results.get('blog_post_original'):
                compare_tab1, compare_tab2 = st.tabs(["✅ Verified Version", "⚠️ Original (Unfiltered)"])
                with compare_tab1:
                    st.markdown(blog_content)
                with compare_tab2:
                    st.warning("This version may contain unverified claims")
                    st.markdown(results['blog_post_original'])
            else:
                st.markdown(blog_content)

            # NEW: Engagement scoring section
            st.divider()
            st.markdown("### 📊 Engagement Analysis")
            add_engagement_scoring_section(
                blog_content,
                st.session_state.selected_platform
            )

            if blog_content != 'No blog post generated.':
                st.download_button(
                    label="📥 Download as Markdown",
                    data=blog_content,
                    file_name="blog_post.md",
                    mime="text/markdown"
                )

        # TAB 3: Social Media (ENHANCED with engagement scoring)
        with tabs[3]:
            st.header("Social Media Post")
            st.caption(f"Optimized for {st.session_state.selected_platform}")

            original_tweet = results.get('social_post', 'No post generated.')

            if not st.session_state.enhanced_tweet:
                st.session_state.enhanced_tweet = original_tweet

            st.markdown("### 📱 Your Post")
            st.markdown(f"> {st.session_state.enhanced_tweet}")

            st.divider()

            # NEW: Engagement scoring section
            st.markdown("### 📊 Engagement Score")

            if st.button("🔍 Analyze Engagement Potential", key="analyze_social"):
                analyze_and_display_score(
                    content=st.session_state.enhanced_tweet,
                    content_type="social_post",
                    target_platform=st.session_state.selected_platform,
                    show_compact=False
                )

            st.divider()

            col1, col2 = st.columns([1, 3])

            with col1:
                if st.button("✨ Enhance Post", key="enhance_tweet"):
                    has_credits, balance, cost = credits_manager.has_sufficient_credits('tweet_enhancement')

                    if not has_credits:
                        st.error(f"❌ Need {cost} credits (have {balance})")
                    else:
                        with st.spinner("Enhancing post..."):
                            enhanced = enhance_tweet_with_llm(
                                st.session_state.enhanced_tweet,
                                st.session_state.selected_platform
                            )

                            if enhanced:
                                credits_manager.deduct_credits(
                                    'tweet_enhancement',
                                    description=f"Enhanced social post for {st.session_state.selected_platform}"
                                )

                                st.session_state.enhanced_tweet = enhanced
                                st.success("✅ Post enhanced! (1 credit used)")
                                st.rerun()

            if st.session_state.enhanced_tweet != "No post generated.":
                st.download_button(
                    label="📥 Copy to Clipboard",
                    data=st.session_state.enhanced_tweet,
                    file_name="social_post.txt",
                    mime="text/plain",
                    key="download_social"
                )

        # TAB 4: Thumbnails
        with tabs[4]:
            st.header("Thumbnail Ideas")
            st.caption(f"Designed for {st.session_state.selected_platform}")

            thumbnail_ideas = results.get('thumbnail_ideas', [])

            if not thumbnail_ideas:
                st.info("No thumbnail ideas generated.")
            else:
                for idx, idea in enumerate(thumbnail_ideas):
                    with st.container(border=True):
                        st.markdown(f"**Idea {idx+1}:** {idea}")

                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if st.button(f"🎨 Generate", key=f"gen_thumb_{idx}"):
                                has_credits, balance, cost = credits_manager.has_sufficient_credits('thumbnail_generation')

                                if not has_credits:
                                    st.error(f"❌ Need {cost} credits (have {balance})")
                                elif hf_client:
                                    with st.spinner(f"Generating thumbnail {idx+1}..."):
                                        img = generate_thumbnail_hf(idea)
                                        if img:
                                            credits_manager.deduct_credits(
                                                'thumbnail_generation',
                                                description=f"Generated thumbnail {idx+1}"
                                            )
                                            st.session_state[f"thumbnail_{idx}"] = img
                                            st.success(f"✅ Generated! (1 credit used)")
                                else:
                                    st.warning("⚠️ Thumbnail generation requires HF_TOKEN")

                        if f"thumbnail_{idx}" in st.session_state:
                            img = st.session_state[f"thumbnail_{idx}"]
                            st.image(img, use_container_width=True)
                            st.download_button(
                                label=f"📥 Download Thumbnail {idx+1}",
                                data=pil_to_bytes(img),
                                file_name=f"thumbnail_{idx+1}.png",
                                mime="image/png",
                                key=f"dl_thumb_{idx}"
                            )

        # TAB 5: Grounding Report
        with tabs[5]:
            st.header("Fact-Grounding Report")

            if not results.get('grounding_metadata', {}).get('enabled'):
                st.info("🔍 Fact-grounding was not enabled for this analysis.")
            else:
                metadata = results['grounding_metadata']

                st.success("✅ All content has been verified against the video transcript")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Blog Post Grounding",
                        f"{metadata.get('blog_grounding_rate', 0):.0%}",
                        help="Percentage of claims verified in transcript"
                    )

                with col2:
                    st.metric(
                        "Social Post Grounding",
                        f"{metadata.get('social_grounding_rate', 0):.0%}",
                        help="Percentage of claims verified in transcript"
                    )

                with col3:
                    st.metric(
                        "Shorts Verification",
                        f"{metadata.get('shorts_verification_rate', 0):.0%}",
                        help="Percentage of shorts with verified content"
                    )

                st.divider()

                st.markdown("### 📋 Grounding Details")

                if metadata.get('blog_filtered_claims'):
                    with st.expander("⚠️ Filtered Blog Claims", expanded=False):
                        st.caption("These claims were removed because they couldn't be verified:")
                        for claim in metadata['blog_filtered_claims']:
                            st.markdown(f"- {claim}")

                if metadata.get('social_filtered_claims'):
                    with st.expander("⚠️ Filtered Social Claims", expanded=False):
                        st.caption("These claims were removed because they couldn't be verified:")
                        for claim in metadata['social_filtered_claims']:
                            st.markdown(f"- {claim}")

        # TAB 6: Engagement Analytics (NEW)
        with tabs[6]:
            st.header("📈 Engagement Analytics")
            st.markdown("Compare engagement potential across all generated content")

            if st.button("📊 Analyze All Content", key="analyze_all"):
                with st.spinner("Analyzing all content for engagement..."):
                    scorer = get_engagement_scorer()

                    # Score blog post
                    blog_score = None
                    if results.get('blog_post'):
                        blog_score = scorer.score_content(
                            results['blog_post'],
                            'blog_post',
                            st.session_state.selected_platform
                        )

                    # Score social post
                    social_score = None
                    if results.get('social_post'):
                        social_score = scorer.score_content(
                            results['social_post'],
                            'social_post',
                            st.session_state.selected_platform
                        )

                    # Score each short idea summary
                    shorts_scores = []
                    for short in results.get('shorts_ideas', []):
                        if short.get('summary'):
                            score = scorer.score_content(
                                short['summary'],
                                'social_post',
                                st.session_state.selected_platform
                            )
                            shorts_scores.append({
                                'topic': short['topic'],
                                'score': score
                            })

                    # Display results
                    st.markdown("### 📊 Content Performance Comparison")

                    # Blog post score
                    if blog_score:
                        with st.expander("📝 Blog Post Analysis", expanded=True):
                            render_engagement_score_card(blog_score, show_details=True)

                    st.divider()

                    # Social post score
                    if social_score:
                        with st.expander("📱 Social Media Post Analysis", expanded=True):
                            render_engagement_score_card(social_score, show_details=True)

                    st.divider()

                    # Shorts scores
                    if shorts_scores:
                        st.markdown("### ✂️ Shorts Ideas Engagement Scores")

                        # Sort by score
                        shorts_scores.sort(key=lambda x: x['score'].overall_score, reverse=True)

                        for i, short_data in enumerate(shorts_scores, 1):
                            with st.expander(f"#{i} - {short_data['topic']}", expanded=False):
                                render_engagement_score_card(short_data['score'], show_details=False)

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Score", f"{short_data['score'].overall_score}/100")
                                with col2:
                                    st.metric("Best Platform", short_data['score'].recommended_platform)

# ---- SIDEBAR + NAVIGATION + ROUTING LOGIC ----
with st.sidebar:
    st.markdown("## 🚀 Creator Catalyst")

    st.divider()
    credits_balance = credits_manager.get_balance()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("💳 Credits", credits_balance)
    with col2:
        if st.button("➕ Add"):
            st.session_state.show_credits_modal = True

    with st.expander("💰 Credit Costs"):
        st.caption("📹 Video Upload: 5 credits")
        st.caption("📝 Blog Post: 2 credits")
        st.caption("📱 Social Post: 1 credit")
        st.caption("✂️ Shorts Clip: 1 credit")
        st.caption("🎨 Thumbnail: 1 credit")

    st.divider()

    # Updated navigation with AI Logs option
    page = st.radio(
        "Navigation", 
        ["Home", "Creator Tool", "History", "Credits", "AI Logs"], 
        label_visibility="hidden"
    )

    st.divider()
    st.caption("📊 Database Statistics")
    try:
        stats = storage_manager.get_statistics()
        st.caption(f"Videos: {stats['total_videos']}")
        st.caption(f"Content Pieces: {stats['total_contents']}")
    except:
        pass
    
    st.divider()
    
    # NEW: AI Usage Stats in Sidebar
    try:
        from src.database.ai_request_logger import get_ai_logger
        logger = get_ai_logger()
        
        # Get current hour stats
        is_allowed, rate_stats = logger.check_rate_limit()
        
        st.caption("⚡ AI Usage (This Hour)")
        st.caption(f"Requests: {rate_stats['requests_used']}/100")
        st.caption(f"Tokens: {rate_stats['tokens_used']:,}")
        
        if not is_allowed:
            st.warning("⚠️ Rate limit reached!")
    except:
        pass

# Route to pages
if page == "Home":
    home_page()
elif page == "Creator Tool":
    creator_tool_page()
elif page == "History":
    from src.ui.pages.history import render_history_page, render_video_details

    if 'selected_video_id' in st.session_state:
        render_video_details(storage_manager, st.session_state.selected_video_id)
    else:
        render_history_page()
elif page == "Credits":
    render_credits_page(credits_manager)
elif page == "AI Logs":
    # NEW: AI Logs Dashboard
    from src.ui.pages.ai_logs_dashboard import render_ai_logs_dashboard
    render_ai_logs_dashboard()