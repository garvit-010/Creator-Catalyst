import streamlit as st
import os
import time
import io
import re
import tempfile
import subprocess
from PIL import Image
from huggingface_hub import InferenceClient
from pathlib import Path
from dotenv import load_dotenv

# Import the wrapper
from llm_wrapper import LLMWrapper

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

# Initialize Wrapper (this now handles all LLM configuration)
llm_client = LLMWrapper()

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

def process_video_with_llm(video_path, target_platform="General"):
    """
    Uploads and analyzes video using the LLM wrapper with platform-specific tone.
    Now fully handles upload, processing, and analysis through the wrapper.
    """
    # Get platform-specific style guide
    style_modifier = get_platform_prompt_modifier(target_platform)
    
    analysis_prompt = f"""
You are an expert video analyst and content strategist. Your primary and most important task is to provide a complete and accurate transcription of the provided video in SRT format.
After the transcription, you will also provide a creative content plan.

IMPORTANT STYLE INSTRUCTION:
The content you generate should be optimized for: {target_platform}
Style Guide: {style_modifier}

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
   Summary: (A one-sentence summary that captures the {target_platform} vibe)

2. Topic: ...

### Blog Post
(Your full, well-structured blog post between 300 and 400 words with markdown formatting here. Write in a style appropriate for {target_platform}.)

### Social Media Post
(Your single, short, and engaging social media post specifically optimized for {target_platform}. Follow the style guide closely.)

### Thumbnail Ideas
(Provide 3 distinct ideas here, each as a numbered list item, keeping {target_platform} aesthetics in mind)
1. (A detailed, visually descriptive prompt for an AI image generator)
2. (Another detailed prompt)
3. (Another detailed prompt)
"""
    
    # Step 1: Upload video through wrapper
    with st.spinner("📤 Uploading video to AI service..."):
        video_file, provider = llm_client.upload_video_file(video_path)
    
    if not video_file:
        st.warning("⚠️ Video upload failed. Generating fallback content...")
        return llm_client.analyze_video(None, analysis_prompt)
    
    # Step 2: Analyze video through wrapper
    with st.spinner(f"🤖 Analyzing video with {provider.upper()} for {target_platform}..."):
        results = llm_client.analyze_video(video_file, analysis_prompt)
    
    return results

def enhance_tweet_with_llm(tweet_text, target_platform="Twitter/X"):
    """Enhances a tweet using the LLM wrapper with platform-specific style."""
    style_modifier = get_platform_prompt_modifier(target_platform)
    
    prompt = f"""You are a social media expert specializing in {target_platform}. 
Enhance the following post to make it more engaging for {target_platform}.

Style Guide: {style_modifier}

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
    - **✍️ Full-Length Blog Post**
    - **📱 Engaging Social Post**
    - **🎨 Clickable Thumbnail Ideas**
    - **🎯 Platform-Specific Tone Optimization** (NEW!)
    """)
    st.divider()
    
    # Show current AI provider status
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

    # Show current provider
    provider = llm_client.get_current_provider()
    if provider != "none":
        st.info(f"🤖 Current AI Provider: **{provider.upper()}**")

    # Platform Selection - Prominent placement
    st.subheader("🎯 Select Target Platform")
    col1, col2 = st.columns([2, 3])
    
    with col1:
        selected_platform = st.selectbox(
            "Choose your content's primary platform:",
            options=list(PLATFORM_TONES.keys()),
            index=list(PLATFORM_TONES.keys()).index(st.session_state.selected_platform),
            help="This will adjust the tone and style of all generated content"
        )
        st.session_state.selected_platform = selected_platform
    
    with col2:
        platform_info = PLATFORM_TONES[selected_platform]
        st.info(f"**{selected_platform}**: {platform_info['description']}")

    st.divider()

    uploaded_file = st.file_uploader("Upload your video file", type=['mp4','mov','webm','mkv'])

    if uploaded_file:
        temp_dir = tempfile.gettempdir()
        video_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save uploaded file
        with open(video_path, "wb") as f: 
            f.write(uploaded_file.getbuffer())
        st.session_state.video_path = video_path
        
        st.video(video_path)

        if st.button("🚀 Analyze Video & Generate All Content", type="primary", use_container_width=True):
            # Process video using the wrapper with platform-specific tone
            st.session_state.results = process_video_with_llm(
                video_path, 
                target_platform=st.session_state.selected_platform
            )
            
            if st.session_state.results and "error" not in st.session_state.results:
                st.success(f"✅ Full analysis complete for {st.session_state.selected_platform}!")
            elif st.session_state.results.get("captions"):
                st.warning("⚠️ Using fallback/mock results (primary AI provider unavailable)")
            else:
                st.error("❌ Analysis failed. Please check your API configuration.")

    # --- Display Results ---
    if st.session_state.results and st.session_state.results.get('captions'):
        results = st.session_state.results
        
        # Show platform badge
        st.success(f"📊 Content optimized for: **{st.session_state.selected_platform}**")
        
        tabs = st.tabs(["🎧 Captions", "✂️ Shorts Ideas", "📝 Blog Post", "📱 Social Media", "🎨 Thumbnails"])

        with tabs[0]:
            st.header("Captions (SRT)")
            captions_text = results.get('captions', "No captions generated.")
            st.text_area("Transcript", captions_text, height=400)
            
            # Download button
            if captions_text and captions_text != "No captions generated.":
                st.download_button(
                    label="📥 Download SRT File",
                    data=captions_text,
                    file_name="captions.srt",
                    mime="text/plain"
                )

        with tabs[1]:
            st.header("Short Clip Ideas")
            st.caption(f"Optimized for {st.session_state.selected_platform}")
            shorts = results.get('shorts_ideas', [])
            
            if not shorts:
                st.info("No shorts ideas generated.")
            
            for i, short in enumerate(shorts):
                with st.container(border=True):
                    st.subheader(f"Idea {i+1}: {short.get('topic', 'N/A')}")
                    st.markdown(f"**Timestamps:** `{short.get('start_time', 'N/A')} - {short.get('end_time', 'N/A')}`")
                    st.markdown(f"**Summary:** {short.get('summary', 'N/A')}")
                    
                    if st.button(f"✂️ Prepare Clip {i+1}", key=f"clip_{i}"):
                        if st.session_state.video_path:
                            with st.spinner("Clipping with ffmpeg..."):
                                clip_path = clip_video_ffmpeg(
                                    st.session_state.video_path, 
                                    short.get('start_time'), 
                                    short.get('end_time')
                                )
                                if clip_path: 
                                    st.session_state[f"clip_path_{i}"] = clip_path
                                    st.success(f"✅ Clip {i+1} ready!")
                    
                    if f"clip_path_{i}" in st.session_state:
                        with open(st.session_state[f"clip_path_{i}"], "rb") as file:
                            st.download_button(
                                label=f"📥 Download Clip {i+1}", 
                                data=file, 
                                file_name=f"clip_{i+1}.mp4", 
                                mime="video/mp4",
                                key=f"download_{i}"
                            )

        with tabs[2]:
            st.header("Blog Post")
            st.caption(f"Written in {st.session_state.selected_platform} style")
            blog_content = results.get('blog_post', 'No blog post generated.')
            st.markdown(blog_content)
            
            if blog_content != 'No blog post generated.':
                st.download_button(
                    label="📥 Download as Markdown",
                    data=blog_content,
                    file_name="blog_post.md",
                    mime="text/markdown"
                )

        with tabs[3]:
            st.header("Social Media Post")
            st.caption(f"Optimized for {st.session_state.selected_platform}")
            original_tweet = results.get('social_post', 'No post generated.')
            
            if not st.session_state.enhanced_tweet:
                st.session_state.enhanced_tweet = original_tweet

            st.markdown(f"> {st.session_state.enhanced_tweet}")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("✨ Enhance Post", key="enhance_tweet"):
                    with st.spinner(f"Refining for {st.session_state.selected_platform}..."):
                        enhanced = enhance_tweet_with_llm(
                            original_tweet, 
                            target_platform=st.session_state.selected_platform
                        )
                        if enhanced and "unavailable" not in enhanced.lower():
                            st.session_state.enhanced_tweet = enhanced
                            st.rerun()
                        else:
                            st.error("Enhancement failed. Using original post.")

        with tabs[4]:
            st.header("Thumbnail Ideas")
            st.caption(f"Visual style for {st.session_state.selected_platform}")
            thumbnail_ideas = results.get('thumbnail_ideas', [])
            
            if not thumbnail_ideas:
                st.info("No thumbnail ideas generated.")
            
            for i, idea in enumerate(thumbnail_ideas):
                with st.container(border=True):
                    st.subheader(f"Idea {i+1}")
                    st.markdown(f"*{idea}*")
                    
                    if st.button(f"🎨 Generate Thumbnail {i+1}", key=f"gen_thumb_{i}"):
                        if hf_client:
                            with st.spinner("Generating thumbnail..."):
                                pil_img = generate_thumbnail_hf(idea)
                                if pil_img:
                                    st.session_state[f"pil_image_{i}"] = pil_img
                        else:
                            st.error("Hugging Face client not configured. Please add HF_TOKEN to .env.local")
                    
                    if f"pil_image_{i}" in st.session_state:
                        image_data = st.session_state[f"pil_image_{i}"]
                        st.image(image_data, use_container_width=True)
                        
                        # Download button
                        img_bytes = pil_to_bytes(image_data)
                        st.download_button(
                            label=f"📥 Download Thumbnail {i+1}",
                            data=img_bytes,
                            file_name=f"thumbnail_{i+1}.png",
                            mime="image/png",
                            key=f"download_thumb_{i}"
                        )

# --- Main App Router ---
with st.sidebar:
    st.markdown("## 🚀 Creator Catalyst")
    page = st.radio("Navigation", ["Home", "Creator Tool"], label_visibility="hidden")

if page == "Home":
    home_page()
elif page == "Creator Tool":
    creator_tool_page()