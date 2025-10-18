import streamlit as st
import google.generativeai as genai
import os
import time
import io
import re
import tempfile
import ssl
import subprocess
from PIL import Image
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from dotenv import load_dotenv

# --- PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Creator Catalyst",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Keys ---
try:
    # Get API keys from Streamlit secrets
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception as e:
    st.error("Error loading API keys. Please check your .streamlit/secrets.toml file")
    st.info("For local development, create .streamlit/secrets.toml")
    st.info("For Streamlit Cloud, add secrets in the dashboard")
    st.stop()

# --- Configure APIs ---
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    analysis_model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    hf_client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=HF_TOKEN)
except Exception as e:
    st.error(f"Failed to initialize APIs: {e}")
    st.stop()


# --- Helper Functions (The Engine of the App) ---

def upload_file_with_retries(path, retries=3, delay=5):
    """Uploads a file to the Gemini API with retries for transient network errors."""
    try:
        with open(path, 'rb') as f:
            file_bytes = f.read()
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None
        
    for attempt in range(retries):
        try:
            file = genai.upload_file(file_bytes)
            return file
        except ssl.SSLEOFError:
            st.warning(f"Network error during upload. Retrying in {delay} seconds...")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                st.error("File upload failed after multiple retries.")
                return None
        except Exception as e:
            st.error(f"An unexpected error occurred during file upload: {e}")
            return None
    return None

def time_str_to_seconds(time_str):
    """Converts MM:SS or HH:MM:SS string to seconds."""
    if not time_str: return 0
    numbers = re.findall(r'\d+', time_str)
    parts = [int(n) for n in numbers]
    if len(parts) == 2: return parts[0] * 60 + parts[1]
    if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 1: return parts[0]
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
        st.error(f"FFmpeg clipping failed. Make sure ffmpeg is installed and in your system's PATH. Error: {e.stderr.decode()}")
        return None
    except Exception as e:
        st.error(f"Failed to clip video: {e}")
        return None

def parse_text_response(text):
    """Parses the structured text response from Gemini into a dictionary."""
    results = {}
    
    captions_match = re.search(r"### Captions\s*\n(.*?)(?=\n###|$)", text, re.DOTALL)
    if captions_match:
        results['captions'] = captions_match.group(1).strip().removeprefix("```srt").removesuffix("```").strip()

    shorts_match = re.search(r"### Shorts Ideas\s*\n(.*?)(?=\n###|$)", text, re.DOTALL)
    if shorts_match:
        shorts_text = shorts_match.group(1).strip()
        ideas = []
        for idea_block in re.split(r'\n\s*\d+\.\s*', shorts_text):
            if not idea_block.strip(): continue
            topic_match = re.search(r"Topic:\s*(.*)", idea_block)
            start_match = re.search(r"Start Time:\s*([\d:]+)", idea_block)
            end_match = re.search(r"End Time:\s*([\d:]+)", idea_block)
            summary_match = re.search(r"Summary:\s*(.*)", idea_block, re.DOTALL)
            if topic_match and start_match and end_match and summary_match:
                ideas.append({
                    "topic": topic_match.group(1).strip(),
                    "start_time": start_match.group(1).strip(),
                    "end_time": end_match.group(1).strip(),
                    "summary": summary_match.group(1).strip()
                })
        results['shorts_ideas'] = ideas

    blog_match = re.search(r"### Blog Post\s*\n(.*?)(?=\n###|$)", text, re.DOTALL)
    if blog_match:
        results['blog_post'] = blog_match.group(1).strip()

    social_match = re.search(r"### Social Media Post\s*\n(.*?)(?=\n###|$)", text, re.DOTALL)
    if social_match:
        results['social_post'] = social_match.group(1).strip()

    thumb_match = re.search(r"### Thumbnail Ideas\s*\n(.*?)(?=\n###|$)", text, re.DOTALL)
    if thumb_match:
        ideas = [idea.strip() for idea in re.split(r'\n\s*\d+\.\s*', thumb_match.group(1).strip()) if idea.strip()]
        results['thumbnail_ideas'] = ideas

    return results

def process_video_with_gemini(video_part):
    """Asks Gemini for a structured text response and parses it."""
    analysis_prompt = """
    You are an expert video analyst and content strategist. Your primary and most important task is to provide a complete and accurate transcription of the provided video in SRT format.
    After the transcription, you will also provide a creative content plan.
    Structure your entire response using the following markdown format, and do not include any other text or explanations.

    ### Captions
    ```srt
    (Your generated SRT captions here)
    ```

    ### Shorts Ideas
    (Provide at least 5 ideas here, each formatted exactly as follows)
    1. Topic: (A short, catchy title)
       Start Time: MM:SS
       End Time: MM:SS
       Summary: (A one-sentence summary of the clip)
    
    2. Topic: ...

    ### Blog Post
    (Your full, well-structured blog post between 300 and 400 words with markdown formatting here)

    ### Social Media Post
    (Your single, short, and engaging social media post suitable for platforms like X/Twitter here)

    ### Thumbnail Ideas
    (Provide 3 distinct ideas here, each as a numbered list item)
    1. (A detailed, visually descriptive prompt for an AI image generator)
    2. (Another detailed prompt)
    3. (Another detailed prompt)
    """
    
    try:
        response = analysis_model.generate_content(
            [video_part, analysis_prompt],
            request_options={"timeout": 600}
        )
        return parse_text_response(response.text)
    except Exception as e:
        st.error(f"Gemini analysis failed: {e}")
        return {"error": str(e)}

def enhance_tweet_with_gemini(tweet_text):
    """Takes a tweet and makes it more engaging using a small, targeted prompt."""
    prompt = f"You are a social media expert. Enhance the following tweet to make it more engaging. Add relevant hashtags and emojis. Keep it under 280 characters. Tweet: '{tweet_text}'"
    try:
        response = analysis_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Tweet enhancement failed: {e}")
        return tweet_text 

def generate_thumbnail_hf(prompt, reference_image=None):
    """Generates a thumbnail using Hugging Face SDXL."""
    enhanced_prompt = f"Ultra-detailed, cinematic, exaggerated, vibrant YouTube thumbnail. Clickable, dramatic lighting, no text. Concept: {prompt}"
    try:
        if reference_image:
            return hf_client.image_to_image(
                image=reference_image, prompt=enhanced_prompt, guidance_scale=8.0, num_inference_steps=30)
        else:
            return hf_client.text_to_image(
                prompt=enhanced_prompt, guidance_scale=8.0, num_inference_steps=30, width=1024, height=576)
    except HfHubHTTPError as e:
         if e.response.status_code == 503:
             st.error("Thumbnail generation failed: The Hugging Face model is currently loading. Please try again in a minute.")
         else:
             st.error(f"Thumbnail generation failed with an HTTP error: {e}")
         return None
    except Exception as e:
        st.error(f"Thumbnail generation failed: {e}")
        return None

def pil_to_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

# --- Page Definitions ---

def home_page():
    """Renders the landing/description page for the app."""
    st.title("🚀 Welcome to Creator Catalyst")
    st.header("Your AI-Powered Content Repurposing Co-Pilot")
    
    st.markdown("""
    Tired of the content grind? **Creator Catalyst** is your secret weapon to multiply your content output without multiplying your effort. 
    Turn a single long-form video into a full-blown marketing campaign, instantly.
    """)
    
    st.divider()

    st.subheader("What It Does")
    st.markdown("""
    Our AI agent analyzes your video and automatically generates a suite of ready-to-use content:
    - **✅ Accurate SRT Captions:** Make your videos accessible and boost SEO.
    - **💡 Viral Shorts Ideas:** Get precise timestamps and summaries for engaging clips.
    - **✍️ Full-Length Blog Post:** A well-structured article ready for your website.
    - **📱 Engaging Social Post:** A perfectly crafted post to drive traffic.
    - **🎨 Clickable Thumbnail Ideas:** AI-generated prompts for eye-catching visuals.
    """)

    st.divider()

    st.subheader("How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div style='text-align: center;'><h3>1. Upload</h3><p>Select the <b>Creator Tool</b> from the sidebar and upload your video file.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='text-align: center;'><h3>2. Analyze</h3><p>Click one button to let our AI agent perform a deep analysis of your content.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div style='text-align: center;'><h3>3. Create</h3><p>Instantly access, download, and enhance all your new content assets.</p></div>", unsafe_allow_html=True)

def creator_tool_page():
    """Renders the main tool for video analysis and content generation."""
    st.title("🛠️ Creator Catalyst Tool")
    st.markdown("##### Upload your video to begin the analysis and content generation process.")

    # Initialize session state variables
    if 'results' not in st.session_state: st.session_state.results = {}
    if 'video_path' not in st.session_state: st.session_state.video_path = None
    if 'enhanced_tweet' not in st.session_state: st.session_state.enhanced_tweet = ""

    # --- Main Upload and Analysis Section ---
    uploaded_file = st.file_uploader("Upload your video file", type=['mp4','mov','webm','mkv'])

    if uploaded_file:
        temp_dir = tempfile.gettempdir()
        video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(video_path, "wb") as f: f.write(uploaded_file.getbuffer())
        st.session_state.video_path = video_path
        
        st.video(video_path)

        if st.button("Analyze Video & Generate All Content", type="primary", use_container_width=True):
            with st.spinner("🚀 Launching AI Agent... This can take several minutes."):
                gemini_file = upload_file_with_retries(path=video_path)
                if gemini_file:
                    while gemini_file.state.name == "PROCESSING":
                        time.sleep(5)
                        gemini_file = genai.get_file(gemini_file.name)
                    if gemini_file.state.name == "ACTIVE":
                        st.session_state.results = process_video_with_gemini(gemini_file)
                        if "error" not in st.session_state.results:
                            st.success("✅ Full analysis complete!")
                    else:
                        st.error(f"Gemini failed to process the video. Status: {gemini_file.state.name}")

    # --- Display Results ---
    if st.session_state.results and 'error' not in st.session_state.results:
        results = st.session_state.results
        
        tabs = st.tabs(["🎧 Captions", "✂️ Shorts Ideas", "📝 Blog Post", "📱 Social Media", "🎨 Thumbnails"])

        with tabs[0]:
            st.header("Captions (SRT)")
            st.text_area("Transcript", results.get('captions', "No captions."), height=400)

        with tabs[1]:
            st.header("Short Clip Ideas")
            for i, short in enumerate(results.get('shorts_ideas', [])):
                with st.container(border=True):
                    st.subheader(f"Idea {i+1}: {short.get('topic', 'N/A')}")
                    st.markdown(f"**Timestamps:** `{short.get('start_time', 'N/A')} - {short.get('end_time', 'N/A')}`")
                    st.markdown(f"**Summary:** {short.get('summary', 'N/A')}")
                    
                    if st.button(f"Prepare Clip {i+1}", key=f"clip_{i}"):
                        with st.spinner("Clipping with ffmpeg..."):
                            clip_path = clip_video_ffmpeg(st.session_state.video_path, short.get('start_time'), short.get('end_time'))
                            if clip_path: st.session_state[f"clip_path_{i}"] = clip_path
                    
                    if f"clip_path_{i}" in st.session_state:
                        with open(st.session_state[f"clip_path_{i}"], "rb") as file:
                            st.download_button(label=f"Download Clip {i+1}", data=file, file_name=os.path.basename(st.session_state[f"clip_path_{i}"]), mime="video/mp4")

        with tabs[2]:
            st.header("Blog Post")
            st.markdown(results.get('blog_post', 'No blog post.'))

        with tabs[3]:
            st.header("Social Media Post")
            original_tweet = results.get('social_post', 'No post generated.')
            
            if 'enhanced_tweet' not in st.session_state or not st.session_state.enhanced_tweet:
                st.session_state.enhanced_tweet = original_tweet

            st.markdown(f"> {st.session_state.enhanced_tweet}")
            
            if st.button("✨ Enhance Post", key="enhance_tweet"):
                with st.spinner("Asking Gemini to spice this up..."):
                    st.session_state.enhanced_tweet = enhance_tweet_with_gemini(original_tweet)
                    st.rerun()

        with tabs[4]:
            st.header("Thumbnail Ideas")
            for i, idea in enumerate(results.get('thumbnail_ideas', [])):
                with st.container(border=True):
                    st.subheader(f"Idea {i+1}")
                    st.markdown(f"*{idea}*")
                    ref_img_file = st.file_uploader("Upload Reference Image (Optional)", type=['png','jpg','jpeg'], key=f"ref_{i}")
                    
                    if st.button(f"Generate Thumbnail {i+1}", key=f"gen_thumb_{i}"):
                        with st.spinner("Generating thumbnail with SDXL..."):
                            ref_img = Image.open(ref_img_file) if ref_img_file else None
                            pil_img = generate_thumbnail_hf(idea, reference_image=ref_img)
                            if pil_img:
                                st.session_state[f"pil_image_{i}"] = pil_img
                    
                    if f"pil_image_{i}" in st.session_state:
                        image_data = st.session_state[f"pil_image_{i}"]
                        st.image(image_data, caption=f"Generated Thumbnail {i+1}")
                        st.download_button(f"Download Thumbnail {i+1}", data=pil_to_bytes(image_data), file_name=f"thumbnail_{i+1}.png", mime="image/png")

# --- Main App Router (in the sidebar) ---

# Custom CSS for the sidebar
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #1a1032;
    }
    [data-testid="stSidebar"] h2 {
        background: -webkit-linear-gradient(45deg, #a855f7, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🚀 Creator Catalyst")
    page = st.radio(
        "Navigation",
        ["Home", "Creator Tool"],
        label_visibility="hidden"
    )

if page == "Home":
    home_page()
elif page == "Creator Tool":
    # Inject the main app CSS here, so it only applies to the tool page
    st.markdown("""
    <style>
        /* Main App Styling */
        .stApp {
            background-color: #0c001f;
            color: #e0e0e0;
        }
        
        /* Headers and Titles */
        h1, h2, h3, h4, h5, h6 {
            background: -webkit-linear-gradient(45deg, #6d28d9, #4f46e5, #2563eb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        
        h1 {
            border-bottom: 2px solid #4f46e5;
            padding-bottom: 10px;
        }
        
        /* Buttons */
        .stButton > button {
            border: 2px solid #4f46e5;
            border-radius: 10px;
            color: #e0e0e0;
            background-image: linear-gradient(45deg, #371a7a, #2b268f);
            transition: all 0.2s ease-in-out;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4);
            border-color: #6d28d9;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-image: linear-gradient(90deg, #371a7a, #2b268f);
            color: white;
            font-weight: bold;
        }

        /* Containers */
        [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] {
            background-color: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* File Uploader */
        .stFileUploader {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 20px;
        }

    </style>
    """, unsafe_allow_html=True)
    creator_tool_page()