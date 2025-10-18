# 🚀 Creator Catalyst

Tired of the content grind? **Creator Catalyst** is your AI-powered co-pilot designed to solve the biggest challenge in the creator economy: content repurposing.  

This tool takes a single long-form video and, in minutes, automatically transforms it into a complete suite of ready-to-publish assets, turning one piece of content into a full-blown marketing campaign.

📍 **Hackathon Winner**  
This project was jury's pick for the [**AI Demos Hackathon: AI for Content Creators**](https://www.linkedin.com/posts/pradipnichite_ai-demos-community-activity-7381660078497497088-Xuhm?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFSO_jUBIMv0FZzJL-pdGOrezAtOwMnSo_A).

---

## 🌟Key Features

Creator Catalyst analyzes your video and generates a multi-format content strategy automatically:

- 🎙️ **Accurate SRT Captions**: Generates a complete, properly formatted `.srt` transcript to make your videos accessible and boost SEO.  
- ✂️ **Viral Shorts Ideas**: Identifies at least five high-impact moments from your video, providing topics, timestamps, and summaries. You can then clip and download these shorts directly from the app.  
- 📝 **Full-Length Blog Post**: Writes a well-structured, engaging blog article (300–400 words) based on the video's content, ready for your website.  
- 📱 **Engaging Social Post**: Crafts a concise and catchy post, complete with hashtags, perfect for sharing on platforms like X (Twitter) or LinkedIn.  
- 🎨 **Clickable Thumbnails**: Generates three distinct, visually descriptive prompts for thumbnails. You can then generate these images with a single click, optionally providing a reference image to guide the AI's style.  

---

## 📹 Video Tutorial

Watch the complete setup and usage tutorial:  
[▶️ Creator Catalyst Tutorial](https://drive.google.com/file/d/1A-Wncmv-4AXm1_OfsrznVbruCf92Xb5L/view?usp=sharing)

---

## 🛠️ Tech Stack & Architecture

As an AI/ML project, Creator Catalyst uses a modern, multi-modal architecture to achieve its results.  

### AI & ML Models
- **Video Analysis & Text Generation**: Powered by *Google Gemini 2.5 Flash* (`gemini-2.5-flash-preview-05-20`) for high-speed, long-context video analysis and text-based content generation.  
- **Image Generation**: Built on *Google Gemini 1.5 Pro* (`gemini-1.5-pro-latest`) for high-quality multimodal thumbnail generation.  

### Core Technologies
- **Framework**: Streamlit for the interactive, web-based user interface.  
- **Video Clipping**: FFMPEG for robust and reliable server-side video processing.  
- **Language**: Python 3.9+  

---

## ⚙️ How It Works

1. **Upload**: Navigate to the "Creator Tool" and upload your long-form video file.  
2. **Analyze**: Click a single button to launch the AI agent. The app uploads your video and sends it to Gemini for a comprehensive analysis.  
3. **Create**: Explore the generated content in organized tabs. Download your captions, clip your favorite shorts, and generate thumbnails instantly.  

---

## 🔧 Local Setup and Installation

Follow these steps to run **Creator Catalyst** on your local machine:  

### 1. Prerequisites
- Python 3.9+ installed  
- FFMPEG installed and accessible in your system's PATH  

**Installation Guides:**  
- Windows: [How to Install FFmpeg on Windows](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/)  
- macOS: `brew install ffmpeg`  

### 2. Clone the Repository
```
git clone https://github.com/garvit-010/Creator-Catalyst.git
cd creator-catalyst
```

### 3. Set Up a Virtual Environment
```
# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies
```
pip install -r requirements.txt
```

### 5. Add API Keys
- Create a new file in the root directory named `.env`  
- Copy the contents of `.env.example` into your new `.env` file  
- Add your secret keys:
```
GOOGLE_API_KEY="AIzaSy..."
```

### 6. Run the Application
```
streamlit run app.py
```

👉 Open the "Network URL" provided by Streamlit in your browser to start using **Creator Catalyst**!

---

## Local development & run (clone and run locally)
1. Clone the repo:
   ```
   git clone <your-repo-url>
   cd Creator-Catalyst
   ```

2. Create a `.env.local` in the project root (do NOT commit this file). Example:
   ```
   GOOGLE_API_KEY=your_google_key_here
   HF_TOKEN=your_huggingface_token_here
   ```

3. Create a venv and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate     # Windows
   source .venv/bin/activate  # macOS / Linux
   pip install -r CodeBase/requirements.txt
   ```

4. Run the app:
   ```
   streamlit run CodeBase/app.py
   ```

**Notes:**
- Keep `.env.local` out of version control.
- If you previously added Streamlit Cloud secrets, remove them while testing locally.

