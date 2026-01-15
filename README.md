# Creator Catalyst

Creator Catalyst is an AI-powered tool designed to solve the primary challenge in the creator economy: content repurposing.

This application takes a single long-form video and automatically transforms it into a complete suite of ready-to-publish assets. It effectively turns one piece of content into a comprehensive marketing campaign in minutes, with built-in hallucination prevention and persistent storage.

**Hackathon Recognition**
This project was a jury's pick for the AI Demos Hackathon: AI for Content Creators.

---

## 🚀 Key Features

Creator Catalyst analyzes your video and generates a multi-format content strategy automatically:

### 🧠 Intelligent Content Generation
* **Accurate SRT Captions**: Generates complete, properly formatted .srt transcripts.
* **Viral Shorts Ideas**: Identifies high-impact moments with topics, timestamps, and summaries.
    * **Auto-Clipping**: Automatically clips and downloads selected shorts using FFmpeg.
* **Platform-Optimized Writing**: Generates Blog Posts and Social Media content tailored to specific platform tones (YouTube, LinkedIn, Twitter/X, Instagram, TikTok).
* **Thumbnail Generation**: Creates detailed prompts and generates actual images using Stable Diffusion.

### 🛡️ Fact-Grounding & Validation (New!)
* **Hallucination Prevention**: Automatically verifies every AI-generated claim against the video transcript.
* **Citation System**: Adds precise timestamps `[Source: MM:SS]` to factual claims.
* **Verification Reports**: Provides "Grounding Rates" for blogs, social posts, and shorts to ensure accuracy.

### 💾 Persistence & Management (New!)
* **Local Database**: Automatically saves all videos, analysis results, and generated content to a local SQLite database.
* **History Browser**: View, search, and export past projects via the "History" tab.
* **JSON Export/Import**: easily backup or transfer analysis results.

### 💳 Credits System (New!)
* **Usage Tracking**: A built-in virtual economy to manage API usage and simulate a SaaS environment.
* **Cost Management**: Track spending for video uploads, content generation, and enhancements.

---

## 🛠️ Tech Stack & Architecture

Creator Catalyst utilizes a modern, multi-modal architecture:

### AI & Machine Learning
* **Video Analysis**: **Google Gemini 2.0 Flash Exp** (Primary) for state-of-the-art multimodal video understanding.
* **Fallback Text Generation**: OpenAI GPT-4 or Ollama (Llama 3) for reliability when the primary provider is offline.
* **Image Generation**: Stability AI (via Hugging Face) for high-quality thumbnail creation.

### Core Technologies
* **Interface**: Streamlit for the interactive web-based UI.
* **Database**: SQLite for local persistence of video data and user credits.
* **Processing**: FFmpeg for precise server-side video clipping.
* **Language**: Python 3.9+

---

## ⚡ Quick Local Deployment

Get Creator Catalyst running locally with these simple steps:

### Prerequisites

* Python 3.9+
* FFmpeg (Required for video clipping)
* Git

### Step 1: Clone the Repository

```bash
git clone [https://github.com/garvit-010/Creator-Catalyst.git](https://github.com/garvit-010/Creator-Catalyst.git)
cd Creator-Catalyst
Step 2: Create Python Environment

Bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
Step 3: Install Dependencies

Bash
pip install -r CodeBase/requirements.txt
Step 4: Configure API Keys

Create a .env.local file in the project root:

Bash
# Copy example template
cp CodeBase/.env.example .env.local
Edit .env.local with your API keys:

Properties
# Required for Video Analysis
GOOGLE_API_KEY="your_google_api_key_here"

# Required for Image Generation
HF_TOKEN="your_huggingface_token_here"

# Optional Fallbacks
OPENAI_API_KEY="your_openai_key_here"
# OR for local Ollama:
# USE_OLLAMA="true"
Step 5: Launch the Application

Bash
streamlit run CodeBase/app.py
✅ Success: Open http://localhost:8501 in your browser!

💻 CLI Tools
Creator Catalyst includes a robust Command Line Interface (CLI) for managing your database and credits without opening the web UI.

Location: CodeBase/db_cli.py

Common Commands

View Database Stats

Bash
python CodeBase/db_cli.py stats
List Processed Videos

Bash
python CodeBase/db_cli.py list
Search Videos

Bash
python CodeBase/db_cli.py search "tutorial"
Manage Credits

Bash
# Check balance
python CodeBase/db_cli.py credits-balance

# Add credits (Admin)
python CodeBase/db_cli.py credits-add 100 -m "Bonus credits"
💰 Credit Economy
The application uses a simulated credit system to track resource usage:

Operation	Cost	Description
Video Upload	5 credits	Full multimodal analysis & grounding
Blog Post	2 credits	SEO-optimized article generation
Social Post	1 credit	Platform-specific post generation
Shorts Clip	1 credit	FFmpeg video clipping
Thumbnail	1 credit	AI image generation
Enhancement	1 credit	Post refinement
Note: New users start with 50 free credits.

License
MIT License