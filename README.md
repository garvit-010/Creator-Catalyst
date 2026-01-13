# Creator Catalyst

Creator Catalyst is an AI-powered tool designed to solve the primary challenge in the creator economy: content repurposing.

This application takes a single long-form video and automatically transforms it into a complete suite of ready-to-publish assets. It effectively turns one piece of content into a comprehensive marketing campaign in minutes.

**Hackathon Recognition**
This project was a jury's pick for the AI Demos Hackathon: AI for Content Creators.

---

## Key Features

Creator Catalyst analyzes your video and generates a multi-format content strategy automatically:

* **Accurate SRT Captions**: Generates a complete, properly formatted .srt transcript to make your videos accessible and improve SEO.
* **Viral Shorts Ideas**: Identifies high-impact moments from your video, providing topics, timestamps, and summaries. You can then clip and download these shorts directly from the application.
* **Full-Length Blog Post**: Writes a well-structured blog article (300–400 words) based on the video's content, ready for your website.
* **Social Media Posts**: Crafts concise and engaging posts, complete with hashtags, optimized for platforms like X (Twitter) or LinkedIn.
* **Thumbnail Generation**: Generates distinct, visually descriptive prompts for thumbnails and allows you to generate the actual images using Stable Diffusion with a single click.
* **Robust AI Fallback**: Built for reliability. The system prioritizes Google Gemini for analysis but automatically falls back to OpenAI or a local Ollama instance if the primary API is unavailable.

---

## Tech Stack & Architecture

Creator Catalyst utilizes a modern, multi-modal architecture to achieve its results.

### AI & Machine Learning
* **Video Analysis**: Google Gemini 2.5 Flash (Primary) for high-speed, long-context video analysis.
* **Fallback Text Generation**: OpenAI GPT-4 or Ollama (Llama 3) for reliability when the primary provider is offline.
* **Image Generation**: Stability AI (via Hugging Face) for high-quality thumbnail creation.

### Core Technologies
* **Interface**: Streamlit for the interactive web-based UI.
* **Processing**: FFmpeg for server-side video clipping and manipulation.
* **Language**: Python 3.9+

---

## Quick Local Deployment 

Get Creator Catalyst running locally with these simple steps:

### Prerequisites Check

Verify you have everything needed:

```bash
python --version    # Should show Python 3.9+
ffmpeg -version     # Should show FFmpeg version
git --version       # Should show Git version
```
#### Step 1: Clone the Repository

```bash
git clone https://github.com/garvit-010/Creator-Catalyst.git
cd Creator-Catalyst
```

#### Step 2: Create Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install -r CodeBase/requirements.txt
```

#### Step 4: Configure API Keys

Create  .env.local  file in project root (same level as README.md):

```bash
# If .env.example exists:
cp CodeBase/.env.example .env.local
# OR create manually:
touch .env.local
```

Edit  .env.local  with your API keys:

```properties
# Required
GOOGLE_API_KEY="your_google_api_key_here"
HF_TOKEN="your_huggingface_token_here"

# Optional fallbacks
OPENAI_API_KEY="your_openai_key_here"
# OR for local Ollama:
# USE_OLLAMA="true"
# OLLAMA_BASE_URL="http://localhost:11434/v1"
```

Step 5: Launch the Application 

```bash
streamlit run CodeBase/app.py
```

✅ Success: Open http://localhost:8501 in your browser!

One-Command Setup (Advanced)

```bash
git clone https://github.com/garvit-010/Creator-Catalyst.git && cd Creator-Catalyst && python -m venv venv && source venv/bin/activate && pip install -r CodeBase/requirements.txt && streamlit run CodeBase/app.py
```

## License

MIT License

```
Copyright (c) 2025 garvit-010

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```