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

## Installation and Setup

Follow these steps to run Creator Catalyst on your local machine.

### 1. Prerequisites
* Python 3.9 or higher installed.
* FFmpeg installed and accessible in your system's PATH.

### 2. Clone the Repository
```bash
git clone https://github.com/garvit-010/Creator-Catalyst.git
cd creator-catalyst

```

### 3. Set Up a Virtual Environment (Recommended)

It is highly recommended to use a virtual environment to manage dependencies.
```bash

# Create the environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```

### 4. Install Dependencies

```bash
pip install -r CodeBase/requirements.txt

```

### 5. Configure API Keys

Create a file named `.env.local` in the project root directory. You can use the example below as a template:

```properties
# Primary Provider (Required)
GOOGLE_API_KEY="your_google_api_key_here"

# Image Generation (Required for Thumbnails)
HF_TOKEN="your_huggingface_token_here"

# Fallback Providers (Optional but Recommended)
OPENAI_API_KEY="your_openai_key_here"
# OR for local Ollama:
# USE_OLLAMA="true"
# OLLAMA_BASE_URL="http://localhost:11434/v1"

```

### 6. Run the Application

Use the following command to start the application:

```bash
python -m streamlit run CodeBase/app.py

```

Once running, open the "Network URL" provided in your terminal (usually http://localhost:8501) to start using the tool.

---

## License

MIT License

```

```