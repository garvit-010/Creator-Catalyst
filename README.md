# Creator Catalyst

**AI-Powered Content Repurposing Platform**

Creator Catalyst is a comprehensive AI-powered tool designed to solve the primary challenge in the creator economy: content repurposing. This application takes a single long-form video and automatically transforms it into a complete suite of ready-to-publish assets, effectively turning one piece of content into a comprehensive marketing campaign in minutes.

**🏆 Hackathon Recognition**  
This project was a jury's pick for the AI Demos Hackathon: AI for Content Creators.

---

## 🚀 Key Features

### 🧠 Intelligent Content Generation
- **Accurate SRT Captions**: Generates complete, properly formatted .srt transcripts with timestamps
- **Viral Shorts Ideas**: Identifies high-impact moments with topics, timestamps, and summaries
  - **Auto-Clipping**: Automatically clips and downloads selected shorts using FFmpeg
- **Platform-Optimized Writing**: Generates Blog Posts and Social Media content tailored to specific platform tones (YouTube, LinkedIn, Twitter/X, Instagram, TikTok)
- **AI Title Generator**: Creates 3 catchy, click-worthy titles for videos and shorts using proven formulas
- **Thumbnail Generation**: Creates detailed prompts and generates actual images using Stable Diffusion

### 🔍 SEO & Discovery
- **Keyword Extraction**: Automatically extracts relevant keywords from all generated content
- **Platform-Specific Optimization**: Keywords optimized for different content types (blog, social, shorts)
- **CSV Export**: Export keywords and content analysis for SEO tools

### 📊 Engagement Analytics
- **Engagement Score Prediction**: AI-powered prediction of content performance (0-100 score)
- **Platform Recommendations**: Suggests optimal platforms for each piece of content
- **Performance Insights**: Detailed breakdown of engagement factors and improvement suggestions
- **Multi-Content Comparison**: Compare engagement scores across blog posts, social posts, and shorts

### 🛡️ Fact-Grounding & Validation 
- **Hallucination Prevention**: Automatically verifies every AI-generated claim against the video transcript
- **Citation System**: Adds precise timestamps `[Source: MM:SS]` to factual claims
- **Verification Reports**: Provides "Grounding Rates" for blogs, social posts, and shorts to ensure accuracy
- **Side-by-Side Comparison**: View filtered (verified) vs original (unfiltered) content

### 💾 Persistence & Management
- **Local Database**: Automatically saves all videos, analysis results, and generated content to a local SQLite database
- **History Browser**: View, search, and export past projects via the "History" tab
- **JSON Export/Import**: Easily backup or transfer analysis results
- **Toolkit ZIP Export**: Download all assets (captions, blog, social, shorts, thumbnails) in one package

### 💳 Credits System 
- **Usage Tracking**: Built-in virtual economy to manage API usage and simulate a SaaS environment
- **Cost Management**: Track spending for video uploads, content generation, and enhancements
- **Transaction History**: Complete audit trail of all credit operations
- **Flexible Packages**: Starter (50), Pro (150), and Business (500) credit packages

### 📈 AI Request Monitoring
- **Comprehensive Logging**: Track all AI API calls with timestamps, costs, and performance metrics
- **Rate Limiting**: Built-in rate limiting to prevent cost overruns (100 requests/hour, 1M tokens/hour)
- **Usage Analytics**: Detailed breakdowns by provider, operation type, and time period
- **Cost Tracking**: Monitor both application credits and estimated USD costs

---

## 🛠️ Tech Stack & Architecture

### AI & Machine Learning
- **Video Analysis**: **Google Gemini 2.0 Flash Exp** (Primary) for state-of-the-art multimodal video understanding
- **Fallback Text Generation**: OpenAI GPT-4 or Ollama (Llama 3) for reliability when the primary provider is offline
- **Image Generation**: Stability AI (via Hugging Face) for high-quality thumbnail creation
- **Engagement Scoring**: Custom AI-powered algorithm analyzing 10+ engagement factors

### Core Technologies
- **Interface**: Streamlit for the interactive web-based UI
- **Database**: SQLite for local persistence with comprehensive schema
  - Videos, content outputs, grounding reports
  - Credits transactions and user management
  - AI request logging and analytics
- **Processing**: FFmpeg for precise server-side video clipping
- **Language**: Python 3.9+

### Architecture Highlights
- **Modular Design**: Separate core logic, database, and UI layers
- **Singleton Patterns**: Efficient resource management for database connections
- **Comprehensive Error Handling**: Try-catch blocks and graceful fallbacks throughout
- **Session State Management**: Streamlit session state for persistent UI interactions

---

## 📁 Project Structure

```
Creator-Catalyst/
├── app/
│   └── app.py                      # Main Streamlit application
├── src/
│   ├── core/                       # Core AI logic
│   │   ├── llm_wrapper.py          # LLM provider abstraction with logging
│   │   ├── fact_grounding.py       # Fact verification system
│   │   ├── engagement_scorer.py    # Engagement prediction algorithm
│   │   ├── keyword_extractor.py    # SEO keyword extraction
│   │   └── title_generator.py      # AI-powered title generation
│   ├── database/                   # Data persistence layer
│   │   ├── database.py             # SQLite schema and base operations
│   │   ├── storage_manager.py      # High-level storage interface
│   │   ├── credits_manager.py      # Credits and billing system
│   │   ├── ai_request_logger.py    # AI usage logging and analytics
│   │   └── csv_exporter.py         # CSV export utilities
│   └── ui/                         # User interface components
│       ├── components/             # Reusable UI components
│       │   ├── engagement_ui.py    # Engagement score displays
│       │   ├── keyword_ui.py       # Keyword badges and sections
│       │   ├── title_ui.py         # Title suggestion interface
│       │   └── theme_manager.py    # Theme switching (light/dark)
│       └── pages/                  # Full page views
│           ├── history.py          # Video history browser
│           ├── credits_page.py     # Credits management
│           └── ai_logs_dashboard.py # AI usage analytics
├── cli/
│   └── db_cli.py                   # Command-line database tools
├── .env.example                    # Environment configuration template
├── requirements.txt                # Python dependencies
├── packages.txt                    # System packages (FFmpeg)
└── README.md                       # This file
```

---

## ⚡ Quick Local Deployment

### Prerequisites

- Python 3.9+
- FFmpeg (Required for video clipping)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/garvit-010/Creator-Catalyst.git
cd Creator-Catalyst
```

### Step 2: Create Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys

Create a `.env.local` file in the project root:

```bash
# Copy example template
cp .env.example .env.local
```

Edit `.env.local` with your API keys:

**Required for Video Analysis:**
```bash
GOOGLE_API_KEY="your_google_api_key_here"
```

**Required for Image Generation:**
```bash
HF_TOKEN="your_huggingface_token_here"
```

**Optional Fallbacks:**
```bash
# OpenAI API (for text generation fallback)
OPENAI_API_KEY="your_openai_key_here"
OPENAI_MODEL="gpt-4o"

# OR use Ollama (local LLM - no API key needed)
USE_OLLAMA="true"
OLLAMA_BASE_URL="http://localhost:11434/v1"
OLLAMA_MODEL="llama3.2"

# Fallback Configuration
ENABLE_FALLBACK="true"
```

### Step 5: Launch the Application

```bash
streamlit run app/app.py
```

✅ **Success**: Open http://localhost:8501 in your browser!

---

## 💻 CLI Tools

Creator Catalyst includes a robust Command Line Interface (CLI) for managing your database and credits without opening the web UI.

**Location**: `cli/db_cli.py`

### Common Commands

**View Database Stats:**
```bash
python cli/db_cli.py stats
```

**List Processed Videos:**
```bash
python cli/db_cli.py list
```

**Search Videos:**
```bash
python cli/db_cli.py search "tutorial"
```

**View Video Details:**
```bash
python cli/db_cli.py show 123
```

**Export Video to JSON:**
```bash
python cli/db_cli.py export 123 -o video.json
```

**Import from JSON:**
```bash
python cli/db_cli.py import video.json
```

**Delete Video:**
```bash
python cli/db_cli.py delete 123 --force
```

### Credits Management

**Check Balance:**
```bash
python cli/db_cli.py credits-balance
```

**Add Credits (Admin):**
```bash
python cli/db_cli.py credits-add 100 -m "Bonus credits"
```

**View Transaction History:**
```bash
python cli/db_cli.py credits-history -l 50
```

**Reset Credits:**
```bash
python cli/db_cli.py credits-reset -a 50 --force
```

### Database Maintenance

**Search Videos:**
```bash
python cli/db_cli.py search "my video"
```

**Recent Activity:**
```bash
python cli/db_cli.py recent -l 20
```

**Cleanup Orphaned Records:**
```bash
python cli/db_cli.py cleanup --force
```

---

## 💰 Credit Economy

The application uses a simulated credit system to track resource usage:

| Operation              | Cost       | Description                                    |
|------------------------|-----------|------------------------------------------------|
| Video Upload           | 5 credits | Full multimodal analysis & grounding           |
| Blog Post Generation   | 2 credits | SEO-optimized article generation               |
| Social Post            | 1 credit  | Platform-specific post generation              |
| Shorts Clip            | 1 credit  | FFmpeg video clipping                          |
| Thumbnail Generation   | 1 credit  | AI image generation                            |
| Tweet Enhancement      | 1 credit  | Post refinement with AI                        |

**Note:** New users start with **50 free credits**.

### Credit Packages

- **🥉 Starter**: 50 credits - $9.99
- **🥈 Pro**: 150 credits - $24.99 (17% savings)
- **🥇 Business**: 500 credits - $79.99 (20% savings)

---

## 🎯 Platform-Specific Optimization

Creator Catalyst adapts content style based on your target platform:

- **YouTube**: Storytelling and engaging narrative style
- **LinkedIn**: Professional and thought-leadership focused
- **Twitter/X**: Punchy, viral-worthy with high energy
- **Instagram**: Visual-first, lifestyle-oriented narrative
- **TikTok**: Fast-paced, trend-aware, Gen-Z friendly
- **General**: Balanced tone suitable for all platforms

---

## 🔍 Fact-Grounding System

### How It Works

1. **Transcript Analysis**: Parses SRT captions into searchable segments
2. **Claim Extraction**: Identifies factual claims in generated content
3. **Evidence Matching**: Matches claims against transcript using word overlap scoring
4. **Timestamp Citations**: Adds `[Source: MM:SS]` citations to verified claims
5. **Filtering**: Removes unverifiable claims (optional strict mode)
6. **Reporting**: Generates comprehensive grounding reports with statistics

### Grounding Metrics

- **Blog Grounding Rate**: % of blog claims verified against transcript
- **Social Grounding Rate**: % of social post claims verified
- **Shorts Verification Rate**: % of shorts ideas with valid timestamps
- **Total vs Verified Claims**: Complete claim audit trail

---

## 📊 Engagement Scoring System

### Analyzed Factors

The engagement scorer evaluates 15+ factors:

- **Brevity**: Optimal length for platform
- **Hashtag Usage**: Quantity and quality of hashtags
- **Emoji Presence**: Strategic emoji usage
- **Call-to-Action**: Clear CTAs that drive engagement
- **Engagement Hooks**: Questions, bold claims, curiosity gaps
- **Storytelling**: Narrative structure and emotional connection
- **Professionalism**: Industry-appropriate tone (LinkedIn)
- **Value Proposition**: Clear benefits and takeaways
- **Viral Potential**: Trending elements and shareability
- **Hook Strength**: Power of opening sentence
- **Visual Appeal**: Formatting and readability

### Score Interpretation

- **85-100**: 🔥 Viral Potential
- **70-84**: 🚀 High Engagement
- **55-69**: 👍 Good Performance
- **40-54**: ⚠️ Moderate Engagement
- **0-39**: ❌ Needs Improvement

---

## 🎬 AI Title Generator

### Title Formulas

The generator uses proven formulas across 6 categories:

1. **Curiosity**: "The Truth About {topic} Nobody Tells You"
2. **Listicle**: "{number} {topic} Tips You Need Right Now"
3. **How-To**: "How to Master {topic} in Minutes"
4. **Secret**: "Secret {topic} Strategy (Explained)"
5. **Urgency**: "Watch This Before You {topic}"
6. **Results**: "{topic} Results in 24 Hours"

### Features

- **AI-Powered**: Uses LLM for contextual title generation
- **Formulaic Fallback**: Rule-based generation when AI unavailable
- **Platform Optimization**: Titles adapted for YouTube, TikTok, Instagram, etc.
- **CTR Estimation**: High/Medium/Low click-through rate predictions
- **Custom Editing**: Select AI suggestions or write your own

---

## 📈 AI Usage Monitoring

### Tracked Metrics

- **Total Requests**: Count of all AI API calls
- **Tokens Consumed**: Total tokens across all operations
- **Credits Spent**: Application credit usage
- **USD Cost Estimation**: Approximate real costs
- **Response Times**: Average latency per operation
- **Success Rate**: % of successful vs failed requests

### Rate Limiting

- **100 requests per hour** (per user)
- **1,000,000 tokens per hour** (per user)
- Automatic enforcement with user-friendly warnings
- Hourly window tracking with reset timers

### Analytics Dashboard

Access comprehensive AI usage analytics:

1. Overview metrics (requests, tokens, costs, success rate)
2. Current rate limit status with visual progress bars
3. Request history with filtering and search
4. Provider breakdown (Gemini, OpenAI, Ollama, HuggingFace)
5. Operation type analysis (video analysis, text generation, etc.)
6. Daily usage trends with charts
7. CSV export for external analysis

---

## 🗄️ Database Schema

### Core Tables

**Videos**
- Metadata, file info, platform, grounding settings, processing status

**Content Outputs**
- Generated content (captions, blog, social, shorts, thumbnails)
- Version tracking, grounding rates, validation status

**Grounding Reports**
- Fact verification statistics and detailed reports

**User Credits**
- Credit balance, total earned, total spent

**Credit Transactions**
- Complete transaction log with descriptions

**AI Requests**
- API call logs with performance metrics

**Rate Limits**
- Hourly usage windows for rate limiting

---

## 🚀 Advanced Features

### Export Options

1. **JSON Export**: Complete video analysis with all content
2. **Toolkit ZIP**: All assets (SRT, blog MD, social TXT, shorts JSON, thumbnails)
3. **CSV Keywords**: SEO-optimized keyword lists
4. **CSV Analytics**: AI usage logs and metrics

### Search & Discovery

- **Filename Search**: Find videos quickly
- **Content Type Filtering**: Filter by blog, social, shorts, etc.
- **Recent Activity**: See latest content generations
- **Statistics Dashboard**: Overall usage metrics

### Version Control

- Multiple versions of each content type
- Original vs grounded content comparison
- Rollback to previous versions
- Version history tracking

---

## 🎨 UI Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Dark/Light Mode**: Persistent theme preferences (via ThemeManager)
- **Progress Tracking**: Real-time progress bars for long operations
- **Interactive Cards**: Expandable content cards with actions
- **Tabbed Navigation**: Organized content in logical tabs
- **Inline Editing**: Edit and customize AI suggestions
- **Download Buttons**: One-click downloads for all content

---

## 🔒 Privacy & Security

- **Local-First**: All data stored locally in SQLite
- **No Cloud Storage**: Videos and content never leave your machine (except API calls)
- **API Key Security**: Keys stored in `.env.local` (gitignored)
- **Rate Limiting**: Prevents excessive API usage and costs
- **Transaction Audit**: Complete log of all credit operations

---

## 🐛 Troubleshooting

### Common Issues

**"Failed to initialize Gemini"**
- Verify `GOOGLE_API_KEY` in `.env.local`
- Check API key is valid and has quota
- Update package: `pip install --upgrade google-generativeai`

**"FFmpeg not found"**
- Install FFmpeg: https://ffmpeg.org/download.html
- On Ubuntu/Debian: `sudo apt-get install ffmpeg`
- On macOS: `brew install ffmpeg`
- On Windows: Download and add to PATH

**"Rate limit exceeded"**
- Wait for hourly window to reset
- Check AI Logs dashboard for usage details
- Consider upgrading API quotas

**"Insufficient credits"**
- Purchase more credits from Credits page
- Or add test credits: `python cli/db_cli.py credits-add 50`

**Database locked errors**
- Close all CLI instances before running app
- Or restart the application

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support

For issues, questions, or feature requests:

- **GitHub Issues**: [Create an issue](https://github.com/garvit-010/Creator-Catalyst/issues)
- **Documentation**: Check this README and code comments
- **CLI Help**: Run `python cli/db_cli.py --help`

---



