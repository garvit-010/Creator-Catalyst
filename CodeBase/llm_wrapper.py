import os
import json
import time
import re
import google.generativeai as genai
from openai import OpenAI, OpenAIError
from pathlib import Path
from dotenv import load_dotenv

# Import fact-grounding system
try:
    from fact_grounding import FactGrounder, create_grounding_prompt_modifier
    GROUNDING_AVAILABLE = True
except ImportError:
    GROUNDING_AVAILABLE = False
    print("⚠️ Fact-grounding module not found. Install fact_grounding.py for validation.")

# Load environment variables
env_path = Path(__file__).parent.parent / '.env.local'
load_dotenv(env_path)

class LLMWrapper:
    """
    Unified interface for LLM providers (Gemini, OpenAI, Ollama) with fallback logic.
    Handles video analysis, text generation, and file uploads with automatic failover.
    Now includes fact-grounding validation to prevent AI hallucinations.
    """
    def __init__(self):
        # API Keys
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
        self.fallback_enabled = os.getenv('ENABLE_FALLBACK', 'true').lower() == 'true'
        
        # Track which provider is currently active
        self.current_provider = None
        
        # Configure Gemini
        self.gemini_model = None
        if self.google_api_key:
            try:
                genai.configure(api_key=self.google_api_key)
                self.gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")
                self.current_provider = "gemini"
                print(f"✅ Gemini initialized successfully (google-generativeai v{genai.__version__})")
            except Exception as e:
                print(f"❌ Failed to initialize Gemini: {e}")
        else:
            print("⚠️ GOOGLE_API_KEY not found in environment")
        
        # Configure OpenAI / Ollama Client
        self.openai_client = None
        self.openai_model = None
        
        if self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o')
                if not self.current_provider:
                    self.current_provider = "openai"
                print(f"✅ OpenAI initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize OpenAI: {e}")
                
        elif os.getenv('USE_OLLAMA', 'false').lower() == 'true':
            try:
                self.openai_client = OpenAI(
                    base_url=self.ollama_base_url,
                    api_key='ollama'  # required but unused
                )
                self.openai_model = os.getenv('OLLAMA_MODEL', 'llama3.2')
                if not self.current_provider:
                    self.current_provider = "ollama"
                print(f"✅ Ollama initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize Ollama: {e}")

    def upload_video_file(self, file_path, retries=3, delay=5):
        """
        Uploads video file to Gemini API with retry logic.
        Returns: (file_object, provider_name) or (None, None) on failure
        """
        if not self.gemini_model:
            print("❌ Gemini not available for video upload")
            return None, None
        
        if not self.google_api_key:
            print("❌ GOOGLE_API_KEY not configured")
            return None, None
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None, None
            
        print(f"📤 Uploading video: {file_path}")
        print(f"   File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
            
        for attempt in range(retries):
            try:
                # Check if the new API is available
                if hasattr(genai, 'upload_file'):
                    # New API (v0.8.0+)
                    print(f"   Attempt {attempt + 1}/{retries} - Using genai.upload_file()")
                    video_file = genai.upload_file(path=file_path)
                else:
                    # Fallback for older versions - use File API directly
                    print(f"   Attempt {attempt + 1}/{retries} - Using legacy File API")
                    from google.generativeai.types import File
                    with open(file_path, 'rb') as f:
                        video_file = File.create(
                            file=f,
                            mime_type='video/mp4'
                        )
                
                print(f"   ✅ Upload successful! File name: {video_file.name}")
                
                # Wait for processing
                print(f"   ⏳ Waiting for processing...")
                max_wait = 300  # 5 minutes max
                wait_time = 0
                
                while video_file.state.name == "PROCESSING":
                    if wait_time >= max_wait:
                        print(f"   ⚠️ Processing timeout after {max_wait}s")
                        return None, None
                    
                    time.sleep(5)
                    wait_time += 5
                    video_file = genai.get_file(video_file.name)
                    print(f"   Processing... ({wait_time}s elapsed, state: {video_file.state.name})")
                
                if video_file.state.name == "ACTIVE":
                    print(f"   ✅ Video ready for analysis!")
                    return video_file, "gemini"
                else:
                    print(f"   ❌ Processing failed with state: {video_file.state.name}")
                    
            except AttributeError as e:
                print(f"   ❌ API Error: {e}")
                print(f"   💡 Suggestion: Update google-generativeai package")
                print(f"      Run: pip install --upgrade google-generativeai")
                return None, None
                
            except Exception as e:
                print(f"   ❌ Upload attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    print(f"   ⏳ Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"   ❌ All upload attempts failed")
                    
        return None, None

    def generate_text(self, prompt, retries=3):
        """
        Generates text with fallback: Gemini -> OpenAI/Ollama -> Mock
        Returns: str (generated text)
        """
        # 1. Try Gemini
        if self.gemini_model:
            for attempt in range(retries):
                try:
                    print(f"🤖 Generating text with Gemini (attempt {attempt + 1}/{retries})")
                    response = self.gemini_model.generate_content(prompt)
                    self.current_provider = "gemini"
                    print(f"   ✅ Success!")
                    return response.text
                except Exception as e:
                    print(f"   ❌ Gemini failed: {e}")
                    if attempt < retries - 1:
                        time.sleep(2)
                    else:
                        print("   ⚠️ Switching to fallback provider...")

        # 2. Try OpenAI / Ollama (Fallback)
        if self.openai_client and self.fallback_enabled:
            try:
                provider_name = "OpenAI" if self.openai_api_key else "Ollama"
                print(f"🤖 Generating text with {provider_name}...")
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=4000
                )
                self.current_provider = "openai" if self.openai_api_key else "ollama"
                print(f"   ✅ Success with {provider_name}!")
                return response.choices[0].message.content
            except Exception as e:
                print(f"   ❌ {provider_name} fallback failed: {e}")

        # 3. Final Fallback (Error message)
        print("❌ All AI providers failed")
        return "⚠️ All AI providers are currently unavailable. Please check your API keys and internet connection, or try again later."

    def analyze_video(self, video_file_obj, prompt, retries=3, enable_grounding=True):
        """
        Analyzes video using Gemini with fallback to mock response.
        Includes fact-grounding validation if enabled.
        
        Args:
            video_file_obj: Gemini file object (from upload_video_file)
            prompt: Analysis instructions
            retries: Number of retry attempts
            enable_grounding: Whether to validate claims against transcript
            
        Returns: dict with parsed content sections and grounding metadata
        """
        # 1. Try Gemini Video Analysis
        if self.gemini_model and video_file_obj:
            for attempt in range(retries):
                try:
                    print(f"🤖 Analyzing video with Gemini (attempt {attempt + 1}/{retries})")
                    
                    # Add grounding instructions to prompt if enabled
                    enhanced_prompt = prompt
                    if enable_grounding and GROUNDING_AVAILABLE:
                        grounding_instructions = """

FACT-GROUNDING REQUIREMENT:
- Every factual claim MUST be verifiable in the video transcript
- Cite timestamps for all statistics, quotes, and specific details
- Format: "claim here [Source: MM:SS]"
- DO NOT include information not present in the video
- If uncertain, omit the claim rather than guess
"""
                        enhanced_prompt = prompt + grounding_instructions
                    
                    response = self.gemini_model.generate_content(
                        [video_file_obj, enhanced_prompt],
                        request_options={"timeout": 600}
                    )
                    self.current_provider = "gemini"
                    print(f"   ✅ Analysis complete!")
                    
                    # Parse response
                    parsed_results = self._parse_response(response.text)
                    
                    # Apply fact-grounding validation if enabled
                    if enable_grounding and GROUNDING_AVAILABLE and parsed_results.get('captions'):
                        print(f"   🔍 Validating claims against transcript...")
                        parsed_results = self._apply_fact_grounding(parsed_results)
                    
                    return parsed_results
                    
                except Exception as e:
                    print(f"   ❌ Analysis attempt {attempt + 1}/{retries} failed: {e}")
                    if attempt < retries - 1:
                        time.sleep(5)
                    else:
                        print("   ❌ All analysis attempts failed")
        
        # 2. OpenAI/Ollama cannot process video directly, so we skip to mock
        # (Future enhancement: implement frame extraction + image analysis)
        
        # 3. Mock Fallback (if enabled)
        if self.fallback_enabled:
            print("⚠️ Using mock analysis response (video analysis requires Gemini)")
            return self._generate_mock_analysis()
            
        return {
            "error": "Video analysis failed. Gemini API is required for video processing.",
            "captions": "",
            "shorts_ideas": [],
            "blog_post": "",
            "social_post": "",
            "thumbnail_ideas": []
        }

    def _parse_response(self, text):
        """Parses the structured markdown response into a dictionary."""
        results = {}
        
        # Extract Captions (SRT format)
        captions_match = re.search(r"### Captions\s*\n```srt\s*\n(.*?)```", text, re.DOTALL)
        if captions_match:
            results['captions'] = captions_match.group(1).strip()
        else:
            # Fallback pattern without code block
            captions_match = re.search(r"### Captions\s*\n(.*?)(?=\n###|$)", text, re.DOTALL)
            if captions_match:
                captions_text = captions_match.group(1).strip()
                results['captions'] = captions_text.removeprefix("```srt").removesuffix("```").strip()

        # Extract Shorts Ideas
        shorts_match = re.search(r"### Shorts Ideas\s*\n(.*?)(?=\n###|$)", text, re.DOTALL)
        if shorts_match:
            shorts_text = shorts_match.group(1).strip()
            ideas = []
            # Split by numbered items (1., 2., etc.)
            for idea_block in re.split(r'\n\s*\d+\.\s*', shorts_text):
                if not idea_block.strip(): 
                    continue
                    
                topic_match = re.search(r"Topic:\s*(.*?)(?=\n|$)", idea_block)
                start_match = re.search(r"Start Time:\s*([\d:]+)", idea_block)
                end_match = re.search(r"End Time:\s*([\d:]+)", idea_block)
                summary_match = re.search(r"Summary:\s*(.*?)(?=\n\n|\Z)", idea_block, re.DOTALL)
                
                if topic_match and start_match and end_match and summary_match:
                    ideas.append({
                        "topic": topic_match.group(1).strip(),
                        "start_time": start_match.group(1).strip(),
                        "end_time": end_match.group(1).strip(),
                        "summary": summary_match.group(1).strip()
                    })
            results['shorts_ideas'] = ideas

        # Extract Blog Post
        blog_match = re.search(r"### Blog Post\s*\n(.*?)(?=\n###|$)", text, re.DOTALL)
        if blog_match:
            results['blog_post'] = blog_match.group(1).strip()

        # Extract Social Media Post
        social_match = re.search(r"### Social Media Post\s*\n(.*?)(?=\n###|$)", text, re.DOTALL)
        if social_match:
            results['social_post'] = social_match.group(1).strip()

        # Extract Thumbnail Ideas
        thumb_match = re.search(r"### Thumbnail Ideas\s*\n(.*?)(?=\n###|$)", text, re.DOTALL)
        if thumb_match:
            thumb_text = thumb_match.group(1).strip()
            ideas = []
            for idea in re.split(r'\n\s*\d+\.\s*', thumb_text):
                if idea.strip():
                    ideas.append(idea.strip())
            results['thumbnail_ideas'] = ideas

        return results

    def _apply_fact_grounding(self, parsed_results: dict) -> dict:
        """
        Apply fact-grounding validation to parsed results.
        Filters ungrounded claims and adds validation metadata.
        
        Args:
            parsed_results: Dictionary with parsed content sections
            
        Returns:
            Enhanced results with grounding validation
        """
        if not GROUNDING_AVAILABLE:
            return parsed_results
        
        try:
            # Initialize fact grounder with transcript
            srt_content = parsed_results.get('captions', '')
            if not srt_content:
                print("   ⚠️ No transcript available for grounding")
                return parsed_results
            
            grounder = FactGrounder(srt_content)
            
            # Generate grounding report
            grounding_report = grounder.generate_grounding_report(parsed_results)
            
            # Replace original content with filtered versions
            if 'blog_post' in grounding_report['filtered_content']:
                original_blog = parsed_results.get('blog_post', '')
                filtered_blog = grounding_report['filtered_content']['blog_post']
                
                # Only replace if filtering actually removed content
                if len(filtered_blog) < len(original_blog) * 0.5:
                    print(f"   ⚠️ Blog post heavily filtered ({len(filtered_blog)}/{len(original_blog)} chars)")
                
                parsed_results['blog_post'] = filtered_blog
                parsed_results['blog_post_original'] = original_blog
            
            if 'social_post' in grounding_report['filtered_content']:
                parsed_results['social_post'] = grounding_report['filtered_content']['social_post']
            
            if 'shorts_ideas' in grounding_report['filtered_content']:
                validated_shorts = grounding_report['filtered_content']['shorts_ideas']
                # Add validation badges to shorts
                for short in validated_shorts:
                    status = short.get('validation_status', 'unknown')
                    if status == 'verified':
                        short['validation_badge'] = '✅ Verified'
                    elif status == 'unverified_summary':
                        short['validation_badge'] = '⚠️ Summary needs review'
                    else:
                        short['validation_badge'] = '❌ Invalid timestamps'
                
                parsed_results['shorts_ideas'] = validated_shorts
            
            # Add grounding statistics
            parsed_results['grounding_metadata'] = {
                'enabled': True,
                'blog_grounding_rate': grounding_report['statistics'].get('blog_grounding_rate', 0),
                'social_grounding_rate': grounding_report['statistics'].get('social_grounding_rate', 0),
                'shorts_verification_rate': grounding_report['statistics'].get('shorts_verification_rate', 0),
                'full_report': grounding_report
            }
            
            # Log validation results
            stats = grounding_report['statistics']
            print(f"   📊 Grounding Stats:")
            print(f"      Blog: {stats.get('blog_grounding_rate', 0):.1%} claims verified")
            print(f"      Social: {stats.get('social_grounding_rate', 0):.1%} claims verified")
            print(f"      Shorts: {stats.get('shorts_verification_rate', 0):.1%} ideas verified")
            
            return parsed_results
            
        except Exception as e:
            print(f"   ⚠️ Fact-grounding validation failed: {e}")
            return parsed_results

    def _generate_mock_analysis(self):
        """Generates a realistic mock response when API is unavailable."""
        return {
            "captions": """1
00:00:01,000 --> 00:00:05,000
[Mock Output] AI provider is currently unavailable.

2
00:00:05,000 --> 00:00:10,000
This is a placeholder transcript for testing purposes.

3
00:00:10,000 --> 00:00:15,000
Please check your GOOGLE_API_KEY in .env.local file.

4
00:00:15,000 --> 00:00:20,000
Or update google-generativeai package: pip install --upgrade google-generativeai""",
            
            "shorts_ideas": [
                {
                    "topic": "API Configuration Issue",
                    "start_time": "00:10",
                    "end_time": "00:30",
                    "summary": "This mock clip indicates that your Gemini API is not properly configured. Check your API key and package version."
                },
                {
                    "topic": "Package Update Required",
                    "start_time": "00:45",
                    "end_time": "01:05",
                    "summary": "The google-generativeai package needs to be updated to version 0.8.0 or higher to support video uploads."
                },
                {
                    "topic": "Fallback System Active",
                    "start_time": "01:20",
                    "end_time": "01:40",
                    "summary": "This demonstrates the app's fallback mechanism working as designed when the primary AI service is unavailable."
                }
            ],
            
            "blog_post": """# AI Provider Configuration Required

## What Happened?
Your video upload to the Gemini API failed. This is typically caused by one of these issues:

### 1. Outdated Package Version
The `google-generativeai` package version in your requirements.txt (0.3.2) is too old. The `upload_file()` method was added in version 0.8.0.

**Fix:**
```bash
pip install --upgrade google-generativeai
```

### 2. Missing or Invalid API Key
Check that your `GOOGLE_API_KEY` is properly set in your `.env.local` file.

**Fix:**
1. Go to https://makersuite.google.com/app/apikey
2. Create a new API key
3. Add it to your `.env.local` file

### 3. Network or API Issues
Sometimes Google's API experiences temporary outages or rate limiting.

**Fix:** Wait a few minutes and try again.

## Next Steps
1. Update your package: `pip install --upgrade google-generativeai`
2. Verify your API key is valid
3. Try uploading your video again

*This is a mock response to help you diagnose the issue.*""",
            
            "social_post": "🔧 Pro tip: Keep your AI packages updated! Just upgraded google-generativeai and now my video analysis is running smooth. #DevLife #AITools #TechTips 🚀",
            
            "thumbnail_ideas": [
                "A frustrated developer looking at a computer screen showing 'API Error', dramatic red lighting, cyberpunk aesthetic",
                "A giant 'UPDATE' button glowing in neon green with sparkles and confetti around it, minimalist tech background",
                "Split screen: left side showing old code with bugs, right side showing updated code running smoothly with checkmarks"
            ]
        }
    
    def get_current_provider(self):
        """Returns the name of the currently active provider."""
        return self.current_provider or "none"