import os
import json
import time
import re
import logging
import google.generativeai as genai
from openai import OpenAI, OpenAIError
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Initialize logger
logger = logging.getLogger(__name__)

# Import Anthropic for Claude support
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic package not installed. Install with: pip install anthropic")

# Import fact-grounding system
try:
    from src.core.fact_grounding import FactGrounder, create_grounding_prompt_modifier
    GROUNDING_AVAILABLE = True
except ImportError:
    GROUNDING_AVAILABLE = False
    logger.warning("Fact-grounding module not found. Install fact_grounding.py for validation.")

# Import AI request logger
try:
    from src.database.ai_request_logger import get_ai_logger
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    logger.warning("AI request logger not found. Logging disabled.")

# Load environment variables
env_path = Path(__file__).parent.parent / '.env.local'
load_dotenv(env_path)

class LLMWrapper:
    """
    Unified interface for LLM providers (Gemini, OpenAI, Claude, Ollama) with fallback logic.
    Now includes comprehensive request logging, rate limiting, and model switching.
    """
    
    # Supported models per provider
    SUPPORTED_MODELS = {
        'gemini': ['gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'],
        'openai': ['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo'],
        'claude': ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229', 'claude-3-haiku-20240307'],
        'ollama': ['llama3.2', 'mistral', 'codellama']
    }
    
    def __init__(self):
        # API Keys
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
        self.fallback_enabled = os.getenv('ENABLE_FALLBACK', 'true').lower() == 'true'
        
        # Primary model selection from ENV
        self.primary_provider = os.getenv('PRIMARY_AI_MODEL', 'gemini').lower()
        
        # Initialize logger
        self.logger = get_ai_logger() if LOGGING_AVAILABLE else None
        
        # Track which provider/model is currently active
        self.current_provider = None
        self.current_model = None
        
        # Store last used model info for display
        self.last_used_provider = None
        self.last_used_model = None
        
        # Configure Gemini
        self.gemini_model = None
        self.gemini_model_name = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')
        if self.google_api_key:
            try:
                genai.configure(api_key=self.google_api_key)
                self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
                if self.primary_provider == "gemini" or not self.current_provider:
                    self.current_provider = "gemini"
                    self.current_model = self.gemini_model_name
                logger.info(f"Gemini initialized successfully (model: {self.gemini_model_name})")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
        else:
            logger.warning("GOOGLE_API_KEY not found in environment")
        
        # Configure Claude/Anthropic
        self.claude_client = None
        self.claude_model_name = os.getenv('CLAUDE_MODEL', 'claude-3-5-sonnet-20241022')
        if self.anthropic_api_key and ANTHROPIC_AVAILABLE:
            try:
                self.claude_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                if self.primary_provider == "claude" or (not self.current_provider and self.primary_provider == "claude"):
                    self.current_provider = "claude"
                    self.current_model = self.claude_model_name
                logger.info(f"Claude initialized successfully (model: {self.claude_model_name})")
            except Exception as e:
                logger.error(f"Failed to initialize Claude: {e}")
        elif self.anthropic_api_key and not ANTHROPIC_AVAILABLE:
            logger.warning("ANTHROPIC_API_KEY found but anthropic package not installed")
        
        # Configure OpenAI / Ollama Client
        self.openai_client = None
        self.openai_model = None
        
        if self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o')
                if self.primary_provider == "openai" or (not self.current_provider and self.primary_provider == "openai"):
                    self.current_provider = "openai"
                    self.current_model = self.openai_model
                logger.info(f"OpenAI initialized successfully (model: {self.openai_model})")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
                
        elif os.getenv('USE_OLLAMA', 'false').lower() == 'true':
            try:
                self.openai_client = OpenAI(
                    base_url=self.ollama_base_url,
                    api_key='ollama'
                )
                self.openai_model = os.getenv('OLLAMA_MODEL', 'llama3.2')
                if self.primary_provider == "ollama" or not self.current_provider:
                    self.current_provider = "ollama"
                    self.current_model = self.openai_model
                logger.info(f"Ollama initialized successfully (model: {self.openai_model})")
            except Exception as e:
                logger.error(f"Failed to initialize Ollama: {e}")
        
        # Set to primary provider if available
        self._set_primary_provider()
    
    def _set_primary_provider(self):
        """Set the current provider based on PRIMARY_AI_MODEL setting."""
        if self.primary_provider == "gemini" and self.gemini_model:
            self.current_provider = "gemini"
            self.current_model = self.gemini_model_name
        elif self.primary_provider == "claude" and self.claude_client:
            self.current_provider = "claude"
            self.current_model = self.claude_model_name
        elif self.primary_provider == "openai" and self.openai_client and self.openai_api_key:
            self.current_provider = "openai"
            self.current_model = self.openai_model
        elif self.primary_provider == "ollama" and self.openai_client:
            self.current_provider = "ollama"
            self.current_model = self.openai_model
    
    def switch_provider(self, provider: str, model: str = None) -> bool:
        """
        Switch to a different AI provider/model at runtime.
        Returns True if switch was successful, False otherwise.
        """
        provider = provider.lower()
        
        if provider == "gemini":
            if not self.gemini_model:
                logger.error(f"Cannot switch to Gemini: not initialized")
                return False
            if model and model in self.SUPPORTED_MODELS['gemini']:
                try:
                    self.gemini_model = genai.GenerativeModel(model)
                    self.gemini_model_name = model
                except Exception as e:
                    logger.error(f"Failed to switch Gemini model: {e}")
                    return False
            self.current_provider = "gemini"
            self.current_model = self.gemini_model_name
            
        elif provider == "claude":
            if not self.claude_client:
                logger.error(f"Cannot switch to Claude: not initialized")
                return False
            if model and model in self.SUPPORTED_MODELS['claude']:
                self.claude_model_name = model
            self.current_provider = "claude"
            self.current_model = self.claude_model_name
            
        elif provider == "openai":
            if not self.openai_client or not self.openai_api_key:
                logger.error(f"Cannot switch to OpenAI: not initialized")
                return False
            if model and model in self.SUPPORTED_MODELS['openai']:
                self.openai_model = model
            self.current_provider = "openai"
            self.current_model = self.openai_model
            
        elif provider == "ollama":
            if not self.openai_client:
                logger.error(f"Cannot switch to Ollama: not initialized")
                return False
            if model:
                self.openai_model = model
            self.current_provider = "ollama"
            self.current_model = self.openai_model
        else:
            logger.error(f"Unknown provider: {provider}")
            return False
        
        logger.info(f"Switched to {self.current_provider.upper()} (model: {self.current_model})")
        return True
    
    def get_available_providers(self) -> dict:
        """Returns a dictionary of available providers and their models."""
        available = {}
        
        if self.gemini_model:
            available['gemini'] = {
                'models': self.SUPPORTED_MODELS['gemini'],
                'current_model': self.gemini_model_name,
                'status': 'active' if self.current_provider == 'gemini' else 'available'
            }
        
        if self.claude_client:
            available['claude'] = {
                'models': self.SUPPORTED_MODELS['claude'],
                'current_model': self.claude_model_name,
                'status': 'active' if self.current_provider == 'claude' else 'available'
            }
        
        if self.openai_client and self.openai_api_key:
            available['openai'] = {
                'models': self.SUPPORTED_MODELS['openai'],
                'current_model': self.openai_model,
                'status': 'active' if self.current_provider == 'openai' else 'available'
            }
        
        if self.openai_client and not self.openai_api_key:
            available['ollama'] = {
                'models': self.SUPPORTED_MODELS['ollama'],
                'current_model': self.openai_model,
                'status': 'active' if self.current_provider == 'ollama' else 'available'
            }
        
        return available
    
    def get_model_info(self) -> dict:
        """Returns current provider and model information for display."""
        return {
            'provider': self.current_provider or 'none',
            'model': self.current_model or 'none',
            'last_used_provider': self.last_used_provider,
            'last_used_model': self.last_used_model
        }

    def _check_rate_limit(self, user_id: str = "default_user") -> bool:
        """Check if user is within rate limits."""
        if not self.logger:
            return True
        
        is_allowed, stats = self.logger.check_rate_limit(
            user_id=user_id,
            max_requests_per_hour=100,
            max_tokens_per_hour=1_000_000
        )
        
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for user {user_id}", extra={"extra_data": {
                "requests_used": stats['requests_used'],
                "tokens_used": stats['tokens_used']
            }})
        
        return is_allowed

    def _log_request(
        self,
        endpoint: str,
        provider: str,
        operation_type: str,
        tokens_used: int,
        cost_credits: float,
        response_time_ms: int,
        success: bool,
        error_message: str = None,
        metadata: dict = None
    ):
        """Log AI request if logging is available."""
        if self.logger:
            self.logger.log_request(
                endpoint=endpoint,
                provider=provider,
                operation_type=operation_type,
                tokens_used=tokens_used,
                cost_credits=cost_credits,
                response_time_ms=response_time_ms,
                success=success,
                error_message=error_message,
                metadata=metadata or {}
            )

    def upload_video_file(self, file_path, retries=3, delay=5, user_id="default_user"):
        """
        Uploads video file to Gemini API with retry logic and logging.
        Returns: (file_object, provider_name) or (None, None) on failure
        """
        # Check rate limit
        if not self._check_rate_limit(user_id):
            return None, None
        
        if not self.gemini_model:
            logger.error("Gemini not available for video upload")
            return None, None
        
        if not self.google_api_key:
            logger.error("GOOGLE_API_KEY not configured")
            return None, None
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None, None
        
        file_size_mb = os.path.getsize(file_path) / (1024*1024)
        logger.info(f"Uploading video: {file_path} (Size: {file_size_mb:.2f} MB)")
        
        start_time = time.time()
        success = False
        error_msg = None
        
        for attempt in range(retries):
            try:
                logger.info(f"Upload attempt {attempt + 1}/{retries} - Using genai.upload_file()")
                
                if hasattr(genai, 'upload_file'):
                    video_file = genai.upload_file(path=file_path)
                else:
                    from google.generativeai.types import File
                    with open(file_path, 'rb') as f:
                        video_file = File.create(file=f, mime_type='video/mp4')
                
                logger.info(f"Upload successful! File name: {video_file.name}")
                
                # Wait for processing
                logger.info(f"Waiting for processing...")
                max_wait = 300
                wait_time = 0
                
                while video_file.state.name == "PROCESSING":
                    if wait_time >= max_wait:
                        error_msg = f"Processing timeout after {max_wait}s"
                        logger.warning(error_msg)
                        break
                    
                    time.sleep(5)
                    wait_time += 5
                    video_file = genai.get_file(video_file.name)
                    logger.debug(f"Processing... ({wait_time}s elapsed, state: {video_file.state.name})")
                
                if video_file.state.name == "ACTIVE":
                    logger.info(f"Video ready for analysis!")
                    success = True
                    
                    # Log successful upload
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    self._log_request(
                        endpoint="/upload_video",
                        provider="gemini",
                        operation_type="video_upload",
                        tokens_used=0,  # Upload doesn't consume tokens
                        cost_credits=0.0,
                        response_time_ms=elapsed_ms,
                        success=True,
                        metadata={
                            'file_size_mb': file_size_mb,
                            'processing_time_s': wait_time
                        }
                    )
                    
                    return video_file, "gemini"
                else:
                    error_msg = f"Processing failed with state: {video_file.state.name}"
                    logger.error(error_msg)
                    
            except Exception as e:
                error_msg = str(e)
                logger.exception(f"Upload attempt {attempt + 1}/{retries} failed")
                if attempt < retries - 1:
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error("All upload attempts failed")
        
        # Log failed upload
        elapsed_ms = int((time.time() - start_time) * 1000)
        self._log_request(
            endpoint="/upload_video",
            provider="gemini",
            operation_type="video_upload",
            tokens_used=0,
            cost_credits=0.0,
            response_time_ms=elapsed_ms,
            success=False,
            error_message=error_msg,
            metadata={'file_size_mb': file_size_mb}
        )
        
        return None, None

    def generate_text(self, prompt, retries=3, user_id="default_user"):
        """
        Generates text using the configured primary provider with fallback support.
        Supports: Gemini, Claude, OpenAI, Ollama
        Includes comprehensive logging and tracks which model was used.
        Returns: str (generated text)
        """
        # Check rate limit
        if not self._check_rate_limit(user_id):
            return "⚠️ Rate limit exceeded. Please try again later."
        
        start_time = time.time()
        success = False
        error_msg = None
        tokens_used = 0
        provider_used = None
        model_used = None
        
        # Estimate tokens (rough approximation)
        estimated_tokens = len(prompt.split()) * 1.3
        
        # Build provider order based on current_provider setting
        providers_to_try = []
        if self.current_provider == "gemini" and self.gemini_model:
            providers_to_try.append("gemini")
        elif self.current_provider == "claude" and self.claude_client:
            providers_to_try.append("claude")
        elif self.current_provider == "openai" and self.openai_client and self.openai_api_key:
            providers_to_try.append("openai")
        elif self.current_provider == "ollama" and self.openai_client:
            providers_to_try.append("ollama")
        
        # Add fallback providers if enabled
        if self.fallback_enabled:
            if self.gemini_model and "gemini" not in providers_to_try:
                providers_to_try.append("gemini")
            if self.claude_client and "claude" not in providers_to_try:
                providers_to_try.append("claude")
            if self.openai_client and self.openai_api_key and "openai" not in providers_to_try:
                providers_to_try.append("openai")
            if self.openai_client and not self.openai_api_key and "ollama" not in providers_to_try:
                providers_to_try.append("ollama")
        
        # Try each provider
        for provider in providers_to_try:
            if provider == "gemini":
                # Try Gemini
                for attempt in range(retries):
                    try:
                        logger.info(f"Generating text with Gemini [{self.gemini_model_name}] (attempt {attempt + 1}/{retries})")
                        response = self.gemini_model.generate_content(prompt)
                        
                        elapsed_ms = int((time.time() - start_time) * 1000)
                        tokens_used = int(estimated_tokens + len(response.text.split()) * 1.3)
                        provider_used = "gemini"
                        model_used = self.gemini_model_name
                        success = True
                        
                        # Log successful request
                        self._log_request(
                            endpoint="/generate_text",
                            provider=provider_used,
                            operation_type="text_generation",
                            tokens_used=tokens_used,
                            cost_credits=0.0,  # Gemini free tier
                            response_time_ms=elapsed_ms,
                            success=True,
                            metadata={'prompt_length': len(prompt), 'model': model_used}
                        )
                        
                        # Track last used model info
                        self.last_used_provider = provider_used
                        self.last_used_model = model_used
                        logger.info(f"Success with {provider_used.upper()} [{model_used}]!")
                        return response.text
                        
                    except Exception as e:
                        error_msg = str(e)
                        logger.exception(f"Gemini attempt {attempt + 1}/{retries} failed")
                        if attempt < retries - 1:
                            time.sleep(2)
                        else:
                            logger.warning("Switching to fallback provider...")
            
            elif provider == "claude":
                # Try Claude
                try:
                    logger.info(f"Generating text with Claude [{self.claude_model_name}]...")
                    
                    message = self.claude_client.messages.create(
                        model=self.claude_model_name,
                        max_tokens=4000,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    tokens_used = message.usage.input_tokens + message.usage.output_tokens if hasattr(message, 'usage') else int(estimated_tokens * 2)
                    provider_used = "claude"
                    model_used = self.claude_model_name
                    success = True
                    
                    # Calculate cost (Claude pricing)
                    cost_credits = 2.0  # Example cost
                    
                    # Log successful request
                    self._log_request(
                        endpoint="/generate_text",
                        provider=provider_used,
                        operation_type="text_generation",
                        tokens_used=tokens_used,
                        cost_credits=cost_credits,
                        response_time_ms=elapsed_ms,
                        success=True,
                        metadata={'prompt_length': len(prompt), 'model': model_used}
                    )
                    
                    # Track last used model info
                    self.last_used_provider = provider_used
                    self.last_used_model = model_used
                    logger.info(f"Success with {provider_used.upper()} [{model_used}]!")
                    return message.content[0].text
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.exception("Claude generation failed")
                    logger.warning("Switching to fallback provider...")
            
            elif provider == "openai":
                # Try OpenAI
                try:
                    logger.info(f"Generating text with OpenAI [{self.openai_model}]...")
                    
                    response = self.openai_client.chat.completions.create(
                        model=self.openai_model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    tokens_used = response.usage.total_tokens
                    provider_used = "openai"
                    model_used = self.openai_model
                    success = True
                    
                    # Log successful request
                    self._log_request(
                        endpoint="/generate_text",
                        provider=provider_used,
                        operation_type="text_generation",
                        tokens_used=tokens_used,
                        cost_credits=0.0, # Will be calculated by logger
                        response_time_ms=elapsed_ms,
                        success=True,
                        metadata={'prompt_length': len(prompt), 'model': model_used}
                    )
                    
                    # Track last used model info
                    self.last_used_provider = provider_used
                    self.last_used_model = model_used
                    logger.info(f"Success with {provider_used.upper()} [{model_used}]!")
                    return response.choices[0].message.content
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.exception("OpenAI generation failed")
                    logger.warning("Switching to fallback provider...")
                    
            elif provider == "ollama":
                # Try Ollama
                try:
                    logger.info(f"Generating text with Ollama [{self.openai_model}]...")
                    
                    response = self.openai_client.chat.completions.create(
                        model=self.openai_model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    tokens_used = int(estimated_tokens + len(response.choices[0].message.content.split()) * 1.3)
                    provider_used = "ollama"
                    model_used = self.openai_model
                    success = True
                    
                    # Log successful request
                    self._log_request(
                        endpoint="/generate_text",
                        provider=provider_used,
                        operation_type="text_generation",
                        tokens_used=tokens_used,
                        cost_credits=0.0,
                        response_time_ms=elapsed_ms,
                        success=True,
                        metadata={'prompt_length': len(prompt), 'model': model_used}
                    )
                    
                    # Track last used model info
                    self.last_used_provider = provider_used
                    self.last_used_model = model_used
                    logger.info(f"Success with {provider_used.upper()} [{model_used}]!")
                    return response.choices[0].message.content
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.exception("Ollama generation failed")
                    logger.warning("Switching to fallback provider...")
        
        # If we get here, all providers failed
        logger.error("All AI providers failed to generate text")
        
        # Log final failure
        elapsed_ms = int((time.time() - start_time) * 1000)
        self._log_request(
            endpoint="/generate_text",
            provider=self.current_provider or "none",
            operation_type="text_generation",
            tokens_used=0,
            cost_credits=0.0,
            response_time_ms=elapsed_ms,
            success=False,
            error_message=error_msg,
            metadata={'prompt_length': len(prompt)}
        )
        
        return f"❌ Error: All AI providers failed. Last error: {error_msg}"

    def analyze_video(self, video_file_obj, prompt, retries=3, enable_grounding=True, user_id="default_user"):
        """
        Analyzes video using Gemini with fallback to mock response.
        Includes comprehensive logging, rate limiting, and model tracking.
        """
        # Check rate limit
        if not self._check_rate_limit(user_id):
            return {
                "error": "Rate limit exceeded. Please try again later.",
                "captions": "",
                "shorts_ideas": [],
                "blog_post": "",
                "social_post": "",
                "thumbnail_ideas": [],
                "model_info": {"provider": "none", "model": "none"}
            }
        
        start_time = time.time()
        success = False
        error_msg = None
        tokens_used = 0
        
        # Estimate tokens
        estimated_tokens = len(prompt.split()) * 1.3 + 5000  # Video analysis uses more tokens
        
        # 1. Try Gemini Video Analysis (only Gemini supports video)
        if self.gemini_model and video_file_obj:
            for attempt in range(retries):
                try:
                    logger.info(f"Analyzing video with Gemini [{self.gemini_model_name}] (attempt {attempt + 1}/{retries})")
                    
                    # Add grounding instructions if enabled
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
                    
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    tokens_used = int(estimated_tokens + len(response.text.split()) * 1.3)
                    success = True
                    
                    # Log successful request
                    self._log_request(
                        endpoint="/analyze_video",
                        provider="gemini",
                        operation_type="video_analysis",
                        tokens_used=tokens_used,
                        cost_credits=5.0,  # As per credit system
                        response_time_ms=elapsed_ms,
                        success=True,
                        metadata={
                            'grounding_enabled': enable_grounding,
                            'prompt_length': len(prompt),
                            'model': self.gemini_model_name
                        }
                    )
                    
                    # Track last used model info
                    self.last_used_provider = "gemini"
                    self.last_used_model = self.gemini_model_name
                    logger.info(f"Analysis complete with GEMINI [{self.gemini_model_name}]!")
                    
                    # Parse response
                    parsed_results = self._parse_response(response.text)
                    
                    # Add model info to results
                    parsed_results['model_info'] = {
                        'provider': 'gemini',
                        'model': self.gemini_model_name
                    }
                    
                    # --- ISSUE #42: Fact-Grounding Integration ---
                    # Apply fact-grounding validation if enabled
                    if enable_grounding and GROUNDING_AVAILABLE and parsed_results.get('captions'):
                        logger.info("Validating claims against transcript...")
                        parsed_results = self._apply_fact_grounding(parsed_results)
                    
                    return parsed_results
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Analysis attempt {attempt + 1}/{retries} failed: {e}")
                    if attempt < retries - 1:
                        time.sleep(5)
                    else:
                        logger.error("All analysis attempts failed")
        
        # 2. Log failure and return mock
        elapsed_ms = int((time.time() - start_time) * 1000)
        self._log_request(
            endpoint="/analyze_video",
            provider="gemini",
            operation_type="video_analysis",
            tokens_used=0,
            cost_credits=0.0,
            response_time_ms=elapsed_ms,
            success=False,
            error_message=error_msg or "Video analysis requires Gemini API",
            metadata={'grounding_enabled': enable_grounding}
        )
        
        if self.fallback_enabled:
            logger.warning("Using mock analysis response (video analysis requires Gemini API key)")
            mock_result = self._generate_mock_analysis()
            mock_result['model_info'] = {'provider': 'mock', 'model': 'fallback'}
            return mock_result
            
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
            captions_match = re.search(r"### Captions\s*\n(.*?)(?=\n###|$)", text, re.DOTALL)
            if captions_match:
                captions_text = captions_match.group(1).strip()
                results['captions'] = captions_text.removeprefix("```srt").removesuffix("```").strip()

        # Extract Shorts Ideas
        shorts_match = re.search(r"### Shorts Ideas\s*\n(.*?)(?=\n###|$)", text, re.DOTALL)
        if shorts_match:
            shorts_text = shorts_match.group(1).strip()
            ideas = []
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
        """Apply fact-grounding validation to parsed results."""
        if not GROUNDING_AVAILABLE:
            return parsed_results
        
        try:
            srt_content = parsed_results.get('captions', '')
            if not srt_content:
                logger.warning("No transcript available for grounding")
                return parsed_results
            
            grounder = FactGrounder(srt_content)
            grounding_report = grounder.generate_grounding_report(parsed_results)
            
            if 'blog_post' in grounding_report['filtered_content']:
                original_blog = parsed_results.get('blog_post', '')
                filtered_blog = grounding_report['filtered_content']['blog_post']
                
                if len(filtered_blog) < len(original_blog) * 0.5:
                    logger.warning(f"Blog post heavily filtered during grounding ({len(filtered_blog)}/{len(original_blog)} chars)")
                
                parsed_results['blog_post'] = filtered_blog
                parsed_results['blog_post_original'] = original_blog
            
            if 'social_post' in grounding_report['filtered_content']:
                parsed_results['social_post'] = grounding_report['filtered_content']['social_post']
            
            if 'shorts_ideas' in grounding_report['filtered_content']:
                validated_shorts = grounding_report['filtered_content']['shorts_ideas']
                for short in validated_shorts:
                    status = short.get('validation_status', 'unknown')
                    if status == 'verified':
                        short['validation_badge'] = '✅ Verified'
                    elif status == 'unverified_summary':
                        short['validation_badge'] = '⚠️ Summary needs review'
                    else:
                        short['validation_badge'] = '❌ Invalid timestamps'
                
                parsed_results['shorts_ideas'] = validated_shorts
            
            parsed_results['grounding_metadata'] = {
                'enabled': True,
                'blog_grounding_rate': grounding_report['statistics'].get('blog_grounding_rate', 0),
                'social_grounding_rate': grounding_report['statistics'].get('social_grounding_rate', 0),
                'shorts_verification_rate': grounding_report['statistics'].get('shorts_verification_rate', 0),
                'full_report': grounding_report
            }
            
            stats = grounding_report['statistics']
            logger.info("Fact-grounding stats: " + 
                        f"Blog: {stats.get('blog_grounding_rate', 0):.1%} verified, " +
                        f"Social: {stats.get('social_grounding_rate', 0):.1%} verified, " +
                        f"Shorts: {stats.get('shorts_verification_rate', 0):.1%} verified")
        except Exception as e:
            logger.error(f"Fact-grounding validation failed: {e}")

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
                }
            ],
            
            "blog_post": """# AI Provider Configuration Required

## What Happened?
Your video upload to the Gemini API failed. This is typically caused by configuration issues.

### Next Steps
1. Update your package: `pip install --upgrade google-generativeai`
2. Verify your API key is valid
3. Try uploading your video again

*This is a mock response to help you diagnose the issue.*""",
            
            "social_post": "🔧 Pro tip: Keep your AI packages updated! #DevLife #AITools #TechTips 🚀",
            
            "thumbnail_ideas": [
                "A frustrated developer looking at a computer screen showing 'API Error', dramatic red lighting, cyberpunk aesthetic"
            ]
        }
    
    def get_current_provider(self):
        """Returns the name of the currently active provider."""
        return self.current_provider or "none"
    
    def get_current_model(self):
        """Returns the name of the currently active model."""
        return self.current_model or "none"
    
    def get_provider_display_name(self):
        """Returns a formatted display string for current provider and model."""
        if self.current_provider and self.current_model:
            return f"{self.current_provider.upper()} ({self.current_model})"
        elif self.current_provider:
            return self.current_provider.upper()
        return "No Provider"

    def generate(self, prompt, retries=3, user_id="default_user"):
        """Alias for generate_text for compatibility."""
        return self.generate_text(prompt, retries, user_id)

    def _extract_json(self, text: str) -> dict:
        """
        Utility to extract JSON from LLM responses that might contain markdown or extra text.
        """
        try:
            # Clean response
            cleaned = text.strip()

            # Remove markdown code blocks if present
            if "```" in cleaned:
                # Try to find json block first
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
                if json_match:
                    cleaned = json_match.group(1).strip()
                else:
                    cleaned = cleaned.split("```")[1].strip()

            # Remove 'json' prefix if present (sometimes models do this outside code blocks)
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

            # Find the first '[' or '{' and last ']' or '}'
            start_idx = -1
            for i, char in enumerate(cleaned):
                if char in "{[":
                    start_idx = i
                    break
            
            end_idx = -1
            for i, char in enumerate(reversed(cleaned)):
                if char in "}]":
                    end_idx = len(cleaned) - i
                    break
            
            if start_idx != -1 and end_idx != -1:
                cleaned = cleaned[start_idx:end_idx]

            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError, IndexError, AttributeError) as e:
            logger.warning(f"Failed to extract JSON from response: {e}")
            return {}
    
    def get_last_used_display(self):
        """Returns a formatted display string for the last used provider and model."""
        if self.last_used_provider and self.last_used_model:
            return f"{self.last_used_provider.upper()} ({self.last_used_model})"
        elif self.last_used_provider:
            return self.last_used_provider.upper()
        return "N/A"