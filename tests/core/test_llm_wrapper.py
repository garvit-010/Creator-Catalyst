import pytest
from src.core.llm_wrapper import LLMWrapper

def test_llm_wrapper_initialization(db_path, mock_genai):
    """Test that LLM wrapper initializes correctly."""
    wrapper = LLMWrapper()
    assert wrapper is not None
    # By default it should try Gemini if key exists, but we've mocked the class behavior
    # We can check if it has the expected methods
    assert hasattr(wrapper, 'generate_text')
    assert hasattr(wrapper, 'analyze_video')

def test_generate_text_gemini(db_path, mocker, mock_genai):
    """Test text generation via Gemini."""
    mocker.patch("os.getenv", side_effect=lambda k, d=None: "fake_key" if "API_KEY" in k else d)
    wrapper = LLMWrapper()
    wrapper.current_provider = "gemini"
    
    response = wrapper.generate_text("Hello")
    assert response == "Mocked Gemini Response"

def test_generate_text_openai_fallback(db_path, mocker, mock_genai, mock_openai):
    """Test fallback to OpenAI when Gemini fails."""
    # Mock Gemini to fail
    mock_genai.generate_content.side_effect = Exception("Gemini Error")
    
    # Mock env to enable fallback and have keys
    mocker.patch("os.getenv", side_effect=lambda k, d=None: "fake_key" if "API_KEY" in k else ("true" if k == "ENABLE_FALLBACK" else d))
    
    wrapper = LLMWrapper()
    wrapper.current_provider = "gemini"
    wrapper.openai_client = mock_openai # Ensure openai client is set
    
    response = wrapper.generate_text("Hello")
    assert response == "Mocked OpenAI Response"

def test_extract_json_from_response(db_path):
    """Test JSON extraction utility."""
    wrapper = LLMWrapper()
    json_str = '{"key": "value"}'
    response = f"Here is the data: ```json\n{json_str}\n```"
    
    extracted = wrapper._extract_json(response)
    assert extracted == {"key": "value"}
