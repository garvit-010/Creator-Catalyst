import pytest
import os
import sys
import tempfile
import json
import uuid
import time
import nltk
import logging
from pathlib import Path
from src.database.database import Database
from src.database.storage_manager import StorageManager
from src.database.credits_manager import CreditsManager
from src.core.llm_wrapper import LLMWrapper

# Initialize logger for tests
logger = logging.getLogger(__name__)

# Ensure NLTK data is available
def pytest_configure(config):
    """Download required NLTK data before running tests."""
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except Exception as e:
        logger.warning(f"Failed to download NLTK data: {e}")

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all singleton instances before each test."""
    from src.database import database, ai_request_logger
    database._db_instance = None
    ai_request_logger._logger_instance = None
    yield

@pytest.fixture
def db_path():
    """Create a temporary database path."""
    # Create a unique filename but don't keep it open
    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, f"test_db_{uuid.uuid4().hex}.db")
    
    # Set environment variable so modules know which DB to use
    os.environ['DATABASE_PATH'] = path
    
    yield path
    
    # Cleanup
    if os.path.exists(path):
        # Try to close any remaining sqlite connections if possible
        # but since we reset singletons, they should be closed if they were in the same process
        try:
            os.remove(path)
        except PermissionError:
            # On Windows, sometimes the file is still locked for a bit
            time.sleep(0.1)
            try:
                os.remove(path)
            except:
                pass 

@pytest.fixture
def temp_db(db_path):
    """Create a temporary database for testing."""
    from src.database import database
    database._db_instance = None # Reset singleton
    return Database(db_path)

@pytest.fixture
def storage_manager(db_path):
    """Create a storage manager with a temporary database."""
    from src.database import database, ai_request_logger
    database._db_instance = None # Reset singleton
    ai_request_logger._logger_instance = None # Reset logger singleton
    sm = StorageManager(db_path=db_path)
    return sm

@pytest.fixture
def credits_manager(db_path):
    """Create a credits manager with a temporary database."""
    from src.database import database, ai_request_logger
    database._db_instance = None # Reset singleton
    ai_request_logger._logger_instance = None # Reset logger singleton
    return CreditsManager(db_path=db_path)

@pytest.fixture
def mock_llm(mocker):
    """Mock the LLM wrapper to avoid API calls."""
    mock = mocker.patch("src.core.llm_wrapper.LLMWrapper")
    instance = mock.return_value
    instance.generate_text.return_value = "Mocked AI Response"
    instance.analyze_video.return_value = {"summary": "Mocked summary", "topics": ["test"]}
    return instance

@pytest.fixture
def mock_genai(mocker):
    """Mock Google Generative AI."""
    mock = mocker.patch("google.generativeai.GenerativeModel")
    instance = mock.return_value
    mock_response = mocker.Mock()
    mock_response.text = "Mocked Gemini Response"
    instance.generate_content.return_value = mock_response
    return instance

@pytest.fixture
def mock_openai(mocker):
    """Mock OpenAI."""
    mock = mocker.patch("openai.OpenAI")
    instance = mock.return_value
    
    mock_usage = mocker.Mock()
    mock_usage.total_tokens = 100
    mock_usage.prompt_tokens = 70
    mock_usage.completion_tokens = 30
    
    instance.chat.completions.create.return_value = mocker.Mock(
        choices=[mocker.Mock(message=mocker.Mock(content="Mocked OpenAI Response"))],
        usage=mock_usage
    )
    return instance

@pytest.fixture
def sample_video_data():
    """Sample video data for testing."""
    return {
        "filename": "test_video.mp4",
        "file_path": "/path/to/test_video.mp4",
        "file_size_mb": 10.5,
        "duration_seconds": 60,
        "platform": "YouTube"
    }
