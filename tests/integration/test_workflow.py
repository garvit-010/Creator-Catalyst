import pytest
import os
import tempfile
from src.database.storage_manager import StorageManager
from src.database.credits_manager import CreditsManager
from src.core.llm_wrapper import LLMWrapper

def test_full_analysis_to_db_workflow(db_path, mock_llm, sample_video_data):
    """
    Test the full workflow:
    1. Check credits
    2. Analyze video (mocked)
    3. Save results to database
    4. Verify persistence
    """
    # 1. Setup
    from src.database import database
    database._db_instance = None # Reset singleton
    storage = StorageManager(db_path=db_path)
    credits = CreditsManager(db_path=db_path)
    
    initial_balance = credits.get_balance()
    
    # 2. Check and deduct credits for upload
    has_credits, balance, cost = credits.has_sufficient_credits('video_upload')
    assert has_credits is True
    credits.deduct_credits('video_upload')
    
    # 3. Simulate analysis results
    results = {
        "summary": "This is a test summary",
        "blog_post": "Test blog content",
        "social_post": "Test tweet #AI",
        "shorts_ideas": [{"topic": "Test short", "hook": "Test hook"}],  
        "keywords": ["test", "integration"],
        "engagement_score": 85
    }
    
    # 4. Save to DB using storage manager
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        temp_video_path = tf.name
    
    try:
        video_id = storage.save_analysis_results(
            video_path=temp_video_path,
            results=results,
            platform=sample_video_data["platform"]
        )
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
    
    # 5. Verify results
    assert video_id > 0
    
    # Check video record
    video = storage.db.get_video(video_id)
    assert video.platform == sample_video_data["platform"]
    
    # Check generated content
    contents = storage.db.get_content_by_video(video_id)
    content_types = [c.content_type for c in contents]
    assert "blog_post" in content_types
    assert "social_post" in content_types
    assert "shorts_idea" in content_types
    
    # Check credits deduction
    final_balance = credits.get_balance()
    assert final_balance == initial_balance - CreditsManager.COSTS['video_upload']
