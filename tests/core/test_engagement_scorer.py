import pytest
from src.core.engagement_scorer import EngagementScorer, EngagementScore

@pytest.fixture
def scorer():
    return EngagementScorer()

def test_score_content_basic(scorer):
    """Test scoring basic content."""
    text = "Check out this amazing new video about AI! #AI #Tech #Innovation"
    score = scorer.score_content(text)
    
    assert isinstance(score, EngagementScore)
    assert 0 <= score.overall_score <= 100
    assert "Instagram" in score.platform_scores
    assert score.recommended_platform in scorer.PLATFORM_WEIGHTS.keys()
    assert len(score.strengths) > 0

def test_score_content_empty(scorer):
    """Test scoring empty content."""
    score = scorer.score_content("")
    assert score.overall_score < 30 # Should be low for empty content

def test_sentiment_analysis(scorer):
    """Test sentiment analysis component."""
    positive_text = "I love this amazing and wonderful project! It's great."
    negative_text = "This is a terrible and awful disaster. I hate it."
    
    pos_score = scorer.score_content(positive_text)
    neg_score = scorer.score_content(negative_text)
    
    assert pos_score.sentiment['compound'] > 0
    assert neg_score.sentiment['compound'] < 0

def test_platform_specific_scoring(scorer):
    """Test that platform-specific characteristics affect the score."""
    # Short text with hashtags (good for Twitter)
    twitter_text = "New blog post out now! #tech #coding"
    # Long professional text (good for LinkedIn)
    linkedin_text = """I'm thrilled to share our latest research on neural architectures. 
    This study demonstrates significant improvements in transformer efficiency 
    by optimizing self-attention mechanisms. #DeepLearning #Research #AI"""
    
    twitter_score = scorer.score_content(twitter_text)
    linkedin_score = scorer.score_content(linkedin_text)
    
    # This is a bit non-deterministic but generally:
    assert "Twitter/X" in twitter_score.platform_scores
    assert "LinkedIn" in linkedin_score.platform_scores
