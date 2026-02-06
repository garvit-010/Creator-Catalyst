import pytest
from src.database.database import Video, ContentOutput, GroundingReport

def test_create_and_get_video(temp_db, sample_video_data):
    video = Video(**sample_video_data)
    video_id = temp_db.create_video(video)
    
    assert video_id > 0
    
    retrieved_video = temp_db.get_video(video_id)
    assert retrieved_video is not None
    assert retrieved_video.filename == sample_video_data["filename"]
    assert retrieved_video.platform == sample_video_data["platform"]

def test_update_video_status(temp_db, sample_video_data):
    video = Video(**sample_video_data)
    video_id = temp_db.create_video(video)
    
    temp_db.update_video_status(video_id, "completed")
    
    retrieved_video = temp_db.get_video(video_id)
    assert retrieved_video.processing_status == "completed"

def test_save_and_get_content(temp_db, sample_video_data):
    video = Video(**sample_video_data)
    video_id = temp_db.create_video(video)
    
    content = ContentOutput(
        video_id=video_id,
        content_type="blog_post",
        content="This is a test blog post",
        metadata='{"key": "value"}'
    )
    content_id = temp_db.save_content(content)
    
    assert content_id > 0
    
    retrieved_content = temp_db.get_content(content_id)
    assert retrieved_content is not None
    assert retrieved_content.content == "This is a test blog post"
    assert retrieved_content.content_type == "blog_post"

def test_delete_video_cascades(temp_db, sample_video_data):
    video = Video(**sample_video_data)
    video_id = temp_db.create_video(video)
    
    content = ContentOutput(
        video_id=video_id,
        content_type="blog_post",
        content="This is a test blog post"
    )
    temp_db.save_content(content)
    
    temp_db.delete_video(video_id)
    
    assert temp_db.get_video(video_id) is None
    assert len(temp_db.get_content_by_video(video_id)) == 0

def test_search_videos(temp_db):
    v1 = Video(filename="apple.mp4", file_path="/path/1", uploaded_at="2023-01-01")
    v2 = Video(filename="banana.mp4", file_path="/path/2", uploaded_at="2023-01-02")
    temp_db.create_video(v1)
    temp_db.create_video(v2)
    
    results = temp_db.search_videos("apple")
    assert len(results) == 1
    assert results[0].filename == "apple.mp4"
