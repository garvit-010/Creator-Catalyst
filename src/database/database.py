"""
Database layer for Creator Catalyst.
Handles persistence of video analysis results, content generations, and metadata.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager


@dataclass
class Video:
    """Represents a processed video."""
    id: Optional[int] = None
    filename: str = ""
    file_path: str = ""
    file_size_mb: float = 0.0
    duration_seconds: Optional[int] = None
    uploaded_at: str = ""
    platform: str = "General"
    grounding_enabled: bool = True
    processing_status: str = "pending"  # pending, processing, completed, failed
    
    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ContentOutput:
    """Represents a generated content output."""
    id: Optional[int] = None
    video_id: int = 0
    content_type: str = ""  # captions, blog_post, social_post, shorts_idea, thumbnail_idea
    content: str = ""
    metadata: str = "{}"  # JSON string for additional data
    version: int = 1
    created_at: str = ""
    grounding_rate: Optional[float] = None
    validation_status: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary."""
        data = asdict(self)
        # Parse JSON metadata
        if isinstance(data['metadata'], str):
            try:
                data['metadata'] = json.loads(data['metadata'])
            except:
                data['metadata'] = {}
        return data


@dataclass
class GroundingReport:
    """Represents a fact-grounding validation report."""
    id: Optional[int] = None
    video_id: int = 0
    blog_grounding_rate: float = 0.0
    social_grounding_rate: float = 0.0
    shorts_verification_rate: float = 0.0
    total_claims: int = 0
    verified_claims: int = 0
    unverified_claims: int = 0
    full_report: str = "{}"  # JSON string
    created_at: str = ""
    
    def to_dict(self):
        """Convert to dictionary."""
        data = asdict(self)
        if isinstance(data['full_report'], str):
            try:
                data['full_report'] = json.loads(data['full_report'])
            except:
                data['full_report'] = {}
        return data


class Database:
    """
    SQLite database manager for Creator Catalyst.
    Handles all persistence operations with proper transaction management.
    """
    
    def __init__(self, db_path: str = "data/creator_catalyst.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Videos table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size_mb REAL DEFAULT 0.0,
                    duration_seconds INTEGER,
                    uploaded_at TEXT NOT NULL,
                    platform TEXT DEFAULT 'General',
                    grounding_enabled INTEGER DEFAULT 1,
                    processing_status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Content outputs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS content_outputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL,
                    content_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    version INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    grounding_rate REAL,
                    validation_status TEXT,
                    FOREIGN KEY (video_id) REFERENCES videos (id) ON DELETE CASCADE
                )
            """)
            
            # Grounding reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS grounding_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL,
                    blog_grounding_rate REAL DEFAULT 0.0,
                    social_grounding_rate REAL DEFAULT 0.0,
                    shorts_verification_rate REAL DEFAULT 0.0,
                    total_claims INTEGER DEFAULT 0,
                    verified_claims INTEGER DEFAULT 0,
                    unverified_claims INTEGER DEFAULT 0,
                    full_report TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_video_type 
                ON content_outputs(video_id, content_type)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_videos_uploaded 
                ON videos(uploaded_at DESC)
            """)
            
            print(f"✅ Database initialized at: {self.db_path}")
    
    # ==================== VIDEO OPERATIONS ====================
    
    def create_video(self, video: Video) -> int:
        """
        Create a new video record.
        
        Args:
            video: Video dataclass instance
            
        Returns:
            video_id: ID of the created video
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO videos (
                    filename, file_path, file_size_mb, duration_seconds,
                    uploaded_at, platform, grounding_enabled, processing_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video.filename,
                video.file_path,
                video.file_size_mb,
                video.duration_seconds,
                video.uploaded_at or datetime.now().isoformat(),
                video.platform,
                1 if video.grounding_enabled else 0,
                video.processing_status
            ))
            
            video_id = cursor.lastrowid
            print(f"✅ Video created: ID={video_id}, filename={video.filename}")
            return video_id
    
    def get_video(self, video_id: int) -> Optional[Video]:
        """Get video by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
            row = cursor.fetchone()
            
            if row:
                return Video(
                    id=row['id'],
                    filename=row['filename'],
                    file_path=row['file_path'],
                    file_size_mb=row['file_size_mb'],
                    duration_seconds=row['duration_seconds'],
                    uploaded_at=row['uploaded_at'],
                    platform=row['platform'],
                    grounding_enabled=bool(row['grounding_enabled']),
                    processing_status=row['processing_status']
                )
            return None
    
    def get_all_videos(self, limit: int = 100, offset: int = 0) -> List[Video]:
        """Get all videos with pagination."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM videos 
                ORDER BY uploaded_at DESC 
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            videos = []
            for row in cursor.fetchall():
                videos.append(Video(
                    id=row['id'],
                    filename=row['filename'],
                    file_path=row['file_path'],
                    file_size_mb=row['file_size_mb'],
                    duration_seconds=row['duration_seconds'],
                    uploaded_at=row['uploaded_at'],
                    platform=row['platform'],
                    grounding_enabled=bool(row['grounding_enabled']),
                    processing_status=row['processing_status']
                ))
            
            return videos
    
    def update_video_status(self, video_id: int, status: str):
        """Update video processing status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE videos 
                SET processing_status = ? 
                WHERE id = ?
            """, (status, video_id))
            print(f"✅ Video {video_id} status updated to: {status}")
    
    def delete_video(self, video_id: int):
        """Delete video and all associated content (cascades)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM videos WHERE id = ?", (video_id,))
            print(f"✅ Video {video_id} deleted (with all associated content)")
    
    # ==================== CONTENT OPERATIONS ====================
    
    def save_content(self, content: ContentOutput) -> int:
        """
        Save a content output.
        
        Args:
            content: ContentOutput dataclass instance
            
        Returns:
            content_id: ID of the saved content
        """
        # Serialize metadata if it's a dict
        metadata_str = content.metadata
        if isinstance(metadata_str, dict):
            metadata_str = json.dumps(metadata_str)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO content_outputs (
                    video_id, content_type, content, metadata, 
                    version, created_at, grounding_rate, validation_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                content.video_id,
                content.content_type,
                content.content,
                metadata_str,
                content.version,
                content.created_at or datetime.now().isoformat(),
                content.grounding_rate,
                content.validation_status
            ))
            
            content_id = cursor.lastrowid
            print(f"✅ Content saved: ID={content_id}, type={content.content_type}")
            return content_id
    
    def get_content(self, content_id: int) -> Optional[ContentOutput]:
        """Get content by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM content_outputs WHERE id = ?", (content_id,))
            row = cursor.fetchone()
            
            if row:
                return ContentOutput(
                    id=row['id'],
                    video_id=row['video_id'],
                    content_type=row['content_type'],
                    content=row['content'],
                    metadata=row['metadata'],
                    version=row['version'],
                    created_at=row['created_at'],
                    grounding_rate=row['grounding_rate'],
                    validation_status=row['validation_status']
                )
            return None
    
    def get_content_by_video(self, video_id: int, content_type: Optional[str] = None) -> List[ContentOutput]:
        """
        Get all content for a video, optionally filtered by type.
        
        Args:
            video_id: Video ID
            content_type: Optional content type filter
            
        Returns:
            List of ContentOutput instances
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if content_type:
                cursor.execute("""
                    SELECT * FROM content_outputs 
                    WHERE video_id = ? AND content_type = ?
                    ORDER BY version DESC, created_at DESC
                """, (video_id, content_type))
            else:
                cursor.execute("""
                    SELECT * FROM content_outputs 
                    WHERE video_id = ?
                    ORDER BY content_type, version DESC, created_at DESC
                """, (video_id,))
            
            contents = []
            for row in cursor.fetchall():
                contents.append(ContentOutput(
                    id=row['id'],
                    video_id=row['video_id'],
                    content_type=row['content_type'],
                    content=row['content'],
                    metadata=row['metadata'],
                    version=row['version'],
                    created_at=row['created_at'],
                    grounding_rate=row['grounding_rate'],
                    validation_status=row['validation_status']
                ))
            
            return contents
    
    def get_latest_content(self, video_id: int, content_type: str) -> Optional[ContentOutput]:
        """Get the latest version of a specific content type for a video."""
        contents = self.get_content_by_video(video_id, content_type)
        return contents[0] if contents else None
    
    def get_content_versions(self, video_id: int, content_type: str) -> List[ContentOutput]:
        """Get all versions of a specific content type for a video."""
        return self.get_content_by_video(video_id, content_type)
    
    def delete_content(self, content_id: int):
        """Delete a specific content output."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM content_outputs WHERE id = ?", (content_id,))
            print(f"✅ Content {content_id} deleted")
    
    # ==================== GROUNDING REPORT OPERATIONS ====================
    
    def save_grounding_report(self, report: GroundingReport) -> int:
        """Save a grounding validation report."""
        # Serialize full_report if it's a dict
        report_str = report.full_report
        if isinstance(report_str, dict):
            report_str = json.dumps(report_str)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO grounding_reports (
                    video_id, blog_grounding_rate, social_grounding_rate,
                    shorts_verification_rate, total_claims, verified_claims,
                    unverified_claims, full_report, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.video_id,
                report.blog_grounding_rate,
                report.social_grounding_rate,
                report.shorts_verification_rate,
                report.total_claims,
                report.verified_claims,
                report.unverified_claims,
                report_str,
                report.created_at or datetime.now().isoformat()
            ))
            
            report_id = cursor.lastrowid
            print(f"✅ Grounding report saved: ID={report_id}")
            return report_id
    
    def get_grounding_report(self, video_id: int) -> Optional[GroundingReport]:
        """Get the latest grounding report for a video."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM grounding_reports 
                WHERE video_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (video_id,))
            
            row = cursor.fetchone()
            if row:
                return GroundingReport(
                    id=row['id'],
                    video_id=row['video_id'],
                    blog_grounding_rate=row['blog_grounding_rate'],
                    social_grounding_rate=row['social_grounding_rate'],
                    shorts_verification_rate=row['shorts_verification_rate'],
                    total_claims=row['total_claims'],
                    verified_claims=row['verified_claims'],
                    unverified_claims=row['unverified_claims'],
                    full_report=row['full_report'],
                    created_at=row['created_at']
                )
            return None
    
    # ==================== ANALYTICS & SEARCH ====================
    
    def get_statistics(self) -> Dict:
        """Get overall statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total videos
            cursor.execute("SELECT COUNT(*) as count FROM videos")
            total_videos = cursor.fetchone()['count']
            
            # Total content pieces
            cursor.execute("SELECT COUNT(*) as count FROM content_outputs")
            total_contents = cursor.fetchone()['count']
            
            # Content by type
            cursor.execute("""
                SELECT content_type, COUNT(*) as count 
                FROM content_outputs 
                GROUP BY content_type
            """)
            content_by_type = {row['content_type']: row['count'] for row in cursor.fetchall()}
            
            # Average grounding rates
            cursor.execute("""
                SELECT 
                    AVG(blog_grounding_rate) as avg_blog,
                    AVG(social_grounding_rate) as avg_social,
                    AVG(shorts_verification_rate) as avg_shorts
                FROM grounding_reports
            """)
            row = cursor.fetchone()
            
            return {
                'total_videos': total_videos,
                'total_contents': total_contents,
                'content_by_type': content_by_type,
                'average_grounding_rates': {
                    'blog': row['avg_blog'] or 0.0,
                    'social': row['avg_social'] or 0.0,
                    'shorts': row['avg_shorts'] or 0.0
                }
            }
    
    def search_videos(self, query: str, limit: int = 50) -> List[Video]:
        """Search videos by filename."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM videos 
                WHERE filename LIKE ? 
                ORDER BY uploaded_at DESC 
                LIMIT ?
            """, (f"%{query}%", limit))
            
            videos = []
            for row in cursor.fetchall():
                videos.append(Video(
                    id=row['id'],
                    filename=row['filename'],
                    file_path=row['file_path'],
                    file_size_mb=row['file_size_mb'],
                    duration_seconds=row['duration_seconds'],
                    uploaded_at=row['uploaded_at'],
                    platform=row['platform'],
                    grounding_enabled=bool(row['grounding_enabled']),
                    processing_status=row['processing_status']
                ))
            
            return videos
    
    def get_recent_activity(self, limit: int = 10) -> List[Dict]:
        """Get recent content generations across all videos."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    c.id,
                    c.video_id,
                    c.content_type,
                    c.created_at,
                    v.filename,
                    v.platform
                FROM content_outputs c
                JOIN videos v ON c.video_id = v.id
                ORDER BY c.created_at DESC
                LIMIT ?
            """, (limit,))
            
            activities = []
            for row in cursor.fetchall():
                activities.append({
                    'content_id': row['id'],
                    'video_id': row['video_id'],
                    'content_type': row['content_type'],
                    'created_at': row['created_at'],
                    'filename': row['filename'],
                    'platform': row['platform']
                })
            
            return activities


# Singleton instance
_db_instance = None

def get_database(db_path: str = "creator_catalyst.db") -> Database:
    """Get or create database singleton instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(db_path)
    return _db_instance