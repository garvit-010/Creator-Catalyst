"""
Storage manager for Creator Catalyst.
Bridges the gap between analysis results and database persistence.
"""

import os
import json
import zipfile
import tempfile
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from src.database.database import get_database, Database, Video, ContentOutput, GroundingReport
from src.core.engagement_scorer import get_engagement_scorer, EngagementScore


class StorageManager:
    """
    High-level interface for storing and retrieving content.
    Handles the conversion between app data structures and database models.
    """

    def __init__(self, db_path: str = "data/creator_catalyst.db"):
        """Initialize storage manager with database connection."""
        self.db = get_database(db_path)
        self.scorer = get_engagement_scorer()
        print(f"✅ Storage manager initialized")

    def save_analysis_results(
        self,
        video_path: str,
        results: Dict,
        platform: str = "General",
        grounding_enabled: bool = True,
    ) -> int:
        """
        Save complete analysis results to database.

        Args:
            video_path: Path to the video file
            results: Dictionary containing all generated content
            platform: Target platform for content
            grounding_enabled: Whether grounding was enabled

        Returns:
            video_id: ID of the created video record
        """
        # --- ISSUE #54: Construct Semantic Metadata Blob ---
        # Concatenate all generated content into a searchable string
        metadata_blob_parts = [
            os.path.basename(video_path),
            platform,
            results.get("captions", ""),
            results.get("blog_post", ""),
            results.get("social_post", "")
        ]
        
        # Add Shorts Topics and Summaries
        if results.get("shorts_ideas"):
            for idea in results["shorts_ideas"]:
                if isinstance(idea, dict):
                    metadata_blob_parts.append(idea.get("topic", ""))
                    metadata_blob_parts.append(idea.get("summary", ""))
        
        # Add Thumbnail Ideas
        if results.get("thumbnail_ideas"):
             for idea in results["thumbnail_ideas"]:
                 if isinstance(idea, str):
                     metadata_blob_parts.append(idea)
                 elif isinstance(idea, dict):
                     metadata_blob_parts.append(idea.get("idea", ""))

        # Join and clean up (limit to ~100k chars to be safe)
        searchable_text = " ".join([str(part) for part in metadata_blob_parts if part])[:100000]

        # Create video record
        file_stats = os.stat(video_path)
        video = Video(
            filename=os.path.basename(video_path),
            file_path=video_path,
            file_size_mb=file_stats.st_size / (1024 * 1024),
            uploaded_at=datetime.now().isoformat(),
            platform=platform,
            grounding_enabled=grounding_enabled,
            processing_status="completed",
            searchable_text=searchable_text  # Save the blob!
        )

        video_id = self.db.create_video(video)

        # Save captions
        if results.get("captions"):
            self._save_captions(video_id, results["captions"])

        # Save blog post
        if results.get("blog_post"):
            self._save_blog_post(
                video_id, results["blog_post"], results.get("blog_post_original")
            )

        # Save social post
        if results.get("social_post"):
            self._save_social_post(video_id, results["social_post"])

        # Save shorts ideas
        if results.get("shorts_ideas"):
            self._save_shorts_ideas(video_id, results["shorts_ideas"])

        # Save thumbnail ideas
        if results.get("thumbnail_ideas"):
            self._save_thumbnail_ideas(video_id, results["thumbnail_ideas"])

        # Save grounding report
        if results.get("grounding_metadata"):
            self._save_grounding_report(video_id, results["grounding_metadata"])

        print(f"✅ All results saved for video ID: {video_id}")
        return video_id

    def _save_captions(self, video_id: int, captions: str):
        """Save video captions/transcript."""
        # Score content
        score = self.scorer.score_content(captions, "captions")
        
        content = ContentOutput(
            video_id=video_id,
            content_type="captions",
            content=captions,
            metadata=json.dumps({
                "format": "srt", 
                "language": "en",
                "engagement_score": score.overall_score,
                "sentiment": score.sentiment,
                "readability": score.readability_score
            }),
            version=self._get_next_version(video_id, "captions"),
        )
        self.db.save_content(content)

    def _save_blog_post(
        self, video_id: int, blog_post: str, original: Optional[str] = None
    ):
        """Save blog post with optional original version."""
        # Score content
        score = self.scorer.score_content(blog_post, "blog_post")
        
        metadata = {
            "word_count": len(blog_post.split()),
            "has_original": original is not None,
            "engagement_score": score.overall_score,
            "sentiment": score.sentiment,
            "readability": score.readability_score,
            "virality": score.virality_score,
            "recommended_platform": score.recommended_platform
        }

        if original:
            metadata["original_word_count"] = len(original.split())

        content = ContentOutput(
            video_id=video_id,
            content_type="blog_post",
            content=blog_post,
            metadata=json.dumps(metadata),
            version=self._get_next_version(video_id, "blog_post"),
        )
        self.db.save_content(content)

        # Save original if filtered
        if original and original != blog_post:
            original_content = ContentOutput(
                video_id=video_id,
                content_type="blog_post_original",
                content=original,
                metadata=json.dumps({"is_unfiltered": True}),
                version=self._get_next_version(video_id, "blog_post_original"),
            )
            self.db.save_content(original_content)

    def _save_social_post(self, video_id: int, social_post: str):
        """Save social media post."""
        # Score content
        score = self.scorer.score_content(social_post, "social_post")
        
        content = ContentOutput(
            video_id=video_id,
            content_type="social_post",
            content=social_post,
            metadata=json.dumps(
                {
                    "character_count": len(social_post),
                    "has_hashtags": "#" in social_post,
                    "engagement_score": score.overall_score,
                    "sentiment": score.sentiment,
                    "readability": score.readability_score,
                    "virality": score.virality_score,
                    "recommended_platform": score.recommended_platform
                }
            ),
            version=self._get_next_version(video_id, "social_post"),
        )
        self.db.save_content(content)

    def _save_shorts_ideas(self, video_id: int, shorts_ideas: List[Dict]):
        """Save short clip ideas."""
        for i, idea in enumerate(shorts_ideas):
            # Score each idea (topic + hook)
            content_to_score = f"{idea.get('topic', '')} {idea.get('hook', '')}"
            score = self.scorer.score_content(content_to_score, "shorts_idea")
            
            metadata = {
                "index": i,
                "topic": idea.get("topic", ""),
                "start_time": idea.get("start_time", ""),
                "end_time": idea.get("end_time", ""),
                "engagement_score": score.overall_score,
                "virality": score.virality_score
            }
            
            content = ContentOutput(
                video_id=video_id,
                content_type="shorts_idea",
                content=json.dumps(idea),
                metadata=json.dumps(metadata),
                version=self._get_next_version(video_id, "shorts_idea"),
                validation_status=idea.get("validation_status"),
                grounding_rate=idea.get("evidence_score"),
            )
            self.db.save_content(content)

    def _save_thumbnail_ideas(self, video_id: int, thumbnail_ideas: List[str]):
        """Save thumbnail generation prompts."""
        for i, idea in enumerate(thumbnail_ideas):
            content = ContentOutput(
                video_id=video_id,
                content_type="thumbnail_idea",
                content=idea,
                metadata=json.dumps({"index": i, "prompt_length": len(idea)}),
                version=self._get_next_version(video_id, "thumbnail_idea"),
            )
            self.db.save_content(content)

    def _save_grounding_report(self, video_id: int, grounding_metadata: Dict):
        """Save grounding validation report."""
        if not grounding_metadata.get("enabled"):
            return

        full_report = grounding_metadata.get("full_report", {})
        stats = full_report.get("statistics", {})

        # Count claims
        validation_results = full_report.get("validation_results", {})
        total_claims = 0
        verified_claims = 0

        for content_type, claims in validation_results.items():
            if isinstance(claims, list):
                total_claims += len(claims)
                verified_claims += sum(1 for c in claims if c.get("is_grounded"))

        report = GroundingReport(
            video_id=video_id,
            blog_grounding_rate=grounding_metadata.get("blog_grounding_rate", 0.0),
            social_grounding_rate=grounding_metadata.get("social_grounding_rate", 0.0),
            shorts_verification_rate=grounding_metadata.get(
                "shorts_verification_rate", 0.0
            ),
            total_claims=total_claims,
            verified_claims=verified_claims,
            unverified_claims=total_claims - verified_claims,
            full_report=json.dumps(full_report),
        )

        self.db.save_grounding_report(report)

    def _get_next_version(self, video_id: int, content_type: str) -> int:
        """Get the next version number for a content type."""
        versions = self.db.get_content_versions(video_id, content_type)
        if not versions:
            return 1
        return max(v.version for v in versions) + 1

    def load_video_results(self, video_id: int) -> Dict:
        """
        Load all results for a video.

        Args:
            video_id: Video ID

        Returns:
            Dictionary with all content organized by type
        """
        video = self.db.get_video(video_id)
        if not video:
            return {}

        # Get all content for video
        all_content = self.db.get_content_by_video(video_id)

        # Organize by type
        results = {
            "video": video.to_dict(),
            "captions": None,
            "blog_post": None,
            "blog_post_original": None,
            "social_post": None,
            "shorts_ideas": [],
            "thumbnail_ideas": [],
            "grounding_report": None,
        }

        for content in all_content:
            content_dict = content.to_dict()

            if content.content_type == "captions":
                results["captions"] = content.content

            elif content.content_type == "blog_post":
                results["blog_post"] = content.content

            elif content.content_type == "blog_post_original":
                results["blog_post_original"] = content.content

            elif content.content_type == "social_post":
                results["social_post"] = content.content

            elif content.content_type == "shorts_idea":
                try:
                    idea = json.loads(content.content)
                    idea["content_id"] = content.id
                    results["shorts_ideas"].append(idea)
                except:
                    pass

            elif content.content_type == "thumbnail_idea":
                results["thumbnail_ideas"].append(
                    {
                        "content_id": content.id,
                        "idea": content.content,
                        "metadata": content_dict["metadata"],
                    }
                )

        # Get grounding report
        report = self.db.get_grounding_report(video_id)
        if report:
            results["grounding_report"] = report.to_dict()

        return results

    def get_all_videos_summary(self, limit: int = 100) -> List[Dict]:
        """
        Get summary of all videos with basic stats.

        Returns:
            List of video summaries with content counts
        """
        videos = self.db.get_all_videos(limit=limit)
        summaries = []

        for video in videos:
            video_dict = video.to_dict()

            # Count content pieces
            contents = self.db.get_content_by_video(video.id)
            content_counts = {}
            for content in contents:
                content_counts[content.content_type] = (
                    content_counts.get(content.content_type, 0) + 1
                )

            video_dict["content_counts"] = content_counts
            video_dict["total_content"] = len(contents)

            summaries.append(video_dict)

        return summaries

    def search_content(
        self, query: str, content_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for content across all videos.

        Args:
            query: Search query
            content_type: Optional content type filter

        Returns:
            List of matching content items
        """
        # First search videos
        videos = self.db.search_videos(query)

        results = []
        for video in videos:
            contents = self.db.get_content_by_video(video.id, content_type)

            for content in contents:
                results.append(
                    {
                        "content_id": content.id,
                        "video_id": video.id,
                        "video_filename": video.filename,
                        "content_type": content.content_type,
                        "content_preview": (
                            content.content[:200] + "..."
                            if len(content.content) > 200
                            else content.content
                        ),
                        "created_at": content.created_at,
                        "platform": video.platform,
                    }
                )

        return results

    def delete_video_and_content(self, video_id: int):
        """Delete video and all associated content."""
        self.db.delete_video(video_id)
        print(f"✅ Video {video_id} and all content deleted")

    def get_statistics(self) -> Dict:
        """Get overall statistics."""
        return self.db.get_statistics()

    def get_recent_activity(self, limit: int = 10) -> List[Dict]:
        """Get recent content generations."""
        return self.db.get_recent_activity(limit)

    def export_video_results(self, video_id: int, export_path: str):
        """
        Export all results for a video to JSON file.

        Args:
            video_id: Video ID
            export_path: Path to save JSON file
        """
        results = self.load_video_results(video_id)

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✅ Results exported to: {export_path}")

    def import_video_results(self, json_path: str) -> int:
        """
        Import results from JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            video_id: ID of the imported video
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Create video record
        video_data = data.get("video", {})
        video = Video(
            filename=video_data.get("filename", "imported_video"),
            file_path=video_data.get("file_path", ""),
            file_size_mb=video_data.get("file_size_mb", 0.0),
            uploaded_at=datetime.now().isoformat(),
            platform=video_data.get("platform", "General"),
            grounding_enabled=video_data.get("grounding_enabled", True),
            processing_status="completed",
            searchable_text=video_data.get("searchable_text", "")
        )

        video_id = self.db.create_video(video)

        # Import all content
        if data.get("captions"):
            self._save_captions(video_id, data["captions"])

        if data.get("blog_post"):
            self._save_blog_post(
                video_id, data["blog_post"], data.get("blog_post_original")
            )

        if data.get("social_post"):
            self._save_social_post(video_id, data["social_post"])

        if data.get("shorts_ideas"):
            self._save_shorts_ideas(video_id, data["shorts_ideas"])

        if data.get("thumbnail_ideas"):
            ideas = [
                item["idea"] if isinstance(item, dict) else item
                for item in data["thumbnail_ideas"]
            ]
            self._save_thumbnail_ideas(video_id, ideas)

        print(f"✅ Results imported with video ID: {video_id}")
        return video_id

    def export_video_toolkit_zip(
        self, video_id: int, output_path: Optional[str] = None
    ) -> str:
        """
        Export all video assets (txt, JSON, thumbnails, clips metadata) to a ZIP file.

        Args:
            video_id: Video ID to export
            output_path: Optional path for the ZIP file. If None, creates in temp directory.

        Returns:
            str: Path to the created ZIP file
        """
        # Load all results
        results = self.load_video_results(video_id)

        if not results:
            raise ValueError(f"Video ID {video_id} not found")

        video = results["video"]
        video_filename = video["filename"].rsplit(".", 1)[0]  # Remove extension

        # Create output path if not provided
        if output_path is None:
            output_path = os.path.join(
                tempfile.gettempdir(), f"toolkit_{video_filename}_{video_id}.zip"
            )

        # Create ZIP file
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add README
            readme_content = f"""Creator Catalyst - Content Toolkit
====================================

Video: {video['filename']}
Platform: {video['platform']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Video ID: {video_id}

This toolkit contains all generated content assets for your video:
- captions.srt: Video transcript in SRT format
- blog_post.md: Full blog post content
- social_post.txt: Social media post
- shorts_ideas.json: Short clip ideas with timestamps
- thumbnail_ideas.txt: Thumbnail concept ideas
- metadata.json: Complete project metadata
- grounding_report.json: Fact verification report (if enabled)

"""
            zipf.writestr("README.txt", readme_content)

            # Add captions
            if results.get("captions"):
                zipf.writestr("captions.srt", results["captions"])

            # Add blog post
            if results.get("blog_post"):
                zipf.writestr("blog_post.md", results["blog_post"])

            # Add original blog post if available
            if results.get("blog_post_original"):
                zipf.writestr("blog_post_original.md", results["blog_post_original"])

            # Add social post
            if results.get("social_post"):
                zipf.writestr("social_post.txt", results["social_post"])

            # Add shorts ideas
            if results.get("shorts_ideas"):
                shorts_json = json.dumps(
                    results["shorts_ideas"], indent=2, ensure_ascii=False
                )
                zipf.writestr("shorts_ideas.json", shorts_json)

                # Also create a readable text version
                shorts_text = "Short Clip Ideas\n" + "=" * 50 + "\n\n"
                for i, short in enumerate(results["shorts_ideas"]):
                    shorts_text += f"Idea {i+1}: {short.get('topic', 'N/A')}\n"
                    shorts_text += f"Timestamps: {short.get('start_time', 'N/A')} - {short.get('end_time', 'N/A')}\n"
                    shorts_text += f"Summary: {short.get('summary', 'N/A')}\n"
                    if short.get("supporting_text"):
                        shorts_text += (
                            f"Evidence: {short['supporting_text'][:200]}...\n"
                        )
                    shorts_text += "\n" + "-" * 50 + "\n\n"
                zipf.writestr("shorts_ideas.txt", shorts_text)

            # Add thumbnail ideas
            if results.get("thumbnail_ideas"):
                thumbnail_text = "Thumbnail Ideas\n" + "=" * 50 + "\n\n"
                for i, thumb in enumerate(results["thumbnail_ideas"]):
                    idea_text = thumb["idea"] if isinstance(thumb, dict) else thumb
                    thumbnail_text += f"{i+1}. {idea_text}\n\n"
                zipf.writestr("thumbnail_ideas.txt", thumbnail_text)

            # Add complete metadata JSON
            metadata = {
                "video": video,
                "export_date": datetime.now().isoformat(),
                "content_summary": {
                    "has_captions": bool(results.get("captions")),
                    "has_blog_post": bool(results.get("blog_post")),
                    "has_social_post": bool(results.get("social_post")),
                    "shorts_count": len(results.get("shorts_ideas", [])),
                    "thumbnails_count": len(results.get("thumbnail_ideas", [])),
                },
            }
            zipf.writestr(
                "metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False)
            )

            # Add grounding report if available
            if results.get("grounding_report"):
                report = results["grounding_report"]
                report_text = f"""Fact-Grounding Report
======================

Blog Grounding Rate: {report['blog_grounding_rate']:.0%}
Social Grounding Rate: {report['social_grounding_rate']:.0%}
Shorts Verification Rate: {report['shorts_verification_rate']:.0%}

Total Claims: {report['total_claims']}
Verified Claims: {report['verified_claims']} ✅
Unverified Claims: {report['unverified_claims']} ❌

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                zipf.writestr("grounding_report.txt", report_text)

                # Also add full JSON report
                zipf.writestr(
                    "grounding_report.json",
                    json.dumps(report, indent=2, ensure_ascii=False),
                )

        print(f"✅ Toolkit ZIP created: {output_path}")
        return output_path


# Singleton instance
_storage_instance = None


def get_storage_manager(db_path: str = "creator_catalyst.db") -> StorageManager:
    """Get or create storage manager singleton instance."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = StorageManager(db_path)
    return _storage_instance