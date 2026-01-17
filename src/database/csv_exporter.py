"""
CSV Export Module for Creator Catalyst
Exports content with keywords in CSV format.
"""

import csv
import io
from typing import List, Dict, Optional
from datetime import datetime


class CSVExporter:
    """Export content analysis results including keywords to CSV."""

    @staticmethod
    def export_keywords_csv(
        keywords_dict: Dict[str, List[str]],
        platform: str = "General",
        title: str = "Analysis",
    ) -> bytes:
        """
        Export keywords from all content types to CSV.

        Args:
            keywords_dict: Dict with content types and keyword lists
            platform: Target platform name
            title: Content title

        Returns:
            CSV data as bytes
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["Creator Catalyst - Keywords Export"])
        writer.writerow(["Title", title])
        writer.writerow(["Platform", platform])
        writer.writerow(["Generated", datetime.now().isoformat()])
        writer.writerow([])

        # Extract all unique keywords with sources
        all_keywords = {}
        for content_type, keywords in keywords_dict.items():
            for keyword in keywords:
                if keyword not in all_keywords:
                    all_keywords[keyword] = []
                all_keywords[keyword].append(content_type.title())

        # Write keywords sorted by frequency of appearance
        writer.writerow(["Rank", "Keyword", "Found In"])
        for rank, (keyword, sources) in enumerate(all_keywords.items(), 1):
            writer.writerow([rank, keyword, "; ".join(sources)])

        writer.writerow([])
        writer.writerow(["Summary by Content Type"])
        writer.writerow(["Content Type", "Keyword Count", "Keywords"])

        for content_type, keywords in keywords_dict.items():
            writer.writerow([content_type.title(), len(keywords), "; ".join(keywords)])

        return output.getvalue().encode("utf-8")

    @staticmethod
    def export_content_with_keywords(
        content_dict: Dict[str, Dict], platform: str = "General"
    ) -> bytes:
        """
        Export full content with keywords.

        Args:
            content_dict: {
                'blog': {'title': '...', 'content': '...', 'keywords': [...]},
                'social': {'title': '...', 'content': '...', 'keywords': [...]}
            }
            platform: Target platform

        Returns:
            CSV data as bytes
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["Creator Catalyst - Full Content Export with Keywords"])
        writer.writerow(["Platform", platform])
        writer.writerow(["Generated", datetime.now().isoformat()])
        writer.writerow([])

        for content_type, data in content_dict.items():
            writer.writerow([f"{content_type.upper()}"])
            writer.writerow(["Title", data.get("title", "N/A")])

            keywords = data.get("keywords", [])
            writer.writerow(["Keywords", "; ".join(keywords)])
            writer.writerow(["Keyword Count", len(keywords)])

            writer.writerow(["Content Preview"])
            content = data.get("content", "")
            # Take first 300 chars as preview
            preview = content[:300] + "..." if len(content) > 300 else content
            writer.writerow([preview])

            writer.writerow([])
            writer.writerow(["---"])
            writer.writerow([])

        return output.getvalue().encode("utf-8")


def get_csv_exporter() -> CSVExporter:
    """Factory function to get a CSV exporter instance."""
    return CSVExporter()
