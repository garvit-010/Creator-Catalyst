"""
Keyword Extraction Module for Creator Catalyst
Extracts relevant keywords from generated content for SEO optimization.
Uses LLM-based approach with optional NLP fallback.
"""

import json
import re
import logging
from typing import List, Optional, Dict

# Initialize logger
logger = logging.getLogger(__name__)

try:
    from src.core.llm_wrapper import LLMWrapper

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class KeywordExtractor:
    """
    Extracts SEO-optimized keywords from content using LLM.
    Falls back to statistical keyword extraction if LLM unavailable.
    """

    def __init__(self):
        """Initialize the keyword extractor."""
        self.llm_client = None

        if LLM_AVAILABLE:
            try:
                self.llm_client = LLMWrapper()
            except Exception as e:
                logger.warning(f"LLM initialization failed for keyword extraction: {e}")

    def extract_keywords(
        self, text: str, num_keywords: int = 8, content_type: str = "general"
    ) -> List[str]:
        """
        Extract keywords from text using LLM.

        Args:
            text: The text to extract keywords from
            num_keywords: Number of keywords to extract (5-10 recommended)
            content_type: Type of content ('blog', 'social', 'shorts', 'general')

        Returns:
            List of extracted keywords sorted by relevance
        """
        if not text or len(text.strip()) < 20:
            return []

        # Ensure num_keywords is in valid range
        num_keywords = max(5, min(10, num_keywords))

        # Try LLM-based extraction
        if self.llm_client:
            keywords = self._extract_with_llm(text, num_keywords, content_type)
            if keywords:
                return keywords

        # Fallback to simple extraction
        return self._extract_simple(text, num_keywords)

    def _extract_with_llm(
        self, text: str, num_keywords: int, content_type: str
    ) -> Optional[List[str]]:
        """Extract keywords using LLM."""
        try:
            context_guidance = {
                "blog": "technical, educational, SEO-friendly keywords for search engines",
                "social": "trending, catchy, viral keywords for social media engagement",
                "shorts": "hook words, quick concepts, trending hashtag-friendly keywords",
                "general": "relevant, searchable, high-impact keywords",
            }

            guidance = context_guidance.get(content_type, context_guidance["general"])

            # Truncate text if too long
            text_preview = text[:1500] if len(text) > 1500 else text

            prompt = f"""Extract exactly {num_keywords} SEO-optimized keywords from this {content_type} content.

Focus on: {guidance}

Content:
{text_preview}

Return ONLY a valid JSON array of keywords in lowercase, nothing else. 
Example response: ["keyword1", "keyword2", "keyword3"]

Respond with ONLY the JSON array:"""

            response = self.llm_client.generate(prompt)

            if response:
                # Parse JSON response
                try:
                    # Clean response
                    cleaned = response.strip()

                    # Remove markdown code blocks if present
                    if "```" in cleaned:
                        cleaned = cleaned.split("```")[1].strip()

                    # Remove 'json' prefix if present
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:].strip()

                    # Parse JSON
                    keywords = json.loads(cleaned)

                    # Validate and clean
                    if isinstance(keywords, list):
                        keywords = [
                            kw.lower().strip()
                            for kw in keywords
                            if isinstance(kw, str) and kw.strip()
                        ]
                        # Remove duplicates while preserving order
                        seen = set()
                        unique = []
                        for kw in keywords:
                            if kw not in seen:
                                unique.append(kw)
                                seen.add(kw)
                        return unique[:num_keywords]
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse LLM response for keywords: {e}")
                    return None
        except Exception as e:
            logger.error(f"LLM keyword extraction failed: {e}")
            return None

        return None

    def _extract_simple(self, text: str, num_keywords: int) -> List[str]:
        """Simple keyword extraction as fallback."""
        # Basic stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
            "as",
            "from",
            "to",
            "of",
            "in",
            "on",
            "at",
            "by",
            "with",
            "for",
            "as",
            "it",
            "just",
            "than",
            "so",
            "up",
            "out",
            "if",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
        }

        # Extract words (3+ characters)
        words = re.findall(r"\b\w+\b", text.lower())
        words = [w for w in words if w not in stop_words and len(w) >= 3]

        # Get unique words preserving frequency
        from collections import Counter

        word_freq = Counter(words)
        keywords = [word for word, _ in word_freq.most_common(num_keywords)]

        return keywords

    def extract_from_multiple(
        self, contents: Dict[str, str], num_keywords: int = 8
    ) -> Dict[str, List[str]]:
        """
        Extract keywords from multiple content pieces.

        Args:
            contents: Dict with keys like 'blog', 'social', 'shorts' and text values
            num_keywords: Number of keywords per content

        Returns:
            Dict with same keys and keyword lists as values
        """
        results = {}

        for content_type, text in contents.items():
            keywords = self.extract_keywords(text, num_keywords, content_type)
            results[content_type] = keywords

        return results


def get_keyword_extractor() -> KeywordExtractor:
    """Factory function to get a keyword extractor instance."""
    return KeywordExtractor()
