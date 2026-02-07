"""
Title Generator for Creator Catalyst
Generates catchy, engaging titles for videos and shorts using AI.
"""

import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

# Initialize logger
logger = logging.getLogger(__name__)


@dataclass
class TitleSuggestion:
    """Represents a single title suggestion with metadata."""
    title: str
    style: str  # e.g., "curiosity", "listicle", "how-to", "secret"
    hook_type: str  # e.g., "question", "number", "power_word", "mystery"
    estimated_ctr: str  # "high", "medium", "low"
    
    def to_dict(self) -> dict:
        return {
            'title': self.title,
            'style': self.style,
            'hook_type': self.hook_type,
            'estimated_ctr': self.estimated_ctr
        }


@dataclass  
class TitleGenerationResult:
    """Contains all generated titles for a video/short."""
    original_titles: List[TitleSuggestion]
    shorts_titles: Dict[int, List[TitleSuggestion]]  # short_index -> titles
    selected_original_title: Optional[str] = None
    selected_shorts_titles: Optional[Dict[int, str]] = None
    
    def to_dict(self) -> dict:
        return {
            'original_titles': [t.to_dict() for t in self.original_titles],
            'shorts_titles': {
                k: [t.to_dict() for t in v] 
                for k, v in self.shorts_titles.items()
            },
            'selected_original_title': self.selected_original_title,
            'selected_shorts_titles': self.selected_shorts_titles
        }


class TitleGenerator:
    """
    Generates catchy, engaging titles for videos and shorts.
    Uses proven title formulas that drive clicks and views.
    """
    
    # Title formulas that work well on different platforms
    TITLE_FORMULAS = {
        "curiosity": [
            "{topic} You'll Wish You Knew Sooner",
            "The Truth About {topic} Nobody Tells You",
            "What {topic} Really Means (Explained)",
            "{topic} – Here's What You're Missing",
            "Why {topic} Changes Everything"
        ],
        "listicle": [
            "{number} {topic} Tips You Need Right Now",
            "{number} Hidden {topic} Secrets Revealed",
            "{number} {topic} Mistakes Everyone Makes",
            "Top {number} {topic} Hacks That Actually Work",
            "{number} {topic} Rules Successful People Follow"
        ],
        "how_to": [
            "How to Master {topic} in Minutes",
            "The Fastest Way to {topic}",
            "{topic} Made Simple (Step-by-Step)",
            "The Ultimate Guide to {topic}",
            "How I {topic} (And You Can Too)"
        ],
        "secret": [
            "Secret {topic} Strategy (Explained)",
            "The {topic} Trick Pros Don't Share",
            "{topic} Secrets They Don't Want You to Know",
            "Hidden {topic} Technique Revealed",
            "The Underground {topic} Method"
        ],
        "urgency": [
            "Watch This Before You {topic}",
            "{topic} – Don't Make This Mistake",
            "Stop {topic} Wrong – Here's How",
            "{topic} Warning: Most People Get This Wrong",
            "You're {topic} Wrong (Here's Why)"
        ],
        "results": [
            "Boost {topic} Fast – Watch Till the End",
            "{topic} Results in 24 Hours",
            "I Tried {topic} for 30 Days (Results)",
            "{topic} That Actually Works",
            "Finally! {topic} That Gets Results"
        ]
    }
    
    POWER_WORDS = [
        "secret", "hidden", "ultimate", "proven", "shocking",
        "essential", "powerful", "game-changing", "instant", "easy",
        "free", "exclusive", "guaranteed", "breakthrough", "incredible"
    ]
    
    HOOK_NUMBERS = ["3", "5", "7", "10"]  # Numbers that perform well
    
    def __init__(self, llm_wrapper=None):
        """
        Initialize the title generator.
        
        Args:
            llm_wrapper: Optional LLMWrapper instance for AI-powered generation
        """
        self.llm_wrapper = llm_wrapper
        
    def generate_titles_for_video(
        self,
        video_summary: str,
        platform: str = "YouTube",
        num_titles: int = 3
    ) -> List[TitleSuggestion]:
        """
        Generate catchy titles for the main video.
        
        Args:
            video_summary: Brief summary or topic of the video
            platform: Target platform (YouTube, TikTok, etc.)
            num_titles: Number of titles to generate
            
        Returns:
            List of TitleSuggestion objects
        """
        if self.llm_wrapper:
            return self._generate_titles_with_ai(
                content=video_summary,
                content_type="video",
                platform=platform,
                num_titles=num_titles
            )
        else:
            return self._generate_titles_formulaic(
                topic=video_summary,
                num_titles=num_titles
            )
    
    def generate_titles_for_short(
        self,
        short_summary: str,
        short_topic: str,
        platform: str = "YouTube",
        num_titles: int = 3
    ) -> List[TitleSuggestion]:
        """
        Generate catchy titles for a short/clip.
        
        Args:
            short_summary: Summary of the short
            short_topic: Topic/title of the short
            platform: Target platform
            num_titles: Number of titles to generate
            
        Returns:
            List of TitleSuggestion objects
        """
        combined_context = f"{short_topic}: {short_summary}"
        
        if self.llm_wrapper:
            return self._generate_titles_with_ai(
                content=combined_context,
                content_type="short",
                platform=platform,
                num_titles=num_titles
            )
        else:
            return self._generate_titles_formulaic(
                topic=short_topic,
                num_titles=num_titles
            )
    
    def _generate_titles_with_ai(
        self,
        content: str,
        content_type: str,
        platform: str,
        num_titles: int
    ) -> List[TitleSuggestion]:
        """Generate titles using AI."""
        
        platform_guidance = self._get_platform_guidance(platform)
        
        prompt = f"""You are a viral content title expert specializing in {platform}. 
Generate {num_titles} catchy, click-worthy titles for this {content_type}.

Content: {content}

{platform_guidance}

TITLE FORMULA EXAMPLES (use these as inspiration):
- "5 Hidden [Topic] Tips You'll Wish You Knew"
- "Boost [Topic] Fast – Watch Till the End"  
- "Secret [Topic] Strategy (Explained)"
- "The Truth About [Topic] Nobody Tells You"
- "Why [Topic] Changes Everything"

RULES:
1. Each title must be unique and engaging
2. Use power words: secret, hidden, ultimate, proven, shocking, essential
3. Consider using numbers (3, 5, 7, 10 work best)
4. Create curiosity gaps - make viewers want to know more
5. Keep titles under 60 characters for YouTube, under 100 for other platforms
6. Avoid clickbait that doesn't deliver - titles must relate to actual content

OUTPUT FORMAT (exactly this format, one per line):
TITLE: [your title here] | STYLE: [curiosity/listicle/how_to/secret/urgency/results] | HOOK: [question/number/power_word/mystery] | CTR: [high/medium/low]

Generate {num_titles} titles now:"""

        try:
            response = self.llm_wrapper.generate_text(prompt)
            return self._parse_ai_titles(response, num_titles)
        except Exception as e:
            logger.error(f"AI title generation failed: {e}, falling back to formulaic")
            # Use the first 50 chars of content as the topic for fallback
            topic = content[:50].split('\n')[0]
            return self._generate_titles_formulaic(topic, num_titles)
    
    def _parse_ai_titles(self, response: str, expected_count: int) -> List[TitleSuggestion]:
        """Parse AI-generated titles from response."""
        titles = []
        
        # Pattern to match: TITLE: ... | STYLE: ... | HOOK: ... | CTR: ...
        pattern = r"TITLE:\s*(.+?)\s*\|\s*STYLE:\s*(\w+)\s*\|\s*HOOK:\s*(\w+)\s*\|\s*CTR:\s*(\w+)"
        
        matches = re.findall(pattern, response, re.IGNORECASE)
        
        for match in matches[:expected_count]:
            title_text, style, hook, ctr = match
            titles.append(TitleSuggestion(
                title=title_text.strip(),
                style=style.lower().strip(),
                hook_type=hook.lower().strip(),
                estimated_ctr=ctr.lower().strip()
            ))
        
        # If parsing failed or not enough titles, add formulaic ones
        if len(titles) < expected_count:
            # Try simpler parsing - just look for numbered titles
            lines = response.strip().split('\n')
            for line in lines:
                if len(titles) >= expected_count:
                    break
                # Look for lines that might be titles
                line = line.strip()
                if line and not line.startswith('#') and len(line) > 10:
                    # Extract title from various formats
                    title_match = re.search(r'(?:TITLE:|^\d+[\.\)]\s*|^-\s*)(.+?)(?:\||$)', line)
                    if title_match:
                        title_text = title_match.group(1).strip()
                        if title_text and title_text not in [t.title for t in titles]:
                            titles.append(TitleSuggestion(
                                title=title_text,
                                style="curiosity",
                                hook_type="power_word",
                                estimated_ctr="medium"
                            ))
        
        return titles
    
    def _generate_titles_formulaic(
        self, 
        topic: str, 
        num_titles: int
    ) -> List[TitleSuggestion]:
        """Generate titles using proven formulas (no AI)."""
        titles = []
        
        # Clean and shorten topic
        topic_clean = self._extract_core_topic(topic)
        
        # Generate one title from each formula category
        formulas = list(self.TITLE_FORMULAS.items())
        
        for i in range(min(num_titles, len(formulas))):
            style, templates = formulas[i]
            template = templates[i % len(templates)]
            
            # Fill in template
            title = template.format(
                topic=topic_clean,
                number=self.HOOK_NUMBERS[i % len(self.HOOK_NUMBERS)]
            )
            
            hook_type = "number" if "{number}" in template else "power_word"
            
            titles.append(TitleSuggestion(
                title=title,
                style=style,
                hook_type=hook_type,
                estimated_ctr="medium"
            ))
        
        return titles[:num_titles]
    
    def _extract_core_topic(self, text: str) -> str:
        """Extract a short, usable topic from longer text."""
        # Take first sentence or first 50 chars
        text = text.strip()
        
        # Remove common prefixes
        text = re.sub(r'^(this video is about|in this video|today we|let me show you)\s*', '', text, flags=re.IGNORECASE)
        
        # Get first sentence
        sentences = re.split(r'[.!?]', text)
        if sentences:
            topic = sentences[0].strip()
        else:
            topic = text
        
        # Truncate if too long
        if len(topic) > 40:
            words = topic.split()[:5]
            topic = ' '.join(words)
        
        return topic.title() if topic else "Video Content"
    
    def _get_platform_guidance(self, platform: str) -> str:
        """Get platform-specific title guidance."""
        guidance = {
            "YouTube": """
YOUTUBE OPTIMIZATION:
- Optimal length: 40-60 characters
- Front-load keywords
- Use brackets for context: (Tutorial), [2024], etc.
- Capitalize important words
- Emojis can work but use sparingly""",
            
            "TikTok": """
TIKTOK OPTIMIZATION:
- Can be longer (up to 100 chars)
- Very casual, trendy language
- Use trending phrases when relevant
- Hook in first 3 words
- Emojis are common and effective""",
            
            "Instagram": """
INSTAGRAM OPTIMIZATION:
- Keep it punchy and visual
- Emojis work well
- Aesthetic and lifestyle focus
- Can use more creative formatting""",
            
            "Twitter/X": """
TWITTER/X OPTIMIZATION:
- Concise and punchy
- Leave room for engagement
- Can be provocative/bold
- Numbers and stats work well""",
            
            "LinkedIn": """
LINKEDIN OPTIMIZATION:
- Professional but engaging
- Focus on value/insights
- Avoid clickbait
- Use industry keywords"""
        }
        
        return guidance.get(platform, guidance.get("YouTube", ""))
    
    def generate_all_titles(
        self,
        video_summary: str,
        shorts_ideas: List[Dict],
        platform: str = "YouTube"
    ) -> TitleGenerationResult:
        """
        Generate titles for both the main video and all shorts.
        
        Args:
            video_summary: Summary of the main video (from blog post or captions)
            shorts_ideas: List of shorts ideas with 'topic' and 'summary' keys
            platform: Target platform
            
        Returns:
            TitleGenerationResult with all generated titles
        """
        # Generate titles for main video
        original_titles = self.generate_titles_for_video(
            video_summary=video_summary,
            platform=platform,
            num_titles=3
        )
        
        # Generate titles for each short
        shorts_titles = {}
        for i, short in enumerate(shorts_ideas):
            topic = short.get('topic', f'Short {i+1}')
            summary = short.get('summary', '')
            
            shorts_titles[i] = self.generate_titles_for_short(
                short_summary=summary,
                short_topic=topic,
                platform=platform,
                num_titles=3
            )
        
        return TitleGenerationResult(
            original_titles=original_titles,
            shorts_titles=shorts_titles
        )


# Singleton instance
_title_generator_instance = None


def get_title_generator(llm_wrapper=None) -> TitleGenerator:
    """Get or create the title generator singleton."""
    global _title_generator_instance
    
    if _title_generator_instance is None:
        _title_generator_instance = TitleGenerator(llm_wrapper)
    elif llm_wrapper and _title_generator_instance.llm_wrapper is None:
        _title_generator_instance.llm_wrapper = llm_wrapper
        
    return _title_generator_instance
