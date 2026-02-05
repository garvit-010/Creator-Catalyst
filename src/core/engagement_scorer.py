"""
Engagement Scoring System for Creator Catalyst
Predicts engagement potential for posts/captions using AI analysis.
Provides platform-specific recommendations and actionable insights.
"""

import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Download required NLTK data
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


@dataclass
class EngagementScore:
    """Represents an engagement score with detailed breakdown."""
    overall_score: int  # 0-100
    platform_scores: Dict[str, int]  # Platform-specific scores
    recommended_platform: str
    strengths: List[str]
    improvements: List[str]
    engagement_factors: Dict[str, float]  # Detailed factor breakdown
    sentiment: Dict[str, float]  # Sentiment analysis results
    readability_score: float  # Flesch-Kincaid Ease or similar
    virality_score: float  # Predicted virality potential
    optimal_posting_time: Optional[str] = None


class EngagementScorer:
    """
    AI-powered engagement prediction system.
    Analyzes posts/captions and predicts performance across platforms.
    """
    
    # Platform characteristics and weights
    PLATFORM_WEIGHTS = {
        'Instagram': {
            'visual_appeal': 0.20,
            'hashtag_usage': 0.15,
            'emoji_presence': 0.10,
            'length_optimal': 0.10,
            'call_to_action': 0.10,
            'storytelling': 0.10,
            'sentiment_impact': 0.15,
            'virality_factor': 0.10
        },
        'Twitter/X': {
            'brevity': 0.20,
            'trending_potential': 0.15,
            'hashtag_usage': 0.10,
            'engagement_hooks': 0.15,
            'thread_worthy': 0.10,
            'controversy_factor': 0.10,
            'sentiment_impact': 0.10,
            'virality_factor': 0.10
        },
        'LinkedIn': {
            'professionalism': 0.20,
            'value_proposition': 0.20,
            'thought_leadership': 0.15,
            'length_optimal': 0.10,
            'call_to_action': 0.10,
            'industry_relevance': 0.05,
            'readability_impact': 0.10,
            'sentiment_impact': 0.10
        },
        'TikTok': {
            'hook_strength': 0.25,
            'trend_alignment': 0.20,
            'emoji_presence': 0.10,
            'brevity': 0.10,
            'viral_potential': 0.20,
            'sentiment_impact': 0.15
        },
        'YouTube': {
            'storytelling': 0.25,
            'seo_optimization': 0.15,
            'value_proposition': 0.15,
            'length_optimal': 0.10,
            'call_to_action': 0.10,
            'virality_factor': 0.15,
            'sentiment_impact': 0.10
        }
    }
    
    def __init__(self):
        """Initialize the engagement scorer."""
        self.sia = SentimentIntensityAnalyzer()
    
    def score_content(
        self, 
        content: str, 
        content_type: str = "social_post",
        target_platform: Optional[str] = None
    ) -> EngagementScore:
        """
        Score content for predicted engagement.
        
        Args:
            content: The post/caption text to analyze
            content_type: Type of content (social_post, blog_post, etc.)
            target_platform: Optional specific platform to optimize for
            
        Returns:
            EngagementScore with detailed analysis
        """
        # Analyze content characteristics
        factors = self._analyze_content_factors(content)
        
        # New Deep Metrics
        sentiment = self._analyze_sentiment(content)
        readability = self._analyze_readability(content)
        virality = self._analyze_virality(content, factors)
        
        # Add deep metrics to factors for scoring
        factors['sentiment_impact'] = sentiment['compound'] * 0.5 + 0.5  # Normalize to 0-1
        factors['readability_impact'] = readability / 100.0
        factors['virality_factor'] = virality
        
        # Calculate platform-specific scores
        platform_scores = {}
        for platform in self.PLATFORM_WEIGHTS.keys():
            score = self._calculate_platform_score(factors, platform)
            platform_scores[platform] = score
        
        # Determine best platform
        if target_platform and target_platform in platform_scores:
            recommended_platform = target_platform
            overall_score = platform_scores[target_platform]
        else:
            recommended_platform = max(platform_scores, key=platform_scores.get)
            overall_score = platform_scores[recommended_platform]
        
        # Generate insights
        strengths = self._identify_strengths(factors, recommended_platform)
        improvements = self._identify_improvements(factors, recommended_platform)
        
        return EngagementScore(
            overall_score=overall_score,
            platform_scores=platform_scores,
            recommended_platform=recommended_platform,
            strengths=strengths,
            improvements=improvements,
            engagement_factors=factors,
            sentiment=sentiment,
            readability_score=readability,
            virality_score=virality * 100,
            optimal_posting_time=self._suggest_posting_time(recommended_platform)
        )
    
    def _analyze_sentiment(self, content: str) -> Dict[str, float]:
        """Perform sentiment analysis on content."""
        return self.sia.polarity_scores(content)

    def _analyze_readability(self, content: str) -> float:
        """Calculate a basic Flesch Reading Ease score."""
        sentences = nltk.sent_tokenize(content)
        words = content.split()
        
        if not words or not sentences:
            return 0.0
            
        avg_sentence_length = len(words) / len(sentences)
        
        # Simple syllable counting (approximate)
        def count_syllables(word):
            word = word.lower()
            count = 0
            vowels = "aeiouy"
            if word[0] in vowels:
                count += 1
            for index in range(1, len(word)):
                if word[index] in vowels and word[index - 1] not in vowels:
                    count += 1
            if word.endswith("e"):
                count -= 1
            if count == 0:
                count = 1
            return count
            
        total_syllables = sum(count_syllables(w) for w in words)
        avg_syllables_per_word = total_syllables / len(words)
        
        # Flesch Reading Ease formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(min(score, 100.0), 0.0)

    def _analyze_virality(self, content: str, factors: Dict[str, float]) -> float:
        """Predict virality potential based on multiple factors."""
        virality_score = 0.0
        
        # Base from existing viral potential
        virality_score += factors.get('viral_potential', 0) * 0.4
        
        # Hook strength is crucial for virality
        virality_score += factors.get('hook_strength', 0) * 0.3
        
        # Controversy/High emotion (sentiment extremity)
        sentiment = self._analyze_sentiment(content)
        extremity = abs(sentiment['compound'])
        virality_score += extremity * 0.2
        
        # Brevity (short, punchy content often goes viral)
        virality_score += factors.get('brevity', 0) * 0.1
        
        return min(virality_score, 1.0)

    def _analyze_content_factors(self, content: str) -> Dict[str, float]:
        """
        Analyze various engagement factors in content.
        Returns scores 0.0-1.0 for each factor.
        """
        factors = {}
        
        # Length analysis
        word_count = len(content.split())
        char_count = len(content)
        
        factors['brevity'] = self._score_brevity(char_count)
        factors['length_optimal'] = self._score_length_optimal(word_count)
        
        # Hashtag analysis
        hashtags = re.findall(r'#\w+', content)
        factors['hashtag_usage'] = self._score_hashtag_usage(len(hashtags), char_count)
        
        # Emoji analysis
        emoji_count = len(re.findall(r'[😀-🙏🌀-🗿🚀-🛿]', content))
        factors['emoji_presence'] = min(emoji_count / 5, 1.0)
        
        # Call to action detection
        cta_patterns = [
            r'\b(click|tap|swipe|comment|share|tag|follow|subscribe|visit|check out|learn more|dm|link in bio)\b',
            r'[!?]{2,}',  # Multiple punctuation
            r'\b(now|today|limited|exclusive|free|new)\b'
        ]
        cta_score = 0
        for pattern in cta_patterns:
            if re.search(pattern, content.lower()):
                cta_score += 0.3
        factors['call_to_action'] = min(cta_score, 1.0)
        
        # Engagement hooks (questions, bold claims)
        hook_score = 0
        if '?' in content:
            hook_score += 0.4
        if any(word in content.lower() for word in ['secret', 'mistake', 'truth', 'nobody', 'never', 'always', 'proven']):
            hook_score += 0.3
        if content.startswith(('Imagine', 'What if', 'Here\'s', 'Did you know', 'The truth')):
            hook_score += 0.3
        factors['engagement_hooks'] = min(hook_score, 1.0)
        
        # Storytelling elements
        story_indicators = ['when', 'then', 'after', 'before', 'finally', 'suddenly', 'realized', 'learned']
        story_score = sum(0.15 for word in story_indicators if word in content.lower())
        factors['storytelling'] = min(story_score, 1.0)
        
        # Professionalism (for LinkedIn)
        professional_terms = ['strategy', 'growth', 'insights', 'industry', 'professional', 'experience', 'team', 'project']
        prof_score = sum(0.15 for term in professional_terms if term in content.lower())
        factors['professionalism'] = min(prof_score, 1.0)
        
        # Value proposition
        value_keywords = ['how to', 'tips', 'guide', 'benefits', 'results', 'proven', 'effective', 'increase', 'improve']
        value_score = sum(0.15 for keyword in value_keywords if keyword in content.lower())
        factors['value_proposition'] = min(value_score, 1.0)
        
        # Viral potential indicators
        viral_patterns = ['breaking', 'shocking', 'unbelievable', 'mind-blowing', 'game-changer', 'you won\'t believe']
        viral_score = sum(0.2 for pattern in viral_patterns if pattern in content.lower())
        factors['viral_potential'] = min(viral_score, 1.0)
        
        # Hook strength (first 5 words)
        first_words = ' '.join(content.split()[:5]).lower()
        hook_strength = 0.5  # baseline
        if any(word in first_words for word in ['stop', 'wait', 'attention', 'breaking', 'urgent', 'new']):
            hook_strength = 0.9
        elif any(word in first_words for word in ['here\'s', 'this', 'the', 'i', 'you']):
            hook_strength = 0.7
        factors['hook_strength'] = hook_strength
        
        # Thread-worthy (for Twitter)
        factors['thread_worthy'] = 0.8 if word_count > 100 else 0.3
        
        # Trending potential
        factors['trending_potential'] = factors['viral_potential'] * 0.7 + factors['engagement_hooks'] * 0.3
        
        # Visual appeal (based on formatting)
        has_line_breaks = '\n' in content
        has_formatting = bool(re.search(r'[*_`]', content))
        factors['visual_appeal'] = 0.5 + (0.3 if has_line_breaks else 0) + (0.2 if has_formatting else 0)
        
        # SEO optimization (keyword density, structure)
        factors['seo_optimization'] = 0.6 if word_count > 50 and len(hashtags) > 0 else 0.4
        
        # Thought leadership
        factors['thought_leadership'] = (factors['professionalism'] + factors['value_proposition']) / 2
        
        # Industry relevance
        factors['industry_relevance'] = factors['professionalism'] * 0.8
        
        # Controversy factor (for Twitter)
        controversy_words = ['debate', 'unpopular opinion', 'hot take', 'controversial', 'disagree']
        controversy_score = sum(0.25 for word in controversy_words if word in content.lower())
        factors['controversy_factor'] = min(controversy_score, 1.0)
        
        # Trend alignment (for TikTok)
        factors['trend_alignment'] = (factors['viral_potential'] + factors['emoji_presence']) / 2
        
        return factors
    
    def _score_brevity(self, char_count: int) -> float:
        """Score content brevity (shorter is better for some platforms)."""
        if char_count <= 100:
            return 1.0
        elif char_count <= 200:
            return 0.8
        elif char_count <= 280:
            return 0.6
        else:
            return 0.3
    
    def _score_length_optimal(self, word_count: int) -> float:
        """Score optimal length (platform-dependent sweet spot)."""
        if 20 <= word_count <= 80:
            return 1.0
        elif 10 <= word_count <= 150:
            return 0.7
        else:
            return 0.4
    
    def _score_hashtag_usage(self, hashtag_count: int, char_count: int) -> float:
        """Score hashtag usage quality."""
        if hashtag_count == 0:
            return 0.2
        elif 1 <= hashtag_count <= 3:
            return 1.0
        elif 4 <= hashtag_count <= 7:
            return 0.8
        else:
            # Too many hashtags
            return 0.4
    
    def _calculate_platform_score(self, factors: Dict[str, float], platform: str) -> int:
        """Calculate weighted score for a specific platform."""
        weights = self.PLATFORM_WEIGHTS.get(platform, {})
        
        score = 0.0
        for factor, weight in weights.items():
            factor_value = factors.get(factor, 0.5)  # Default to 0.5 if factor missing
            score += factor_value * weight
        
        # Convert to 0-100 scale
        return int(score * 100)
    
    def _identify_strengths(self, factors: Dict[str, float], platform: str) -> List[str]:
        """Identify top strengths of the content."""
        weights = self.PLATFORM_WEIGHTS.get(platform, {})
        
        strengths = []
        for factor, weight in weights.items():
            factor_value = factors.get(factor, 0)
            if factor_value >= 0.7:  # Strong factor
                strength_message = self._factor_to_message(factor, factor_value, True)
                strengths.append(strength_message)
        
        return strengths[:3]  # Top 3 strengths
    
    def _identify_improvements(self, factors: Dict[str, float], platform: str) -> List[str]:
        """Identify areas for improvement."""
        weights = self.PLATFORM_WEIGHTS.get(platform, {})
        
        improvements = []
        for factor, weight in weights.items():
            factor_value = factors.get(factor, 0)
            if factor_value < 0.5 and weight > 0.15:  # Important weak factor
                improvement_message = self._factor_to_message(factor, factor_value, False)
                improvements.append(improvement_message)
        
        return improvements[:3]  # Top 3 improvements
    
    def _factor_to_message(self, factor: str, value: float, is_strength: bool) -> str:
        """Convert factor to human-readable message."""
        messages = {
            'brevity': {
                True: "Concise and punchy - perfect length",
                False: "Consider shortening for better impact"
            },
            'hashtag_usage': {
                True: "Excellent hashtag strategy",
                False: "Add 2-3 relevant hashtags to increase discoverability"
            },
            'emoji_presence': {
                True: "Great use of emojis for visual appeal",
                False: "Add 1-2 emojis to make it more eye-catching"
            },
            'call_to_action': {
                True: "Strong call-to-action drives engagement",
                False: "Add a clear CTA (e.g., 'Comment below', 'Share if you agree')"
            },
            'engagement_hooks': {
                True: "Compelling hooks that grab attention",
                False: "Start with a question or bold statement to hook readers"
            },
            'storytelling': {
                True: "Engaging narrative structure",
                False: "Incorporate storytelling elements to connect emotionally"
            },
            'professionalism': {
                True: "Professional tone resonates with business audience",
                False: "Use more industry-specific terminology"
            },
            'value_proposition': {
                True: "Clear value and actionable insights",
                False: "Highlight specific benefits or takeaways"
            },
            'viral_potential': {
                True: "High viral potential with trending elements",
                False: "Add trending topics or controversial angles"
            },
            'hook_strength': {
                True: "Powerful opening grabs immediate attention",
                False: "Strengthen the first sentence with urgency or curiosity"
            },
            'visual_appeal': {
                True: "Well-formatted and visually scannable",
                False: "Use line breaks and formatting for better readability"
            },
            'seo_optimization': {
                True: "SEO-friendly with good keyword usage",
                False: "Include more relevant keywords and hashtags"
            }
        }
        
        return messages.get(factor, {True: f"Strong {factor}", False: f"Improve {factor}"})[is_strength]
    
    def _suggest_posting_time(self, platform: str) -> str:
        """Suggest optimal posting time based on platform."""
        timing_suggestions = {
            'Instagram': "Best: 6-9 PM weekdays, 11 AM-1 PM weekends",
            'Twitter/X': "Best: 8-10 AM, 6-9 PM weekdays",
            'LinkedIn': "Best: 7-9 AM, 12 PM, 5-6 PM Tuesday-Thursday",
            'TikTok': "Best: 6-10 PM, especially Tuesday-Thursday",
            'YouTube': "Best: 2-4 PM weekdays, 9-11 AM weekends"
        }
        return timing_suggestions.get(platform, "Post during peak engagement hours")
    
    def generate_scoring_prompt(self, content: str, platform: str) -> str:
        """
        Generate an AI prompt for more sophisticated scoring.
        This can be passed to the LLM for enhanced analysis.
        """
        return f"""Analyze this {platform} post for engagement potential:

Content: "{content}"

Evaluate on these criteria:
1. Hook strength and attention-grabbing opening
2. Emotional resonance and relatability
3. Value proposition and actionable insights
4. Call-to-action effectiveness
5. Platform-specific optimization ({platform} best practices)
6. Viral potential and shareability
7. Visual formatting and readability
8. Hashtag and keyword strategy
9. Authenticity and brand voice
10. Timing and trend relevance

Provide:
- Overall engagement score (0-100)
- Top 3 strengths
- Top 3 improvements
- Predicted engagement level (Low/Medium/High/Viral)
- Recommended posting time

Format as structured JSON."""


def get_engagement_scorer() -> EngagementScorer:
    """Get or create engagement scorer singleton instance."""
    return EngagementScorer()