"""
AI Strategy Advisor for Creator Catalyst
Analyzes historical content performance and generates actionable "Next Steps".
"""

import json
from typing import List, Dict, Optional
from src.core.llm_wrapper import LLMWrapper
from src.core.engagement_scorer import EngagementScorer, get_engagement_scorer
from src.database.database import Database, Video, ContentOutput

class StrategyAdvisor:
    """Provides AI-generated strategy tips based on content performance."""

    def __init__(self, db: Database, llm: LLMWrapper):
        self.db = db
        self.llm = llm
        self.scorer = get_engagement_scorer()

    def generate_next_steps(self, video_id: int) -> str:
        """
        Generate actionable next steps for a specific video campaign.
        """
        # 1. Fetch data
        video = self.db.get_video(video_id)
        if not video:
            return "Video not found."
            
        outputs = self.db.get_content_by_video(video_id)
        if not outputs:
            return "No content generated for this video yet."

        # 2. Analyze performance of each content piece
        performance_data = []
        for output in outputs:
            if output.content_type in ["blog_post", "social_post", "shorts_idea"]:
                # Check if score is already in metadata
                metadata = output.to_dict().get('metadata', {})
                
                if 'engagement_score' in metadata:
                    performance_data.append({
                        "type": output.content_type,
                        "score": metadata.get('engagement_score'),
                        "sentiment": metadata.get('sentiment'),
                        "readability": metadata.get('readability'),
                        "virality": metadata.get('virality')
                    })
                else:
                    # Calculate if missing (for legacy data)
                    content_text = output.content
                    if output.content_type == "shorts_idea":
                        try:
                            idea = json.loads(output.content)
                            content_text = f"Topic: {idea.get('topic')}. Hook: {idea.get('hook')}"
                        except:
                            pass
                    
                    score = self.scorer.score_content(content_text, output.content_type)
                    performance_data.append({
                        "type": output.content_type,
                        "score": score.overall_score,
                        "sentiment": score.sentiment,
                        "readability": score.readability_score,
                        "virality": score.virality_score
                    })

        # 3. Create prompt for LLM
        prompt = self._create_strategy_prompt(video, performance_data)
        
        # 4. Generate tips
        tips = self.llm.generate_text(prompt)
        return tips

    def _create_strategy_prompt(self, video: Video, performance: List[Dict]) -> str:
        perf_summary = json.dumps(performance, indent=2)
        return f"""You are a senior Content Strategy Advisor for a digital creator platform.
Analyze the performance metrics of the content generated for the video campaign: "{video.filename}".

Performance Data:
{perf_summary}

Based on this data, provide 3-5 highly actionable "Next Steps" to improve future campaigns or optimize the current results.
Focus on:
1. Content gaps identified in improvements (low scores).
2. Leveraging existing strengths (high scores).
3. Sentiment and readability adjustments.
4. Specific platform optimizations based on the content types.

Format your response as a clear, bulleted list of "Next Steps" with a brief explanation for each. Use emojis for readability. 
Keep the tone professional yet encouraging."""

def get_strategy_advisor():
    """Helper to get advisor instance."""
    from src.core.llm_wrapper import LLMWrapper
    from src.database.database import Database
    db = Database()
    llm = LLMWrapper()
    return StrategyAdvisor(db, llm)
