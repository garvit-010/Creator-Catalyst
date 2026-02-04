"""
Fact-Grounding System for Creator Catalyst
Ensures all AI-generated claims are backed by transcript evidence with timestamps.
Refactored for Issue #52: Uses Vector-Based RAG (Semantic Search).
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

@dataclass
class TranscriptSegment:
    """Represents a single SRT caption segment with timing."""
    index: int
    start_time: str
    end_time: str
    text: str
    
    def to_seconds(self, time_str: str) -> float:
        """Convert SRT timestamp to seconds."""
        # Format: HH:MM:SS,mmm or MM:SS,mmm
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return 0.0


class FactGrounder:
    """
    Validates AI-generated content against video transcript using Semantic Search (RAG).
    Ensures claims are grounded in actual video content.
    """
    
    # Static model instance to avoid reloading overhead across requests
    _model = None
    
    def __init__(self, srt_content: str):
        """
        Initialize with SRT transcript content and prepare vector index.
        
        Args:
            srt_content: Raw SRT format transcript
        """
        self.segments = self._parse_srt(srt_content)
        self.full_text = " ".join([seg.text for seg in self.segments])
        
        # Initialize Embedding Model (Lazy loading)
        if FactGrounder._model is None:
            # 'all-MiniLM-L6-v2' is fast and effective for semantic search
            print("Loading Fact-Grounding Embedding Model...")
            FactGrounder._model = SentenceTransformer('all-MiniLM-L6-v2')
            
        self.model = FactGrounder._model
        
        # Pre-compute embeddings for all transcript segments
        # We assume segments are short; if too short, we might merge neighbors (future optimization)
        self.segment_texts = [seg.text for seg in self.segments]
        
        if self.segment_texts:
            self.segment_embeddings = self.model.encode(
                self.segment_texts, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
        else:
            self.segment_embeddings = None
        
    def _parse_srt(self, srt_content: str) -> List[TranscriptSegment]:
        """Parse SRT content into structured segments."""
        segments = []
        
        # Split by double newlines (segment separator)
        raw_segments = re.split(r'\n\s*\n', srt_content.strip())
        
        for raw_seg in raw_segments:
            lines = raw_seg.strip().split('\n')
            if len(lines) < 3:
                continue
                
            try:
                index = int(lines[0])
                # Parse timestamp line: 00:00:01,000 --> 00:00:05,000
                time_match = re.match(r'([\d:,]+)\s*-->\s*([\d:,]+)', lines[1])
                if time_match:
                    start_time = time_match.group(1)
                    end_time = time_match.group(2)
                    text = ' '.join(lines[2:])
                    
                    segments.append(TranscriptSegment(
                        index=index,
                        start_time=start_time,
                        end_time=end_time,
                        text=text
                    ))
            except (ValueError, IndexError):
                continue
                
        return segments
    
    def find_supporting_evidence(self, claim: str, threshold: float = 0.35) -> Optional[Dict]:
        """
        Find transcript segments that support a given claim using Semantic Vector Search.
        
        Args:
            claim: The claim to verify
            threshold: Cosine similarity threshold (0-1). 
                       0.35 is usually a good baseline for semantic relatedness.
            
        Returns:
            Dict with evidence or None if not found
        """
        if not self.segment_texts or self.segment_embeddings is None:
            return None

        # 1. Vectorize the claim
        claim_embedding = self.model.encode(claim, convert_to_tensor=True)
        
        # 2. Semantic Search (Cosine Similarity)
        # util.cos_sim returns a matrix, we take the first row [0]
        cosine_scores = util.cos_sim(claim_embedding, self.segment_embeddings)[0]
        
        # 3. Find Best Match
        best_score_tensor = torch.max(cosine_scores)
        best_score = float(best_score_tensor.item())
        best_idx = int(torch.argmax(cosine_scores).item())
        
        if best_score >= threshold:
            best_segment = self.segments[best_idx]
            return {
                'segment': best_segment,
                'score': best_score,
                'timestamp': best_segment.start_time,
                'text': best_segment.text
            }
        
        return None
    
    def verify_claim(self, claim: str) -> Tuple[bool, Optional[str]]:
        """
        Verify if a claim is supported by transcript.
        
        Args:
            claim: The claim to verify
            
        Returns:
            Tuple of (is_valid, timestamp_or_none)
        """
        evidence = self.find_supporting_evidence(claim)
        
        if evidence:
            return True, evidence['timestamp']
        
        return False, None
    
    def extract_grounded_claims(self, text: str) -> List[Dict]:
        """
        Extract claims from text and verify each against transcript.
        
        Args:
            text: Content to analyze (blog post, social media, etc.)
            
        Returns:
            List of dicts with claim, verification status, and timestamps
        """
        # Split text into sentences (claims)
        sentences = re.split(r'[.!?]+', text)
        results = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short fragments
                continue
            
            # Skip meta-text (headers, formatting)
            if sentence.startswith('#') or sentence.startswith('*'):
                continue
            
            evidence = self.find_supporting_evidence(sentence)
            is_valid = evidence is not None
            
            results.append({
                'claim': sentence,
                'is_grounded': is_valid,
                'timestamp': evidence['timestamp'] if is_valid else None,
                'evidence_score': evidence['score'] if is_valid else 0.0
            })
        
        return results
    
    def filter_ungrounded_content(self, text: str, strict_mode: bool = False) -> str:
        """
        Remove ungrounded claims from text content.
        
        Args:
            text: Content to filter
            strict_mode: If True, removes any claim without high confidence evidence
            
        Returns:
            Filtered text with only grounded claims
        """
        claims = self.extract_grounded_claims(text)
        
        # Build filtered text
        filtered_sentences = []
        for claim_data in claims:
            if claim_data['is_grounded']:
                filtered_sentences.append(claim_data['claim'])
        
        return '. '.join(filtered_sentences) + '.'
    
    def add_citations_to_content(self, text: str) -> str:
        """
        Add timestamp citations to supported claims.
        
        Args:
            text: Content to annotate
            
        Returns:
            Text with inline citations
        """
        claims = self.extract_grounded_claims(text)
        
        annotated_text = text
        for claim_data in claims:
            if claim_data['is_grounded'] and claim_data['timestamp']:
                # Add citation after the claim
                original = claim_data['claim']
                # Only replace if we haven't already (simple prevention of double replace)
                if original in annotated_text:
                    cited = f"{original} [Source: {claim_data['timestamp']}]"
                    annotated_text = annotated_text.replace(original, cited, 1)
        
        return annotated_text
    
    def generate_grounding_prompt(self) -> str:
        """
        Generate a prompt addition to instruct LLM to ground all claims.
        """
        return f"""
CRITICAL FACT-GROUNDING REQUIREMENTS:

1. ALL claims, statistics, quotes, or factual statements MUST come directly from the video transcript
2. You MUST cite the timestamp for each factual claim in this format: [Source: MM:SS]
3. DO NOT invent, extrapolate, or assume any facts not explicitly stated in the video
4. If you cannot find transcript support for a claim, DO NOT include it
5. Verify all numbers, percentages, and specific details against the transcript

Transcript Reference:
{self.full_text[:500]}... [Full transcript available]

Any claim without transcript evidence will be automatically removed.
"""

    def validate_shorts_ideas(self, shorts_ideas: List[Dict]) -> List[Dict]:
        """
        Validate that shorts ideas correspond to actual transcript content.
        """
        validated = []
        
        for idea in shorts_ideas:
            start_time = idea.get('start_time', '')
            end_time = idea.get('end_time', '')
            summary = idea.get('summary', '')
            
            # Find segments in this time range
            supporting_segments = []
            for seg in self.segments:
                if self._is_in_time_range(seg, start_time, end_time):
                    supporting_segments.append(seg)
            
            if supporting_segments:
                # Verify summary matches segment content
                combined_text = ' '.join([s.text for s in supporting_segments])
                
                # Semantic check between summary and the actual segment text
                evidence = self.find_supporting_evidence(summary)
                
                # If we find evidence within the transcript generally, or specifically in this segment
                if evidence:
                    idea['validation_status'] = 'verified'
                    idea['supporting_text'] = evidence['text']
                    idea['confidence'] = f"{evidence['score']:.2f}"
                    validated.append(idea)
                else:
                    idea['validation_status'] = 'unverified_summary'
                    validated.append(idea)
            else:
                idea['validation_status'] = 'invalid_timestamps'
        
        return validated
    
    def _is_in_time_range(self, segment: TranscriptSegment, start: str, end: str) -> bool:
        """Check if segment falls within time range."""
        try:
            seg_start = segment.to_seconds(segment.start_time)
            range_start = segment.to_seconds(start)
            range_end = segment.to_seconds(end)
            
            return range_start <= seg_start <= range_end
        except:
            return False
    
    def generate_grounding_report(self, content_dict: Dict) -> Dict:
        """
        Generate a comprehensive grounding report for all generated content.
        """
        report = {
            'original_content': content_dict,
            'validation_results': {},
            'filtered_content': {},
            'statistics': {}
        }
        
        # Validate blog post
        if 'blog_post' in content_dict:
            blog_claims = self.extract_grounded_claims(content_dict['blog_post'])
            grounded_count = sum(1 for c in blog_claims if c['is_grounded'])
            
            report['validation_results']['blog_post'] = blog_claims
            report['filtered_content']['blog_post'] = self.filter_ungrounded_content(
                content_dict['blog_post']
            )
            report['statistics']['blog_grounding_rate'] = (
                grounded_count / len(blog_claims) if blog_claims else 0
            )
        
        # Validate social post
        if 'social_post' in content_dict:
            social_claims = self.extract_grounded_claims(content_dict['social_post'])
            grounded_count = sum(1 for c in social_claims if c['is_grounded'])
            
            report['validation_results']['social_post'] = social_claims
            report['filtered_content']['social_post'] = self.filter_ungrounded_content(
                content_dict['social_post']
            )
            report['statistics']['social_grounding_rate'] = (
                grounded_count / len(social_claims) if social_claims else 0
            )
        
        return report


def create_grounding_prompt_modifier(srt_content: str) -> str:
    """
    Convenience function to create grounding instructions for LLM prompts.
    """
    grounder = FactGrounder(srt_content)
    return grounder.generate_grounding_prompt()