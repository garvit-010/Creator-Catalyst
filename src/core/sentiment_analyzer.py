import re
from textblob import TextBlob
import pandas as pd

class SentimentAnalyzer:
    def __init__(self):
        pass

    def parse_srt(self, srt_text):
        """
        Parses SRT text into a list of {'start': seconds, 'text': string}.
        """
        if not srt_text:
            return []
            
        entries = []
        # Regex to find blocks: Index -> Timestamp -> Text
        # timestamps look like 00:00:01,000 --> 00:00:05,000
        pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)', re.DOTALL)
        
        matches = pattern.findall(srt_text + "\n\n") # Append newlines to catch last block
        
        for match in matches:
            start_str = match[1]
            text_content = match[3].replace('\n', ' ').strip()
            
            # Convert start timestamp to seconds
            h, m, s_ms = start_str.split(':')
            s, ms = s_ms.split(',')
            seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms)/1000.0
            
            entries.append({
                'start': seconds,
                'text': text_content
            })
            
        return entries

    def analyze_emotional_arc(self, srt_text, chunk_duration=30):
        """
        Splits transcript into time chunks and calculates sentiment.
        Returns a DataFrame suitable for Streamlit charting.
        """
        entries = self.parse_srt(srt_text)
        if not entries:
            return None

        # Group by chunks (e.g., every 30 seconds)
        max_time = entries[-1]['start']
        num_chunks = int(max_time // chunk_duration) + 1
        
        chunks = [{'time': i * chunk_duration, 'text': [], 'start_sec': i * chunk_duration} for i in range(num_chunks)]
        
        for entry in entries:
            chunk_idx = int(entry['start'] // chunk_duration)
            if chunk_idx < len(chunks):
                chunks[chunk_idx]['text'].append(entry['text'])
        
        # Calculate sentiment for each chunk
        results = []
        for chunk in chunks:
            full_text = " ".join(chunk['text'])
            if not full_text.strip():
                sentiment = 0.0 # Neutral if silence
            else:
                blob = TextBlob(full_text)
                sentiment = blob.sentiment.polarity # Range: -1.0 to 1.0
            
            # Create a label for the x-axis (e.g., "01:30")
            m = int(chunk['start_sec'] // 60)
            s = int(chunk['start_sec'] % 60)
            time_label = f"{m:02d}:{s:02d}"
            
            results.append({
                'Time': time_label,
                'Sentiment': sentiment,
                'Text': full_text[:100] + "..." if full_text else "(Silence)" # For hover tooltip
            })
            
        return pd.DataFrame(results)