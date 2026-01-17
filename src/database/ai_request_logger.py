"""
AI Request Logger for Creator Catalyst
Tracks all AI API calls with timestamps, costs, and usage metrics.
Provides analytics and prevents cost overruns.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AIRequest:
    """Represents a single AI API request."""
    id: Optional[int] = None
    user_id: str = "default_user"
    endpoint: str = ""
    provider: str = ""  # gemini, openai, ollama, huggingface
    operation_type: str = ""  # video_analysis, text_generation, image_generation
    tokens_used: int = 0
    cost_credits: float = 0.0
    cost_usd: float = 0.0
    response_time_ms: int = 0
    success: bool = True
    error_message: Optional[str] = None
    request_metadata: str = "{}"  # JSON string
    created_at: str = ""


class AIRequestLogger:
    """
    Centralized logging system for all AI API requests.
    Tracks usage, costs, and provides analytics.
    """
    
    # Token cost mappings (approximate USD per 1M tokens)
    TOKEN_COSTS = {
        'gemini-2.0-flash-exp': {
            'input': 0.0,  # Free tier
            'output': 0.0
        },
        'gpt-4o': {
            'input': 5.00,
            'output': 15.00
        },
        'gpt-4-turbo': {
            'input': 10.00,
            'output': 30.00
        },
        'ollama': {
            'input': 0.0,  # Local, free
            'output': 0.0
        },
        'stable-diffusion-xl': {
            'per_image': 0.02  # Approximate HF cost
        }
    }
    
    def __init__(self, db_path: str = "data/creator_catalyst.db"):
        """Initialize AI request logger with database connection."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_logger_tables()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_logger_tables(self):
        """Create logging tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # AI requests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT 'default_user',
                    endpoint TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    tokens_used INTEGER DEFAULT 0,
                    cost_credits REAL DEFAULT 0.0,
                    cost_usd REAL DEFAULT 0.0,
                    response_time_ms INTEGER DEFAULT 0,
                    success INTEGER DEFAULT 1,
                    error_message TEXT,
                    request_metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Rate limit tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rate_limits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    window_start TIMESTAMP NOT NULL,
                    window_end TIMESTAMP NOT NULL,
                    request_count INTEGER DEFAULT 0,
                    tokens_used INTEGER DEFAULT 0,
                    credits_spent REAL DEFAULT 0.0,
                    UNIQUE(user_id, window_start)
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ai_requests_user_time 
                ON ai_requests(user_id, created_at DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ai_requests_provider 
                ON ai_requests(provider, created_at DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rate_limits_user 
                ON rate_limits(user_id, window_start)
            """)
            
            print(f"✅ AI Request Logger initialized")
    
    def log_request(
        self,
        endpoint: str,
        provider: str,
        operation_type: str,
        tokens_used: int = 0,
        cost_credits: float = 0.0,
        response_time_ms: int = 0,
        success: bool = True,
        error_message: Optional[str] = None,
        user_id: str = "default_user",
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Log an AI API request.
        
        Args:
            endpoint: API endpoint called (e.g., /analyze_video, /generate_text)
            provider: AI provider used (gemini, openai, ollama, huggingface)
            operation_type: Type of operation (video_analysis, text_generation, etc.)
            tokens_used: Number of tokens consumed
            cost_credits: Cost in application credits
            response_time_ms: Response time in milliseconds
            success: Whether request succeeded
            error_message: Error message if failed
            user_id: User identifier
            metadata: Additional request metadata
            
        Returns:
            request_id: ID of logged request
        """
        # Calculate USD cost
        cost_usd = self._calculate_usd_cost(provider, tokens_used, operation_type)
        
        # Serialize metadata
        metadata_str = json.dumps(metadata or {})
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ai_requests (
                    user_id, endpoint, provider, operation_type,
                    tokens_used, cost_credits, cost_usd, response_time_ms,
                    success, error_message, request_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, endpoint, provider, operation_type,
                tokens_used, cost_credits, cost_usd, response_time_ms,
                1 if success else 0, error_message, metadata_str
            ))
            
            request_id = cursor.lastrowid
            
            # Update rate limit tracking
            self._update_rate_limit_window(user_id, tokens_used, cost_credits)
            
            return request_id
    
    def _calculate_usd_cost(self, provider: str, tokens: int, operation: str) -> float:
        """Calculate approximate USD cost for a request."""
        model_key = provider.lower()
        
        if operation == "image_generation":
            return self.TOKEN_COSTS.get('stable-diffusion-xl', {}).get('per_image', 0.02)
        
        costs = self.TOKEN_COSTS.get(model_key, {'input': 0, 'output': 0})
        
        # Approximate: 70% input, 30% output tokens
        input_tokens = int(tokens * 0.7)
        output_tokens = int(tokens * 0.3)
        
        input_cost = (input_tokens / 1_000_000) * costs.get('input', 0)
        output_cost = (output_tokens / 1_000_000) * costs.get('output', 0)
        
        return input_cost + output_cost
    
    def _update_rate_limit_window(self, user_id: str, tokens: int, credits: float):
        """Update rate limit tracking for current time window."""
        now = datetime.now()
        window_start = now.replace(minute=0, second=0, microsecond=0)  # Hourly windows
        window_end = window_start + timedelta(hours=1)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Try to update existing window
            cursor.execute("""
                UPDATE rate_limits 
                SET request_count = request_count + 1,
                    tokens_used = tokens_used + ?,
                    credits_spent = credits_spent + ?
                WHERE user_id = ? AND window_start = ?
            """, (tokens, credits, user_id, window_start.isoformat()))
            
            # If no rows updated, create new window
            if cursor.rowcount == 0:
                cursor.execute("""
                    INSERT INTO rate_limits (
                        user_id, window_start, window_end,
                        request_count, tokens_used, credits_spent
                    ) VALUES (?, ?, ?, 1, ?, ?)
                """, (user_id, window_start.isoformat(), window_end.isoformat(), tokens, credits))
    
    def check_rate_limit(
        self,
        user_id: str = "default_user",
        max_requests_per_hour: int = 100,
        max_tokens_per_hour: int = 1_000_000
    ) -> Tuple[bool, Dict]:
        """
        Check if user is within rate limits.
        
        Args:
            user_id: User identifier
            max_requests_per_hour: Maximum requests per hour
            max_tokens_per_hour: Maximum tokens per hour
            
        Returns:
            Tuple of (is_allowed, usage_stats)
        """
        now = datetime.now()
        window_start = now.replace(minute=0, second=0, microsecond=0)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT request_count, tokens_used, credits_spent
                FROM rate_limits
                WHERE user_id = ? AND window_start = ?
            """, (user_id, window_start.isoformat()))
            
            row = cursor.fetchone()
            
            if not row:
                return True, {
                    'requests_used': 0,
                    'tokens_used': 0,
                    'credits_spent': 0.0,
                    'requests_remaining': max_requests_per_hour,
                    'tokens_remaining': max_tokens_per_hour
                }
            
            requests_used = row['request_count']
            tokens_used = row['tokens_used']
            credits_spent = row['credits_spent']
            
            is_allowed = (
                requests_used < max_requests_per_hour and
                tokens_used < max_tokens_per_hour
            )
            
            return is_allowed, {
                'requests_used': requests_used,
                'tokens_used': tokens_used,
                'credits_spent': credits_spent,
                'requests_remaining': max_requests_per_hour - requests_used,
                'tokens_remaining': max_tokens_per_hour - tokens_used,
                'window_start': window_start.isoformat(),
                'window_end': (window_start + timedelta(hours=1)).isoformat()
            }
    
    def get_request_history(
        self,
        user_id: str = "default_user",
        limit: int = 100,
        provider: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """Get AI request history with optional filters."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM ai_requests WHERE user_id = ?"
            params = [user_id]
            
            if provider:
                query += " AND provider = ?"
                params.append(provider)
            
            if start_date:
                query += " AND created_at >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND created_at <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            requests = []
            for row in cursor.fetchall():
                requests.append({
                    'id': row['id'],
                    'endpoint': row['endpoint'],
                    'provider': row['provider'],
                    'operation_type': row['operation_type'],
                    'tokens_used': row['tokens_used'],
                    'cost_credits': row['cost_credits'],
                    'cost_usd': row['cost_usd'],
                    'response_time_ms': row['response_time_ms'],
                    'success': bool(row['success']),
                    'error_message': row['error_message'],
                    'metadata': json.loads(row['request_metadata']),
                    'created_at': row['created_at']
                })
            
            return requests
    
    def get_usage_analytics(
        self,
        user_id: str = "default_user",
        days: int = 30
    ) -> Dict:
        """Get comprehensive usage analytics."""
        start_date = datetime.now() - timedelta(days=days)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(tokens_used) as total_tokens,
                    SUM(cost_credits) as total_credits,
                    SUM(cost_usd) as total_usd,
                    AVG(response_time_ms) as avg_response_time,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_requests
                FROM ai_requests
                WHERE user_id = ? AND created_at >= ?
            """, (user_id, start_date.isoformat()))
            
            totals = cursor.fetchone()
            
            # By provider
            cursor.execute("""
                SELECT 
                    provider,
                    COUNT(*) as request_count,
                    SUM(tokens_used) as tokens,
                    SUM(cost_credits) as credits,
                    SUM(cost_usd) as usd
                FROM ai_requests
                WHERE user_id = ? AND created_at >= ?
                GROUP BY provider
            """, (user_id, start_date.isoformat()))
            
            by_provider = {row['provider']: dict(row) for row in cursor.fetchall()}
            
            # By operation type
            cursor.execute("""
                SELECT 
                    operation_type,
                    COUNT(*) as request_count,
                    SUM(tokens_used) as tokens,
                    AVG(response_time_ms) as avg_response_time
                FROM ai_requests
                WHERE user_id = ? AND created_at >= ?
                GROUP BY operation_type
            """, (user_id, start_date.isoformat()))
            
            by_operation = {row['operation_type']: dict(row) for row in cursor.fetchall()}
            
            # Daily breakdown
            cursor.execute("""
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as requests,
                    SUM(tokens_used) as tokens,
                    SUM(cost_credits) as credits
                FROM ai_requests
                WHERE user_id = ? AND created_at >= ?
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            """, (user_id, start_date.isoformat()))
            
            daily = [dict(row) for row in cursor.fetchall()]
            
            return {
                'period_days': days,
                'total_requests': totals['total_requests'] or 0,
                'total_tokens': totals['total_tokens'] or 0,
                'total_credits': totals['total_credits'] or 0.0,
                'total_usd': totals['total_usd'] or 0.0,
                'avg_response_time_ms': totals['avg_response_time'] or 0,
                'failed_requests': totals['failed_requests'] or 0,
                'success_rate': (
                    ((totals['total_requests'] - totals['failed_requests']) / totals['total_requests'] * 100)
                    if totals['total_requests'] else 100
                ),
                'by_provider': by_provider,
                'by_operation': by_operation,
                'daily_breakdown': daily
            }
    
    def cleanup_old_logs(self, days_to_keep: int = 90):
        """Delete logs older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM ai_requests 
                WHERE created_at < ?
            """, (cutoff_date.isoformat(),))
            
            deleted = cursor.rowcount
            
            cursor.execute("""
                DELETE FROM rate_limits 
                WHERE window_end < ?
            """, (cutoff_date.isoformat(),))
            
            print(f"✅ Cleaned up {deleted} old log entries")
            return deleted


# Singleton instance
_logger_instance = None

def get_ai_logger(db_path: str = "creator_catalyst.db") -> AIRequestLogger:
    """Get or create AI logger singleton instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AIRequestLogger(db_path)
    return _logger_instance