"""
Credits management system for Creator Catalyst.
Handles credit tracking, deduction, and validation for monetization.
"""

import sqlite3
from datetime import datetime
from typing import Optional, Dict, Tuple
from contextlib import contextmanager
from pathlib import Path


class CreditsManager:
    """
    Manages user credits for Creator Catalyst operations.
    Tracks usage and enforces limits for monetization.
    """
    
    # Credit costs for different operations
    COSTS = {
        'video_upload': 5,
        'blog_generation': 2,
        'social_post': 1,
        'shorts_clip': 1,
        'thumbnail_generation': 1,
        'tweet_enhancement': 1
    }
    
    # Default starting credits for new users
    DEFAULT_CREDITS = 50
    
    def __init__(self, db_path: str = "data/creator_catalyst.db"):
        """Initialize credits manager with database connection."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_credits_table()
    
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
    
    def _init_credits_table(self):
        """Create credits-related tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # User credits table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_credits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL UNIQUE DEFAULT 'default_user',
                    current_balance INTEGER NOT NULL DEFAULT 0,
                    total_earned INTEGER DEFAULT 0,
                    total_spent INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Credit transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS credit_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT 'default_user',
                    transaction_type TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    balance_after INTEGER NOT NULL,
                    operation_type TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_transactions_user 
                ON credit_transactions(user_id, created_at DESC)
            """)
            
            # Initialize default user if not exists
            cursor.execute("""
                INSERT OR IGNORE INTO user_credits (user_id, current_balance, total_earned)
                VALUES ('default_user', ?, ?)
            """, (self.DEFAULT_CREDITS, self.DEFAULT_CREDITS))
            
            print(f"✅ Credits system initialized")
    
    def get_balance(self, user_id: str = 'default_user') -> int:
        """Get current credit balance for user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT current_balance FROM user_credits WHERE user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            return row['current_balance'] if row else 0
    
    def has_sufficient_credits(self, operation: str, user_id: str = 'default_user') -> Tuple[bool, int, int]:
        """
        Check if user has enough credits for operation.
        
        Args:
            operation: Operation type (key from COSTS dict)
            user_id: User identifier
            
        Returns:
            Tuple of (has_credits, current_balance, required_amount)
        """
        cost = self.COSTS.get(operation, 0)
        balance = self.get_balance(user_id)
        
        return (balance >= cost, balance, cost)
    
    def deduct_credits(
        self, 
        operation: str, 
        user_id: str = 'default_user',
        description: Optional[str] = None
    ) -> Tuple[bool, int]:
        """
        Deduct credits for an operation.
        
        Args:
            operation: Operation type
            user_id: User identifier
            description: Optional description of the transaction
            
        Returns:
            Tuple of (success, new_balance)
        """
        cost = self.COSTS.get(operation, 0)
        
        if cost == 0:
            return True, self.get_balance(user_id)
        
        # Check if user has enough credits
        has_credits, current_balance, _ = self.has_sufficient_credits(operation, user_id)
        
        if not has_credits:
            print(f"❌ Insufficient credits: need {cost}, have {current_balance}")
            return False, current_balance
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Deduct from balance
            cursor.execute("""
                UPDATE user_credits 
                SET current_balance = current_balance - ?,
                    total_spent = total_spent + ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (cost, cost, user_id))
            
            # Get new balance
            cursor.execute("""
                SELECT current_balance FROM user_credits WHERE user_id = ?
            """, (user_id,))
            new_balance = cursor.fetchone()['current_balance']
            
            # Record transaction
            cursor.execute("""
                INSERT INTO credit_transactions (
                    user_id, transaction_type, amount, balance_after,
                    operation_type, description
                ) VALUES (?, 'debit', ?, ?, ?, ?)
            """, (user_id, cost, new_balance, operation, description or f"{operation} operation"))
            
            print(f"✅ Deducted {cost} credits for {operation}. New balance: {new_balance}")
            return True, new_balance
    
    def add_credits(
        self, 
        amount: int, 
        user_id: str = 'default_user',
        description: str = "Credits added"
    ) -> int:
        """
        Add credits to user account.
        
        Args:
            amount: Number of credits to add
            user_id: User identifier
            description: Transaction description
            
        Returns:
            New balance
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Add to balance
            cursor.execute("""
                UPDATE user_credits 
                SET current_balance = current_balance + ?,
                    total_earned = total_earned + ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (amount, amount, user_id))
            
            # Get new balance
            cursor.execute("""
                SELECT current_balance FROM user_credits WHERE user_id = ?
            """, (user_id,))
            new_balance = cursor.fetchone()['current_balance']
            
            # Record transaction
            cursor.execute("""
                INSERT INTO credit_transactions (
                    user_id, transaction_type, amount, balance_after, description
                ) VALUES (?, 'credit', ?, ?, ?)
            """, (user_id, amount, new_balance, description))
            
            print(f"✅ Added {amount} credits. New balance: {new_balance}")
            return new_balance
    
    def get_transaction_history(
        self, 
        user_id: str = 'default_user',
        limit: int = 50
    ) -> list:
        """Get transaction history for user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM credit_transactions 
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit))
            
            transactions = []
            for row in cursor.fetchall():
                transactions.append({
                    'id': row['id'],
                    'type': row['transaction_type'],
                    'amount': row['amount'],
                    'balance_after': row['balance_after'],
                    'operation': row['operation_type'],
                    'description': row['description'],
                    'created_at': row['created_at']
                })
            
            return transactions
    
    def get_user_stats(self, user_id: str = 'default_user') -> Dict:
        """Get comprehensive stats for user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get user credits info
            cursor.execute("""
                SELECT * FROM user_credits WHERE user_id = ?
            """, (user_id,))
            credits_row = cursor.fetchone()
            
            if not credits_row:
                return {
                    'current_balance': 0,
                    'total_earned': 0,
                    'total_spent': 0,
                    'operation_counts': {}
                }
            
            # Get operation counts
            cursor.execute("""
                SELECT operation_type, COUNT(*) as count, SUM(amount) as total_cost
                FROM credit_transactions
                WHERE user_id = ? AND transaction_type = 'debit'
                GROUP BY operation_type
            """, (user_id,))
            
            operation_counts = {}
            for row in cursor.fetchall():
                if row['operation_type']:
                    operation_counts[row['operation_type']] = {
                        'count': row['count'],
                        'total_cost': row['total_cost']
                    }
            
            return {
                'current_balance': credits_row['current_balance'],
                'total_earned': credits_row['total_earned'],
                'total_spent': credits_row['total_spent'],
                'last_updated': credits_row['last_updated'],
                'operation_counts': operation_counts
            }
    
    def reset_credits(self, user_id: str = 'default_user', new_balance: int = None):
        """Reset user credits (admin function)."""
        if new_balance is None:
            new_balance = self.DEFAULT_CREDITS
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE user_credits 
                SET current_balance = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (new_balance, user_id))
            
            # Record transaction
            cursor.execute("""
                INSERT INTO credit_transactions (
                    user_id, transaction_type, amount, balance_after, description
                ) VALUES (?, 'credit', ?, ?, 'Credits reset')
            """, (user_id, new_balance, new_balance))
            
            print(f"✅ Credits reset to {new_balance}")


# Singleton instance
_credits_instance = None

def get_credits_manager(db_path: str = "creator_catalyst.db") -> CreditsManager:
    """Get or create credits manager singleton instance."""
    global _credits_instance
    if _credits_instance is None:
        _credits_instance = CreditsManager(db_path)
    return _credits_instance