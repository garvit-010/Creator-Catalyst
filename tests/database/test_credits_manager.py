import pytest
from src.database.credits_manager import CreditsManager

def test_initial_balance(credits_manager):
    """Test that the initial balance is set correctly."""
    balance = credits_manager.get_balance()
    assert balance == CreditsManager.DEFAULT_CREDITS

def test_add_credits(credits_manager):
    """Test adding credits to a user."""
    initial_balance = credits_manager.get_balance()
    amount_to_add = 100
    new_balance = credits_manager.add_credits(amount_to_add, description="Test add")
    
    assert new_balance == initial_balance + amount_to_add
    assert credits_manager.get_balance() == new_balance

def test_deduct_credits_success(credits_manager):
    """Test successful credit deduction."""
    initial_balance = credits_manager.get_balance()
    operation = 'video_upload'
    cost = CreditsManager.COSTS[operation]
    
    success, new_balance = credits_manager.deduct_credits(operation)
    
    assert success is True
    assert new_balance == initial_balance - cost
    assert credits_manager.get_balance() == new_balance

def test_deduct_credits_insufficient(credits_manager):
    """Test credit deduction with insufficient balance."""
    # Deduct more than available
    credits_manager.deduct_credits('video_upload') # -5
    credits_manager.deduct_credits('video_upload') # -5
    # ... keep deducting or just set balance to 0 if there was a method, 
    # but we can just use a lot of deductions or a custom add with negative if supported.
    # Let's just add a large amount and then try to deduct more than that.
    
    # Or better, just test with a high cost operation if any, 
    # or just deduct until 0.
    current_balance = credits_manager.get_balance()
    for _ in range(current_balance // 5 + 1):
        credits_manager.deduct_credits('video_upload')
        
    success, balance = credits_manager.deduct_credits('video_upload')
    assert success is False
    assert balance < 5

def test_has_sufficient_credits(credits_manager):
    """Test checking for sufficient credits."""
    operation = 'video_upload'
    cost = CreditsManager.COSTS[operation]
    
    has_credits, balance, required = credits_manager.has_sufficient_credits(operation)
    assert has_credits is True
    assert balance >= required
    assert required == cost

def test_get_user_stats(credits_manager):
    """Test retrieving user statistics."""
    credits_manager.add_credits(100, description="Earned")
    credits_manager.deduct_credits('video_upload')
    
    stats = credits_manager.get_user_stats()
    assert stats['current_balance'] == CreditsManager.DEFAULT_CREDITS + 100 - CreditsManager.COSTS['video_upload']
    assert stats['total_earned'] == CreditsManager.DEFAULT_CREDITS + 100
    assert stats['total_spent'] == CreditsManager.COSTS['video_upload']
