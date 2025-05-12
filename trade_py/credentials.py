"""Manages credentials (e.g. API keys)."""

import os


def get_alpaca_keys():
    """Get Alpaca paper trading API credentials from environment variables."""
    assert 'ALPACA_API_KEY_PAPER' in os.environ
    assert 'ALPACA_API_SECRET_PAPER' in os.environ
    return {
        'api_key': os.environ['ALPACA_API_KEY_PAPER'],
        'api_secret': os.environ['ALPACA_API_SECRET_PAPER']
    }