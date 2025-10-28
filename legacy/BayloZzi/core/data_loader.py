"""
Data Loader Module - Handles data fetching
"""
import pandas as pd
import numpy as np
from datetime import datetime

def download_alpha_fx_daily(symbol='EURUSD', api_key=None):
    """
    Download forex daily data (stub function)
    Returns sample data for testing
    """
    # Return sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 1.1000,
        'high': np.random.randn(100).cumsum() + 1.1100,
        'low': np.random.randn(100).cumsum() + 1.0900,
        'close': np.random.randn(100).cumsum() + 1.1000,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    return df

def get_live_data(pair='EURUSD'):
    """Get live market data"""
    return download_alpha_fx_daily(pair)