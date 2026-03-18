"""
Services module for forex trading system.

Provides data acquisition, feature engineering, and other business logic services.
"""

from .feature_engineering import FeatureEngineer

try:
    from .data_service import DataService
    __all__ = ["DataService", "FeatureEngineer"]
except ImportError:
    # SQLAlchemy not installed — DataService unavailable, FeatureEngineer still works
    __all__ = ["FeatureEngineer"]
