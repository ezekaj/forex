"""
Broker Integration Module
=========================
Connects the trading system to real brokers.
"""

from .oanda_broker import OandaBroker, get_broker

__all__ = ["OandaBroker", "get_broker"]
