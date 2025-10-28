"""
Configuration management for the Forex Trading System.

This module loads and validates all configuration from environment variables,
ensuring secure credential management and proper validation on startup.

Never hardcode credentials - all sensitive values come from .env file.
"""
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when configuration is invalid or missing required values."""
    pass


@dataclass
class Config:
    """
    Centralized configuration management.
    Loads from .env file using python-dotenv.
    Validates all required settings on startup.
    """

    # System Settings
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Broker Configuration
    BROKER: str = "demo"

    # MT5 Credentials (NEVER hardcode in source)
    MT5_LOGIN: Optional[str] = None
    MT5_PASSWORD: Optional[str] = None
    MT5_SERVER: Optional[str] = None

    # OANDA Credentials
    OANDA_API_KEY: Optional[str] = None
    OANDA_ACCOUNT_ID: Optional[str] = None
    OANDA_ENVIRONMENT: str = "practice"

    # Demo Broker Settings
    DEMO_INITIAL_BALANCE: float = 10000.0
    DEMO_SLIPPAGE_PIPS: float = 0.5

    # Telegram Bot
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_USER_ID: Optional[int] = None

    # Database Configuration
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "forex_trading"
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    DATABASE_URL: Optional[str] = None

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_URL: Optional[str] = None

    # Risk Management
    RISK_PER_TRADE: float = 0.03  # 3% per trade
    HIGH_CONFIDENCE_RISK: float = 0.05  # 5% for auto-execute trades
    DAILY_LOSS_LIMIT: float = 0.10  # 10% daily loss limit
    WEEKLY_LOSS_LIMIT: float = 0.20  # 20% weekly loss limit
    MAX_DRAWDOWN: float = 0.30  # 30% max drawdown threshold
    MAX_POSITIONS: int = 5  # Maximum concurrent positions
    MAX_POSITIONS_PER_PAIR: int = 2  # Max positions per currency pair

    # Trading Strategy Configuration
    ENABLED_STRATEGIES: List[str] = field(default_factory=lambda: ["scalping", "day_trading", "swing_trading"])

    # Auto-Execute Thresholds
    AUTO_EXECUTE_CONFIDENCE: float = 0.80  # 80% minimum confidence
    AUTO_EXECUTE_RISK_REWARD: float = 3.0  # 1:3 minimum R/R
    AUTO_EXECUTE_MIN_INDICATORS: int = 4  # At least 4 indicators aligned

    # Currency Pairs
    TRADING_PAIRS: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURJPY", "GBPJPY"])

    # Leverage Settings
    LEVERAGE_SCALPING: int = 50
    LEVERAGE_DAY_TRADING: int = 30
    LEVERAGE_SWING_TRADING: int = 20

    # Market Data
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    TWELVE_DATA_API_KEY: Optional[str] = None

    # Data refresh intervals (seconds)
    DATA_REFRESH_SCALPING: int = 5  # 5 seconds for scalping
    DATA_REFRESH_DAY_TRADING: int = 60  # 1 minute for day trading
    DATA_REFRESH_SWING_TRADING: int = 300  # 5 minutes for swing trading

    # Monitoring & Alerting
    ENABLE_PERFORMANCE_MONITORING: bool = True
    ENABLE_ERROR_ALERTS: bool = True
    HEALTH_CHECK_INTERVAL: int = 60  # seconds

    # Advanced Settings
    ENABLE_NEWS_FILTER: bool = True
    NEWS_MINUTES_BEFORE: int = 30
    NEWS_MINUTES_AFTER: int = 30

    # Spread Filters (pips)
    MAX_SPREAD_SCALPING: float = 3.0
    MAX_SPREAD_DAY_TRADING: float = 5.0
    MAX_SPREAD_SWING_TRADING: float = 8.0

    # Correlation Filter
    ENABLE_CORRELATION_FILTER: bool = True
    MAX_CORRELATED_PAIRS: int = 2

    # Backtesting
    BACKTEST_START_DATE: str = "2023-01-01"
    BACKTEST_END_DATE: str = "2024-12-31"

    @classmethod
    def load(cls, env_file: Optional[str] = None) -> 'Config':
        """
        Load configuration from .env file with validation.

        Args:
            env_file: Optional path to .env file (defaults to .env in current directory)

        Returns:
            Validated Config instance

        Raises:
            ConfigError: If required configuration is missing or invalid
        """
        # Load .env file
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        config = cls()

        # Load and convert environment variables
        config._load_from_env()

        # Validate configuration
        config._validate()

        return config

    def _load_from_env(self) -> None:
        """Load configuration values from environment variables."""

        # System Settings
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
        self.DEBUG = self._get_bool("DEBUG", False)
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

        # Broker Configuration
        self.BROKER = os.getenv("BROKER", "demo")

        # MT5 Credentials
        self.MT5_LOGIN = os.getenv("MT5_LOGIN")
        self.MT5_PASSWORD = os.getenv("MT5_PASSWORD")
        self.MT5_SERVER = os.getenv("MT5_SERVER")

        # OANDA Credentials
        self.OANDA_API_KEY = os.getenv("OANDA_API_KEY")
        self.OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
        self.OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")

        # Demo Broker Settings
        self.DEMO_INITIAL_BALANCE = self._get_float("DEMO_INITIAL_BALANCE", 10000.0)
        self.DEMO_SLIPPAGE_PIPS = self._get_float("DEMO_SLIPPAGE_PIPS", 0.5)

        # Telegram Bot
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        self.TELEGRAM_USER_ID = self._get_int("TELEGRAM_USER_ID")

        # Database Configuration
        self.POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
        self.POSTGRES_PORT = self._get_int("POSTGRES_PORT", 5432)
        self.POSTGRES_DB = os.getenv("POSTGRES_DB", "forex_trading")
        self.POSTGRES_USER = os.getenv("POSTGRES_USER")
        self.POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
        self.DATABASE_URL = os.getenv("DATABASE_URL")

        # If DATABASE_URL not provided, construct it
        if not self.DATABASE_URL and self.POSTGRES_USER and self.POSTGRES_PASSWORD:
            self.DATABASE_URL = f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

        # Redis
        self.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        self.REDIS_PORT = self._get_int("REDIS_PORT", 6379)
        self.REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
        self.REDIS_DB = self._get_int("REDIS_DB", 0)
        self.REDIS_URL = os.getenv("REDIS_URL")

        # If REDIS_URL not provided, construct it
        if not self.REDIS_URL:
            if self.REDIS_PASSWORD:
                self.REDIS_URL = f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
            else:
                self.REDIS_URL = f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

        # Risk Management
        self.RISK_PER_TRADE = self._get_float("RISK_PER_TRADE", 0.03)
        self.HIGH_CONFIDENCE_RISK = self._get_float("HIGH_CONFIDENCE_RISK", 0.05)
        self.DAILY_LOSS_LIMIT = self._get_float("DAILY_LOSS_LIMIT", 0.10)
        self.WEEKLY_LOSS_LIMIT = self._get_float("WEEKLY_LOSS_LIMIT", 0.20)
        self.MAX_DRAWDOWN = self._get_float("MAX_DRAWDOWN", 0.30)
        self.MAX_POSITIONS = self._get_int("MAX_POSITIONS", 5)
        self.MAX_POSITIONS_PER_PAIR = self._get_int("MAX_POSITIONS_PER_PAIR", 2)

        # Trading Strategy Configuration
        strategies_str = os.getenv("ENABLED_STRATEGIES", "scalping,day_trading,swing_trading")
        self.ENABLED_STRATEGIES = [s.strip() for s in strategies_str.split(",")]

        # Auto-Execute Thresholds
        self.AUTO_EXECUTE_CONFIDENCE = self._get_float("AUTO_EXECUTE_CONFIDENCE", 0.80)
        self.AUTO_EXECUTE_RISK_REWARD = self._get_float("AUTO_EXECUTE_RISK_REWARD", 3.0)
        self.AUTO_EXECUTE_MIN_INDICATORS = self._get_int("AUTO_EXECUTE_MIN_INDICATORS", 4)

        # Currency Pairs
        pairs_str = os.getenv("TRADING_PAIRS", "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,EURJPY,GBPJPY")
        self.TRADING_PAIRS = [s.strip() for s in pairs_str.split(",")]

        # Leverage Settings
        self.LEVERAGE_SCALPING = self._get_int("LEVERAGE_SCALPING", 50)
        self.LEVERAGE_DAY_TRADING = self._get_int("LEVERAGE_DAY_TRADING", 30)
        self.LEVERAGE_SWING_TRADING = self._get_int("LEVERAGE_SWING_TRADING", 20)

        # Market Data
        self.ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

        # Data refresh intervals
        self.DATA_REFRESH_SCALPING = self._get_int("DATA_REFRESH_SCALPING", 5)
        self.DATA_REFRESH_DAY_TRADING = self._get_int("DATA_REFRESH_DAY_TRADING", 60)
        self.DATA_REFRESH_SWING_TRADING = self._get_int("DATA_REFRESH_SWING_TRADING", 300)

        # Monitoring & Alerting
        self.ENABLE_PERFORMANCE_MONITORING = self._get_bool("ENABLE_PERFORMANCE_MONITORING", True)
        self.ENABLE_ERROR_ALERTS = self._get_bool("ENABLE_ERROR_ALERTS", True)
        self.HEALTH_CHECK_INTERVAL = self._get_int("HEALTH_CHECK_INTERVAL", 60)

        # Advanced Settings
        self.ENABLE_NEWS_FILTER = self._get_bool("ENABLE_NEWS_FILTER", True)
        self.NEWS_MINUTES_BEFORE = self._get_int("NEWS_MINUTES_BEFORE", 30)
        self.NEWS_MINUTES_AFTER = self._get_int("NEWS_MINUTES_AFTER", 30)

        # Spread Filters
        self.MAX_SPREAD_SCALPING = self._get_float("MAX_SPREAD_SCALPING", 3.0)
        self.MAX_SPREAD_DAY_TRADING = self._get_float("MAX_SPREAD_DAY_TRADING", 5.0)
        self.MAX_SPREAD_SWING_TRADING = self._get_float("MAX_SPREAD_SWING_TRADING", 8.0)

        # Correlation Filter
        self.ENABLE_CORRELATION_FILTER = self._get_bool("ENABLE_CORRELATION_FILTER", True)
        self.MAX_CORRELATED_PAIRS = self._get_int("MAX_CORRELATED_PAIRS", 2)

        # Backtesting
        self.BACKTEST_START_DATE = os.getenv("BACKTEST_START_DATE", "2023-01-01")
        self.BACKTEST_END_DATE = os.getenv("BACKTEST_END_DATE", "2024-12-31")

    def _get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    def _get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Get integer value from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            if default is not None:
                return default
            raise ConfigError(f"Invalid integer value for {key}: {value}")

    def _get_float(self, key: str, default: float) -> float:
        """Get float value from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            raise ConfigError(f"Invalid float value for {key}: {value}")

    def _validate(self) -> None:
        """Validate configuration values."""

        # Validate environment
        if self.ENVIRONMENT not in ["development", "staging", "production"]:
            raise ConfigError(f"Invalid ENVIRONMENT: {self.ENVIRONMENT}. Must be development, staging, or production")

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.LOG_LEVEL not in valid_log_levels:
            raise ConfigError(f"Invalid LOG_LEVEL: {self.LOG_LEVEL}. Must be one of {valid_log_levels}")

        # Validate broker
        if self.BROKER not in ["mt5", "oanda", "demo"]:
            raise ConfigError(f"Invalid BROKER: {self.BROKER}. Must be mt5, oanda, or demo")

        # Validate broker-specific credentials
        if self.BROKER == "mt5":
            if not self.MT5_LOGIN or not self.MT5_PASSWORD or not self.MT5_SERVER:
                raise ConfigError("MT5 broker requires MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER to be set")

        elif self.BROKER == "oanda":
            if not self.OANDA_API_KEY or not self.OANDA_ACCOUNT_ID:
                raise ConfigError("OANDA broker requires OANDA_API_KEY and OANDA_ACCOUNT_ID to be set")

            if self.OANDA_ENVIRONMENT not in ["practice", "live"]:
                raise ConfigError(f"Invalid OANDA_ENVIRONMENT: {self.OANDA_ENVIRONMENT}. Must be practice or live")

        # Telegram bot validation (required for notifications)
        if not self.TELEGRAM_BOT_TOKEN:
            raise ConfigError("TELEGRAM_BOT_TOKEN is required for user notifications")

        if not self.TELEGRAM_USER_ID:
            raise ConfigError("TELEGRAM_USER_ID is required for user notifications")

        # Database validation
        if not self.DATABASE_URL:
            raise ConfigError("DATABASE_URL is required (or POSTGRES_USER/POSTGRES_PASSWORD for auto-generation)")

        # Risk management validation
        if not (0 < self.RISK_PER_TRADE <= 0.1):
            raise ConfigError(f"RISK_PER_TRADE must be between 0 and 0.1 (0-10%), got {self.RISK_PER_TRADE}")

        if not (0 < self.HIGH_CONFIDENCE_RISK <= 0.2):
            raise ConfigError(f"HIGH_CONFIDENCE_RISK must be between 0 and 0.2 (0-20%), got {self.HIGH_CONFIDENCE_RISK}")

        if not (0 < self.DAILY_LOSS_LIMIT <= 0.5):
            raise ConfigError(f"DAILY_LOSS_LIMIT must be between 0 and 0.5 (0-50%), got {self.DAILY_LOSS_LIMIT}")

        if not (0 < self.WEEKLY_LOSS_LIMIT <= 1.0):
            raise ConfigError(f"WEEKLY_LOSS_LIMIT must be between 0 and 1.0 (0-100%), got {self.WEEKLY_LOSS_LIMIT}")

        if not (0 < self.MAX_DRAWDOWN <= 1.0):
            raise ConfigError(f"MAX_DRAWDOWN must be between 0 and 1.0 (0-100%), got {self.MAX_DRAWDOWN}")

        # Auto-execute validation
        if not (0.5 <= self.AUTO_EXECUTE_CONFIDENCE <= 1.0):
            raise ConfigError(f"AUTO_EXECUTE_CONFIDENCE must be between 0.5 and 1.0, got {self.AUTO_EXECUTE_CONFIDENCE}")

        if not (1.0 <= self.AUTO_EXECUTE_RISK_REWARD <= 10.0):
            raise ConfigError(f"AUTO_EXECUTE_RISK_REWARD must be between 1.0 and 10.0, got {self.AUTO_EXECUTE_RISK_REWARD}")

        # Strategy validation
        valid_strategies = ["scalping", "day_trading", "swing_trading"]
        for strategy in self.ENABLED_STRATEGIES:
            if strategy not in valid_strategies:
                raise ConfigError(f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}")

        if not self.ENABLED_STRATEGIES:
            raise ConfigError("At least one strategy must be enabled")

        # Currency pair validation
        if not self.TRADING_PAIRS:
            raise ConfigError("At least one trading pair must be specified")

        # Validate currency pair format (basic check)
        for pair in self.TRADING_PAIRS:
            if len(pair) != 6:
                raise ConfigError(f"Invalid currency pair format: {pair}. Must be 6 characters (e.g., EURUSD)")

    def get_broker_config(self) -> Dict[str, Any]:
        """Get broker-specific configuration."""
        if self.BROKER == "mt5":
            return {
                "login": self.MT5_LOGIN,
                "password": self.MT5_PASSWORD,
                "server": self.MT5_SERVER
            }
        elif self.BROKER == "oanda":
            return {
                "api_key": self.OANDA_API_KEY,
                "account_id": self.OANDA_ACCOUNT_ID,
                "environment": self.OANDA_ENVIRONMENT
            }
        elif self.BROKER == "demo":
            return {
                "initial_balance": self.DEMO_INITIAL_BALANCE,
                "slippage_pips": self.DEMO_SLIPPAGE_PIPS
            }
        else:
            raise ConfigError(f"Unknown broker: {self.BROKER}")

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"


# Global config instance (loaded once at startup)
config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Config: The loaded configuration

    Raises:
        ConfigError: If configuration hasn't been loaded yet
    """
    global config
    if config is None:
        raise ConfigError("Configuration not loaded. Call load_config() first.")
    return config


def load_config(env_file: Optional[str] = None) -> Config:
    """
    Load and validate configuration from environment.

    Args:
        env_file: Optional path to .env file

    Returns:
        Config: The loaded and validated configuration
    """
    global config
    config = Config.load(env_file)
    return config