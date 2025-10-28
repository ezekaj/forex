"""
Test configuration management and validation.

Tests for Config class including:
- Environment variable loading
- Validation rules
- Error handling
- Broker-specific configurations
"""
import pytest
import os
from unittest.mock import patch
from forex_system.config import Config, ConfigError


class TestConfigLoading:
    """Test configuration loading from environment variables."""

    def test_default_values(self):
        """Test default configuration values."""
        config = Config()
        assert config.ENVIRONMENT == "development"
        assert config.DEBUG is False
        assert config.LOG_LEVEL == "INFO"
        assert config.BROKER == "demo"
        assert config.RISK_PER_TRADE == 0.03
        assert config.MAX_POSITIONS == 5

    @patch.dict(os.environ, {"ENVIRONMENT": "production", "DEBUG": "true"})
    def test_env_variable_loading(self):
        """Test loading values from environment variables."""
        config = Config()
        config._load_from_env()
        assert config.ENVIRONMENT == "production"
        assert config.DEBUG is True

    @patch.dict(os.environ, {"RISK_PER_TRADE": "0.05", "MAX_POSITIONS": "10"})
    def test_numeric_env_loading(self):
        """Test loading numeric values from environment."""
        config = Config()
        config._load_from_env()
        assert config.RISK_PER_TRADE == 0.05
        assert config.MAX_POSITIONS == 10

    @patch.dict(os.environ, {"TRADING_PAIRS": "EURUSD,GBPUSD,USDJPY"})
    def test_list_env_loading(self):
        """Test loading comma-separated lists from environment."""
        config = Config()
        config._load_from_env()
        assert config.TRADING_PAIRS == ["EURUSD", "GBPUSD", "USDJPY"]

    def test_database_url_construction(self):
        """Test automatic DATABASE_URL construction."""
        config = Config()
        config.POSTGRES_USER = "testuser"
        config.POSTGRES_PASSWORD = "testpass"
        config.POSTGRES_HOST = "localhost"
        config.POSTGRES_PORT = 5432
        config.POSTGRES_DB = "testdb"
        config._load_from_env()

        expected_url = "postgresql://testuser:testpass@localhost:5432/testdb"
        assert config.DATABASE_URL == expected_url

    def test_redis_url_construction_with_password(self):
        """Test Redis URL construction with password."""
        config = Config()
        config.REDIS_HOST = "localhost"
        config.REDIS_PORT = 6379
        config.REDIS_PASSWORD = "secret"
        config.REDIS_DB = 0
        config._load_from_env()

        expected_url = "redis://:secret@localhost:6379/0"
        assert config.REDIS_URL == expected_url

    def test_redis_url_construction_without_password(self):
        """Test Redis URL construction without password."""
        config = Config()
        config.REDIS_HOST = "localhost"
        config.REDIS_PORT = 6379
        config.REDIS_PASSWORD = None
        config.REDIS_DB = 0
        config._load_from_env()

        expected_url = "redis://localhost:6379/0"
        assert config.REDIS_URL == expected_url


class TestConfigValidation:
    """Test configuration validation rules."""

    def test_invalid_environment(self):
        """Test validation fails with invalid environment."""
        config = Config()
        config.ENVIRONMENT = "invalid"

        with pytest.raises(ConfigError, match="Invalid ENVIRONMENT"):
            config._validate()

    def test_valid_environments(self):
        """Test all valid environment values."""
        for env in ["development", "staging", "production"]:
            config = Config()
            config.ENVIRONMENT = env
            config.BROKER = "demo"
            config.LOG_LEVEL = "INFO"
            config.DATABASE_URL = "sqlite:///:memory:"
            # Should not raise
            try:
                config._validate()
            except ConfigError:
                pytest.fail(f"Validation failed for valid environment: {env}")

    def test_invalid_log_level(self):
        """Test validation fails with invalid log level."""
        config = Config()
        config.LOG_LEVEL = "INVALID"
        config.BROKER = "demo"

        with pytest.raises(ConfigError, match="Invalid LOG_LEVEL"):
            config._validate()

    def test_valid_log_levels(self):
        """Test all valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = Config()
            config.LOG_LEVEL = level
            config.BROKER = "demo"
            config.DATABASE_URL = "sqlite:///:memory:"
            # Should not raise
            try:
                config._validate()
            except ConfigError:
                pytest.fail(f"Validation failed for valid log level: {level}")

    def test_invalid_broker(self):
        """Test validation fails with invalid broker."""
        config = Config()
        config.BROKER = "invalid"

        with pytest.raises(ConfigError, match="Invalid BROKER"):
            config._validate()

    def test_mt5_broker_missing_credentials(self):
        """Test MT5 broker requires credentials."""
        config = Config()
        config.BROKER = "mt5"
        config.MT5_LOGIN = None
        config.DATABASE_URL = "sqlite:///:memory:"

        with pytest.raises(ConfigError, match="MT5 broker requires"):
            config._validate()

    def test_mt5_broker_with_credentials(self):
        """Test MT5 broker validation passes with credentials."""
        config = Config()
        config.BROKER = "mt5"
        config.MT5_LOGIN = "12345"
        config.MT5_PASSWORD = "password"
        config.MT5_SERVER = "Demo-Server"
        config.DATABASE_URL = "sqlite:///:memory:"

        # Should not raise
        try:
            config._validate()
        except ConfigError:
            pytest.fail("MT5 validation failed with valid credentials")

    def test_oanda_broker_missing_credentials(self):
        """Test OANDA broker requires credentials."""
        config = Config()
        config.BROKER = "oanda"
        config.OANDA_API_KEY = None
        config.DATABASE_URL = "sqlite:///:memory:"

        with pytest.raises(ConfigError, match="OANDA broker requires"):
            config._validate()

    def test_oanda_broker_with_credentials(self):
        """Test OANDA broker validation passes with credentials."""
        config = Config()
        config.BROKER = "oanda"
        config.OANDA_API_KEY = "test-api-key"
        config.OANDA_ACCOUNT_ID = "001-001-123456-001"
        config.OANDA_ENVIRONMENT = "practice"
        config.DATABASE_URL = "sqlite:///:memory:"

        # Should not raise
        try:
            config._validate()
        except ConfigError:
            pytest.fail("OANDA validation failed with valid credentials")

    def test_invalid_oanda_environment(self):
        """Test OANDA environment must be practice or live."""
        config = Config()
        config.BROKER = "oanda"
        config.OANDA_API_KEY = "test-key"
        config.OANDA_ACCOUNT_ID = "test-account"
        config.OANDA_ENVIRONMENT = "invalid"
        config.DATABASE_URL = "sqlite:///:memory:"

        with pytest.raises(ConfigError, match="Invalid OANDA_ENVIRONMENT"):
            config._validate()


class TestRiskManagementValidation:
    """Test risk management parameter validation."""

    def test_risk_per_trade_too_high(self):
        """Test RISK_PER_TRADE must be <= 10%."""
        config = Config()
        config.RISK_PER_TRADE = 0.15  # 15%
        config.BROKER = "demo"
        config.DATABASE_URL = "sqlite:///:memory:"

        with pytest.raises(ConfigError, match="RISK_PER_TRADE must be between"):
            config._validate()

    def test_risk_per_trade_negative(self):
        """Test RISK_PER_TRADE must be positive."""
        config = Config()
        config.RISK_PER_TRADE = -0.01
        config.BROKER = "demo"
        config.DATABASE_URL = "sqlite:///:memory:"

        with pytest.raises(ConfigError, match="RISK_PER_TRADE must be between"):
            config._validate()

    def test_risk_per_trade_valid_range(self):
        """Test valid risk per trade values."""
        for risk in [0.01, 0.03, 0.05, 0.10]:
            config = Config()
            config.RISK_PER_TRADE = risk
            config.BROKER = "demo"
            config.DATABASE_URL = "sqlite:///:memory:"

            try:
                config._validate()
            except ConfigError:
                pytest.fail(f"Validation failed for valid risk: {risk}")

    def test_high_confidence_risk_too_high(self):
        """Test HIGH_CONFIDENCE_RISK must be <= 20%."""
        config = Config()
        config.HIGH_CONFIDENCE_RISK = 0.25
        config.BROKER = "demo"
        config.DATABASE_URL = "sqlite:///:memory:"

        with pytest.raises(ConfigError, match="HIGH_CONFIDENCE_RISK must be between"):
            config._validate()

    def test_daily_loss_limit_too_high(self):
        """Test DAILY_LOSS_LIMIT must be <= 50%."""
        config = Config()
        config.DAILY_LOSS_LIMIT = 0.60
        config.BROKER = "demo"
        config.DATABASE_URL = "sqlite:///:memory:"

        with pytest.raises(ConfigError, match="DAILY_LOSS_LIMIT must be between"):
            config._validate()

    def test_max_drawdown_valid(self):
        """Test MAX_DRAWDOWN validation."""
        config = Config()
        config.MAX_DRAWDOWN = 0.30
        config.BROKER = "demo"
        config.DATABASE_URL = "sqlite:///:memory:"

        try:
            config._validate()
        except ConfigError:
            pytest.fail("MAX_DRAWDOWN validation failed for valid value")


class TestAutoExecuteValidation:
    """Test auto-execute threshold validation."""

    def test_auto_execute_confidence_too_low(self):
        """Test AUTO_EXECUTE_CONFIDENCE must be >= 50%."""
        config = Config()
        config.AUTO_EXECUTE_CONFIDENCE = 0.40
        config.BROKER = "demo"
        config.DATABASE_URL = "sqlite:///:memory:"

        with pytest.raises(ConfigError, match="AUTO_EXECUTE_CONFIDENCE must be between"):
            config._validate()

    def test_auto_execute_confidence_too_high(self):
        """Test AUTO_EXECUTE_CONFIDENCE must be <= 100%."""
        config = Config()
        config.AUTO_EXECUTE_CONFIDENCE = 1.1
        config.BROKER = "demo"
        config.DATABASE_URL = "sqlite:///:memory:"

        with pytest.raises(ConfigError, match="AUTO_EXECUTE_CONFIDENCE must be between"):
            config._validate()

    def test_auto_execute_risk_reward_too_low(self):
        """Test AUTO_EXECUTE_RISK_REWARD must be >= 1.0."""
        config = Config()
        config.AUTO_EXECUTE_RISK_REWARD = 0.5
        config.BROKER = "demo"
        config.DATABASE_URL = "sqlite:///:memory:"

        with pytest.raises(ConfigError, match="AUTO_EXECUTE_RISK_REWARD must be between"):
            config._validate()


class TestStrategyValidation:
    """Test trading strategy validation."""

    def test_invalid_strategy(self):
        """Test validation fails with invalid strategy."""
        config = Config()
        config.ENABLED_STRATEGIES = ["invalid_strategy"]
        config.BROKER = "demo"
        config.DATABASE_URL = "sqlite:///:memory:"

        with pytest.raises(ConfigError, match="Invalid strategy"):
            config._validate()

    def test_no_strategies_enabled(self):
        """Test at least one strategy must be enabled."""
        config = Config()
        config.ENABLED_STRATEGIES = []
        config.BROKER = "demo"
        config.DATABASE_URL = "sqlite:///:memory:"

        with pytest.raises(ConfigError, match="At least one strategy must be enabled"):
            config._validate()

    def test_valid_strategies(self):
        """Test all valid strategy values."""
        valid_strategies = ["scalping", "day_trading", "swing_trading"]
        config = Config()
        config.ENABLED_STRATEGIES = valid_strategies
        config.BROKER = "demo"
        config.DATABASE_URL = "sqlite:///:memory:"

        try:
            config._validate()
        except ConfigError:
            pytest.fail("Validation failed for valid strategies")


class TestCurrencyPairValidation:
    """Test currency pair validation."""

    def test_no_trading_pairs(self):
        """Test at least one trading pair must be specified."""
        config = Config()
        config.TRADING_PAIRS = []
        config.BROKER = "demo"
        config.DATABASE_URL = "sqlite:///:memory:"

        with pytest.raises(ConfigError, match="At least one trading pair must be specified"):
            config._validate()

    def test_invalid_pair_length(self):
        """Test currency pair must be 6 characters."""
        config = Config()
        config.TRADING_PAIRS = ["EUR"]
        config.BROKER = "demo"
        config.DATABASE_URL = "sqlite:///:memory:"

        with pytest.raises(ConfigError, match="Invalid currency pair format"):
            config._validate()

    def test_valid_currency_pairs(self):
        """Test valid currency pairs."""
        config = Config()
        config.TRADING_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
        config.BROKER = "demo"
        config.DATABASE_URL = "sqlite:///:memory:"

        try:
            config._validate()
        except ConfigError:
            pytest.fail("Validation failed for valid currency pairs")


class TestHelperMethods:
    """Test configuration helper methods."""

    def test_get_broker_config_demo(self):
        """Test get_broker_config for demo broker."""
        config = Config()
        config.BROKER = "demo"
        config.DEMO_INITIAL_BALANCE = 15000.0
        config.DEMO_SLIPPAGE_PIPS = 1.0

        broker_config = config.get_broker_config()

        assert broker_config["initial_balance"] == 15000.0
        assert broker_config["slippage_pips"] == 1.0

    def test_get_broker_config_mt5(self):
        """Test get_broker_config for MT5 broker."""
        config = Config()
        config.BROKER = "mt5"
        config.MT5_LOGIN = "12345"
        config.MT5_PASSWORD = "password"
        config.MT5_SERVER = "Demo-Server"

        broker_config = config.get_broker_config()

        assert broker_config["login"] == "12345"
        assert broker_config["password"] == "password"
        assert broker_config["server"] == "Demo-Server"

    def test_get_broker_config_oanda(self):
        """Test get_broker_config for OANDA broker."""
        config = Config()
        config.BROKER = "oanda"
        config.OANDA_API_KEY = "test-key"
        config.OANDA_ACCOUNT_ID = "test-account"
        config.OANDA_ENVIRONMENT = "practice"

        broker_config = config.get_broker_config()

        assert broker_config["api_key"] == "test-key"
        assert broker_config["account_id"] == "test-account"
        assert broker_config["environment"] == "practice"

    def test_is_production(self):
        """Test is_production() method."""
        config = Config()
        config.ENVIRONMENT = "production"
        assert config.is_production() is True

        config.ENVIRONMENT = "development"
        assert config.is_production() is False

    def test_is_development(self):
        """Test is_development() method."""
        config = Config()
        config.ENVIRONMENT = "development"
        assert config.is_development() is True

        config.ENVIRONMENT = "production"
        assert config.is_development() is False


class TestDatabaseValidation:
    """Test database configuration validation."""

    def test_missing_database_url(self):
        """Test DATABASE_URL is required."""
        config = Config()
        config.BROKER = "demo"
        config.DATABASE_URL = None
        config.POSTGRES_USER = None

        with pytest.raises(ConfigError, match="DATABASE_URL is required"):
            config._validate()

    def test_database_url_auto_generated(self):
        """Test DATABASE_URL can be auto-generated from components."""
        config = Config()
        config.BROKER = "demo"
        config.POSTGRES_USER = "user"
        config.POSTGRES_PASSWORD = "pass"
        config.POSTGRES_HOST = "localhost"
        config.POSTGRES_PORT = 5432
        config.POSTGRES_DB = "forex"
        config._load_from_env()

        assert config.DATABASE_URL is not None
        assert "user" in config.DATABASE_URL
        assert "pass" in config.DATABASE_URL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
