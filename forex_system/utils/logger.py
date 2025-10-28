"""
Structured logging system for the Forex Trading System.

Replaces all print() statements with proper structured logging.
Supports JSON formatting for production and human-readable for development.
Implements log rotation, multiple handlers, and sensitive data sanitization.
"""
import logging
import logging.handlers
import sys
import os
import re
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from pythonjsonlogger import jsonlogger


class SensitiveDataFilter(logging.Filter):
    """
    Filter to redact sensitive data from logs.
    Prevents passwords, API keys, and credentials from being logged.
    """

    SENSITIVE_PATTERNS = [
        (re.compile(r'(password["\']?\s*[:=]\s*["\']?)([^"\'}\s]+)', re.IGNORECASE), r'\1***'),
        (re.compile(r'(api[_-]?key["\']?\s*[:=]\s*["\']?)([^"\'}\s]+)', re.IGNORECASE), r'\1***'),
        (re.compile(r'(token["\']?\s*[:=]\s*["\']?)([^"\'}\s]+)', re.IGNORECASE), r'\1***'),
        (re.compile(r'(secret["\']?\s*[:=]\s*["\']?)([^"\'}\s]+)', re.IGNORECASE), r'\1***'),
        (re.compile(r'(MT5_PASSWORD["\']?\s*[:=]\s*["\']?)([^"\'}\s]+)', re.IGNORECASE), r'\1***'),
        (re.compile(r'(OANDA_API_KEY["\']?\s*[:=]\s*["\']?)([^"\'}\s]+)', re.IGNORECASE), r'\1***'),
        (re.compile(r'(TELEGRAM_BOT_TOKEN["\']?\s*[:=]\s*["\']?)([^"\'}\s]+)', re.IGNORECASE), r'\1***'),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Sanitize the log message before it's emitted."""
        if isinstance(record.msg, str):
            for pattern, replacement in self.SENSITIVE_PATTERNS:
                record.msg = pattern.sub(replacement, record.msg)

        # Also sanitize args if present
        if record.args:
            sanitized_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    for pattern, replacement in self.SENSITIVE_PATTERNS:
                        arg = pattern.sub(replacement, arg)
                sanitized_args.append(arg)
            record.args = tuple(sanitized_args)

        return True


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter for development environments.
    Makes logs easier to read with color-coded levels.
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Add color to level name
        record.levelname = f"{log_color}{record.levelname}{reset}"

        return super().format(record)


class StructuredLogger:
    """
    Centralized logging setup.
    Creates loggers for different services with consistent formatting.
    """

    _loggers: Dict[str, logging.Logger] = {}
    _initialized = False

    @classmethod
    def initialize(cls, log_level: str = "INFO", environment: str = "development", log_dir: str = "logs") -> None:
        """
        Initialize the logging system.
        Should be called once at application startup.

        Args:
            log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            environment: Application environment (development, staging, production)
            log_dir: Directory for log files
        """
        if cls._initialized:
            return

        # Create logs directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        cls.log_level = getattr(logging, log_level.upper())
        cls.environment = environment
        cls.log_dir = log_dir
        cls._initialized = True

    @classmethod
    def get_logger(cls, service_name: str, extra_context: Optional[Dict[str, Any]] = None) -> logging.Logger:
        """
        Get or create a logger for a specific service.

        Args:
            service_name: Name of the service (e.g., 'market_analyzer', 'trade_executor')
            extra_context: Additional context to include in all log messages

        Returns:
            Configured logger instance
        """
        if not cls._initialized:
            cls.initialize()

        # Return existing logger if already created
        if service_name in cls._loggers:
            return cls._loggers[service_name]

        # Create new logger
        logger = logging.getLogger(service_name)
        logger.setLevel(cls.log_level)
        logger.propagate = False  # Don't propagate to root logger

        # Add sensitive data filter
        logger.addFilter(SensitiveDataFilter())

        # Console handler (stdout) - human-readable for development
        if cls.environment == "development":
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(cls.log_level)

            # Colored formatter for console
            console_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            console_formatter = ColoredFormatter(
                console_format,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # Main log file handler - JSON format for easy parsing
        main_log_file = os.path.join(cls.log_dir, 'forex_system.log')
        main_handler = logging.handlers.TimedRotatingFileHandler(
            main_log_file,
            when='midnight',
            interval=1,
            backupCount=30,  # Keep 30 days
            encoding='utf-8'
        )
        main_handler.setLevel(logging.INFO)

        # JSON formatter for file logs
        json_formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s %(context)s',
            rename_fields={'levelname': 'level', 'name': 'service'}
        )
        main_handler.setFormatter(json_formatter)
        logger.addHandler(main_handler)

        # Error log file handler - WARNING and above only
        error_log_file = os.path.join(cls.log_dir, 'errors.log')
        error_handler = logging.handlers.TimedRotatingFileHandler(
            error_log_file,
            when='midnight',
            interval=1,
            backupCount=90,  # Keep 90 days for errors
            encoding='utf-8'
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(json_formatter)
        logger.addHandler(error_handler)

        # Store logger
        cls._loggers[service_name] = logger

        return logger

    @classmethod
    def get_trade_logger(cls) -> logging.Logger:
        """
        Get specialized logger for trade-related events.
        Trades are logged separately for audit purposes.

        Returns:
            Trade logger instance
        """
        if 'trades' in cls._loggers:
            return cls._loggers['trades']

        if not cls._initialized:
            cls.initialize()

        logger = logging.getLogger('trades')
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Add sensitive data filter
        logger.addFilter(SensitiveDataFilter())

        # Trade log file - rotates weekly, keeps 1 year for tax/audit
        trade_log_file = os.path.join(cls.log_dir, 'trades.log')
        trade_handler = logging.handlers.TimedRotatingFileHandler(
            trade_log_file,
            when='W0',  # Weekly rotation on Monday
            interval=1,
            backupCount=52,  # Keep 1 year
            encoding='utf-8'
        )
        trade_handler.setLevel(logging.INFO)

        # JSON formatter for trade logs
        json_formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(level)s %(message)s %(trade_data)s',
            rename_fields={'levelname': 'level'}
        )
        trade_handler.setFormatter(json_formatter)
        logger.addHandler(trade_handler)

        cls._loggers['trades'] = logger
        return logger

    @classmethod
    def log_trade_event(cls, event_type: str, trade_data: Dict[str, Any]) -> None:
        """
        Log a trade-related event with structured data.

        Args:
            event_type: Type of event (opportunity_generated, trade_executed, position_closed, etc.)
            trade_data: Trade-specific data to log
        """
        logger = cls.get_trade_logger()

        # Add timestamp and event type to trade data
        log_data = {
            'event': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            **trade_data
        }

        logger.info(f"Trade event: {event_type}", extra={'trade_data': log_data})

    @classmethod
    def log_opportunity(cls, opportunity_data: Dict[str, Any]) -> None:
        """Log a generated trading opportunity."""
        cls.log_trade_event('opportunity_generated', opportunity_data)

    @classmethod
    def log_trade_execution(cls, trade_data: Dict[str, Any]) -> None:
        """Log a trade execution."""
        cls.log_trade_event('trade_executed', trade_data)

    @classmethod
    def log_position_closed(cls, position_data: Dict[str, Any]) -> None:
        """Log a position closure."""
        cls.log_trade_event('position_closed', position_data)

    @classmethod
    def log_error_with_context(cls, logger: logging.Logger, error_msg: str, context: Dict[str, Any], exc_info: bool = True) -> None:
        """
        Log an error with full context and stack trace.

        Args:
            logger: Logger instance to use
            error_msg: Error message
            context: Additional context data
            exc_info: Include exception info (stack trace)
        """
        logger.error(
            error_msg,
            extra={'context': context},
            exc_info=exc_info
        )


def get_logger(service_name: str, extra_context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Convenience function to get a logger.
    Automatically initializes logging system if needed.

    Args:
        service_name: Name of the service
        extra_context: Additional context to include in all log messages

    Returns:
        Configured logger instance
    """
    return StructuredLogger.get_logger(service_name, extra_context)


# Example usage patterns for different scenarios
if __name__ == "__main__":
    # Initialize logging system
    StructuredLogger.initialize(log_level="INFO", environment="development")

    # Get logger for a service
    logger = get_logger("market_analyzer")

    # Simple logging
    logger.info("Market analysis started")

    # Structured logging with context
    logger.info("Trading signal generated", extra={
        'context': {
            'pair': 'EURUSD',
            'confidence': 0.85,
            'direction': 'BUY',
            'indicators': ['EMA_cross', 'RSI_oversold', 'MACD_bullish']
        }
    })

    # Error logging
    try:
        # Simulated error
        raise ValueError("Test error")
    except Exception as e:
        logger.error(
            "Order placement failed",
            extra={'context': {'pair': 'EURUSD', 'error': str(e)}},
            exc_info=True
        )

    # Trade event logging
    StructuredLogger.log_trade_execution({
        'trade_id': 'uuid-1234',
        'pair': 'EURUSD',
        'direction': 'BUY',
        'entry_price': 1.0850,
        'size': 0.1,
        'confidence': 0.82
    })

    logger.info("Example logging complete")