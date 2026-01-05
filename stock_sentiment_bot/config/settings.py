"""
Configuration management for Stock Sentiment Bot.
Loads from environment variables with sensible defaults.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class AlpacaConfig:
    """Alpaca API configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv(
        "ALPACA_BASE_URL",
        "https://paper-api.alpaca.markets"  # Paper trading by default
    ))

    @property
    def is_paper(self) -> bool:
        return "paper" in self.base_url.lower()

    def validate(self) -> bool:
        if not self.api_key or not self.secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        return True


@dataclass
class RedditConfig:
    """Reddit API configuration."""
    client_id: str = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_ID", ""))
    client_secret: str = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_SECRET", ""))
    user_agent: str = field(default_factory=lambda: os.getenv(
        "REDDIT_USER_AGENT",
        "SentimentBot/1.0 (by /u/your_username)"
    ))

    # Subreddits to monitor
    subreddits: list = field(default_factory=lambda: [
        "wallstreetbets",
        "stocks",
        "options",
    ])

    def validate(self) -> bool:
        if not self.client_id or not self.client_secret:
            raise ValueError("REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set")
        return True


@dataclass
class TelegramConfig:
    """Telegram bot configuration for alerts."""
    bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)


@dataclass
class DatabaseConfig:
    """SQLite database configuration."""
    path: Path = field(default_factory=lambda: Path(
        os.getenv("DATABASE_PATH", "data/sentiment_bot.db")
    ))

    def ensure_directory(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    """Main settings container."""
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    reddit: RedditConfig = field(default_factory=RedditConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    # Global settings
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    dry_run: bool = field(default_factory=lambda: os.getenv("DRY_RUN", "true").lower() == "true")

    def validate(self) -> bool:
        """Validate all required configurations."""
        self.alpaca.validate()
        # Reddit validation optional - can work with news only
        # Telegram validation optional - alerts are nice-to-have
        return True

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from environment."""
        settings = cls()
        settings.database.ensure_directory()
        return settings


# Singleton instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get or create settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings
