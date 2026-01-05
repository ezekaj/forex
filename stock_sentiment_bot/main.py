"""
Stock Sentiment Trading Bot - Entry Point

A sentiment-driven stock trading bot using Reddit/news sentiment
with Quarter-Kelly position sizing and hard circuit breakers.

Usage:
    python main.py              # Run the bot
    python main.py --dry-run    # Simulate without executing trades
    python main.py --backtest   # Run backtesting
"""
import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from config.constants import LOG_FORMAT, LOG_DATE_FORMAT


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                f"logs/bot_{datetime.now().strftime('%Y%m%d')}.log",
                mode='a'
            )
        ]
    )
    return logging.getLogger("sentiment_bot")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stock Sentiment Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Simulate trades without executing (overrides .env setting)"
    )

    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtesting mode"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Backtest start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="Backtest end date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level (overrides .env setting)"
    )

    return parser.parse_args()


class SentimentBot:
    """Main bot orchestrator."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.running = False
        self.logger = logging.getLogger("sentiment_bot.main")

        # Components (will be initialized in setup)
        self.alpaca_client = None
        self.reddit_scraper = None
        self.news_scraper = None
        self.sentiment_scorer = None
        self.signal_generator = None
        self.risk_manager = None
        self.executor = None

    async def setup(self):
        """Initialize all components."""
        self.logger.info("Initializing bot components...")

        settings = get_settings()

        # Validate required configs
        try:
            settings.validate()
        except ValueError as e:
            self.logger.error(f"Configuration error: {e}")
            raise

        # Import and initialize components
        # TODO: Uncomment as components are implemented
        #
        # from data.alpaca_client import AlpacaClient
        # from data.reddit_scraper import RedditScraper
        # from data.news_scraper import NewsScraper
        # from analysis.sentiment_scorer import SentimentScorer
        # from trading.signal_generator import SignalGenerator
        # from trading.risk_manager import RiskManager
        # from trading.executor import TradeExecutor
        #
        # self.alpaca_client = AlpacaClient(settings.alpaca)
        # self.reddit_scraper = RedditScraper(settings.reddit)
        # self.news_scraper = NewsScraper()
        # self.sentiment_scorer = SentimentScorer()
        # self.signal_generator = SignalGenerator()
        # self.risk_manager = RiskManager()
        # self.executor = TradeExecutor(self.alpaca_client, dry_run=self.dry_run)

        mode = "DRY RUN" if self.dry_run else "LIVE"
        self.logger.info(f"Bot initialized in {mode} mode")

    async def run(self):
        """Main bot loop."""
        self.running = True
        self.logger.info("Starting bot main loop...")

        while self.running:
            try:
                await self._trading_cycle()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in trading cycle: {e}", exc_info=True)
                await asyncio.sleep(300)  # Wait 5 min on error

    async def _trading_cycle(self):
        """Single trading cycle - analyze and potentially execute trades."""
        # TODO: Implement full trading cycle
        #
        # 1. Check if market is open
        # 2. Scrape Reddit/news for sentiment
        # 3. Score tickers by sentiment momentum
        # 4. Generate signals for high-scoring tickers
        # 5. Validate against risk manager (circuit breakers)
        # 6. Execute approved trades
        # 7. Manage existing positions (stops, take profits)
        #
        self.logger.debug("Trading cycle executed (not yet implemented)")

    async def shutdown(self):
        """Graceful shutdown."""
        self.logger.info("Shutting down bot...")
        self.running = False

        # Close connections
        # TODO: Implement cleanup for each component


async def run_backtest(start_date: str, end_date: str):
    """Run backtesting mode."""
    logger = logging.getLogger("sentiment_bot.backtest")
    logger.info(f"Starting backtest from {start_date} to {end_date}")

    # TODO: Implement backtesting
    # from backtest.engine import BacktestEngine
    # engine = BacktestEngine(start_date, end_date)
    # results = await engine.run()
    # engine.print_report(results)

    logger.warning("Backtesting not yet implemented")


def main():
    """Main entry point."""
    args = parse_args()

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    # Setup logging
    settings = get_settings()
    log_level = args.log_level or settings.log_level
    logger = setup_logging(log_level)

    logger.info("=" * 60)
    logger.info("STOCK SENTIMENT BOT - Starting")
    logger.info("=" * 60)

    # Determine dry run mode
    dry_run = args.dry_run if args.dry_run is not None else settings.dry_run

    if dry_run:
        logger.info("Running in DRY RUN mode - no real trades will be executed")
    else:
        logger.warning("Running in LIVE mode - real money at risk!")

    # Handle mode selection
    if args.backtest:
        # Backtesting mode
        start = args.start_date or "2024-01-01"
        end = args.end_date or datetime.now().strftime("%Y-%m-%d")
        asyncio.run(run_backtest(start, end))
    else:
        # Live/paper trading mode
        bot = SentimentBot(dry_run=dry_run)

        # Setup signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            asyncio.create_task(bot.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Run the bot
        try:
            asyncio.run(bot.setup())
            asyncio.run(bot.run())
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            sys.exit(1)

    logger.info("Bot shutdown complete")


if __name__ == "__main__":
    main()
