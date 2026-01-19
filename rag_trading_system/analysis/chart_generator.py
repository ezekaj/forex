"""
Chart Generator
===============
Creates candlestick chart images for vision model analysis.
"""

import io
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import mplfinance, fall back to basic matplotlib
try:
    import mplfinance as mpf
    HAS_MPLFINANCE = True
except ImportError:
    HAS_MPLFINANCE = False
    logger.warning("mplfinance not installed. Run: pip install mplfinance")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHARTS_DIR, CHART_CANDLES, CHART_WIDTH, CHART_HEIGHT, CHART_STYLE


class ChartGenerator:
    """Generates candlestick charts for LLM vision analysis."""

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or CHARTS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_chart(
        self,
        df: pd.DataFrame,
        pair: str,
        save_path: str = None,
        show_volume: bool = False,
        show_ema: bool = True,
        show_bb: bool = True,
        title: str = None
    ) -> Optional[str]:
        """
        Generate a candlestick chart image.

        Args:
            df: DataFrame with OHLCV data (must have datetime index)
            pair: Currency pair name
            save_path: Where to save the image
            show_volume: Whether to show volume
            show_ema: Whether to show EMA lines
            show_bb: Whether to show Bollinger Bands

        Returns:
            Path to saved image or None if failed
        """
        if not HAS_MPLFINANCE and not HAS_MATPLOTLIB:
            logger.error("No charting library available")
            return None

        # Ensure we have the right columns
        required_cols = ['open', 'high', 'low', 'close']
        df_cols_lower = [c.lower() for c in df.columns]

        if not all(col in df_cols_lower for col in required_cols):
            logger.error(f"Missing required columns. Has: {df.columns.tolist()}")
            return None

        # Normalize column names
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Take last N candles
        df = df.tail(CHART_CANDLES)

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                logger.error("Failed to convert index to datetime")
                return None

        # Generate filename
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"{pair}_{timestamp}.png"

        save_path = Path(save_path)

        if HAS_MPLFINANCE:
            return self._generate_with_mplfinance(
                df, pair, save_path, show_volume, show_ema, show_bb, title
            )
        else:
            return self._generate_with_matplotlib(df, pair, save_path, title)

    def _generate_with_mplfinance(
        self,
        df: pd.DataFrame,
        pair: str,
        save_path: Path,
        show_volume: bool,
        show_ema: bool,
        show_bb: bool,
        title: str
    ) -> str:
        """Generate chart using mplfinance."""
        try:
            # Build additional plots
            addplots = []

            if show_ema:
                # Calculate EMAs
                df['ema9'] = df['close'].ewm(span=9).mean()
                df['ema21'] = df['close'].ewm(span=21).mean()
                df['ema50'] = df['close'].ewm(span=50).mean()

                addplots.extend([
                    mpf.make_addplot(df['ema9'], color='blue', width=0.7),
                    mpf.make_addplot(df['ema21'], color='orange', width=0.7),
                    mpf.make_addplot(df['ema50'], color='purple', width=0.7),
                ])

            if show_bb:
                # Calculate Bollinger Bands
                df['bb_mid'] = df['close'].rolling(20).mean()
                df['bb_std'] = df['close'].rolling(20).std()
                df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
                df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)

                addplots.extend([
                    mpf.make_addplot(df['bb_upper'], color='gray', width=0.5, linestyle='--'),
                    mpf.make_addplot(df['bb_lower'], color='gray', width=0.5, linestyle='--'),
                ])

            # Chart style
            mc = mpf.make_marketcolors(
                up='#26a69a',
                down='#ef5350',
                edge='inherit',
                wick='inherit',
                volume='in'
            )
            style = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor='#e0e0e0'
            )

            # Generate chart
            fig, axes = mpf.plot(
                df,
                type='candle',
                style=style,
                title=title or f"{pair} - {df.index[-1].strftime('%Y-%m-%d %H:%M')}",
                volume=show_volume and 'volume' in df.columns,
                addplot=addplots if addplots else None,
                figsize=(CHART_WIDTH/100, CHART_HEIGHT/100),
                returnfig=True,
                tight_layout=True
            )

            fig.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            logger.info(f"Chart saved to: {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"Failed to generate chart with mplfinance: {e}")
            return None

    def _generate_with_matplotlib(
        self,
        df: pd.DataFrame,
        pair: str,
        save_path: Path,
        title: str
    ) -> str:
        """Fallback chart generation using basic matplotlib."""
        try:
            fig, ax = plt.subplots(figsize=(CHART_WIDTH/100, CHART_HEIGHT/100))

            # Simple line chart as fallback
            ax.plot(df.index, df['close'], 'b-', linewidth=1)
            ax.fill_between(df.index, df['low'], df['high'], alpha=0.3)

            ax.set_title(title or f"{pair} Price")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.grid(True, alpha=0.3)

            plt.xticks(rotation=45)
            plt.tight_layout()

            fig.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            logger.info(f"Chart saved to: {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"Failed to generate chart with matplotlib: {e}")
            return None

    def generate_analysis_chart(
        self,
        df: pd.DataFrame,
        pair: str,
        indicators: Dict = None,
        support_resistance: Dict = None,
        annotations: list = None
    ) -> Optional[str]:
        """
        Generate a comprehensive analysis chart with indicators and annotations.

        Args:
            df: OHLCV DataFrame
            pair: Currency pair
            indicators: Dict of indicator values to overlay
            support_resistance: Dict with 'support' and 'resistance' levels
            annotations: List of (date, price, text) annotations

        Returns:
            Path to saved chart
        """
        if not HAS_MPLFINANCE:
            return self.generate_chart(df, pair)

        try:
            df = df.copy().tail(CHART_CANDLES)
            df.columns = [c.lower() for c in df.columns]

            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            addplots = []

            # Add RSI if provided
            if indicators and 'rsi' in indicators:
                rsi = indicators['rsi'].tail(CHART_CANDLES)
                addplots.append(
                    mpf.make_addplot(rsi, panel=1, color='purple', ylabel='RSI')
                )

            # Add MACD if provided
            if indicators and 'macd' in indicators:
                macd = indicators['macd'].tail(CHART_CANDLES)
                macd_signal = indicators.get('macd_signal', pd.Series()).tail(CHART_CANDLES)
                if len(macd) > 0:
                    addplots.append(
                        mpf.make_addplot(macd, panel=2, color='blue', ylabel='MACD')
                    )
                    if len(macd_signal) > 0:
                        addplots.append(
                            mpf.make_addplot(macd_signal, panel=2, color='orange')
                        )

            # Chart style
            mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit', wick='inherit')
            style = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', gridcolor='#e0e0e0')

            # Support/Resistance lines
            hlines = {}
            if support_resistance:
                levels = []
                colors = []
                if 'support' in support_resistance:
                    for level in support_resistance['support']:
                        levels.append(level)
                        colors.append('green')
                if 'resistance' in support_resistance:
                    for level in support_resistance['resistance']:
                        levels.append(level)
                        colors.append('red')
                if levels:
                    hlines = {'hlines': levels, 'colors': colors, 'linestyle': '--', 'linewidths': 0.5}

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"{pair}_analysis_{timestamp}.png"

            fig, axes = mpf.plot(
                df,
                type='candle',
                style=style,
                title=f"{pair} Analysis - {df.index[-1].strftime('%Y-%m-%d %H:%M')}",
                volume=False,
                addplot=addplots if addplots else None,
                figsize=(14, 10),
                returnfig=True,
                tight_layout=True,
                **hlines
            )

            fig.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return str(save_path)

        except Exception as e:
            logger.error(f"Failed to generate analysis chart: {e}")
            return None


def generate_sample_data(pair: str = "EURUSD", bars: int = 200, end_date: datetime = None) -> pd.DataFrame:
    """Generate sample OHLCV data for testing.

    Args:
        pair: Currency pair
        bars: Number of bars to generate
        end_date: End date for the data (defaults to now)
    """
    np.random.seed(42 + hash(pair) % 1000)  # Different seed per pair

    end_date = end_date or datetime.now()
    dates = pd.date_range(end=end_date, periods=bars, freq='4h')

    # Starting price based on pair
    start_prices = {
        "EURUSD": 1.0850,
        "GBPUSD": 1.2650,
        "USDJPY": 149.50,
    }
    start_price = start_prices.get(pair, 1.0)

    # Generate random walk
    returns = np.random.normal(0, 0.001, bars)
    close = start_price * np.cumprod(1 + returns)

    # Generate OHLC from close
    volatility = np.abs(np.random.normal(0.0003, 0.0001, bars))
    high = close + volatility
    low = close - volatility
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.randint(1000, 10000, bars)

    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test chart generation
    generator = ChartGenerator()

    for pair in ["EURUSD", "GBPUSD"]:
        df = generate_sample_data(pair)
        chart_path = generator.generate_chart(df, pair)
        if chart_path:
            print(f"Generated chart: {chart_path}")
