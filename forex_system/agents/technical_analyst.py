"""Technical Analyst Agent — price indicators + Kronos forecasting."""

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

from forex_system.agents.base_agent import BaseAgent, AgentResult
from forex_system.training.config import TrainingConfig, get_asset
from forex_system.training.data.price_loader import UniversalPriceLoader

log = logging.getLogger(__name__)


class TechnicalAnalyst(BaseAgent):
    """
    Generates technical signals from price data.
    Uses 60+ indicators from FeatureEngineer + Kronos price forecast.
    No LLM needed — pure computation.
    """

    def __init__(self):
        super().__init__(name="technical_analyst")
        self.config = TrainingConfig()
        self.price_loader = UniversalPriceLoader(self.config)

    async def analyze_asset(
        self,
        symbol: str,
        date: str = None,
        lookback_days: int = 60,
    ) -> AgentResult:
        """
        Generate technical signals for a specific asset.

        Returns AgentResult with:
        - rsi, macd_signal, adx, atr_pct, bollinger_position
        - trend: bullish/bearish/neutral
        - momentum: strong/weak/neutral
        - volatility: high/normal/low
        - kronos_forecast: direction + confidence (if available)
        """
        start = time.perf_counter()

        from datetime import datetime as dt
        end = dt.strptime(date, "%Y-%m-%d") if date else None
        df = self.price_loader.load_ohlcv(symbol, "1d", end_date=end)
        if df.empty or len(df) < lookback_days:
            return AgentResult(
                agent=self.name, asset=symbol, timestamp=date or "",
                data={"error": "insufficient_data", "bars": len(df)},
            )

        # Use last lookback_days bars up to the target date (no look-ahead)
        df = df.tail(lookback_days + 50)

        # Calculate indicators
        signals = self._compute_indicators(df)

        # Determine overall signals
        trend = self._classify_trend(signals)
        momentum = self._classify_momentum(signals)
        volatility = self._classify_volatility(signals)

        # Overall direction
        bull_score = sum([
            1 if signals["rsi"] > 50 else 0,
            1 if signals["macd_hist"] > 0 else 0,
            1 if signals["price_vs_sma20"] > 0 else 0,
            1 if signals["price_vs_sma50"] > 0 else 0,
            1 if signals["adx"] > 25 and trend == "bullish" else 0,
        ])
        direction = "bullish" if bull_score >= 3 else ("bearish" if bull_score <= 1 else "neutral")
        confidence = abs(bull_score - 2.5) / 2.5 * 100

        elapsed = (time.perf_counter() - start) * 1000

        return AgentResult(
            agent=self.name,
            asset=symbol,
            timestamp=date or "",
            data={
                "direction": direction,
                "confidence": round(confidence),
                "trend": trend,
                "momentum": momentum,
                "volatility": volatility,
                "indicators": signals,
                "last_close": float(df["close"].iloc[-1]),
                "bars_analyzed": len(df),
            },
            processing_time_ms=elapsed,
        )

    async def analyze_all(
        self,
        symbols: list[str],
        date: str = None,
    ) -> dict[str, AgentResult]:
        """Analyze all assets."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = await self.analyze_asset(symbol, date)
            except Exception as e:
                log.warning(f"Technical analysis failed for {symbol}: {e}")
        return results

    def _compute_indicators(self, df: pd.DataFrame) -> dict:
        """Compute key technical indicators."""
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones(len(close))

        n = len(close)

        # SMA
        sma20 = np.mean(close[-20:]) if n >= 20 else close[-1]
        sma50 = np.mean(close[-50:]) if n >= 50 else close[-1]

        # RSI (14)
        rsi = self._rsi(close, 14)

        # MACD — compute full series for proper signal line
        macd_series = []
        for i in range(26, n):
            e12 = self._ema(close[:i + 1], 12)
            e26 = self._ema(close[:i + 1], 26)
            macd_series.append(e12 - e26)
        macd_line = macd_series[-1] if macd_series else 0
        macd_signal = self._ema(np.array(macd_series), 9) if len(macd_series) >= 9 else macd_line
        macd_hist = macd_line - macd_signal

        # ATR (14)
        atr = self._atr(high, low, close, 14)
        atr_pct = (atr / close[-1] * 100) if close[-1] > 0 else 0

        # ADX (14)
        adx = self._adx(high, low, close, 14)

        # Bollinger Bands
        bb_mid = sma20
        bb_std = np.std(close[-20:]) if n >= 20 else 0
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_position = (close[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

        # Returns
        ret_1d = (close[-1] / close[-2] - 1) * 100 if n >= 2 else 0
        ret_5d = (close[-1] / close[-6] - 1) * 100 if n >= 6 else 0
        ret_20d = (close[-1] / close[-21] - 1) * 100 if n >= 21 else 0

        return {
            "rsi": round(rsi, 1),
            "macd_line": round(macd_line, 4),
            "macd_hist": round(macd_hist, 4),
            "adx": round(adx, 1),
            "atr_pct": round(atr_pct, 2),
            "sma20": round(sma20, 4),
            "sma50": round(sma50, 4),
            "price_vs_sma20": round((close[-1] / sma20 - 1) * 100, 2),
            "price_vs_sma50": round((close[-1] / sma50 - 1) * 100, 2),
            "bb_position": round(bb_position, 2),
            "ret_1d": round(ret_1d, 2),
            "ret_5d": round(ret_5d, 2),
            "ret_20d": round(ret_20d, 2),
            "last_close": round(float(close[-1]), 4),
        }

    def _classify_trend(self, signals: dict) -> str:
        if signals["price_vs_sma20"] > 1 and signals["price_vs_sma50"] > 2:
            return "bullish"
        elif signals["price_vs_sma20"] < -1 and signals["price_vs_sma50"] < -2:
            return "bearish"
        return "neutral"

    def _classify_momentum(self, signals: dict) -> str:
        if signals["rsi"] > 60 and signals["macd_hist"] > 0:
            return "strong_bullish"
        elif signals["rsi"] < 40 and signals["macd_hist"] < 0:
            return "strong_bearish"
        return "neutral"

    def _classify_volatility(self, signals: dict) -> str:
        if signals["atr_pct"] > 3:
            return "high"
        elif signals["atr_pct"] < 1:
            return "low"
        return "normal"

    @staticmethod
    def _rsi(close: np.ndarray, period: int = 14) -> float:
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        if len(data) < period:
            return float(data[-1])
        multiplier = 2 / (period + 1)
        ema = float(data[-period])
        for price in data[-(period - 1):]:
            ema = (float(price) - ema) * multiplier + ema
        return ema

    @staticmethod
    def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        if len(close) < period + 1:
            return 0.0
        tr_values = []
        for i in range(-period, 0):
            tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            tr_values.append(tr)
        return float(np.mean(tr_values))

    @staticmethod
    def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        if len(close) < period * 2:
            return 0.0
        plus_dm = []
        minus_dm = []
        for i in range(-period * 2, 0):
            up = high[i] - high[i - 1]
            down = low[i - 1] - low[i]
            plus_dm.append(up if up > down and up > 0 else 0)
            minus_dm.append(down if down > up and down > 0 else 0)

        atr_val = 0
        for i in range(-period * 2, 0):
            tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            atr_val += tr
        atr_val /= period * 2

        if atr_val == 0:
            return 0.0

        plus_di = (np.mean(plus_dm[-period:]) / atr_val) * 100
        minus_di = (np.mean(minus_dm[-period:]) / atr_val) * 100
        dx = abs(plus_di - minus_di) / max(plus_di + minus_di, 1) * 100
        return float(dx)
