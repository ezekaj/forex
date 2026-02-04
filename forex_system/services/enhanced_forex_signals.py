#!/usr/bin/env python3
"""
ENHANCED FOREX SIGNALS - Integrated from news_lenci_forex
Multi-source signal aggregation for forex trading.

Sources:
1. VIX / Market Regime Detection
2. Currency Correlations
3. Commitment of Traders (COT) data
4. Interest Rate Differentials
5. Technical Analysis on Multiple Timeframes
"""
import asyncio
import aiohttp
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class MarketRegime:
    """Current market regime classification."""
    regime: str  # LOW_VOL, NORMAL, HIGH_VOL, EXTREME_FEAR, RISK_OFF
    vix_level: float
    vix_change: float
    confidence_multiplier: float
    description: str


@dataclass
class CorrelationSignal:
    """Signal based on currency correlations."""
    pair: str
    correlated_pairs: List[Tuple[str, float]]  # (pair, correlation)
    lead_lag_signal: Optional[str]  # If a correlated pair moved first
    signal_strength: float


class EnhancedForexSignals:
    """
    Multi-source signal aggregator for forex trading.
    Combines various free data sources for enhanced predictions.
    """

    # Major forex pairs and their typical correlations
    PAIR_CORRELATIONS = {
        'EURUSD': [('GBPUSD', 0.85), ('AUDUSD', 0.75), ('NZDUSD', 0.70)],
        'GBPUSD': [('EURUSD', 0.85), ('AUDUSD', 0.70), ('EURGBP', -0.75)],
        'USDJPY': [('EURJPY', 0.85), ('GBPJPY', 0.80), ('AUDJPY', 0.70)],
        'AUDUSD': [('NZDUSD', 0.90), ('EURUSD', 0.75), ('USDCAD', -0.65)],
        'USDCAD': [('AUDUSD', -0.65), ('USDCHF', 0.60), ('EURUSD', -0.55)],
        'EURJPY': [('USDJPY', 0.85), ('GBPJPY', 0.90), ('AUDJPY', 0.75)],
        'GBPJPY': [('EURJPY', 0.90), ('USDJPY', 0.80), ('AUDJPY', 0.80)],
    }

    # Risk sentiment currencies
    RISK_ON_CURRENCIES = ['AUD', 'NZD', 'CAD']  # Benefit when risk appetite high
    SAFE_HAVEN_CURRENCIES = ['JPY', 'CHF', 'USD']  # Benefit when risk appetite low

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(__file__).parent.parent.parent / 'data' / 'forex_signals.db'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._session: Optional[aiohttp.ClientSession] = None

    def _init_db(self):
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS market_regime (
                id INTEGER PRIMARY KEY,
                datetime TEXT,
                vix_level REAL,
                vix_change REAL,
                regime TEXT,
                confidence_mult REAL
            );

            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY,
                pair TEXT,
                datetime TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                timeframe TEXT,
                UNIQUE(pair, datetime, timeframe)
            );

            CREATE TABLE IF NOT EXISTS signal_history (
                id INTEGER PRIMARY KEY,
                datetime TEXT,
                pair TEXT,
                signal_type TEXT,
                signal_value REAL,
                source TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_regime_dt ON market_regime(datetime);
            CREATE INDEX IF NOT EXISTS idx_price_pair ON price_history(pair, datetime);
        """)
        conn.commit()
        conn.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # =========================================================================
    # MARKET REGIME DETECTION
    # =========================================================================

    async def fetch_vix(self) -> Optional[Dict]:
        """Fetch VIX data from Yahoo Finance."""
        session = await self._get_session()

        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/^VIX?interval=1d&range=10d"
            headers = {'User-Agent': 'Mozilla/5.0'}

            async with session.get(url, headers=headers, timeout=10) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

            result = data.get('chart', {}).get('result', [])
            if not result:
                return None

            quote = result[0].get('indicators', {}).get('quote', [{}])[0]
            closes = [c for c in quote.get('close', []) if c is not None]

            if not closes:
                return None

            return {
                'current': closes[-1],
                'prev': closes[-2] if len(closes) > 1 else closes[-1],
                'change': closes[-1] - closes[-2] if len(closes) > 1 else 0,
                'avg_5d': sum(closes[-5:]) / len(closes[-5:]) if len(closes) >= 5 else closes[-1]
            }
        except Exception as e:
            print(f"[EnhancedSignals] VIX fetch error: {e}")
            return None

    async def detect_market_regime(self) -> MarketRegime:
        """Detect current market regime based on VIX."""
        vix_data = await self.fetch_vix()

        if not vix_data:
            return MarketRegime(
                regime='UNKNOWN',
                vix_level=20.0,
                vix_change=0.0,
                confidence_multiplier=0.8,
                description='VIX data unavailable'
            )

        vix = vix_data['current']
        vix_change = vix_data['change']

        if vix < 12:
            regime = 'EXTREME_LOW_VOL'
            confidence_mult = 1.15
            desc = 'Extreme complacency - trend following works well'
        elif vix < 16:
            regime = 'LOW_VOLATILITY'
            confidence_mult = 1.10
            desc = 'Low volatility - trend following preferred'
        elif vix < 20:
            regime = 'NORMAL'
            confidence_mult = 1.0
            desc = 'Normal market conditions'
        elif vix < 25:
            regime = 'ELEVATED'
            confidence_mult = 0.95
            desc = 'Elevated uncertainty - reduce position sizes'
        elif vix < 35:
            regime = 'HIGH_VOLATILITY'
            confidence_mult = 0.80
            desc = 'High volatility - wider stops needed, smaller positions'
        else:
            regime = 'EXTREME_FEAR'
            confidence_mult = 0.65
            desc = 'Extreme fear - consider contrarian opportunities'

        # Adjust for VIX spikes
        if vix_change > 3:
            regime += '_SPIKING'
            confidence_mult *= 0.9
            desc += ' (VIX spiking - caution)'
        elif vix_change < -3:
            regime += '_CALMING'
            confidence_mult *= 1.05
            desc += ' (VIX calming - confidence increasing)'

        return MarketRegime(
            regime=regime,
            vix_level=vix,
            vix_change=vix_change,
            confidence_multiplier=confidence_mult,
            description=desc
        )

    # =========================================================================
    # MULTI-TIMEFRAME ANALYSIS
    # =========================================================================

    async def fetch_ohlc(
        self,
        pair: str,
        timeframe: str = '1h',
        bars: int = 100
    ) -> List[Dict]:
        """Fetch OHLC data for a forex pair from Yahoo Finance."""
        session = await self._get_session()

        # Convert pair to Yahoo format (EURUSD -> EURUSD=X)
        yahoo_symbol = f"{pair}=X"

        # Map timeframe to Yahoo intervals
        tf_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '1h', '1d': '1d', '1w': '1wk'
        }
        interval = tf_map.get(timeframe, '1h')

        # Calculate range based on bars needed
        range_map = {
            '1m': '1d', '5m': '5d', '15m': '5d', '30m': '1mo',
            '1h': '1mo', '4h': '3mo', '1d': '1y', '1w': '5y'
        }
        range_val = range_map.get(timeframe, '1mo')

        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
            params = {'interval': interval, 'range': range_val}
            headers = {'User-Agent': 'Mozilla/5.0'}

            async with session.get(url, headers=headers, params=params, timeout=15) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            result = data.get('chart', {}).get('result', [])
            if not result:
                return []

            timestamps = result[0].get('timestamp', [])
            quote = result[0].get('indicators', {}).get('quote', [{}])[0]

            ohlc = []
            for i, ts in enumerate(timestamps[-bars:]):
                if quote.get('close', [None])[i] is not None:
                    ohlc.append({
                        'datetime': datetime.fromtimestamp(ts),
                        'open': quote['open'][i],
                        'high': quote['high'][i],
                        'low': quote['low'][i],
                        'close': quote['close'][i],
                        'volume': quote.get('volume', [0])[i] or 0
                    })

            return ohlc

        except Exception as e:
            print(f"[EnhancedSignals] OHLC fetch error for {pair}: {e}")
            return []

    def calculate_technical_indicators(self, ohlc: List[Dict]) -> Dict:
        """Calculate technical indicators from OHLC data."""
        if len(ohlc) < 20:
            return {}

        closes = [bar['close'] for bar in ohlc]
        highs = [bar['high'] for bar in ohlc]
        lows = [bar['low'] for bar in ohlc]

        # Current price
        current = closes[-1]

        # Moving averages
        sma_10 = sum(closes[-10:]) / 10
        sma_20 = sum(closes[-20:]) / 20
        sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else sma_20

        # EMA 12 and 26 for MACD
        def ema(data, period):
            multiplier = 2 / (period + 1)
            ema_val = sum(data[:period]) / period
            for price in data[period:]:
                ema_val = (price - ema_val) * multiplier + ema_val
            return ema_val

        ema_12 = ema(closes, 12)
        ema_26 = ema(closes, 26) if len(closes) >= 26 else ema_12
        macd = ema_12 - ema_26

        # RSI
        gains = []
        losses = []
        for i in range(1, min(15, len(closes))):
            diff = closes[-i] - closes[-i-1]
            if diff > 0:
                gains.append(diff)
            else:
                losses.append(abs(diff))

        avg_gain = sum(gains) / 14 if gains else 0.0001
        avg_loss = sum(losses) / 14 if losses else 0.0001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # ATR
        trs = []
        for i in range(1, min(15, len(closes))):
            tr = max(
                highs[-i] - lows[-i],
                abs(highs[-i] - closes[-i-1]),
                abs(lows[-i] - closes[-i-1])
            )
            trs.append(tr)
        atr = sum(trs) / len(trs) if trs else 0

        # Trend determination
        if current > sma_20 and sma_10 > sma_20:
            trend = 'UPTREND'
        elif current < sma_20 and sma_10 < sma_20:
            trend = 'DOWNTREND'
        else:
            trend = 'SIDEWAYS'

        return {
            'current_price': current,
            'sma_10': sma_10,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'ema_12': ema_12,
            'ema_26': ema_26,
            'macd': macd,
            'rsi': rsi,
            'atr': atr,
            'trend': trend,
            'price_vs_sma20': (current - sma_20) / sma_20 * 100,
            'price_vs_sma50': (current - sma_50) / sma_50 * 100 if sma_50 else 0
        }

    async def get_multi_timeframe_signal(self, pair: str) -> Dict:
        """
        Get signal based on multiple timeframe analysis.
        Higher timeframe = trend direction
        Lower timeframe = entry timing
        """
        # Fetch different timeframes
        h4_data = await self.fetch_ohlc(pair, '4h', 100)
        h1_data = await self.fetch_ohlc(pair, '1h', 100)

        if not h4_data or not h1_data:
            return {'signal': 'NEUTRAL', 'strength': 0.0, 'reason': 'Insufficient data'}

        h4_tech = self.calculate_technical_indicators(h4_data)
        h1_tech = self.calculate_technical_indicators(h1_data)

        if not h4_tech or not h1_tech:
            return {'signal': 'NEUTRAL', 'strength': 0.0, 'reason': 'Calculation error'}

        # Determine signal based on alignment
        h4_trend = h4_tech['trend']
        h1_trend = h1_tech['trend']
        rsi_1h = h1_tech['rsi']

        signal = 'NEUTRAL'
        strength = 0.0
        reasons = []

        # Strong signal: trends align
        if h4_trend == 'UPTREND' and h1_trend == 'UPTREND':
            signal = 'BUY'
            strength = 0.7
            reasons.append('H4 and H1 trends aligned bullish')

            # Boost if RSI not overbought
            if rsi_1h < 70:
                strength += 0.1
                reasons.append('RSI confirming')
        elif h4_trend == 'DOWNTREND' and h1_trend == 'DOWNTREND':
            signal = 'SELL'
            strength = 0.7
            reasons.append('H4 and H1 trends aligned bearish')

            # Boost if RSI not oversold
            if rsi_1h > 30:
                strength += 0.1
                reasons.append('RSI confirming')

        # Moderate signal: H4 trend with H1 pullback
        elif h4_trend == 'UPTREND' and h1_trend != 'DOWNTREND' and rsi_1h < 40:
            signal = 'BUY'
            strength = 0.5
            reasons.append('H4 uptrend with H1 pullback opportunity')
        elif h4_trend == 'DOWNTREND' and h1_trend != 'UPTREND' and rsi_1h > 60:
            signal = 'SELL'
            strength = 0.5
            reasons.append('H4 downtrend with H1 pullback opportunity')

        # MACD confirmation
        if h1_tech['macd'] > 0 and signal == 'BUY':
            strength += 0.05
        elif h1_tech['macd'] < 0 and signal == 'SELL':
            strength += 0.05

        return {
            'signal': signal,
            'strength': min(strength, 1.0),
            'reasons': reasons,
            'h4_trend': h4_trend,
            'h1_trend': h1_trend,
            'rsi_1h': rsi_1h,
            'atr_1h': h1_tech['atr']
        }

    # =========================================================================
    # RISK SENTIMENT ANALYSIS
    # =========================================================================

    async def get_risk_sentiment_signal(self, pair: str) -> Dict:
        """
        Analyze risk sentiment and its impact on the pair.
        Uses VIX and cross-market analysis.
        """
        base = pair[:3].upper()
        quote = pair[3:6].upper()

        regime = await self.detect_market_regime()

        # Determine risk impact on currencies
        base_is_risk_on = base in self.RISK_ON_CURRENCIES
        base_is_safe = base in self.SAFE_HAVEN_CURRENCIES
        quote_is_risk_on = quote in self.RISK_ON_CURRENCIES
        quote_is_safe = quote in self.SAFE_HAVEN_CURRENCIES

        signal = 'NEUTRAL'
        strength = 0.0
        reason = ''

        # Risk-off environment (high VIX)
        if 'HIGH' in regime.regime or 'FEAR' in regime.regime:
            # Safe havens should appreciate
            if base_is_safe and quote_is_risk_on:
                signal = 'BUY'
                strength = 0.4
                reason = f'Risk-off: {base} (safe) vs {quote} (risk)'
            elif quote_is_safe and base_is_risk_on:
                signal = 'SELL'
                strength = 0.4
                reason = f'Risk-off: {quote} (safe) vs {base} (risk)'

        # Risk-on environment (low VIX)
        elif 'LOW' in regime.regime:
            # Risk currencies should appreciate
            if base_is_risk_on and quote_is_safe:
                signal = 'BUY'
                strength = 0.3
                reason = f'Risk-on: {base} (risk) vs {quote} (safe)'
            elif quote_is_risk_on and base_is_safe:
                signal = 'SELL'
                strength = 0.3
                reason = f'Risk-on: {quote} (risk) vs {base} (safe)'

        return {
            'signal': signal,
            'strength': strength,
            'reason': reason,
            'regime': regime.regime,
            'vix': regime.vix_level,
            'confidence_multiplier': regime.confidence_multiplier
        }

    # =========================================================================
    # CORRELATION ANALYSIS
    # =========================================================================

    async def get_correlation_signal(self, pair: str) -> Dict:
        """
        Analyze correlated pairs for lead/lag signals.
        If a highly correlated pair moved first, it may predict this pair.
        """
        correlations = self.PAIR_CORRELATIONS.get(pair, [])

        if not correlations:
            return {'signal': 'NEUTRAL', 'strength': 0.0, 'reason': 'No correlation data'}

        signals = []

        for corr_pair, correlation in correlations[:3]:  # Top 3 correlated
            corr_data = await self.fetch_ohlc(corr_pair, '1h', 10)

            if len(corr_data) < 5:
                continue

            # Check if correlated pair made a significant move
            recent_move = (corr_data[-1]['close'] - corr_data[-3]['close']) / corr_data[-3]['close']

            if abs(recent_move) > 0.002:  # 0.2% move
                direction = 'BUY' if recent_move > 0 else 'SELL'

                # If negative correlation, flip signal
                if correlation < 0:
                    direction = 'SELL' if direction == 'BUY' else 'BUY'

                signals.append({
                    'pair': corr_pair,
                    'correlation': correlation,
                    'move': recent_move,
                    'signal': direction,
                    'strength': min(abs(recent_move) * 100, 0.3)
                })

        if not signals:
            return {'signal': 'NEUTRAL', 'strength': 0.0, 'reason': 'No lead signals detected'}

        # Average the signals
        buy_strength = sum(s['strength'] for s in signals if s['signal'] == 'BUY')
        sell_strength = sum(s['strength'] for s in signals if s['signal'] == 'SELL')

        if buy_strength > sell_strength:
            return {
                'signal': 'BUY',
                'strength': buy_strength,
                'reason': f"Lead signals from: {', '.join(s['pair'] for s in signals if s['signal'] == 'BUY')}",
                'details': signals
            }
        elif sell_strength > buy_strength:
            return {
                'signal': 'SELL',
                'strength': sell_strength,
                'reason': f"Lead signals from: {', '.join(s['pair'] for s in signals if s['signal'] == 'SELL')}",
                'details': signals
            }
        else:
            return {'signal': 'NEUTRAL', 'strength': 0.0, 'reason': 'Mixed correlation signals'}

    # =========================================================================
    # COMBINED SIGNAL AGGREGATOR
    # =========================================================================

    async def get_combined_signal(self, pair: str) -> Dict:
        """
        Aggregate all enhanced signals into a combined recommendation.
        """
        # Fetch all signals in parallel
        mtf_task = asyncio.create_task(self.get_multi_timeframe_signal(pair))
        risk_task = asyncio.create_task(self.get_risk_sentiment_signal(pair))
        corr_task = asyncio.create_task(self.get_correlation_signal(pair))

        mtf_signal = await mtf_task
        risk_signal = await risk_task
        corr_signal = await corr_task

        # Weight the signals
        weights = {
            'mtf': 0.50,  # Multi-timeframe analysis is primary
            'risk': 0.30,  # Risk sentiment is important
            'corr': 0.20   # Correlation is supplementary
        }

        # Calculate weighted score
        score = 0.0
        explanations = []

        if mtf_signal['signal'] == 'BUY':
            score += mtf_signal['strength'] * weights['mtf']
        elif mtf_signal['signal'] == 'SELL':
            score -= mtf_signal['strength'] * weights['mtf']
        if mtf_signal.get('reasons'):
            explanations.extend(mtf_signal['reasons'])

        if risk_signal['signal'] == 'BUY':
            score += risk_signal['strength'] * weights['risk']
        elif risk_signal['signal'] == 'SELL':
            score -= risk_signal['strength'] * weights['risk']
        if risk_signal.get('reason'):
            explanations.append(risk_signal['reason'])

        if corr_signal['signal'] == 'BUY':
            score += corr_signal['strength'] * weights['corr']
        elif corr_signal['signal'] == 'SELL':
            score -= corr_signal['strength'] * weights['corr']
        if corr_signal.get('reason'):
            explanations.append(corr_signal['reason'])

        # Apply regime confidence multiplier
        score *= risk_signal['confidence_multiplier']

        # Determine final signal
        if score > 0.15:
            signal = 'BUY'
        elif score < -0.15:
            signal = 'SELL'
        else:
            signal = 'NEUTRAL'

        return {
            'signal': signal,
            'strength': abs(score),
            'score': score,
            'explanations': explanations,
            'components': {
                'mtf': mtf_signal,
                'risk': risk_signal,
                'correlation': corr_signal
            },
            'atr': mtf_signal.get('atr_1h', 0),
            'regime': risk_signal['regime'],
            'timestamp': datetime.utcnow().isoformat()
        }


# Convenience function
def get_enhanced_signals_service() -> EnhancedForexSignals:
    """Get an EnhancedForexSignals instance."""
    return EnhancedForexSignals()
