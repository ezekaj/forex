"""
Feature engineering service for ML model training.

Generates 50+ features from OHLCV data and technical indicators.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict
from ..indicators import (
    SMA, EMA, MACD, ADX,
    RSI, Stochastic, CCI, ROC,
    ATR, BollingerBands, StandardDeviation,
    OBV, VolumeSMA, MFI
)


class FeatureEngineer:
    """Generate ML features from OHLCV data."""

    def __init__(self):
        """Initialize feature engineer with indicators."""
        # Trend indicators
        self.sma_20 = SMA(period=20)
        self.sma_50 = SMA(period=50)
        self.sma_200 = SMA(period=200)
        self.ema_12 = EMA(period=12)
        self.ema_26 = EMA(period=26)
        self.macd = MACD(fast=12, slow=26, signal=9)
        self.adx = ADX(period=14)

        # Momentum indicators
        self.rsi = RSI(period=14)
        self.stoch = Stochastic(k_period=14, d_period=3)
        self.cci = CCI(period=20)
        self.roc = ROC(period=12)

        # Volatility indicators
        self.atr = ATR(period=14)
        self.bb = BollingerBands(period=20, std_dev=2.0)
        self.std_dev = StandardDeviation(period=20)

        # Volume indicators
        self.obv = OBV()
        self.vol_sma = VolumeSMA(period=20)
        self.mfi = MFI(period=14)

    def generate_features(
        self,
        df: pd.DataFrame,
        include_multi_timeframe: bool = False
    ) -> pd.DataFrame:
        """
        Generate comprehensive feature set from OHLCV data.

        Args:
            df: DataFrame with OHLCV data
            include_multi_timeframe: Whether to add multi-timeframe features

        Returns:
            DataFrame with original data + engineered features
        """
        if df.empty or len(df) < 200:  # Need enough data for 200-period SMA
            raise ValueError("Insufficient data for feature engineering. Need at least 200 bars.")

        # Make a copy to avoid modifying original
        features_df = df.copy()

        # Calculate all indicators
        features_df = self._add_trend_indicators(features_df)
        features_df = self._add_momentum_indicators(features_df)
        features_df = self._add_volatility_indicators(features_df)
        features_df = self._add_volume_indicators(features_df)

        # Derived features
        features_df = self._add_price_action_features(features_df)
        features_df = self._add_crossover_features(features_df)
        features_df = self._add_divergence_features(features_df)
        features_df = self._add_volatility_features(features_df)
        features_df = self._add_time_features(features_df)

        # Drop rows with NaN (from indicators with lookback)
        features_df = features_df.dropna()

        return features_df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicator features."""
        # Moving averages
        df['sma_20'] = self.sma_20.calculate(df)
        df['sma_50'] = self.sma_50.calculate(df)
        df['sma_200'] = self.sma_200.calculate(df)
        df['ema_12'] = self.ema_12.calculate(df)
        df['ema_26'] = self.ema_26.calculate(df)

        # MACD
        macd_df = self.macd.calculate(df)
        df['macd'] = macd_df['macd']
        df['macd_signal'] = macd_df['macd_signal']
        df['macd_diff'] = macd_df['macd_diff']

        # ADX
        adx_df = self.adx.calculate(df)
        df['adx'] = adx_df['adx']
        df['adx_pos'] = adx_df['adx_pos']
        df['adx_neg'] = adx_df['adx_neg']

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicator features."""
        df['rsi'] = self.rsi.calculate(df)
        df['cci'] = self.cci.calculate(df)
        df['roc'] = self.roc.calculate(df)

        # Stochastic
        stoch_df = self.stoch.calculate(df)
        df['stoch_k'] = stoch_df['stoch_k']
        df['stoch_d'] = stoch_df['stoch_d']

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicator features."""
        df['atr'] = self.atr.calculate(df)
        df['std_dev'] = self.std_dev.calculate(df)

        # Bollinger Bands
        bb_df = self.bb.calculate(df)
        df['bb_upper'] = bb_df['bb_upper']
        df['bb_middle'] = bb_df['bb_middle']
        df['bb_lower'] = bb_df['bb_lower']
        df['bb_width'] = bb_df['bb_width']
        df['bb_pct'] = bb_df['bb_pct']

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicator features."""
        df['obv'] = self.obv.calculate(df)
        df['vol_sma'] = self.vol_sma.calculate(df)
        df['mfi'] = self.mfi.calculate(df)

        return df

    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action-derived features."""
        # Price vs moving averages
        df['price_vs_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['price_vs_sma_50'] = (df['close'] - df['sma_50']) / df['sma_50']
        df['price_vs_sma_200'] = (df['close'] - df['sma_200']) / df['sma_200']

        # Returns (percentage change)
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)

        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_sma'] = df['hl_range'].rolling(window=20).mean()

        # Close position in daily range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        # Candle body size
        df['body_size'] = abs(df['close'] - df['open']) / df['close']

        # Upper/lower wicks
        df['upper_wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['lower_wick'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']

        return df

    def _add_crossover_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add crossover-based features."""
        # SMA crossovers
        df['sma_20_50_cross'] = np.where(
            df['sma_20'] > df['sma_50'], 1,
            np.where(df['sma_20'] < df['sma_50'], -1, 0)
        )

        df['sma_50_200_cross'] = np.where(
            df['sma_50'] > df['sma_200'], 1,
            np.where(df['sma_50'] < df['sma_200'], -1, 0)
        )

        # Price vs EMA
        df['price_above_ema_12'] = (df['close'] > df['ema_12']).astype(int)
        df['price_above_ema_26'] = (df['close'] > df['ema_26']).astype(int)

        # MACD crossover
        df['macd_cross'] = np.where(
            df['macd'] > df['macd_signal'], 1,
            np.where(df['macd'] < df['macd_signal'], -1, 0)
        )

        # Stochastic crossover
        df['stoch_cross'] = np.where(
            df['stoch_k'] > df['stoch_d'], 1,
            np.where(df['stoch_k'] < df['stoch_d'], -1, 0)
        )

        return df

    def _add_divergence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add divergence-based features."""
        # RSI overbought/oversold
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)

        # Stochastic overbought/oversold
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)

        # MFI overbought/oversold
        df['mfi_overbought'] = (df['mfi'] > 80).astype(int)
        df['mfi_oversold'] = (df['mfi'] < 20).astype(int)

        # CCI extreme levels
        df['cci_overbought'] = (df['cci'] > 100).astype(int)
        df['cci_oversold'] = (df['cci'] < -100).astype(int)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-derived features."""
        # ATR as percentage of price
        df['atr_pct'] = df['atr'] / df['close']

        # Volatility regime (low/medium/high)
        df['vol_regime'] = pd.cut(
            df['atr_pct'],
            bins=[0, df['atr_pct'].quantile(0.33), df['atr_pct'].quantile(0.67), float('inf')],
            labels=[0, 1, 2]
        )
        df['vol_regime'] = df['vol_regime'].fillna(1).astype(int)  # Fill NaN with medium (1)

        # Bollinger Band squeeze (low volatility)
        bb_width_sma = df['bb_width'].rolling(window=20).mean()
        df['bb_squeeze'] = (df['bb_width'] < bb_width_sma * 0.8).astype(int)

        # Price near bands
        df['near_bb_upper'] = (df['close'] > df['bb_upper'] * 0.98).astype(int)
        df['near_bb_lower'] = (df['close'] < df['bb_lower'] * 1.02).astype(int)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['day_of_month'] = pd.to_datetime(df['timestamp']).dt.day

            # Trading session indicators
            df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['american_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)

        return df

    def get_feature_names(self) -> List[str]:
        """
        Get list of all generated feature names.

        Returns:
            List of feature column names
        """
        return [
            # Original OHLCV
            'open', 'high', 'low', 'close', 'volume',

            # Trend indicators
            'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_diff',
            'adx', 'adx_pos', 'adx_neg',

            # Momentum indicators
            'rsi', 'cci', 'roc', 'stoch_k', 'stoch_d',

            # Volatility indicators
            'atr', 'std_dev', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct',

            # Volume indicators
            'obv', 'vol_sma', 'mfi',

            # Price action
            'price_vs_sma_20', 'price_vs_sma_50', 'price_vs_sma_200',
            'return_1', 'return_5', 'return_10',
            'hl_range', 'hl_range_sma', 'close_position',
            'body_size', 'upper_wick', 'lower_wick',

            # Crossovers
            'sma_20_50_cross', 'sma_50_200_cross',
            'price_above_ema_12', 'price_above_ema_26',
            'macd_cross', 'stoch_cross',

            # Divergences
            'rsi_overbought', 'rsi_oversold',
            'stoch_overbought', 'stoch_oversold',
            'mfi_overbought', 'mfi_oversold',
            'cci_overbought', 'cci_oversold',

            # Volatility features
            'atr_pct', 'vol_regime', 'bb_squeeze',
            'near_bb_upper', 'near_bb_lower',

            # Time features
            'hour', 'day_of_week', 'day_of_month',
            'asian_session', 'european_session', 'american_session'
        ]

    def get_feature_count(self) -> int:
        """Get total number of features generated."""
        return len(self.get_feature_names())
