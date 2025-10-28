"""
Test technical indicators module.

Comprehensive tests for all indicator calculations, validation, and edge cases.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from forex_system.indicators import (
    # Trend
    SMA, EMA, MACD, ADX,
    # Momentum
    RSI, Stochastic, CCI, ROC,
    # Volatility
    ATR, BollingerBands, StandardDeviation, KeltnerChannel,
    # Volume
    OBV, VolumeSMA, MFI, VWAP, ADI, CMF
)


# ==================== Fixtures ====================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')

    # Generate realistic price movement
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.randn(100) * 0.1,
        'high': close_prices + abs(np.random.randn(100) * 0.3),
        'low': close_prices - abs(np.random.randn(100) * 0.3),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 100)
    })

    return df


@pytest.fixture
def minimal_ohlcv_data():
    """Generate minimal OHLCV data for edge case testing."""
    df = pd.DataFrame({
        'open': [100.0, 101.0, 99.0],
        'high': [102.0, 103.0, 100.0],
        'low': [99.0, 100.0, 98.0],
        'close': [101.0, 102.0, 99.0],
        'volume': [1000, 1500, 1200]
    })
    return df


# ==================== Trend Indicators ====================

class TestSMA:
    """Test Simple Moving Average indicator."""

    def test_sma_calculation(self, sample_ohlcv_data):
        """Test SMA calculates correctly."""
        sma = SMA(period=20)
        result = sma.calculate(sample_ohlcv_data)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        assert not result.iloc[-1] == np.nan  # Last value should exist

    def test_sma_custom_period(self, sample_ohlcv_data):
        """Test SMA with custom period."""
        sma_10 = SMA(period=10)
        sma_50 = SMA(period=50)

        result_10 = sma_10.calculate(sample_ohlcv_data)
        result_50 = sma_50.calculate(sample_ohlcv_data)

        # 10-period SMA should respond faster than 50-period
        assert not result_10.equals(result_50)

    def test_sma_insufficient_data(self, minimal_ohlcv_data):
        """Test SMA with insufficient data."""
        sma = SMA(period=10)

        with pytest.raises(ValueError, match="Insufficient data"):
            sma.calculate(minimal_ohlcv_data)

    def test_sma_invalid_period(self):
        """Test SMA with invalid period."""
        with pytest.raises(ValueError):
            SMA(period=0)

        with pytest.raises(ValueError):
            SMA(period=-5)


class TestEMA:
    """Test Exponential Moving Average indicator."""

    def test_ema_calculation(self, sample_ohlcv_data):
        """Test EMA calculates correctly."""
        ema = EMA(period=20)
        result = ema.calculate(sample_ohlcv_data)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)

    def test_ema_vs_sma(self, sample_ohlcv_data):
        """Test EMA differs from SMA (more responsive)."""
        ema = EMA(period=20)
        sma = SMA(period=20)

        ema_result = ema.calculate(sample_ohlcv_data)
        sma_result = sma.calculate(sample_ohlcv_data)

        # EMA and SMA should differ
        assert not ema_result.equals(sma_result)


class TestMACD:
    """Test MACD indicator."""

    def test_macd_calculation(self, sample_ohlcv_data):
        """Test MACD calculates all components."""
        macd = MACD(fast=12, slow=26, signal=9)
        result = macd.calculate(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)
        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_diff' in result.columns

    def test_macd_invalid_parameters(self):
        """Test MACD with invalid fast/slow periods."""
        with pytest.raises(ValueError, match="Fast period must be less than slow"):
            MACD(fast=26, slow=12, signal=9)


class TestADX:
    """Test Average Directional Index."""

    def test_adx_calculation(self, sample_ohlcv_data):
        """Test ADX calculates all components."""
        adx = ADX(period=14)
        result = adx.calculate(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)
        assert 'adx' in result.columns
        assert 'adx_pos' in result.columns
        assert 'adx_neg' in result.columns


# ==================== Momentum Indicators ====================

class TestRSI:
    """Test Relative Strength Index."""

    def test_rsi_calculation(self, sample_ohlcv_data):
        """Test RSI calculates correctly."""
        rsi = RSI(period=14)
        result = rsi.calculate(sample_ohlcv_data)

        assert isinstance(result, pd.Series)
        # RSI should be between 0 and 100
        valid_values = result.dropna()
        assert valid_values.min() >= 0
        assert valid_values.max() <= 100

    def test_rsi_insufficient_data(self, minimal_ohlcv_data):
        """Test RSI with insufficient data."""
        rsi = RSI(period=14)

        with pytest.raises(ValueError, match="Insufficient data"):
            rsi.calculate(minimal_ohlcv_data)

    def test_rsi_custom_period(self, sample_ohlcv_data):
        """Test RSI with custom period."""
        rsi_9 = RSI(period=9)
        rsi_21 = RSI(period=21)

        result_9 = rsi_9.calculate(sample_ohlcv_data)
        result_21 = rsi_21.calculate(sample_ohlcv_data)

        # Different periods should give different results
        assert not result_9.equals(result_21)


class TestStochastic:
    """Test Stochastic Oscillator."""

    def test_stochastic_calculation(self, sample_ohlcv_data):
        """Test Stochastic calculates %K and %D."""
        stoch = Stochastic(k_period=14, d_period=3)
        result = stoch.calculate(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)
        assert 'stoch_k' in result.columns
        assert 'stoch_d' in result.columns

        # Values should be between 0 and 100
        valid_k = result['stoch_k'].dropna()
        assert valid_k.min() >= 0
        assert valid_k.max() <= 100


class TestCCI:
    """Test Commodity Channel Index."""

    def test_cci_calculation(self, sample_ohlcv_data):
        """Test CCI calculates correctly."""
        cci = CCI(period=20)
        result = cci.calculate(sample_ohlcv_data)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)


class TestROC:
    """Test Rate of Change."""

    def test_roc_calculation(self, sample_ohlcv_data):
        """Test ROC calculates correctly."""
        roc = ROC(period=12)
        result = roc.calculate(sample_ohlcv_data)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)


# ==================== Volatility Indicators ====================

class TestATR:
    """Test Average True Range."""

    def test_atr_calculation(self, sample_ohlcv_data):
        """Test ATR calculates correctly."""
        atr = ATR(period=14)
        result = atr.calculate(sample_ohlcv_data)

        assert isinstance(result, pd.Series)
        # ATR should be positive
        valid_values = result.dropna()
        assert (valid_values >= 0).all()

    def test_atr_custom_period(self, sample_ohlcv_data):
        """Test ATR with custom period."""
        atr_10 = ATR(period=10)
        atr_20 = ATR(period=20)

        result_10 = atr_10.calculate(sample_ohlcv_data)
        result_20 = atr_20.calculate(sample_ohlcv_data)

        # Different periods should give different results
        assert not result_10.equals(result_20)


class TestBollingerBands:
    """Test Bollinger Bands."""

    def test_bollinger_calculation(self, sample_ohlcv_data):
        """Test Bollinger Bands calculates all components."""
        bb = BollingerBands(period=20, std_dev=2.0)
        result = bb.calculate(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)
        assert 'bb_upper' in result.columns
        assert 'bb_middle' in result.columns
        assert 'bb_lower' in result.columns
        assert 'bb_width' in result.columns
        assert 'bb_pct' in result.columns

    def test_bollinger_band_order(self, sample_ohlcv_data):
        """Test that upper > middle > lower."""
        bb = BollingerBands(period=20, std_dev=2.0)
        result = bb.calculate(sample_ohlcv_data)

        # Remove NaN values
        valid_data = result.dropna()

        # Upper should be > middle
        assert (valid_data['bb_upper'] >= valid_data['bb_middle']).all()
        # Middle should be > lower
        assert (valid_data['bb_middle'] >= valid_data['bb_lower']).all()

    def test_bollinger_invalid_std_dev(self):
        """Test Bollinger Bands with invalid std dev."""
        with pytest.raises(ValueError, match="Standard deviation must be positive"):
            BollingerBands(period=20, std_dev=-1.0)


class TestStandardDeviation:
    """Test Standard Deviation indicator."""

    def test_std_dev_calculation(self, sample_ohlcv_data):
        """Test Standard Deviation calculates correctly."""
        std_dev = StandardDeviation(period=20)
        result = std_dev.calculate(sample_ohlcv_data)

        assert isinstance(result, pd.Series)
        # Standard deviation should be non-negative
        valid_values = result.dropna()
        assert (valid_values >= 0).all()


class TestKeltnerChannel:
    """Test Keltner Channel."""

    def test_keltner_calculation(self, sample_ohlcv_data):
        """Test Keltner Channel calculates all bands."""
        kc = KeltnerChannel(period=20, atr_period=10, multiplier=2.0)
        result = kc.calculate(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)
        assert 'kc_upper' in result.columns
        assert 'kc_middle' in result.columns
        assert 'kc_lower' in result.columns


# ==================== Volume Indicators ====================

class TestOBV:
    """Test On-Balance Volume."""

    def test_obv_calculation(self, sample_ohlcv_data):
        """Test OBV calculates correctly."""
        obv = OBV()
        result = obv.calculate(sample_ohlcv_data)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)


class TestVolumeSMA:
    """Test Volume Simple Moving Average."""

    def test_volume_sma_calculation(self, sample_ohlcv_data):
        """Test Volume SMA calculates correctly."""
        vol_sma = VolumeSMA(period=20)
        result = vol_sma.calculate(sample_ohlcv_data)

        assert isinstance(result, pd.Series)
        # Volume should be positive
        valid_values = result.dropna()
        assert (valid_values >= 0).all()


class TestMFI:
    """Test Money Flow Index."""

    def test_mfi_calculation(self, sample_ohlcv_data):
        """Test MFI calculates correctly."""
        mfi = MFI(period=14)
        result = mfi.calculate(sample_ohlcv_data)

        assert isinstance(result, pd.Series)
        # MFI should be between 0 and 100
        valid_values = result.dropna()
        assert valid_values.min() >= 0
        assert valid_values.max() <= 100


class TestVWAP:
    """Test Volume Weighted Average Price."""

    def test_vwap_calculation(self, sample_ohlcv_data):
        """Test VWAP calculates correctly."""
        vwap = VWAP()
        result = vwap.calculate(sample_ohlcv_data)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)


class TestADI:
    """Test Accumulation/Distribution Index."""

    def test_adi_calculation(self, sample_ohlcv_data):
        """Test ADI calculates correctly."""
        adi = ADI()
        result = adi.calculate(sample_ohlcv_data)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)


class TestCMF:
    """Test Chaikin Money Flow."""

    def test_cmf_calculation(self, sample_ohlcv_data):
        """Test CMF calculates correctly."""
        cmf = CMF(period=20)
        result = cmf.calculate(sample_ohlcv_data)

        assert isinstance(result, pd.Series)
        # CMF typically ranges from -1 to 1
        valid_values = result.dropna()
        assert valid_values.min() >= -1
        assert valid_values.max() <= 1


# ==================== Edge Cases ====================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test indicators with empty dataframe."""
        empty_df = pd.DataFrame()
        rsi = RSI(period=14)

        with pytest.raises(ValueError, match="empty"):
            rsi.calculate(empty_df)

    def test_missing_columns(self, sample_ohlcv_data):
        """Test indicators with missing required columns."""
        incomplete_df = sample_ohlcv_data[['open', 'high']].copy()
        rsi = RSI(period=14)

        with pytest.raises(ValueError, match="Missing required columns"):
            rsi.calculate(incomplete_df)

    def test_nan_values(self, sample_ohlcv_data):
        """Test indicators handle NaN values."""
        df_with_nan = sample_ohlcv_data.copy()
        df_with_nan.loc[10:15, 'close'] = np.nan

        # Indicators should handle NaN gracefully
        rsi = RSI(period=14)
        result = rsi.calculate(df_with_nan)

        assert isinstance(result, pd.Series)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
