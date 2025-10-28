"""
SYNTHETIC GENERATOR - Create 1000+ Trading Scenarios from 1 API Call
Methods: Bootstrapping, Monte Carlo, Pattern Variations
Output: Synthetic OHLC data for backtesting and trading
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SyntheticGenerator:
    """
    Generate synthetic market data to maximize API efficiency
    1 real data point -> 1000+ trading scenarios
    """
    
    def __init__(self):
        self.scenarios_generated = 0
        self.methods = ['bootstrap', 'monte_carlo', 'pattern_variation', 'regime_switch', 'fractal']
        
    def generate_all_scenarios(self, df: pd.DataFrame, n_scenarios: int = 100) -> List[pd.DataFrame]:
        """
        Generate scenarios using all methods
        """
        scenarios = []
        
        # Distribute scenarios across methods
        per_method = n_scenarios // len(self.methods)
        
        # 1. Bootstrap scenarios
        scenarios.extend(self.generate_bootstrap_scenarios(df, per_method))
        
        # 2. Monte Carlo scenarios
        scenarios.extend(self.generate_monte_carlo_scenarios(df, per_method))
        
        # 3. Pattern variation scenarios
        scenarios.extend(self.generate_pattern_variations(df, per_method))
        
        # 4. Regime switch scenarios
        scenarios.extend(self.generate_regime_scenarios(df, per_method))
        
        # 5. Fractal scenarios
        scenarios.extend(self.generate_fractal_scenarios(df, n_scenarios - len(scenarios)))
        
        self.scenarios_generated += len(scenarios)
        
        return scenarios
        
    def generate_bootstrap_scenarios(self, df: pd.DataFrame, n: int = 20) -> List[pd.DataFrame]:
        """
        Bootstrap resampling with replacement
        Preserves actual market microstructure
        """
        scenarios = []
        
        if len(df) < 60:
            return scenarios
            
        returns = df['close'].pct_change().dropna().values
        
        for _ in range(n):
            # Bootstrap returns
            bootstrapped_returns = np.random.choice(returns, size=len(df), replace=True)
            
            # Reconstruct prices
            prices = [df['close'].iloc[0]]
            for ret in bootstrapped_returns[1:]:
                prices.append(prices[-1] * (1 + ret))
                
            # Create OHLC
            scenario = self.create_ohlc_from_prices(prices, df.index)
            scenarios.append(scenario)
            
        return scenarios
        
    def generate_monte_carlo_scenarios(self, df: pd.DataFrame, n: int = 20) -> List[pd.DataFrame]:
        """
        Monte Carlo simulation using historical parameters
        """
        scenarios = []
        
        if len(df) < 30:
            return scenarios
            
        # Calculate parameters from real data
        returns = df['close'].pct_change().dropna().values
        mu = returns.mean()
        sigma = returns.std()
        
        # Additional parameters
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        for _ in range(n):
            # Generate returns with similar distribution
            if abs(skew) > 0.5 or abs(kurt) > 1:
                # Use skewed distribution for non-normal returns
                simulated_returns = stats.skewnorm.rvs(
                    a=skew, 
                    loc=mu, 
                    scale=sigma, 
                    size=len(df)
                )
            else:
                # Normal distribution
                simulated_returns = np.random.normal(mu, sigma, len(df))
                
            # Add autocorrelation (momentum/mean reversion)
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            for i in range(1, len(simulated_returns)):
                simulated_returns[i] += autocorr * simulated_returns[i-1] * 0.3
                
            # Reconstruct prices
            prices = [df['close'].iloc[0]]
            for ret in simulated_returns[1:]:
                prices.append(prices[-1] * (1 + ret))
                
            # Create OHLC
            scenario = self.create_ohlc_from_prices(prices, df.index)
            scenarios.append(scenario)
            
        return scenarios
        
    def generate_pattern_variations(self, df: pd.DataFrame, n: int = 20) -> List[pd.DataFrame]:
        """
        Create variations of detected patterns
        """
        scenarios = []
        
        if len(df) < 50:
            return scenarios
            
        closes = df['close'].values
        
        for _ in range(n):
            # Identify trend
            trend = np.polyfit(range(len(closes)), closes, 1)[0]
            
            # Generate variation
            variation = closes.copy()
            
            # Random walk around trend
            noise = np.random.normal(0, abs(trend) * 10, len(variation))
            noise = np.cumsum(noise) * 0.0001
            
            # Add patterns
            pattern_type = np.random.choice(['sine', 'triangle', 'sawtooth', 'square'])
            
            if pattern_type == 'sine':
                # Sine wave pattern
                freq = np.random.uniform(0.05, 0.2)
                amplitude = np.random.uniform(0.0005, 0.002)
                pattern = amplitude * np.sin(2 * np.pi * freq * np.arange(len(variation)))
                
            elif pattern_type == 'triangle':
                # Triangle wave
                period = np.random.randint(10, 30)
                amplitude = np.random.uniform(0.0005, 0.002)
                pattern = amplitude * signal.sawtooth(2 * np.pi * np.arange(len(variation)) / period, 0.5)
                
            elif pattern_type == 'sawtooth':
                # Sawtooth pattern
                period = np.random.randint(10, 30)
                amplitude = np.random.uniform(0.0005, 0.002)
                pattern = amplitude * signal.sawtooth(2 * np.pi * np.arange(len(variation)) / period)
                
            else:  # square
                # Square wave
                period = np.random.randint(10, 30)
                amplitude = np.random.uniform(0.0005, 0.002)
                pattern = amplitude * signal.square(2 * np.pi * np.arange(len(variation)) / period)
                
            variation = variation + pattern + noise
            
            # Create OHLC
            scenario = self.create_ohlc_from_prices(variation, df.index)
            scenarios.append(scenario)
            
        return scenarios
        
    def generate_regime_scenarios(self, df: pd.DataFrame, n: int = 20) -> List[pd.DataFrame]:
        """
        Generate scenarios with different market regimes
        (trending, ranging, volatile)
        """
        scenarios = []
        
        if len(df) < 30:
            return scenarios
            
        base_price = df['close'].iloc[-1]
        
        for _ in range(n):
            regime = np.random.choice(['trending', 'ranging', 'volatile'])
            
            prices = [base_price]
            
            if regime == 'trending':
                # Strong trend
                trend = np.random.uniform(-0.0002, 0.0002)
                volatility = 0.0005
                
                for i in range(1, len(df)):
                    change = trend + np.random.normal(0, volatility)
                    prices.append(prices[-1] * (1 + change))
                    
            elif regime == 'ranging':
                # Range-bound market
                range_center = base_price
                range_width = 0.002
                
                for i in range(1, len(df)):
                    # Mean reversion to center
                    distance_from_center = (prices[-1] - range_center) / range_center
                    reversion = -distance_from_center * 0.1
                    noise = np.random.normal(0, 0.0003)
                    change = reversion + noise
                    
                    new_price = prices[-1] * (1 + change)
                    
                    # Keep within range
                    new_price = np.clip(
                        new_price,
                        range_center * (1 - range_width),
                        range_center * (1 + range_width)
                    )
                    prices.append(new_price)
                    
            else:  # volatile
                # High volatility regime
                volatility = 0.002
                
                for i in range(1, len(df)):
                    # Fat-tailed distribution for extreme moves
                    if np.random.random() < 0.05:  # 5% chance of extreme move
                        change = np.random.normal(0, volatility * 3)
                    else:
                        change = np.random.normal(0, volatility)
                        
                    prices.append(prices[-1] * (1 + change))
                    
            # Create OHLC
            scenario = self.create_ohlc_from_prices(prices, df.index)
            scenarios.append(scenario)
            
        return scenarios
        
    def generate_fractal_scenarios(self, df: pd.DataFrame, n: int = 20) -> List[pd.DataFrame]:
        """
        Generate fractal/self-similar scenarios
        Markets exhibit fractal properties
        """
        scenarios = []
        
        if len(df) < 20:
            return scenarios
            
        # Extract fractal dimension from real data
        closes = df['close'].values
        fractal_dim = self.calculate_fractal_dimension(closes)
        
        for _ in range(n):
            # Generate fractional Brownian motion
            hurst = fractal_dim / 2  # Approximate Hurst exponent
            prices = self.fractional_brownian_motion(
                n=len(df),
                hurst=hurst,
                start_price=df['close'].iloc[0]
            )
            
            # Create OHLC
            scenario = self.create_ohlc_from_prices(prices, df.index)
            scenarios.append(scenario)
            
        return scenarios
        
    def create_ohlc_from_prices(self, prices: List[float], index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Create OHLC data from price series
        """
        prices = np.array(prices)
        
        # Create realistic OHLC
        df = pd.DataFrame(index=index)
        
        df['close'] = prices
        df['open'] = np.roll(prices, 1)
        df['open'][0] = prices[0]
        
        # High and low with realistic wicks
        wick_size = np.abs(np.random.normal(0, 0.0002, len(prices)))
        df['high'] = np.maximum(df['open'], df['close']) + wick_size
        df['low'] = np.minimum(df['open'], df['close']) - wick_size
        
        # Volume (synthetic)
        avg_volume = 1000
        df['volume'] = np.random.poisson(avg_volume, len(prices))
        
        return df
        
    def calculate_fractal_dimension(self, prices: np.array) -> float:
        """
        Calculate fractal dimension using box-counting method
        """
        if len(prices) < 10:
            return 1.5  # Default
            
        # Normalize prices
        prices = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)
        
        # Box sizes
        scales = [2, 4, 8, 16]
        counts = []
        
        for scale in scales:
            # Count boxes needed to cover the price series
            boxes = len(prices) // scale
            if boxes > 0:
                counts.append(boxes)
            else:
                counts.append(1)
                
        # Calculate dimension
        if len(counts) > 1:
            coeffs = np.polyfit(np.log(scales[:len(counts)]), np.log(counts), 1)
            return abs(coeffs[0])
        else:
            return 1.5
            
    def fractional_brownian_motion(self, n: int, hurst: float = 0.5, start_price: float = 1.0) -> np.array:
        """
        Generate fractional Brownian motion
        Hurst < 0.5: mean reverting
        Hurst = 0.5: random walk
        Hurst > 0.5: trending
        """
        
        # Ensure valid Hurst exponent
        hurst = np.clip(hurst, 0.1, 0.9)
        
        # Generate correlated increments
        increments = np.random.randn(n)
        
        # Apply long-range dependence
        fBm = np.zeros(n)
        fBm[0] = increments[0]
        
        for t in range(1, n):
            # Weight past increments based on Hurst exponent
            weights = np.power(np.arange(1, t+1), hurst - 0.5)
            weights = weights / weights.sum()
            
            # Weighted sum of past increments
            if t < 100:  # Limit lookback for efficiency
                fBm[t] = np.sum(weights * increments[:t]) + increments[t] * 0.1
            else:
                fBm[t] = np.sum(weights[-100:] * increments[t-100:t]) + increments[t] * 0.1
                
        # Scale and add to start price
        scale = 0.001  # Volatility scale
        prices = start_price * np.exp(np.cumsum(fBm * scale))
        
        return prices
        
    def add_market_events(self, df: pd.DataFrame, event_probability: float = 0.05) -> pd.DataFrame:
        """
        Add realistic market events (news, announcements, etc.)
        """
        df = df.copy()
        
        for i in range(len(df)):
            if np.random.random() < event_probability:
                # Random event impact
                impact = np.random.choice([-1, 1]) * np.random.uniform(0.001, 0.005)
                
                # Apply to OHLC
                df.loc[df.index[i], 'close'] *= (1 + impact)
                df.loc[df.index[i], 'high'] *= (1 + abs(impact))
                df.loc[df.index[i], 'low'] *= (1 - abs(impact) * 0.5)
                
                # Increase volume
                df.loc[df.index[i], 'volume'] *= np.random.uniform(2, 5)
                
        return df


# Import scipy.signal if available
try:
    from scipy import signal
except ImportError:
    # Simple fallback implementations
    class signal:
        @staticmethod
        def sawtooth(t, width=1):
            t = np.mod(t, 2*np.pi)
            return 2 * (t / (2*np.pi) - 0.5)
            
        @staticmethod
        def square(t):
            return np.sign(np.sin(t))


if __name__ == "__main__":
    # Test synthetic generator
    print("Testing Synthetic Generator...")
    
    # Create sample data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='1H')
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 1.0850,
        'high': np.random.randn(100).cumsum() + 1.0860,
        'low': np.random.randn(100).cumsum() + 1.0840,
        'close': np.random.randn(100).cumsum() + 1.0850,
        'volume': np.random.randint(100, 1000, 100)
    }, index=dates)
    
    # Generate scenarios
    generator = SyntheticGenerator()
    scenarios = generator.generate_all_scenarios(df, n_scenarios=10)
    
    print(f"Generated {len(scenarios)} scenarios")
    print(f"First scenario shape: {scenarios[0].shape}")
    print(f"Price range: {scenarios[0]['close'].min():.4f} - {scenarios[0]['close'].max():.4f}")