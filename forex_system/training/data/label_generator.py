"""Label generation for supervised learning: simple threshold and triple barrier (de Prado)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forex_system.training.config import TrainingConfig


@dataclass
class BarrierResult:
    label: int          # 1=PT hit (buy), -1=SL hit (sell), 0=time barrier
    ret: float          # actual return at exit
    holding_period: int # bars held
    barrier_hit: str    # "pt", "sl", "time"


class LabelGenerator:
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()

    def generate_simple_labels(
        self,
        df: pd.DataFrame,
        lookahead: int = 5,
        buy_threshold: float = 0.003,
        sell_threshold: float = -0.003,
    ) -> pd.Series:
        """
        Simple future-return labeling.
        Returns Series: 1=BUY, 0=HOLD, -1=SELL
        """
        close = df["close"]
        future_return = close.shift(-lookahead) / close - 1.0

        labels = pd.Series(0, index=df.index, dtype=int)
        labels[future_return > buy_threshold] = 1
        labels[future_return < sell_threshold] = -1

        # Drop last `lookahead` rows (no future data)
        labels.iloc[-lookahead:] = np.nan
        return labels

    def generate_triple_barrier_labels(
        self,
        df: pd.DataFrame,
        pt_sl_ratio: float = None,
        max_holding: int = None,
        atr_multiplier: float = None,
        atr_period: int = 14,
    ) -> pd.DataFrame:
        """
        Triple barrier labeling (Marcos Lopez de Prado).

        Three barriers:
        - Upper (profit target): entry + ATR * multiplier * pt_sl_ratio
        - Lower (stop loss): entry - ATR * multiplier
        - Vertical (time): max_holding bars

        First barrier touched determines the label.

        Returns DataFrame with columns:
            label, ret, holding_period, barrier_hit
        """
        pt_sl_ratio = pt_sl_ratio or self.config.TRIPLE_BARRIER_PT_SL_RATIO
        max_holding = max_holding or self.config.TRIPLE_BARRIER_MAX_HOLDING
        atr_multiplier = atr_multiplier or self.config.TRIPLE_BARRIER_ATR_MULTIPLIER

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # Compute ATR
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        atr = pd.Series(np.concatenate([[np.nan], tr])).rolling(atr_period).mean().values

        n = len(close)
        labels = np.full(n, np.nan)
        returns = np.full(n, np.nan)
        holdings = np.full(n, np.nan)
        barriers = np.full(n, "", dtype=object)

        for i in range(n - max_holding):
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue

            entry = close[i]
            sl_width = atr[i] * atr_multiplier
            pt_width = sl_width * pt_sl_ratio

            upper = entry + pt_width
            lower = entry - sl_width

            result = self._check_barriers(
                high, low, close, i, upper, lower, max_holding, entry
            )
            labels[i] = result.label
            returns[i] = result.ret
            holdings[i] = result.holding_period
            barriers[i] = result.barrier_hit

        result_df = pd.DataFrame(
            {"label": labels, "ret": returns, "holding_period": holdings, "barrier_hit": barriers},
            index=df.index,
        )
        return result_df

    @staticmethod
    def _check_barriers(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        entry_idx: int,
        upper: float,
        lower: float,
        max_holding: int,
        entry_price: float,
    ) -> BarrierResult:
        """Check which barrier is hit first after entry."""
        for j in range(1, max_holding + 1):
            idx = entry_idx + j
            if idx >= len(close):
                break

            # Check upper barrier (profit target)
            if high[idx] >= upper:
                ret = upper / entry_price - 1.0
                return BarrierResult(label=1, ret=ret, holding_period=j, barrier_hit="pt")

            # Check lower barrier (stop loss)
            if low[idx] <= lower:
                ret = lower / entry_price - 1.0
                return BarrierResult(label=-1, ret=ret, holding_period=j, barrier_hit="sl")

        # Time barrier hit
        exit_idx = min(entry_idx + max_holding, len(close) - 1)
        ret = close[exit_idx] / entry_price - 1.0
        label = 1 if ret > 0 else (-1 if ret < 0 else 0)
        return BarrierResult(
            label=label, ret=ret, holding_period=exit_idx - entry_idx, barrier_hit="time"
        )

    def generate_news_labels(
        self,
        price_df: pd.DataFrame,
        news_timestamps: pd.Series,
        forward_windows: tuple[int, ...] = None,
    ) -> pd.DataFrame:
        """
        For each news timestamp, compute forward returns at multiple windows.

        Returns DataFrame with columns:
            news_idx, best_window, label, return_2d, return_3d, return_5d
        """
        forward_windows = forward_windows or self.config.NEWS_FORWARD_WINDOWS
        close = price_df["close"]

        results = []
        for news_idx, ts in news_timestamps.items():
            # Find nearest price bar at or after the news timestamp
            valid_dates = close.index[close.index >= ts]
            if valid_dates.empty:
                continue
            entry_date = valid_dates[0]
            entry_price = close[entry_date]

            fwd_returns = {}
            for window in forward_windows:
                future_dates = close.index[close.index > entry_date]
                if len(future_dates) >= window:
                    exit_price = close[future_dates[window - 1]]
                    fwd_returns[window] = exit_price / entry_price - 1.0
                else:
                    fwd_returns[window] = np.nan

            if all(np.isnan(v) for v in fwd_returns.values()):
                continue

            # Best window = largest absolute return
            valid_returns = {k: v for k, v in fwd_returns.items() if not np.isnan(v)}
            if not valid_returns:
                continue
            best_window = max(valid_returns, key=lambda k: abs(valid_returns[k]))
            best_return = valid_returns[best_window]
            label = 1 if best_return > 0 else -1

            row = {
                "news_idx": news_idx,
                "entry_date": entry_date,
                "best_window": best_window,
                "label": label,
            }
            for w in forward_windows:
                row[f"return_{w}d"] = fwd_returns.get(w, np.nan)

            results.append(row)

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results)
