"""
Signal Validation — Step 1 of Investment AI v2.

Before building ANY strategies, prove that tradeable signals exist in our data.
Four tests:
  A. News sentiment → future returns correlation
  B. RSI(2) mean reversion backtest (Connors-style)
  C. News volume → next-day volatility correlation
  D. Event study — returns after high-impact news days

Usage:
    python -m forex_system.analysis.signal_validation
"""

import logging
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from forex_system.training.config import TrainingConfig, ASSET_REGISTRY, get_asset
from forex_system.training.data.price_loader import UniversalPriceLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-25s] %(levelname)s  %(message)s",
)
log = logging.getLogger("signal_validation")


# ── Symbols to validate ──
# Focus assets: 4 stocks + 4 crypto (best signal-to-noise from Run 1)
VALIDATION_SYMBOLS = [
    # Stocks (high volume, lots of news)
    "AAPL", "NVDA", "TSLA", "AMZN",
    # Crypto (24/7 trading, BTC had best Run 1 results at 45%)
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD",
]


# ═══════════════════════════════════════════════════════════════════════
# TEST A: Does news sentiment correlate with future returns?
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SentimentCorrelationResult:
    symbol: str
    n_days: int
    n_articles: int
    corr_1d: float   # Spearman correlation: sentiment vs next-1-day return
    pval_1d: float
    corr_3d: float   # sentiment vs next-3-day return
    pval_3d: float
    corr_5d: float   # sentiment vs next-5-day return
    pval_5d: float
    avg_return_positive: float  # avg 3d return when sentiment > +0.3
    avg_return_negative: float  # avg 3d return when sentiment < -0.3
    signal_exists: bool = False

    def __post_init__(self):
        self.signal_exists = (
            abs(self.corr_3d) > 0.05 and self.pval_3d < 0.05
        )


def test_a_sentiment_correlation(
    config: TrainingConfig,
    price_loader: UniversalPriceLoader,
    symbols: list[str],
) -> list[SentimentCorrelationResult]:
    """
    For each asset, for each day:
      sentiment = avg DistilRoBERTa score of articles on that day
      return_Nd = price change over next N days
      correlation(sentiment, return_Nd)

    Signal threshold: |corr| > 0.05 AND p-value < 0.05
    """
    log.info("=" * 70)
    log.info("TEST A: News Sentiment → Future Returns Correlation")
    log.info("=" * 70)

    results = []
    news_db = config.NEWS_DB_PATH
    sentiment_db = config.SENTIMENT_CACHE_DB_PATH

    for symbol in symbols:
        try:
            asset = get_asset(symbol)
            keywords = asset.keywords
        except KeyError:
            log.warning(f"  {symbol}: not in asset registry, skipping")
            continue

        # Load news articles for this asset
        try:
            from forex_system.training.data.news_loader import NewsLoader
            loader = NewsLoader(news_db)
            articles = loader.load_for_asset(
                keywords,
                start_date="2024-06-01",
                end_date="2026-01-15",
            )
        except Exception as e:
            log.warning(f"  {symbol}: news load failed — {e}")
            continue

        if articles.empty or len(articles) < 30:
            log.info(f"  {symbol}: only {len(articles)} articles, need ≥30. Skipping.")
            continue

        # Score sentiment (using cached scores)
        articles = articles.dropna(subset=["title"])
        articles = articles[articles["title"].str.len() > 10]

        try:
            from forex_system.models.sentiment import SentimentScorer
            scorer = SentimentScorer(cache_db=sentiment_db, device="cpu")
            articles = scorer.score_dataframe(articles, text_col="title")
        except Exception as e:
            log.warning(f"  {symbol}: sentiment scoring failed — {e}")
            continue

        # Aggregate daily: average sentiment score per day
        articles["date_day"] = articles["date"].dt.date
        daily_sent = (
            articles.groupby("date_day")
            .agg(avg_sentiment=("sent_score", "mean"), article_count=("sent_score", "count"))
            .reset_index()
        )
        daily_sent["date_day"] = pd.to_datetime(daily_sent["date_day"])

        # Load price data
        df_price = price_loader.load_ohlcv(symbol, "1d")
        if df_price.empty or len(df_price) < 100:
            log.info(f"  {symbol}: insufficient price data ({len(df_price)} bars)")
            continue

        # Compute forward returns
        df_price = df_price.copy()
        df_price["ret_1d"] = df_price["close"].pct_change(1).shift(-1) * 100
        df_price["ret_3d"] = df_price["close"].pct_change(3).shift(-3) * 100
        df_price["ret_5d"] = df_price["close"].pct_change(5).shift(-5) * 100
        df_price["date_day"] = df_price.index.date
        df_price["date_day"] = pd.to_datetime(df_price["date_day"])

        # Merge sentiment with price
        merged = daily_sent.merge(
            df_price[["date_day", "ret_1d", "ret_3d", "ret_5d"]],
            on="date_day",
            how="inner",
        ).dropna()

        if len(merged) < 30:
            log.info(f"  {symbol}: only {len(merged)} matched days after merge. Skipping.")
            continue

        # Spearman correlations (rank-based, robust to outliers)
        c1, p1 = scipy_stats.spearmanr(merged["avg_sentiment"], merged["ret_1d"])
        c3, p3 = scipy_stats.spearmanr(merged["avg_sentiment"], merged["ret_3d"])
        c5, p5 = scipy_stats.spearmanr(merged["avg_sentiment"], merged["ret_5d"])

        # Conditional returns
        pos_mask = merged["avg_sentiment"] > 0.3
        neg_mask = merged["avg_sentiment"] < -0.3
        avg_ret_pos = merged.loc[pos_mask, "ret_3d"].mean() if pos_mask.sum() > 5 else float("nan")
        avg_ret_neg = merged.loc[neg_mask, "ret_3d"].mean() if neg_mask.sum() > 5 else float("nan")

        result = SentimentCorrelationResult(
            symbol=symbol,
            n_days=len(merged),
            n_articles=len(articles),
            corr_1d=round(c1, 4),
            pval_1d=round(p1, 4),
            corr_3d=round(c3, 4),
            pval_3d=round(p3, 4),
            corr_5d=round(c5, 4),
            pval_5d=round(p5, 4),
            avg_return_positive=round(avg_ret_pos, 3) if not np.isnan(avg_ret_pos) else None,
            avg_return_negative=round(avg_ret_neg, 3) if not np.isnan(avg_ret_neg) else None,
        )
        results.append(result)

        sig = "SIGNAL" if result.signal_exists else "no signal"
        log.info(
            f"  {symbol}: corr_3d={c3:+.4f} (p={p3:.4f}) | "
            f"{len(articles)} articles, {len(merged)} days | {sig}"
        )

    _print_test_a_summary(results)
    return results


def _print_test_a_summary(results: list[SentimentCorrelationResult]):
    log.info("")
    log.info("─── TEST A SUMMARY ─────────────────────────────────────────")
    log.info(f"{'Symbol':<10} {'Days':>5} {'Articles':>8} "
             f"{'Corr1d':>8} {'Corr3d':>8} {'Corr5d':>8} "
             f"{'AvgRet+':>8} {'AvgRet-':>8} {'Signal':>8}")
    log.info("─" * 85)

    for r in results:
        pos = f"{r.avg_return_positive:+.3f}" if r.avg_return_positive is not None else "  N/A"
        neg = f"{r.avg_return_negative:+.3f}" if r.avg_return_negative is not None else "  N/A"
        sig = "YES" if r.signal_exists else "no"
        log.info(
            f"{r.symbol:<10} {r.n_days:>5} {r.n_articles:>8} "
            f"{r.corr_1d:>+8.4f} {r.corr_3d:>+8.4f} {r.corr_5d:>+8.4f} "
            f"{pos:>8} {neg:>8} {sig:>8}"
        )

    signals = [r for r in results if r.signal_exists]
    log.info("─" * 85)
    log.info(f"Signal found in {len(signals)}/{len(results)} assets: "
             f"{[r.symbol for r in signals]}")
    log.info("")


# ═══════════════════════════════════════════════════════════════════════
# TEST B: RSI(2) Mean Reversion Backtest
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RSI2BacktestResult:
    symbol: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    # Long/short breakdown
    long_trades: int
    long_wins: int
    long_win_rate: float
    short_trades: int
    short_wins: int
    short_win_rate: float
    test_period: str
    signal_exists: bool = False

    def __post_init__(self):
        self.signal_exists = self.win_rate > 0.55 and self.total_trades >= 30


def _compute_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    """Compute RSI using Wilder's smoothing (exponential moving average of gains/losses)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def test_b_rsi2_mean_reversion(
    price_loader: UniversalPriceLoader,
    symbols: list[str],
) -> list[RSI2BacktestResult]:
    """
    Connors-style RSI(2) mean reversion on daily data.

    Rules:
      LONG:  RSI(2) < 10 AND price > SMA(200)
      SHORT: RSI(2) > 90 AND price < SMA(200)
      EXIT:  RSI(2) crosses 50 (mean reversion complete)

    Walk-forward: train on nothing (rules are fixed), test on full history.
    This IS the baseline. Published research: 60-91% win rate on stocks.
    """
    log.info("=" * 70)
    log.info("TEST B: RSI(2) Mean Reversion Backtest")
    log.info("=" * 70)

    results = []

    for symbol in symbols:
        df = price_loader.load_ohlcv(symbol, "1d")
        if df.empty or len(df) < 250:
            log.info(f"  {symbol}: insufficient data ({len(df)} bars, need 250+)")
            continue

        df = df.copy()
        df["rsi2"] = _compute_rsi(df["close"], period=2)
        df["sma200"] = _compute_sma(df["close"], period=200)
        df = df.dropna(subset=["rsi2", "sma200"])

        if len(df) < 100:
            log.info(f"  {symbol}: insufficient data after indicator warmup")
            continue

        # Walk through the data simulating trades
        trades = []
        position = None  # None, "long", "short"
        entry_price = 0.0
        entry_date = None

        for i in range(1, len(df)):
            row = df.iloc[i]
            rsi = row["rsi2"]
            price = row["close"]
            sma200 = row["sma200"]
            date = df.index[i]

            # Exit logic
            if position == "long" and rsi > 50:
                pnl_pct = (price / entry_price - 1) * 100
                trades.append({
                    "direction": "LONG",
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl_pct": pnl_pct,
                    "holding_days": (date - entry_date).days,
                })
                position = None

            elif position == "short" and rsi < 50:
                pnl_pct = (entry_price / price - 1) * 100
                trades.append({
                    "direction": "SHORT",
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl_pct": pnl_pct,
                    "holding_days": (date - entry_date).days,
                })
                position = None

            # Entry logic (only if flat)
            if position is None:
                if rsi < 10 and price > sma200:
                    position = "long"
                    entry_price = price
                    entry_date = date
                elif rsi > 90 and price < sma200:
                    position = "short"
                    entry_price = price
                    entry_date = date

        if not trades:
            log.info(f"  {symbol}: no trades triggered")
            continue

        trades_df = pd.DataFrame(trades)

        # Statistics
        total = len(trades_df)
        wins = (trades_df["pnl_pct"] > 0).sum()
        losses = total - wins
        win_rate = wins / total if total > 0 else 0

        winning = trades_df[trades_df["pnl_pct"] > 0]["pnl_pct"]
        losing = trades_df[trades_df["pnl_pct"] <= 0]["pnl_pct"]
        avg_win = winning.mean() if len(winning) > 0 else 0
        avg_loss = abs(losing.mean()) if len(losing) > 0 else 0
        profit_factor = (winning.sum() / abs(losing.sum())) if losing.sum() != 0 else float("inf")

        # Equity curve
        equity = (1 + trades_df["pnl_pct"] / 100).cumprod()
        total_return = (equity.iloc[-1] - 1) * 100 if len(equity) > 0 else 0

        # Max drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_dd = abs(drawdown.min()) * 100

        # Sharpe (annualized, assuming avg 5 trades/month)
        ret_series = trades_df["pnl_pct"] / 100
        if ret_series.std() > 0:
            sharpe = (ret_series.mean() / ret_series.std()) * np.sqrt(252 / max(trades_df["holding_days"].mean(), 1))
        else:
            sharpe = 0.0

        # Long/short breakdown
        longs = trades_df[trades_df["direction"] == "LONG"]
        shorts = trades_df[trades_df["direction"] == "SHORT"]
        long_wins = (longs["pnl_pct"] > 0).sum() if len(longs) > 0 else 0
        short_wins = (shorts["pnl_pct"] > 0).sum() if len(shorts) > 0 else 0

        result = RSI2BacktestResult(
            symbol=symbol,
            total_trades=total,
            wins=int(wins),
            losses=int(losses),
            win_rate=round(win_rate, 4),
            avg_win_pct=round(avg_win, 3),
            avg_loss_pct=round(avg_loss, 3),
            profit_factor=round(profit_factor, 2),
            total_return_pct=round(total_return, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            long_trades=len(longs),
            long_wins=int(long_wins),
            long_win_rate=round(long_wins / max(len(longs), 1), 4),
            short_trades=len(shorts),
            short_wins=int(short_wins),
            short_win_rate=round(short_wins / max(len(shorts), 1), 4),
            test_period=f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
        )
        results.append(result)

        sig = "SIGNAL" if result.signal_exists else "no signal"
        log.info(
            f"  {symbol}: {total} trades, win_rate={win_rate:.1%}, "
            f"PF={profit_factor:.2f}, return={total_return:+.1f}%, "
            f"sharpe={sharpe:.2f} | {sig}"
        )

    _print_test_b_summary(results)
    return results


def _print_test_b_summary(results: list[RSI2BacktestResult]):
    log.info("")
    log.info("─── TEST B SUMMARY ─────────────────────────────────────────")
    log.info(f"{'Symbol':<10} {'Trades':>6} {'WinRate':>8} {'PF':>6} "
             f"{'Return%':>8} {'MaxDD%':>7} {'Sharpe':>7} "
             f"{'L_WR':>6} {'S_WR':>6} {'Signal':>7}")
    log.info("─" * 85)

    for r in results:
        sig = "YES" if r.signal_exists else "no"
        l_wr = f"{r.long_win_rate:.0%}" if r.long_trades > 0 else "N/A"
        s_wr = f"{r.short_win_rate:.0%}" if r.short_trades > 0 else "N/A"
        log.info(
            f"{r.symbol:<10} {r.total_trades:>6} {r.win_rate:>7.1%} {r.profit_factor:>6.2f} "
            f"{r.total_return_pct:>+7.1f}% {r.max_drawdown_pct:>6.1f}% {r.sharpe_ratio:>7.2f} "
            f"{l_wr:>6} {s_wr:>6} {sig:>7}"
        )

    signals = [r for r in results if r.signal_exists]
    log.info("─" * 85)
    log.info(f"Signal found in {len(signals)}/{len(results)} assets: "
             f"{[r.symbol for r in signals]}")

    if signals:
        avg_wr = np.mean([r.win_rate for r in signals])
        avg_pf = np.mean([r.profit_factor for r in signals])
        log.info(f"Average among signals: win_rate={avg_wr:.1%}, profit_factor={avg_pf:.2f}")
    log.info("")


# ═══════════════════════════════════════════════════════════════════════
# TEST C: News Volume → Next-Day Volatility
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class NewsVolumeResult:
    symbol: str
    n_days: int
    corr_volume_vs_absret: float    # Spearman: article_count vs |next_day_return|
    pval_volume_vs_absret: float
    corr_volume_vs_range: float     # Spearman: article_count vs next_day (high-low)/close
    pval_volume_vs_range: float
    avg_vol_high_news: float        # avg |return| on days with > 2x avg articles
    avg_vol_low_news: float         # avg |return| on days with < 0.5x avg articles
    signal_exists: bool = False

    def __post_init__(self):
        self.signal_exists = (
            self.corr_volume_vs_absret > 0.05 and self.pval_volume_vs_absret < 0.05
        )


def test_c_news_volume_volatility(
    config: TrainingConfig,
    price_loader: UniversalPriceLoader,
    symbols: list[str],
) -> list[NewsVolumeResult]:
    """
    Test: does news volume predict next-day volatility?
    If more articles = bigger moves, we can predict WHEN to trade (volatility timing).
    """
    log.info("=" * 70)
    log.info("TEST C: News Volume → Next-Day Volatility")
    log.info("=" * 70)

    results = []
    news_db = config.NEWS_DB_PATH

    for symbol in symbols:
        try:
            asset = get_asset(symbol)
            keywords = asset.keywords
        except KeyError:
            continue

        try:
            from forex_system.training.data.news_loader import NewsLoader
            loader = NewsLoader(news_db)
            articles = loader.load_for_asset(
                keywords,
                start_date="2024-06-01",
                end_date="2026-01-15",
            )
        except Exception as e:
            log.warning(f"  {symbol}: news load failed — {e}")
            continue

        if articles.empty or len(articles) < 30:
            log.info(f"  {symbol}: insufficient news data ({len(articles)} articles)")
            continue

        # Daily article count
        articles["date_day"] = articles["date"].dt.date
        daily_count = (
            articles.groupby("date_day")
            .size()
            .reset_index(name="article_count")
        )
        daily_count["date_day"] = pd.to_datetime(daily_count["date_day"])

        # Load price data
        df_price = price_loader.load_ohlcv(symbol, "1d")
        if df_price.empty or len(df_price) < 100:
            continue

        df_price = df_price.copy()
        # Next-day absolute return and range
        df_price["abs_ret_1d"] = df_price["close"].pct_change(1).shift(-1).abs() * 100
        df_price["range_1d"] = ((df_price["high"].shift(-1) - df_price["low"].shift(-1))
                                / df_price["close"].shift(-1) * 100)
        df_price["date_day"] = df_price.index.date
        df_price["date_day"] = pd.to_datetime(df_price["date_day"])

        # Merge
        merged = daily_count.merge(
            df_price[["date_day", "abs_ret_1d", "range_1d"]],
            on="date_day",
            how="inner",
        ).dropna()

        if len(merged) < 30:
            continue

        # Correlations
        c_ret, p_ret = scipy_stats.spearmanr(merged["article_count"], merged["abs_ret_1d"])
        c_rng, p_rng = scipy_stats.spearmanr(merged["article_count"], merged["range_1d"])

        # Conditional volatility
        avg_count = merged["article_count"].mean()
        high_news = merged[merged["article_count"] > avg_count * 2]
        low_news = merged[merged["article_count"] < avg_count * 0.5]
        avg_vol_high = high_news["abs_ret_1d"].mean() if len(high_news) > 5 else float("nan")
        avg_vol_low = low_news["abs_ret_1d"].mean() if len(low_news) > 5 else float("nan")

        result = NewsVolumeResult(
            symbol=symbol,
            n_days=len(merged),
            corr_volume_vs_absret=round(c_ret, 4),
            pval_volume_vs_absret=round(p_ret, 4),
            corr_volume_vs_range=round(c_rng, 4),
            pval_volume_vs_range=round(p_rng, 4),
            avg_vol_high_news=round(avg_vol_high, 3) if not np.isnan(avg_vol_high) else None,
            avg_vol_low_news=round(avg_vol_low, 3) if not np.isnan(avg_vol_low) else None,
        )
        results.append(result)

        sig = "SIGNAL" if result.signal_exists else "no signal"
        log.info(
            f"  {symbol}: corr(volume, |ret|)={c_ret:+.4f} (p={p_ret:.4f}), "
            f"high_news_vol={avg_vol_high:.3f}% vs low={avg_vol_low:.3f}% | {sig}"
        )

    _print_test_c_summary(results)
    return results


def _print_test_c_summary(results: list[NewsVolumeResult]):
    log.info("")
    log.info("─── TEST C SUMMARY ─────────────────────────────────────────")
    log.info(f"{'Symbol':<10} {'Days':>5} {'Corr|Ret|':>10} {'pval':>7} "
             f"{'CorrRange':>10} {'HighVol':>8} {'LowVol':>8} {'Signal':>7}")
    log.info("─" * 75)

    for r in results:
        sig = "YES" if r.signal_exists else "no"
        hv = f"{r.avg_vol_high_news:.3f}" if r.avg_vol_high_news is not None else "N/A"
        lv = f"{r.avg_vol_low_news:.3f}" if r.avg_vol_low_news is not None else "N/A"
        log.info(
            f"{r.symbol:<10} {r.n_days:>5} {r.corr_volume_vs_absret:>+10.4f} "
            f"{r.pval_volume_vs_absret:>7.4f} {r.corr_volume_vs_range:>+10.4f} "
            f"{hv:>8} {lv:>8} {sig:>7}"
        )

    signals = [r for r in results if r.signal_exists]
    log.info("─" * 75)
    log.info(f"Signal found in {len(signals)}/{len(results)} assets: "
             f"{[r.symbol for r in signals]}")
    log.info("")


# ═══════════════════════════════════════════════════════════════════════
# TEST D: Event Study — Returns After High-Impact News Days
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EventStudyResult:
    symbol: str
    event_type: str
    n_events: int
    avg_ret_1d: float
    avg_ret_3d: float
    avg_ret_5d: float
    std_ret_3d: float
    t_stat_3d: float
    p_value_3d: float
    positive_pct: float  # % of events with positive 3d return
    significant: bool = False

    def __post_init__(self):
        self.significant = abs(self.t_stat_3d) > 2.0 and self.p_value_3d < 0.05


def test_d_event_study(
    config: TrainingConfig,
    price_loader: UniversalPriceLoader,
    symbols: list[str],
) -> list[EventStudyResult]:
    """
    Find days with extreme sentiment (proxy for high-impact events).
    Measure average return 1/3/5 days after.
    Test if the post-event returns are statistically different from zero.
    """
    log.info("=" * 70)
    log.info("TEST D: Event Study — Returns After High-Impact News Days")
    log.info("=" * 70)

    results = []
    news_db = config.NEWS_DB_PATH
    sentiment_db = config.SENTIMENT_CACHE_DB_PATH

    for symbol in symbols:
        try:
            asset = get_asset(symbol)
            keywords = asset.keywords
        except KeyError:
            continue

        try:
            from forex_system.training.data.news_loader import NewsLoader
            loader = NewsLoader(news_db)
            articles = loader.load_for_asset(
                keywords,
                start_date="2024-06-01",
                end_date="2026-01-15",
            )
        except Exception as e:
            log.warning(f"  {symbol}: news load failed — {e}")
            continue

        if articles.empty or len(articles) < 30:
            continue

        # Score sentiment
        articles = articles.dropna(subset=["title"])
        articles = articles[articles["title"].str.len() > 10]

        try:
            from forex_system.models.sentiment import SentimentScorer
            scorer = SentimentScorer(cache_db=sentiment_db, device="cpu")
            articles = scorer.score_dataframe(articles, text_col="title")
        except Exception:
            continue

        # Daily aggregation
        articles["date_day"] = articles["date"].dt.date
        daily = (
            articles.groupby("date_day")
            .agg(
                avg_sentiment=("sent_score", "mean"),
                max_sentiment=("sent_score", "max"),
                min_sentiment=("sent_score", "min"),
                article_count=("sent_score", "count"),
            )
            .reset_index()
        )
        daily["date_day"] = pd.to_datetime(daily["date_day"])

        # Load price data
        df_price = price_loader.load_ohlcv(symbol, "1d")
        if df_price.empty:
            continue

        df_price = df_price.copy()
        df_price["ret_1d"] = df_price["close"].pct_change(1).shift(-1) * 100
        df_price["ret_3d"] = df_price["close"].pct_change(3).shift(-3) * 100
        df_price["ret_5d"] = df_price["close"].pct_change(5).shift(-5) * 100
        df_price["date_day"] = df_price.index.date
        df_price["date_day"] = pd.to_datetime(df_price["date_day"])

        merged = daily.merge(
            df_price[["date_day", "ret_1d", "ret_3d", "ret_5d"]],
            on="date_day",
            how="inner",
        ).dropna()

        if len(merged) < 20:
            continue

        # Define event types using sentiment thresholds
        event_types = {
            "strong_positive": merged[merged["avg_sentiment"] > 0.5],
            "strong_negative": merged[merged["avg_sentiment"] < -0.5],
            "high_volume": merged[merged["article_count"] > merged["article_count"].quantile(0.9)],
            "extreme_positive_article": merged[merged["max_sentiment"] > 0.9],
            "extreme_negative_article": merged[merged["min_sentiment"] < -0.9],
        }

        for event_type, event_df in event_types.items():
            if len(event_df) < 5:
                continue

            ret_3d = event_df["ret_3d"]
            t_stat, p_value = scipy_stats.ttest_1samp(ret_3d, 0)

            result = EventStudyResult(
                symbol=symbol,
                event_type=event_type,
                n_events=len(event_df),
                avg_ret_1d=round(event_df["ret_1d"].mean(), 3),
                avg_ret_3d=round(ret_3d.mean(), 3),
                avg_ret_5d=round(event_df["ret_5d"].mean(), 3),
                std_ret_3d=round(ret_3d.std(), 3),
                t_stat_3d=round(t_stat, 3),
                p_value_3d=round(p_value, 4),
                positive_pct=round((ret_3d > 0).mean() * 100, 1),
            )
            results.append(result)

            if result.significant:
                log.info(
                    f"  {symbol} [{event_type}]: {len(event_df)} events, "
                    f"avg_3d_ret={ret_3d.mean():+.3f}%, t={t_stat:.2f}, "
                    f"p={p_value:.4f} *** SIGNIFICANT ***"
                )

    _print_test_d_summary(results)
    return results


def _print_test_d_summary(results: list[EventStudyResult]):
    log.info("")
    log.info("─── TEST D SUMMARY ─────────────────────────────────────────")
    log.info(f"{'Symbol':<10} {'Event Type':<25} {'N':>4} "
             f"{'Ret1d':>7} {'Ret3d':>7} {'Ret5d':>7} "
             f"{'t-stat':>7} {'p-val':>7} {'%Pos':>5} {'Sig':>5}")
    log.info("─" * 95)

    for r in results:
        sig = "***" if r.significant else ""
        log.info(
            f"{r.symbol:<10} {r.event_type:<25} {r.n_events:>4} "
            f"{r.avg_ret_1d:>+7.3f} {r.avg_ret_3d:>+7.3f} {r.avg_ret_5d:>+7.3f} "
            f"{r.t_stat_3d:>7.2f} {r.p_value_3d:>7.4f} {r.positive_pct:>4.0f}% {sig:>5}"
        )

    sig_results = [r for r in results if r.significant]
    log.info("─" * 95)
    log.info(f"Statistically significant events: {len(sig_results)}/{len(results)}")
    if sig_results:
        for r in sig_results:
            log.info(f"  {r.symbol} [{r.event_type}]: avg_3d={r.avg_ret_3d:+.3f}%, "
                     f"t={r.t_stat_3d:.2f}, p={r.p_value_3d:.4f}")
    log.info("")


# ═══════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════

def print_final_verdict(
    results_a: list[SentimentCorrelationResult],
    results_b: list[RSI2BacktestResult],
    results_c: list[NewsVolumeResult],
    results_d: list[EventStudyResult],
):
    log.info("=" * 70)
    log.info("FINAL VERDICT — Where Is the Signal?")
    log.info("=" * 70)

    # Test A
    a_signals = [r for r in results_a if r.signal_exists]
    log.info(f"\nTest A (Sentiment → Returns): "
             f"{'SIGNAL' if a_signals else 'NO SIGNAL'} "
             f"({len(a_signals)}/{len(results_a)} assets)")
    if a_signals:
        for r in a_signals:
            log.info(f"  {r.symbol}: corr_3d={r.corr_3d:+.4f} (p={r.pval_3d:.4f})")

    # Test B
    b_signals = [r for r in results_b if r.signal_exists]
    log.info(f"\nTest B (RSI(2) Mean Reversion): "
             f"{'SIGNAL' if b_signals else 'NO SIGNAL'} "
             f"({len(b_signals)}/{len(results_b)} assets)")
    if b_signals:
        for r in b_signals:
            log.info(f"  {r.symbol}: win_rate={r.win_rate:.1%}, "
                     f"PF={r.profit_factor:.2f}, trades={r.total_trades}")

    # Test C
    c_signals = [r for r in results_c if r.signal_exists]
    log.info(f"\nTest C (News Volume → Volatility): "
             f"{'SIGNAL' if c_signals else 'NO SIGNAL'} "
             f"({len(c_signals)}/{len(results_c)} assets)")
    if c_signals:
        for r in c_signals:
            log.info(f"  {r.symbol}: corr={r.corr_volume_vs_absret:+.4f} "
                     f"(p={r.pval_volume_vs_absret:.4f})")

    # Test D
    d_signals = [r for r in results_d if r.significant]
    log.info(f"\nTest D (Event Study): "
             f"{'SIGNAL' if d_signals else 'NO SIGNAL'} "
             f"({len(d_signals)}/{len(results_d)} event types)")
    if d_signals:
        for r in d_signals:
            log.info(f"  {r.symbol} [{r.event_type}]: avg_3d={r.avg_ret_3d:+.3f}%, "
                     f"t={r.t_stat_3d:.2f}")

    # Overall recommendation
    log.info("\n" + "=" * 70)
    has_technical = len(b_signals) > 0
    has_news_direction = len(a_signals) > 0
    has_news_timing = len(c_signals) > 0
    has_events = len(d_signals) > 0

    if has_technical:
        log.info("RECOMMENDATION: PROCEED with statistical strategies.")
        log.info(f"  RSI(2) mean reversion works on: {[r.symbol for r in b_signals]}")
        if has_news_direction:
            log.info(f"  News sentiment adds directional signal for: "
                     f"{[r.symbol for r in a_signals]}")
        if has_news_timing:
            log.info(f"  News volume helps timing for: "
                     f"{[r.symbol for r in c_signals]}")
        if has_events:
            log.info(f"  Event-driven edges found in: "
                     f"{list(set(r.symbol for r in d_signals))}")
    elif has_news_direction or has_events:
        log.info("RECOMMENDATION: PROCEED with caution.")
        log.info("  Technical baseline (RSI2) did not reach 55% win rate.")
        log.info("  But news-based signals exist. Build news-enhanced model.")
    else:
        log.info("RECOMMENDATION: STOP. No tradeable signal found.")
        log.info("  Neither technical nor news-based approaches show edge.")
        log.info("  Rethink data sources, timeframes, or asset selection.")

    log.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def run_all(symbols: list[str] = None):
    """Run all 4 signal validation tests."""
    symbols = symbols or VALIDATION_SYMBOLS
    config = TrainingConfig()
    price_loader = UniversalPriceLoader(config)

    log.info("Signal Validation — Step 1 of Investment AI v2")
    log.info(f"Testing {len(symbols)} assets: {symbols}")
    log.info(f"Price DB: {config.PRICE_DB_PATH}")
    log.info(f"News DB: {config.NEWS_DB_PATH}")
    log.info(f"Sentiment cache: {config.SENTIMENT_CACHE_DB_PATH}")
    log.info("")

    results_a = test_a_sentiment_correlation(config, price_loader, symbols)
    results_b = test_b_rsi2_mean_reversion(price_loader, symbols)
    results_c = test_c_news_volume_volatility(config, price_loader, symbols)
    results_d = test_d_event_study(config, price_loader, symbols)

    print_final_verdict(results_a, results_b, results_c, results_d)

    return {
        "test_a": results_a,
        "test_b": results_b,
        "test_c": results_c,
        "test_d": results_d,
    }


if __name__ == "__main__":
    symbols = sys.argv[1:] if len(sys.argv) > 1 else None
    run_all(symbols)
