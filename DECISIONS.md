# Investment AI — Decisions & Learnings Log

This file records every major decision, finding, and lesson learned during development.
**Read this at the start of every session to avoid repeating mistakes.**

---

## Session: 2026-03-16 to 2026-03-19

### What We Tried and What Failed

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| Run 1: LLM-as-predictor (Qwen3.5-35B) | 40.8% win rate on 731 trades | LLMs cannot predict price direction from news+technicals |
| Run 2 v1: XGBoost ensemble (22 features) | -16.5% walk-forward on BTC-USD | No stop-loss, 50% model accuracy, -11.4% in costs |
| Run 2 v2: 3-model ensemble (114 features, de Prado) | -14.1% walk-forward | Models overfit on every fold (90%+ train, ~50% calib). <2000 daily bars is insufficient for ML |
| Time-series momentum (single asset) | +4.2% NVDA, -4.2% BTC | Inconsistent — works on trending assets, fails on choppy ones |
| Cross-sectional momentum on crypto | -60% | Crypto too volatile, momentum selects crypto exclusively then crashes |
| News sentiment → direction prediction | ~50% accuracy | Confirmed in Signal Validation (4 tests), SSD system's own backtests, AND Qwen 4-day analysis |
| ATR-based 4-layer exit engine | Didn't help ML | Good exits can't save garbage entries |
| RSI(2) mean reversion | 68% in-sample, underperforms B&H in 7/8 assets | Just buying dips in uptrends — survivorship bias |

### What Actually Works

| Approach | Result | Why |
|----------|--------|-----|
| **Stocks-only cross-sectional momentum** | **+214%, Sharpe 1.92** (raw) | Jegadeesh-Titman 1993 — most documented alpha in finance, 30+ years of evidence |
| **+ Sector caps + hold buffer** | **+122.7%, Sharpe 1.01** | Sector diversification trades some return for lower risk |
| **+ Volatility scaling (20%) + market state filter** | **+143.2%, Sharpe 1.17, DD 16.9%** | Barroso 2015 + Cooper 2004 — research-backed enhancements |

### Strategy Parameters (proven)

```
Universe:        46 US stocks (Yahoo cache, 1256+ daily bars each)
Momentum:        Vol-adjusted 6-1 month return (126d lookback, skip last 21d)
Selection:       Top 10 by score
Sizing:          Equal weight (10% each)
Rebalance:       Every 21 trading days
Hold buffer:     Don't sell unless rank drops below 15; don't buy unless enters top 8
Sector caps:     Tech:4, Finance:3, Healthcare:2, Consumer:2, Energy:2, Industrial:2, Auto:1
Vol scaling:     target_vol=0.20, scale = min(target_vol / realized_vol_126d, 2.0)
Market filter:   If equal-weight market 252d return < 0 → reduce to 25% exposure
```

### Key Technical Decisions

**Storage format: SQLite (not JSON)**
- 4M articles need fast querying by date/ticker/sentiment
- Resumable writes with INSERT OR IGNORE
- 4 parallel processes → separate DBs, zero conflict
- Aggregation via SQL GROUP BY (no memory issues)
- Migration to PostgreSQL is easy if we scale later

**LLM for news: Qwen3.5-35B via vLLM (not DistilRoBERTa)**
- DistilRoBERTa: fast (185/sec) but only gives sentiment score — no tickers, events, importance
- Qwen3.5-35B: slower (19/sec) but gives structured output — tickers, sentiment, event_type, importance
- Quality matters more than speed for training data
- Completions API with JSON prefilling bypasses thinking chain

**Qwen prompt that works:**
```
System: You classify news articles for market impact analysis.
Output ONE JSON object. Choose sentiment from: BULLISH, BEARISH, NEUTRAL.
Choose event_type from: earnings, merger, product, fda, analyst, regulation,
layoff, macro, geopolitical, tech, energy, consumer, crime, weather, political, social, general.
Set importance 1-10 where 1=irrelevant to markets, 5=moderate, 8=significant, 10=market-moving.
For tickers, list stock symbols directly affected. Empty array if none.

User: {headline}
Assistant: {"sentiment":"   ← JSON prefill forces structured output
```

**4 parallel Qwen chunks (not 1 process):**
- vLLM handles concurrent requests via batching
- 4 processes × 16 workers = 64 concurrent requests
- Each chunk writes to its own DB — no race conditions
- Combined throughput: ~19/sec vs ~8/sec single process
- GPU utilization: 85-95%

### Research Findings (applied)

| Paper | Finding | How We Use It |
|-------|---------|---------------|
| Barroso & Santa-Clara 2015 | Vol scaling doubles Sharpe, halves drawdown | `scale = target_vol / realized_vol` in portfolio.py |
| Cooper et al. 2004 | Momentum works in UP markets (+0.93%/mo), fails in DOWN (-0.37%/mo) | Market state filter: reduce to 25% when S&P 252d return < 0 |
| George & Hwang 2004 | 52-week high momentum: 0.65%/mo vs 0.38%/mo traditional | Tested — improves return but not Sharpe in our data |
| Tetlock 2008 | Negative news language predicts low earnings + returns | Use as FILTER for momentum, not standalone |
| Da et al. 2014 | Continuous movers: +5.94%, discrete jumpers: -2.07% | Filter for gradual momentum, article count as proxy |
| De Prado 2018 | Need 50:1 samples/features for financial ML, max_depth 3-4 | Why our ML failed — 940 samples with 114 features |
| AlphaAgent (KDD 2025) | LLM generates alpha factors, 11% excess return | Use LLM to discover factors, not predict direction |
| Alpha decay research | 5.6%/year decay in US markets, half-life ~12 months | Need continuous alpha discovery pipeline |

### De Prado ML Constraints (why ML failed)

- **Samples/features ratio:** Need ≥20:1 (preferably 50:1). We had 940/114 = 8.2:1.
- **Train accuracy 100% = memorization.** Healthy financial model: 55-65% train, 52-55% test.
- **max_depth 3-4** for <2K samples. We used 6-8 initially → total overfit.
- **Early stopping breaks** on tiny noisy validation sets. Use fixed tree count + regularization instead.
- **Feature selection mandatory:** MDA found <5 features with significant OOS power.
- **Binary UP/DOWN is the wrong target.** Triple barrier (de Prado) or momentum reversal prediction is better.

### News Analysis Quality Issues (and fixes)

**Bad prompt (killed after 810K articles):**
- 66% BULLISH (everything classified positive)
- Event types wrong ("News", "Corporate" instead of earnings/merger/etc.)
- Tickers garbage ("MISSING", "B", "ROSEN" — matched non-stock words)
- Importance inflated (avg 7.0, should be ~4)

**Fixed prompt (current, running):**
- Sentiment balanced: ~55% NEUTRAL, ~25% BULLISH, ~20% BEARISH
- Event types match schema: general, macro, political, earnings, etc.
- Importance calibrated: avg 4.0-4.5
- Tickers are real stock symbols when detected

### What News CAN and CANNOT Do

**CANNOT:** Predict price direction (confirmed 5 separate times — Run 1, Signal Validation, SSD backtests, Qwen 4-day analysis, DistilRoBERTa correlation tests)

**CAN:**
- Event DETECTION: know when earnings/M&A/FDA happened (useful for filtering)
- Risk management: cluster of negative events → reduce exposure
- Conviction sizing: momentum + positive news → increase position
- Market-wide sentiment: aggregate bearish spike → defensive mode
- Magnitude estimation: high-importance events → bigger moves

### Hardware Setup

```
2x DGX GB10 (Blackwell, aarch64)
- 128GB unified LPDDR5X each (256GB total)
- ConnectX-7 200Gbps RoCE RDMA (inter-node)
- vLLM serving Qwen3.5-35B-A3B-FP8 at ~19 articles/sec
- GPU utilization: 85-95% during processing
- Launch: spark-vllm-docker with --max-num-batched-tokens 16384

Data:
- /home/user/Backup-SSD/investment-monitor/historical_news.db (12.8 GB, 4.08M articles)
- /home/user/Backup-SSD/investment-monitor/forex_prices.db (30 GB, 28 symbols, 146M bars)
- forex_system/training/artifacts/yahoo_cache.db (73 symbols daily, 68 hourly)
- forex_system/training/artifacts/sentiment_cache_v2.db (3.8M scored)
- alpha_memory.db (740 rules, 908 reflections from Run 1)
```

### Upcoming (Plan)

**Phase 1 (done):** Volatility scaling + market state filter → Sharpe 1.17
**Phase 2 (after Qwen finishes ~Mar 19):** Aggregate news → daily features per stock → news filter for momentum
**Phase 3 (~Mar 20):** Vision chart analysis with Qwen2-VL
**Phase 4 (~Mar 21):** Production daily_runner.py + paper trading via cron
**Phase 5 (~Mar 25):** Pairs trading overlay + forex ML signals
**Phase 6 (~Apr 1):** Alpha discovery loop (LLM generates factors monthly)

---

### NautilusTrader Analysis (2026-03-19)

**Verdict: Don't switch frameworks. Extract useful patterns.**

NautilusTrader (21K stars, Rust core, 8 years old) is built for event-driven, high-frequency, per-instrument strategies. Our batch ranking of 46 stocks every 21 days is a completely different paradigm. Also: no Alpaca support (dealbreaker), still in Beta, single maintainer (95% of commits).

**What we extracted and tested:**

| Technique | Impact on our system |
|-----------|---------------------|
| **EfficiencyRatio (Kaufman)** | Sharpe 1.01 → 1.38 in isolation test, 1.18 in production combo |
| **KeltnerPosition** | Sharpe 1.01 → 1.37 in isolation test |
| **Pressure indicator** | No improvement alone, slight help in combos |
| VolumeSensitiveFillModel | Not backtested yet — for realistic fill simulation |
| TWAP execution | Not implemented yet — for live order splitting |
| SlidingWindowThrottler | Not needed yet — for Alpaca rate limiting |
| 3-state CircuitBreaker | Built, not deployed — ACTIVE/REDUCING/HALTED |
| FuzzyCandlesticks | Built — for LLM chart analysis token compression |
| SpreadAnalyzer | Built — filter stocks with wide spreads |

**Current best backtest result:**
```
+146.1%, Sharpe 1.18, Max DD 16.9%, $10K → $24,614
Components: vol-adjusted momentum + sector caps + hold buffer + vol scaling(20%) + market state filter + EfficiencyRatio filter
```

### Qwen Article Processing (2026-03-17 to 2026-03-19)

- Processing ALL 4,082,690 articles through Qwen3.5-35B via vLLM
- 4 parallel chunks, 16 workers each, ~19 articles/sec combined
- Output: `qwen_chunk_{1-4}.db` in SQLite (not JSON — faster queries, resumable)
- Fixed prompt after first attempt (old: 66% BULLISH everything; new: balanced sentiment, calibrated importance)
- Status: 95.5% complete, ~3 hours remaining

**Prompt that works (completions API with JSON prefilling):**
```
System: You classify news articles for market impact analysis...
User: {headline}
Assistant: {"sentiment":"   ← prefill forces JSON output, bypasses thinking chain
```

### Research Findings Applied (2026-03-18)

**From academic papers:**
- Barroso & Santa-Clara 2015 (vol scaling): +20% return, -30% drawdown → APPLIED
- Cooper et al. 2004 (market state filter): eliminates crash scenarios → APPLIED
- George & Hwang 2004 (52-week high): tested, didn't improve Sharpe in our data → REJECTED

**From NautilusTrader:**
- EfficiencyRatio: filters noisy stocks → APPLIED (+3% return improvement)
- KeltnerPosition: alternative ranking → available but not in production (would need more testing)

**From 2024-2026 research:**
- AlphaAgent (KDD 2025): LLM generates alpha factors → PLANNED for Phase 6
- Alpha decay 5.6%/year → need continuous discovery pipeline → PLANNED
- MarketSenseAI: LLM stock selection → PLANNED after Qwen processing completes
- Forex ML Sharpe 2.84-3.91 → use SSD forex data as context features → PLANNED

### Storage Decision: SQLite vs JSON

SQLite chosen over JSON for 4M article analysis storage because:
- Fast queries: `WHERE ticker='NVDA' AND importance >= 8` → instant
- Resumable: `INSERT OR IGNORE` → restart anytime
- 4 parallel writers → separate DBs, zero conflict
- Aggregation: `GROUP BY date, ticker` in SQL → no memory issues
- **Format has ZERO impact on model results** — same data either way
- Can export to JSON/JSONL anytime if needed for portability

---

## Rules for Future Sessions

1. **Read this file first** — don't repeat experiments that already failed
2. **News cannot predict direction** — use it for filtering/risk/events, not BUY/SELL signals
3. **ML needs 50:1 samples/features** — don't train with <20:1 ratio
4. **Always compare to buy-and-hold** — if it doesn't beat B&H, it doesn't work
5. **Volatility scaling is mandatory** — never run momentum without it
6. **Market state filter is mandatory** — reduce exposure in bear markets
7. **Test one enhancement at a time** — compare to baseline, keep what improves Sharpe
8. **SQLite for data storage** — not JSON, not CSV. Migrate to PostgreSQL only if multi-machine
9. **vLLM completions API with JSON prefilling** — bypasses thinking chain, 20x faster than chat API
10. **4 parallel Qwen chunks** — separate DBs, 64 concurrent requests, ~19/sec combined
