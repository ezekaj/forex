"""Training configuration for the Universal Investment AI."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetConfig:
    symbol: str
    asset_class: str      # stock, crypto, forex, commodity, index
    name: str
    keywords: tuple[str, ...]
    sector: str = ""
    data_source: str = "yahoo"  # yahoo, ssd_db


# ── Asset Registry (94 assets) ──

STOCKS = {
    # Tech
    "AAPL": AssetConfig("AAPL", "stock", "Apple", ("Apple", "AAPL", "iPhone", "Tim Cook", "iOS"), "tech"),
    "MSFT": AssetConfig("MSFT", "stock", "Microsoft", ("Microsoft", "MSFT", "Azure", "Satya Nadella", "Windows"), "tech"),
    "NVDA": AssetConfig("NVDA", "stock", "NVIDIA", ("NVIDIA", "NVDA", "GPU", "Jensen Huang", "AI chip"), "tech"),
    "GOOGL": AssetConfig("GOOGL", "stock", "Alphabet", ("Google", "Alphabet", "GOOGL", "Gemini", "YouTube"), "tech"),
    "AMZN": AssetConfig("AMZN", "stock", "Amazon", ("Amazon", "AMZN", "AWS", "Prime"), "tech"),
    "META": AssetConfig("META", "stock", "Meta", ("Meta", "Facebook", "META", "Zuckerberg", "Instagram", "WhatsApp"), "tech"),
    "TSLA": AssetConfig("TSLA", "stock", "Tesla", ("Tesla", "TSLA", "Elon Musk", "EV", "electric vehicle"), "auto"),
    "AVGO": AssetConfig("AVGO", "stock", "Broadcom", ("Broadcom", "AVGO", "VMware"), "tech"),
    "ORCL": AssetConfig("ORCL", "stock", "Oracle", ("Oracle", "ORCL", "cloud"), "tech"),
    "CRM": AssetConfig("CRM", "stock", "Salesforce", ("Salesforce", "CRM"), "tech"),
    "AMD": AssetConfig("AMD", "stock", "AMD", ("AMD", "Advanced Micro", "Lisa Su", "Ryzen"), "tech"),
    "NFLX": AssetConfig("NFLX", "stock", "Netflix", ("Netflix", "NFLX", "streaming"), "tech"),
    "INTC": AssetConfig("INTC", "stock", "Intel", ("Intel", "INTC", "semiconductor"), "tech"),
    # Finance
    "JPM": AssetConfig("JPM", "stock", "JPMorgan", ("JPMorgan", "JPM", "Dimon", "JP Morgan"), "finance"),
    "GS": AssetConfig("GS", "stock", "Goldman Sachs", ("Goldman Sachs", "GS", "Goldman"), "finance"),
    "MS": AssetConfig("MS", "stock", "Morgan Stanley", ("Morgan Stanley", "MS"), "finance"),
    "BAC": AssetConfig("BAC", "stock", "Bank of America", ("Bank of America", "BAC", "BofA"), "finance"),
    "V": AssetConfig("V", "stock", "Visa", ("Visa",), "finance"),
    "MA": AssetConfig("MA", "stock", "Mastercard", ("Mastercard", "MA"), "finance"),
    # Healthcare
    "JNJ": AssetConfig("JNJ", "stock", "Johnson & Johnson", ("Johnson", "JNJ", "J&J"), "healthcare"),
    "UNH": AssetConfig("UNH", "stock", "UnitedHealth", ("UnitedHealth", "UNH"), "healthcare"),
    "PFE": AssetConfig("PFE", "stock", "Pfizer", ("Pfizer", "PFE", "vaccine"), "healthcare"),
    "LLY": AssetConfig("LLY", "stock", "Eli Lilly", ("Eli Lilly", "LLY", "Mounjaro", "obesity"), "healthcare"),
    # Consumer
    "WMT": AssetConfig("WMT", "stock", "Walmart", ("Walmart", "WMT"), "consumer"),
    "KO": AssetConfig("KO", "stock", "Coca-Cola", ("Coca-Cola", "KO", "Coke"), "consumer"),
    "PG": AssetConfig("PG", "stock", "Procter & Gamble", ("Procter", "P&G", "PG"), "consumer"),
    "DIS": AssetConfig("DIS", "stock", "Disney", ("Disney", "DIS", "Disney+"), "consumer"),
    "NKE": AssetConfig("NKE", "stock", "Nike", ("Nike", "NKE"), "consumer"),
    "MCD": AssetConfig("MCD", "stock", "McDonald's", ("McDonald", "MCD"), "consumer"),
    # Energy
    "XOM": AssetConfig("XOM", "stock", "ExxonMobil", ("Exxon", "XOM", "ExxonMobil"), "energy"),
    "CVX": AssetConfig("CVX", "stock", "Chevron", ("Chevron", "CVX"), "energy"),
    # Industrial
    "BA": AssetConfig("BA", "stock", "Boeing", ("Boeing", "BA", "737", "787"), "industrial"),
    "CAT": AssetConfig("CAT", "stock", "Caterpillar", ("Caterpillar", "CAT"), "industrial"),
    "GE": AssetConfig("GE", "stock", "GE Aerospace", ("GE", "General Electric"), "industrial"),
    # Other mega caps
    "BRK-B": AssetConfig("BRK-B", "stock", "Berkshire Hathaway", ("Berkshire", "BRK", "Buffett"), "finance"),
    "COST": AssetConfig("COST", "stock", "Costco", ("Costco", "COST"), "consumer"),
    "ABBV": AssetConfig("ABBV", "stock", "AbbVie", ("AbbVie", "ABBV"), "healthcare"),
    "PLTR": AssetConfig("PLTR", "stock", "Palantir", ("Palantir", "PLTR"), "tech"),
    "COIN": AssetConfig("COIN", "stock", "Coinbase", ("Coinbase", "COIN"), "finance"),
    "HOOD": AssetConfig("HOOD", "stock", "Robinhood", ("Robinhood", "HOOD"), "finance"),
    "NIO": AssetConfig("NIO", "stock", "NIO", ("NIO", "Chinese EV"), "auto"),
    "SOFI": AssetConfig("SOFI", "stock", "SoFi", ("SoFi", "SOFI"), "finance"),
    "RIVN": AssetConfig("RIVN", "stock", "Rivian", ("Rivian", "RIVN"), "auto"),
    "F": AssetConfig("F", "stock", "Ford", ("Ford", "F150"), "auto"),
    "GM": AssetConfig("GM", "stock", "General Motors", ("General Motors", "GM"), "auto"),
    "TGT": AssetConfig("TGT", "stock", "Target", ("Target", "TGT"), "consumer"),
    "SHOP": AssetConfig("SHOP", "stock", "Shopify", ("Shopify", "SHOP"), "tech"),
    "SQ": AssetConfig("SQ", "stock", "Block", ("Block", "Square", "SQ"), "finance"),
    "UBER": AssetConfig("UBER", "stock", "Uber", ("Uber", "UBER", "rideshare"), "tech"),
    "SNAP": AssetConfig("SNAP", "stock", "Snap", ("Snap", "Snapchat", "SNAP"), "tech"),
}

CRYPTO = {
    "BTC-USD": AssetConfig("BTC-USD", "crypto", "Bitcoin", ("Bitcoin", "BTC", "crypto", "halving", "Satoshi"), "crypto"),
    "ETH-USD": AssetConfig("ETH-USD", "crypto", "Ethereum", ("Ethereum", "ETH", "smart contract", "Vitalik"), "crypto"),
    "SOL-USD": AssetConfig("SOL-USD", "crypto", "Solana", ("Solana", "SOL"), "crypto"),
    "XRP-USD": AssetConfig("XRP-USD", "crypto", "Ripple", ("Ripple", "XRP", "SEC lawsuit"), "crypto"),
    "ADA-USD": AssetConfig("ADA-USD", "crypto", "Cardano", ("Cardano", "ADA"), "crypto"),
    "DOGE-USD": AssetConfig("DOGE-USD", "crypto", "Dogecoin", ("Dogecoin", "DOGE", "meme coin"), "crypto"),
    "AVAX-USD": AssetConfig("AVAX-USD", "crypto", "Avalanche", ("Avalanche", "AVAX"), "crypto"),
    "LINK-USD": AssetConfig("LINK-USD", "crypto", "Chainlink", ("Chainlink", "LINK", "oracle"), "crypto"),
    "DOT-USD": AssetConfig("DOT-USD", "crypto", "Polkadot", ("Polkadot", "DOT"), "crypto"),
    "MATIC-USD": AssetConfig("MATIC-USD", "crypto", "Polygon", ("Polygon", "MATIC", "POL"), "crypto"),
    "UNI-USD": AssetConfig("UNI-USD", "crypto", "Uniswap", ("Uniswap", "UNI", "DEX"), "crypto"),
    "ATOM-USD": AssetConfig("ATOM-USD", "crypto", "Cosmos", ("Cosmos", "ATOM"), "crypto"),
    "FIL-USD": AssetConfig("FIL-USD", "crypto", "Filecoin", ("Filecoin", "FIL"), "crypto"),
    "APT-USD": AssetConfig("APT-USD", "crypto", "Aptos", ("Aptos", "APT"), "crypto"),
    "ARB-USD": AssetConfig("ARB-USD", "crypto", "Arbitrum", ("Arbitrum", "ARB"), "crypto"),
    "OP-USD": AssetConfig("OP-USD", "crypto", "Optimism", ("Optimism", "OP"), "crypto"),
    "NEAR-USD": AssetConfig("NEAR-USD", "crypto", "NEAR Protocol", ("NEAR",), "crypto"),
    "SUI-USD": AssetConfig("SUI-USD", "crypto", "Sui", ("Sui", "SUI"), "crypto"),
    "PEPE-USD": AssetConfig("PEPE-USD", "crypto", "Pepe", ("Pepe", "PEPE", "meme"), "crypto"),
    "RENDER-USD": AssetConfig("RENDER-USD", "crypto", "Render", ("Render", "RNDR"), "crypto"),
}

FOREX = {
    "EURUSD": AssetConfig("EURUSD", "forex", "EUR/USD", ("EUR", "USD", "euro", "dollar", "ECB", "Fed"), data_source="ssd_db"),
    "GBPUSD": AssetConfig("GBPUSD", "forex", "GBP/USD", ("GBP", "pound", "sterling", "BOE", "Bank of England"), data_source="ssd_db"),
    "USDJPY": AssetConfig("USDJPY", "forex", "USD/JPY", ("JPY", "yen", "BOJ", "Bank of Japan"), data_source="ssd_db"),
    "AUDUSD": AssetConfig("AUDUSD", "forex", "AUD/USD", ("AUD", "Australian dollar", "RBA"), data_source="ssd_db"),
    "USDCAD": AssetConfig("USDCAD", "forex", "USD/CAD", ("CAD", "Canadian dollar", "Bank of Canada"), data_source="ssd_db"),
    "USDCHF": AssetConfig("USDCHF", "forex", "USD/CHF", ("CHF", "Swiss franc", "SNB"), data_source="ssd_db"),
    "NZDUSD": AssetConfig("NZDUSD", "forex", "NZD/USD", ("NZD", "New Zealand dollar", "RBNZ"), data_source="ssd_db"),
    "EURGBP": AssetConfig("EURGBP", "forex", "EUR/GBP", ("EURGBP",), data_source="ssd_db"),
    "EURJPY": AssetConfig("EURJPY", "forex", "EUR/JPY", ("EURJPY",), data_source="ssd_db"),
    "GBPJPY": AssetConfig("GBPJPY", "forex", "GBP/JPY", ("GBPJPY",), data_source="ssd_db"),
    "AUDJPY": AssetConfig("AUDJPY", "forex", "AUD/JPY", ("AUDJPY",), data_source="ssd_db"),
    "EURAUD": AssetConfig("EURAUD", "forex", "EUR/AUD", ("EURAUD",), data_source="ssd_db"),
    "GBPAUD": AssetConfig("GBPAUD", "forex", "GBP/AUD", ("GBPAUD",), data_source="ssd_db"),
}

COMMODITIES = {
    "XAUUSD": AssetConfig("XAUUSD", "commodity", "Gold", ("gold", "XAUUSD", "precious metal", "safe haven", "bullion"), data_source="ssd_db"),
    "XAGUSD": AssetConfig("XAGUSD", "commodity", "Silver", ("silver", "XAGUSD", "precious metal"), data_source="ssd_db"),
    "BRENTCMDUSD": AssetConfig("BRENTCMDUSD", "commodity", "Brent Crude", ("Brent", "crude oil", "OPEC", "oil price"), data_source="ssd_db"),
    "LIGHTCMDUSD": AssetConfig("LIGHTCMDUSD", "commodity", "WTI Crude", ("WTI", "crude oil", "oil"), data_source="ssd_db"),
    "GC=F": AssetConfig("GC=F", "commodity", "Gold Futures", ("gold futures", "GC", "COMEX gold")),
    "CL=F": AssetConfig("CL=F", "commodity", "Crude Oil Futures", ("crude futures", "CL", "NYMEX crude")),
}

INDICES = {
    "USA500IDXUSD": AssetConfig("USA500IDXUSD", "index", "S&P 500", ("S&P 500", "SPX", "SP500", "S&P"), data_source="ssd_db"),
    "USATECHIDXUSD": AssetConfig("USATECHIDXUSD", "index", "NASDAQ 100", ("NASDAQ", "QQQ", "tech index"), data_source="ssd_db"),
    "USA30IDXUSD": AssetConfig("USA30IDXUSD", "index", "Dow Jones", ("Dow Jones", "DJIA", "Dow"), data_source="ssd_db"),
    "^FTSE": AssetConfig("^FTSE", "index", "FTSE 100", ("FTSE", "London Stock Exchange", "UK market")),
    "^N225": AssetConfig("^N225", "index", "Nikkei 225", ("Nikkei", "Tokyo Stock Exchange", "Japan market")),
}

# Combined registry
ASSET_REGISTRY: dict[str, AssetConfig] = {**STOCKS, **CRYPTO, **FOREX, **COMMODITIES, **INDICES}


def get_symbols_by_class(asset_class: str) -> list[str]:
    return [s for s, a in ASSET_REGISTRY.items() if a.asset_class == asset_class]


def get_asset(symbol: str) -> AssetConfig:
    if symbol not in ASSET_REGISTRY:
        raise KeyError(f"Unknown symbol: {symbol}. Register it in config.py ASSET_REGISTRY.")
    return ASSET_REGISTRY[symbol]


def _detect_paths() -> dict[str, str]:
    """Auto-detect paths based on environment (host vs container)."""
    import os
    if os.path.exists("/data/ssd/forex_prices.db"):
        # Inside Docker container
        return {
            "ssd": "/data/ssd",
            "project": "/workspace/forex",
        }
    else:
        # Host machine
        return {
            "ssd": "/home/user/Backup-SSD",
            "project": "/home/user/forex",
        }

_paths = _detect_paths()
_SSD = _paths["ssd"]
_PROJECT = _paths["project"]


@dataclass(frozen=True)
class TrainingConfig:
    # ── Read-only data paths (SSD) ──
    PRICE_DB_PATH: str = f"{_SSD}/investment-monitor/forex_prices.db"
    NEWS_DB_PATH: str = f"{_SSD}/investment-monitor/historical_news.db"

    # ── Writable data paths (project directory) ──
    # These are COPIES of the SSD data, made writable for the learning loop.
    # On first run, copy from SSD: cp /data/ssd/investment-monitor/data/alpha_memory.db {_PROJECT}/
    ALPHA_MEMORY_DB_PATH: str = f"{_PROJECT}/alpha_memory.db"
    PREDICTIONS_DB_PATH: str = f"{_PROJECT}/coordinator_predictions.db"
    YAHOO_CACHE_DB_PATH: str = f"{_PROJECT}/forex_system/training/artifacts/yahoo_cache.db"
    MODEL_OUTPUT_DIR: str = f"{_PROJECT}/forex_system/training/artifacts"
    EXPERIMENT_DB_PATH: str = f"{_PROJECT}/forex_system/training/experiments.db"
    PAPER_TRADES_DB_PATH: str = f"{_PROJECT}/paper_trades.db"
    SENTIMENT_CACHE_DB_PATH: str = f"{_PROJECT}/forex_system/training/artifacts/sentiment_cache_v2.db"

    # ── Labeling ──
    LOOKAHEAD_BARS_DAILY: int = 5
    LOOKAHEAD_BARS_HOURLY: int = 24
    BUY_THRESHOLD: float = 0.003
    SELL_THRESHOLD: float = -0.003
    TRIPLE_BARRIER_PT_SL_RATIO: float = 2.0
    TRIPLE_BARRIER_MAX_HOLDING: int = 10
    TRIPLE_BARRIER_ATR_MULTIPLIER: float = 1.5

    # ── Walk-forward ──
    TRAIN_WINDOW_DAYS: int = 252
    TEST_WINDOW_DAYS: int = 42
    PURGE_DAYS: int = 5
    EMBARGO_DAYS: int = 2

    # ── News ──
    NEWS_FORWARD_WINDOWS: tuple = (2, 3, 5)
    NEGATIVE_NEWS_WEIGHT: float = 1.5

    # ── Model defaults ──
    RANDOM_SEED: int = 42
    XGBOOST_N_ESTIMATORS: int = 300
    XGBOOST_MAX_DEPTH: int = 8
    XGBOOST_LEARNING_RATE: float = 0.05
    XGBOOST_SUBSAMPLE: float = 0.8
    XGBOOST_COLSAMPLE_BYTREE: float = 0.8

    # ── Phase gate criteria ──
    MIN_WIN_RATE: float = 0.53
    MIN_SHARPE: float = 1.0
    MIN_PROFIT_FACTOR: float = 1.2
    MAX_DRAWDOWN_PCT: float = 0.15
    MIN_TRADES_PER_FOLD: int = 50

    # ── Cluster ──
    NODE1_IP: str = "10.0.0.1"
    NODE2_IP: str = "10.0.0.2"
    RAY_HEAD_ADDRESS: str = "10.0.0.1:6379"
    VLLM_ENDPOINT: str = "http://localhost:8000"
    NUM_GPUS: int = 2
    NUM_CPUS: int = 40

    @property
    def model_output_path(self) -> Path:
        path = Path(self.MODEL_OUTPUT_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path
