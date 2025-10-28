import MetaTrader5 as mt5

# Initialize and login
mt5.initialize()
mt5.login(95948709, 'To-4KyLg', 'MetaQuotes-Demo')

# Get all symbols
symbols = mt5.symbols_get()
print(f"Total symbols available: {len(symbols)}")
print("="*60)

# Categorize by group
forex = []
crypto = []
indices = []
stocks = []
commodities = []
others = []

for symbol in symbols:
    name = symbol.name
    path = symbol.path if symbol.path else ""
    
    # Only visible/tradeable symbols
    if symbol.visible:
        if "Forex" in path or any(cur in name for cur in ["EUR", "USD", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"]):
            if len(name) <= 10:  # Likely forex pairs
                forex.append(name)
        elif "Crypto" in path or any(crypto in name for crypto in ["BTC", "ETH", "LTC", "XRP", "DOGE"]):
            crypto.append(name)
        elif any(idx in name for idx in ["US30", "US100", "US500", "DAX", "FTSE", "ASX"]):
            indices.append(name)
        elif "Stock" in path or "Shares" in path:
            stocks.append(name)
        elif any(comm in name for comm in ["GOLD", "SILVER", "OIL", "GAS", "XAUUSD", "XAGUSD"]):
            commodities.append(name)
        else:
            others.append(name)

print(f"FOREX PAIRS: {len(forex)}")
if forex:
    print(f"Examples: {', '.join(forex[:10])}")
print()

print(f"CRYPTO: {len(crypto)}")  
if crypto:
    print(f"Examples: {', '.join(crypto[:5])}")
print()

print(f"INDICES: {len(indices)}")
if indices:
    print(f"Examples: {', '.join(indices[:5])}")
print()

print(f"COMMODITIES: {len(commodities)}")
if commodities:
    print(f"Examples: {', '.join(commodities[:5])}")
print()

print(f"STOCKS: {len(stocks)}")
if stocks:
    print(f"Examples: {', '.join(stocks[:10])}")
print()

# Find most liquid/popular symbols by checking spreads
print("="*60)
print("CHECKING BEST SYMBOLS BY SPREAD...")
print("="*60)

best_symbols = []
for symbol in forex[:20] + crypto[:5] + commodities[:5]:  # Check top symbols
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick and info and info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
        if tick.bid > 0 and tick.ask > 0:
            spread_points = tick.ask - tick.bid
            spread_pips = spread_points / info.point / 10 if info.point > 0 else 999
            best_symbols.append((symbol, spread_pips))

# Sort by lowest spread
best_symbols.sort(key=lambda x: x[1])

print("BEST SYMBOLS TO TRADE (by spread):")
for sym, spread in best_symbols[:15]:
    print(f"  {sym}: {spread:.1f} pips")

mt5.shutdown()