import MetaTrader5 as mt5

# Initialize and login
mt5.initialize()
mt5.login(95948709, 'To-4KyLg', 'MetaQuotes-Demo')

# Get all positions
positions = mt5.positions_get()

if positions:
    print(f"Found {len(positions)} open positions")
    print("="*50)
    
    for position in positions:
        symbol = position.symbol
        ticket = position.ticket
        volume = position.volume
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            continue
        
        # Determine close price based on position type
        if position.type == 0:  # Buy position
            close_price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
        else:  # Sell position
            close_price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY
        
        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": close_price,
            "deviation": 20,
            "magic": 0,
            "comment": "close_all",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Send close order
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"CLOSED: {symbol} #{ticket} | P&L: ${position.profit:.2f}")
        else:
            print(f"FAILED to close {symbol}: {result.comment}")
    
    print("="*50)
    
    # Check final balance
    account = mt5.account_info()
    print(f"Final Balance: ${account.balance:.2f}")
    print(f"Final Equity: ${account.equity:.2f}")
else:
    print("No open positions")

mt5.shutdown()