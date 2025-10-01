"""
Close all open positions
"""
import MetaTrader5 as mt5

mt5.initialize()
mt5.login(95948709, 'To-4KyLg', 'MetaQuotes-Demo')

positions = mt5.positions_get()
if positions:
    print(f"Closing {len(positions)} positions...")
    for pos in positions:
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick:
            close_price = tick.bid if pos.type == 0 else tick.ask
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': pos.symbol,
                'volume': pos.volume,
                'type': mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                'position': pos.ticket,
                'price': close_price,
                'deviation': 20,
                'magic': pos.magic,
                'comment': 'close_all',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_FOK
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"  Closed: {pos.symbol} {'BUY' if pos.type == 0 else 'SELL'}")
    print("All positions closed!")
else:
    print("No positions to close")

mt5.shutdown()