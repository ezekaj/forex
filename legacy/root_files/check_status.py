import MetaTrader5 as mt5
import os

mt5.initialize()
mt5.login(95948709, 'To-4KyLg', 'MetaQuotes-Demo')

account = mt5.account_info()
positions = mt5.positions_get()

print('='*50)
print('ACCOUNT STATUS')
print('='*50)
print(f'Balance: ${account.balance:.2f}')
print(f'Equity: ${account.equity:.2f}')
print(f'Profit/Loss: ${account.profit:.2f}')

if positions:
    print(f'\nOPEN POSITIONS: {len(positions)}')
    print('-'*50)
    for pos in positions:
        direction = 'BUY' if pos.type == 0 else 'SELL'
        pnl = '+' if pos.profit > 0 else ''
        print(f'{pos.symbol} | {direction} | Volume: {pos.volume} | P&L: {pnl}${pos.profit:.2f}')
        print(f'  Entry: {pos.price_open:.5f} | Current: {pos.price_current:.5f}')
        print(f'  SL: {pos.sl:.5f} | TP: {pos.tp:.5f}')
        print(f'  Magic: {pos.magic} | Comment: {pos.comment}')
else:
    print('\nNo open positions')

print('='*50)

# Check if trade history file was created
if os.path.exists('trade_history.json'):
    print('\nTrade history file exists - bot is learning!')
    import json
    with open('trade_history.json', 'r') as f:
        data = json.load(f)
        print(f"Historical trades recorded: {len(data.get('trades', []))}")
        if data.get('patterns'):
            print(f"Patterns being tracked: {len(data['patterns'])}")
    
mt5.shutdown()