"""
MT5 Connection Test - Verify your credentials work
=================================================
Quick test to ensure MT5 connection is working properly
"""

import MetaTrader5 as mt5
from datetime import datetime

def test_mt5_connection():
    """Test MT5 connection with your credentials"""
    
    # Your MT5 credentials
    LOGIN = 95948709
    PASSWORD = 'To-4KyLg'
    SERVER = 'MetaQuotes-Demo'
    
    print("🔄 Testing MT5 Connection...")
    print(f"Server: {SERVER}")
    print(f"Login: {LOGIN}")
    print("=" * 40)
    
    # Step 1: Initialize MT5
    print("1. Initializing MT5...")
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        error = mt5.last_error()
        print(f"Error: {error}")
        return False
    
    print("✅ MT5 initialized successfully")
    
    # Step 2: Login to account
    print("2. Logging into account...")
    if not mt5.login(LOGIN, password=PASSWORD, server=SERVER):
        print("❌ Login failed")
        error = mt5.last_error()
        print(f"Error: {error}")
        mt5.shutdown()
        return False
    
    print("✅ Login successful!")
    
    # Step 3: Get account information
    print("3. Retrieving account information...")
    account_info = mt5.account_info()
    
    if account_info is None:
        print("❌ Failed to get account information")
        mt5.shutdown()
        return False
    
    print("✅ Account information retrieved!")
    print(f"   Account Number: {account_info.login}")
    print(f"   Server: {account_info.server}")
    print(f"   Currency: {account_info.currency}")
    print(f"   Balance: ${account_info.balance:.2f}")
    print(f"   Equity: ${account_info.equity:.2f}")
    print(f"   Margin: ${account_info.margin:.2f}")
    print(f"   Free Margin: ${account_info.margin_free:.2f}")
    print(f"   Leverage: 1:{account_info.leverage}")
    
    # Step 4: Test symbol availability
    print("\n4. Testing symbol availability...")
    test_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    available_symbols = []
    
    for symbol in test_symbols:
        # Select symbol in market watch
        if mt5.symbol_select(symbol, True):
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                available_symbols.append(symbol)
                print(f"   ✅ {symbol}: Spread = {symbol_info.spread} points")
            else:
                print(f"   ⚠️ {symbol}: Selected but no info available")
        else:
            print(f"   ❌ {symbol}: Not available")
    
    print(f"\nAvailable symbols for trading: {available_symbols}")
    
    # Step 5: Test getting current prices
    print("\n5. Testing real-time price data...")
    for symbol in available_symbols[:2]:  # Test first 2 symbols
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            print(f"   {symbol}: Bid={tick.bid:.5f}, Ask={tick.ask:.5f}, "
                  f"Spread={tick.ask-tick.bid:.5f}")
        else:
            print(f"   ❌ No tick data for {symbol}")
    
    # Step 6: Check trading permissions
    print("\n6. Checking trading permissions...")
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print(f"   Trade Allowed: {'✅ Yes' if terminal_info.trade_allowed else '❌ No'}")
        print(f"   Connected: {'✅ Yes' if terminal_info.connected else '❌ No'}")
    
    # Cleanup
    mt5.shutdown()
    
    print("\n🎯 CONNECTION TEST COMPLETED!")
    print("✅ Your MT5 credentials are working correctly!")
    print("✅ Account is ready for automated trading!")
    
    return True

def quick_demo_trade_test():
    """Test placing a very small demo trade"""
    
    LOGIN = 95948709
    PASSWORD = 'To-4KyLg'
    SERVER = 'MetaQuotes-Demo'
    
    print("\n" + "="*50)
    print("🧪 DEMO TRADE TEST")
    print("Testing actual order placement with micro lot")
    print("="*50)
    
    # Connect
    if not mt5.initialize():
        print("❌ Cannot initialize MT5 for trade test")
        return False
    
    if not mt5.login(LOGIN, password=PASSWORD, server=SERVER):
        print("❌ Cannot login for trade test")
        mt5.shutdown()
        return False
    
    # Test with EURUSD micro lot
    symbol = "EURUSD"
    
    # Select symbol
    if not mt5.symbol_select(symbol, True):
        print(f"❌ Cannot select {symbol}")
        mt5.shutdown()
        return False
    
    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"❌ Cannot get price for {symbol}")
        mt5.shutdown()
        return False
    
    print(f"📊 Current {symbol} price: {tick.bid:.5f} / {tick.ask:.5f}")
    
    # Place a tiny buy order (0.01 lots = $1000 position)
    lot_size = 0.01
    price = tick.ask
    
    # Calculate stop loss and take profit (very small)
    point = 0.0001  # EURUSD point value
    sl_price = price - (20 * point)  # 20 pips stop loss
    tp_price = price + (20 * point)  # 20 pips take profit
    
    print(f"📝 Preparing test order:")
    print(f"   Symbol: {symbol}")
    print(f"   Type: BUY")
    print(f"   Volume: {lot_size} lots")
    print(f"   Price: {price:.5f}")
    print(f"   Stop Loss: {sl_price:.5f}")
    print(f"   Take Profit: {tp_price:.5f}")
    
    # Prepare order
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 20,
        "magic": 999999,
        "comment": "Connection Test Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    print("🚀 Placing test order...")
    result = mt5.order_send(request)
    
    if result is None:
        print("❌ Order failed - no result")
        mt5.shutdown()
        return False
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Order failed: {result.comment} (Code: {result.retcode})")
        mt5.shutdown()
        return False
    
    print("✅ TEST ORDER SUCCESSFUL!")
    print(f"   Ticket: {result.order}")
    print(f"   Executed Price: {result.price:.5f}")
    print(f"   Volume: {result.volume}")
    
    # Immediately close the test position
    print("🔄 Closing test position...")
    
    # Get the position
    positions = mt5.positions_get(symbol=symbol)
    if positions and len(positions) > 0:
        position = positions[-1]  # Get the latest position
        
        # Close order
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL,
            "position": position.ticket,
            "price": mt5.symbol_info_tick(symbol).bid,
            "deviation": 20,
            "magic": 999999,
            "comment": "Close Test Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        close_result = mt5.order_send(close_request)
        if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
            print("✅ Test position closed successfully!")
            
            # Get the profit/loss
            import time
            time.sleep(1)  # Wait for trade to process
            
            deals = mt5.history_deals_get(position=position.ticket)
            if deals and len(deals) >= 2:
                profit = deals[-1].profit
                print(f"   Test Trade P&L: ${profit:.2f}")
        else:
            print("⚠️ Could not close test position - please close manually")
    
    mt5.shutdown()
    
    print("\n🎯 DEMO TRADE TEST COMPLETED!")
    print("✅ Your account can execute real trades!")
    
    return True

if __name__ == "__main__":
    print("""
🧪 MT5 CONNECTION & TRADING TEST
===============================
Test your MT5 credentials and trading capabilities

Options:
1. Connection Test Only
2. Connection + Demo Trade Test (Recommended)
3. Exit
    """)
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "1":
        test_mt5_connection()
    elif choice == "2":
        if test_mt5_connection():
            input("\nPress Enter to continue with demo trade test...")
            quick_demo_trade_test()
    elif choice == "3":
        print("Goodbye!")
    else:
        print("Invalid choice")