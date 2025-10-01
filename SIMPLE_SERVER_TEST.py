"""
Simple MT5 Server Test - Find your correct server
================================================
"""

import MetaTrader5 as mt5

def test_servers():
    """Test different servers"""
    
    LOGIN = 95948709
    PASSWORD = 'To-4KyLg'
    
    # Common server variations
    servers = [
        'MetaQuotes-Demo',
        'XMGlobal-MT5 3', 
        'XMGlobal-Demo',
        'XM-Demo',
        'XMGlobal-MT5-3',
        'XM Global Limited-Demo'
    ]
    
    print("Testing MT5 Servers...")
    print(f"Login: {LOGIN}")
    print("-" * 40)
    
    for server in servers:
        print(f"\nTesting: {server}")
        
        if not mt5.initialize():
            print("  Failed to initialize MT5")
            continue
        
        success = mt5.login(LOGIN, password=PASSWORD, server=server)
        
        if success:
            account_info = mt5.account_info()
            if account_info:
                print(f"  SUCCESS!")
                print(f"  Account: {account_info.login}")
                print(f"  Server: {account_info.server}")
                print(f"  Balance: ${account_info.balance:.2f}")
                
                mt5.shutdown()
                print(f"\nCORRECT SERVER: {server}")
                return server
            else:
                print("  Login ok but no account info")
        else:
            error = mt5.last_error()
            print(f"  Failed (Error: {error})")
        
        mt5.shutdown()
    
    print("\nNo working server found!")
    return None

def check_current_connection():
    """Check if MT5 is already connected"""
    
    print("Checking current MT5 connection...")
    
    if not mt5.initialize():
        print("MT5 not initialized")
        return None
    
    account_info = mt5.account_info()
    if account_info:
        print("MT5 already connected!")
        print(f"Server: {account_info.server}")
        print(f"Account: {account_info.login}")
        print(f"Balance: ${account_info.balance:.2f}")
        
        server = account_info.server
        mt5.shutdown()
        return server
    else:
        print("MT5 not connected")
        mt5.shutdown()
        return None

if __name__ == "__main__":
    
    # First check current connection
    current_server = check_current_connection()
    
    if current_server:
        print(f"\nYour server is: {current_server}")
    else:
        print("\nTesting server combinations...")
        found_server = test_servers()
        
        if found_server:
            print(f"\nUse this server: {found_server}")
        else:
            print("\nManually check your MT5 terminal for server name")