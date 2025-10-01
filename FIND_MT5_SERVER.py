"""
Find Correct MT5 Server - Auto-detect your server
=================================================
Try multiple server combinations to find the right one
"""

import MetaTrader5 as mt5

def test_mt5_servers():
    """Test different server combinations to find the correct one"""
    
    # Your credentials
    LOGIN = 95948709
    PASSWORD = 'To-4KyLg'
    
    # Possible server names to try
    possible_servers = [
        'MetaQuotes-Demo',
        'XMGlobal-MT5 3',
        'XMGlobal-Demo',
        'XM-Demo',
        'XM-MT5',
        'XMGlobal-MT5-3',
        'XMGlobal-Demo-3',
        'MetaQuotes-Demo-3',
        'XM-Demo-EU3',
        'XMGlobal-EU3',
        'XM Global Limited-Demo',
        'XM Global-Demo'
    ]
    
    print("🔍 SEARCHING FOR CORRECT MT5 SERVER")
    print(f"Login: {LOGIN}")
    print("=" * 50)
    
    successful_servers = []
    
    for server in possible_servers:
        print(f"\n🔄 Testing server: {server}")
        
        # Initialize MT5
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            continue
        
        # Try to login
        success = mt5.login(LOGIN, password=PASSWORD, server=server)
        
        if success:
            # Get account info to verify connection
            account_info = mt5.account_info()
            if account_info:
                print(f"✅ SUCCESS! Server: {server}")
                print(f"   Account: {account_info.login}")
                print(f"   Balance: ${account_info.balance:.2f}")
                print(f"   Server: {account_info.server}")
                print(f"   Currency: {account_info.currency}")
                
                successful_servers.append({
                    'server': server,
                    'account_server': account_info.server,
                    'balance': account_info.balance,
                    'login': account_info.login
                })
            else:
                print(f"⚠️ Login successful but no account info for: {server}")
        else:
            error = mt5.last_error()
            print(f"❌ Failed: {server} (Error: {error})")
        
        # Shutdown before next attempt
        mt5.shutdown()
    
    print("\n" + "=" * 50)
    print("🎯 RESULTS:")
    
    if successful_servers:
        print(f"✅ Found {len(successful_servers)} working server(s):")
        
        for i, server_info in enumerate(successful_servers, 1):
            print(f"\n{i}. Server Name: {server_info['server']}")
            print(f"   Actual Server: {server_info['account_server']}")
            print(f"   Account: {server_info['login']}")
            print(f"   Balance: ${server_info['balance']:.2f}")
        
        # Recommend the first working server
        recommended = successful_servers[0]
        print(f"\n🎯 RECOMMENDED SERVER: {recommended['server']}")
        print(f"   (Actual server name: {recommended['account_server']})")
        
        return recommended['server']
    else:
        print("❌ No working servers found!")
        print("\nPossible issues:")
        print("1. MT5 terminal not running")
        print("2. Wrong login credentials")
        print("3. Server name not in our test list")
        print("\nTry manually checking your MT5 terminal for the exact server name.")
        
        return None

def get_server_from_terminal():
    """Try to get server info from MT5 terminal if already connected"""
    
    print("\n🔍 CHECKING CURRENT MT5 TERMINAL CONNECTION")
    print("=" * 50)
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 not initialized")
        return None
    
    # Check if already connected
    account_info = mt5.account_info()
    if account_info:
        print("✅ MT5 terminal is already connected!")
        print(f"Account: {account_info.login}")
        print(f"Server: {account_info.server}")
        print(f"Balance: ${account_info.balance:.2f}")
        
        mt5.shutdown()
        return account_info.server
    else:
        print("⚠️ MT5 terminal not connected to any account")
        mt5.shutdown()
        return None

if __name__ == "__main__":
    print("""
🔍 MT5 SERVER FINDER
===================
Find your correct MT5 server automatically

Options:
1. Check current MT5 terminal connection
2. Test multiple server combinations
3. Both (recommended)
    """)
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "1":
        server = get_server_from_terminal()
        if server:
            print(f"\n🎯 Your server is: {server}")
    elif choice == "2":
        server = test_mt5_servers()
        if server:
            print(f"\n🎯 Use this server: {server}")
    elif choice == "3":
        # First check terminal
        current_server = get_server_from_terminal()
        if current_server:
            print(f"\n🎯 Current terminal server: {current_server}")
        
        # Then test combinations
        input("\nPress Enter to test server combinations...")
        found_server = test_mt5_servers()
        
        if found_server:
            print(f"\n🎯 Working server found: {found_server}")
            
            if current_server and current_server != found_server:
                print(f"⚠️ Note: Terminal shows '{current_server}' but '{found_server}' works")
    else:
        print("Invalid choice")