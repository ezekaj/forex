# HOW TO ENABLE AUTO TRADING IN MT5

## The AutoTrading Button Location:

Look at the **TOP TOOLBAR** in your MT5 window. The AutoTrading button is the **GREEN/RED button** that looks like a play button with an "EA" or arrow symbol.

In your screenshot, I can see the toolbar with these buttons:
- File, View, Insert, Charts, Tools, Window, Help
- Below that: Algo Trading, New Order, and other trading buttons

## Steps to Enable AutoTrading:

### Method 1: Toolbar Button
1. Look for the **Algo Trading button** in the toolbar (it might show as a triangle/play button)
2. If it's **RED**, click it to turn it **GREEN**
3. GREEN = AutoTrading Enabled
4. RED = AutoTrading Disabled

### Method 2: Menu Option
1. Go to **Tools** menu (in the top menu bar)
2. Select **Options** (or press Ctrl+O)
3. Go to **Expert Advisors** tab
4. Check these boxes:
   - ✅ Allow automated trading
   - ✅ Allow DLL imports (for Python connection)
   - ✅ Disable 'Confirm DLL function calls'
5. Click **OK**

### Method 3: Quick Enable
Press **Ctrl+E** on your keyboard - this toggles AutoTrading on/off

## Visual Indicators:
- **AutoTrading ON**: You'll see a small smiley face 😊 in the top right corner of charts
- **AutoTrading OFF**: You'll see a sad face ☹️ or no icon

## For Python Trading to Work:

You need to:
1. Enable AutoTrading (green button/Ctrl+E)
2. Allow automated trading in Options
3. Allow DLL imports in Options

## Important Notes:
- Your account shows "95948709 - MetaQuotes-Demo Demo Account"
- You're logged in correctly to the demo server
- Balance shows 18,533.55 (demo money)
- Once AutoTrading is enabled, the Python bot can execute trades

## Test After Enabling:
```python
# Run this to test:
python MT5_AUTO_TRADER.py
# Select option 3 to check connection
# Select option 5 to test a trade
```

The bot will only work when AutoTrading is GREEN/ENABLED!