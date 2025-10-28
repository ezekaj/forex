import MetaTrader5 as mt5

mt5.initialize()
mt5.login(95948709, 'To-4KyLg', 'MetaQuotes-Demo')

positions = mt5.positions_get()

print("="*50)
print("QUANTUM TRADER POSITIONS (Magic: 222222)")
print("="*50)

if positions:
    quantum_positions = [p for p in positions if p.magic == 222222]
    
    if quantum_positions:
        print(f"Found {len(quantum_positions)} Quantum positions:")
        for pos in quantum_positions:
            direction = "BUY" if pos.type == 0 else "SELL"
            print(f"{pos.symbol} | {direction} | Vol: {pos.volume} | P&L: ${pos.profit:.2f}")
            print(f"  Comment: {pos.comment}")
    else:
        print("No Quantum Trader positions found")
else:
    print("No positions at all")

print("\n" + "="*50)
print("ALL ACTIVE POSITIONS BY MAGIC NUMBER")
print("="*50)

if positions:
    magic_groups = {}
    for pos in positions:
        if pos.magic not in magic_groups:
            magic_groups[pos.magic] = []
        magic_groups[pos.magic].append(pos)
    
    for magic, group_positions in magic_groups.items():
        total_pnl = sum(p.profit for p in group_positions)
        print(f"Magic {magic}: {len(group_positions)} positions | Total P&L: ${total_pnl:.2f}")

mt5.shutdown()