"""
START THE AUTONOMOUS EVOLUTION SYSTEM
=====================================
This launches the new balanced trading system
"""

import subprocess
import time
import os

print("="*70)
print("LAUNCHING AUTONOMOUS EVOLUTION SYSTEM")
print("="*70)
print("\nThis system features:")
print("- BALANCED BUY/SELL trading (no more 100% buy bias)")
print("- 100 competing strategies that evolve")
print("- Automatic start/stop based on market conditions")
print("- Continuous self-improvement")
print("- Saves learning to evolution_state.json")
print("\n" + "="*70)

# First, close old positions if user wants
response = input("\nDo you want to close OLD bot positions first? (y/n): ")
if response.lower() == 'y':
    print("\nClosing old positions...")
    os.system("python close_all_positions.py")
    time.sleep(2)

print("\nStarting Evolution System...")
print("Press Ctrl+C to stop\n")

# Run the evolution system
try:
    subprocess.run(["python", "MT5_AUTONOMOUS_EVOLUTION.py"])
except KeyboardInterrupt:
    print("\n\nEvolution System stopped.")
    print("The system has saved its learning to evolution_state.json")
    print("It will continue evolving from where it left off next time!")