#!/usr/bin/env python
"""
TEST ALL FOREX MODES - Quick demo of all trading modes
"""

import subprocess
import time

print("="*60)
print("FOREX SYSTEM - TESTING ALL MODES")
print("="*60)

modes = [
    ("status", "System Status Check"),
    ("analyze", "Market Analysis"),
    ("demo", "Demo Trading"),
    ("safe -c 10", "Safe Mode with EUR 10"),
    ("turbo -c 10", "Turbo Mode with EUR 10"),
]

for mode, description in modes:
    print(f"\n[TEST] {description}")
    print("-"*40)
    
    try:
        result = subprocess.run(
            f"python forex.py {mode}", 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=10
        )
        
        # Show output
        output_lines = result.stdout.split('\n')
        for line in output_lines[:20]:  # First 20 lines
            if line.strip():
                print(line)
                
        # Check for errors
        if result.returncode != 0 and result.stderr:
            print(f"[ERROR] {result.stderr[:200]}")
            
    except subprocess.TimeoutExpired:
        print("[TIMEOUT] Mode took too long, skipping...")
    except Exception as e:
        print(f"[ERROR] {e}")
        
    time.sleep(1)

print("\n" + "="*60)
print("TESTING COMPLETE")
print("="*60)
print("\nUsage:")
print("  Interactive menu:     python forex.py")
print("  Turbo mode:          python forex.py turbo")
print("  Safe mode:           python forex.py safe -c 100")
print("  Scalp mode:          python forex.py scalp")
print("  Market analysis:     python forex.py analyze")
print("  System status:       python forex.py status")