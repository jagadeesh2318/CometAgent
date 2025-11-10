#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

print("Importing the module...")
exec(open('agentic-trader.py').read().replace('if __name__ == "__main__":', 'if False:'))

print("Module loaded successfully")

# Test directly calling functions
print("Testing _read_portfolio...")
positions = _read_portfolio('test_portfolio.csv', 'crypto')
print(f"Positions loaded: {[p.symbol for p in positions]}")

print("Testing single position analysis...")
pos = positions[0]
print(f"Processing {pos.symbol}...")

try:
    print("Testing download...")
    df = _download_history(pos.symbol, 'short')
    print(f"Downloaded {len(df)} rows")

    print("Testing indicators...")
    ind = _indicators(df)
    print(f"Indicators calculated: close={ind.close}")

except Exception as e:
    print(f"Error: {e}")