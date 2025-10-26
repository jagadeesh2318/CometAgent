#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

print("Importing the module...")
exec(open('agentic-trader.py').read().replace('if __name__ == "__main__":', 'if False:'))

print("Module loaded successfully")

print("Testing with Excel file...")
try:
    positions = _read_portfolio('CometCryptoPortfolio.xlsx', 'crypto')
    print(f"Positions loaded: {[p.symbol for p in positions]}")

    print("Testing analyze_positions...")
    df_signals = analyze_positions('crypto', 'short', 'coinbase', positions)
    print("Analysis completed successfully!")
    print(df_signals.head())

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()