#!/usr/bin/env python3
print("Starting test...")

import argparse
print("argparse imported")

parser = argparse.ArgumentParser()
parser.add_argument("--test", required=True)
print("Parser created")

if __name__ == "__main__":
    print("In main block")
    args = parser.parse_args()
    print(f"Args parsed: {args}")