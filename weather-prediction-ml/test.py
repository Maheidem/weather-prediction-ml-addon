#!/usr/bin/env python3
import time
import os
import json

print("Weather Prediction ML addon starting...")
print("This is a minimal test version")

# Read config
with open('/data/options.json') as f:
    config = json.load(f)
    print(f"Config loaded: {config}")

# Just loop and print
while True:
    print("Weather addon is running...")
    time.sleep(60)