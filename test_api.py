#!/usr/bin/env python3
"""Test script to validate Home Assistant API access"""

import requests
import json
from datetime import datetime, timedelta

# Configuration
HA_URL = "http://192.168.31.114:8123"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJiMzRjMTdmOTIzMjQ0MTI4OWZjOGE0MTE0NGExMTQxZCIsImlhdCI6MTczNTc5NTAzMSwiZXhwIjoyMDUxMTU1MDMxfQ.a_1F85H1MJvqnhQlzaC_Mxw7u2r6sAr_gIdkLrCLGQo"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

print("Testing Home Assistant API Access")
print("=" * 50)

# Test 1: Check API status
try:
    response = requests.get(f"{HA_URL}/api/", headers=headers)
    print(f"API Status Test: {response.status_code}")
    if response.status_code == 200:
        print("✓ API is accessible")
    else:
        print(f"✗ API error: {response.text}")
except Exception as e:
    print(f"✗ Connection error: {e}")

# Test 2: Get specific sensor state
sensor_id = "sensor.sensor_varanda_temperature"
try:
    response = requests.get(f"{HA_URL}/api/states/{sensor_id}", headers=headers)
    print(f"\nSensor State Test: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Current temperature: {data['state']}°C")
        print(f"  Last updated: {data['last_updated']}")
    else:
        print(f"✗ Sensor error: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Get sensor history
try:
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=2)
    
    url = f"{HA_URL}/api/history/period/{start_time.isoformat()}"
    params = {
        "filter_entity_id": sensor_id,
        "end_time": end_time.isoformat()
    }
    
    response = requests.get(url, headers=headers, params=params)
    print(f"\nHistory Test: {response.status_code}")
    if response.status_code == 200:
        history = response.json()
        if history and len(history) > 0:
            print(f"✓ Got {len(history[0])} history entries")
            # Show first few entries
            for i, entry in enumerate(history[0][:3]):
                print(f"  Entry {i}: state={entry['state']}, time={entry['last_changed']}")
            
            # Show the exact format
            print("\nRaw entry format:")
            print(json.dumps(history[0][0], indent=2))
    else:
        print(f"✗ History error: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 50)
print("Test complete")