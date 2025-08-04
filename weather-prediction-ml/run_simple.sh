#!/usr/bin/env bashio

# Simple run script without bashio logging
set -e

echo "Starting Weather Prediction ML addon (Simplified)..."

# Read configuration
CONFIG_PATH=/data/options.json
export MQTT_BROKER=$(jq -r '.mqtt_broker' $CONFIG_PATH)
export MQTT_PORT=$(jq -r '.mqtt_port' $CONFIG_PATH)
export MQTT_USERNAME=$(jq -r '.mqtt_username' $CONFIG_PATH)
export MQTT_PASSWORD=$(jq -r '.mqtt_password' $CONFIG_PATH)
export UPDATE_INTERVAL=$(jq -r '.update_interval' $CONFIG_PATH)
export LOG_LEVEL=$(jq -r '.log_level' $CONFIG_PATH)

echo "Configuration loaded:"
echo "  MQTT Broker: ${MQTT_BROKER}:${MQTT_PORT}"
echo "  Update Interval: ${UPDATE_INTERVAL}s"

# Run the Python application
cd /app
exec python3 -u main.py