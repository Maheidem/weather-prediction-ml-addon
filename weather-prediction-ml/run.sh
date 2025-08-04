#!/usr/bin/env bashio

# Enable error handling
set -e

bashio::log.info "Starting Weather Prediction ML addon..."

# Read configuration
CONFIG_PATH=/data/options.json
MQTT_BROKER=$(jq -r '.mqtt_broker' $CONFIG_PATH)
MQTT_PORT=$(jq -r '.mqtt_port' $CONFIG_PATH)
MQTT_USERNAME=$(jq -r '.mqtt_username' $CONFIG_PATH)
MQTT_PASSWORD=$(jq -r '.mqtt_password' $CONFIG_PATH)
UPDATE_INTERVAL=$(jq -r '.update_interval' $CONFIG_PATH)
TEMP_SENSOR=$(jq -r '.temperature_sensor' $CONFIG_PATH)
HUMIDITY_SENSOR=$(jq -r '.humidity_sensor' $CONFIG_PATH)
PRESSURE_SENSOR=$(jq -r '.pressure_sensor' $CONFIG_PATH)
LOG_LEVEL=$(jq -r '.log_level' $CONFIG_PATH)

# Export environment variables for Python
export MQTT_BROKER
export MQTT_PORT
export MQTT_USERNAME
export MQTT_PASSWORD
export UPDATE_INTERVAL
export TEMP_SENSOR
export HUMIDITY_SENSOR
export PRESSURE_SENSOR
export LOG_LEVEL

# Try to get the supervisor token using bashio
if bashio::var.has_value "supervisor_token"; then
    export SUPERVISOR_TOKEN=$(bashio::var.supervisor_token)
    export HOMEASSISTANT_URL="http://supervisor/core"
    bashio::log.info "Home Assistant API token obtained"
else
    bashio::log.warning "Home Assistant API token not available - predictions will use mock data"
fi

bashio::log.info "Configuration loaded:"
bashio::log.info "  MQTT Broker: ${MQTT_BROKER}:${MQTT_PORT}"
bashio::log.info "  Update Interval: ${UPDATE_INTERVAL}s"
bashio::log.info "  Temperature Sensor: ${TEMP_SENSOR}"
bashio::log.info "  Humidity Sensor: ${HUMIDITY_SENSOR}"
bashio::log.info "  Pressure Sensor: ${PRESSURE_SENSOR}"

# Wait for MQTT to be available
bashio::log.info "Waiting for MQTT broker..."
for i in {1..30}; do
    if nc -z ${MQTT_BROKER} ${MQTT_PORT} 2>/dev/null; then
        bashio::log.info "MQTT broker is available"
        break
    fi
    if [ $i -eq 30 ]; then
        bashio::log.error "MQTT broker not available after 30 seconds"
        exit 1
    fi
    sleep 1
done

# Run the Python application
bashio::log.info "Starting prediction service..."
cd /app
exec python3 -u main.py