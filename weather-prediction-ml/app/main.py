#!/usr/bin/env python3
"""Weather Prediction ML Addon - Main Service"""

import os
import sys
import time
import json
import logging
import signal
from datetime import datetime
from weather_service import WeatherPredictionService

# Setup logging
log_level = os.environ.get('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('weather_prediction_ml')

# Global service instance
service = None

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info("Received shutdown signal, cleaning up...")
    if service:
        service.cleanup()
    sys.exit(0)

def main():
    """Main service entry point"""
    global service
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("Weather Prediction ML Addon starting...")
    
    # Load configuration from environment
    ha_token = os.environ.get('SUPERVISOR_TOKEN')
    
    # Determine the correct HA URL based on token type
    # Long-lived tokens need direct API access, not through supervisor
    ha_url = os.environ.get('HOMEASSISTANT_URL', 'http://supervisor/core')
    if ha_token and len(ha_token) > 100:  # Long-lived tokens are typically longer
        # This is likely a Long-Lived Access Token, use direct API
        # Use IP address as .local domains don't resolve in Docker containers
        ha_url = 'http://192.168.31.114:8123'
        logger.info("Detected Long-Lived Access Token, using direct API access")
    
    config = {
        'mqtt_broker': os.environ.get('MQTT_BROKER', 'core-mosquitto'),
        'mqtt_port': int(os.environ.get('MQTT_PORT', 1883)),
        'mqtt_username': os.environ.get('MQTT_USERNAME', ''),
        'mqtt_password': os.environ.get('MQTT_PASSWORD', ''),
        'update_interval': int(os.environ.get('UPDATE_INTERVAL', 3600)),
        'temperature_sensor': os.environ.get('TEMP_SENSOR'),
        'humidity_sensor': os.environ.get('HUMIDITY_SENSOR'),
        'pressure_sensor': os.environ.get('PRESSURE_SENSOR'),
        'ha_token': ha_token,
        'ha_url': ha_url
    }
    
    logger.info(f"Configuration loaded: {json.dumps({k: v for k, v in config.items() if k not in ['mqtt_password', 'ha_token']}, indent=2)}")
    
    try:
        # Initialize service
        service = WeatherPredictionService(config)
        
        # Start the service
        service.run()
        
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if service:
            service.cleanup()

if __name__ == "__main__":
    main()