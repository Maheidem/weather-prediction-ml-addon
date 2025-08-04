#!/usr/bin/env python3
"""Simplified Weather Prediction ML Add-on - No ML dependencies"""

import os
import json
import time
import logging
import random
from datetime import datetime
from ha_mqtt_discoverable import Settings, DeviceInfo
from ha_mqtt_discoverable.sensors import Sensor, SensorInfo
import paho.mqtt.client as mqtt

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleWeatherPredictor:
    """Simplified weather predictor using random values for testing"""
    
    def __init__(self):
        self.mqtt_settings = None
        self.device_info = None
        self.sensors = {}
        
    def setup_mqtt(self):
        """Setup MQTT connection"""
        self.mqtt_settings = Settings.MQTT(
            host=os.environ.get('MQTT_BROKER', 'localhost'),
            port=int(os.environ.get('MQTT_PORT', 1883)),
            username=os.environ.get('MQTT_USERNAME'),
            password=os.environ.get('MQTT_PASSWORD'),
        )
        
        self.device_info = DeviceInfo(
            identifiers=["weather_prediction_ml"],
            name="Weather Prediction ML",
            model="Simplified Test Version",
            manufacturer="Home Assistant Add-on",
            sw_version="2.0.1"
        )
        
        logger.info(f"MQTT configured for {self.mqtt_settings.host}:{self.mqtt_settings.port}")
        
    def create_sensors(self):
        """Create Home Assistant sensors via MQTT discovery"""
        sensors_config = [
            {
                "name": "Weather Prediction",
                "unique_id": "weather_prediction",
                "icon": "mdi:weather-partly-cloudy",
                "unit": None,
            },
            {
                "name": "Weather Prediction Confidence",
                "unique_id": "weather_prediction_confidence", 
                "icon": "mdi:percent",
                "unit": "%",
            },
            {
                "name": "Weather Prediction Trend",
                "unique_id": "weather_prediction_trend",
                "icon": "mdi:trending-up",
                "unit": None,
            },
            {
                "name": "Weather Prediction Last Update",
                "unique_id": "weather_prediction_last_update",
                "icon": "mdi:clock-outline",
                "unit": None,
            }
        ]
        
        for config in sensors_config:
            sensor_info = SensorInfo(
                name=config["name"],
                unique_id=config["unique_id"],
                device=self.device_info,
                icon=config["icon"],
                unit_of_measurement=config["unit"],
            )
            
            settings = Settings(mqtt=self.mqtt_settings, entity=sensor_info)
            sensor = Sensor(settings)
            self.sensors[config["unique_id"]] = sensor
            logger.info(f"Created sensor: {config['name']}")
            
    def get_mock_prediction(self):
        """Get mock prediction data for testing"""
        conditions = ["sunny", "cloudy", "rainy"]
        prediction = random.choice(conditions)
        confidence = random.randint(70, 95)
        trends = ["improving", "stable", "worsening"]
        trend = random.choice(trends)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "trend": trend,
            "timestamp": datetime.now().isoformat()
        }
        
    def update_sensors(self, data):
        """Update sensor values"""
        try:
            self.sensors["weather_prediction"].set_state(data["prediction"])
            self.sensors["weather_prediction_confidence"].set_state(data["confidence"])
            self.sensors["weather_prediction_trend"].set_state(data["trend"])
            self.sensors["weather_prediction_last_update"].set_state(data["timestamp"])
            
            logger.info(f"Updated sensors: {data['prediction']} ({data['confidence']}%)")
        except Exception as e:
            logger.error(f"Error updating sensors: {e}")
            
    def run(self):
        """Main run loop"""
        update_interval = int(os.environ.get('UPDATE_INTERVAL', 3600))
        
        logger.info("Starting simplified weather prediction service...")
        logger.info(f"Update interval: {update_interval} seconds")
        
        self.setup_mqtt()
        self.create_sensors()
        
        logger.info("Service started successfully!")
        
        while True:
            try:
                # Get mock prediction
                prediction_data = self.get_mock_prediction()
                
                # Update sensors
                self.update_sensors(prediction_data)
                
                # Wait for next update
                time.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait a minute before retrying

if __name__ == "__main__":
    predictor = SimpleWeatherPredictor()
    predictor.run()