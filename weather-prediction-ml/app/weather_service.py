"""Weather Prediction Service with MQTT Autodiscovery"""

import os
import json
import time
import logging
import requests
import numpy as np
from datetime import datetime, timedelta
from ha_mqtt_discoverable import Settings, DeviceInfo
from ha_mqtt_discoverable.sensors import Sensor, SensorInfo
from predictor import WeatherPredictor

logger = logging.getLogger('weather_prediction_ml.service')

class WeatherPredictionService:
    """Main service class for weather predictions"""
    
    def __init__(self, config):
        self.config = config
        self.running = True
        self.last_prediction = None
        
        # Initialize predictor
        logger.info("Loading ML models...")
        self.predictor = WeatherPredictor()
        
        # MQTT settings
        mqtt_settings = Settings.MQTT(
            host=config['mqtt_broker'],
            port=config['mqtt_port'],
            username=config['mqtt_username'] if config['mqtt_username'] else None,
            password=config['mqtt_password'] if config['mqtt_password'] else None
        )
        
        # Device info for all sensors
        device_info = DeviceInfo(
            name="Weather Prediction ML",
            identifiers=["weather_prediction_ml_addon"],
            manufacturer="Custom",
            model="Ensemble ML (XGBoost + Random Forest)",
            sw_version="4.0.9",
            configuration_url="http://homeassistant.local:8123/hassio/addon/weather_prediction_ml"
        )
        
        # Create sensors with autodiscovery
        logger.info("Creating MQTT sensors with autodiscovery...")
        self.sensors = self._create_sensors(mqtt_settings, device_info)
        
        # Home Assistant API headers
        self.ha_headers = {}
        if config.get('ha_token'):
            self.ha_headers = {
                "Authorization": f"Bearer {config['ha_token']}",
                "Content-Type": "application/json"
            }
        else:
            logger.warning("No Home Assistant token available, sensor history will not be available")
        
        logger.info("Service initialized successfully")
    
    def _create_sensors(self, mqtt_settings, device_info):
        """Create all sensors with MQTT autodiscovery"""
        sensors = {}
        
        # Main prediction sensor
        sensors['prediction'] = Sensor(Settings(
            mqtt=mqtt_settings,
            entity=SensorInfo(
                name="Weather Prediction",
                unique_id="weather_prediction_ml_prediction",
                device=device_info,
                icon="mdi:weather-partly-cloudy",
                state_class="measurement"
            )
        ))
        
        # Trend sensor
        sensors['trend'] = Sensor(Settings(
            mqtt=mqtt_settings,
            entity=SensorInfo(
                name="Weather Trend",
                unique_id="weather_prediction_ml_trend",
                device=device_info,
                icon="mdi:trending-up"
            )
        ))
        
        # Confidence sensor
        sensors['confidence'] = Sensor(Settings(
            mqtt=mqtt_settings,
            entity=SensorInfo(
                name="Prediction Confidence",
                unique_id="weather_prediction_ml_confidence",
                device=device_info,
                unit_of_measurement="%",
                icon="mdi:gauge",
                state_class="measurement"
            )
        ))
        
        # Probability sensors
        for prob_type in ['increase', 'decrease', 'stable']:
            icon_map = {
                'increase': 'mdi:thermometer-chevron-up',
                'decrease': 'mdi:thermometer-chevron-down',
                'stable': 'mdi:thermometer'
            }
            
            sensors[f'{prob_type}_probability'] = Sensor(Settings(
                mqtt=mqtt_settings,
                entity=SensorInfo(
                    name=f"Temperature {prob_type.capitalize()} Probability",
                    unique_id=f"weather_prediction_ml_{prob_type}_probability",
                    device=device_info,
                    unit_of_measurement="%",
                    icon=icon_map.get(prob_type, 'mdi:percent'),
                    state_class="measurement"
                )
            ))
        
        # Last update sensor
        sensors['last_update'] = Sensor(Settings(
            mqtt=mqtt_settings,
            entity=SensorInfo(
                name="Last Prediction Update",
                unique_id="weather_prediction_ml_last_update",
                device=device_info,
                icon="mdi:clock-outline",
                device_class="timestamp"
            )
        ))
        
        # Model accuracy sensor (static)
        sensors['model_accuracy'] = Sensor(Settings(
            mqtt=mqtt_settings,
            entity=SensorInfo(
                name="Model Accuracy",
                unique_id="weather_prediction_ml_accuracy",
                device=device_info,
                unit_of_measurement="%",
                icon="mdi:chart-line",
                state_class="measurement"
            )
        ))
        
        # Set static accuracy
        sensors['model_accuracy'].set_state(82.2)
        
        return sensors
    
    def get_sensor_history(self, entity_id, hours=48):
        """Get sensor history from Home Assistant"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        url = f"{self.config['ha_url']}/api/history/period/{start_time.isoformat()}"
        params = {
            "filter_entity_id": entity_id,
            "end_time": end_time.isoformat()
        }
        
        try:
            if not self.ha_headers:
                logger.warning("Cannot get sensor history without HA token")
                return []
            response = requests.get(url, headers=self.ha_headers, params=params)
            response.raise_for_status()
            
            history = response.json()
            if history and len(history) > 0:
                # Extract state values
                states = []
                for entry in history[0]:
                    try:
                        value = float(entry['state'])
                        timestamp = entry['last_changed']
                        states.append({
                            'timestamp': timestamp,
                            'value': value
                        })
                    except (ValueError, KeyError):
                        continue
                
                return states
            
        except Exception as e:
            logger.error(f"Failed to get history for {entity_id}: {e}")
        
        return []
    
    def make_prediction(self):
        """Make a weather prediction"""
        logger.info("Making weather prediction...")
        
        try:
            # Get sensor history
            temp_history = self.get_sensor_history(self.config['temperature_sensor'])
            humidity_history = self.get_sensor_history(self.config['humidity_sensor'])
            pressure_history = self.get_sensor_history(self.config['pressure_sensor'])
            
            # Extract just the values from the history
            temp_values = [entry['value'] for entry in temp_history] if temp_history else []
            humidity_values = [entry['value'] for entry in humidity_history] if humidity_history else []
            pressure_values = [entry['value'] for entry in pressure_history] if pressure_history else []
            
            # Use mock data if no sensor history available
            if not temp_values:
                logger.warning("No temperature history, using mock data")
                temp_values = self._generate_mock_history('temperature', 20, 2)
            if not humidity_values:
                logger.warning("No humidity history, using mock data")
                humidity_values = self._generate_mock_history('humidity', 60, 10)
            if not pressure_values:
                logger.warning("No pressure history, using mock data")
                pressure_values = self._generate_mock_history('pressure', 1013, 5)
            
            # Log some stats
            logger.info(f"Got {len(temp_values)} temperature, {len(humidity_values)} humidity, {len(pressure_values)} pressure readings")
            
            # Make prediction with separate arguments
            result = self.predictor.predict(temp_values, humidity_values, pressure_values)
            
            # Update sensors
            self._update_sensors(result)
            
            self.last_prediction = result
            logger.info(f"Prediction complete: {result['prediction']} ({result['confidence']:.1f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return None
    
    def _update_sensors(self, prediction):
        """Update all sensors with prediction results"""
        if not prediction:
            return
        
        # Main prediction
        self.sensors['prediction'].set_state(prediction['prediction'])
        
        # Trend indicator
        trend_map = {
            'increase': '↑',
            'decrease': '↓',
            'stable': '→'
        }
        self.sensors['trend'].set_state(trend_map.get(prediction['prediction'], '?'))
        
        # Confidence
        self.sensors['confidence'].set_state(round(prediction['confidence'], 1))
        
        # Probabilities
        for prob_type in ['increase', 'decrease', 'stable']:
            prob_value = prediction['probabilities'].get(prob_type, 0)
            self.sensors[f'{prob_type}_probability'].set_state(round(prob_value, 1))
        
        # Last update
        self.sensors['last_update'].set_state(datetime.now().isoformat())
        
        # Set attributes on main sensor
        self.sensors['prediction'].set_attributes({
            "confidence": prediction['confidence'],
            "probabilities": prediction['probabilities'],
            "model_type": "Ensemble (XGBoost + Random Forest)",
            "model_accuracy": 82.2,
            "next_update": (datetime.now() + timedelta(seconds=self.config['update_interval'])).isoformat()
        })
    
    def run(self):
        """Main service loop"""
        logger.info("Starting prediction service loop...")
        
        # Make initial prediction
        self.make_prediction()
        
        # Main loop
        while self.running:
            try:
                # Wait for next update
                logger.info(f"Waiting {self.config['update_interval']} seconds until next prediction...")
                time.sleep(self.config['update_interval'])
                
                # Make prediction
                self.make_prediction()
                
            except Exception as e:
                logger.error(f"Error in service loop: {e}", exc_info=True)
                # Wait a bit before retrying
                time.sleep(60)
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up service...")
        self.running = False
        # MQTT connections are cleaned up automatically by ha-mqtt-discoverable
    
    def _generate_mock_history(self, sensor_type, base_value, variation):
        """Generate mock sensor history for testing"""
        values = []
        now = datetime.now()
        
        # Generate 48 hours of hourly data
        for hours_ago in range(48, 0, -1):
            timestamp = now - timedelta(hours=hours_ago)
            # Add some realistic variation
            value = base_value + np.random.normal(0, variation) + \
                   5 * np.sin(2 * np.pi * timestamp.hour / 24)  # Daily cycle
            
            values.append(float(value))
        
        return values