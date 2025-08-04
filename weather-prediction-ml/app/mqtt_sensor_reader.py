"""MQTT Sensor Reader - Gets real sensor data via MQTT"""

import json
import logging
import time
from collections import deque
from datetime import datetime, timedelta
import paho.mqtt.client as mqtt
from threading import Lock

logger = logging.getLogger('weather_prediction_ml.mqtt_reader')

class MQTTSensorReader:
    """Reads sensor values from Home Assistant via MQTT"""
    
    def __init__(self, mqtt_config):
        self.mqtt_config = mqtt_config
        self.sensor_data = {}
        self.data_lock = Lock()
        self.client = None
        self.connected = False
        
        # Keep history for each sensor (48 hours of data)
        self.history_hours = 48
        self.max_entries = self.history_hours * 60  # Store minute-level data
        
    def start(self):
        """Start MQTT client and subscribe to sensors"""
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        # Set authentication
        if self.mqtt_config.get('username'):
            self.client.username_pw_set(
                self.mqtt_config['username'],
                self.mqtt_config.get('password', '')
            )
        
        try:
            self.client.connect(
                self.mqtt_config['broker'],
                self.mqtt_config['port'],
                60
            )
            self.client.loop_start()
            logger.info("MQTT sensor reader started")
            
            # Wait for connection
            for _ in range(10):
                if self.connected:
                    break
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Failed to connect to MQTT: {e}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for MQTT connection"""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            self.connected = True
            
            # Subscribe to sensor topics
            sensors = self.mqtt_config.get('sensors', {})
            for sensor_name, entity_id in sensors.items():
                # Home Assistant publishes states to homeassistant/+/+/state
                # We need to extract the topic from the entity_id
                topic = f"homeassistant/sensor/{entity_id.replace('.', '/')}/state"
                client.subscribe(topic)
                logger.info(f"Subscribed to {topic} for {sensor_name}")
                
                # Initialize history
                with self.data_lock:
                    self.sensor_data[sensor_name] = deque(maxlen=self.max_entries)
        else:
            logger.error(f"Failed to connect to MQTT: RC={rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection"""
        self.connected = False
        logger.warning(f"Disconnected from MQTT broker: RC={rc}")
    
    def _on_message(self, client, userdata, msg):
        """Process incoming sensor data"""
        try:
            # Parse the topic to get sensor type
            topic_parts = msg.topic.split('/')
            
            # Try to parse the payload
            payload = msg.payload.decode('utf-8')
            
            # Try to extract numeric value
            try:
                value = float(payload)
            except ValueError:
                # Maybe it's JSON
                try:
                    data = json.loads(payload)
                    value = float(data.get('value', data.get('state', 0)))
                except:
                    logger.debug(f"Could not parse value from: {payload}")
                    return
            
            # Determine sensor type from topic
            sensor_type = None
            if 'temperature' in msg.topic.lower():
                sensor_type = 'temperature'
            elif 'humidity' in msg.topic.lower():
                sensor_type = 'humidity'
            elif 'pressure' in msg.topic.lower():
                sensor_type = 'pressure'
            
            if sensor_type:
                with self.data_lock:
                    if sensor_type not in self.sensor_data:
                        self.sensor_data[sensor_type] = deque(maxlen=self.max_entries)
                    
                    self.sensor_data[sensor_type].append({
                        'timestamp': datetime.now(),
                        'value': value
                    })
                    
                logger.debug(f"Stored {sensor_type} value: {value}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def get_history(self, sensor_type, hours=48):
        """Get sensor history as a list of values"""
        with self.data_lock:
            if sensor_type not in self.sensor_data:
                return []
            
            # Get data from the last N hours
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            values = []
            for entry in self.sensor_data[sensor_type]:
                if entry['timestamp'] > cutoff_time:
                    values.append(entry['value'])
            
            # If we have less than 48 values, pad with the average
            if len(values) < hours:
                if values:
                    avg_value = sum(values) / len(values)
                    # Pad to get hourly values
                    while len(values) < hours:
                        values.insert(0, avg_value)
                else:
                    # No data at all, return empty
                    return []
            
            return values
    
    def stop(self):
        """Stop MQTT client"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("MQTT sensor reader stopped")