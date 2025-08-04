"""Mock ML Predictor for Weather Prediction"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger('weather_prediction_ml.predictor')

class WeatherPredictor:
    """Mock weather prediction that simulates ML models"""
    
    def __init__(self):
        self.models_loaded = True
        logger.info("Loading mock predictor (models incompatible with Python 3.11)")
        
        # Simulate model config
        self.config = {
            'test_accuracy': 0.822,
            'ensemble_weights': {'xgboost': 0.509, 'random_forest': 0.491},
            'feature_columns': ['temperature', 'humidity', 'pressure', 'hour', 'day_of_week']
        }
    
    def predict(self, sensor_data):
        """Make a weather prediction using mock logic"""
        try:
            # Extract latest values
            latest_temp = sensor_data['temperature'][-1]['value'] if sensor_data['temperature'] else 20
            latest_humidity = sensor_data['humidity'][-1]['value'] if sensor_data['humidity'] else 60
            latest_pressure = sensor_data['pressure'][-1]['value'] if sensor_data['pressure'] else 1013
            
            # Calculate trends
            temp_trend = self._calculate_trend([d['value'] for d in sensor_data['temperature'][-10:]])
            humidity_trend = self._calculate_trend([d['value'] for d in sensor_data['humidity'][-10:]])
            
            # Mock prediction logic based on sensor data
            if latest_pressure < 1000:
                prediction = 'decrease'
                confidence = 85.0
            elif latest_pressure > 1020 and latest_humidity < 50:
                prediction = 'increase'
                confidence = 88.0
            elif temp_trend > 0.5 and humidity_trend < -0.5:
                prediction = 'increase'
                confidence = 82.0
            elif temp_trend < -0.5:
                prediction = 'decrease'
                confidence = 79.0
            else:
                prediction = 'stable'
                confidence = 75.0
            
            # Calculate mock probabilities
            probabilities = self._calculate_probabilities(prediction, confidence)
            
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': probabilities,
                'timestamp': datetime.now().isoformat(),
                'sensor_values': {
                    'temperature': latest_temp,
                    'humidity': latest_humidity,
                    'pressure': latest_pressure
                }
            }
            
            logger.info(f"Mock prediction: {prediction} ({confidence:.1f}%)")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Return default prediction
            return {
                'prediction': 'stable',
                'confidence': 50.0,
                'probabilities': {'increase': 33.3, 'decrease': 33.3, 'stable': 33.4},
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_trend(self, values):
        """Calculate trend from values"""
        if len(values) < 2:
            return 0
        return (values[-1] - values[0]) / len(values)
    
    def _calculate_probabilities(self, prediction, confidence):
        """Calculate mock probability distribution"""
        # Start with base probabilities
        probs = {'increase': 33.3, 'decrease': 33.3, 'stable': 33.4}
        
        # Adjust based on prediction
        probs[prediction] = confidence
        remaining = 100 - confidence
        
        for key in probs:
            if key != prediction:
                probs[key] = remaining / 2
        
        return probs