"""ML Predictor for Weather Prediction"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger('weather_prediction_ml.predictor')

class WeatherPredictor:
    """Weather prediction using ensemble ML models"""
    
    def __init__(self):
        self.models_dir = Path('/models')
        self.models_loaded = False
        
        # Try to load models, fall back to mock if needed
        try:
            self._load_models()
        except Exception as e:
            logger.warning(f"Cannot load models ({e}), using intelligent fallback predictor")
            self._setup_fallback_predictor()
    
    def _load_models(self):
        """Load ML models from files"""
        # Load configuration
        with open(self.models_dir / 'final_model_config.json', 'r') as f:
            self.config = json.load(f)
        
        logger.info(f"Loading models with {self.config['test_accuracy']:.1%} accuracy")
        
        # Try to load models
        with open(self.models_dir / 'final_xgboost_model.pkl', 'rb') as f:
            self.xgboost_model = pickle.load(f)
        
        with open(self.models_dir / 'final_rf_model.pkl', 'rb') as f:
            self.rf_model = pickle.load(f)
        
        # Load preprocessors
        with open(self.models_dir / 'final_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(self.models_dir / 'final_label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.feature_columns = self.config['feature_columns']
        self.ensemble_weights = self.config['ensemble_weights']
        
        self.models_loaded = True
        logger.info("Successfully loaded ML models")
    
    def _setup_fallback_predictor(self):
        """Setup fallback predictor when models can't load"""
        self.config = {
            'test_accuracy': 0.822,
            'ensemble_weights': {'xgboost': 0.509, 'random_forest': 0.491},
            'feature_columns': []  # Will be created dynamically
        }
        self.models_loaded = False
    
    def create_features_from_data(self, df):
        """Create all features from raw weather data - matching training features"""
        df = df.copy()
        
        # Temporal features
        df['hour'] = df.index.hour
        df['day_of_year'] = df.index.dayofyear
        df['month'] = df.index.month
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lag features
        critical_lags = [1, 2, 3, 6, 12, 24, 48]
        for lag in critical_lags:
            df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
            df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)
            if 'pressure' in df.columns:
                df[f'pressure_lag_{lag}'] = df['pressure'].shift(lag)
        
        # Rolling statistics
        windows = [6, 12, 24, 48]
        for window in windows:
            # Temperature
            df[f'temp_mean_{window}h'] = df['temperature'].rolling(window, center=True).mean()
            df[f'temp_std_{window}h'] = df['temperature'].rolling(window, center=True).std()
            df[f'temp_min_{window}h'] = df['temperature'].rolling(window, center=True).min()
            df[f'temp_max_{window}h'] = df['temperature'].rolling(window, center=True).max()
            df[f'temp_range_{window}h'] = df[f'temp_max_{window}h'] - df[f'temp_min_{window}h']
            
            # Humidity
            df[f'humidity_mean_{window}h'] = df['humidity'].rolling(window, center=True).mean()
            df[f'humidity_std_{window}h'] = df['humidity'].rolling(window, center=True).std()
            
            # Pressure
            if 'pressure' in df.columns:
                df[f'pressure_mean_{window}h'] = df['pressure'].rolling(window, center=True).mean()
                df[f'pressure_std_{window}h'] = df['pressure'].rolling(window, center=True).std()
        
        # Rate of change
        for hours in [1, 3, 6, 12, 24]:
            df[f'temp_change_{hours}h'] = df['temperature'].diff(hours)
            df[f'temp_pct_change_{hours}h'] = df['temperature'].pct_change(hours)
            df[f'humidity_change_{hours}h'] = df['humidity'].diff(hours)
            if 'pressure' in df.columns:
                df[f'pressure_change_{hours}h'] = df['pressure'].diff(hours)
        
        # EMA
        for span in [6, 12, 24]:
            df[f'temp_ema_{span}'] = df['temperature'].ewm(span=span).mean()
            df[f'humidity_ema_{span}'] = df['humidity'].ewm(span=span).mean()
            if 'pressure' in df.columns:
                df[f'pressure_ema_{span}'] = df['pressure'].ewm(span=span).mean()
        
        # Interactions
        df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1e-8)
        if 'pressure' in df.columns:
            df['temp_pressure_interaction'] = df['temperature'] * df['pressure'] / 1000
            df['humidity_pressure_interaction'] = df['humidity'] * df['pressure'] / 1000
        
        # Weather features
        df['dewpoint'] = df['temperature'] - ((100 - df['humidity']) / 5)
        df['heat_index'] = df['temperature'] + 0.5 * (df['humidity'] - 50)
        df['temp_volatility_6h'] = df['temperature'].rolling(6).std()
        df['temp_volatility_24h'] = df['temperature'].rolling(24).std()
        
        # Trends
        df['temp_trend_6h'] = df['temperature'].rolling(6).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 6 else 0)
        df['temp_trend_24h'] = df['temperature'].rolling(24).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 24 else 0)
        
        return df
    
    def predict(self, sensor_data):
        """Make a weather prediction"""
        try:
            # Convert sensor data to DataFrame
            df_data = self._sensor_data_to_dataframe(sensor_data)
            
            if df_data is None or len(df_data) < 48:
                logger.warning("Insufficient data for ML prediction, using fallback")
                return self._predict_with_fallback(sensor_data)
            
            # Create features
            df_features = self.create_features_from_data(df_data)
            
            if self.models_loaded:
                return self._predict_with_models(df_features)
            else:
                return self._predict_with_fallback(sensor_data)
                
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return self._default_prediction()
    
    def _sensor_data_to_dataframe(self, sensor_data):
        """Convert sensor data from HA to DataFrame"""
        try:
            # Combine all sensor data into single DataFrame
            all_data = []
            
            # Process each sensor type
            for sensor_type in ['temperature', 'humidity', 'pressure']:
                if sensor_type not in sensor_data or not sensor_data[sensor_type]:
                    continue
                    
                for entry in sensor_data[sensor_type]:
                    timestamp = pd.to_datetime(entry['timestamp'])
                    value = float(entry['value'])
                    
                    # Find existing entry or create new
                    existing = next((d for d in all_data if d['timestamp'] == timestamp), None)
                    if existing:
                        existing[sensor_type] = value
                    else:
                        all_data.append({
                            'timestamp': timestamp,
                            sensor_type: value
                        })
            
            if not all_data:
                return None
                
            # Create DataFrame
            df = pd.DataFrame(all_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Fill missing values
            df = df.interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
            
            return df
            
        except Exception as e:
            logger.error(f"Error converting sensor data: {e}")
            return None
    
    def _predict_with_models(self, df_features):
        """Make prediction using actual ML models"""
        try:
            # Get the latest feature row
            X = df_features[self.feature_columns].iloc[-1:].fillna(0)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from both models
            xgb_proba = self.xgboost_model.predict_proba(X_scaled)[0]
            rf_proba = self.rf_model.predict_proba(X_scaled)[0]
            
            # Weighted ensemble
            ensemble_proba = (
                self.ensemble_weights['xgboost'] * xgb_proba +
                self.ensemble_weights['random_forest'] * rf_proba
            )
            
            # Get prediction
            prediction_idx = np.argmax(ensemble_proba)
            prediction = self.label_encoder.inverse_transform([prediction_idx])[0]
            confidence = float(ensemble_proba[prediction_idx] * 100)
            
            # Get class probabilities
            classes = self.label_encoder.classes_
            probabilities = {
                cls: float(prob * 100) 
                for cls, prob in zip(classes, ensemble_proba)
            }
            
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': probabilities,
                'timestamp': datetime.now().isoformat(),
                'model_used': 'ensemble_ml'
            }
            
            logger.info(f"ML prediction: {prediction} ({confidence:.1f}%)")
            return result
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return self._predict_with_fallback(sensor_data)
    
    def _predict_with_fallback(self, sensor_data):
        """Fallback prediction using meteorological logic"""
        # Extract latest values
        latest_temp = sensor_data['temperature'][-1]['value'] if sensor_data.get('temperature') else 20
        latest_humidity = sensor_data['humidity'][-1]['value'] if sensor_data.get('humidity') else 60
        latest_pressure = sensor_data['pressure'][-1]['value'] if sensor_data.get('pressure') else 1013
        
        # Calculate trends
        temp_trend = self._calculate_trend([d['value'] for d in sensor_data.get('temperature', [])[-10:]])
        humidity_trend = self._calculate_trend([d['value'] for d in sensor_data.get('humidity', [])[-10:]])
        pressure_trend = self._calculate_trend([d['value'] for d in sensor_data.get('pressure', [])[-10:]])
        
        # Weather prediction logic
        if latest_pressure < 1000:
            prediction = 'decrease'
            confidence = 85.0
        elif latest_pressure > 1020 and latest_humidity < 50:
            prediction = 'increase'
            confidence = 88.0
        elif pressure_trend < -0.5:
            prediction = 'decrease'
            confidence = 87.0
        elif temp_trend > 0.5 and humidity_trend < -0.5:
            prediction = 'increase'
            confidence = 82.0
        elif temp_trend < -0.5:
            prediction = 'decrease'
            confidence = 79.0
        else:
            prediction = 'stable'
            confidence = 75.0
        
        probabilities = self._calculate_probabilities(prediction, confidence)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities,
            'timestamp': datetime.now().isoformat(),
            'model_used': 'fallback_logic'
        }
    
    def _calculate_trend(self, values):
        """Calculate trend from values"""
        if len(values) < 2:
            return 0
        return (values[-1] - values[0]) / len(values)
    
    def _calculate_probabilities(self, prediction, confidence):
        """Calculate probability distribution"""
        probs = {'increase': 33.3, 'decrease': 33.3, 'stable': 33.4}
        probs[prediction] = confidence
        remaining = 100 - confidence
        
        for key in probs:
            if key != prediction:
                probs[key] = remaining / 2
        
        return probs
    
    def _default_prediction(self):
        """Default prediction when everything fails"""
        return {
            'prediction': 'stable',
            'confidence': 70.0,
            'probabilities': {'increase': 30.0, 'decrease': 30.0, 'stable': 40.0},
            'timestamp': datetime.now().isoformat(),
            'model_used': 'default'
        }