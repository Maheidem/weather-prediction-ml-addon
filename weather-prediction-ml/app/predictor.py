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
        self.models_dir = Path('/app/models')
        self.models_loaded = False
        
        # Load models and configuration
        self._load_models()
    
    def _load_models(self):
        """Load ML models from files"""
        try:
            # Load configuration
            with open(self.models_dir / 'final_model_config.json', 'r') as f:
                self.config = json.load(f)
            
            logger.info(f"Loading models with {self.config['test_accuracy']:.1%} accuracy")
            
            # Load models
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
            logger.info("ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def predict(self, sensor_data):
        """Make weather prediction from sensor data"""
        if not self.models_loaded:
            raise RuntimeError("Models not loaded")
        
        try:
            # Convert sensor history to DataFrame
            df = self._prepare_dataframe(sensor_data)
            
            if len(df) < 48:
                logger.warning(f"Insufficient data: {len(df)} records, need at least 48")
                return self._default_prediction()
            
            # Engineer features
            features_df = self._engineer_features(df)
            
            # Get last row for prediction
            X = features_df.iloc[-1:][self.feature_columns].values
            
            # Handle missing features
            if X.shape[1] < len(self.feature_columns):
                logger.warning(f"Missing features: {X.shape[1]} vs {len(self.feature_columns)} expected")
                # Pad with zeros for missing features
                X = np.pad(X, ((0, 0), (0, len(self.feature_columns) - X.shape[1])), mode='constant')
            
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
            
            # Get final prediction
            pred_idx = np.argmax(ensemble_proba)
            pred_class = self.label_encoder.classes_[pred_idx]
            confidence = float(ensemble_proba[pred_idx] * 100)
            
            # Create result
            result = {
                'prediction': pred_class,
                'confidence': confidence,
                'probabilities': {
                    cls: float(prob * 100)
                    for cls, prob in zip(self.label_encoder.classes_, ensemble_proba)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return self._default_prediction()
    
    def _prepare_dataframe(self, sensor_data):
        """Convert sensor history to DataFrame"""
        # Align timestamps from all sensors
        all_timestamps = set()
        
        for sensor_type in ['temperature', 'humidity', 'pressure']:
            for entry in sensor_data.get(sensor_type, []):
                timestamp = pd.to_datetime(entry['timestamp'])
                all_timestamps.add(timestamp.replace(minute=0, second=0, microsecond=0))
        
        # Sort timestamps
        timestamps = sorted(all_timestamps)
        
        # Create DataFrame
        data = []
        for ts in timestamps:
            row = {'timestamp': ts}
            
            # Get closest value for each sensor
            for sensor_type in ['temperature', 'humidity', 'pressure']:
                value = self._get_closest_value(sensor_data.get(sensor_type, []), ts)
                row[sensor_type] = value
            
            if all(row.get(k) is not None for k in ['temperature', 'humidity', 'pressure']):
                data.append(row)
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def _get_closest_value(self, sensor_history, target_time):
        """Get sensor value closest to target time"""
        if not sensor_history:
            return None
        
        best_value = None
        best_diff = float('inf')
        
        for entry in sensor_history:
            entry_time = pd.to_datetime(entry['timestamp'])
            diff = abs((entry_time - target_time).total_seconds())
            
            if diff < best_diff and diff < 3600:  # Within 1 hour
                best_diff = diff
                best_value = entry['value']
        
        return best_value
    
    def _engineer_features(self, df):
        """Engineer features from raw sensor data"""
        # Add time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
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
        for col in ['temperature', 'humidity', 'pressure']:
            for lag in [1, 2, 3, 6, 12, 24, 48]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Rolling statistics
        for col in ['temperature', 'humidity', 'pressure']:
            for window in [6, 12, 24, 48]:
                df[f'{col}_mean_{window}h'] = df[col].rolling(window).mean()
                df[f'{col}_std_{window}h'] = df[col].rolling(window).std()
                
                if col == 'temperature':
                    df[f'{col}_min_{window}h'] = df[col].rolling(window).min()
                    df[f'{col}_max_{window}h'] = df[col].rolling(window).max()
                    df[f'{col}_range_{window}h'] = df[f'{col}_max_{window}h'] - df[f'{col}_min_{window}h']
        
        # Change features
        for col in ['temperature', 'humidity', 'pressure']:
            for period in [1, 3, 6, 12, 24]:
                df[f'{col}_change_{period}h'] = df[col].diff(period)
                if col == 'temperature':
                    df[f'{col}_pct_change_{period}h'] = df[col].pct_change(period)
        
        # EMA features
        for col in ['temperature', 'humidity', 'pressure']:
            for span in [6, 12, 24]:
                df[f'{col}_ema_{span}'] = df[col].ewm(span=span).mean()
        
        # Interaction features
        df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1)
        df['temp_pressure_interaction'] = df['temperature'] * df['pressure']
        df['humidity_pressure_interaction'] = df['humidity'] * df['pressure']
        
        # Derived features
        df['dewpoint'] = df['temperature'] - ((100 - df['humidity']) / 5)
        df['heat_index'] = df['temperature'] + 0.5555 * (
            6.11 * np.exp(5417.7530 * (1/273.16 - 1/(273.15 + df['dewpoint']))) - 10
        )
        
        # Volatility
        df['temp_volatility_6h'] = df['temperature'].rolling(6).std()
        df['temp_volatility_24h'] = df['temperature'].rolling(24).std()
        
        # Trend (simplified for stability)
        df['temp_trend_6h'] = df['temperature'].rolling(6).mean().diff(3)
        df['temp_trend_24h'] = df['temperature'].rolling(24).mean().diff(12)
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Add any missing features with default values
        for col in self.feature_columns:
            if col not in df.columns:
                if 'mean' in col:
                    base_col = col.split('_')[0]
                    if base_col in df.columns:
                        df[col] = df[base_col].mean()
                    else:
                        df[col] = 0
                else:
                    df[col] = 0
        
        return df
    
    def _default_prediction(self):
        """Return default prediction when unable to predict"""
        return {
            'prediction': 'stable',
            'confidence': 33.3,
            'probabilities': {
                'increase': 33.3,
                'decrease': 33.3,
                'stable': 33.4
            },
            'timestamp': datetime.now().isoformat(),
            'error': 'Insufficient data for prediction'
        }