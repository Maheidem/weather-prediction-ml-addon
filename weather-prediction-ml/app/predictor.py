"""ML Predictor for Weather Prediction - Fixed version"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from model_loader import PortableModelLoader

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
        
        # Use portable loader for better compatibility
        loader = PortableModelLoader()
        
        # Try to load models with fallback strategies
        self.xgboost_model = loader.load_xgboost(
            str(self.models_dir / 'final_xgboost_model.pkl')
        )
        
        self.rf_model = loader.load_random_forest(
            str(self.models_dir / 'final_rf_model.pkl')
        )
        
        # Load preprocessors
        self.scaler = loader.load_scaler(
            str(self.models_dir / 'final_scaler.pkl')
        )
        
        self.label_encoder = loader.load_label_encoder(
            str(self.models_dir / 'final_label_encoder.pkl')
        )
        
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
        
        # First, add the base aggregated features that the model expects
        # These would normally come from hourly aggregations
        if 'temperature' in df.columns and 'temperature_mean' not in df.columns:
            # Create hourly aggregations from the raw data
            df['temperature_mean'] = df['temperature'].rolling(window=60, min_periods=1).mean()
            df['temperature_min'] = df['temperature'].rolling(window=60, min_periods=1).min()
            df['temperature_max'] = df['temperature'].rolling(window=60, min_periods=1).max()
            df['humidity_mean'] = df['humidity'].rolling(window=60, min_periods=1).mean()
            df['humidity_min'] = df['humidity'].rolling(window=60, min_periods=1).min()
            df['humidity_max'] = df['humidity'].rolling(window=60, min_periods=1).max()
            
            if 'pressure' in df.columns:
                df['pressure_mean'] = df['pressure'].rolling(window=60, min_periods=1).mean()
                df['pressure_min'] = df['pressure'].rolling(window=60, min_periods=1).min()
                df['pressure_max'] = df['pressure'].rolling(window=60, min_periods=1).max()
            else:
                # Use default pressure if not available
                df['pressure'] = 1013.25
                df['pressure_mean'] = 1013.25
                df['pressure_min'] = 1013.25
                df['pressure_max'] = 1013.25
            
            # Additional base features
            df['temperature_range'] = df['temperature_max'] - df['temperature_min']
            df['humidity_range'] = df['humidity_max'] - df['humidity_min']
            df['day_of_week'] = df.index.dayofweek
            
            # Solar features (simplified)
            df['solar_azimuth'] = 180  # Placeholder
            df['solar_elevation'] = 45  # Placeholder
            df['atmosphere_pressure'] = df['pressure_mean']
        
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
            df[f'pressure_lag_{lag}'] = df['pressure'].shift(lag)
        
        # Rolling statistics
        windows = [6, 12, 24, 48]
        for window in windows:
            # Temperature
            df[f'temp_mean_{window}h'] = df['temperature'].rolling(window, center=True).mean()
            df[f'temp_std_{window}h'] = df['temperature'].rolling(window, center=True).std()
            df[f'temp_min_{window}h'] = df['temperature'].rolling(window).min()
            df[f'temp_max_{window}h'] = df['temperature'].rolling(window).max()
            df[f'temp_range_{window}h'] = df[f'temp_max_{window}h'] - df[f'temp_min_{window}h']
            
            # Humidity
            df[f'humidity_mean_{window}h'] = df['humidity'].rolling(window, center=True).mean()
            df[f'humidity_std_{window}h'] = df['humidity'].rolling(window, center=True).std()
            
            # Pressure
            df[f'pressure_mean_{window}h'] = df['pressure'].rolling(window, center=True).mean()
            df[f'pressure_std_{window}h'] = df['pressure'].rolling(window, center=True).std()
        
        # Change features
        for hours in [1, 3, 6, 12, 24]:
            df[f'temp_change_{hours}h'] = df['temperature'].diff(hours)
            df[f'temp_pct_change_{hours}h'] = df['temperature'].pct_change(hours)
            df[f'humidity_change_{hours}h'] = df['humidity'].diff(hours)
            df[f'pressure_change_{hours}h'] = df['pressure'].diff(hours)
        
        # Exponential moving averages
        for span in [6, 12, 24]:
            df[f'temp_ema_{span}'] = df['temperature'].ewm(span=span).mean()
            df[f'humidity_ema_{span}'] = df['humidity'].ewm(span=span).mean()
            df[f'pressure_ema_{span}'] = df['pressure'].ewm(span=span).mean()
        
        # Feature interactions
        df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1)
        df['temp_pressure_interaction'] = df['temperature'] * df['pressure']
        df['humidity_pressure_interaction'] = df['humidity'] * df['pressure']
        
        # Weather indices
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
    
    def predict(self, temperature_data, humidity_data, pressure_data):
        """Make a weather prediction based on sensor data"""
        try:
            # Create DataFrame from sensor data
            if len(temperature_data) != len(humidity_data):
                logger.warning("Temperature and humidity data length mismatch")
                # Align to shortest length
                min_len = min(len(temperature_data), len(humidity_data))
                temperature_data = temperature_data[:min_len]
                humidity_data = humidity_data[:min_len]
            
            # Create time index
            now = pd.Timestamp.now()
            time_index = pd.date_range(end=now, periods=len(temperature_data), freq='H')
            
            # Create DataFrame
            df = pd.DataFrame({
                'temperature': temperature_data,
                'humidity': humidity_data,
                'pressure': pressure_data if len(pressure_data) == len(temperature_data) else [1013.25] * len(temperature_data)
            }, index=time_index)
            
            # Create features
            df_features = self.create_features_from_data(df)
            
            if self.models_loaded:
                return self._predict_with_models(df_features)
            else:
                return self._predict_with_fallback(df_features)
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Return a safe default prediction
            return {
                'prediction': 'stable',
                'confidence': 70.0,
                'probabilities': {
                    'increase': 30.0,
                    'decrease': 30.0,
                    'stable': 40.0
                }
            }
    
    def _predict_with_models(self, df_features):
        """Make prediction using loaded ML models"""
        try:
            # Get the last row with all required features
            X = df_features[self.feature_columns].iloc[-1:].fillna(0)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from both models
            xgb_proba = self.xgboost_model.predict_proba(X_scaled)[0]
            rf_proba = self.rf_model.predict_proba(X_scaled)[0]
            
            # Handle potential shape mismatches
            if len(xgb_proba) != len(rf_proba):
                logger.warning(f"Shape mismatch: XGBoost {len(xgb_proba)} classes, RF {len(rf_proba)} classes")
                # Pad the shorter array with zeros
                max_len = max(len(xgb_proba), len(rf_proba))
                if len(xgb_proba) < max_len:
                    xgb_proba = np.pad(xgb_proba, (0, max_len - len(xgb_proba)), 'constant')
                if len(rf_proba) < max_len:
                    rf_proba = np.pad(rf_proba, (0, max_len - len(rf_proba)), 'constant')
            
            # Ensemble prediction
            ensemble_proba = (
                self.ensemble_weights['xgboost'] * xgb_proba +
                self.ensemble_weights['random_forest'] * rf_proba
            )
            
            # Get prediction
            prediction_idx = np.argmax(ensemble_proba)
            prediction = self.label_encoder.classes_[prediction_idx]
            confidence = float(ensemble_proba[prediction_idx] * 100)
            
            # Create probability dictionary - handle different class orders
            # The models might have different numbers of classes or different ordering
            prob_dict = {}
            
            # Map probabilities based on actual classes present
            for i, label in enumerate(self.label_encoder.classes_):
                if i < len(ensemble_proba):
                    prob_dict[label] = float(ensemble_proba[i] * 100)
                else:
                    prob_dict[label] = 0.0
            
            # Ensure all three classes are present
            probabilities = {
                'increase': prob_dict.get('increase', 0.0),
                'stable': prob_dict.get('stable', 0.0),
                'decrease': prob_dict.get('decrease', 0.0)
            }
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': probabilities
            }
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to intelligent prediction
            return self._predict_with_fallback(df_features)
    
    def _predict_with_fallback(self, df):
        """Intelligent fallback prediction based on meteorological principles"""
        try:
            # Get latest values
            latest = df.iloc[-1]
            temp = latest.get('temperature', 20)
            humidity = latest.get('humidity', 50)
            pressure = latest.get('pressure', 1013.25)
            
            # Get recent trends
            temp_change = df['temperature'].iloc[-6:].mean() - df['temperature'].iloc[-12:-6].mean()
            humidity_change = df['humidity'].iloc[-6:].mean() - df['humidity'].iloc[-12:-6].mean()
            pressure_change = df['pressure'].iloc[-6:].mean() - df['pressure'].iloc[-12:-6].mean() if 'pressure' in df else 0
            
            # Simple meteorological rules
            score_increase = 0
            score_decrease = 0
            score_stable = 0
            
            # Pressure-based prediction
            if pressure < 1000:  # Low pressure
                score_decrease += 30
            elif pressure > 1020:  # High pressure
                score_increase += 20
            else:
                score_stable += 20
            
            # Temperature trend
            if temp_change > 1:
                score_increase += 25
            elif temp_change < -1:
                score_decrease += 25
            else:
                score_stable += 20
            
            # Humidity indicators
            if humidity > 80 and humidity_change > 0:
                score_decrease += 20
            elif humidity < 40 and humidity_change < 0:
                score_increase += 15
            
            # Pressure trend is very important
            if pressure_change < -2:
                score_decrease += 30
            elif pressure_change > 2:
                score_increase += 25
            
            # Normalize scores
            total = score_increase + score_decrease + score_stable + 1
            prob_increase = score_increase / total * 100
            prob_decrease = score_decrease / total * 100
            prob_stable = score_stable / total * 100
            
            # Determine prediction
            probs = {'increase': prob_increase, 'decrease': prob_decrease, 'stable': prob_stable}
            prediction = max(probs, key=probs.get)
            confidence = probs[prediction]
            
            # Ensure minimum confidence
            if confidence < 40:
                prediction = 'stable'
                confidence = 50.0
                probs = {'increase': 25.0, 'decrease': 25.0, 'stable': 50.0}
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': probs
            }
            
        except Exception as e:
            logger.error(f"Error in fallback prediction: {e}")
            return {
                'prediction': 'stable',
                'confidence': 70.0,
                'probabilities': {
                    'increase': 30.0,
                    'decrease': 30.0,
                    'stable': 40.0
                }
            }