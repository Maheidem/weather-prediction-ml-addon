"""
Portable model loader that tries multiple loading strategies
"""
import os
import json
import numpy as np
import joblib
import pickle
import logging

logger = logging.getLogger(__name__)

class PortableModelLoader:
    """Load models with multiple fallback strategies"""
    
    @staticmethod
    def load_xgboost(model_path):
        """Try to load XGBoost model with multiple strategies"""
        import xgboost as xgb
        
        # Try native XGBoost JSON format first
        json_path = model_path.replace('.pkl', '.json')
        if os.path.exists(json_path):
            try:
                model = xgb.XGBClassifier()
                model.load_model(json_path)
                logger.info(f"Loaded XGBoost from JSON: {json_path}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load XGBoost JSON: {e}")
        
        # Try native XGBoost binary format
        bin_path = model_path.replace('.pkl', '.bin')
        if os.path.exists(bin_path):
            try:
                model = xgb.XGBClassifier()
                model.load_model(bin_path)
                logger.info(f"Loaded XGBoost from binary: {bin_path}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load XGBoost binary: {e}")
        
        # Try joblib with different protocols
        if os.path.exists(model_path):
            for protocol in [None, 4, 3, 2]:
                try:
                    if protocol is None:
                        model = joblib.load(model_path)
                    else:
                        # Try loading with mmap_mode for large files
                        model = joblib.load(model_path, mmap_mode='r')
                    logger.info(f"Loaded XGBoost with joblib (protocol={protocol})")
                    return model
                except Exception as e:
                    logger.warning(f"Failed joblib load (protocol={protocol}): {e}")
            
            # Try pickle as last resort
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info("Loaded XGBoost with pickle")
                return model
            except Exception as e:
                logger.warning(f"Failed pickle load: {e}")
        
        raise RuntimeError(f"Could not load XGBoost model from {model_path}")
    
    @staticmethod
    def load_random_forest(model_path):
        """Try to load Random Forest model with multiple strategies"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Try joblib first (most common for sklearn)
        for protocol in [None, 4, 3, 2]:
            try:
                if protocol is None:
                    model = joblib.load(model_path)
                else:
                    model = joblib.load(model_path, mmap_mode='r')
                logger.info(f"Loaded Random Forest with joblib (protocol={protocol})")
                return model
            except Exception as e:
                logger.warning(f"Failed joblib load (protocol={protocol}): {e}")
        
        # Try pickle
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("Loaded Random Forest with pickle")
            return model
        except Exception as e:
            logger.warning(f"Failed pickle load: {e}")
        
        raise RuntimeError(f"Could not load Random Forest model from {model_path}")
    
    @staticmethod
    def load_scaler(scaler_path):
        """Load scaler with fallback to JSON parameters"""
        # Try joblib/pickle first
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                logger.info("Loaded scaler with joblib")
                return scaler
            except:
                try:
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    logger.info("Loaded scaler with pickle")
                    return scaler
                except:
                    pass
        
        # Try JSON parameters
        json_path = scaler_path.replace('.pkl', '_params.json')
        if os.path.exists(json_path):
            try:
                from sklearn.preprocessing import StandardScaler
                with open(json_path, 'r') as f:
                    params = json.load(f)
                
                scaler = StandardScaler()
                scaler.mean_ = np.array(params['mean'])
                scaler.scale_ = np.array(params['scale'])
                scaler.var_ = np.array(params['var'])
                scaler.n_features_in_ = params['n_features']
                scaler.n_samples_seen_ = len(params['mean'])
                scaler.feature_names_in_ = params.get('feature_names', None)
                
                logger.info("Loaded scaler from JSON parameters")
                return scaler
            except Exception as e:
                logger.warning(f"Failed to load scaler from JSON: {e}")
        
        raise RuntimeError(f"Could not load scaler from {scaler_path}")
    
    @staticmethod
    def load_label_encoder(encoder_path):
        """Load label encoder with fallback to JSON"""
        if os.path.exists(encoder_path):
            try:
                encoder = joblib.load(encoder_path)
                logger.info("Loaded label encoder with joblib")
                return encoder
            except:
                try:
                    with open(encoder_path, 'rb') as f:
                        encoder = pickle.load(f)
                    logger.info("Loaded label encoder with pickle")
                    return encoder
                except:
                    pass
        
        # Create a simple encoder if loading fails
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        encoder.classes_ = np.array(['decrease', 'increase', 'stable'])
        logger.warning("Created default label encoder")
        return encoder