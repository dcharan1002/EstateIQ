"""Model registry for EstateIQ."""
import logging
from pathlib import Path
from random_forest import RandomForestModel
from xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)

# Constants from base model
MODEL_DIR = Path("artifacts")
RESULTS_DIR = Path("results")

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

AVAILABLE_MODELS = {
    'random_forest': {
        'class': RandomForestModel,
        'default_params': {
            'n_estimators': 5,
            'random_state': 42
        }
    },
    'xgboost': {
        'class': XGBoostModel,
        'default_params': {
            'n_estimators': 5,
            'learning_rate': 0.1,
            'random_state': 42
        }
    }
}

def get_model(model_name, **kwargs):
    """
    Get a model instance by name.
    
    Args:
        model_name (str): Name of the model to get
        **kwargs: Additional parameters to pass to the model
    
    Returns:
        BaseModel: Instance of the requested model
        
    Raises:
        ValueError: If model_name is not found in AVAILABLE_MODELS
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {list(AVAILABLE_MODELS.keys())}"
        )
    
    model_info = AVAILABLE_MODELS[model_name]
    model_class = model_info['class']
    
    # Combine default params with provided kwargs
    params = model_info['default_params'].copy()
    params.update(kwargs)
    
    return model_class(**params)

def list_models():
    """List all available models and their default parameters."""
    return {
        name: {
            'class': info['class'].__name__,
            'default_params': info['default_params']
        }
        for name, info in AVAILABLE_MODELS.items()
    }
