import logging
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
from sklearn.preprocessing import LabelEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path("artifacts")
RESULTS_DIR = Path("results")

def setup_directories():
    """Create necessary directories if they don't exist."""
    logger.info("Setting up directories...")
    try:
        for directory in [MODEL_DIR, RESULTS_DIR]:
            if not directory.exists():
                directory.mkdir(parents=True)
                logger.info(f"Created directory: {directory}")
            elif not directory.is_dir():
                directory.unlink()
                directory.mkdir(parents=True)
                logger.info(f"Replaced file with directory: {directory}")
    except Exception as e:
        logger.error(f"Failed to setup directories: {str(e)}")
        raise

def load_data(data_path):
    """Load training and test datasets."""
    try:
        data_path = Path(data_path)
        X_train = pd.read_csv(data_path / "X_train.csv", low_memory=False)
        X_test = pd.read_csv(data_path / "X_test.csv", low_memory=False)
        y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
        y_test = pd.read_csv(data_path / "y_test.csv").squeeze()
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def encode_categorical_features(X_train, X_test):
    """Encode categorical features using LabelEncoder."""
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    cat_cols = X_train.select_dtypes(include=['object']).columns
    
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([X_train[col], X_test[col]], axis=0).astype(str))
        X_train_encoded[col] = le.transform(X_train[col].astype(str))
        X_test_encoded[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le
    
    # Save encoders for future use
    np.save(MODEL_DIR / "label_encoders.npy", encoders)
    return X_train_encoded, X_test_encoded

def evaluate_model(model_name, y_true, y_pred):
    """Evaluate model performance and log metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    logger.info(f"\n{model_name} Evaluation:")
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")

    mlflow.log_metrics({
        f"{model_name}_R2": r2,
        f"{model_name}_MAE": mae,
        f"{model_name}_MSE": mse,
        f"{model_name}_RMSE": rmse
    })
    
    return {
        'r2': r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }

class BaseModel:
    def __init__(self, name):
        self.name = name
        
    def train(self, X_train, y_train):
        """Train the model."""
        raise NotImplementedError
        
    def predict(self, X):
        """Make predictions."""
        raise NotImplementedError
        
    def tune_hyperparameters(self, X_train, y_train):
        """Tune model hyperparameters."""
        raise NotImplementedError
        
    def save(self, path):
        """Save model to disk."""
        raise NotImplementedError
        
    def load(self, path):
        """Load model from disk."""
        raise NotImplementedError
