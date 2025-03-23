import logging
from pathlib import Path
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
