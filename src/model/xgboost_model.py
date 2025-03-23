import logging
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from pathlib import Path
from .base import BaseModel, MODEL_DIR, RESULTS_DIR, logger

class XGBoostModel(BaseModel):
    def __init__(self, name="XGBoost", n_estimators=100, **kwargs):
        super().__init__(name)
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            tree_method='hist',  # Better handling of categorical features
            enable_categorical=True,
            **kwargs
        )
        
    def train(self, X_train, y_train):
        """Train the XGBoost model."""
        logger.info(f"\nTraining {self.name}...")
        start_time = pd.Timestamp.now()
        
        # Convert data to DMatrix for better performance
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        
        # Set parameters explicitly
        params = self.model.get_params()
        params['tree_method'] = 'hist'  # Better handling of categorical features
        
        # Train using the native API
        self.model.fit(X_train, y_train)
        
        training_time = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(f"Training completed in {training_time:.2f} seconds")
        mlflow.log_metric(f"{self.name}_training_time", training_time)
        
        return self
        
    def predict(self, X):
        """Make predictions using the XGBoost model."""
        return self.model.predict(X)
        
    def get_feature_importance(self, feature_names):
        """Get feature importance scores."""
        return pd.Series(self.model.feature_importances_, index=feature_names)
        
    def tune_hyperparameters(self, X_train, y_train):
        """Tune XGBoost hyperparameters."""
        logger.info(f"\nTuning hyperparameters for {self.name}...")
        
        param_grid = {
            'n_estimators': [100, 300],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6],
            'subsample': [0.8, 1],
            'colsample_bytree': [0.8, 1]
        }
        
        with mlflow.start_run(run_name=f"{self.name}_tuning", nested=True) as run:
            logger.info(f"Started hyperparameter tuning run: {run.info.run_id}")
            
            # Create base estimator with categorical feature handling
            base_estimator = xgb.XGBRegressor(
                random_state=42,
                tree_method='hist',
                enable_categorical=True
            )
            
            random_search = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=param_grid,
                n_iter=5,
                cv=3,
                verbose=1,
                random_state=42,
                n_jobs=-1
            )
            
            random_search.fit(X_train, y_train)
            
            # Save results
            results = pd.DataFrame(random_search.cv_results_)
            results_path = RESULTS_DIR / f"{self.name.lower()}_hyperparameter_results.csv"
            results.to_csv(results_path, index=False)
            mlflow.log_artifact(str(results_path))
            
            # Log metrics and parameters
            mlflow.log_metric(f"best_cv_score", random_search.best_score_)
            mlflow.log_params(random_search.best_params_)
            
            logger.info(f"Best CV score: {random_search.best_score_:.4f}")
            logger.info("Best parameters:")
            for param, value in random_search.best_params_.items():
                logger.info(f"{param}: {value}")
            
            self.model = random_search.best_estimator_
            
        return self
        
    def save(self, path=None):
        """Save XGBoost model."""
        if path is None:
            path = MODEL_DIR / f"{self.name.lower()}_model.json"
        mlflow.xgboost.log_model(self.model, f"{self.name}_model")
        self.model.save_model(path)
        logger.info(f"Model saved to {path}")
        
    def load(self, path):
        """Load XGBoost model."""
        self.model = xgb.XGBRegressor(
            tree_method='hist',
            enable_categorical=True
        )
        self.model.load_model(path)
        return self
