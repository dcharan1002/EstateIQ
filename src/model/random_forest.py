import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from .base import BaseModel, MODEL_DIR, RESULTS_DIR, logger

class RandomForestModel(BaseModel):
    def __init__(self, name="RandomForest", n_estimators=100, random_state=42, **kwargs):
        super().__init__(name)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs
        )
        
    def train(self, X_train, y_train):
        logger.info(f"\nTraining {self.name}...")
        start_time = pd.Timestamp.now()
        
        self.model.fit(X_train, y_train)
        
        training_time = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(f"Training completed in {training_time:.2f} seconds")
        mlflow.log_metric(f"{self.name}_training_time", training_time)
        
        return self
        
    def predict(self, X):
        return self.model.predict(X)
        
    def get_feature_importance(self, feature_names):
        return pd.Series(self.model.feature_importances_, index=feature_names)
        
    def tune_hyperparameters(self, X_train, y_train):
        logger.info(f"\nTuning hyperparameters for {self.name}...")
        
        bootstrap_params = {
            'n_estimators': [100, 200, 300, 500, 1000],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 6, 8],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True],
            'max_samples': [0.7, 0.8, 0.9, None]
        }
        
        no_bootstrap_params = {
            'n_estimators': [100, 200, 300, 500, 1000],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 6, 8],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [False],
            'max_samples': [None]  # Only None when bootstrap is False
        }
        
        param_grid = [bootstrap_params, no_bootstrap_params]
        
        with mlflow.start_run(run_name=f"{self.name}_tuning", nested=True) as run:
            logger.info(f"Started hyperparameter tuning run: {run.info.run_id}")
            
            random_search = RandomizedSearchCV(
                estimator=RandomForestRegressor(random_state=42),
                param_distributions=param_grid,
                n_iter=1,
                cv=1,
                verbose=1,
                scoring='neg_root_mean_squared_error',
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
        if path is None:
            path = MODEL_DIR / f"{self.name.lower()}_model.pkl"
        mlflow.sklearn.log_model(self.model, f"{self.name}_model")
        pd.to_pickle(self.model, path)
        logger.info(f"Model saved to {path}")
        
    def load(self, path):
        self.model = pd.read_pickle(path)
        return self
