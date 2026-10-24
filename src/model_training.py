# Standard library imports
import logging
from typing import Any, Dict, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import uniform

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class ModelTraining:
    """
    Class managing train/val/test splitting, baseline model training,
    hyperparameter tuning, and performance evaluations.
    """
    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        self.config = config
        self.preprocessor = preprocessor

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Splits features and target into Train (80%), Val (10%), and Test (10%) sets."""
        X = df.drop(columns=[self.config['target_column'], self.config['log_target_column']])
        y = df[self.config['log_target_column']]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.config['val_test_size'], random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.config['val_size'], random_state=42
        )

        logging.info(f"Data split successfully - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def evaluate_model(self, model: Any, X_data: pd.DataFrame, y_true: pd.Series, label: str = "Evaluation") -> Dict[str, float]:
        """Computes MAE, MSE, RMSE, and R2 regression metrics."""
        y_pred = model.predict(X_data)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
        logging.info(f"{label} Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        return metrics

    def train_baseline_models(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Pipeline]:
        """Fits Linear, Ridge, and Lasso baseline models."""
        pipelines = {
            'LinearRegression': Pipeline([('preprocessor', self.preprocessor), ('regressor', LinearRegression())]),
            'Ridge': Pipeline([('preprocessor', self.preprocessor), ('regressor', Ridge(alpha=1.0, random_state=42))]),
            'Lasso': Pipeline([('preprocessor', self.preprocessor), ('regressor', Lasso(alpha=0.01, random_state=42))])
        }

        fitted_models = {}
        for name, pipe in pipelines.items():
            logging.info(f"Training baseline model: {name}...")
            pipe.fit(X_train, y_train)
            self.evaluate_model(pipe, X_val, y_val, label=f"{name} (Val)")
            fitted_models[name] = pipe

        return fitted_models

    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """Tunes Ridge regression via GridSearchCV."""
        logging.info("Starting GridSearchCV hyperparameter tuning for Ridge Regression...")
        base_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', Ridge(random_state=42))
        ])

        grid_search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=self.config['param_grid'],
            cv=self.config['cv'],
            scoring=self.config['scoring'],
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        logging.info(f"Best Ridge Alpha: {grid_search.best_params_} | Best CV R2: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
