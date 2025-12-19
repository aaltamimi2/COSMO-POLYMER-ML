"""
Correction model for COSMO-therm SLE predictions
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional
import joblib
from pathlib import Path


class SLECorrectionModel:
    """
    Model to correct COSMO-therm SLE predictions

    Usage:
        model = SLECorrectionModel(model_type='ridge')
        model.fit(X_train, y_train)
        corrections = model.predict(X_test)
        corrected_sle = sle_predictions - corrections
    """

    def __init__(self, model_type: str = 'ridge', scale_features: bool = True, **model_params):
        """
        Initialize correction model

        Parameters:
        -----------
        model_type : str
            Type of model: 'ridge', 'lasso', 'elastic', 'rf', 'gbm'
        scale_features : bool
            Whether to standardize features
        **model_params : dict
            Additional parameters for the model
        """
        self.model_type = model_type
        self.scale_features = scale_features
        self.model_params = model_params

        # Initialize model
        self.model = self._create_model()

        # Scaler
        self.scaler = StandardScaler() if scale_features else None

        # Store feature names
        self.feature_names = None

    def _create_model(self):
        """Create the underlying model"""
        if self.model_type == 'ridge':
            return Ridge(alpha=self.model_params.get('alpha', 1.0))
        elif self.model_type == 'lasso':
            return Lasso(alpha=self.model_params.get('alpha', 1.0))
        elif self.model_type == 'elastic':
            return ElasticNet(
                alpha=self.model_params.get('alpha', 1.0),
                l1_ratio=self.model_params.get('l1_ratio', 0.5)
            )
        elif self.model_type == 'rf':
            return RandomForestRegressor(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', None),
                min_samples_split=self.model_params.get('min_samples_split', 2),
                random_state=42
            )
        elif self.model_type == 'gbm':
            return GradientBoostingRegressor(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 3),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the correction model

        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target (error to correct)
        """
        self.feature_names = list(X.columns)

        X_array = X.values

        if self.scaler:
            X_array = self.scaler.fit_transform(X_array)

        self.model.fit(X_array, y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict corrections

        Parameters:
        -----------
        X : pd.DataFrame
            Features

        Returns:
        --------
        Array of predicted corrections
        """
        # Ensure columns match training
        if self.feature_names:
            X = X[self.feature_names]

        X_array = X.values

        if self.scaler:
            X_array = self.scaler.transform(X_array)

        return self.model.predict(X_array)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance

        Returns:
        --------
        Dictionary with MAE, RMSE, R2
        """
        y_pred = self.predict(X)

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation

        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        cv : int
            Number of folds

        Returns:
        --------
        Dictionary with cross-validation scores
        """
        X_array = X.values
        if self.scaler:
            X_array = self.scaler.fit_transform(X_array)

        mae_scores = -cross_val_score(
            self.model, X_array, y,
            cv=cv, scoring='neg_mean_absolute_error'
        )

        rmse_scores = np.sqrt(-cross_val_score(
            self.model, X_array, y,
            cv=cv, scoring='neg_mean_squared_error'
        ))

        r2_scores = cross_val_score(
            self.model, X_array, y,
            cv=cv, scoring='r2'
        )

        return {
            'MAE': mae_scores,
            'RMSE': rmse_scores,
            'R2': r2_scores
        }

    def save(self, filepath: str):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'model_params': self.model_params
        }
        joblib.dump(model_data, filepath)

    @classmethod
    def load(cls, filepath: str):
        """Load model from disk"""
        model_data = joblib.load(filepath)

        instance = cls(
            model_type=model_data['model_type'],
            **model_data['model_params']
        )
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']

        return instance

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance (for tree-based models)

        Returns:
        --------
        DataFrame with feature names and importances
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            return df
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            coefs = np.abs(self.model.coef_)
            df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': coefs
            }).sort_values('importance', ascending=False)
            return df
        else:
            return None


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5
) -> pd.DataFrame:
    """
    Compare different model types using cross-validation

    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    cv : int
        Number of CV folds

    Returns:
    --------
    DataFrame with comparison results
    """
    models_to_test = {
        'Ridge': SLECorrectionModel('ridge', alpha=1.0),
        'Lasso': SLECorrectionModel('lasso', alpha=0.1),
        'ElasticNet': SLECorrectionModel('elastic', alpha=0.1, l1_ratio=0.5),
        'Random Forest': SLECorrectionModel('rf', n_estimators=100, max_depth=10),
        'Gradient Boosting': SLECorrectionModel('gbm', n_estimators=100, max_depth=3, learning_rate=0.1)
    }

    results = []

    for name, model in models_to_test.items():
        print(f"Testing {name}...")
        cv_scores = model.cross_validate(X, y, cv=cv)

        results.append({
            'Model': name,
            'MAE_mean': cv_scores['MAE'].mean(),
            'MAE_std': cv_scores['MAE'].std(),
            'RMSE_mean': cv_scores['RMSE'].mean(),
            'RMSE_std': cv_scores['RMSE'].std(),
            'R2_mean': cv_scores['R2'].mean(),
            'R2_std': cv_scores['R2'].std()
        })

    df_results = pd.DataFrame(results).sort_values('MAE_mean')
    return df_results


def apply_correction(
    sle_predictions: pd.Series,
    corrections: np.ndarray
) -> pd.Series:
    """
    Apply corrections to SLE predictions

    Parameters:
    -----------
    sle_predictions : pd.Series
        Original SLE predictions
    corrections : np.ndarray
        Predicted corrections (error in SLE)

    Returns:
    --------
    Corrected predictions
    """
    return sle_predictions - corrections
