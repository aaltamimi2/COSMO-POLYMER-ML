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


def train_universal_model(
    polymer_data_dict: Dict[str, pd.DataFrame],
    target_variable: str = 'log10_x',
    include_polymer_id: bool = True,
    model_type: str = 'ridge',
    cv: int = 5,
    **model_params
) -> Tuple[SLECorrectionModel, pd.DataFrame]:
    """
    Train a universal correction model on multiple polymers

    This function combines data from multiple polymers to train a single
    correction model that can generalize across different polymer types.

    Parameters:
    -----------
    polymer_data_dict : Dict[str, pd.DataFrame]
        Dictionary mapping polymer names to merged dataframes
        e.g., {'evoh': df_evoh, 'ps': df_ps, 'pc': df_pc}
    target_variable : str
        Variable to correct ('log10_x', 'log10_S', 'w')
    include_polymer_id : bool
        Whether to include polymer type as a feature (one-hot encoded)
        Set to True for polymer-aware universal model
        Set to False for fully polymer-agnostic model
    model_type : str
        Type of model: 'ridge', 'lasso', 'elastic', 'rf', 'gbm'
    cv : int
        Number of CV folds for evaluation
    **model_params : dict
        Additional parameters for the model

    Returns:
    --------
    model : SLECorrectionModel
        Trained universal model
    cv_results : pd.DataFrame
        Cross-validation results showing performance

    Example:
    --------
    >>> data_dict = {
    ...     'evoh': pd.read_csv('evoh/merged_comparison.csv'),
    ...     'ps': pd.read_csv('ps/merged_comparison.csv'),
    ...     'pc': pd.read_csv('pc/merged_comparison.csv')
    ... }
    >>> model, cv_results = train_universal_model(data_dict, model_type='ridge')
    """
    from models.feature_engineering import prepare_training_data

    print("="*80)
    print(f"Training Universal Correction Model")
    print("="*80)
    print(f"Polymers: {list(polymer_data_dict.keys())}")
    print(f"Target variable: {target_variable}")
    print(f"Include polymer ID: {include_polymer_id}")
    print(f"Model type: {model_type}")

    # Combine all polymer data
    combined_data = []
    for polymer_name, df in polymer_data_dict.items():
        df_copy = df.copy()
        df_copy['polymer'] = polymer_name
        combined_data.append(df_copy)

    combined_df = pd.concat(combined_data, ignore_index=True)
    print(f"\nTotal data points: {len(combined_df)}")
    for polymer_name in polymer_data_dict.keys():
        count = (combined_df['polymer'] == polymer_name).sum()
        print(f"  {polymer_name}: {count}")

    # Prepare features
    print("\nPreparing features...")
    X, y, feature_names = prepare_training_data(
        combined_df,
        target_variable=target_variable,
        feature_cols=None
    )

    # Add polymer one-hot encoding if requested
    if include_polymer_id:
        polymer_dummies = pd.get_dummies(combined_df['polymer'], prefix='polymer')
        X = pd.concat([X, polymer_dummies], axis=1)
        feature_names = list(X.columns)
        print(f"Added polymer identifiers: {list(polymer_dummies.columns)}")

    print(f"Total features: {len(feature_names)}")

    # Train model
    print(f"\nTraining {model_type} model...")
    model = SLECorrectionModel(model_type=model_type, **model_params)
    model.fit(X, y)

    # Cross-validation
    print(f"\nPerforming {cv}-fold cross-validation...")
    cv_scores = model.cross_validate(X, y, cv=cv)

    cv_results = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'R2'],
        'Mean': [cv_scores['MAE'].mean(), cv_scores['RMSE'].mean(), cv_scores['R2'].mean()],
        'Std': [cv_scores['MAE'].std(), cv_scores['RMSE'].std(), cv_scores['R2'].std()]
    })

    print("\nCross-Validation Results:")
    print(cv_results.to_string(index=False))

    # Feature importance
    importance = model.get_feature_importance()
    if importance is not None:
        print("\nTop 10 Most Important Features:")
        print(importance.head(10).to_string(index=False))

    return model, cv_results


def compare_polymer_performance(
    polymer_data_dict: Dict[str, pd.DataFrame],
    model: SLECorrectionModel,
    target_variable: str = 'log10_x'
) -> pd.DataFrame:
    """
    Evaluate model performance on each polymer separately

    This function shows how well a universal model performs on each
    individual polymer type, helping identify if the model works better
    for some polymers than others.

    Parameters:
    -----------
    polymer_data_dict : Dict[str, pd.DataFrame]
        Dictionary mapping polymer names to merged dataframes
    model : SLECorrectionModel
        Trained correction model (can be universal or polymer-specific)
    target_variable : str
        Variable being corrected ('log10_x', 'log10_S', 'w')

    Returns:
    --------
    pd.DataFrame
        Performance metrics (MAE, RMSE, R2) for each polymer, plus improvement metrics

    Example:
    --------
    >>> results = compare_polymer_performance(data_dict, trained_model)
    >>> print(results.sort_values('MAE_corrected'))
    """
    from models.feature_engineering import prepare_training_data

    print("="*80)
    print("Comparing Performance Across Polymers")
    print("="*80)

    results = []

    for polymer_name, df in polymer_data_dict.items():
        print(f"\nEvaluating on {polymer_name} ({len(df)} points)...")

        # Add polymer identifier
        df_copy = df.copy()
        df_copy['polymer'] = polymer_name

        # Prepare features
        X, y, _ = prepare_training_data(
            df_copy,
            target_variable=target_variable,
            feature_cols=None
        )

        # Add polymer dummies if model expects them
        if any('polymer_' in feat for feat in model.feature_names):
            polymer_dummies = pd.get_dummies(df_copy['polymer'], prefix='polymer')
            X = pd.concat([X, polymer_dummies], axis=1)

        # Ensure all expected features are present
        for feat in model.feature_names:
            if feat not in X.columns:
                X[feat] = 0.0

        # Get predictions
        y_pred = model.predict(X[model.feature_names])

        # Calculate metrics for correction quality
        mae_correction = mean_absolute_error(y, y_pred)
        rmse_correction = np.sqrt(mean_squared_error(y, y_pred))
        r2_correction = r2_score(y, y_pred)

        # Calculate improvement in SLE predictions
        sle_values = df[f'{target_variable}_SLE']
        ref_values = df[f'{target_variable}_REF']
        corrected_values = apply_correction(sle_values, y_pred)

        mae_original = mean_absolute_error(ref_values, sle_values)
        mae_corrected = mean_absolute_error(ref_values, corrected_values)
        improvement = mae_original - mae_corrected
        improvement_pct = 100 * (1 - mae_corrected / mae_original)

        results.append({
            'Polymer': polymer_name,
            'N_points': len(df),
            'MAE_correction': mae_correction,
            'RMSE_correction': rmse_correction,
            'R2_correction': r2_correction,
            'MAE_original': mae_original,
            'MAE_corrected': mae_corrected,
            'Improvement': improvement,
            'Improvement_pct': improvement_pct
        })

    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    print(results_df.to_string(index=False))

    return results_df


def cross_polymer_validation(
    polymer_data_dict: Dict[str, pd.DataFrame],
    target_variable: str = 'log10_x',
    model_type: str = 'ridge',
    include_polymer_id: bool = False,
    **model_params
) -> pd.DataFrame:
    """
    Leave-one-polymer-out cross-validation

    This function trains on N-1 polymers and tests on the remaining one,
    rotating through all polymers. This shows how well the model generalizes
    to completely new polymer types.

    Parameters:
    -----------
    polymer_data_dict : Dict[str, pd.DataFrame]
        Dictionary mapping polymer names to merged dataframes
    target_variable : str
        Variable to correct ('log10_x', 'log10_S', 'w')
    model_type : str
        Type of model to use
    include_polymer_id : bool
        Whether to include polymer type as feature (typically False for this test)
    **model_params : dict
        Additional model parameters

    Returns:
    --------
    pd.DataFrame
        Results showing performance when each polymer is held out

    Example:
    --------
    >>> # Test generalization to new polymers
    >>> results = cross_polymer_validation(data_dict, model_type='ridge')
    >>> print(results)
    """
    from models.feature_engineering import prepare_training_data

    print("="*80)
    print("Leave-One-Polymer-Out Cross-Validation")
    print("="*80)
    print(f"Polymers: {list(polymer_data_dict.keys())}")
    print(f"Model: {model_type}")
    print(f"Include polymer ID: {include_polymer_id}")

    results = []
    polymer_names = list(polymer_data_dict.keys())

    for test_polymer in polymer_names:
        print(f"\n{'='*80}")
        print(f"Holding out: {test_polymer}")
        print('='*80)

        # Split into train and test
        train_polymers = [p for p in polymer_names if p != test_polymer]
        print(f"Training on: {train_polymers}")

        # Prepare training data
        train_data = []
        for polymer_name in train_polymers:
            df_copy = polymer_data_dict[polymer_name].copy()
            df_copy['polymer'] = polymer_name
            train_data.append(df_copy)

        train_df = pd.concat(train_data, ignore_index=True)
        print(f"Training points: {len(train_df)}")

        # Prepare test data
        test_df = polymer_data_dict[test_polymer].copy()
        test_df['polymer'] = test_polymer
        print(f"Test points: {len(test_df)}")

        # Prepare features for training
        X_train, y_train, feature_names = prepare_training_data(
            train_df,
            target_variable=target_variable,
            feature_cols=None
        )

        if include_polymer_id:
            polymer_dummies = pd.get_dummies(train_df['polymer'], prefix='polymer')
            X_train = pd.concat([X_train, polymer_dummies], axis=1)

        # Train model
        model = SLECorrectionModel(model_type=model_type, **model_params)
        model.fit(X_train, y_train)

        # Prepare test features
        X_test, y_test, _ = prepare_training_data(
            test_df,
            target_variable=target_variable,
            feature_cols=None
        )

        if include_polymer_id:
            test_dummies = pd.get_dummies(test_df['polymer'], prefix='polymer')
            X_test = pd.concat([X_test, test_dummies], axis=1)

        # Ensure all features are present
        for feat in model.feature_names:
            if feat not in X_test.columns:
                X_test[feat] = 0.0

        # Predict on test polymer
        y_pred = model.predict(X_test[model.feature_names])

        # Evaluate correction quality
        mae_correction = mean_absolute_error(y_test, y_pred)
        rmse_correction = np.sqrt(mean_squared_error(y_test, y_pred))
        r2_correction = r2_score(y_test, y_pred)

        # Evaluate SLE improvement
        sle_values = test_df[f'{target_variable}_SLE']
        ref_values = test_df[f'{target_variable}_REF']
        corrected_values = apply_correction(sle_values, y_pred)

        mae_original = mean_absolute_error(ref_values, sle_values)
        mae_corrected = mean_absolute_error(ref_values, corrected_values)
        improvement = mae_original - mae_corrected
        improvement_pct = 100 * (1 - mae_corrected / mae_original)

        print(f"\nResults for {test_polymer}:")
        print(f"  MAE (original): {mae_original:.3f}")
        print(f"  MAE (corrected): {mae_corrected:.3f}")
        print(f"  Improvement: {improvement:.3f} ({improvement_pct:.1f}%)")

        results.append({
            'Test_Polymer': test_polymer,
            'Train_Polymers': ', '.join(train_polymers),
            'N_train': len(train_df),
            'N_test': len(test_df),
            'MAE_correction': mae_correction,
            'RMSE_correction': rmse_correction,
            'R2_correction': r2_correction,
            'MAE_original': mae_original,
            'MAE_corrected': mae_corrected,
            'Improvement': improvement,
            'Improvement_pct': improvement_pct
        })

    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("Leave-One-Polymer-Out Results Summary")
    print("="*80)
    print(results_df[['Test_Polymer', 'MAE_original', 'MAE_corrected', 'Improvement_pct']].to_string(index=False))

    return results_df
