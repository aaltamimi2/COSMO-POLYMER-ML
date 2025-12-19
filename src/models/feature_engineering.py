"""
Feature engineering for COSMO-therm SLE correction model
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temperature-based features

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'Temperature_K' column

    Returns:
    --------
    DataFrame with additional temperature features
    """
    df = df.copy()

    # Temperature in Celsius
    df['Temperature_C'] = df['Temperature_K'] - 273.15

    # Inverse temperature (1/T) - often linear relationship
    df['inv_T'] = 1.0 / df['Temperature_K']

    # Polynomial features
    df['T_squared'] = df['Temperature_K'] ** 2
    df['T_cubed'] = df['Temperature_K'] ** 3

    # Log temperature
    df['log_T'] = np.log(df['Temperature_K'])

    return df


def add_solvent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add solvent-based features from existing data

    Note: This uses basic features from COSMO data.
    For advanced modeling, consider adding:
    - Hansen solubility parameters
    - Dielectric constant
    - Dipole moment
    - H-bond donor/acceptor counts

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with solvent data

    Returns:
    --------
    DataFrame with additional solvent features
    """
    df = df.copy()

    # Already have: Solvent_density, Solvent_MolWeight

    # Log of molecular weight (often better for ML)
    if 'Solvent_MolWeight' in df.columns:
        df['log_MolWeight'] = np.log(df['Solvent_MolWeight'])

    # Molar volume estimate (MW / density)
    if 'Solvent_MolWeight' in df.columns and 'Solvent_density' in df.columns:
        df['molar_volume'] = df['Solvent_MolWeight'] / df['Solvent_density']

    return df


def add_interaction_features(df: pd.DataFrame, method_suffix: str = '') -> pd.DataFrame:
    """
    Add features based on solute-solvent interactions

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with SLE or REF data
    method_suffix : str
        Suffix for method-specific columns (e.g., '_SLE', '_REF')

    Returns:
    --------
    DataFrame with interaction features
    """
    df = df.copy()

    # Chemical potential difference (if not already present)
    mu_self = f'mu_self{method_suffix}'
    mu_solv = f'mu_solv{method_suffix}'
    delta_mu = f'delta_mu{method_suffix}'

    if mu_self in df.columns and mu_solv in df.columns and delta_mu not in df.columns:
        df[delta_mu] = df[mu_solv] - df[mu_self]

    # Normalized by temperature (ΔG/RT)
    if delta_mu in df.columns and 'Temperature_K' in df.columns:
        R = 1.987204e-3  # kcal/(mol·K)
        df[f'delta_mu_norm{method_suffix}'] = df[delta_mu] / (R * df['Temperature_K'])

    return df


def create_feature_matrix(
    df: pd.DataFrame,
    include_sle_predictions: bool = True,
    method_suffix: str = '_SLE'
) -> pd.DataFrame:
    """
    Create complete feature matrix for modeling

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (typically merged REF-SLE data)
    include_sle_predictions : bool
        Whether to include SLE predictions as features
    method_suffix : str
        Suffix for method columns to use as features

    Returns:
    --------
    DataFrame with all features
    """
    df_feat = df.copy()

    # Add temperature features
    df_feat = add_temperature_features(df_feat)

    # Add solvent features
    df_feat = add_solvent_features(df_feat)

    # Add interaction features
    df_feat = add_interaction_features(df_feat, method_suffix)

    # One-hot encode solvent names (for capturing solvent-specific effects)
    df_feat = pd.get_dummies(df_feat, columns=['Solvent'], prefix='solvent')

    return df_feat


def prepare_training_data(
    merged_df: pd.DataFrame,
    target_variable: str = 'log10_x',
    feature_cols: Optional[List[str]] = None
) -> tuple:
    """
    Prepare X, y for model training

    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged REF-SLE dataframe with features
    target_variable : str
        Which variable to correct ('log10_x', 'log10_S', 'w')
    feature_cols : List[str], optional
        Specific feature columns to use. If None, uses recommended defaults.

    Returns:
    --------
    X_train, y_train, feature_names
    """

    # Create features
    df_feat = create_feature_matrix(merged_df, include_sle_predictions=True)

    # Target: error in SLE prediction (SLE - REF)
    target_ref = f'{target_variable}_REF'
    target_sle = f'{target_variable}_SLE'

    if target_ref not in df_feat.columns or target_sle not in df_feat.columns:
        raise ValueError(f"Target variable {target_variable} not found in dataframe")

    # We're predicting the error (correction needed)
    y = df_feat[target_sle] - df_feat[target_ref]

    # Define default features if not provided
    if feature_cols is None:
        feature_cols = []

        # Temperature features (most important)
        feature_cols.extend(['Temperature_K', 'inv_T', 'Temperature_C'])

        # SLE predictions (what we have for new polymers)
        feature_cols.extend([f'{target_variable}_SLE', f'delta_mu_SLE'])

        # Solvent properties
        if 'Solvent_MolWeight' in df_feat.columns:
            feature_cols.extend(['Solvent_MolWeight', 'log_MolWeight'])
        if 'Solvent_density' in df_feat.columns:
            feature_cols.append('Solvent_density')
        if 'molar_volume' in df_feat.columns:
            feature_cols.append('molar_volume')

        # Chemical potential features
        if 'delta_mu_norm_SLE' in df_feat.columns:
            feature_cols.append('delta_mu_norm_SLE')

        # Solvent one-hot encodings (if you want solvent-specific corrections)
        solvent_cols = [c for c in df_feat.columns if c.startswith('solvent_')]
        # Note: Including these makes the model polymer-specific
        # For generalization, you might want to skip these
        # feature_cols.extend(solvent_cols)

    # Filter to available features
    available_features = [f for f in feature_cols if f in df_feat.columns]

    X = df_feat[available_features]

    return X, y, available_features


def get_simple_features() -> List[str]:
    """
    Get list of simple, generalizable features (no solvent one-hot encoding)
    These features should work across different polymers
    """
    return [
        'Temperature_K',
        'inv_T',
        'Temperature_C',
        'log10_x_SLE',
        'delta_mu_SLE',
        'Solvent_MolWeight',
        'Solvent_density'
    ]


def get_advanced_features() -> List[str]:
    """
    Get list of advanced features including polynomial terms
    """
    base = get_simple_features()
    advanced = base + [
        'T_squared',
        'log_T',
        'log_MolWeight',
        'molar_volume',
        'delta_mu_norm_SLE'
    ]
    return advanced
