"""
Train correction model for COSMO-therm SLE predictions

This script trains a model on EVOH data that can be applied to predict
corrections for polymers without experimental reference data.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.feature_engineering import (
    prepare_training_data,
    get_simple_features,
    get_advanced_features
)
from models.correction_model import (
    SLECorrectionModel,
    compare_models,
    apply_correction
)


def train_and_evaluate(
    merged_csv: str,
    target_variable: str = 'log10_x',
    model_type: str = 'ridge',
    output_dir: str = None
):
    """
    Train correction model on single polymer data

    Parameters:
    -----------
    merged_csv : str
        Path to merged REF-SLE CSV file
    target_variable : str
        Variable to correct ('log10_x', 'log10_S', 'w')
    model_type : str
        Type of model to use
    output_dir : str
        Directory to save outputs
    """

    print("="*80)
    print(f"Training SLE Correction Model for {target_variable}")
    print("="*80)

    # Load data
    print(f"\nLoading data from {merged_csv}...")
    merged_df = pd.read_csv(merged_csv)
    print(f"  Loaded {len(merged_df)} data points")

    # Prepare training data
    print("\nPreparing features...")
    X, y, feature_names = prepare_training_data(
        merged_df,
        target_variable=target_variable,
        feature_cols=None  # Use defaults
    )

    print(f"  Features ({len(feature_names)}): {feature_names}")
    print(f"\nTarget statistics:")
    print(f"  Mean error: {y.mean():.3f}")
    print(f"  Std error: {y.std():.3f}")
    print(f"  Min error: {y.min():.3f}")
    print(f"  Max error: {y.max():.3f}")

    # Compare models if requested
    print("\n" + "="*80)
    print("Comparing Different Models (5-fold CV)")
    print("="*80)
    comparison = compare_models(X, y, cv=5)
    print("\n", comparison.to_string(index=False))

    # Train best model
    print("\n" + "="*80)
    print(f"Training {model_type} model on full dataset")
    print("="*80)

    model = SLECorrectionModel(model_type=model_type)
    model.fit(X, y)

    # Evaluate on training data (just to see fit quality)
    metrics = model.evaluate(X, y)
    print(f"\nTraining Set Performance:")
    print(f"  MAE:  {metrics['MAE']:.3f}")
    print(f"  RMSE: {metrics['RMSE']:.3f}")
    print(f"  R²:   {metrics['R2']:.3f}")

    # Get predictions
    y_pred = model.predict(X)

    # Calculate corrected SLE values
    sle_values = merged_df[f'{target_variable}_SLE']
    ref_values = merged_df[f'{target_variable}_REF']
    corrected_values = apply_correction(sle_values, y_pred)

    # Calculate improvement
    original_mae = np.abs(sle_values - ref_values).mean()
    corrected_mae = np.abs(corrected_values - ref_values).mean()

    print(f"\nImprovement:")
    print(f"  Original SLE MAE:  {original_mae:.3f}")
    print(f"  Corrected SLE MAE: {corrected_mae:.3f}")
    print(f"  Improvement:       {original_mae - corrected_mae:.3f} ({100*(1-corrected_mae/original_mae):.1f}% reduction)")

    # Feature importance
    importance = model.get_feature_importance()
    if importance is not None:
        print(f"\nTop 10 Most Important Features:")
        print(importance.head(10).to_string(index=False))

    # Save outputs
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = output_path / f'correction_model_{target_variable}.joblib'
        model.save(str(model_file))
        print(f"\nSaved model to: {model_file}")

        # Save comparison results
        comparison_file = output_path / f'model_comparison_{target_variable}.csv'
        comparison.to_csv(comparison_file, index=False)
        print(f"Saved comparison to: {comparison_file}")

        # Save predictions
        results_df = merged_df.copy()
        results_df[f'{target_variable}_corrected'] = corrected_values
        results_df[f'{target_variable}_correction'] = y_pred
        results_file = output_path / f'corrected_predictions_{target_variable}.csv'
        results_df.to_csv(results_file, index=False)
        print(f"Saved predictions to: {results_file}")

        # Create visualization
        create_correction_plot(
            ref_values, sle_values, corrected_values,
            target_variable,
            output_path / f'correction_results_{target_variable}.png'
        )

    return model, metrics, comparison


def create_correction_plot(
    ref_values: pd.Series,
    sle_values: pd.Series,
    corrected_values: pd.Series,
    variable_name: str,
    output_file: str
):
    """Create visualization showing correction results"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Before correction
    ax = axes[0]
    ax.scatter(ref_values, sle_values, alpha=0.5, s=30)

    # 1:1 line
    min_val = min(ref_values.min(), sle_values.min())
    max_val = max(ref_values.max(), sle_values.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.5)

    mae = np.abs(sle_values - ref_values).mean()
    r2 = 1 - np.sum((sle_values - ref_values)**2) / np.sum((ref_values - ref_values.mean())**2)

    ax.text(0.05, 0.95, f'MAE = {mae:.3f}\nR² = {r2:.3f}',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(f'{variable_name} (REF)', fontsize=12)
    ax.set_ylabel(f'{variable_name} (SLE Original)', fontsize=12)
    ax.set_title('Before Correction', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # After correction
    ax = axes[1]
    ax.scatter(ref_values, corrected_values, alpha=0.5, s=30, color='green')
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.5)

    mae_corr = np.abs(corrected_values - ref_values).mean()
    r2_corr = 1 - np.sum((corrected_values - ref_values)**2) / np.sum((ref_values - ref_values.mean())**2)

    ax.text(0.05, 0.95, f'MAE = {mae_corr:.3f}\nR² = {r2_corr:.3f}',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(f'{variable_name} (REF)', fontsize=12)
    ax.set_ylabel(f'{variable_name} (SLE Corrected)', fontsize=12)
    ax.set_title('After Correction', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved correction plot to: {output_file}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_correction_model.py <merged_csv> [target_variable] [model_type]")
        print("\nExample:")
        print("  python train_correction_model.py ../data/processed/evoh/merged_comparison.csv")
        print("  python train_correction_model.py ../data/processed/evoh/merged_comparison.csv log10_x ridge")
        print("\nTarget variables: log10_x (default), log10_S, w")
        print("Model types: ridge (default), lasso, elastic, rf, gbm")
        sys.exit(1)

    merged_csv = sys.argv[1]
    target_var = sys.argv[2] if len(sys.argv) > 2 else 'log10_x'
    model_type = sys.argv[3] if len(sys.argv) > 3 else 'ridge'

    # Determine output directory
    polymer_name = 'evoh'  # Could be extracted from path
    output_dir = Path(__file__).parent.parent / 'models' / polymer_name

    model, metrics, comparison = train_and_evaluate(
        merged_csv,
        target_variable=target_var,
        model_type=model_type,
        output_dir=str(output_dir)
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel saved and ready to use for predictions on new polymers")
