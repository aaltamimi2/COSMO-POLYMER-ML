"""
Cross-polymer analysis and universal correction model training

This script demonstrates how to:
1. Train a universal correction model on multiple polymers
2. Compare performance across different polymers
3. Evaluate generalization to new polymers (leave-one-out)
4. Compare polymer-aware vs polymer-agnostic models

Usage:
    python compare_polymers.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.correction_model import (
    train_universal_model,
    compare_polymer_performance,
    cross_polymer_validation,
    compare_models
)


def load_polymer_data(data_dir: str = '../data/processed') -> dict:
    """
    Load data for all available polymers

    Parameters:
    -----------
    data_dir : str
        Base directory containing processed polymer data

    Returns:
    --------
    dict mapping polymer names to DataFrames
    """
    data_dir = Path(data_dir)
    polymer_data = {}

    # Auto-detect polymers
    for polymer_dir in data_dir.iterdir():
        if polymer_dir.is_dir():
            merged_file = polymer_dir / 'merged_comparison.csv'
            if merged_file.exists():
                polymer_name = polymer_dir.name
                print(f"Loading {polymer_name} data from {merged_file}")
                polymer_data[polymer_name] = pd.read_csv(merged_file)
                print(f"  {polymer_name}: {len(polymer_data[polymer_name])} data points")

    return polymer_data


def plot_strategy_comparison(
    strategy_results: dict,
    output_file: str
):
    """
    Create bar plot comparing polymer-aware vs polymer-agnostic strategies
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    strategies = list(strategy_results.keys())
    strategy_names = [strategy_results[s]['name'] for s in strategies]

    # Extract metrics
    metrics = ['MAE', 'RMSE', 'R2']
    colors = ['#3498db', '#e74c3c']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = []
        for s in strategies:
            cv_results = strategy_results[s]['cv_results']
            val = cv_results.loc[cv_results['Metric'] == metric, 'Mean'].values[0]
            values.append(val)

        bars = ax.bar(strategy_names, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved strategy comparison plot: {output_file}")
    plt.close()


def plot_polymer_performance(
    strategy_results: dict,
    output_file: str
):
    """
    Create grouped bar plot comparing performance across polymers
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Collect data
    all_data = []
    for strategy_name, data in strategy_results.items():
        perf_df = data['polymer_performance'].copy()
        perf_df['Strategy'] = data['name']
        all_data.append(perf_df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Metrics to plot
    metrics = [
        ('MAE_original', 'Original MAE'),
        ('MAE_corrected', 'Corrected MAE'),
        ('Improvement', 'Improvement (MAE)'),
        ('Improvement_pct', 'Improvement (%)')
    ]

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]

        # Pivot for grouped bar plot
        pivot_df = combined_df.pivot(index='Polymer', columns='Strategy', values=metric)

        pivot_df.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'],
                     alpha=0.7, edgecolor='black', width=0.7)

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_xlabel('Polymer', fontsize=11)
        ax.legend(title='Strategy', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved polymer performance plot: {output_file}")
    plt.close()


def plot_lopo_results(
    lopo_results: pd.DataFrame,
    output_file: str
):
    """
    Create visualization for leave-one-polymer-out results
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Original vs Corrected MAE
    ax = axes[0]
    x = np.arange(len(lopo_results))
    width = 0.35

    bars1 = ax.bar(x - width/2, lopo_results['MAE_original'], width,
                   label='Original SLE', color='#e74c3c', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, lopo_results['MAE_corrected'], width,
                   label='Corrected SLE', color='#2ecc71', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Test Polymer', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax.set_title('Leave-One-Polymer-Out: MAE Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(lopo_results['Test_Polymer'])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)

    # Plot 2: Improvement percentage
    ax = axes[1]
    bars = ax.bar(lopo_results['Test_Polymer'], lopo_results['Improvement_pct'],
                  color='#3498db', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Test Polymer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Leave-One-Polymer-Out: Improvement', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved LOPO results plot: {output_file}")
    plt.close()


def plot_parity_plots(
    polymer_data_dict: dict,
    model: object,
    target_variable: str,
    output_file: str
):
    """
    Create parity plots showing original vs corrected predictions for each polymer
    """
    from models.feature_engineering import prepare_training_data
    from models.correction_model import apply_correction

    n_polymers = len(polymer_data_dict)
    fig, axes = plt.subplots(2, n_polymers, figsize=(5*n_polymers, 10))

    if n_polymers == 1:
        axes = axes.reshape(2, 1)

    for idx, (polymer_name, df) in enumerate(polymer_data_dict.items()):
        # Prepare data
        df_copy = df.copy()
        df_copy['polymer'] = polymer_name

        X, y, _ = prepare_training_data(df_copy, target_variable=target_variable)

        # Add polymer dummies if needed
        if any('polymer_' in feat for feat in model.feature_names):
            polymer_dummies = pd.get_dummies(df_copy['polymer'], prefix='polymer')
            X = pd.concat([X, polymer_dummies], axis=1)

        # Ensure all features present
        for feat in model.feature_names:
            if feat not in X.columns:
                X[feat] = 0.0

        # Get predictions
        y_pred = model.predict(X[model.feature_names])

        sle_values = df[f'{target_variable}_SLE'].values
        ref_values = df[f'{target_variable}_REF'].values
        corrected_values = apply_correction(pd.Series(sle_values), y_pred).values

        # Plot 1: Original SLE vs REF
        ax = axes[0, idx]
        ax.scatter(ref_values, sle_values, alpha=0.5, s=30, color='#e74c3c', edgecolor='black', linewidth=0.5)

        min_val = min(ref_values.min(), sle_values.min())
        max_val = max(ref_values.max(), sle_values.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.5)

        mae_orig = np.abs(sle_values - ref_values).mean()
        r2_orig = 1 - np.sum((sle_values - ref_values)**2) / np.sum((ref_values - ref_values.mean())**2)

        ax.text(0.05, 0.95, f'MAE = {mae_orig:.3f}\nR² = {r2_orig:.3f}',
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

        ax.set_xlabel(f'{target_variable} (REF)', fontsize=11)
        ax.set_ylabel(f'{target_variable} (SLE)', fontsize=11)
        ax.set_title(f'{polymer_name.upper()} - Original', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Plot 2: Corrected SLE vs REF
        ax = axes[1, idx]
        ax.scatter(ref_values, corrected_values, alpha=0.5, s=30, color='#2ecc71', edgecolor='black', linewidth=0.5)
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.5)

        mae_corr = np.abs(corrected_values - ref_values).mean()
        r2_corr = 1 - np.sum((corrected_values - ref_values)**2) / np.sum((ref_values - ref_values.mean())**2)
        improvement_pct = 100 * (1 - mae_corr / mae_orig)

        ax.text(0.05, 0.95, f'MAE = {mae_corr:.3f}\nR² = {r2_corr:.3f}\nImpr. = {improvement_pct:.1f}%',
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

        ax.set_xlabel(f'{target_variable} (REF)', fontsize=11)
        ax.set_ylabel(f'{target_variable} (Corrected)', fontsize=11)
        ax.set_title(f'{polymer_name.upper()} - Corrected', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved parity plots: {output_file}")
    plt.close()


def compare_universal_vs_specific(
    polymer_data_dict: dict,
    target_variable: str = 'log10_x',
    output_dir: str = None
):
    """
    Compare polymer-aware universal model vs polymer-agnostic model

    This helps determine if including polymer identity as a feature improves
    the model or if a truly universal correction factor works better.
    """
    print("\n" + "="*80)
    print("COMPARING UNIVERSAL MODEL STRATEGIES")
    print("="*80)

    results = {}

    # Strategy 1: Polymer-agnostic (universal correction factor)
    print("\n1. Training POLYMER-AGNOSTIC model (universal correction factor)...")
    model_agnostic, cv_agnostic = train_universal_model(
        polymer_data_dict,
        target_variable=target_variable,
        include_polymer_id=False,
        model_type='ridge'
    )
    results['agnostic'] = {
        'model': model_agnostic,
        'cv_results': cv_agnostic,
        'name': 'Polymer-Agnostic'
    }

    # Strategy 2: Polymer-aware (includes polymer ID as feature)
    print("\n" + "="*80)
    print("\n2. Training POLYMER-AWARE model (polymer-specific corrections)...")
    model_aware, cv_aware = train_universal_model(
        polymer_data_dict,
        target_variable=target_variable,
        include_polymer_id=True,
        model_type='ridge'
    )
    results['aware'] = {
        'model': model_aware,
        'cv_results': cv_aware,
        'name': 'Polymer-Aware'
    }

    # Compare CV performance
    print("\n" + "="*80)
    print("CROSS-VALIDATION COMPARISON")
    print("="*80)
    comparison_df = pd.DataFrame({
        'Strategy': ['Polymer-Agnostic', 'Polymer-Aware'],
        'MAE': [cv_agnostic.loc[cv_agnostic['Metric'] == 'MAE', 'Mean'].values[0],
                cv_aware.loc[cv_aware['Metric'] == 'MAE', 'Mean'].values[0]],
        'RMSE': [cv_agnostic.loc[cv_agnostic['Metric'] == 'RMSE', 'Mean'].values[0],
                 cv_aware.loc[cv_aware['Metric'] == 'RMSE', 'Mean'].values[0]],
        'R2': [cv_agnostic.loc[cv_agnostic['Metric'] == 'R2', 'Mean'].values[0],
               cv_aware.loc[cv_aware['Metric'] == 'R2', 'Mean'].values[0]]
    })
    print(comparison_df.to_string(index=False))

    # Compare per-polymer performance
    print("\n" + "="*80)
    print("PER-POLYMER PERFORMANCE")
    print("="*80)

    for strategy_name, strategy_data in results.items():
        print(f"\n{strategy_data['name']}:")
        perf = compare_polymer_performance(
            polymer_data_dict,
            strategy_data['model'],
            target_variable=target_variable
        )
        results[strategy_name]['polymer_performance'] = perf

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save models
        model_agnostic.save(str(output_path / f'universal_agnostic_{target_variable}.joblib'))
        model_aware.save(str(output_path / f'universal_aware_{target_variable}.joblib'))

        # Save comparison
        comparison_df.to_csv(output_path / f'strategy_comparison_{target_variable}.csv', index=False)

        # Save per-polymer results
        for strategy_name, strategy_data in results.items():
            perf_df = strategy_data['polymer_performance']
            perf_df.to_csv(
                output_path / f'polymer_performance_{strategy_name}_{target_variable}.csv',
                index=False
            )

        print(f"\nResults saved to: {output_path}")

        # Create visualizations
        print("\nCreating visualizations...")
        plot_strategy_comparison(results, str(output_path / f'strategy_comparison_{target_variable}.png'))
        plot_polymer_performance(results, str(output_path / f'polymer_performance_{target_variable}.png'))

        # Parity plots for both strategies
        plot_parity_plots(polymer_data_dict, results['agnostic']['model'], target_variable,
                         str(output_path / f'parity_agnostic_{target_variable}.png'))
        plot_parity_plots(polymer_data_dict, results['aware']['model'], target_variable,
                         str(output_path / f'parity_aware_{target_variable}.png'))

    return results


def main():
    """Main execution function"""

    print("="*80)
    print("CROSS-POLYMER CORRECTION MODEL ANALYSIS")
    print("="*80)

    # Configuration
    target_variable = 'log10_x'
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    output_dir = Path(__file__).parent.parent / 'results' / 'cross_polymer_analysis'

    # Load data
    print("\n" + "="*80)
    print("LOADING POLYMER DATA")
    print("="*80)
    polymer_data = load_polymer_data(data_dir)

    if len(polymer_data) == 0:
        print("ERROR: No polymer data found!")
        return

    print(f"\nLoaded {len(polymer_data)} polymers: {list(polymer_data.keys())}")

    # Analysis 1: Compare universal model strategies
    print("\n" + "="*80)
    print("ANALYSIS 1: Universal Model Strategies")
    print("="*80)
    strategy_results = compare_universal_vs_specific(
        polymer_data,
        target_variable=target_variable,
        output_dir=str(output_dir)
    )

    # Analysis 2: Leave-one-polymer-out validation
    print("\n" + "="*80)
    print("ANALYSIS 2: Leave-One-Polymer-Out Cross-Validation")
    print("="*80)
    print("\nTesting polymer-agnostic model generalization...")
    lopo_results = cross_polymer_validation(
        polymer_data,
        target_variable=target_variable,
        model_type='ridge',
        include_polymer_id=False
    )

    # Save LOPO results
    if output_dir:
        output_path = Path(output_dir)
        lopo_results.to_csv(
            output_path / f'leave_one_polymer_out_{target_variable}.csv',
            index=False
        )

        # Create LOPO visualization
        plot_lopo_results(lopo_results, str(output_path / f'leave_one_polymer_out_{target_variable}.png'))

    # Analysis 3: Compare different model types on combined data
    print("\n" + "="*80)
    print("ANALYSIS 3: Compare Model Types on Combined Data")
    print("="*80)

    # Combine all data
    combined_data = []
    for polymer_name, df in polymer_data.items():
        df_copy = df.copy()
        df_copy['polymer'] = polymer_name
        combined_data.append(df_copy)
    combined_df = pd.concat(combined_data, ignore_index=True)

    # Prepare features
    from models.feature_engineering import prepare_training_data
    X, y, _ = prepare_training_data(combined_df, target_variable=target_variable)

    # Compare models
    model_comparison = compare_models(X, y, cv=5)
    print("\n", model_comparison.to_string(index=False))

    if output_dir:
        output_path = Path(output_dir)
        model_comparison.to_csv(
            output_path / f'model_type_comparison_{target_variable}.csv',
            index=False
        )

    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nKey files generated:")
    print("\nModels:")
    print("  - universal_agnostic_*.joblib: Polymer-agnostic universal model")
    print("  - universal_aware_*.joblib: Polymer-aware universal model")
    print("\nData (CSV):")
    print("  - strategy_comparison_*.csv: Comparison of modeling strategies")
    print("  - polymer_performance_*.csv: Per-polymer performance metrics")
    print("  - leave_one_polymer_out_*.csv: Generalization test results")
    print("  - model_type_comparison_*.csv: Comparison of different ML algorithms")
    print("\nFigures (PNG):")
    print("  - strategy_comparison_*.png: Bar plots comparing strategies")
    print("  - polymer_performance_*.png: Per-polymer performance comparison")
    print("  - parity_agnostic_*.png: Parity plots for polymer-agnostic model")
    print("  - parity_aware_*.png: Parity plots for polymer-aware model")
    print("  - leave_one_polymer_out_*.png: LOPO cross-validation results")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Best strategy based on CV
    best_strategy = min(strategy_results.items(),
                       key=lambda x: x[1]['cv_results'].loc[
                           x[1]['cv_results']['Metric'] == 'MAE', 'Mean'].values[0])

    print(f"\nBest overall strategy: {best_strategy[1]['name']}")

    # LOPO generalization
    avg_lopo_improvement = lopo_results['Improvement_pct'].mean()
    print(f"\nLeave-one-polymer-out average improvement: {avg_lopo_improvement:.1f}%")

    if avg_lopo_improvement > 20:
        print("  -> Model generalizes well to new polymers")
    elif avg_lopo_improvement > 0:
        print("  -> Model shows some generalization, but could be improved")
    else:
        print("  -> WARNING: Model may not generalize well to new polymers")

    # Best model type
    best_model = model_comparison.iloc[0]
    print(f"\nBest model type: {best_model['Model']}")
    print(f"  MAE: {best_model['MAE_mean']:.3f} ± {best_model['MAE_std']:.3f}")


if __name__ == "__main__":
    main()
