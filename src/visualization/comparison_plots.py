"""
Visualization functions for comparing REF and SLE methods
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple


# Set style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)


def plot_parity(
    merged_df: pd.DataFrame,
    variable: str = 'log10_x',
    ax: Optional[plt.Axes] = None,
    show_stats: bool = True
) -> plt.Axes:
    """
    Create parity plot comparing REF vs SLE predictions

    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged dataframe with both REF and SLE columns
    variable : str
        Variable to plot ('log10_x', 'log10_S', 'w', 'delta_mu')
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    show_stats : bool
        Whether to show statistics on plot

    Returns:
    --------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ref_col = f"{variable}_REF"
    sle_col = f"{variable}_SLE"

    x = merged_df[ref_col]
    y = merged_df[sle_col]

    # Scatter plot with temperature-based coloring
    scatter = ax.scatter(
        x, y,
        c=merged_df['Temperature_K'],
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5,
        s=50
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Temperature (K)', rotation=270, labelpad=20)

    # Plot ideal 1:1 line
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='1:1 line', alpha=0.5)

    # Calculate and plot regression line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    x_line = np.array([min_val, max_val])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', lw=2, label=f'Fit: y={slope:.3f}x+{intercept:.3f}', alpha=0.7)

    # Calculate statistics
    error = y - x
    mae = np.abs(error).mean()
    rmse = np.sqrt((error**2).mean())
    bias = error.mean()

    # Add statistics text
    if show_stats:
        stats_text = f'R² = {r_value**2:.3f}\n'
        stats_text += f'MAE = {mae:.3f}\n'
        stats_text += f'RMSE = {rmse:.3f}\n'
        stats_text += f'Bias = {bias:.3f}\n'
        stats_text += f'N = {len(x)}'

        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Labels
    var_labels = {
        'log10_x': 'log₁₀(x)',
        'log10_S': 'log₁₀(S) [mol/L]',
        'w': 'Mass Fraction w',
        'delta_mu': 'Δμ [kcal/mol]'
    }
    label = var_labels.get(variable, variable)

    ax.set_xlabel(f'{label} (REF)', fontsize=12)
    ax.set_ylabel(f'{label} (SLE)', fontsize=12)
    ax.set_title(f'Parity Plot: {label}', fontsize=14, fontweight='bold')

    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    return ax


def plot_error_distribution(
    merged_df: pd.DataFrame,
    variable: str = 'log10_x',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot distribution of errors (SLE - REF)"""

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    error_col = f"error_{variable}"
    errors = merged_df[error_col]

    # Histogram
    ax.hist(errors, bins=30, alpha=0.7, edgecolor='black', density=True)

    # Add normal distribution fit
    from scipy.stats import norm
    mu, std = errors.mean(), errors.std()
    x = np.linspace(errors.min(), errors.max(), 100)
    ax.plot(x, norm.pdf(x, mu, std), 'r-', lw=2, label=f'Normal fit\nμ={mu:.3f}, σ={std:.3f}')

    # Add vertical line at zero
    ax.axvline(0, color='black', linestyle='--', lw=2, alpha=0.5, label='Zero error')

    # Labels
    var_labels = {
        'log10_x': 'log₁₀(x)',
        'log10_S': 'log₁₀(S)',
        'w': 'Mass Fraction w',
        'delta_mu': 'Δμ'
    }
    label = var_labels.get(variable, variable)

    ax.set_xlabel(f'Error in {label} (SLE - REF)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Error Distribution: {label}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_error_vs_temperature(
    merged_df: pd.DataFrame,
    variable: str = 'log10_x',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot error as a function of temperature"""

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    error_col = f"error_{variable}"

    # Group by temperature and calculate statistics
    temp_stats = merged_df.groupby('Temperature_K')[error_col].agg(['mean', 'std', 'count'])

    # Plot mean error with error bars
    ax.errorbar(
        temp_stats.index,
        temp_stats['mean'],
        yerr=temp_stats['std'],
        fmt='o-',
        capsize=5,
        capthick=2,
        markersize=8,
        linewidth=2,
        label='Mean ± Std'
    )

    # Add zero line
    ax.axhline(0, color='black', linestyle='--', lw=2, alpha=0.5, label='Zero error')

    # Labels
    var_labels = {
        'log10_x': 'log₁₀(x)',
        'log10_S': 'log₁₀(S)',
        'w': 'Mass Fraction w',
        'delta_mu': 'Δμ'
    }
    label = var_labels.get(variable, variable)

    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel(f'Error in {label} (SLE - REF)', fontsize=12)
    ax.set_title(f'Error vs Temperature: {label}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_error_vs_solvent(
    merged_df: pd.DataFrame,
    variable: str = 'log10_x',
    top_n: int = 15,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot error by solvent (showing top N by error magnitude)"""

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    error_col = f"error_{variable}"

    # Calculate mean absolute error per solvent
    solvent_stats = merged_df.groupby('Solvent')[error_col].agg(['mean', 'std', 'count'])
    solvent_stats['abs_mean'] = solvent_stats['mean'].abs()
    solvent_stats = solvent_stats.sort_values('abs_mean', ascending=False).head(top_n)

    # Create bar plot
    x_pos = np.arange(len(solvent_stats))
    ax.bar(x_pos, solvent_stats['mean'], yerr=solvent_stats['std'],
           capsize=5, alpha=0.7, edgecolor='black')

    # Add zero line
    ax.axhline(0, color='black', linestyle='--', lw=2, alpha=0.5)

    # Labels
    var_labels = {
        'log10_x': 'log₁₀(x)',
        'log10_S': 'log₁₀(S)',
        'w': 'Mass Fraction w',
        'delta_mu': 'Δμ'
    }
    label = var_labels.get(variable, variable)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(solvent_stats.index, rotation=45, ha='right')
    ax.set_ylabel(f'Mean Error in {label} (SLE - REF)', fontsize=12)
    ax.set_title(f'Error by Solvent (Top {top_n}): {label}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    return ax


def create_comprehensive_comparison(
    merged_df: pd.DataFrame,
    variable: str = 'log10_x',
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive 4-panel comparison plot

    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged dataframe with REF and SLE data
    variable : str
        Variable to analyze
    output_path : str, optional
        Path to save the figure

    Returns:
    --------
    plt.Figure
    """

    fig = plt.figure(figsize=(16, 12))

    # Create subplots
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    # Generate plots
    plot_parity(merged_df, variable, ax=ax1)
    plot_error_distribution(merged_df, variable, ax=ax2)
    plot_error_vs_temperature(merged_df, variable, ax=ax3)
    plot_error_vs_solvent(merged_df, variable, ax=ax4)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    return fig


def plot_all_variables(
    merged_df: pd.DataFrame,
    output_dir: Optional[str] = None
) -> None:
    """Create comprehensive comparison plots for all key variables"""

    variables = ['log10_x', 'log10_S', 'w', 'delta_mu']

    for var in variables:
        print(f"\nCreating plots for {var}...")

        fig = create_comprehensive_comparison(merged_df, variable=var)

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            filename = output_path / f'comparison_{var}.png'
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Saved: {filename}")

        plt.close(fig)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python comparison_plots.py <merged_csv_file> [output_dir]")
        sys.exit(1)

    merged_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'results/figures'

    # Load data
    print(f"Loading merged data from {merged_file}...")
    merged_df = pd.read_csv(merged_file)

    # Create all plots
    plot_all_variables(merged_df, output_dir=output_dir)

    print("\nDone! All plots created.")
