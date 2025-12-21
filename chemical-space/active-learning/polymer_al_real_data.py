"""
Active Learning with YOUR REAL Solubility Data

This script shows how to adapt the active learning framework to use your
actual solubility predictions instead of synthetic data.

Expected data format:
  CSV with columns: polymer, solvent, solubility_g_L, temperature_C
  OR
  Excel with sheets for each polymer or solvent

Usage:
  python polymer_al_real_data.py --data your_solubility_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# Import the active learning framework
from polymer_solubility_active_learning import (
    build_polymer_database,
    get_common_solvents,
    compute_solubility_scores,
    featurize_polymers,
    PolymerActiveLearner,
    plot_gp_predictions,
    plot_acquisition_comparison,
    plot_learning_curve
)


def load_solubility_data(filepath):
    """
    Load solubility data from file or directory.

    Supports:
    - CSV: polymer, solvent, solubility_g_L, temperature_C
    - Excel: multiple formats
    - Directory: Multiple CSV files (REF-DATA format)

    Returns:
        DataFrame with standardized format
    """
    filepath = Path(filepath)

    # Check if it's a directory (e.g., REF-DATA)
    if filepath.is_dir():
        print(f"[INFO] Loading multiple CSV files from directory: {filepath}")
        dfs = []
        csv_files = list(filepath.glob("*.csv"))

        if not csv_files:
            raise ValueError(f"No CSV files found in {filepath}")

        for csv_file in csv_files:
            print(f"  Loading: {csv_file.name}")
            df_temp = pd.read_csv(csv_file, encoding='utf-8-sig')  # Handle BOM

            # Extract polymer name from filename if not in data
            # Pattern: COMMON-solvents-{POLYMER}.csv
            if 'Polymer' not in df_temp.columns and 'polymer' not in df_temp.columns:
                polymer_name = csv_file.stem.replace('COMMON-solvents-', '')
                df_temp['Polymer'] = polymer_name
                print(f"    Extracted polymer name from filename: {polymer_name}")

            dfs.append(df_temp)

        df = pd.concat(dfs, ignore_index=True)
        print(f"[INFO] Loaded {len(csv_files)} files, {len(df)} total rows")

    elif filepath.suffix == '.csv':
        df = pd.read_csv(filepath)

    elif filepath.suffix in ['.xlsx', '.xls']:
        # Try to infer format
        xl = pd.ExcelFile(filepath)

        if 'solubility' in xl.sheet_names or 'data' in xl.sheet_names:
            # Single sheet with all data
            sheet = 'solubility' if 'solubility' in xl.sheet_names else 'data'
            df = pd.read_excel(filepath, sheet_name=sheet)
        else:
            # Multiple sheets - try to combine
            dfs = []
            for sheet in xl.sheet_names:
                df_sheet = pd.read_excel(filepath, sheet_name=sheet)
                df_sheet['source_sheet'] = sheet
                dfs.append(df_sheet)
            df = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    # Standardize column names
    col_mapping = {
        'Polymer': 'polymer',
        'polymer_name': 'polymer',
        'Solvent': 'solvent',
        'solvent_name': 'solvent',
        'Solubility': 'solubility_g_L',
        'Solubility (%)': 'solubility_g_L',
        'solubility': 'solubility_g_L',
        'Temperature': 'temperature_C',
        'Temperature (°C)': 'temperature_C',
        'temp': 'temperature_C',
        'Temp': 'temperature_C'
    }

    df = df.rename(columns=col_mapping)

    # Validate required columns
    required = ['polymer', 'solvent', 'solubility_g_L']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")

    # Add temperature if missing
    if 'temperature_C' not in df.columns:
        print("[WARN] No temperature column found. Assuming 25°C")
        df['temperature_C'] = 25.0

    # Note: solubility_g_L might actually be in % (mass fraction) depending on source
    # For active learning purposes, the relative ranking is what matters

    return df


def custom_solubility_score(solubility_df, method='versatility', **kwargs):
    """
    Define custom solubility scoring functions.
    
    Methods:
    - 'versatility': Breadth × magnitude (default)
    - 'max': Best single solvent
    - 'mean': Average across all solvents
    - 'count': Number of "good" solvents (>threshold)
    - 'weighted': Weighted by solvent importance
    - 'custom': User-defined function
    
    Examples:
    --------
    # Optimize for water solubility specifically
    scores = custom_solubility_score(df, method='specific', solvent='Water')
    
    # Optimize for organic solvent compatibility
    scores = custom_solubility_score(
        df, method='weighted',
        solvent_weights={'Acetone': 2.0, 'THF': 2.0, 'Toluene': 1.5}
    )
    
    # Count solvents with >50 g/L
    scores = custom_solubility_score(df, method='count', threshold=50)
    """
    scores = []
    
    for polymer in solubility_df['polymer'].unique():
        poly_data = solubility_df[solubility_df['polymer'] == polymer]
        
        if method == 'versatility':
            # Default: log-scale versatility
            sols = poly_data['solubility_g_L'].values
            log_sols = np.log10(sols + 1)
            soluble_frac = np.mean(sols > kwargs.get('threshold', 10))
            score = np.mean(log_sols) * (1 + soluble_frac)
            
        elif method == 'max':
            # Best single solvent
            score = poly_data['solubility_g_L'].max()
            
        elif method == 'mean':
            # Average solubility
            score = poly_data['solubility_g_L'].mean()
            
        elif method == 'count':
            # Number of solvents above threshold
            threshold = kwargs.get('threshold', 10)
            score = (poly_data['solubility_g_L'] > threshold).sum()
            
        elif method == 'specific':
            # Specific solvent
            solvent = kwargs.get('solvent', 'Water')
            solvent_data = poly_data[poly_data['solvent'] == solvent]
            if len(solvent_data) > 0:
                score = solvent_data['solubility_g_L'].mean()
            else:
                score = 0.0
                
        elif method == 'weighted':
            # Weighted by solvent importance
            weights = kwargs.get('solvent_weights', {})
            weighted_sum = 0
            total_weight = 0
            for _, row in poly_data.iterrows():
                weight = weights.get(row['solvent'], 1.0)
                weighted_sum += row['solubility_g_L'] * weight
                total_weight += weight
            score = weighted_sum / total_weight if total_weight > 0 else 0.0
            
        elif method == 'custom':
            # User-defined function
            score_func = kwargs.get('score_function')
            if score_func is None:
                raise ValueError("Must provide 'score_function' for custom method")
            score = score_func(poly_data)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        scores.append({'polymer': polymer, 'score': score})
    
    scores_df = pd.DataFrame(scores)
    
    # Normalize to [0, 1]
    scores_df['score_norm'] = (
        (scores_df['score'] - scores_df['score'].min()) /
        (scores_df['score'].max() - scores_df['score'].min() + 1e-10)
    )
    
    return scores_df


def run_campaign_with_real_data(
    solubility_data_path,
    scoring_method='versatility',
    n_initial=3,
    n_iterations=10,
    acquisition='ei',
    temperature_filter=None,
    output_dir='active_learning_real_data',
    **scoring_kwargs
):
    """
    Run active learning campaign with real data.

    Args:
        solubility_data_path: Path to data file or directory
        scoring_method: Method for computing polymer scores
        n_initial: Number of initial random samples
        n_iterations: Number of AL iterations
        acquisition: Acquisition function ('ei', 'ucb', 'pi', 'greedy')
        temperature_filter: Temperature to filter to (e.g., 25). If None, uses all temps
        **scoring_kwargs: Additional arguments for scoring method
    """
    print("="*70)
    print("ACTIVE LEARNING WITH REAL SOLUBILITY DATA")
    print("="*70)

    # 1) Load data
    print(f"\n[1/5] Loading solubility data from {solubility_data_path}...")
    solubility_df = load_solubility_data(solubility_data_path)
    print(f"      Data shape: {solubility_df.shape}")
    print(f"      Polymers: {solubility_df['polymer'].nunique()}")
    print(f"      Solvents: {solubility_df['solvent'].nunique()}")
    print(f"      Temperature range: {solubility_df['temperature_C'].min():.1f} - {solubility_df['temperature_C'].max():.1f}°C")

    # Filter to specific temperature if requested
    if temperature_filter is not None:
        print(f"\n      Filtering to temperature = {temperature_filter}°C...")
        # Find closest temperature to requested
        temps = solubility_df['temperature_C'].unique()
        closest_temp = temps[np.argmin(np.abs(temps - temperature_filter))]
        solubility_df = solubility_df[solubility_df['temperature_C'] == closest_temp].copy()
        print(f"      Using temperature: {closest_temp}°C")
        print(f"      Filtered data shape: {solubility_df.shape}")
    
    # 2) Compute scores
    print(f"\n[2/5] Computing solubility scores (method='{scoring_method}')...")
    scores_df = custom_solubility_score(solubility_df, method=scoring_method, **scoring_kwargs)
    print(f"      Score range: [{scores_df['score'].min():.2f}, {scores_df['score'].max():.2f}]")
    print(f"      Top 3 polymers:")
    top3 = scores_df.nlargest(3, 'score')
    for i, row in top3.iterrows():
        print(f"        {row['polymer']}: {row['score']:.2f}")
    
    # 3) Build polymer database (match with scored polymers)
    print(f"\n[3/5] Building polymer feature database...")
    polymers_df = build_polymer_database()
    
    # Filter to only polymers with scores
    scored_polymers = set(scores_df['polymer'].unique())
    polymers_df = polymers_df[polymers_df['polymer'].isin(scored_polymers)].reset_index(drop=True)
    print(f"      Polymers with features: {len(polymers_df)}")
    
    # 4) Featurize
    print(f"\n[4/5] Computing molecular features...")
    X_pool, polymer_names, scaler, pca = featurize_polymers(polymers_df)
    print(f"      Feature matrix: {X_pool.shape}")
    
    # 5) Run active learning
    print(f"\n[5/5] Starting active learning campaign...")
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    learner = PolymerActiveLearner(X_pool, polymer_names, metric='score_norm')
    
    # Initial random sampling
    print(f"\n--- Initial Random Sampling ({n_initial} polymers) ---")
    initial_indices = np.random.choice(len(polymer_names), size=n_initial, replace=False)
    
    for idx in initial_indices:
        polymer_name = polymer_names[idx]
        score = scores_df[scores_df['polymer'] == polymer_name]['score_norm'].values[0]
        learner.add_observation(idx, score)
        print(f"  ✓ {polymer_name}: score = {score:.3f}")
    
    learner.fit()
    
    # Active learning loop
    history = []
    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration}/{n_iterations} ---")
        
        suggestions = learner.suggest_next(method=acquisition, n_suggestions=5)
        
        if not suggestions:
            print("  All polymers tested!")
            break
        
        print(f"  Top 5 suggestions ({acquisition.upper()}):")
        for i, (idx, name, acq_val) in enumerate(suggestions, 1):
            pred_score, pred_std = learner.predict(X_pool[idx:idx+1], return_std=True)
            print(f"    {i}. {name} (acq={acq_val:.4f}, μ={pred_score[0]:.3f}, σ={pred_std[0]:.3f})")
        
        # Test top suggestion
        test_idx, test_name, _ = suggestions[0]
        true_score = scores_df[scores_df['polymer'] == test_name]['score_norm'].values[0]
        
        learner.add_observation(test_idx, true_score)
        learner.fit()
        
        best_idx = learner.tested_indices[np.argmax(learner.y_train)]
        history.append({
            "iteration": iteration,
            "tested_polymer": test_name,
            "tested_score": true_score,
            "best_polymer": polymer_names[best_idx],
            "best_score": np.max(learner.y_train)
        })
        
        print(f"  ✓ Tested: {test_name} → score = {true_score:.3f}")
        print(f"  Current best: {polymer_names[best_idx]} ({np.max(learner.y_train):.3f})")
        
        # Visualizations
        if iteration % 2 == 0:
            plot_gp_predictions(learner, X_pool, polymer_names, scores_df, 'score_norm', iteration, outdir)
    
    # Summary
    print("\n" + "="*70)
    print("CAMPAIGN COMPLETE")
    print("="*70)
    
    best_found_idx = learner.tested_indices[np.argmax(learner.y_train)]
    true_best_idx = scores_df['score_norm'].idxmax()
    
    print(f"Best found: {polymer_names[best_found_idx]} ({np.max(learner.y_train):.3f})")
    print(f"True best:  {scores_df.loc[true_best_idx, 'polymer']} ({scores_df.loc[true_best_idx, 'score_norm']:.3f})")
    print(f"Efficiency: Found rank {(scores_df['score_norm'] > np.max(learner.y_train)).sum() + 1} / {len(scores_df)}")
    
    # Save
    history_df = pd.DataFrame(history)
    history_df.to_csv(outdir / "campaign_history.csv", index=False)
    plot_learning_curve(history_df, scores_df['score_norm'].max(), outdir)
    
    return learner, scores_df, history_df


# ==========================================
# EXAMPLE USAGE PATTERNS
# ==========================================

def example_water_solubility():
    """Example: Optimize for water solubility specifically."""
    print("\n" + "="*70)
    print("EXAMPLE: Optimize for Water Solubility")
    print("="*70 + "\n")
    
    # This assumes you have solubility_data.csv
    # Replace with your actual file path
    learner, scores, history = run_campaign_with_real_data(
        solubility_data_path="solubility_data.csv",
        scoring_method='specific',
        solvent='Water',
        n_initial=3,
        n_iterations=8,
        acquisition='ucb'  # UCB often good for maximizing specific properties
    )
    
    return learner, scores, history


def example_organic_solvents():
    """Example: Optimize for organic solvent compatibility."""
    print("\n" + "="*70)
    print("EXAMPLE: Optimize for Organic Solvent Compatibility")
    print("="*70 + "\n")
    
    # Weight important organic solvents
    organic_weights = {
        'Acetone': 2.0,
        'THF': 2.0,
        'Chloroform': 1.5,
        'Toluene': 1.5,
        'DCM': 1.5,
        'Ethyl_acetate': 1.0
    }
    
    learner, scores, history = run_campaign_with_real_data(
        solubility_data_path="solubility_data.csv",
        scoring_method='weighted',
        solvent_weights=organic_weights,
        n_initial=3,
        n_iterations=8,
        acquisition='ei'
    )
    
    return learner, scores, history


def example_custom_score():
    """Example: Custom scoring function."""
    print("\n" + "="*70)
    print("EXAMPLE: Custom Scoring Function")
    print("="*70 + "\n")
    
    # Define custom score: high water solubility + moderate organic solubility
    def custom_score_func(poly_data):
        water_sol = poly_data[poly_data['solvent'] == 'Water']['solubility_g_L'].mean()
        organic_solvents = ['Acetone', 'THF', 'Chloroform']
        organic_sol = poly_data[poly_data['solvent'].isin(organic_solvents)]['solubility_g_L'].mean()
        
        # Combined score: prioritize water, but also want some organic solubility
        score = water_sol * 0.7 + organic_sol * 0.3
        return score
    
    learner, scores, history = run_campaign_with_real_data(
        solubility_data_path="solubility_data.csv",
        scoring_method='custom',
        score_function=custom_score_func,
        n_initial=3,
        n_iterations=8,
        acquisition='ei'
    )
    
    return learner, scores, history


# ==========================================
# COMMAND LINE INTERFACE
# ==========================================

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Active learning for polymer solubility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default versatility score
  python polymer_al_real_data.py --data solubility.csv
  
  # Optimize for water solubility
  python polymer_al_real_data.py --data solubility.csv --method specific --solvent Water
  
  # Use UCB acquisition with 15 iterations
  python polymer_al_real_data.py --data solubility.csv --acquisition ucb --iterations 15
        """
    )
    
    parser.add_argument('--data', type=str, default='REF-DATA',
                       help='Path to solubility data (CSV, Excel, or directory). Default: REF-DATA')
    parser.add_argument('--method', type=str, default='versatility',
                       choices=['versatility', 'max', 'mean', 'count', 'specific', 'weighted'],
                       help='Scoring method')
    parser.add_argument('--solvent', type=str, help='Solvent name for "specific" method')
    parser.add_argument('--threshold', type=float, default=10.0, help='Threshold for "count" method')
    parser.add_argument('--temperature', type=float, default=25.0,
                       help='Temperature to filter to (°C). Default: 25')
    parser.add_argument('--initial', type=int, default=3, help='Number of initial random samples')
    parser.add_argument('--iterations', type=int, default=10, help='Number of AL iterations')
    parser.add_argument('--acquisition', type=str, default='ei', choices=['ei', 'ucb', 'pi', 'greedy'],
                       help='Acquisition function')
    parser.add_argument('--output', type=str, default='active_learning_real_data',
                       help='Output directory for results. Default: active_learning_real_data')
    
    args = parser.parse_args()
    
    # Prepare kwargs for scoring method
    scoring_kwargs = {}
    if args.method == 'specific':
        if args.solvent is None:
            parser.error("--method specific requires --solvent")
        scoring_kwargs['solvent'] = args.solvent
    elif args.method == 'count':
        scoring_kwargs['threshold'] = args.threshold
    
    # Run campaign
    learner, scores, history = run_campaign_with_real_data(
        solubility_data_path=args.data,
        scoring_method=args.method,
        n_initial=args.initial,
        n_iterations=args.iterations,
        acquisition=args.acquisition,
        temperature_filter=args.temperature,
        output_dir=args.output,
        **scoring_kwargs
    )

    print(f"\nDone! Check '{args.output}/' for results.")


if __name__ == "__main__":
    # If no command line args, show examples
    import sys
    if len(sys.argv) == 1:
        print("No arguments provided. Running with defaults (REF-DATA directory)...\n")
        print("EXAMPLE USAGE:")
        print("-" * 70)
        print("\n1. Basic usage with REF-DATA directory (default):")
        print("   python polymer_al_real_data.py")
        print("   python polymer_al_real_data.py --data REF-DATA")
        print("\n2. Use different temperature:")
        print("   python polymer_al_real_data.py --temperature 50")
        print("\n3. Optimize for specific solvent:")
        print("   python polymer_al_real_data.py --method specific --solvent propanol")
        print("\n4. Different acquisition function:")
        print("   python polymer_al_real_data.py --acquisition ucb --iterations 15")
        print("\n5. Use custom data file:")
        print("   python polymer_al_real_data.py --data your_data.csv")
        print("\n" + "-" * 70)
        print("\nRunning with default settings...\n")
        # Run with defaults
    main()
