"""
Process COSMO-therm tab files and generate cleaned CSV files
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from .tab_parser import parse_tab_file, extract_common_columns


def load_solvent_list(filepath: str) -> List[str]:
    """
    Load a list of solvents from a text file (one solvent per line)

    Parameters:
    -----------
    filepath : str
        Path to text file with solvent names

    Returns:
    --------
    List of solvent names
    """
    with open(filepath, 'r') as f:
        solvents = [line.strip() for line in f if line.strip()]
    return solvents


def process_and_merge_data(
    ref_filepath: str,
    sle_filepath: str,
    output_dir: Optional[str] = None,
    common_solvents_file: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process REF and SLE tab files and create matched datasets

    Parameters:
    -----------
    ref_filepath : str
        Path to reference method tab file
    sle_filepath : str
        Path to SLE method tab file
    output_dir : str, optional
        Directory to save output CSV files
    common_solvents_file : str, optional
        Path to text file with list of common solvents (one per line)
        If provided, REF data will be filtered to only these solvents

    Returns:
    --------
    Tuple of (ref_df, sle_df, merged_df)
    """

    # Parse tab files
    print("Parsing REF tab file...")
    ref_raw = parse_tab_file(ref_filepath)
    print(f"  Found {len(ref_raw)} data points")

    print("Parsing SLE tab file...")
    sle_raw = parse_tab_file(sle_filepath)
    print(f"  Found {len(sle_raw)} data points")

    # Extract common columns
    print("\nExtracting and standardizing columns...")
    ref_df = extract_common_columns(ref_raw, 'REF')
    print(f"  REF after cleanup: {len(ref_df)} valid data points")
    sle_df = extract_common_columns(sle_raw, 'SLE')
    print(f"  SLE after cleanup: {len(sle_df)} valid data points")

    # Filter REF by common solvents list if provided
    if common_solvents_file:
        print(f"\nFiltering REF by common solvents list: {common_solvents_file}")
        common_solvents = load_solvent_list(common_solvents_file)
        print(f"  Loaded {len(common_solvents)} common solvents")
        ref_df = ref_df[ref_df['Solvent'].isin(common_solvents)].copy()
        print(f"  REF after solvent filter: {len(ref_df)} data points")

    # Filter REF data to match SLE solvents and temperature range
    print("\nFiltering to match solvents present in BOTH datasets...")
    sle_solvents = set(sle_df['Solvent'].unique())
    sle_temps = set(sle_df['Temperature_K'].unique())

    print(f"  SLE has {len(sle_solvents)} unique solvents")
    print(f"  SLE has {len(sle_temps)} unique temperatures")
    print(f"  SLE temperature range: {min(sle_temps):.2f} - {max(sle_temps):.2f} K")

    ref_filtered = ref_df[
        (ref_df['Solvent'].isin(sle_solvents)) &
        (ref_df['Temperature_K'].isin(sle_temps))
    ].copy()

    print(f"  Filtered REF from {len(ref_df)} to {len(ref_filtered)} data points")

    # Merge datasets for comparison
    print("\nMerging datasets...")
    merged_df = pd.merge(
        ref_filtered,
        sle_df,
        on=['Solvent', 'Temperature_K'],
        suffixes=('_REF', '_SLE'),
        how='inner'
    )

    print(f"  Merged dataset has {len(merged_df)} matched data points")
    print(f"  Covering {len(merged_df['Solvent'].unique())} solvents")
    print(f"  Covering {len(merged_df['Temperature_K'].unique())} temperatures")

    # Calculate differences
    merged_df['error_log10_x'] = merged_df['log10_x_SLE'] - merged_df['log10_x_REF']
    merged_df['error_log10_S'] = merged_df['log10_S_SLE'] - merged_df['log10_S_REF']
    merged_df['error_w'] = merged_df['w_SLE'] - merged_df['w_REF']
    merged_df['error_delta_mu'] = merged_df['delta_mu_SLE'] - merged_df['delta_mu_REF']

    # Save to CSV if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        ref_output = output_path / 'ref_filtered.csv'
        sle_output = output_path / 'sle.csv'
        merged_output = output_path / 'merged_comparison.csv'

        print(f"\nSaving CSV files to {output_dir}...")
        ref_filtered.to_csv(ref_output, index=False)
        print(f"  Saved: {ref_output}")

        sle_df.to_csv(sle_output, index=False)
        print(f"  Saved: {sle_output}")

        merged_df.to_csv(merged_output, index=False)
        print(f"  Saved: {merged_output}")

    return ref_filtered, sle_df, merged_df


def print_data_summary(df: pd.DataFrame, name: str):
    """Print summary statistics for a dataset"""
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")

    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    if 'Solvent' in df.columns:
        print(f"\nSolvents ({len(df['Solvent'].unique())}): {sorted(df['Solvent'].unique())[:10]}...")

    if 'Temperature_K' in df.columns:
        temps = df['Temperature_K'].unique()
        print(f"\nTemperatures ({len(temps)}): {sorted(temps)}")

    # Print statistics for key columns
    key_cols = ['log10_x', 'log10_S', 'w', 'delta_mu']

    for col in key_cols:
        # Check with both REF and SLE suffixes
        for suffix in ['', '_REF', '_SLE']:
            full_col = col + suffix
            if full_col in df.columns:
                print(f"\n{full_col}:")
                print(f"  Mean: {df[full_col].mean():.4f}")
                print(f"  Std:  {df[full_col].std():.4f}")
                print(f"  Min:  {df[full_col].min():.4f}")
                print(f"  Max:  {df[full_col].max():.4f}")

    # Print error statistics if available
    error_cols = [c for c in df.columns if c.startswith('error_')]
    if error_cols:
        print(f"\nError Statistics:")
        for col in error_cols:
            print(f"\n{col}:")
            print(f"  Mean:  {df[col].mean():.4f}")
            print(f"  Std:   {df[col].std():.4f}")
            print(f"  MAE:   {df[col].abs().mean():.4f}")
            print(f"  RMSE:  {np.sqrt((df[col]**2).mean()):.4f}")


if __name__ == "__main__":
    # Example usage - modify paths as needed
    import sys

    if len(sys.argv) != 3:
        print("Usage: python process_tab_files.py <ref_tab_file> <sle_tab_file>")
        sys.exit(1)

    ref_file = sys.argv[1]
    sle_file = sys.argv[2]

    # Process files
    ref_df, sle_df, merged_df = process_and_merge_data(
        ref_file,
        sle_file,
        output_dir='data/processed'
    )

    # Print summaries
    print_data_summary(ref_df, "REF (Filtered)")
    print_data_summary(sle_df, "SLE")
    print_data_summary(merged_df, "Merged Comparison")
