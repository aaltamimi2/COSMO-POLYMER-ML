"""
Main analysis script for EVOH data comparison
Run this script to process tab files and generate comparison plots
"""
import sys
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_processing.process_tab_files import process_and_merge_data, print_data_summary
from visualization.comparison_plots import plot_all_variables


def extract_polymer_name(ref_file: str, sle_file: str) -> str:
    """
    Extract polymer name from input filenames

    Looks for patterns like:
    - all-solvents-PS.tab -> PS
    - EVOH-REF.tab -> EVOH
    - SLE-COMMON-PS.tab -> PS

    Parameters:
    -----------
    ref_file : str
        Path to first tab file
    sle_file : str
        Path to second tab file

    Returns:
    --------
    str : Detected polymer name
    """
    # Try both files
    for filepath in [ref_file, sle_file]:
        filename = Path(filepath).stem  # Get filename without extension

        # Pattern 1: Look for uppercase sequences after hyphen or underscore
        # e.g., "all-solvents-PS" -> "PS", "EVOH-REF" -> "EVOH"
        matches = re.findall(r'[-_]([A-Z][A-Z0-9]*)', filename)

        # Filter out common suffixes like REF, SLE, COMMON
        polymer_candidates = [m for m in matches if m not in ['REF', 'SLE', 'COMMON', 'ALL', 'TAB']]

        if polymer_candidates:
            # Return the last match (usually the polymer name)
            return polymer_candidates[-1]

    # Fallback to EVOH if can't detect
    print("Warning: Could not auto-detect polymer name from filenames. Using default 'EVOH'")
    return 'EVOH'


def main(ref_tab_file: str, sle_tab_file: str, polymer_name: str = 'EVOH',
         common_solvents_file: str = None):
    """
    Complete analysis pipeline for comparing REF and SLE methods

    Parameters:
    -----------
    ref_tab_file : str
        Path to REF method tab file
    sle_tab_file : str
        Path to SLE method tab file
    polymer_name : str
        Name of polymer (for file naming)
    common_solvents_file : str, optional
        Path to text file with common solvents list (one per line)
    """

    print("="*80)
    print(f"COSMO-therm SLE Correction Analysis: {polymer_name}")
    print("="*80)

    # Set up paths
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / 'data' / 'processed' / polymer_name.lower()
    figures_dir = base_dir / 'results' / 'figures' / polymer_name.lower()

    # Process data
    print("\n" + "="*80)
    print("STEP 1: Processing Tab Files")
    print("="*80)

    ref_df, sle_df, merged_df = process_and_merge_data(
        ref_tab_file,
        sle_tab_file,
        output_dir=str(processed_dir),
        common_solvents_file=common_solvents_file
    )

    # Print summaries
    print_data_summary(ref_df, f"{polymer_name} - REF (Filtered)")
    print_data_summary(sle_df, f"{polymer_name} - SLE")
    print_data_summary(merged_df, f"{polymer_name} - Merged Comparison")

    # Create visualizations
    print("\n" + "="*80)
    print("STEP 2: Creating Comparison Plots")
    print("="*80)

    plot_all_variables(merged_df, output_dir=str(figures_dir))

    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nProcessed data saved to: {processed_dir}")
    print(f"Figures saved to: {figures_dir}")
    print("\nGenerated files:")
    print(f"  - {processed_dir}/ref_filtered.csv")
    print(f"  - {processed_dir}/sle.csv")
    print(f"  - {processed_dir}/merged_comparison.csv")
    print(f"  - {figures_dir}/comparison_log10_x.png")
    print(f"  - {figures_dir}/comparison_log10_S.png")
    print(f"  - {figures_dir}/comparison_w.png")
    print(f"  - {figures_dir}/comparison_delta_mu.png")

    return ref_df, sle_df, merged_df


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python analyze_evoh.py <ref_tab_file> <sle_tab_file> [common_solvents_file]")
        print("\nExample:")
        print("  python analyze_evoh.py ../data/raw/EVOH-REF.tab ../data/raw/EVOH-SLE.tab")
        print("  python analyze_evoh.py ../data/raw/all-solvents-PS.tab ../data/raw/SLE-COMMON-PS.tab ../common-solvents-LIST.txt")
        sys.exit(1)

    ref_file = sys.argv[1]
    sle_file = sys.argv[2]
    common_solvents = sys.argv[3] if len(sys.argv) == 4 else None

    # Auto-detect polymer name from filenames
    polymer_name = extract_polymer_name(ref_file, sle_file)
    print(f"Auto-detected polymer: {polymer_name}\n")

    main(ref_file, sle_file, polymer_name=polymer_name, common_solvents_file=common_solvents)
