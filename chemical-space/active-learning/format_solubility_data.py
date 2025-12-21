"""
Helper: Format Your Solubility Data for Active Learning

This script helps you convert your solubility predictions into the format
expected by the active learning framework.

Expected output format:
  polymer | solvent | solubility_g_L | temperature_C
  --------|---------|----------------|---------------
  PMMA    | Acetone | 150.5          | 25
  PMMA    | Water   | 0.01           | 25
  ...

Supports multiple input formats - see examples below.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def format_wide_to_long(df, polymer_col='polymer', temp_col=None):
    """
    Convert wide format (solvents as columns) to long format.
    
    Input (wide):
        polymer | Acetone | Water | THF | ...
        --------|---------|-------|-----|----
        PMMA    | 150.5   | 0.01  | 180 | ...
        PS      | 120     | 0.02  | 160 | ...
    
    Output (long):
        polymer | solvent | solubility_g_L | temperature_C
        --------|---------|----------------|---------------
        PMMA    | Acetone | 150.5          | 25
        PMMA    | Water   | 0.01           | 25
        ...
    """
    # Identify solvent columns (everything except polymer and temp)
    exclude_cols = [polymer_col]
    if temp_col and temp_col in df.columns:
        exclude_cols.append(temp_col)
    
    solvent_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Melt to long format
    df_long = df.melt(
        id_vars=exclude_cols,
        value_vars=solvent_cols,
        var_name='solvent',
        value_name='solubility_g_L'
    )
    
    # Add temperature if not provided
    if temp_col and temp_col in df.columns:
        df_long = df_long.rename(columns={temp_col: 'temperature_C'})
    else:
        df_long['temperature_C'] = 25  # Default room temperature
    
    # Rename polymer column if needed
    if polymer_col != 'polymer':
        df_long = df_long.rename(columns={polymer_col: 'polymer'})
    
    return df_long[['polymer', 'solvent', 'solubility_g_L', 'temperature_C']]


def format_multi_sheet_excel(filepath):
    """
    Convert multi-sheet Excel where each sheet is a polymer.
    
    Format:
        Sheet "PMMA":
            solvent | solubility_g_L | temperature_C
            --------|----------------|---------------
            Acetone | 150.5          | 25
            Water   | 0.01           | 25
    """
    xl = pd.ExcelFile(filepath)
    dfs = []
    
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        df['polymer'] = sheet_name  # Use sheet name as polymer name
        dfs.append(df)
    
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Ensure correct column names
    if 'temperature_C' not in df_combined.columns and 'temp' in df_combined.columns:
        df_combined = df_combined.rename(columns={'temp': 'temperature_C'})
    if 'temperature_C' not in df_combined.columns:
        df_combined['temperature_C'] = 25
    
    return df_combined[['polymer', 'solvent', 'solubility_g_L', 'temperature_C']]


def format_separate_files(directory, pattern='*_solubility.csv'):
    """
    Combine separate files where each file is a polymer.
    
    Files:
        PMMA_solubility.csv
        PS_solubility.csv
        ...
    
    Each file:
        solvent | solubility_g_L | temperature_C
        --------|----------------|---------------
        Acetone | 150.5          | 25
        Water   | 0.01           | 25
    """
    directory = Path(directory)
    dfs = []
    
    for filepath in directory.glob(pattern):
        polymer_name = filepath.stem.replace('_solubility', '')
        df = pd.read_csv(filepath)
        df['polymer'] = polymer_name
        dfs.append(df)
    
    df_combined = pd.concat(dfs, ignore_index=True)
    
    if 'temperature_C' not in df_combined.columns:
        df_combined['temperature_C'] = 25
    
    return df_combined[['polymer', 'solvent', 'solubility_g_L', 'temperature_C']]


def validate_solubility_data(df):
    """
    Validate and clean solubility data.
    """
    print("Validating solubility data...")
    
    # Check required columns
    required = ['polymer', 'solvent', 'solubility_g_L', 'temperature_C']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Remove NaN values
    n_before = len(df)
    df = df.dropna(subset=['solubility_g_L'])
    n_after = len(df)
    if n_before != n_after:
        print(f"  Warning: Removed {n_before - n_after} rows with NaN solubility")
    
    # Check for negative solubilities
    negative = (df['solubility_g_L'] < 0).sum()
    if negative > 0:
        print(f"  Warning: {negative} negative solubilities found. Setting to 0.")
        df.loc[df['solubility_g_L'] < 0, 'solubility_g_L'] = 0.0
    
    # Summary statistics
    print(f"\n  ✓ Valid data shape: {df.shape}")
    print(f"  ✓ Polymers: {df['polymer'].nunique()}")
    print(f"  ✓ Solvents: {df['solvent'].nunique()}")
    print(f"  ✓ Solubility range: [{df['solubility_g_L'].min():.2e}, {df['solubility_g_L'].max():.2e}] g/L")
    print(f"  ✓ Temperatures: {df['temperature_C'].unique()}")
    
    # Check for complete data
    expected_rows = df['polymer'].nunique() * df['solvent'].nunique()
    actual_rows = len(df)
    completeness = (actual_rows / expected_rows) * 100
    print(f"  ✓ Data completeness: {completeness:.1f}%")
    
    if completeness < 90:
        print(f"    Warning: Missing {expected_rows - actual_rows} polymer-solvent combinations")
    
    return df


# ==========================================
# EXAMPLE WORKFLOWS
# ==========================================

def example_1_wide_format():
    """Example: Convert wide format CSV."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Wide Format CSV → Long Format")
    print("="*70)
    
    # Create example wide format data
    wide_data = {
        'polymer': ['PMMA', 'PS', 'PVA', 'PAN'],
        'Acetone': [150.5, 80.2, 5.1, 2.3],
        'Water': [0.01, 0.02, 200.5, 0.05],
        'THF': [180.3, 120.4, 0.8, 15.6],
        'Chloroform': [200.1, 150.8, 0.03, 1.2]
    }
    df_wide = pd.DataFrame(wide_data)
    
    print("\nInput (wide format):")
    print(df_wide.to_string(index=False))
    
    # Convert
    df_long = format_wide_to_long(df_wide, polymer_col='polymer')
    
    print("\nOutput (long format):")
    print(df_long.head(10).to_string(index=False))
    
    # Validate
    df_validated = validate_solubility_data(df_long)
    
    # Save
    output_path = "solubility_data_formatted.csv"
    df_validated.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    
    return df_validated


def example_2_temperature_dependent():
    """Example: Handle temperature-dependent data."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Temperature-Dependent Solubility")
    print("="*70)
    
    # Create example with multiple temperatures
    data = {
        'polymer': ['PMMA', 'PMMA', 'PMMA', 'PS', 'PS', 'PS'],
        'solvent': ['Acetone', 'Acetone', 'Acetone', 'Toluene', 'Toluene', 'Toluene'],
        'solubility_g_L': [150, 180, 220, 100, 130, 170],
        'temperature_C': [25, 50, 80, 25, 50, 80]
    }
    df = pd.DataFrame(data)
    
    print("\nTemperature-dependent data:")
    print(df.to_string(index=False))
    
    # Option 1: Use highest temperature
    df_high_temp = df.loc[df.groupby(['polymer', 'solvent'])['temperature_C'].idxmax()]
    print("\nOption 1: Keep highest temperature only")
    print(df_high_temp.to_string(index=False))
    
    # Option 2: Average across temperatures
    df_avg = df.groupby(['polymer', 'solvent']).agg({
        'solubility_g_L': 'mean',
        'temperature_C': 'mean'
    }).reset_index()
    print("\nOption 2: Average across temperatures")
    print(df_avg.to_string(index=False))
    
    return df_high_temp


def example_3_real_workflow():
    """Example: Complete workflow for your data."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Complete Workflow for Your Data")
    print("="*70)
    
    print("""
Step-by-step guide:

1. Prepare your data in one of these formats:
   
   a) Wide CSV (solvents as columns):
      polymer,Acetone,Water,THF,Chloroform,...
      PMMA,150.5,0.01,180.3,200.1,...
      PS,80.2,0.02,120.4,150.8,...
   
   b) Long CSV (one row per measurement):
      polymer,solvent,solubility_g_L,temperature_C
      PMMA,Acetone,150.5,25
      PMMA,Water,0.01,25
   
   c) Multi-sheet Excel (one sheet per polymer):
      Sheet "PMMA": solvent,solubility_g_L,temperature_C
      Sheet "PS": solvent,solubility_g_L,temperature_C

2. Load and format your data:
    """)
    
    # Show code examples
    print("   # For wide CSV:")
    print("   df = pd.read_csv('your_data.csv')")
    print("   df_long = format_wide_to_long(df, polymer_col='polymer')")
    print()
    print("   # For multi-sheet Excel:")
    print("   df_long = format_multi_sheet_excel('your_data.xlsx')")
    print()
    print("   # For separate files:")
    print("   df_long = format_separate_files('solubility_data/', pattern='*.csv')")
    
    print("""
3. Validate the formatted data:
   df_validated = validate_solubility_data(df_long)

4. Save in standardized format:
   df_validated.to_csv('solubility_data_formatted.csv', index=False)

5. Run active learning:
   python polymer_al_real_data.py --data solubility_data_formatted.csv
    """)


# ==========================================
# INTERACTIVE DATA FORMATTER
# ==========================================

def interactive_formatter():
    """Interactive tool to help format your data."""
    print("\n" + "="*70)
    print("INTERACTIVE DATA FORMATTER")
    print("="*70)
    
    print("""
I'll help you format your solubility data for active learning.

What format is your data currently in?

1. Wide CSV (solvents as columns)
2. Long CSV (already in correct format, just need validation)
3. Multi-sheet Excel (one sheet per polymer)
4. Separate files (one file per polymer)
5. Other (I'll need to see your data first)

Enter choice (1-5): """, end='')
    
    try:
        choice = input()
        
        if choice == '1':
            filepath = input("Enter CSV file path: ")
            df = pd.read_csv(filepath)
            print("\nDetected columns:", list(df.columns))
            polymer_col = input("Which column contains polymer names? [polymer]: ") or 'polymer'
            df_long = format_wide_to_long(df, polymer_col=polymer_col)
            
        elif choice == '2':
            filepath = input("Enter CSV file path: ")
            df_long = pd.read_csv(filepath)
            
        elif choice == '3':
            filepath = input("Enter Excel file path: ")
            df_long = format_multi_sheet_excel(filepath)
            
        elif choice == '4':
            directory = input("Enter directory path: ")
            pattern = input("Enter file pattern [*_solubility.csv]: ") or '*_solubility.csv'
            df_long = format_separate_files(directory, pattern)
            
        else:
            print("Please run one of the examples above to see the expected format.")
            return
        
        # Validate
        df_validated = validate_solubility_data(df_long)
        
        # Save
        output_path = input("\nSave formatted data as [solubility_data_formatted.csv]: ") or "solubility_data_formatted.csv"
        df_validated.to_csv(output_path, index=False)
        print(f"\n✓ Success! Data saved to: {output_path}")
        print("\nYou can now run:")
        print(f"  python polymer_al_real_data.py --data {output_path}")
        
    except KeyboardInterrupt:
        print("\n\nCancelled.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease check your data format and try again.")


# ==========================================
# MAIN
# ==========================================

def main():
    """Run examples or interactive formatter."""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--interactive':
            interactive_formatter()
        else:
            print("Usage: python format_solubility_data.py [--interactive]")
    else:
        # Show all examples
        example_1_wide_format()
        example_2_temperature_dependent()
        example_3_real_workflow()
        
        print("\n" + "="*70)
        print("Want help formatting YOUR data?")
        print("Run: python format_solubility_data.py --interactive")
        print("="*70)


if __name__ == "__main__":
    main()
