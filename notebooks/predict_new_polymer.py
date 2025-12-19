"""
Apply trained correction model to new polymer SLE predictions

Use this script to correct SLE predictions for polymers where you don't
have experimental reference solubility data.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_processing.tab_parser import parse_tab_file, extract_common_columns
from models.feature_engineering import create_feature_matrix
from models.correction_model import SLECorrectionModel, apply_correction


def predict_corrected_solubility(
    sle_tab_file: str,
    model_file: str,
    target_variable: str = 'log10_x',
    output_file: str = None
) -> pd.DataFrame:
    """
    Apply correction model to new polymer SLE data

    Parameters:
    -----------
    sle_tab_file : str
        Path to SLE tab file for new polymer
    model_file : str
        Path to trained correction model (.joblib)
    target_variable : str
        Variable being corrected
    output_file : str, optional
        Path to save corrected predictions

    Returns:
    --------
    DataFrame with original SLE and corrected predictions
    """

    print("="*80)
    print("Applying SLE Correction Model to New Polymer")
    print("="*80)

    # Parse SLE data
    print(f"\nParsing SLE tab file: {sle_tab_file}")
    sle_raw = parse_tab_file(sle_tab_file)
    print(f"  Found {len(sle_raw)} data points")

    # Extract and clean
    print("Extracting and standardizing columns...")
    sle_df = extract_common_columns(sle_raw, 'SLE')
    print(f"  Valid data points: {len(sle_df)}")

    # Load trained model
    print(f"\nLoading trained model: {model_file}")
    model = SLECorrectionModel.load(model_file)
    print(f"  Model type: {model.model_type}")
    print(f"  Features: {model.feature_names}")

    # Create features for prediction
    print("\nCreating features...")
    df_feat = create_feature_matrix(sle_df, include_sle_predictions=True, method_suffix='_SLE')

    # Ensure all required features are present
    missing_features = set(model.feature_names) - set(df_feat.columns)
    if missing_features:
        print(f"WARNING: Missing features: {missing_features}")
        print("Adding zero-filled columns for missing features")
        for feat in missing_features:
            df_feat[feat] = 0.0

    # Extract features in correct order
    X_pred = df_feat[model.feature_names]

    # Predict corrections
    print("\nPredicting corrections...")
    corrections = model.predict(X_pred)

    # Apply corrections
    sle_values = sle_df[f'{target_variable}']
    corrected_values = apply_correction(sle_values, corrections)

    # Create results dataframe
    results = sle_df.copy()
    results[f'{target_variable}_original'] = sle_values
    results[f'{target_variable}_correction'] = corrections
    results[f'{target_variable}_corrected'] = corrected_values

    # Summary statistics
    print("\n" + "="*80)
    print("Correction Summary")
    print("="*80)
    print(f"\nOriginal {target_variable} range: {sle_values.min():.3f} to {sle_values.max():.3f}")
    print(f"Correction range: {corrections.min():.3f} to {corrections.max():.3f}")
    print(f"Corrected {target_variable} range: {corrected_values.min():.3f} to {corrected_values.max():.3f}")

    print(f"\nMean correction: {corrections.mean():.3f} ± {corrections.std():.3f}")

    # Save if output file specified
    if output_file:
        results.to_csv(output_file, index=False)
        print(f"\nSaved corrected predictions to: {output_file}")

    return results


def batch_predict(
    sle_tab_files: list,
    model_file: str,
    target_variable: str = 'log10_x',
    output_dir: str = None
) -> dict:
    """
    Apply correction to multiple polymer SLE files

    Parameters:
    -----------
    sle_tab_files : list
        List of paths to SLE tab files
    model_file : str
        Path to trained correction model
    target_variable : str
        Variable to correct
    output_dir : str, optional
        Directory to save outputs

    Returns:
    --------
    Dictionary mapping filenames to result DataFrames
    """

    results = {}

    for sle_file in sle_tab_files:
        polymer_name = Path(sle_file).stem

        print(f"\n{'='*80}")
        print(f"Processing: {polymer_name}")
        print('='*80)

        output_file = None
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = str(output_path / f'{polymer_name}_corrected.csv')

        try:
            result = predict_corrected_solubility(
                sle_file,
                model_file,
                target_variable=target_variable,
                output_file=output_file
            )
            results[polymer_name] = result
        except Exception as e:
            print(f"ERROR processing {polymer_name}: {e}")
            continue

    print(f"\n{'='*80}")
    print(f"Batch Processing Complete")
    print('='*80)
    print(f"Successfully processed {len(results)}/{len(sle_tab_files)} files")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict_new_polymer.py <sle_tab_file> <model_file> [target_variable]")
        print("\nExample:")
        print("  python predict_new_polymer.py ../data/raw/NEW-POLYMER-SLE.tab ../models/evoh/correction_model_log10_x.joblib")
        print("  python predict_new_polymer.py ../data/raw/NEW-POLYMER-SLE.tab ../models/evoh/correction_model_log10_x.joblib log10_x")
        print("\nFor batch processing, use:")
        print("  python predict_new_polymer.py batch <model_file> <sle_file1> <sle_file2> ...")
        sys.exit(1)

    if sys.argv[1] == 'batch':
        # Batch mode
        model_file = sys.argv[2]
        sle_files = sys.argv[3:]
        target_var = 'log10_x'  # Could add as parameter

        output_dir = Path(__file__).parent.parent / 'results' / 'predictions'

        results = batch_predict(
            sle_files,
            model_file,
            target_variable=target_var,
            output_dir=str(output_dir)
        )

    else:
        # Single file mode
        sle_file = sys.argv[1]
        model_file = sys.argv[2]
        target_var = sys.argv[3] if len(sys.argv) > 3 else 'log10_x'

        # Determine output file
        polymer_name = Path(sle_file).stem
        output_dir = Path(__file__).parent.parent / 'results' / 'predictions'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(output_dir / f'{polymer_name}_corrected.csv')

        results = predict_corrected_solubility(
            sle_file,
            model_file,
            target_variable=target_var,
            output_file=output_file
        )

        print("\n" + "="*80)
        print("PREDICTION COMPLETE!")
        print("="*80)
        print(f"\nCorrected predictions saved to: {output_file}")
