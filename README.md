# COSMO-Therm SLE Correction Model

Machine learning model to correct COSMO-therm Solid-Liquid Equilibrium (SLE) calculations using reference solubility data.

## Project Overview

This project aims to build a correction model for COSMO-therm SLE calculations by training on experimental reference data. The SLE method requires no experimental reference input but is less accurate. By training on data from the more accurate reference solubility method, we can develop corrections for the SLE predictions.

### Dataset
- **Polymers**: ~10 solvents (starting with EVOH)
- **Solvents**: 30 common solvents + 1009 total solvents
- **Temperature Range**: 25°C to 130°C in 5°C increments
- **Methods**:
  - REF: Reference solubility (more accurate, requires experimental input)
  - SLE: Solid-liquid equilibrium (less accurate, no experimental input needed)

## Directory Structure

```
COSMO-POLYMER-ML/
├── data/
│   ├── raw/              # Original CSV files from COSMO-therm
│   └── processed/        # Cleaned and processed datasets
├── notebooks/            # Jupyter notebooks for analysis
├── src/
│   ├── data_processing/  # Data loading and preprocessing utilities
│   ├── visualization/    # Plotting and visualization functions
│   └── models/          # ML model definitions and training
├── results/
│   ├── figures/         # Generated plots and visualizations
│   └── metrics/         # Model performance metrics
└── models/              # Saved trained models
```

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start: Analyze EVOH Data

1. Place your COSMO-therm tab files in `data/raw/`:
   - `EVOH-REF.tab` (Reference method)
   - `EVOH-SLE.tab` (SLE method)

2. Run the analysis:
   ```bash
   cd notebooks
   python analyze_evoh.py ../data/raw/EVOH-REF.tab ../data/raw/EVOH-SLE.tab
   ```

3. View results:
   - Processed CSVs: `data/processed/evoh/`
   - Comparison plots: `results/figures/evoh/`

See [notebooks/README.md](notebooks/README.md) for detailed usage instructions.

### Quick Start: Train Correction Model

After analyzing your reference polymers:

```bash
cd notebooks
# Train correction model on EVOH data
python train_correction_model.py ../data/processed/evoh/merged_comparison.csv

# Apply to new polymer (no experimental data needed!)
python predict_new_polymer.py \
    ../data/raw/NEW-POLYMER-SLE.tab \
    ../models/evoh/correction_model_log10_x.joblib
```

See [notebooks/README_MODELING.md](notebooks/README_MODELING.md) for complete modeling workflow.

### Quick Start: Model the common-solvents CSVs

The `data/common-solvents` folder contains solubility–temperature curves on a
5 °C grid for multiple polymers. You can train and score a model directly from
those CSVs with the new utilities in `src/models/common_solvent_model.py`:

```bash
python - <<'PY'
from pathlib import Path
from src.models.common_solvent_model import (
    run_common_solvent_active_learning,
    run_common_solvent_cv_and_visualize,
)

data_dir = Path("data/common-solvents")

# 1) Grouped CV holding out entire polymers (saves CSV + figure to results/common-solvents/cv)
cv_df, cv_paths = run_common_solvent_cv_and_visualize(
    data_dir,
    model_types=("ridge", "gbr", "rf"),
    n_splits=3,
)
print("CV metrics saved to:", cv_paths)

# 2) Train an ensemble and rank candidate experiments (CSV + figure to results/common-solvents/active_learning)
ranking_df, al_paths = run_common_solvent_active_learning(
    data_dir,
    n_members=5,
    model_type="gbr",
    top_k=10,
)
print("Active-learning ranking saved to:", al_paths)
PY
```

## Workflow

1. **Data Comparison** (`analyze_evoh.py`): Compare REF vs SLE for polymers with experimental data
2. **Model Training** (`train_correction_model.py`): Learn systematic corrections from ~10 reference polymers
3. **Prediction** (`predict_new_polymer.py`): Apply corrections to new polymers without experimental data

## Key Features

### Data Processing
- **Tab File Parser**: Reads COSMO-therm output tab files (REF and SLE formats)
- **Automatic Filtering**: Matches REF data to SLE solvents and temperature ranges
- **Standardized Output**: Generates clean CSV files with consistent column names

### Variables Compared
| Variable | Description | Units |
|----------|-------------|-------|
| **log10(x)** | Mole fraction solubility | - |
| **log10(S)** | Volume-based solubility | log₁₀(mol/L) |
| **w** | Mass fraction solubility | g/g |
| **Δμ** | Chemical potential difference | kcal/mol |

### Visualizations
Each comparison includes 4-panel plots:
1. **Parity Plot**: REF vs SLE with temperature coloring and statistics (R², MAE, RMSE)
2. **Error Distribution**: Histogram showing (SLE - REF) errors
3. **Error vs Temperature**: Systematic bias across temperature range
4. **Error by Solvent**: Identifies solvents with largest prediction errors

## Data Files

- Tab files from COSMO-therm calculations (REF and SLE methods)
- Starting with EVOH polymer + common solvents
- Temperature range: 25-130°C in 5°C increments
