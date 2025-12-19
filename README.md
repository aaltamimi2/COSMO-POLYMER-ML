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
