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

1. Place raw CSV files in `data/raw/`
2. Install dependencies: `pip install -r requirements.txt`
3. Run analysis notebooks in `notebooks/`

## Data Files

- `EVOH-SLE-COMMON.csv`: EVOH polymer with common solvents using SLE method
- `EVOH-REF-COMMON.csv`: EVOH polymer with common solvents using reference method
