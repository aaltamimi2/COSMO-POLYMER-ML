# SLE Correction Model Workflow

## Overview

This workflow trains a machine learning model to correct COSMO-therm SLE predictions using experimental reference solubility data from ~10 polymers. Once trained, the model can predict corrections for new polymers without experimental data.

## Quick Start

### 1. Train Correction Model (on EVOH or other reference polymer)

```bash
cd notebooks
python train_correction_model.py ../data/processed/evoh/merged_comparison.csv
```

This will:
- Compare multiple model types (Ridge, Lasso, Random Forest, Gradient Boosting)
- Train the best model on your data
- Save the trained model to `models/evoh/correction_model_log10_x.joblib`
- Generate diagnostic plots showing improvement

### 2. Apply Model to New Polymer (no experimental data needed)

```bash
python predict_new_polymer.py \
    ../data/raw/NEW-POLYMER-SLE.tab \
    ../models/evoh/correction_model_log10_x.joblib
```

This will:
- Parse the new polymer's SLE data
- Apply learned corrections
- Save corrected predictions to `results/predictions/`

### 3. Batch Processing for Multiple Polymers

```bash
python predict_new_polymer.py batch \
    ../models/evoh/correction_model_log10_x.joblib \
    ../data/raw/POLYMER1-SLE.tab \
    ../data/raw/POLYMER2-SLE.tab \
    ../data/raw/POLYMER3-SLE.tab
```

## Model Features

Based on your EVOH analysis, the model uses these key features (in order of importance):

### Temperature Features (CRITICAL - explains most variance)
- `Temperature_K` - Absolute temperature
- `inv_T` - Inverse temperature (1/T)
- `Temperature_C` - Celsius temperature

### SLE Predictions (available for new polymers)
- `log10_x_SLE` - Original SLE prediction
- `delta_mu_SLE` - Chemical potential difference (μ_solv - μ_self)

### Solvent Properties
- `Solvent_MolWeight` - Molecular weight
- `Solvent_density` - Density
- `molar_volume` - MW/density

## Key Insights from EVOH Data

1. **Temperature Dependence is Dominant**
   - Error decreases from ~3.2 (300K) to ~0.7 (400K)
   - Linear or polynomial in 1/T works well

2. **Systematic Positive Bias**
   - SLE overestimates solubility by ~1.8 log units on average
   - Correctable with simple models

3. **Chemical Potential (Δμ) is Highly Predictive**
   - R² = 0.929 (better than log10_x directly)
   - Use as auxiliary feature

4. **Solvent Effects are Moderate**
   - Some solvents have consistently larger errors
   - Captured by molecular descriptors

## Model Types

### Ridge Regression (Default - Recommended)
**Pros:** Fast, interpretable, robust
**Cons:** Assumes linear relationships
**Use when:** You want interpretability and fast training

```bash
python train_correction_model.py data.csv log10_x ridge
```

### Random Forest
**Pros:** Captures non-linear effects, feature importance
**Cons:** Slower, can overfit
**Use when:** You have enough data (>500 points) and want to capture complex patterns

```bash
python train_correction_model.py data.csv log10_x rf
```

### Gradient Boosting
**Pros:** Often best performance, handles non-linearity
**Cons:** Slow training, can overfit
**Use when:** You need maximum accuracy and have sufficient data

```bash
python train_correction_model.py data.csv log10_x gbm
```

## Multi-Polymer Training Strategy

Once you have ~10 polymers with REF data:

### Option 1: Pooled Model (Recommended to start)
Train single model on all polymers combined:

```bash
# Merge all polymer data
cat ../data/processed/*/merged_comparison.csv > ../data/processed/all_polymers_merged.csv

# Train on combined data
python train_correction_model.py ../data/processed/all_polymers_merged.csv
```

**Pros:** Maximum data, learns general patterns
**Cons:** May miss polymer-specific effects

### Option 2: Polymer-Specific Models
Train separate model for each polymer, use ensemble:

```bash
for polymer in evoh pva pvac ...; do
    python train_correction_model.py ../data/processed/$polymer/merged_comparison.csv
done
```

**Pros:** Captures polymer-specific behavior
**Cons:** Less data per model, need strategy for new polymers

### Option 3: Two-Stage Model (Advanced)
1. Train global model on all data
2. Train polymer-specific correction on residuals

## Expected Performance

Based on EVOH single-polymer results:

| Metric | Original SLE | After Correction (Ridge) |
|--------|--------------|--------------------------|
| MAE (log10_x) | 1.836 | **~0.5-0.8** (expected) |
| RMSE (log10_x) | 2.008 | **~0.7-1.0** (expected) |
| R² | 0.853 | **~0.95+** (expected) |

With ~10 polymers, expect even better generalization.

## Validation Strategy

### For Single Polymer (Current)
Use cross-validation (5-fold):
```python
# Built into train_correction_model.py
```

### For Multiple Polymers
Use leave-one-polymer-out (LOPO):
```python
# Train on 9 polymers, test on 10th
# Repeat for each polymer
# This tests generalization to new polymers
```

## Files and Outputs

### Input
- `merged_comparison.csv` - REF-SLE comparison data from `analyze_evoh.py`

### Output
- `correction_model_log10_x.joblib` - Trained model
- `model_comparison_log10_x.csv` - Cross-validation results for different models
- `corrected_predictions_log10_x.csv` - Training data with corrections
- `correction_results_log10_x.png` - Before/after visualization

## Tips for Success

1. **Always check temperature range match**
   - Model trained on 300-400K won't extrapolate well to 450K
   - Ensure new polymers use similar temperature range

2. **Start simple (Ridge), add complexity if needed**
   - Ridge regression often works as well as complex models
   - Simpler = more interpretable and robust

3. **Monitor for overfitting**
   - Check cross-validation scores
   - If training R² >> CV R², you're overfitting

4. **Use common solvents for training**
   - Model learns better from solvents it has seen
   - Filter to common solvents across all polymers

5. **Check residual patterns**
   - If errors are systematic (vs temperature, solvent type), add features
   - If random, you're done!

## Next Steps

Once you have data for ~10 polymers:

1. Run `analyze_evoh.py` on each polymer
2. Check if error patterns are similar across polymers
3. Merge data and train pooled model
4. Evaluate with leave-one-polymer-out CV
5. Apply to new polymers!
