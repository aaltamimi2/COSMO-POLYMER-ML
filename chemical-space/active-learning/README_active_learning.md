# Polymer Solubility Active Learning System

A complete Bayesian optimization / active learning framework for discovering polymers with optimal solubility properties.

## 📦 Files

### 1. `polymer_solubility_active_learning.py` (Main Framework)
Complete active learning system with:
- Gaussian Process surrogate model
- Multiple acquisition functions (EI, UCB, PI, Greedy)
- Normalized solubility scoring (mean, max, versatility)
- Comprehensive visualizations
- Synthetic data generation (for testing)

**Usage:**
```bash
python polymer_solubility_active_learning.py
```

**Output:**
- `active_learning_output/gp_predictions_iterXX.png` - Model predictions & uncertainty
- `active_learning_output/acquisition_comparison_iterXX.png` - Acquisition functions
- `active_learning_output/learning_curve.png` - Progress over iterations
- `active_learning_output/campaign_history.csv` - Full history

---

### 2. `polymer_al_real_data.py` (For Your Data)
Adapts the framework to use YOUR solubility predictions.

**Usage:**
```bash
# Basic usage
python polymer_al_real_data.py --data your_solubility.csv

# Optimize for water solubility
python polymer_al_real_data.py --data your_solubility.csv --method specific --solvent Water

# Use UCB with 15 iterations
python polymer_al_real_data.py --data your_solubility.csv --acquisition ucb --iterations 15
```

**Scoring Methods:**
- `versatility` (default): Breadth × magnitude across all solvents
- `max`: Best single solvent
- `mean`: Average solubility
- `count`: Number of solvents above threshold
- `specific`: Target specific solvent (e.g., Water)
- `weighted`: Weight solvents by importance
- `custom`: User-defined function

---

### 3. `format_solubility_data.py` (Data Formatter)
Helps convert your data into the required format.

**Required Format:**
```
polymer | solvent | solubility_g_L | temperature_C
--------|---------|----------------|---------------
PMMA    | Acetone | 150.5          | 25
PMMA    | Water   | 0.01           | 25
PS      | Toluene | 100.0          | 25
...
```

**Usage:**
```bash
# See examples
python format_solubility_data.py

# Interactive formatter
python format_solubility_data.py --interactive
```

**Supports:**
- Wide CSV (solvents as columns)
- Long CSV (one row per measurement)
- Multi-sheet Excel (one polymer per sheet)
- Separate files (one polymer per file)

---

## 🚀 Quick Start

### Option 1: Test with Synthetic Data
```bash
python polymer_solubility_active_learning.py
```
Runs a demo with synthetic solubility data for 29 polymers × 30 solvents.

### Option 2: Use Your Data
```bash
# Step 1: Format your data
python format_solubility_data.py --interactive

# Step 2: Run active learning
python polymer_al_real_data.py --data solubility_data_formatted.csv
```

---

## 📊 How It Works

### 1. **Initialization**
- Start with 3 random polymers (or smart initial selection)
- Measure/predict their solubility
- Fit initial Gaussian Process model

### 2. **Active Learning Loop**
For each iteration:
1. **Predict**: GP predicts solubility for all untested polymers
2. **Acquire**: Acquisition function balances exploration vs exploitation
   - **EI** (Expected Improvement): Good all-arounder
   - **UCB** (Upper Confidence Bound): Optimistic exploration
   - **PI** (Probability of Improvement): Conservative
   - **Greedy**: Pure exploitation
3. **Test**: Measure/predict top suggestion
4. **Update**: Add to training data, refit GP
5. **Repeat**

### 3. **Output**
- Suggestions for next polymers to test
- Uncertainty maps
- Model predictions
- Learning curves

---

## 🎯 Acquisition Functions

| Function | When to Use | Parameter |
|----------|-------------|-----------|
| **EI** (Expected Improvement) | Default, balanced | `xi=0.01` (exploration) |
| **UCB** (Upper Confidence Bound) | Want to explore broadly | `kappa=2.0` (exploration) |
| **PI** (Probability of Improvement) | Conservative, risk-averse | `xi=0.01` |
| **Greedy** | Late-stage, exploit known good regions | - |

**Tuning:**
- Higher `xi` or `kappa` → more exploration (try uncertain regions)
- Lower values → more exploitation (focus on predicted best)

---

## 📈 Solubility Metrics

### Versatility Score (Default)
Combines breadth (how many solvents) and magnitude (how much dissolves):
```python
log_sols = log10(solubilities + 1)
versatility = mean(log_sols) × (1 + fraction_soluble)
```
**Best for:** Finding polymers that dissolve in many solvents

### Max Solubility
Simply the best single solvent.
**Best for:** When you only care about one very good solvent

### Specific Solvent
Target a particular solvent (e.g., water, acetone).
**Best for:** Application-specific optimization

### Custom
Define your own scoring function:
```python
def custom_score(poly_data):
    water = poly_data[poly_data['solvent']=='Water']['solubility_g_L'].mean()
    organic = poly_data[poly_data['solvent']=='Acetone']['solubility_g_L'].mean()
    return 0.7 * water + 0.3 * organic  # Prioritize water
```

---

## 🔧 Advanced Usage

### Custom Acquisition Function
```python
from polymer_solubility_active_learning import PolymerActiveLearner

learner = PolymerActiveLearner(X_pool, names, metric='versatility_score_norm')
# ... add observations ...

# Custom acquisition: pure uncertainty sampling
def acquisition_uncertainty(X):
    _, sigma = learner.predict(X, return_std=True)
    return sigma

suggestions = learner.suggest_next(method='custom', acquisition_func=acquisition_uncertainty)
```

### Multi-Objective Optimization
```python
# Optimize for both water AND organic solubility
def multi_objective_score(poly_data):
    water = poly_data[poly_data['solvent']=='Water']['solubility_g_L'].mean()
    organic_solvents = ['Acetone', 'THF', 'Chloroform']
    organic = poly_data[poly_data['solvent'].isin(organic_solvents)]['solubility_g_L'].mean()
    
    # Pareto-inspired: multiply objectives
    return np.sqrt(water * organic)
```

### Batch Suggestions
```python
# Get top 10 suggestions at once (for parallel experiments)
suggestions = learner.suggest_next(method='ei', n_suggestions=10)

for idx, name, acq_val in suggestions:
    print(f"{name}: acq={acq_val:.4f}")
```

---

## 📝 Example Workflows

### Workflow 1: Find Best Water-Soluble Polymer
```bash
python polymer_al_real_data.py \
    --data solubility.csv \
    --method specific \
    --solvent Water \
    --acquisition ei \
    --iterations 15
```

### Workflow 2: Optimize for Coating Applications
(Needs good solubility in acetone, THF, and toluene)
```python
# In Python:
from polymer_al_real_data import run_campaign_with_real_data

coating_weights = {
    'Acetone': 2.0,
    'THF': 2.0,
    'Toluene': 1.5,
    'Ethyl_acetate': 1.0
}

learner, scores, history = run_campaign_with_real_data(
    'solubility.csv',
    scoring_method='weighted',
    solvent_weights=coating_weights,
    acquisition='ucb',
    n_iterations=12
)
```

### Workflow 3: Active Learning with Temperature
```python
# Filter to high-temperature data only
df = pd.read_csv('solubility.csv')
df_high_temp = df[df['temperature_C'] >= 50]
df_high_temp.to_csv('solubility_high_temp.csv', index=False)

# Then run AL on high-temp data
python polymer_al_real_data.py --data solubility_high_temp.csv
```

---

## 📚 Key Algorithms

### Gaussian Process Regression
- **Kernel**: Matérn (ν=2.5) + White noise
- **Normalization**: Standardized features + normalized targets
- **Optimizer**: 10 random restarts for hyperparameters

### Feature Engineering
- **Molecular fingerprints**: Morgan (radius=2, 512 bits)
- **Dimensionality reduction**: PCA (50 components)
- **Standardization**: Zero mean, unit variance

### Acquisition Optimization
- Evaluates all untested candidates
- Returns top-k by acquisition value
- Handles edge cases (no data, all tested)

---

## 🐛 Troubleshooting

**GP not fitting:**
- Check for duplicate observations
- Ensure y-values are not constant
- Try increasing `alpha` parameter (noise level)

**Slow performance:**
- Reduce PCA components (currently 50)
- Use greedy acquisition (faster than EI/UCB)
- Batch process suggestions

**Poor predictions:**
- Need more initial data (try n_initial=5-10)
- Check feature quality (are SMILES valid?)
- Verify solubility data isn't too noisy

**All suggestions similar:**
- Increase exploration (higher `xi` or `kappa`)
- Try UCB instead of EI
- Add diversity constraint

---

## 📖 References

**Bayesian Optimization:**
- Mockus et al. (1978) - Global optimization with Bayesian approach
- Jones et al. (1998) - Efficient Global Optimization (EGO)
- Snoek et al. (2012) - Practical Bayesian Optimization

**Active Learning:**
- Settles (2009) - Active Learning Literature Survey
- Lookman et al. (2019) - Active learning in materials science

**Polymer Informatics:**
- Venkatram et al. (2021) - Polymer genome and informatics
- Chen et al. (2022) - Machine learning for polymer properties

---

## 💡 Tips

1. **Start with 3-5 random samples** to seed the GP
2. **Use EI for general exploration**, UCB when data is sparse
3. **Normalize your solubility data** if values span orders of magnitude
4. **Validate with held-out test set** if you have known good polymers
5. **Plot uncertainty maps** to understand model confidence
6. **Compare acquisition functions** - they often suggest different polymers
7. **Use domain knowledge** to design custom scoring functions

---

## 📧 Questions?

This framework is designed to be:
- **Flexible**: Easy to adapt to different metrics and constraints
- **Transparent**: Full visibility into predictions and uncertainty
- **Practical**: Tested with realistic polymer solubility data

Happy optimizing! 🚀
