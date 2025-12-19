# Analysis Notebooks

## Quick Start: Analyzing EVOH Data

### Step 1: Place your tab files

Put your COSMO-therm tab files in the `data/raw/` directory:
```bash
data/raw/
├── EVOH-REF.tab      # Reference method results
└── EVOH-SLE.tab      # SLE method results
```

### Step 2: Run the analysis

```bash
cd notebooks
python analyze_evoh.py ../data/raw/EVOH-REF.tab ../data/raw/EVOH-SLE.tab
```

This will:
1. Parse both tab files
2. Filter REF data to match SLE solvents and temperatures
3. Generate cleaned CSV files in `data/processed/evoh/`
4. Create comprehensive comparison plots in `results/figures/evoh/`

### Generated Outputs

**CSV Files** (`data/processed/evoh/`):
- `ref_filtered.csv` - REF data filtered to match SLE conditions
- `sle.csv` - SLE data
- `merged_comparison.csv` - Matched REF-SLE pairs with error metrics

**Plots** (`results/figures/evoh/`):
- `comparison_log10_x.png` - Mole fraction solubility comparison
- `comparison_log10_S.png` - Volume-based solubility comparison
- `comparison_w.png` - Mass fraction comparison
- `comparison_delta_mu.png` - Chemical potential difference comparison

Each plot contains 4 panels:
1. **Parity plot**: REF vs SLE with temperature coloring
2. **Error distribution**: Histogram of (SLE - REF) errors
3. **Error vs Temperature**: How errors vary with temperature
4. **Error by Solvent**: Top solvents with largest errors

## Variables Compared

The analysis focuses on these key variables:

| Variable | Description | Units | Column Name |
|----------|-------------|-------|-------------|
| **log10(x)** | Mole fraction solubility (log scale) | - | `log10_x` |
| **log10(S)** | Volume-based solubility (log scale) | log₁₀(mol/L) | `log10_S` |
| **w** | Mass fraction solubility | g/g | `w` |
| **Δμ** | Chemical potential difference | kcal/mol | `delta_mu` |

### Why these variables?

- **log10(x)**: Most fundamental measure, directly from thermodynamics
- **log10(S)**: Practical measure accounting for density
- **w**: Industrial/formulation relevance
- **Δμ**: Thermodynamic driving force for dissolution

## Using the Modules Directly

### Parse a single tab file

```python
import sys
sys.path.insert(0, '../src')

from data_processing.tab_parser import parse_tab_file

df = parse_tab_file('../data/raw/EVOH-REF.tab')
print(df.head())
```

### Process and merge datasets

```python
from data_processing.process_tab_files import process_and_merge_data

ref_df, sle_df, merged_df = process_and_merge_data(
    '../data/raw/EVOH-REF.tab',
    '../data/raw/EVOH-SLE.tab',
    output_dir='../data/processed/evoh'
)
```

### Create custom plots

```python
from visualization.comparison_plots import plot_parity, create_comprehensive_comparison
import matplotlib.pyplot as plt

# Single parity plot
fig, ax = plt.subplots(figsize=(8, 8))
plot_parity(merged_df, variable='log10_x', ax=ax)
plt.show()

# Complete 4-panel comparison
fig = create_comprehensive_comparison(
    merged_df,
    variable='log10_x',
    output_path='../results/figures/my_plot.png'
)
```

## Next Steps for Other Polymers

To analyze other polymers, simply run:

```bash
python analyze_evoh.py ../data/raw/POLYMER-REF.tab ../data/raw/POLYMER-SLE.tab
```

The script will automatically organize outputs by polymer name.
