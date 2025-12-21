"""
Bayesian Optimization / Active Learning for Polymer Solubility Discovery

Features:
  ✅ Gaussian Process surrogate model
  ✅ Multiple acquisition functions (EI, UCB, PI, Greedy)
  ✅ Normalized solubility scoring
  ✅ Uncertainty visualization
  ✅ Iterative suggestions with simulated experiments
  ✅ Chemistry-aware feature space (Morgan FPs)

Usage:
  python polymer_solubility_active_learning.py

Install:
  pip install numpy pandas matplotlib scikit-learn scikit-optimize rdkit-pypi
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import norm
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

import warnings
warnings.filterwarnings('ignore', category=UserWarning)


# ==========================================
# 1) POLYMER + SOLVENT DATABASE
# ==========================================

def build_polymer_database():
    """Build comprehensive polymer database with SMILES."""
    polymers = {
        # Base polymers matching REF-DATA
        "HDPE": "CC",
        "LDPE": "CC",
        "PP-LMW": "CC(C)",  # Low MW polypropylene (in REF-DATA)
        "PS": "CC(c1ccccc1)",
        "PVC": "CC(Cl)",
        "PC": "CC(C)(c1ccc(OC(=O)Oc2ccc(C(C)(C)c3ccc(O)cc3)cc2)cc1)",
        "PET": "O=C(c1ccc(C(=O)OCCO)cc1)",
        "EVOH": "CC(O)",
        "nylon66": "NCCCCCCNC(=O)CCCCC(=O)",  # Match REF-DATA naming

        # Additional polymers for expanded search space
        "PETG": "O=C(c1ccc(C(=O)OCC(O)CO)cc1)",
        "PVDF": "CC(F)F",
        "Nylon6": "NCCCCCC(=O)",
        "Polysulfone": "O=S(=O)(c1ccc(Oc2ccc(C(C)(C)c3ccc(O)cc3)cc2)cc1)c4ccc(O)cc4",
        "Polyurethane": "CCCCCCNC(=O)OCCCCOC(=O)N",
        
        # Dissolved polymers (original)
        "PMMA": "CC(C)(C(=O)OC)",
        "PVA": "CC(O)",
        "PAN": "CC(C#N)",
        "PVP": "CC(N1CCCC1=O)",
        "Nitrocellulose": "C([C@@H]1[C@H]([C@@H]([C@H]([C@@H](O1)O)O)O)O)O[N+](=O)[O-]",  # Simplified
        
        # Dissolved polymers (new)
        "PVB": "CC(O)CC(c1ccccc1)",
        "PEO": "OCCO",
        "PLA": "CC(O)C(=O)",
        "CA": "CC(=O)OC[C@H]1O[C@H](OC(C)=O)[C@H](OC(C)=O)[C@@H](OC(C)=O)[C@@H]1OC(C)=O",
        "PAA": "CC(C(=O)O)",
        "PAAm": "CC(C(=O)N)",
        "PEI": "NCCN",
        "PVAc": "CC(OC(=O)C)",
        "PDMS": "C[Si](C)(C)O[Si](C)(C)C",
        "PU-ester": "CCCCOC(=O)NCCCCCCNC(=O)OCCCCOC(=O)N",
    }
    
    df = pd.DataFrame([
        {"polymer": name, "smiles": smiles}
        for name, smiles in polymers.items()
    ])
    return df


def get_common_solvents():
    """30 common industrial solvents with properties."""
    return {
        # Water & alcohols
        "Water": {"polarity": "high", "category": "protic"},
        "Methanol": {"polarity": "high", "category": "protic"},
        "Ethanol": {"polarity": "high", "category": "protic"},
        "Isopropanol": {"polarity": "medium", "category": "protic"},
        "Butanol": {"polarity": "medium", "category": "protic"},
        
        # Ketones
        "Acetone": {"polarity": "medium", "category": "aprotic_polar"},
        "MEK": {"polarity": "medium", "category": "aprotic_polar"},
        "Cyclohexanone": {"polarity": "medium", "category": "aprotic_polar"},
        
        # Esters
        "Ethyl_acetate": {"polarity": "medium", "category": "aprotic_polar"},
        "Butyl_acetate": {"polarity": "medium", "category": "aprotic_polar"},
        
        # Ethers
        "THF": {"polarity": "medium", "category": "aprotic_polar"},
        "Dioxane": {"polarity": "medium", "category": "aprotic_polar"},
        "Diethyl_ether": {"polarity": "low", "category": "aprotic_polar"},
        
        # Chlorinated
        "Chloroform": {"polarity": "low", "category": "aprotic_nonpolar"},
        "DCM": {"polarity": "low", "category": "aprotic_polar"},
        "Carbon_tetrachloride": {"polarity": "low", "category": "aprotic_nonpolar"},
        
        # Aromatics
        "Benzene": {"polarity": "low", "category": "aprotic_nonpolar"},
        "Toluene": {"polarity": "low", "category": "aprotic_nonpolar"},
        "Xylene": {"polarity": "low", "category": "aprotic_nonpolar"},
        
        # Hydrocarbons
        "Hexane": {"polarity": "low", "category": "aprotic_nonpolar"},
        "Heptane": {"polarity": "low", "category": "aprotic_nonpolar"},
        "Cyclohexane": {"polarity": "low", "category": "aprotic_nonpolar"},
        
        # Dipolar aprotic
        "DMF": {"polarity": "high", "category": "aprotic_polar"},
        "DMSO": {"polarity": "high", "category": "aprotic_polar"},
        "NMP": {"polarity": "high", "category": "aprotic_polar"},
        "Acetonitrile": {"polarity": "high", "category": "aprotic_polar"},
        
        # Others
        "Acetic_acid": {"polarity": "high", "category": "protic"},
        "Pyridine": {"polarity": "medium", "category": "aprotic_polar"},
        "Nitromethane": {"polarity": "high", "category": "aprotic_polar"},
        "Formic_acid": {"polarity": "high", "category": "protic"},
    }


def generate_solubility_data(polymers_df, seed=42):
    """
    Generate realistic solubility data for polymers in 30 solvents.
    
    Returns:
        DataFrame with columns: polymer, solvent, solubility (g/L), temperature (C)
    """
    np.random.seed(seed)
    solvents = list(get_common_solvents().keys())
    
    # Define solubility rules based on polymer chemistry
    rules = {
        # Water-soluble polymers
        "PVA": {"Water": (50, 80), "Methanol": (10, 30), "Ethanol": (5, 15)},
        "PEO": {"Water": (100, 200), "Chloroform": (50, 100), "Acetone": (20, 50)},
        "PVP": {"Water": (100, 300), "Ethanol": (50, 150), "Chloroform": (30, 80)},
        "PAA": {"Water": (30, 100), "Methanol": (10, 40)},
        "PAAm": {"Water": (50, 150), "Methanol": (15, 50)},
        "PEI": {"Water": (40, 120), "Methanol": (20, 60), "Ethanol": (10, 30)},
        
        # Organic-soluble polymers
        "PMMA": {"Acetone": (50, 150), "Chloroform": (100, 250), "THF": (80, 200), "Toluene": (30, 80)},
        "PS": {"Toluene": (100, 250), "Chloroform": (80, 200), "THF": (60, 150), "Benzene": (90, 220)},
        "PAN": {"DMF": (50, 150), "DMSO": (40, 120), "NMP": (45, 135)},
        "Nitrocellulose": {"Acetone": (100, 250), "Ethyl_acetate": (80, 200), "MEK": (70, 180)},
        "PVAc": {"Acetone": (80, 200), "Ethanol": (40, 100), "Ethyl_acetate": (60, 150)},
        "PVB": {"Ethanol": (50, 150), "Isopropanol": (40, 120), "Butanol": (30, 90)},
        "CA": {"Acetone": (100, 250), "Ethyl_acetate": (70, 180), "Acetic_acid": (50, 130)},
        "PLA": {"Chloroform": (100, 250), "DCM": (90, 220), "Dioxane": (60, 150)},
        "PDMS": {"Toluene": (80, 200), "Hexane": (60, 150), "Chloroform": (70, 180)},
        
        # Special cases
        "Polysulfone": {"NMP": (40, 100), "DMF": (35, 90), "DMSO": (30, 80)},
        "PU-ester": {"DMF": (40, 100), "THF": (30, 80), "MEK": (25, 70)},
    }
    
    data = []
    for _, row in polymers_df.iterrows():
        polymer = row['polymer']
        
        # Get polymer-specific solubility ranges
        poly_rules = rules.get(polymer, {})
        
        for solvent in solvents:
            if solvent in poly_rules:
                # Good solvent - high solubility
                min_sol, max_sol = poly_rules[solvent]
                solubility = np.random.uniform(min_sol, max_sol)
            else:
                # Poor solvent - low solubility with occasional moderate
                if np.random.random() < 0.1:  # 10% chance of moderate solubility
                    solubility = np.random.uniform(5, 20)
                else:
                    solubility = np.random.uniform(0.01, 2)  # Very low
            
            # Temperature (most measurements at room temp, some elevated)
            temp = np.random.choice([25, 50, 80], p=[0.7, 0.2, 0.1])
            
            data.append({
                "polymer": polymer,
                "solvent": solvent,
                "solubility_g_L": solubility,
                "temperature_C": temp
            })
    
    return pd.DataFrame(data)


# ==========================================
# 2) SOLUBILITY METRICS
# ==========================================

def compute_solubility_scores(solubility_df):
    """
    Compute normalized solubility scores for each polymer.
    
    Multiple metrics:
    - mean_solubility: Average across all solvents
    - max_solubility: Best solvent
    - soluble_fraction: Fraction of solvents with solubility > 10 g/L
    - versatility_score: Combination of breadth and magnitude
    """
    scores = []
    
    for polymer in solubility_df['polymer'].unique():
        poly_data = solubility_df[solubility_df['polymer'] == polymer]
        sols = poly_data['solubility_g_L'].values
        
        mean_sol = np.mean(sols)
        max_sol = np.max(sols)
        soluble_frac = np.mean(sols > 10)  # Fraction with >10 g/L
        
        # Versatility: combines breadth (how many solvents) and magnitude
        # Uses log scale to handle wide range
        log_sols = np.log10(sols + 1)
        versatility = np.mean(log_sols) * (1 + soluble_frac)
        
        scores.append({
            "polymer": polymer,
            "mean_solubility": mean_sol,
            "max_solubility": max_sol,
            "soluble_fraction": soluble_frac,
            "versatility_score": versatility,
            "n_solvents_tested": len(sols)
        })
    
    scores_df = pd.DataFrame(scores)
    
    # Normalize scores to [0, 1] for easier interpretation
    for col in ["mean_solubility", "max_solubility", "versatility_score"]:
        scores_df[f"{col}_norm"] = (
            (scores_df[col] - scores_df[col].min()) / 
            (scores_df[col].max() - scores_df[col].min())
        )
    
    return scores_df


# ==========================================
# 3) FEATURE GENERATION
# ==========================================

def smiles_to_morgan(smiles, radius=2, nbits=512):
    """Convert SMILES to Morgan fingerprint."""
    gen = GetMorganGenerator(radius=radius, fpSize=nbits)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((nbits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize_polymers(polymers_df, use_pca=True, n_components=50):
    """
    Convert polymers to feature vectors.
    
    Args:
        polymers_df: DataFrame with 'polymer' and 'smiles'
        use_pca: Whether to apply PCA dimensionality reduction
        n_components: Number of PCA components
    
    Returns:
        X: Feature matrix (n_polymers, n_features)
        feature_names: List of polymer names
        scaler: StandardScaler object
        pca: PCA object (or None)
    """
    fps = []
    names = []
    
    for _, row in polymers_df.iterrows():
        fp = smiles_to_morgan(row['smiles'])
        if fp is not None:
            fps.append(fp)
            names.append(row['polymer'])
    
    X = np.array(fps)
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Optional PCA
    pca = None
    if use_pca:
        n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        X = pca.fit_transform(X)
        print(f"[INFO] PCA: {X.shape[1]} components, {pca.explained_variance_ratio_.sum()*100:.1f}% variance")
    
    return X, names, scaler, pca


# ==========================================
# 4) BAYESIAN OPTIMIZATION
# ==========================================

class PolymerActiveLearner:
    """
    Bayesian Optimization for Polymer Discovery using Gaussian Processes.
    """
    
    def __init__(self, X_pool, polymer_names, metric="versatility_score_norm"):
        """
        Args:
            X_pool: Feature matrix of all candidate polymers
            polymer_names: List of polymer names
            metric: Which solubility metric to optimize
        """
        self.X_pool = X_pool
        self.polymer_names = np.array(polymer_names)
        self.metric = metric
        
        # Training data (initially empty)
        self.X_train = np.empty((0, X_pool.shape[1]))
        self.y_train = np.empty(0)
        self.tested_indices = []
        self.tested_polymers = []
        
        # Gaussian Process model
        kernel = C(1.0, (1e-3, 1e3)) * Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(1e-5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True,
            random_state=42
        )
        
    def add_observation(self, polymer_idx, score):
        """Add a new observation to the training set."""
        self.X_train = np.vstack([self.X_train, self.X_pool[polymer_idx]])
        self.y_train = np.append(self.y_train, score)
        self.tested_indices.append(polymer_idx)
        self.tested_polymers.append(self.polymer_names[polymer_idx])
        
    def fit(self):
        """Fit the Gaussian Process to current observations."""
        if len(self.y_train) == 0:
            raise ValueError("No observations yet. Add at least one.")
        self.gp.fit(self.X_train, self.y_train)
        
    def predict(self, X, return_std=True):
        """Predict mean and std for given features."""
        if len(self.y_train) == 0:
            # No data yet - return uniform predictions
            mean = np.ones(len(X)) * 0.5
            std = np.ones(len(X)) * 0.5
            return (mean, std) if return_std else mean
        
        return self.gp.predict(X, return_std=return_std)
    
    def acquisition_ei(self, X, xi=0.01):
        """
        Expected Improvement acquisition function.
        
        Args:
            X: Feature matrix
            xi: Exploration-exploitation trade-off (higher = more exploration)
        """
        mu, sigma = self.predict(X, return_std=True)
        
        if len(self.y_train) == 0:
            return sigma  # Pure exploration
        
        mu_best = np.max(self.y_train)
        
        with np.errstate(divide='warn'):
            imp = mu - mu_best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def acquisition_ucb(self, X, kappa=2.0):
        """
        Upper Confidence Bound acquisition function.
        
        Args:
            X: Feature matrix
            kappa: Exploration parameter (higher = more exploration)
        """
        mu, sigma = self.predict(X, return_std=True)
        return mu + kappa * sigma
    
    def acquisition_pi(self, X, xi=0.01):
        """
        Probability of Improvement acquisition function.
        """
        mu, sigma = self.predict(X, return_std=True)
        
        if len(self.y_train) == 0:
            return np.ones(len(X)) * 0.5
        
        mu_best = np.max(self.y_train)
        
        with np.errstate(divide='warn'):
            Z = (mu - mu_best - xi) / sigma
            pi = norm.cdf(Z)
            pi[sigma == 0.0] = 0.0
        
        return pi
    
    def acquisition_greedy(self, X):
        """Greedy acquisition: just pick highest predicted mean."""
        mu, _ = self.predict(X, return_std=True)
        return mu
    
    def suggest_next(self, method="ei", n_suggestions=5, **kwargs):
        """
        Suggest next polymers to test.
        
        Args:
            method: 'ei', 'ucb', 'pi', 'greedy'
            n_suggestions: Number of suggestions to return
            **kwargs: Parameters for acquisition function (xi, kappa, etc.)
        
        Returns:
            List of (polymer_idx, polymer_name, acquisition_value) tuples
        """
        # Get untested polymers
        untested_mask = np.ones(len(self.X_pool), dtype=bool)
        untested_mask[self.tested_indices] = False
        untested_indices = np.where(untested_mask)[0]
        
        if len(untested_indices) == 0:
            print("[WARN] All polymers have been tested!")
            return []
        
        X_untested = self.X_pool[untested_indices]
        
        # Compute acquisition function
        if method == "ei":
            acq_values = self.acquisition_ei(X_untested, **kwargs)
        elif method == "ucb":
            acq_values = self.acquisition_ucb(X_untested, **kwargs)
        elif method == "pi":
            acq_values = self.acquisition_pi(X_untested, **kwargs)
        elif method == "greedy":
            acq_values = self.acquisition_greedy(X_untested)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Get top suggestions
        top_k = min(n_suggestions, len(untested_indices))
        top_acq_idx = np.argsort(acq_values)[-top_k:][::-1]
        
        suggestions = []
        for i in top_acq_idx:
            poly_idx = untested_indices[i]
            suggestions.append((
                poly_idx,
                self.polymer_names[poly_idx],
                acq_values[i]
            ))
        
        return suggestions


# ==========================================
# 5) VISUALIZATION
# ==========================================

def plot_gp_predictions(learner, X_pool, polymer_names, scores_df, metric, iteration, outdir):
    """Plot GP predictions with uncertainty."""
    polymer_names = np.array(polymer_names)  # Convert to numpy array for boolean indexing
    mu, sigma = learner.predict(X_pool, return_std=True)
    
    # Get true scores (for comparison)
    true_scores = []
    for name in polymer_names:
        score_row = scores_df[scores_df['polymer'] == name]
        if len(score_row) > 0:
            true_scores.append(score_row[metric].values[0])
        else:
            true_scores.append(np.nan)
    true_scores = np.array(true_scores)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1) Predictions vs True
    ax = axes[0, 0]
    tested_mask = np.zeros(len(polymer_names), dtype=bool)
    tested_mask[learner.tested_indices] = True
    
    ax.scatter(true_scores[~tested_mask], mu[~tested_mask], alpha=0.6, label='Untested', s=80)
    ax.scatter(true_scores[tested_mask], mu[tested_mask], alpha=0.8, label='Tested', s=100, marker='s')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('True Score', fontsize=11)
    ax.set_ylabel('Predicted Score', fontsize=11)
    ax.set_title(f'GP Predictions (Iteration {iteration})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2) Uncertainty (std)
    ax = axes[0, 1]
    sorted_idx = np.argsort(sigma)
    colors = ['orange' if i in learner.tested_indices else 'steelblue' for i in sorted_idx]
    ax.barh(range(len(sigma)), sigma[sorted_idx], color=colors, alpha=0.7)
    ax.set_xlabel('Prediction Uncertainty (σ)', fontsize=11)
    ax.set_ylabel('Polymer Index', fontsize=11)
    ax.set_title('Model Uncertainty', fontsize=12, fontweight='bold')
    ax.set_yticks([])
    
    # 3) Predicted scores bar chart
    ax = axes[1, 0]
    sorted_idx = np.argsort(mu)[-15:]  # Top 15
    names_sorted = [polymer_names[i] for i in sorted_idx]
    colors = ['orange' if i in learner.tested_indices else 'steelblue' for i in sorted_idx]
    ax.barh(range(len(sorted_idx)), mu[sorted_idx], color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.set_xlabel('Predicted Score', fontsize=11)
    ax.set_title('Top 15 Predicted Polymers', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4) Exploration-Exploitation
    ax = axes[1, 1]
    ax.errorbar(range(len(mu)), mu, yerr=sigma, fmt='o', alpha=0.4, capsize=3, elinewidth=1)
    tested_idx = np.array(learner.tested_indices)
    ax.scatter(tested_idx, mu[tested_idx], c='red', s=120, marker='*', label='Tested', zorder=10)
    ax.set_xlabel('Polymer Index', fontsize=11)
    ax.set_ylabel('Score (μ ± σ)', fontsize=11)
    ax.set_title('Exploration-Exploitation Landscape', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    outpath = outdir / f"gp_predictions_iter{iteration:02d}.png"
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {outpath}")


def plot_acquisition_comparison(learner, X_pool, polymer_names, iteration, outdir):
    """Compare different acquisition functions."""
    polymer_names = np.array(polymer_names)  # Convert to numpy array for boolean indexing
    untested_mask = np.ones(len(X_pool), dtype=bool)
    untested_mask[learner.tested_indices] = False
    X_untested = X_pool[untested_mask]
    names_untested = polymer_names[untested_mask]
    
    if len(X_untested) == 0:
        return
    
    # Compute all acquisition functions
    acq_ei = learner.acquisition_ei(X_untested, xi=0.01)
    acq_ucb = learner.acquisition_ucb(X_untested, kappa=2.0)
    acq_pi = learner.acquisition_pi(X_untested, xi=0.01)
    mu, sigma = learner.predict(X_untested, return_std=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = [
        ("Expected Improvement", acq_ei),
        ("Upper Confidence Bound", acq_ucb),
        ("Probability of Improvement", acq_pi),
        ("Greedy (Mean)", mu)
    ]
    
    for ax, (name, values) in zip(axes.flat, methods):
        top_10_idx = np.argsort(values)[-10:][::-1]
        top_names = [names_untested[i] for i in top_10_idx]
        top_values = values[top_10_idx]
        
        ax.barh(range(len(top_values)), top_values, color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(top_values)))
        ax.set_yticklabels(top_names, fontsize=9)
        ax.set_xlabel('Acquisition Value', fontsize=10)
        ax.set_title(f'{name} - Top 10', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
    
    plt.tight_layout()
    outpath = outdir / f"acquisition_comparison_iter{iteration:02d}.png"
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {outpath}")


# ==========================================
# 6) ACTIVE LEARNING LOOP
# ==========================================

def run_active_learning_campaign(
    n_initial=3,
    n_iterations=10,
    n_suggestions=5,
    acquisition_method="ei",
    metric="versatility_score_norm",
    seed=42
):
    """
    Run a full active learning campaign.
    
    Args:
        n_initial: Number of random initial experiments
        n_iterations: Number of AL iterations
        n_suggestions: Number of suggestions per iteration
        acquisition_method: 'ei', 'ucb', 'pi', 'greedy'
        metric: Which solubility metric to optimize
        seed: Random seed
    """
    np.random.seed(seed)
    outdir = Path("active_learning_output")
    outdir.mkdir(exist_ok=True)
    
    print("="*70)
    print("BAYESIAN OPTIMIZATION FOR POLYMER SOLUBILITY DISCOVERY")
    print("="*70)
    
    # 1) Build polymer database
    print("\n[1/6] Building polymer database...")
    polymers_df = build_polymer_database()
    print(f"      Total polymers: {len(polymers_df)}")
    
    # 2) Generate solubility data (this would be your real measurements)
    print("\n[2/6] Generating solubility data...")
    solubility_df = generate_solubility_data(polymers_df, seed=seed)
    solvents = list(get_common_solvents().keys())
    print(f"      Solvents: {len(solvents)}")
    print(f"      Total data points: {len(solubility_df)}")
    
    # 3) Compute solubility scores
    print("\n[3/6] Computing solubility scores...")
    scores_df = compute_solubility_scores(solubility_df)
    print(f"      Metrics: {[c for c in scores_df.columns if 'norm' in c]}")
    print(f"      Optimizing: {metric}")
    
    # 4) Featurize polymers
    print("\n[4/6] Featurizing polymers...")
    X_pool, polymer_names, scaler, pca = featurize_polymers(polymers_df, use_pca=True, n_components=50)
    print(f"      Feature dimensions: {X_pool.shape}")
    
    # 5) Initialize learner
    print("\n[5/6] Initializing Bayesian optimizer...")
    learner = PolymerActiveLearner(X_pool, polymer_names, metric=metric)
    
    # Initial random experiments
    print(f"\n[6/6] Starting active learning campaign...")
    print(f"\n--- Initial Random Sampling ({n_initial} polymers) ---")
    available_indices = list(range(len(polymer_names)))
    initial_indices = np.random.choice(available_indices, size=n_initial, replace=False)
    
    for idx in initial_indices:
        polymer_name = polymer_names[idx]
        score = scores_df[scores_df['polymer'] == polymer_name][metric].values[0]
        learner.add_observation(idx, score)
        print(f"  ✓ {polymer_name}: {metric} = {score:.3f}")
    
    # Fit initial model
    learner.fit()
    print(f"\n  Best so far: {polymer_names[learner.tested_indices[np.argmax(learner.y_train)]]} ({np.max(learner.y_train):.3f})")
    
    # Save initial state
    plot_gp_predictions(learner, X_pool, polymer_names, scores_df, metric, 0, outdir)
    
    # Active learning iterations
    history = []
    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration}/{n_iterations} ---")
        
        # Get suggestions
        suggestions = learner.suggest_next(method=acquisition_method, n_suggestions=n_suggestions)
        
        if not suggestions:
            print("  All polymers tested!")
            break
        
        print(f"  Acquisition function: {acquisition_method.upper()}")
        print(f"  Top {len(suggestions)} suggestions:")
        for i, (idx, name, acq_val) in enumerate(suggestions, 1):
            print(f"    {i}. {name} (acq={acq_val:.4f})")
        
        # "Test" the top suggestion (in real scenario, this would be an experiment)
        test_idx, test_name, _ = suggestions[0]
        true_score = scores_df[scores_df['polymer'] == test_name][metric].values[0]
        
        # Add observation and refit
        learner.add_observation(test_idx, true_score)
        learner.fit()
        
        # Track history
        best_idx = learner.tested_indices[np.argmax(learner.y_train)]
        history.append({
            "iteration": iteration,
            "tested_polymer": test_name,
            "tested_score": true_score,
            "best_polymer": polymer_names[best_idx],
            "best_score": np.max(learner.y_train),
            "n_tested": len(learner.tested_indices)
        })
        
        print(f"\n  ✓ Tested: {test_name} → {metric} = {true_score:.3f}")
        print(f"  Current best: {polymer_names[best_idx]} ({np.max(learner.y_train):.3f})")
        
        # Visualizations every few iterations
        if iteration % 2 == 0 or iteration == n_iterations:
            plot_gp_predictions(learner, X_pool, polymer_names, scores_df, metric, iteration, outdir)
            plot_acquisition_comparison(learner, X_pool, polymer_names, iteration, outdir)
    
    # Final summary
    print("\n" + "="*70)
    print("CAMPAIGN SUMMARY")
    print("="*70)
    print(f"Total polymers tested: {len(learner.tested_indices)}/{len(polymer_names)}")
    print(f"Best polymer found: {polymer_names[learner.tested_indices[np.argmax(learner.y_train)]]}")
    print(f"Best score: {np.max(learner.y_train):.3f}")
    
    # True best for comparison
    true_best_idx = scores_df[metric].idxmax()
    true_best = scores_df.loc[true_best_idx]
    print(f"\nTrue best (if all tested): {true_best['polymer']} ({true_best[metric]:.3f})")
    
    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(outdir / "campaign_history.csv", index=False)
    print(f"\n[OK] Saved campaign history: {outdir / 'campaign_history.csv'}")
    
    # Plot learning curve
    plot_learning_curve(history_df, true_best[metric], outdir)
    
    return learner, scores_df, history_df


def plot_learning_curve(history_df, true_best_score, outdir):
    """Plot how the best found score improves over iterations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = history_df['iteration'].values
    best_scores = history_df['best_score'].values
    
    ax.plot(iterations, best_scores, 'o-', linewidth=2, markersize=8, label='Best score found')
    ax.axhline(true_best_score, color='red', linestyle='--', linewidth=2, label='True optimum', alpha=0.7)
    
    # Shade the gap
    ax.fill_between(iterations, best_scores, true_best_score, alpha=0.2, color='gray')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Score Found', fontsize=12)
    ax.set_title('Active Learning Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    outpath = outdir / "learning_curve.png"
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {outpath}")


# ==========================================
# 7) MAIN
# ==========================================

def main():
    """Run the active learning campaign."""
    learner, scores_df, history = run_active_learning_campaign(
        n_initial=3,           # Start with 3 random polymers
        n_iterations=10,       # Run 10 AL iterations
        n_suggestions=5,       # Show top 5 suggestions each time
        acquisition_method="ei",  # Expected Improvement
        metric="versatility_score_norm",  # Optimize versatility
        seed=42
    )
    
    print("\n" + "="*70)
    print("DONE! Check 'active_learning_output/' for results")
    print("="*70)


if __name__ == "__main__":
    main()
