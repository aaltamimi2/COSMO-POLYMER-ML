
import time
import numpy as np
import pandas as pd

# Set matplotlib to non-interactive backend (prevents window popups)
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False
    print("[WARN] adjustText not available - labels may overlap. Install: pip install adjustText")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("[WARN] plotly not available - skipping interactive plots. Install: pip install plotly")


# -----------------------------
# 1) Polymers + SMILES proxies
# -----------------------------
def build_polymer_table() -> pd.DataFrame:
    """
    Build polymer table with accurate SMILES and industrial applications.
    Note: SMILES represent repeat units. For copolymers, we use characteristic monomer.
    """
    # Base polymers (mostly insoluble commodity plastics)
    base = {
        "HDPE": "CC",  # High-density polyethylene
        "LDPE": "CC",  # Low-density polyethylene (branched)
        "PP": "CC(C)",  # Polypropylene
        "PS": "CC(c1ccccc1)",  # Polystyrene
        "PVC": "CC(Cl)",  # Polyvinyl chloride
        "PC": "CC(C)(c1ccc(OC(=O)Oc2ccc(C(C)(C)c3ccc(O)cc3)cc2)cc1)",  # IMPROVED: Bisphenol-A polycarbonate
        "PET": "O=C(c1ccc(C(=O)OCCO)cc1)",  # IMPROVED: Proper terephthalate
        "PETG": "O=C(c1ccc(C(=O)OCC(O)CO)cc1)",  # IMPROVED: CHDM-modified PET
        "PVDF": "CC(F)F",  # FIXED: Polyvinylidene fluoride repeat unit
        "Nylon6": "NCCCCCC(=O)",  # Nylon 6 (caprolactam)
        "Nylon66": "NCCCCCCNC(=O)CCCCC(=O)",  # IMPROVED: Proper Nylon 6,6
        "Polysulfone": "O=S(=O)(c1ccc(Oc2ccc(C(C)(C)c3ccc(O)cc3)cc2)cc1)c4ccc(O)cc4",  # PSU
        "Polyurethane": "CCCCCCNC(=O)OCCCCOC(=O)N",  # IMPROVED: Typical aliphatic PU
        "EVOH": "CC(O)",  # Ethylene-vinyl alcohol copolymer (VA unit)
    }

    # Commonly dissolved polymers - ORIGINAL 5
    dissolved_original = {
        "PMMA": "CC(C)(C(=O)OC)",  # Poly(methyl methacrylate) - acetone, CHCl3, THF
        "PVA": "CC(O)",  # Polyvinyl alcohol - water (hot)
        "PAN": "CC(C#N)",  # Polyacrylonitrile - DMF, DMSO
        "PVP": "CC(N1CCCC1=O)",  # FIXED: Polyvinylpyrrolidone - water, alcohols
        "Nitrocellulose": "C([C@@H]1[C@H]([C@@H]([C@H]([C@@H](O1)O[C@@H]2[C@H](O[C@H]([C@@H]([C@H]2O[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-])CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-])O[C@H]3[C@@H]([C@H]([C@@H]([C@H](O3)CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]",  # Acetone, esters
    }
    
    # NEW: Additional industrially important dissolved polymers
    dissolved_new = {
        "PVB": "CC(O)CC(c1ccccc1)",  # Polyvinyl butyral - alcohols (laminated glass)
        "PEO/PEG": "OCCO",  # Polyethylene oxide/glycol - water, CHCl3
        "PLA": "CC(O)C(=O)",  # Polylactic acid - CHCl3, DCM (biodegradable)
        "CA": "CC(=O)OC[C@H]1O[C@H](OC(C)=O)[C@H](OC(C)=O)[C@@H](OC(C)=O)[C@@H]1OC(C)=O",  # Cellulose acetate - acetone
        "PAA": "CC(C(=O)O)",  # Polyacrylic acid - water (pH>4)
        "PAAm": "CC(C(=O)N)",  # Polyacrylamide - water (flocculants, gels)
        "PEI": "NCCN",  # Polyethyleneimine - water, alcohols (paper, adhesives)
        "PVAc": "CC(OC(=O)C)",  # Polyvinyl acetate - acetone, alcohols (wood glue)
        "PDMS": "C[Si](C)(C)O[Si](C)(C)C",  # IMPROVED: Polydimethylsiloxane - toluene, hexane
        "PU-ester": "CCCCOC(=O)NCCCCCCNC(=O)OCCCCOC(=O)N",  # Polyurethane (ester-based) - DMF, THF
    }

    rows = []
    for k, v in base.items():
        rows.append({"polymer": k, "smiles": v, "group": "base"})
    for k, v in dissolved_original.items():
        rows.append({"polymer": k, "smiles": v, "group": "dissolved_orig"})
    for k, v in dissolved_new.items():
        rows.append({"polymer": k, "smiles": v, "group": "dissolved_new"})

    df = pd.DataFrame(rows).sort_values(["group", "polymer"]).reset_index(drop=True)
    return df


# -----------------------------
# 2) SMILES -> Morgan fingerprints (optimized)
# -----------------------------
def make_morgan_generator(radius: int, nbits: int):
    """Create Morgan fingerprint generator."""
    return GetMorganGenerator(radius=radius, fpSize=nbits)


def smiles_to_morgan_bits(smiles: str, gen, nbits: int) -> np.ndarray:
    """Convert SMILES to Morgan fingerprint bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles}")
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize(df: pd.DataFrame, radius: int = 2, nbits: int = 512):
    """
    Convert SMILES to fingerprints.
    
    NOTE: Reduced from 2048 to 512 bits - sufficient for ~20 polymers
    and much faster for UMAP.
    """
    gen = make_morgan_generator(radius=radius, nbits=nbits)

    rows, fps = [], []
    for _, r in df.iterrows():
        smi = r["smiles"]
        if smi is None or (isinstance(smi, float) and np.isnan(smi)) or str(smi).strip() == "":
            continue
        try:
            fp = smiles_to_morgan_bits(str(smi), gen=gen, nbits=nbits)
            fps.append(fp)
            rows.append(r)
        except Exception as e:
            print(f"[WARN] Skipping {r['polymer']}: {e}")

    if not fps:
        raise RuntimeError("No valid SMILES found.")

    df_ok = pd.DataFrame(rows).reset_index(drop=True)
    X = np.vstack(fps)
    return df_ok, X


# -----------------------------
# 3) Improved plotting
# -----------------------------
def plot_embedding_improved(Z, df_ok, title, xlabel, ylabel, outpath=None, var_explained=None):
    """
    Plot with better styling and label positioning.
    Now handles 3 groups: base, dissolved_orig, dissolved_new
    """
    colors = {
        "base": "#1f77b4",           # Blue - insoluble base polymers
        "dissolved_orig": "#ff7f0e",  # Orange - original 5 dissolved
        "dissolved_new": "#2ca02c"    # Green - new 10 dissolved
    }
    labels = {
        "base": "Insoluble (base)",
        "dissolved_orig": "Dissolved (original 5)",
        "dissolved_new": "Dissolved (new 10)"
    }
    
    fig, ax = plt.subplots(figsize=(12, 9))

    texts = []
    for grp in ["base", "dissolved_orig", "dissolved_new"]:
        mask = (df_ok["group"].values == grp)
        if mask.sum() == 0:
            continue
        
        # Plot points with different markers
        marker = 'o' if grp == "base" else 's' if grp == "dissolved_orig" else '^'
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            s=150, alpha=0.7,
            label=labels[grp],
            c=colors[grp],
            edgecolors='white',
            linewidths=1.5,
            zorder=2,
            marker=marker
        )

        # Collect text labels for adjustment
        for i in np.where(mask)[0]:
            txt = ax.text(
                Z[i, 0], Z[i, 1], 
                df_ok.loc[i, "polymer"], 
                fontsize=8, 
                color=colors[grp],
                fontweight='bold',
                zorder=3
            )
            texts.append(txt)

    # Adjust text positions to avoid overlap
    if HAS_ADJUSTTEXT and len(texts) > 0:
        adjust_text(
            texts, 
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5),
            ax=ax
        )

    # Title with variance if available
    if var_explained is not None:
        title = f"{title}\n({var_explained[0]:.1f}% and {var_explained[1]:.1f}% variance explained)"
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(frameon=True, loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {outpath}")
    
    plt.close()  # Close without displaying


def plot_interactive_plotly(Z, df_ok, title, xlabel, ylabel, outpath=None):
    """Create interactive Plotly version with 3 groups."""
    if not HAS_PLOTLY:
        return
    
    colors = {
        "base": "#1f77b4",
        "dissolved_orig": "#ff7f0e",
        "dissolved_new": "#2ca02c"
    }
    labels = {
        "base": "Insoluble (base)",
        "dissolved_orig": "Dissolved (original 5)",
        "dissolved_new": "Dissolved (new 10)"
    }
    markers = {
        "base": "circle",
        "dissolved_orig": "square",
        "dissolved_new": "diamond"
    }
    
    fig = go.Figure()
    
    for grp in ["base", "dissolved_orig", "dissolved_new"]:
        mask = (df_ok["group"].values == grp)
        if mask.sum() == 0:
            continue
        
        df_grp = df_ok[mask].reset_index(drop=True)
        
        fig.add_trace(go.Scatter(
            x=Z[mask, 0],
            y=Z[mask, 1],
            mode='markers+text',
            name=labels[grp],
            text=df_grp['polymer'],
            textposition='top center',
            textfont=dict(size=9, color=colors[grp]),
            marker=dict(
                size=14,
                color=colors[grp],
                symbol=markers[grp],
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>' + 
                         f'{xlabel}: %{{x:.3f}}<br>' +
                         f'{ylabel}: %{{y:.3f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        hovermode='closest',
        template='plotly_white',
        width=900,
        height=700,
        showlegend=True
    )
    
    if outpath:
        fig.write_html(outpath)
        print(f"[OK] Saved interactive: {outpath}")


# -----------------------------
# 4) Main pipeline
# -----------------------------
def main():
    print("="*60)
    print("POLYMER CHEMICAL SPACE VISUALIZATION - IMPROVED VERSION")
    print("="*60)
    
    df = build_polymer_table()
    n_base = (df['group']=='base').sum()
    n_orig = (df['group']=='dissolved_orig').sum()
    n_new = (df['group']=='dissolved_new').sum()
    print(f"\n[INFO] Total polymers: {len(df)} ({n_base} base + {n_orig} dissolved_orig + {n_new} dissolved_new)")

    # Featurize with smaller fingerprints (512 vs 2048)
    t0 = time.time()
    df_ok, X = featurize(df, radius=2, nbits=512)
    print(f"[TIMING] Featurization: {time.time() - t0:.3f} sec")
    print(f"[INFO] Feature matrix shape: {X.shape}")

    # -----------------------------
    # PCA (with variance explained)
    # -----------------------------
    print("\n--- Running PCA ---")
    t0 = time.time()
    pca = PCA(n_components=2, random_state=0)
    Z_pca = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100
    print(f"[TIMING] PCA: {time.time() - t0:.3f} sec")
    print(f"[INFO] Variance explained: PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%")

    plot_embedding_improved(
        Z_pca, df_ok,
        title="PCA Chemical Space (Morgan FP, r=2, 512 bits)",
        xlabel="PC1", ylabel="PC2",
        outpath="polymer_space_pca.png",
        var_explained=var_exp
    )

    # -----------------------------
    # UMAP (OPTIMIZED)
    # -----------------------------
    print("\n--- Running UMAP (optimized) ---")
    
    # OPTIMIZATION 1: Pre-reduce with PCA (standard practice)
    # This is the KEY speedup for high-dimensional data
    pca_pre = PCA(n_components=min(50, X.shape[0]-1, X.shape[1]), random_state=0)
    X_pca = pca_pre.fit_transform(X)
    print(f"[INFO] Pre-reduced to {X_pca.shape[1]} dims with PCA")
    
    # OPTIMIZATION 2: Use cosine metric (faster than Jaccard for normalized data)
    # OPTIMIZATION 3: Reduced n_epochs (50 is plenty for exploration)
    um = umap.UMAP(
        n_neighbors=min(10, len(df_ok) - 1),
        min_dist=0.1,
        metric='cosine',        # Faster than Jaccard
        n_epochs=50,            # Reduced from 100
        n_components=2,
        low_memory=True,
        random_state=0,
        verbose=False
    )

    t0 = time.time()
    Z_umap = um.fit_transform(X_pca)  # Use PCA-reduced data
    umap_time = time.time() - t0
    print(f"[TIMING] UMAP fit_transform: {umap_time:.3f} sec")
    
    if umap_time > 5:
        print("[WARN] UMAP still slow? Check numba installation and first-run JIT compilation")

    plot_embedding_improved(
        Z_umap, df_ok,
        title="UMAP Chemical Space (cosine, PCA-pre-reduced)",
        xlabel="UMAP-1", ylabel="UMAP-2",
        outpath="polymer_space_umap.png"
    )

    # Interactive version
    if HAS_PLOTLY:
        print("\n--- Creating interactive plots ---")
        plot_interactive_plotly(
            Z_umap, df_ok,
            title="UMAP Chemical Space (Interactive)",
            xlabel="UMAP-1", ylabel="UMAP-2",
            outpath="polymer_space_umap_interactive.html"
        )

    # Save embeddings
    out = df_ok.copy()
    out["pca1"], out["pca2"] = Z_pca[:, 0], Z_pca[:, 1]
    out["umap1"], out["umap2"] = Z_umap[:, 0], Z_umap[:, 1]
    out.to_csv("polymer_embeddings.csv", index=False)
    print("\n[OK] Saved: polymer_embeddings.csv")
    print("="*60)


if __name__ == "__main__":
    main()