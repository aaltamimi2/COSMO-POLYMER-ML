"""
Modeling utilities for the common-solvents CSVs (solubility vs temperature).

This module follows the workflow outlined in the modeling plan:
- Point prediction of solubility as a function of (polymer, solvent, temperature)
- Grouped splits that hold out entire polymers
- A flexible feature stack (descriptors + one-hot fallbacks) that supports
  temperature smoothing and active learning via deep ensembles.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_BASE_FEATURES = [
    "temperature_c",
    "temperature_k",
    "inv_temperature_k",
    "temperature_c_squared",
    "temperature_c_cubed",
    "log_temperature_k",
]


@dataclass
class ModelResult:
    """Container for cross-validation metrics."""

    model_name: str
    mae_mean: float
    mae_std: float
    rmse_mean: float
    rmse_std: float
    r2_mean: float
    r2_std: float
    smoothness_mean: float
    smoothness_std: float


def _normalize_polymer_name(csv_path: Path) -> str:
    """Infer polymer name from filename such as COMMON-solvents-PVC.csv."""
    stem = csv_path.stem.replace("COMMON-solvents-", "")
    return stem.lower()


def load_common_solvent_csv(
    csv_path: Path,
    polymer: Optional[str] = None,
    anchor_temperatures: Sequence[float] = (25.0, 50.0, 75.0),
) -> pd.DataFrame:
    """
    Load a single polymer-solvent CSV and append engineered columns.

    Parameters
    ----------
    csv_path : Path
        Path to a COMMON-solvents-*.csv file.
    polymer : str, optional
        Polymer name override. If None, the name is inferred from the filename.
    anchor_temperatures : Sequence[float]
        Anchor temperatures used for curve-regularization features.

    Returns
    -------
    pd.DataFrame
        Standardized dataframe with columns:
        [polymer, solvent, temperature_c, solubility_pct, solubility_fraction,
         log_solubility, temperature_k, inv_temperature_k, ... anchor deltas ...]
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    col_map = {
        "Solvent": "solvent",
        "Temperature (°C)": "temperature_c",
        "Solubility (%)": "solubility_pct",
    }
    df = df.rename(columns=col_map)

    if polymer is None:
        polymer = _normalize_polymer_name(csv_path)

    df["polymer"] = polymer
    df["temperature_k"] = df["temperature_c"] + 273.15
    df["inv_temperature_k"] = 1.0 / df["temperature_k"]
    df["temperature_c_squared"] = df["temperature_c"] ** 2
    df["temperature_c_cubed"] = df["temperature_c"] ** 3
    df["log_temperature_k"] = np.log(df["temperature_k"])

    for anchor in anchor_temperatures:
        delta = df["temperature_c"] - anchor
        df[f"delta_from_{anchor:.0f}C"] = delta
        df[f"abs_delta_from_{anchor:.0f}C"] = delta.abs()

    df["solubility_fraction"] = df["solubility_pct"] / 100.0
    df["log_solubility"] = np.log(df["solubility_fraction"].clip(lower=1e-6))

    return df


def load_common_solvent_directory(
    data_dir: Path, anchor_temperatures: Sequence[float] = (25.0, 50.0, 75.0)
) -> pd.DataFrame:
    """
    Load and concatenate all COMMON-solvents CSVs.

    Parameters
    ----------
    data_dir : Path
        Directory containing COMMON-solvents-*.csv files.
    anchor_temperatures : Sequence[float]
        Anchor temperatures used for curve-regularization features.

    Returns
    -------
    pd.DataFrame
        Combined dataframe across polymers.
    """
    frames = []
    for csv_path in sorted(data_dir.glob("COMMON-solvents-*.csv")):
        frames.append(load_common_solvent_csv(csv_path, anchor_temperatures=anchor_temperatures))

    if not frames:
        raise FileNotFoundError(f"No COMMON-solvents CSVs found in {data_dir}")

    return pd.concat(frames, ignore_index=True)


def build_preprocessor(
    df: pd.DataFrame,
    solvent_descriptor_cols: Optional[Sequence[str]] = None,
    polymer_descriptor_cols: Optional[Sequence[str]] = None,
    include_solvent_identity: bool = True,
    include_polymer_identity: bool = True,
    anchor_temperatures: Sequence[float] = (25.0, 50.0, 75.0),
) -> ColumnTransformer:
    """
    Create a ColumnTransformer that mirrors the two-tower + interaction idea.

    The transformer keeps:
    - Numeric temperature features (including anchor deltas)
    - Optional solvent descriptors
    - Optional polymer descriptors
    - One-hot encodings for solvent/polymer identities (ignored for unseen values)
    """
    numeric_features: List[str] = []
    numeric_features.extend(NUMERIC_BASE_FEATURES)

    for anchor in anchor_temperatures:
        numeric_features.append(f"delta_from_{anchor:.0f}C")
        numeric_features.append(f"abs_delta_from_{anchor:.0f}C")

    if solvent_descriptor_cols:
        numeric_features.extend(list(solvent_descriptor_cols))
    if polymer_descriptor_cols:
        numeric_features.extend(list(polymer_descriptor_cols))

    numeric_features = [feat for feat in numeric_features if feat in df.columns]
    if not numeric_features:
        raise ValueError("No numeric features available for preprocessing.")

    transformers: List[Tuple[str, Pipeline, List[str]]] = []
    transformers.append(
        (
            "numeric",
            Pipeline(steps=[("scaler", StandardScaler())]),
            numeric_features,
        )
    )

    if include_solvent_identity:
        transformers.append(
            (
                "solvent",
                OneHotEncoder(handle_unknown="ignore"),
                ["solvent"],
            )
        )

    if include_polymer_identity:
        transformers.append(
            (
                "polymer",
                OneHotEncoder(handle_unknown="ignore"),
                ["polymer"],
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _make_estimator(model_type: str, random_state: int) -> object:
    """Return the regressor corresponding to the requested model type."""
    model_type = model_type.lower()
    if model_type == "ridge":
        return Ridge(alpha=1.0)
    if model_type == "gbr":
        return GradientBoostingRegressor(random_state=random_state)
    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_type == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=2000,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def build_model_pipeline(
    df: pd.DataFrame,
    model_type: str = "gbr",
    random_state: int = 0,
    solvent_descriptor_cols: Optional[Sequence[str]] = None,
    polymer_descriptor_cols: Optional[Sequence[str]] = None,
    include_solvent_identity: bool = True,
    include_polymer_identity: bool = True,
    anchor_temperatures: Sequence[float] = (25.0, 50.0, 75.0),
) -> Pipeline:
    """
    Assemble a preprocessing + model pipeline suitable for fitting/predicting.

    Unknown solvents or polymers at prediction time are handled gracefully by
    OneHotEncoder(handle_unknown="ignore"), which supports "new solvent" and
    "new polymer" scenarios while still using shared temperature features.
    """
    preprocessor = build_preprocessor(
        df,
        solvent_descriptor_cols=solvent_descriptor_cols,
        polymer_descriptor_cols=polymer_descriptor_cols,
        include_solvent_identity=include_solvent_identity,
        include_polymer_identity=include_polymer_identity,
        anchor_temperatures=anchor_temperatures,
    )
    estimator = _make_estimator(model_type, random_state=random_state)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def _compute_smoothness_penalty(
    df: pd.DataFrame,
    value_col: str,
    temp_col: str = "temperature_c",
    group_cols: Sequence[str] = ("polymer", "solvent"),
) -> float:
    """
    Penalize wiggly temperature curves by averaging absolute second differences.
    """
    penalties = []
    for _, group in df.groupby(list(group_cols)):
        if len(group) < 3:
            continue
        ordered = group.sort_values(temp_col)
        second_diff = np.diff(ordered[value_col], n=2)
        penalties.append(np.mean(np.abs(second_diff)))
    return float(np.mean(penalties)) if penalties else np.nan


def evaluate_models(
    df: pd.DataFrame,
    model_types: Iterable[str] = ("ridge", "gbr", "rf", "mlp"),
    target_col: str = "log_solubility",
    group_col: str = "polymer",
    n_splits: int = 3,
    random_state: int = 0,
    **preprocess_kwargs,
) -> pd.DataFrame:
    """
    Grouped cross-validation that holds out entire polymers.
    """
    n_groups = df[group_col].nunique()
    if n_groups < 2:
        raise ValueError("Grouped CV needs at least two distinct polymers.")
    n_splits = min(n_splits, n_groups)
    splitter = GroupKFold(n_splits=n_splits)

    results: List[ModelResult] = []

    for model_name in model_types:
        fold_mae: List[float] = []
        fold_rmse: List[float] = []
        fold_r2: List[float] = []
        fold_smooth: List[float] = []

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(df, groups=df[group_col])):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            model = build_model_pipeline(
                train_df,
                model_type=model_name,
                random_state=random_state + fold_idx,
                **preprocess_kwargs,
            )
            model.fit(train_df, train_df[target_col])
            preds = model.predict(test_df)

            fold_mae.append(mean_absolute_error(test_df[target_col], preds))
            fold_rmse.append(np.sqrt(mean_squared_error(test_df[target_col], preds)))
            fold_r2.append(r2_score(test_df[target_col], preds))

            pred_df = test_df.copy()
            pred_df["prediction"] = preds
            smoothness = _compute_smoothness_penalty(pred_df, "prediction")
            fold_smooth.append(smoothness)

        results.append(
            ModelResult(
                model_name=model_name,
                mae_mean=float(np.mean(fold_mae)),
                mae_std=float(np.std(fold_mae)),
                rmse_mean=float(np.mean(fold_rmse)),
                rmse_std=float(np.std(fold_rmse)),
                r2_mean=float(np.mean(fold_r2)),
                r2_std=float(np.std(fold_r2)),
                smoothness_mean=float(np.nanmean(fold_smooth)),
                smoothness_std=float(np.nanstd(fold_smooth)),
            )
        )

    return pd.DataFrame([r.__dict__ for r in results])


def fit_ensemble(
    df: pd.DataFrame,
    n_members: int = 5,
    model_type: str = "gbr",
    random_state: int = 0,
    **preprocess_kwargs,
) -> List[Pipeline]:
    """
    Train a deep-ensemble style collection of models with different seeds.
    """
    models: List[Pipeline] = []
    for idx in range(n_members):
        model = build_model_pipeline(
            df,
            model_type=model_type,
            random_state=random_state + idx,
            **preprocess_kwargs,
        )
        model.fit(df, df["log_solubility"])
        models.append(model)
    return models


def ensemble_predict(models: Sequence[Pipeline], df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return mean and standard deviation predictions from an ensemble.
    """
    predictions = np.stack([model.predict(df) for model in models], axis=0)
    return predictions.mean(axis=0), predictions.std(axis=0)


def rank_active_learning_batch(
    models: Sequence[Pipeline],
    candidate_df: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Score candidate (polymer, solvent, temperature) points by ensemble uncertainty.
    """
    mean_pred, std_pred = ensemble_predict(models, candidate_df)
    scored = candidate_df.copy()
    scored["prediction_mean"] = mean_pred
    scored["prediction_std"] = std_pred
    return scored.sort_values("prediction_std", ascending=False).head(top_k)


def _ensure_output_dir(path: Path) -> Path:
    """Create an output directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def visualize_cv_results(cv_df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    """
    Save CSV + plots summarizing grouped CV performance.
    """
    paths: Dict[str, Path] = {}
    output_dir = _ensure_output_dir(output_dir)

    csv_path = output_dir / "cv_metrics.csv"
    cv_df.to_csv(csv_path, index=False)
    paths["csv"] = csv_path

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    positions = np.arange(len(cv_df))
    labels = list(cv_df["model_name"])

    # MAE
    axes[0].errorbar(
        positions, cv_df["mae_mean"], yerr=cv_df["mae_std"], fmt="o", capsize=4
    )
    axes[0].set_title("MAE (log-solubility)")
    axes[0].set_ylabel("MAE")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")

    # RMSE
    axes[1].errorbar(
        positions, cv_df["rmse_mean"], yerr=cv_df["rmse_std"], fmt="o", capsize=4
    )
    axes[1].set_title("RMSE (log-solubility)")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")

    # R2 + smoothness overlay
    ax3 = axes[2]
    ax3.errorbar(
        positions, cv_df["r2_mean"], yerr=cv_df["r2_std"], fmt="o", color="C0", capsize=4
    )
    ax3.set_ylabel("R2", color="C0")
    ax3.tick_params(axis="y", labelcolor="C0")
    ax3.set_xticks(positions)
    ax3.set_xticklabels(labels, rotation=30, ha="right")
    ax3.set_title("R2 and smoothness")

    ax4 = ax3.twinx()
    ax4.errorbar(
        positions,
        cv_df["smoothness_mean"],
        yerr=cv_df["smoothness_std"],
        fmt="s",
        color="C3",
        capsize=4,
    )
    ax4.set_ylabel("Smoothness penalty (abs 2nd diff)", color="C3")
    ax4.tick_params(axis="y", labelcolor="C3")

    fig.tight_layout()
    fig_path = output_dir / "cv_metrics.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["fig"] = fig_path

    return paths


def visualize_active_learning(
    ranking_df: pd.DataFrame,
    output_dir: Path,
    top_k: int = 20,
    score_col: str = "prediction_std",
) -> Dict[str, Path]:
    """
    Save CSV + plots for active-learning candidate ranking.
    """
    paths: Dict[str, Path] = {}
    output_dir = _ensure_output_dir(output_dir)

    ranking_df.to_csv(output_dir / "active_learning_ranking.csv", index=False)
    paths["csv"] = output_dir / "active_learning_ranking.csv"

    head = ranking_df.head(top_k)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        range(len(head)),
        head[score_col],
        color="C1",
    )
    ax.set_yticks(range(len(head)))
    labels = (
        head["polymer"].astype(str)
        + " | "
        + head["solvent"].astype(str)
        + " @ "
        + head["temperature_c"].round(1).astype(str)
        + "C"
    )
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Ensemble std (uncertainty)")
    ax.set_title("Top candidate points by uncertainty")
    fig.tight_layout()

    fig_path = output_dir / "active_learning_ranking.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["fig"] = fig_path
    return paths


def run_common_solvent_cv_and_visualize(
    data_dir: Path,
    output_dir: Path = Path("results/common-solvents/cv"),
    model_types: Iterable[str] = ("ridge", "gbr", "rf"),
    n_splits: int = 3,
    random_state: int = 0,
    **preprocess_kwargs,
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """
    Convenience wrapper: load data, run grouped CV, save CSV + plots.
    """
    data = load_common_solvent_directory(data_dir)
    cv_df = evaluate_models(
        data,
        model_types=model_types,
        n_splits=n_splits,
        random_state=random_state,
        **preprocess_kwargs,
    )
    paths = visualize_cv_results(cv_df, output_dir)
    return cv_df, paths


def run_common_solvent_active_learning(
    data_dir: Path,
    output_dir: Path = Path("results/common-solvents/active_learning"),
    n_members: int = 5,
    model_type: str = "gbr",
    top_k: int = 20,
    random_state: int = 0,
    candidate_df: Optional[pd.DataFrame] = None,
    **preprocess_kwargs,
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """
    Train an ensemble, score candidates, and save ranking CSV + plots.
    """
    data = load_common_solvent_directory(data_dir)
    ensemble = fit_ensemble(
        data,
        n_members=n_members,
        model_type=model_type,
        random_state=random_state,
        **preprocess_kwargs,
    )

    if candidate_df is None:
        candidate_df = data.drop(columns=["log_solubility"])

    ranking = rank_active_learning_batch(ensemble, candidate_df, top_k=top_k)
    paths = visualize_active_learning(ranking, output_dir, top_k=top_k)
    return ranking, paths
