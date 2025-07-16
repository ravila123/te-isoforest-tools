#!/usr/bin/env python3
"""
train_isoforest_folder.py

Train an Isolation Forest anomaly model from **all per‑run Parquet files** in a folder.

Usage (no args; edit the CONFIG block below):
    python train_isoforest_folder.py

What it does:
  • Scans DATA_DIR for *.parquet files *other than* any file whose name starts with
    "baseline_" (to avoid double counting).
  • Concatenates them row‑wise (union of columns across runs).
  • Keeps only numeric measurement columns (float/int) for modeling.
  • Drops columns with all NaN and (optionally) columns that are constant.
  • Imputes remaining NaNs with column medians (robust to outliers).
  • Trains & light‑tunes an IsolationForest via a small parameter grid.
  • If sample count < MIN_VAL_SAMPLES, skips validation split and fits on all data
    (prints a warning).
  • Saves the trained pipeline (scaler + IF) to MODELS_DIR / MODEL_NAME.

Outputs:
  models/isoforest_v1.pkl     (change MODEL_NAME below to version bump)

Edit CONFIG paths below to match your system.
"""

from __future__ import annotations
import warnings
import argparse  # kept if you want to wire CLI later; not strictly needed
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# ------------------------------------------------------------------
# CONFIG – EDIT ME
# ------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "Dataset"          # where per‑run parquet files live
MODELS_DIR = ROOT / "models"      # where to write model
MODEL_NAME = "isoforest_v1.pkl"   # change to v2 when you retrain
EXCLUDE_PREFIXES = ("baseline_",) # ignore files starting w/ these when loading runs
MIN_VAL_SAMPLES = 5                # need ≥5 rows to do a train/val split
VAL_FRACTION = 0.20                # 20% validation when enough rows
DROP_CONST_COLS = True             # drop zero‑variance cols before training
RANDOM_SEED = 0

# small hyper‑param grid
PARAM_GRID = {
    "iso__n_estimators":  [100, 200],
    "iso__max_samples":   ["auto", 256],
    "iso__contamination": ["auto", 0.01, 0.03],  # tune expected outlier fraction
}
# ------------------------------------------------------------------


def load_all_runs(data_dir: Path, exclude_prefixes=EXCLUDE_PREFIXES) -> pd.DataFrame:
    """Load every Parquet in data_dir except excluded prefixes; stack rows."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    files = sorted(p for p in data_dir.glob("*.parquet")
                   if not any(p.name.startswith(pref) for pref in exclude_prefixes))
    if not files:
        raise SystemExit(f"No per‑run parquet files found in {data_dir}.")

    frames = []
    for fp in files:
        try:
            df = pd.read_parquet(fp)
            df["__source_file__"] = fp.name  # track origin
            frames.append(df)
            print(f"[load] {fp.name}: shape {df.shape}")
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Failed to read {fp}: {e}")
    data = pd.concat(frames, ignore_index=True, sort=True)  # union of cols
    return data


def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select numeric cols, drop all‑NaN & constant ones, impute remaining NaNs."""
    # select numeric
    num_df = df.select_dtypes(include=["number"]).copy()

    # drop all‑NaN columns
    all_nan_cols = num_df.columns[num_df.isna().all()].tolist()
    if all_nan_cols:
        num_df = num_df.drop(columns=all_nan_cols)
        print(f"[prep] dropped {len(all_nan_cols)} all‑NaN cols")

    # drop constant columns (variance == 0)
    if DROP_CONST_COLS:
        nunique = num_df.nunique(dropna=True)
        const_cols = nunique[nunique <= 1].index.tolist()
        if const_cols:
            num_df = num_df.drop(columns=const_cols)
            print(f"[prep] dropped {len(const_cols)} constant cols")

    # impute NaNs w/ median (robust)
    if num_df.isna().any().any():
        med = num_df.median(numeric_only=True)
        num_df = num_df.fillna(med)
        print("[prep] filled NaNs w/ column median")

    return num_df


def fit_grid(X_train: pd.DataFrame, X_val: pd.DataFrame | None = None):
    """Grid‑search a small set of params; return best pipeline & stats.
    If X_val is None (not enough samples), fit first grid candidate on full data.
    """
    best_model = None
    best_params = None
    best_fp = np.inf

    for params in ParameterGrid(PARAM_GRID):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("iso", IsolationForest(random_state=RANDOM_SEED, n_jobs=-1,
                                    **{k.split("__")[1]: v for k, v in params.items()})),
        ])

        if X_val is None:  # no validation; just fit & return first candidate
            pipe.fit(X_train)
            return pipe, params, None

        pipe.fit(X_train)
        preds = pipe["iso"].predict(X_val)  # +1 normal, -1 anomaly
        fp_rate = (preds == -1).mean()       # val set assumed normal
        print(f"  {params} => FP={fp_rate:.3%}")
        if fp_rate < best_fp:
            best_model, best_params, best_fp = pipe, params, fp_rate

    return best_model, best_params, best_fp


def main():
    print(f"[config] DATA_DIR={DATA_DIR}")
    print(f"[config] MODELS_DIR={MODELS_DIR}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # LOAD all runs
    raw = load_all_runs(DATA_DIR)
    print(f"[load] combined raw shape: {raw.shape}")

    # PREP numeric features
    X = prep_features(raw)
    print(f"[prep] numeric feature matrix shape: {X.shape}")

    # decide whether to split
    if len(X) >= MIN_VAL_SAMPLES:
        X_train, X_val = train_test_split(X, test_size=VAL_FRACTION,
                                          random_state=RANDOM_SEED, shuffle=True)
        print(f"[split] train={X_train.shape}, val={X_val.shape}")
    else:
        X_train, X_val = X, None
        print(f"[split] <{MIN_VAL_SAMPLES} samples; fitting on all data (no validation)")

    # FIT / GRID
    print("[train] grid search…")
    best_model, best_params, best_fp = fit_grid(X_train, X_val)

    print("[train] done.")
    if best_params is not None:
        print(f"[train] best params: {best_params}")
    if best_fp is not None:
        print(f"[train] validation FP rate: {best_fp:.3%}")

    # SAVE
    out_path = MODELS_DIR / MODEL_NAME
    joblib.dump(best_model, out_path)
    print(f"[save] model → {out_path}")

    # optional: save feature list used at train time
    feat_path = out_path.with_suffix(".features.txt")
    with feat_path.open("w") as f:
        for col in X.columns:
            f.write(col + "\n")
    print(f"[save] feature list → {feat_path}")


if __name__ == "__main__":
    main()
