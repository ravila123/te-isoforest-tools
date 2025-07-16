#!/usr/bin/env python3
"""
Numeric_feature.py

Inspect per-feature deviations for a selected run against the population
used by the IsolationForest model.

Usage:
    python Numeric_feature.py                 # auto-pick most anomalous run
    python Numeric_feature.py <run_name>      # inspect specific run
"""

import argparse
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd


# --- PATH CONFIG -------------------------------------------------
ROOT = pathlib.Path("/Users/ravikiranl/Documents/TE_2_O")
DATA_DIR = ROOT / "Dataset"
MODEL_DIR = ROOT / "models"
MODEL_PKL = MODEL_DIR / "isoforest_v1.pkl"
FEAT_TXT = MODEL_DIR / "isoforest_v1.features.txt"
# -----------------------------------------------------------------


def load_all_runs() -> pd.DataFrame:
    """Stack every *.parquet file in DATA_DIR and return a DataFrame
    containing *all* rows across all runs, plus a __run__ column."""
    files = sorted(DATA_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found in {DATA_DIR}")

    frames = []
    for f in files:
        df = pd.read_parquet(f).copy()
        df["__run__"] = f.stem
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run",
        nargs="?",
        help="Run name to inspect. If omitted, the run with the lowest IsolationForest score (most anomalous) is used.",
    )
    args = parser.parse_args(argv)

    # --- Load model + feature list ---
    model = joblib.load(MODEL_PKL)
    feat_list = FEAT_TXT.read_text().strip().splitlines()

    # --- Load & align data ---
    all_df = load_all_runs()
    run_names = all_df.pop("__run__")  # aligned Series

    # ensure all expected features exist (missing columns will be NaN)
    X = all_df.reindex(columns=feat_list)
    X_raw = X.copy()

    # numeric-only statistics for imputation
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    train_medians = X[numeric_cols].median()
    X_filled = X.copy()
    X_filled[numeric_cols] = X_filled[numeric_cols].fillna(train_medians)

    # --- Score all rows with the trained IsolationForest ---
    scores = model.decision_function(X_filled)
    preds = model.predict(X_filled)

    # --- Determine which run to inspect ---
    if args.run is not None:
        bad_run = args.run
        if bad_run not in run_names.unique():
            raise SystemExit(
                f"Run '{bad_run}' not found. Available: {sorted(run_names.unique())[:10]}..."
            )
    else:
        # pick the run whose *mean* score across its rows is lowest (most anomalous)
        mean_scores_by_run = (
            pd.Series(scores, index=run_names).groupby(level=0).mean()
        )
        bad_run = mean_scores_by_run.idxmin()

    # choose a representative row from that run: the row with the lowest score in the run
    run_mask = run_names == bad_run
    run_indices = np.flatnonzero(run_mask.to_numpy())
    if len(run_indices) == 0:
        raise SystemExit(f"No rows found for run '{bad_run}' (this should not happen).")
    run_scores = scores[run_mask.to_numpy()]
    worst_in_run_local = np.argmin(run_scores)
    i = run_indices[worst_in_run_local]

    row_raw = X_raw.iloc[i]
    row_filled = X_filled.iloc[i]
    row_score = scores[i]
    row_pred = preds[i]

    print(f"\nInspecting run: {bad_run}")
    print(
        f"Row index={i}  Model score={row_score:+.6f}  pred={'ANOM' if row_pred==-1 else 'ok'}"
    )

    # --- Training distribution stats (numeric only) ---
    # Use the filled matrix so stats are computed on imputed values (consistent with model input).
    train_mean = X_filled[numeric_cols].mean()
    train_std = X_filled[numeric_cols].std(ddof=1)
    train_p05 = X_filled[numeric_cols].quantile(0.05)
    train_p95 = X_filled[numeric_cols].quantile(0.95)

    summary = pd.DataFrame(
        {
            "train_median": train_medians,
            "train_mean": train_mean,
            "train_std": train_std,
            "train_p05": train_p05,
            "train_p95": train_p95,
            "run_value": row_filled[numeric_cols],
            "raw_value": row_raw[numeric_cols],
        }
    )

    iqr = summary["train_p95"] - summary["train_p05"]
    summary["robust_dev"] = (
        (summary["run_value"] - summary["train_median"]).abs()
        / iqr.replace(0, np.nan)
    )
    summary["zscore"] = (
        (summary["run_value"] - summary["train_mean"])
        / summary["train_std"].replace(0, np.nan)
    )

    # --- Report top deviants ---
    top = summary.sort_values("robust_dev", ascending=False).head(15)
    print("\nTop deviant features (run vs training distribution):")
    print(
        top[
            [
                "run_value",
                "train_median",
                "train_p05",
                "train_p95",
                "robust_dev",
                "zscore",
            ]
        ].to_string()
    )

    # --- Feature contribution heuristic: clamp each feature to its median and rescore ---
    deltas = []
    row_vec = row_filled.to_frame().T  # DataFrame with single row
    for col in numeric_cols:
        tmp = row_vec.copy()
        tmp[col] = train_medians[col]
        s = model.decision_function(tmp)[0]
        deltas.append((col, s - row_score))

    deltas = (
        pd.DataFrame(deltas, columns=["feature", "score_delta"])
        .sort_values("score_delta", ascending=False)
        .reset_index(drop=True)
    )
    print(
        "\nFeatures that most improve score when reset to median (bigger delta = more suspicious):"
    )
    print(deltas.head(15).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())