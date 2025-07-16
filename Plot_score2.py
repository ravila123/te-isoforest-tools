#!/usr/bin/env python3
"""
Plot Isolation Forest scores for all per-run Parquet files in DATA_DIR.
Also write a CSV of scores so you can review worst runs.

Run:
    python Plot_score2.py
"""

import pathlib
import numpy as np
import pandas as pd
import joblib

# Use non-GUI backend (saves PNGs; safe over SSH / headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- CONFIG ---------------------------------------------------------------
ROOT      = pathlib.Path(__file__).parent
DATA_DIR  = ROOT / "Dataset"   # where per-run parquet files live
MODEL_DIR = ROOT / "models"
MODEL_PKL = MODEL_DIR / "isoforest_v1.pkl"
FEAT_TXT  = MODEL_DIR / "isoforest_v1.features.txt"
# -------------------------------------------------------------------------


def load_runs() -> pd.DataFrame:
    """Load & stack all parquet runs; tag each row with its source file."""
    files = sorted(DATA_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files in {DATA_DIR}")
    frames = []
    for f in files:
        df = pd.read_parquet(f).copy()
        df["__run_name__"] = f.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    model = joblib.load(MODEL_PKL)
    feat_list = FEAT_TXT.read_text().strip().splitlines()

    df_all = load_runs()
    run_names = df_all.pop("__run_name__").tolist()  # aligns w/ rows

    # align & fill
    X = df_all.reindex(columns=feat_list)
    X = X.fillna(X.median(numeric_only=True))

    scores = model.decision_function(X)  # higher = more normal
    preds  = model.predict(X)            # +1 normal / -1 anomaly (@model cutoff)

    print(f"{len(scores)} runs scored.")
    print("Pred counts:", (preds == 1).sum(), "normal /", (preds == -1).sum(), "anomaly")

    pct = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\nScore percentiles:")
    for p in pct:
        print(f"  p{p:>2}: {np.percentile(scores, p):+.6f}")

    # save CSV
    out_csv = MODEL_DIR / "isoforest_v1.training_scores.csv"
    pd.DataFrame({"run": run_names, "score": scores, "pred": preds}) \
      .sort_values("score") \
      .to_csv(out_csv, index=False)
    print("\nWrote per-run scores →", out_csv)

    # histogram
    plt.figure()
    plt.hist(scores, bins=20)
    plt.axvline(0.0, ls="--", label="score=0 model cutoff")
    plt.xlabel("score (higher = more normal)")
    plt.ylabel("count")
    plt.title("Isolation Forest scores")
    plt.legend()
    plt.tight_layout()
    hist_png = MODEL_DIR / "isoforest_v1.score_hist.png"
    plt.savefig(hist_png, dpi=150)
    print("Saved histogram →", hist_png)

    # sorted curve
    plt.figure()
    plt.plot(np.sort(scores))
    plt.axhline(0.0, ls="--")
    plt.xlabel("run index (sorted asc)")
    plt.ylabel("score")
    plt.title("Sorted Isolation Forest scores")
    plt.tight_layout()
    sorted_png = MODEL_DIR / "isoforest_v1.score_sorted.png"
    plt.savefig(sorted_png, dpi=150)
    print("Saved sorted curve →", sorted_png)


if __name__ == "__main__":
    main()
