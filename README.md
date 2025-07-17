# TE IsoForest Tools

Collection of small utilities to parse test-equipment logs and train an
[Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
model for anomaly detection.

These scripts expect Python 3.8+ and the packages listed in
[`requirements.txt`](requirements.txt).

## Quick start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Adjust the path constants near the top of each script to match your
   environment.
3. Run the tools in the order below to build a model and inspect results.

## Workflow

### 1. Parse raw logs
`Log_Parser.py` scans a folder of text logs and converts each run into a
Parquet row of numeric measurements. The script writes one Parquet file per
source log under `Dataset/` and also builds `baseline_v1.parquet` containing
all parsed rows.

```
python Log_Parser.py
```

### 2. Train the Isolation Forest
`isolation_forest.py` loads all Parquet files from `Dataset/`, selects numeric
columns, imputes missing values with the median and trains an Isolation Forest.
The trained pipeline and the list of features used during training are saved in
`models/`.

```
python isolation_forest.py
```

### 3. Review training scores
`Plot_score2.py` rescoring every run using the trained model and writes a CSV of
scores plus PNG plots summarising the score distribution.

```
python Plot_score2.py
```

### 4. Inspect deviations for a run
`Numeric_feature.py` reports which features contribute most to an anomalous run.
Provide a run name or omit it to inspect the most anomalous run automatically.

```
python Numeric_feature.py <run_name>
```

### Additional helper scripts
- `raw_deltas.py` – quick check of raw value deltas versus training medians.
- `sanity_baseline.py` – simple demo printing the shape of the baseline
  Parquet file.

## Repository layout

```
Dataset/   # generated per-run Parquet files
models/    # saved models and artifacts
```

The scripts assume these folders exist in the repository root (they will be
created on first run if missing).

## Notes

- Edit the path variables at the top of each script to point to your log and
  output directories.
- The model assumes the runs in `Dataset/` are mostly normal. Outliers will have
  negative scores.
- See each script's docstring for further usage details.


