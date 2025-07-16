import pandas as pd
df = pd.read_parquet("baseline_run_2025‑03‑27.parquet")   # or baseline_v1.parquet
print(df.shape)                 # e.g., (1, 180)  ← 1 run, 180 numeric columns
print(df.select_dtypes(float).iloc[0, :10])  # spot‑check first 10 measurements
