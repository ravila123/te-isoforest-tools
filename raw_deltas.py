import pathlib, pandas as pd, joblib

ROOT = pathlib.Path("/Users/ravikiranl/Documents/TE_2_O")
DATA_DIR = ROOT / "Dataset"
MODEL_DIR = ROOT / "models"
BAD_RUN  = "run1"   # change if needed

# load features & data
feat_list = (MODEL_DIR / "isoforest_v1.features.txt").read_text().strip().splitlines()
frames=[]
for f in sorted(DATA_DIR.glob("*.parquet")):
    df=pd.read_parquet(f).copy(); df["__run__"]=f.stem; frames.append(df)
all_df=pd.concat(frames, ignore_index=True)
names=all_df.pop("__run__")
X=all_df.reindex(columns=feat_list)
med=X.median(numeric_only=True)
i = names[names==BAD_RUN].index[0]
row=X.iloc[i]
sub = med.to_frame("train_median").join(row.rename("run_value"))
sub["delta"] = sub["run_value"] - sub["train_median"]
sub["delta_mV"] = sub["delta"]*1000
print(sub.sort_values("delta_mV").head(20))
print("\nLargest positive deltas:")
print(sub.sort_values("delta_mV", ascending=False).head(20))
#PY