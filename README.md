1. Log_parser.py  (pre-run Paraquet files for all golden runs)
-> Input: Raw TE Logs.
-> Output: Clean numeric Table (1 row = 1 run) saved as Dataset/<run_name>.parquet

2. isolation_forest.py
-> Input: All the pre-run Parquet files in Dataset/.
-> Output: A trained model saved to models/isofrest_v1.pkl, plus a tet file lissting exactly which feature columns were used in feature.txt

3. Plot_score2.py (model review/score visualization)
-> Input: Tranined models/isoforest_v1.plk
          feature.txt
          All pre-run Parquet files in Dataset/.
-> Output: Scores each run (how normal/abnormal), writ4es a .csv of runs+scores+model prediction, savved plots i.e. histograms of scores & sorted score curve in PNG file.


 
 
 
 
 
 
 
 ┌────────────────────────────┐
 │   Raw TE Logs (*.log)      │
 │  (instrument text output)  │
 └────────────┬───────────────┘
              │
              │ parse
              ▼
 ┌────────────────────────────┐
 │  Log Parsing Scripts       │
 │  • Log_Parser.py           │
 │  • parse_log.py (baseline) │
 └────────────┬───────────────┘
              │ 1 row per run
              ▼  (numeric cols)
 ┌────────────────────────────┐
 │   Dataset/  (Per-run       │
 │   *.parquet feature rows)  │
 └────────────┬───────────────┘
              │ offline model dev
              ▼
 ┌────────────────────────────┐
 │ train_isoforest_folder.py  │
 │  • load all Dataset rows   │
 │  • numeric select / clean  │
 │  • drop NaN & constant     │
 │  • median impute           │
 │  • scale + IsolationForest │
 │  • light grid search       │
 └────────────┬───────────────┘
              │ saves model artifacts
              ▼
 ┌─────────────────────────────────────────────┐
 │ models/                                     │
 │  • isoforest_v1.pkl (pipeline)              │
 │  • isoforest_v1.features.txt (train cols)   │
 │  • isoforest_v1.medians.parquet (optional)  │
 │  • isoforest_thresholds.json (you decide)   │
 └────────────┬────────────────────────────────┘
              │ analysis / threshold tuning
              ▼
 ┌────────────────────────────┐
 │ Plot_score2.py             │
 │  • load model + features   │
 │  • rescore all Dataset     │
 │  • CSV + plots             │
 └────────────┬───────────────┘
              │ drilldown
              ▼
 ┌────────────────────────────┐
 │ inspect_run.py (optional)  │
 │  • per-run deviation rank  │
 │  • "what-if" feature clamp │
 └────────────┬───────────────┘
              │ choose thresholds
              ▼
 ┌────────────────────────────┐
 │ score_new_run.py (TE hook) │
 │  • load model + thresholds │
 │  • score 1 new run parquet │
 │  • PASS / WARN / FAIL      │
 └────────────────────────────┘
