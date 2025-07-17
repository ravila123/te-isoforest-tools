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
