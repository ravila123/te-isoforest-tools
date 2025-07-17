#!/usr/bin/env python3
"""
Log_Parser_fixed.py

Hard‑path version: no command‑line args needed.

• Scans a *configured* input folder for .log/.txt files (non‑recursively by default).
• Parses each file, extracting numeric *measurement* results (Read/Measure/ADC/Volt/Res/Curr/Pwr/SHLD/Short).
• Writes ONE Parquet per source log into an output folder.
• Builds/updates a combined baseline_v1.parquet stacking all parsed rows.

Edit ONLY the CONFIG section below to change paths/behavior.

Works cross‑platform (macOS, Windows, Linux) because all paths go through pathlib.
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# ------------------------------------------------------------------
# CONFIG –‑‑ EDIT THESE PATHS FOR YOUR MACHINE
# ------------------------------------------------------------------
# Absolute paths are safest; use Path.home() if you move folders.
PROJECT_ROOT = Path(__file__).parent.resolve()
INDIR = Path("/Users/ravikiranl/Documents/TE_2_O/Log")  # folder that holds raw log files
OUTDIR = Path("/Users/ravikiranl/Documents/TE_2_O/Dataset") # where Parquet outputs go
# Whether to descend into subfolders under INDIR.
RECURSIVE = False
# Extensions we treat as logs (lower‑cased compare)
EXTENSIONS = {".log", ".txt"}
# Name of the combined baseline parquet we auto‑write after parsing.
BASELINE_NAME = "baseline_v1.parquet"
# ------------------------------------------------------------------

# Regex to capture header metadata.
HEADER_PAT = re.compile(r"^(Operator|Part number|Serial number|Start time):\s*(.+)")
# Regex to capture End: lines. We stop step name at the first comma.
END_PAT = re.compile(r"^\d{2}:\d{2}:\d{2}:\s+End:\s+(.+?),\s+Result:\s+(.+?),\s+Status:\s+(Pass|FAIL),")

# Keywords that indicate *measurement* steps we want to KEEP.
# Add/remove based on your test content.
KEEP_PAT = re.compile(
    r"(?i)\b(read|measure|meas|adc|volt|voltage|res|ohm|short|curr|current|pwr|power|shld)\b"
)


def iter_log_files(indir: Path, recursive: bool = False) -> Iterable[Path]:
    """Yield all log files under *indir* matching EXTENSIONS."""
    if not indir.is_dir():
        raise FileNotFoundError(f"Input log directory not found: {indir}")
    if recursive:
        yield from (p for p in indir.rglob('*') if p.suffix.lower() in EXTENSIONS and p.is_file())
    else:
        yield from (p for p in indir.iterdir() if p.suffix.lower() in EXTENSIONS and p.is_file())


def parse_one_file(fp: Path) -> Optional[pd.DataFrame]:
    """Parse ONE log file → 1×N DataFrame row (numeric measurement columns).

    Returns None if no numeric measurement rows found.
    """
    text_lines = fp.read_text(errors="ignore").splitlines()

    meta = {}
    rows = []
    for ln in text_lines:
        if m := HEADER_PAT.match(ln):
            key = m.group(1).lower().replace(" ", "_")  # operator -> operator
            meta[key] = m.group(2).strip()
            continue
        m = END_PAT.match(ln)
        if not m:
            continue
        step, result, status = m.groups()
        step = step.strip()

        # keep only PASS steps
        if status != "Pass":
            continue
        # keep only measurement-ish steps
        if not KEEP_PAT.search(step):
            continue
        # try to capture final numeric token in result string
        num_match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", result)
        if not num_match:
            continue
        val = float(num_match.group(1))
        rows.append({"step": step, "value": val})

    if not rows:
        print(f"  [WARN] {fp.name}: no numeric measurement rows captured.")
        return None

    df = pd.DataFrame(rows)
    # pivot to wide row
    pivot = df.pivot_table(index=None, columns="step", values="value", aggfunc="first")
    # attach metadata
    for k, v in meta.items():
        pivot[k] = v
    # mark overall pass (since we filtered FAILs, this simply True if rows exist)
    pivot["overall_pass"] = True
    return pivot


def main() -> None:
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input logs   : {INDIR}")
    print(f"Output dir   : {OUTDIR}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    parsed_rows = []
    for log_fp in iter_log_files(INDIR, RECURSIVE):
        print(f"\nParsing {log_fp.name} …")
        try:
            pivot_df = parse_one_file(log_fp)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  [ERROR] Failed to parse {log_fp}: {exc}")
            continue
        if pivot_df is None:
            continue

        # write one parquet per log, named after source file stem
        out_path = OUTDIR / f"{log_fp.stem}.parquet"
        pivot_df.to_parquet(out_path, index=False)
        print(f"  → wrote {out_path}  ({pivot_df.shape[1]} cols)")
        parsed_rows.append(pivot_df)

    if not parsed_rows:
        print("\nNo logs parsed; exiting.")
        sys.exit(1)

    # combine to baseline
    baseline_path = OUTDIR / BASELINE_NAME
    baseline_df = pd.concat(parsed_rows, ignore_index=True)
    baseline_df.to_parquet(baseline_path, index=False)
    print(f"\nCombined baseline → {baseline_path}  rows={len(baseline_df)} cols={baseline_df.shape[1]}")


if __name__ == "__main__":
    main()
