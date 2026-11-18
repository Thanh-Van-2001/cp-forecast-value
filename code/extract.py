"""Extract the four balancing authorities from the EIA-930 six-month files.

Keeps: authority, local date, hour ending (1-24 in local time as reported),
UTC end time, Demand (MW), Demand Forecast (MW). Writes data/eia930_4ba.csv.gz
and data/extract_log.json (rows, missing counts per authority and year).
Run: python -I extract.py <repo_root>
"""
import glob
import json
import os
import sys

import pandas as pd

ROOT = sys.argv[1]
BAS = ["ERCO", "PJM", "NYIS", "ISNE"]
COLS = {"Balancing Authority": "ba", "Data Date": "date", "Hour Number": "hour",
        "UTC Time at End of Hour": "utc_end", "Local Time at End of Hour": "local_end",
        "Demand (MW)": "demand", "Demand Forecast (MW)": "forecast"}
parts, log = [], {"files": []}
for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "EIA930_BALANCE_*.csv"))):
    df = pd.read_csv(f, usecols=list(COLS), dtype=str, low_memory=False).rename(columns=COLS)
    df = df[df["ba"].isin(BAS)].copy()
    for c in ("demand", "forecast"):
        df[c] = pd.to_numeric(df[c].str.replace(",", ""), errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    log["files"].append({"file": os.path.basename(f), "rows": int(len(df))})
    parts.append(df)
d = pd.concat(parts, ignore_index=True)
d["date"] = pd.to_datetime(d["date"], format="%m/%d/%Y", errors="coerce")
d["utc_end"] = pd.to_datetime(d["utc_end"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
dups = int(d.duplicated(["ba", "utc_end"]).sum())
d = d.drop_duplicates(["ba", "utc_end"], keep="last").sort_values(["ba", "utc_end"])
log["duplicates_dropped"] = dups
log["by_ba_year"] = {}
for (ba, yr), g in d.groupby(["ba", d["date"].dt.year]):
    log["by_ba_year"][f"{ba}_{yr}"] = {"rows": int(len(g)), "demand_missing": int(g["demand"].isna().sum()),
                                       "forecast_missing": int(g["forecast"].isna().sum())}
d.to_csv(os.path.join(ROOT, "data", "eia930_4ba.csv.gz"), index=False)
json.dump(log, open(os.path.join(ROOT, "data", "extract_log.json"), "w"), indent=1)
print(len(d), "rows;", dups, "duplicates dropped")
