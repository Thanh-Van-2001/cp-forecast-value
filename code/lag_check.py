"""Amendment 5 diagnostic (exploratory): alignment of each market's EIA-930
forecast with its demand. For every market and calendar year, the MAE (% of
mean load) of the forecast shifted by k hours, k in -2..2 (forecast row k
hours away paired with demand: k = -1 pairs demand at hour t with the
forecast stamped t-1). Uses the same screen as cp_study.py, without the PJM
realignment. Writes out_realign/lag_check.json.
Run from the repository root: python -I code/lag_check.py
"""
import json
import os
import sys

import numpy as np

sys.argv = ["cp_study.py", "."]
sys.path.insert(0, "code")
import cp_study as C  # noqa: E402

d = C.load()
out = {}
for ba in ["ERCO", "PJM", "NYIS", "ISNE"]:
    H = C.hourly_frame(d, ba)
    y = H["demand"].values
    f = H["forecast"].values
    yr = H.index.get_level_values("date").year.values
    mo = H.index.get_level_values("date").month.values
    out[ba] = {}
    for Y in range(2016, 2027):
        r = {}
        for label, sel in (("year", yr == Y), ("summer", (yr == Y) & (mo >= 6) & (mo <= 9))):
            j = np.where(sel)[0]
            m = {}
            for k in range(-2, 3):
                jj = j[(j + k >= 0) & (j + k < len(y))]
                ok = np.isfinite(f[jj + k]) & np.isfinite(y[jj])
                if ok.sum() >= 500:
                    m[k] = float(np.mean(np.abs(f[jj + k][ok] - y[jj][ok])) / np.mean(y[jj][ok]) * 100)
            if 0 in m:
                r[label] = {"best": min(m, key=m.get), "mae": m}
        if r:
            out[ba][Y] = r
os.makedirs("out_realign", exist_ok=True)
json.dump(out, open("out_realign/lag_check.json", "w"), indent=1)
for ba, r in out.items():
    print(ba, {Y: (v["year"]["best"], round(v["year"]["mae"][0], 2), round(min(v["year"]["mae"].values()), 2))
               for Y, v in r.items() if "year" in v})
