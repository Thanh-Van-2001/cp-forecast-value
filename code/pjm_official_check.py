"""Amendment 5 diagnostics (exploratory), added after the internal review:
(a) PJM's official summer 5CP hours (PJM, "Summer YYYY Peaks and 5CPs", files and SHA-256 in
    data_pjm5cp/) against the EIA-930 data: is EIA demand's daily peak at the official hour
    (demand correctly stamped), and where is the forecast's daily peak, as published and realigned?
(b) the offset by season: best shift of PJM's forecast in standard-time months (Dec-Feb) and in
    daylight-saving months (Jun-Aug), per year, to rule out a daylight-saving error.
Writes out_realign/pjm_official_check.json. Run from the repository root: python -I code/pjm_official_check.py
"""
import json
import sys

import numpy as np
import pandas as pd

sys.argv = ["cp_study.py", "."]
sys.path.insert(0, "code")
import cp_study as C  # noqa: E402

# official RTO 5CP (date, hour ending EPT), transcribed from the PJM files in data_pjm5cp/
OFFICIAL = {
    2022: [("2022-07-20", 18), ("2022-07-21", 17), ("2022-07-22", 18), ("2022-08-08", 16), ("2022-08-03", 18)],
    2023: [("2023-07-27", 18), ("2023-09-05", 17), ("2023-07-28", 18), ("2023-09-06", 17), ("2023-07-05", 18)],
    2024: [("2024-07-16", 18), ("2024-07-15", 18), ("2024-08-01", 18), ("2024-06-21", 18), ("2024-08-28", 18)],
    2025: [("2025-06-23", 18), ("2025-06-24", 18), ("2025-07-29", 18), ("2025-06-25", 15), ("2025-07-28", 18)],
    2026: [("2026-07-02", 18), ("2026-07-01", 18), ("2026-07-15", 18), ("2026-07-16", 18), ("2026-09-01", 18)],
}
d = C.load()
H = C.hourly_frame(d, "PJM")
dem, fc = H["demand"], H["forecast"]
fc_ra = fc.shift(1)  # forecast delayed by one hour (the realignment of amendment 5)
rows = []
for yr, lst in OFFICIAL.items():
    for ds, he in lst:
        day = pd.Timestamp(ds)
        dd, ff, fr = dem.loc[day], fc.loc[day], fc_ra.loc[day]
        rows.append(dict(year=yr, date=ds, official_he=he,
                         eia_demand_peak_he=int(dd.idxmax()) if dd.notna().any() else None,
                         forecast_peak_he=int(ff.idxmax()) if ff.notna().any() else None,
                         forecast_realigned_peak_he=int(fr.idxmax()) if fr.notna().any() else None,
                         demand_at_official=float(dd.get(he, np.nan))))
R = pd.DataFrame(rows)
res = {"n": len(R),
       "demand_peak_at_official_hour": int((R.eia_demand_peak_he == R.official_he).sum()),
       "demand_peak_within_1h": int(((R.eia_demand_peak_he - R.official_he).abs() <= 1).sum()),
       "forecast_peak_at_official_hour": int((R.forecast_peak_he == R.official_he).sum()),
       "forecast_realigned_peak_at_official_hour": int((R.forecast_realigned_peak_he == R.official_he).sum()),
       "forecast_peak_minus_official_mean": float((R.forecast_peak_he - R.official_he).mean()),
       "forecast_realigned_peak_minus_official_mean": float((R.forecast_realigned_peak_he - R.official_he).mean())}
# are the official CP days the five highest EIA-demand days (our CP identification)?
match = {}
for yr, lst in OFFICIAL.items():
    ours = C.cp_hours(H, "PJM", yr, C.holidays(range(2015, 2027)))
    ours_d = {str(k[0].date()) for k in ours}
    match[yr] = len(ours_d & {ds for ds, _ in lst})
res["cp_days_matching_official"] = match
res["cp_days_matching_total"] = int(sum(match.values()))
# (b) offset by season
y = dem.values
f = fc.values
dates = H.index.get_level_values("date")
season = {}
for Y in range(2016, 2027):
    for lab, months in (("std_DecFeb", (12, 1, 2)), ("dst_JunAug", (6, 7, 8))):
        sel = np.where((dates.year == Y) & np.isin(dates.month, months))[0]
        m = {}
        for k in (-1, 0, 1):
            jj = sel[(sel + k >= 0) & (sel + k < len(y))]
            ok = np.isfinite(f[jj + k]) & np.isfinite(y[jj])
            if ok.sum() >= 500:
                m[k] = float(np.mean(np.abs(f[jj + k][ok] - y[jj][ok])) / np.mean(y[jj][ok]) * 100)
        if 0 in m:
            season[f"{Y}_{lab}"] = {"best": min(m, key=m.get), "mae": m}
res["season"] = season
post = {k: v for k, v in season.items() if int(k[:4]) >= 2019 and not k.startswith("2019_std")}
res["season_post2019_best_minus1"] = int(sum(v["best"] == -1 for v in post.values()))
res["season_post2019_n"] = len(post)
res["rows"] = rows
json.dump(res, open("out_realign/pjm_official_check.json", "w"), indent=1)
print({k: v for k, v in res.items() if k not in ("rows", "season")})
print(R.to_string())
