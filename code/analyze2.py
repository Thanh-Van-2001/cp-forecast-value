"""Secondary analyses (PLAN: secondary and robustness; amendment 2), from
out/panel.csv only. Writes out/secondary.json.
Run from the repository root: python -I code/analyze2.py [outdir]
(amendment 5: outdir=out_realign for the PJM-realigned run)"""
import json
import os
import sys

import numpy as np
import pandas as pd

OD = sys.argv[1] if len(sys.argv) > 1 else "out"
sys.argv = ["cp_study.py", "."]
sys.path.insert(0, "code")
import cp_study as C  # noqa: E402

P = pd.read_csv(os.path.join(OD, "panel.csv"))
out = {}
for scale in ("forecast", "actual", "exact", "nomax"):
    for mult in (3, 5, 8):
        for w in (1, 3, 5):
            sub = P[(P.scale == scale) & (P.mult == mult) & (P.window == w)]
            # reuse the pre-specified tests on this configuration
            Q = sub.assign(scale="forecast", mult=5, window=3)
            H = C.hypotheses(Q)
            out[f"{scale}_m{mult}_w{w}"] = dict(
                H1_tau=H["H1"]["mean_tau"], H1_ci=H["H1"]["ci"], H1_p=H["H1"]["p_one_sided"],
                H2_capture=H["H2"]["mean_capture"], H2_p_holm=H["H2"]["p_holm"],
                H3_tail=H["H3"]["mean_tau_tail"], H3_p=H["H3"]["p_one_sided"],
                H4=H["H4"]["per_market"],
                capture={f: float(g.capture.mean()) for f, g in sub.groupby("forecaster")},
                alerts={f: float(g.alerts.mean()) for f, g in sub.groupby("forecaster")})
# excluding 2020
sub = P[(P.scale == "forecast") & (P.mult == 5) & (P.window == 3) & (P.year != 2020)]
H = C.hypotheses(sub)
out["no2020"] = dict(H1_tau=H["H1"]["mean_tau"], H1_ci=H["H1"]["ci"], H3_tail=H["H3"]["mean_tau_tail"],
                     H3_p=H["H3"]["p_one_sided"], H2_capture=H["H2"]["mean_capture"])
# per-market capture table (primary)
prim = P[(P.scale == "forecast") & (P.mult == 5) & (P.window == 3)]
out["per_market"] = {ba: {f: dict(hits=int(h.hits.sum()), n_cp=int(h.n_cp.sum()), capture=float(h.capture.mean()),
                                  alerts=float(h.alerts.mean()), mae_pct=float(h.mae_pct.mean()))
                          for f, h in g.groupby("forecaster")} for ba, g in prim.groupby("ba")}
# exploratory: within the synthetic same-MAE group, capture vs tail score
syn = prim[prim.forecaster.str.startswith("S_")]
out["synthetic_tail_tau"] = float(np.nanmean(C.tau_by_cell(syn, "tail")))
# exploratory: within real forecasters only
real = prim[prim.forecaster.str.startswith("F_")]
out["real_only_tau_mae"] = float(np.nanmean(C.tau_by_cell(real, "mae_pct")))
out["real_only_tau_tail"] = float(np.nanmean(C.tau_by_cell(real, "tail")))
# --- amendment 4 exploratory analyses ---
from scipy import stats as st  # noqa: E402

# H1 on subsets without the weakest forecasters
subs = {"no_week": ["F_week"], "no_week_pers": ["F_week", "F_pers"], "no_week_pers_ml": ["F_week", "F_pers", "F_ml"]}
out["h1_subsets"] = {}
for k, drop in subs.items():
    sub_ = prim[~prim.forecaster.isin(drop)]
    t = C.tau_by_cell(sub_, "mae_pct")
    t2 = C.tau_by_cell(sub_, "tail")
    out["h1_subsets"][k] = dict(tau=float(np.nanmean(t)), n=int(t.notna().sum()), tau_tail=float(np.nanmean(t2)))
# year-clustered bootstrap for H1 and H3
tm = C.tau_by_cell(prim, "mae_pct")
tt = C.tau_by_cell(prim, "tail")
dfc = pd.DataFrame({"mae": tm, "tail": tt}).reset_index()
dfc.columns = ["ba", "year", "mae", "tail"]
yrs = dfc.year.unique()
rng = np.random.default_rng(5)
bm, btl = [], []
for _ in range(10000):
    pick = rng.choice(yrs, size=yrs.size, replace=True)
    s = pd.concat([dfc[dfc.year == y] for y in pick])
    bm.append(s["mae"].mean())
    btl.append(s["tail"].mean())
out["h1_year_cluster_ci"] = [float(np.nanpercentile(bm, 2.5)), float(np.nanpercentile(bm, 97.5))]
out["h3_year_cluster_ci"] = [float(np.nanpercentile(btl, 2.5)), float(np.nanpercentile(btl, 97.5))]
# per-market H1 and H3
out["per_market_tau"] = {ba: dict(mae=float(np.nanmean(C.tau_by_cell(g, "mae_pct"))),
                                  tail=float(np.nanmean(C.tau_by_cell(g, "tail")))) for ba, g in prim.groupby("ba")}
# out-of-sample: previous season's tail and MAE vs this season's capture
prev = prim[["ba", "year", "forecaster", "tail", "mae_pct"]].copy()
prev["year"] = prev["year"] + 1
oos = prim.drop(columns=["tail", "mae_pct"]).merge(prev, on=["ba", "year", "forecaster"], how="inner")
out["oos_tau"] = dict(mae=float(np.nanmean(C.tau_by_cell(oos, "mae_pct"))),
                      tail=float(np.nanmean(C.tau_by_cell(oos, "tail"))), n=int(oos.groupby(["ba", "year"]).ngroups))
# H4 with exact (Clopper-Pearson) intervals, primary policy
h4 = {}
for ba, g in prim.groupby("ba"):
    r = {}
    for fc in ("F_ISO", "F_ml"):
        h = g[g.forecaster == fc]
        k, n = int(h.hits.sum()), int(h.n_cp.sum())
        ci = st.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
        r[fc] = dict(hits=k, n=n, lo=float(ci.low), hi=float(ci.high))
    h4[ba] = r
out["h4_exact"] = h4
# day-level capture and alerts per forecaster (primary, exact-budget and nomax)
for sc in ("forecast", "exact", "nomax", "actual"):
    q = P[(P.scale == sc) & (P.mult == 5) & (P.window == 3)]
    out[f"daycap_{sc}"] = {f: float(g.day_capture.mean()) for f, g in q.groupby("forecaster")}
    out[f"alerts_market_{sc}"] = {ba: {f: float(h.alerts.mean()) for f, h in g.groupby("forecaster")}
                                  for ba, g in q.groupby("ba")}
out["years"] = sorted(int(y) for y in P.year.unique())
out["n_market_years"] = int(prim.groupby(["ba", "year"]).ngroups)
json.dump(out, open(os.path.join(OD, "secondary.json"), "w"), indent=1, default=float)
print("h1 subsets", out["h1_subsets"], "year-cluster", out["h1_year_cluster_ci"], out["h3_year_cluster_ci"])
print("per market", out["per_market_tau"], "oos", out["oos_tau"])
print("h4", out["h4_exact"])
print("daycap forecast", {k: round(v, 2) for k, v in out["daycap_forecast"].items()})
print("daycap exact", {k: round(v, 2) for k, v in out["daycap_exact"].items()})
print("daycap nomax", {k: round(v, 2) for k, v in out["daycap_nomax"].items()})
print("alerts PJM forecast", {k: round(v, 1) for k, v in out["alerts_market_forecast"]["PJM"].items()})
print("alerts PJM nomax", {k: round(v, 1) for k, v in out["alerts_market_nomax"]["PJM"].items()})
for k in ("forecast_m5_w3", "exact_m5_w3", "actual_m5_w3", "nomax_m5_w3", "forecast_m3_w3", "forecast_m8_w3",
          "forecast_m5_w1", "forecast_m5_w5"):
    v = out[k]
    print(k, "H1", round(v["H1_tau"], 3), "H3", round(v["H3_tail"], 3), round(v["H3_p"], 4),
          "shrink", round(v["H2_capture"]["S_shrink"], 2), "noise", round(v["H2_capture"]["S_noise"], 2),
          "bias", round(v["H2_capture"]["S_bias"], 2))
print("no2020", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["no2020"].items() if k != "H2_capture"})
print("syn tail tau", round(out["synthetic_tail_tau"], 3), "real-only mae/tail", round(out["real_only_tau_mae"], 3),
      round(out["real_only_tau_tail"], 3))
