"""Every number in the paper. Reads out/results.json, out/secondary.json,
out/panel.csv, out/extract_log.json; writes paper/macros.tex and out/macros.json.
Run from the repository root: python -I code/make_macros.py"""
import json

import numpy as np
import pandas as pd

R = json.load(open("out/results.json"))
S2 = json.load(open("out/secondary.json"))
P = pd.read_csv("out/panel.csv")
X = json.load(open("out/extract_log.json"))
M = {}
MK = {"ERCO": "ERCOT", "PJM": "PJM", "NYIS": "NYISO", "ISNE": "ISONE"}
TAG = {"ERCO": "Ercot", "PJM": "Pjm", "NYIS": "Nyiso", "ISNE": "Isone"}
FT = {"F_ISO": "Iso", "F_ISObc": "Isobc", "F_pers": "Pers", "F_week": "Week", "F_ml": "Ml", "F_mix": "Mix",
      "S_bias": "Bias", "S_under": "Under", "S_noise": "Noise", "S_ar1": "Arone", "S_shrink": "Shrink"}


def f(x, d=2):
    return f"{x:.{d}f}"


def pct(x, d=0):
    return f"{100 * x:.{d}f}"


prim = P[(P.scale == "forecast") & (P.mult == 5) & (P.window == 3)]
M["nMarketYears"] = str(prim.groupby(["ba", "year"]).ngroups)
M["firstYear"], M["lastYear"] = str(int(prim.year.min())), str(int(prim.year.max()))
M["nYears"] = str(prim.year.nunique())
M["nForecasters"] = str(prim.forecaster.nunique())
M["nCPtotal"] = str(int(prim[prim.forecaster == "F_ISO"].n_cp.sum()))
M["nRows"] = f"{sum(v['rows'] for v in X['files']):,}"
miss = {k: v for k, v in X["by_ba_year"].items()}
mx = max(max(v["demand_missing"], v["forecast_missing"]) / v["rows"] for v in miss.values())
M["maxMissPct"] = pct(mx, 1)
for ba in MK:
    g = prim[(prim.ba == ba) & (prim.forecaster == "F_ISO")]
    M[f"nCP{TAG[ba]}"] = str(int(g.n_cp.sum()))
    M[f"budget{TAG[ba]}"] = str(int(g.budget.iloc[0]))

# primary summary per forecaster
for fc, t in FT.items():
    g = prim[prim.forecaster == fc]
    M[f"cap{t}"] = pct(g.capture.mean())
    M[f"mae{t}"] = f(g.mae_pct.mean(), 1)
    M[f"alerts{t}"] = f(g.alerts.mean(), 1)
    M[f"tail{t}"] = f(g["tail"].mean(), 2)
    for ba in MK:
        h = g[g.ba == ba]
        M[f"hits{t}{TAG[ba]}"] = str(int(h.hits.sum()))
        M[f"cap{t}{TAG[ba]}"] = pct(h.capture.mean())
M["alertsMin"] = f(prim.groupby("forecaster").alerts.mean().min(), 1)
M["alertsMax"] = f(prim.groupby("forecaster").alerts.mean().max(), 1)
M["budgetMean"] = f(prim.groupby(["ba", "year"]).budget.first().mean(), 1)

H = R["hypotheses"]
M["hOneTau"] = f(H["H1"]["mean_tau"])
M["hOneLo"], M["hOneHi"] = f(H["H1"]["ci"][0]), f(H["H1"]["ci"][1])
M["hOneN"] = str(H["H1"]["n_cells"])
M["hOneUndef"] = str(H["H1"]["n_undefined"])
M["hOneP"] = "<0.001" if H["H1"]["p_one_sided"] < 0.001 else f(H["H1"]["p_one_sided"], 3)
for k, t in (("S_bias", "Bias"), ("S_under", "Under"), ("S_noise", "Noise")):
    M[f"hTwoP{t}"] = f(H["H2"]["p"][k], 2)
M["hThreeTail"] = f(H["H3"]["mean_tau_tail"])
M["hThreeP"] = "<0.001" if H["H3"]["p_one_sided"] < 0.001 else f(H["H3"]["p_one_sided"], 3)
for ba in MK:
    v = H["H4"]["per_market"][ba]
    M[f"hFourIso{TAG[ba]}"], M[f"hFourMl{TAG[ba]}"], M[f"hFourN{TAG[ba]}"] = str(v["iso_hits"]), str(v["ml_hits"]), str(v["n_cp"])

# secondary configurations
for key, tag in [("exact_m5_w3", "Exact"), ("actual_m5_w3", "Actual"), ("forecast_m3_w3", "Mthree"),
                 ("forecast_m8_w3", "Meight"), ("forecast_m5_w1", "Wone"), ("forecast_m5_w5", "Wfive")]:
    v = S2[key]
    M[f"tau{tag}"] = f(v["H1_tau"])
    M[f"tail{tag}"] = f(v["H3_tail"])
    M[f"tailP{tag}"] = "<0.001" if v["H3_p"] < 0.001 else f(v["H3_p"], 3)
    for fc in ("S_bias", "S_under", "S_noise", "S_ar1", "S_shrink", "F_ISO", "F_ISObc", "F_ml"):
        M[f"cap{FT[fc]}{tag}"] = pct(v["capture"][fc])
    M[f"alerts{tag}Min"] = f(min(v["alerts"].values()), 1)
    M[f"alerts{tag}Max"] = f(max(v["alerts"].values()), 1)
taus = [S2[k]["H1_tau"] for k in S2 if k.startswith("forecast_") or k.startswith("exact_") or k.startswith("actual_")]
tails = [S2[k]["H3_tail"] for k in S2 if k.startswith("forecast_") or k.startswith("exact_") or k.startswith("actual_")]
M["tauAllMin"], M["tauAllMax"] = f(min(taus)), f(max(taus))
M["tailAllMin"], M["tailAllMax"] = f(min(tails)), f(max(tails))
M["nConfigs"] = str(len(taus))
M["tauNoCovid"] = f(S2["no2020"]["H1_tau"])
M["tailNoCovid"] = f(S2["no2020"]["H3_tail"])
M["tauSynTail"] = f(S2["synthetic_tail_tau"])
M["tauRealMae"] = f(S2["real_only_tau_mae"])
M["tauRealTail"] = f(S2["real_only_tau_tail"])
# spread of same-MAE synthetic capture (primary)
syn = [prim[prim.forecaster == k].capture.mean() for k in ("S_bias", "S_under", "S_noise", "S_ar1", "S_shrink")]
M["synSpread"] = pct(max(syn) - min(syn))
# amendment 3: unscreened run (out/v1) for disclosure
V1 = json.load(open("out/v1/results.json"))["hypotheses"]
M["vOneTau"], M["vOneTail"] = f(V1["H1"]["mean_tau"]), f(V1["H3"]["mean_tau_tail"])
SC = R["notes"]["screened_values"]
M["screenDemand"] = str(sum(v for k, v in SC.items() if k.endswith("demand")))
M["screenForecast"] = str(sum(v for k, v in SC.items() if k.endswith("forecast")))
M["excludedSeasons"] = ", ".join(R["notes"]["excluded"]) or "none"

with open("paper/macros.tex", "w", encoding="utf-8") as fh:
    fh.write("% generated by code/make_macros.py -- do not edit\n\\makeatletter\n")
    for k, v in sorted(M.items()):
        fh.write(f"\\expandafter\\def\\csname res@{k}\\endcsname{{{v}}}\n")
    fh.write("\\makeatother\n\\newcommand{\\res}[1]{\\ifcsname res@#1\\endcsname\\csname res@#1\\endcsname\\else\\textbf{??#1??}\\fi}\n")
json.dump(M, open("out/macros.json", "w"), indent=0)
print(len(M), "macros")
