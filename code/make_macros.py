"""Every number in the paper. Reads out/results.json, out/secondary.json,
out/panel.csv, out/extract_log.json; writes paper/macros.tex and out/macros.json.
Run from the repository root: python -I code/make_macros.py
Amendment 5: the same macros from out_realign/ (PJM forecast realigned) carry the prefix "ra"."""
import json

import numpy as np
import pandas as pd

X = json.load(open("out/extract_log.json"))


def build(od):
    R = json.load(open(f"{od}/results.json"))
    S2 = json.load(open(f"{od}/secondary.json"))
    P = pd.read_csv(f"{od}/panel.csv")
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
    M["hThreeP"] = "<0.001" if H["H3"]["p_one_sided"] < 0.001 else f(H["H3"]["p_one_sided"], 2)
    for ba in MK:
        v = H["H4"]["per_market"][ba]
        M[f"hFourIso{TAG[ba]}"], M[f"hFourMl{TAG[ba]}"], M[f"hFourN{TAG[ba]}"] = str(v["iso_hits"]), str(v["ml_hits"]), str(v["n_cp"])

    # secondary configurations
    for key, tag in [("exact_m5_w3", "Exact"), ("actual_m5_w3", "Actual"), ("nomax_m5_w3", "Nomax"), ("forecast_m3_w3", "Mthree"),
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
    EXCL = {"ERCO_2016_missing": "ERCOT 2016 (a training season; missing data)",
            "PJM_2020_incomplete_CP_day_2020-07-27": "PJM 2020 (a removed recording error on a CP day)"}
    M["excludedSeasons"] = "; ".join(EXCL.get(e, e.replace("_", " ")) for e in R["notes"]["excluded"]) or "none"
    M["nScreenSummer"] = str(len(R["notes"]["screened_summer_list"]))
    # amendment 4 exploratory
    HS = S2["h1_subsets"]
    M["tauNoWeek"] = f(HS["no_week"]["tau"])
    M["tauStrong"], M["tauTailStrong"], M["nStrong"] = f(HS["no_week_pers_ml"]["tau"]), f(HS["no_week_pers_ml"]["tau_tail"]), str(HS["no_week_pers_ml"]["n"])
    M["hOneYlo"], M["hOneYhi"] = f(S2["h1_year_cluster_ci"][0]), f(S2["h1_year_cluster_ci"][1])
    M["hThreeYlo"], M["hThreeYhi"] = f(S2["h3_year_cluster_ci"][0]), f(S2["h3_year_cluster_ci"][1])
    for ba in MK:
        M[f"tauMae{TAG[ba]}"] = f(S2["per_market_tau"][ba]["mae"])
        M[f"tauTail{TAG[ba]}"] = f(S2["per_market_tau"][ba]["tail"])
        for fc, t in (("F_ISO", "Iso"), ("F_ml", "Ml")):
            e = S2["h4_exact"][ba][fc]
            M[f"ci{t}{TAG[ba]}"] = f"{100 * e['lo']:.0f}--{100 * e['hi']:.0f}"
    M["oosMae"], M["oosTail"], M["oosN"] = f(S2["oos_tau"]["mae"]), f(S2["oos_tau"]["tail"]), str(S2["oos_tau"]["n"])
    for sc, tag in (("forecast", ""), ("exact", "Exact"), ("nomax", "Nomax")):
        for fc in ("F_ISO", "F_ISObc", "F_ml", "S_bias", "S_under", "S_noise", "S_ar1", "S_shrink"):
            M[f"day{FT[fc]}{tag}"] = pct(S2[f"daycap_{sc}"][fc])
    AP = S2["alerts_market_forecast"]["PJM"]
    M["alertsShrinkPjm"] = f(AP["S_shrink"], 1)
    others = [v for k, v in AP.items() if k.startswith("S_") and k != "S_shrink"]
    M["alertsSynPjmMin"], M["alertsSynPjmMax"] = f(min(others), 1), f(max(others), 1)
    AN = S2["alerts_market_nomax"]
    allv = [v for m in AN.values() for v in m.values()]
    M["alertsNomaxPjmMin"], M["alertsNomaxPjmMax"] = f(min(AN["PJM"].values()), 1), f(max(AN["PJM"].values()), 1)
    AF = S2["alerts_market_forecast"]["PJM"]
    M["alertsFcPjmMin"], M["alertsFcPjmMax"] = f(min(AF.values()), 1), f(max(AF.values()), 1)
    M["alertsNomaxMin"], M["alertsNomaxMax"] = f(min(allv), 1), f(max(allv), 1)
    return M


M = build("out")
for k, v in build("out_realign").items():
    M["ra" + k[0].upper() + k[1:]] = v

# amendment 5: PJM detail, data as published vs realigned, budget x5
FTS = {"F_ISO": "Iso", "F_ISObc": "Isobc", "F_mix": "Mix", "F_ml": "Ml"}
SCS = {"forecast": "Prim", "nomax": "Nomax", "exact": "Exact", "actual": "Actual"}
for od, src in (("out", "Pub"), ("out_realign", "Ra")):
    Q = pd.read_csv(f"{od}/panel.csv")
    Q = Q[(Q.ba == "PJM") & (Q.mult == 5)]
    for sc, st in SCS.items():
        for fc, ft in FTS.items():
            for w in (1, 3):
                g = Q[(Q.scale == sc) & (Q.forecaster == fc) & (Q.window == w)]
                M[f"pjm{src}{ft}{st}W{w}"] = str(int(g.hits.sum()))
            g = Q[(Q.scale == sc) & (Q.forecaster == fc) & (Q.window == 3)]
            M[f"pjmDay{src}{ft}{st}"] = f"{100 * g.day_capture.mean():.0f}"
            M[f"pjmMae{src}{ft}"] = f"{g.mae_pct.mean():.1f}"
M["pjmNcp"] = str(int(Q[(Q.scale == "forecast") & (Q.forecaster == "F_ISO") & (Q.window == 3)].n_cp.sum()))

# amendment 5 diagnostic: alignment of each market's forecast (code/lag_check.py)
LC = json.load(open("out_realign/lag_check.json"))
pj = LC["PJM"]
pre = [v["year"]["mae"]["0"] for y, v in pj.items() if int(y) <= 2018]
post0 = [v["year"]["mae"]["0"] for y, v in pj.items() if int(y) >= 2019]
post1 = [v["year"]["mae"]["-1"] for y, v in pj.items() if int(y) >= 2019]
M["lagPjmPre"] = f"{np.mean(pre):.1f}"
M["lagPjmPostZero"] = f"{np.mean(post0):.1f}"
M["lagPjmPostShift"] = f"{np.mean(post1):.1f}"
M["lagPjmPostZeroMin"], M["lagPjmPostZeroMax"] = f"{min(post0):.1f}", f"{max(post0):.1f}"
M["lagPjmPostShiftMin"], M["lagPjmPostShiftMax"] = f"{min(post1):.1f}", f"{max(post1):.1f}"
M["lagPjmShare"] = f"{100 * (1 - np.mean(post1) / np.mean(post0)):.0f}"
M["lagPjmYearsBest"] = str(sum(1 for y, v in pj.items() if int(y) >= 2019 and v["year"]["best"] == -1))
M["lagPjmYearsPost"] = str(len(post0))
M["lagOtherShifted"] = str(sum(1 for b in ("ERCO", "NYIS", "ISNE") for v in LC[b].values() if v["year"]["best"] != 0))
M["lagOtherN"] = str(sum(len(LC[b]) for b in ("ERCO", "NYIS", "ISNE")))
ps = [v["summer"]["mae"]["0"] for y, v in pj.items() if int(y) >= 2019 and "summer" in v]
ps1 = [v["summer"]["mae"]["-1"] for y, v in pj.items() if int(y) >= 2019 and "summer" in v]
M["lagPjmSummerZero"], M["lagPjmSummerShift"] = f"{np.mean(ps):.1f}", f"{np.mean(ps1):.1f}"
RA = json.load(open("out_realign/results.json"))["hypotheses"]
M["raHThreePexact"] = f"{RA['H3']['p_one_sided']:.3f}"


with open("paper/macros.tex", "w", encoding="utf-8") as fh:
    fh.write("% generated by code/make_macros.py -- do not edit\n\\makeatletter\n")
    for k, v in sorted(M.items()):
        fh.write(f"\\expandafter\\def\\csname res@{k}\\endcsname{{{v}}}\n")
    fh.write("\\makeatother\n\\newcommand{\\res}[1]{\\ifcsname res@#1\\endcsname\\csname res@#1\\endcsname\\else\\textbf{??#1??}\\fi}\n")
json.dump(M, open("out/macros.json", "w"), indent=0)
print(len(M), "macros")
