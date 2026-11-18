"""Secondary analyses (PLAN: secondary and robustness; amendment 2), from
out/panel.csv only. Writes out/secondary.json.
Run from the repository root: python -I code/analyze2.py"""
import json
import sys

import numpy as np
import pandas as pd

sys.argv = ["cp_study.py", "."]
sys.path.insert(0, "code")
import cp_study as C  # noqa: E402

P = pd.read_csv("out/panel.csv")
out = {}
for scale in ("forecast", "actual", "exact"):
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
out["years"] = sorted(int(y) for y in P.year.unique())
out["n_market_years"] = int(prim.groupby(["ba", "year"]).ngroups)
json.dump(out, open("out/secondary.json", "w"), indent=1, default=float)
for k in ("forecast_m5_w3", "exact_m5_w3", "actual_m5_w3", "forecast_m3_w3", "forecast_m8_w3", "forecast_m5_w1",
          "forecast_m5_w5"):
    v = out[k]
    print(k, "H1", round(v["H1_tau"], 3), "H3", round(v["H3_tail"], 3), round(v["H3_p"], 4),
          "shrink", round(v["H2_capture"]["S_shrink"], 2), "noise", round(v["H2_capture"]["S_noise"], 2),
          "bias", round(v["H2_capture"]["S_bias"], 2))
print("no2020", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["no2020"].items() if k != "H2_capture"})
print("syn tail tau", round(out["synthetic_tail_tau"], 3), "real-only mae/tail", round(out["real_only_tau_mae"], 3),
      round(out["real_only_tau_tail"], 3))
