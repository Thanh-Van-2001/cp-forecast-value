"""Figures. Run from the repository root: python -I code/make_figures.py"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P = pd.read_csv("out/panel.csv")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "xtick.color": "#444444", "ytick.color": "#444444",
                     "savefig.dpi": 300})
LAB = {"F_ISO": "ISO day-ahead", "F_ISObc": "ISO, bias-corrected", "F_pers": "Yesterday", "F_week": "Last week",
       "F_ml": "LightGBM (no weather)", "F_mix": "ISO + LightGBM", "S_bias": "Over-forecast", "S_under": "Under-forecast",
       "S_noise": "Independent noise", "S_ar1": "AR(1) noise", "S_shrink": "Shrink to mean"}
prim = P[(P.scale == "forecast") & (P.mult == 5) & (P.window == 3)]
g = prim.groupby("forecaster").agg(cap=("capture", "mean"), mae=("mae_pct", "mean"), tail=("tail", "mean"))

# Figure 1: capture vs error, real (circles) and same-MAE synthetic (diamonds)
fig, ax = plt.subplots(figsize=(6.3, 3.8))
for f, r in g.iterrows():
    syn = f.startswith("S_")
    ax.scatter(r.mae, 100 * r.cap, s=40, marker="D" if syn else "o", color="#D55E00" if syn else "#0072B2",
               edgecolor="white", linewidth=0.7, zorder=3)
    if syn:
        ylab = {"S_shrink": 89, "S_under": 85.5, "S_bias": 82, "S_ar1": 68, "S_noise": 63.5}[f]
        ax.annotate(LAB[f], (r.mae, 100 * r.cap), xytext=(4.9, ylab), fontsize=7, ha="left", va="center",
                    color="#333333", arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.6))
    else:
        ax.annotate(LAB[f], (r.mae, 100 * r.cap), xytext=(r.mae + 0.25, 100 * r.cap + {"F_ISObc": 0.4, "F_ISO": 0.6, "F_mix": -2.4}.get(f, 0)),
                    fontsize=7, ha="left", va="center", color="#333333")
ax.scatter([], [], marker="o", color="#0072B2", label="real forecasts")
ax.scatter([], [], marker="D", color="#D55E00", label="synthetic, same MAE as ISO")
ax.set_xlabel("Mean absolute error (% of mean summer load)")
ax.set_ylabel("Coincident peaks captured (%)")
ax.set_xlim(1.5, 14)
ax.set_ylim(10, 90)
ax.grid(color="#eeeeee", lw=0.6)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout()
fig.savefig("paper/fig_capture_mae.pdf")
fig.savefig("out/fig_capture_mae.png")
plt.close(fig)

# Figure 2: same-MAE error types under three ways of using the forecast
pol = [("forecast", "Own-scale threshold\n(primary)", "#0072B2"), ("actual", "Threshold from\nactual peaks", "#E69F00"),
       ("exact", "Exactly B best days\n(ex post)", "#009E73")]
types = ["S_bias", "S_under", "S_shrink", "F_ISO", "S_noise", "S_ar1"]
fig, ax = plt.subplots(figsize=(6.3, 3.3))
y = np.arange(len(types))
for i, (sc, lab, col) in enumerate(pol):
    q = P[(P.scale == sc) & (P.mult == 5) & (P.window == 3)]
    v = [100 * q[q.forecaster == t].capture.mean() for t in types]
    ax.scatter(v, y + (i - 1) * 0.22, s=30, color=col, label=lab, zorder=3,
               marker=["o", "s", "^"][i], edgecolor="white", linewidth=0.6)
for k in range(len(types)):
    ax.axhline(k, color="#f0f0f0", lw=6, zorder=0)
ax.set_yticks(y, [LAB[t] if t != "F_ISO" else "ISO day-ahead (its own error)" for t in types])
ax.invert_yaxis()
ax.set_xlabel("Coincident peaks caught (%); every row has the ISO forecast's MAE")
ax.set_xlim(40, 102)
ax.grid(axis="x", color="#eeeeee", lw=0.6)
ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=1)
fig.tight_layout()
fig.savefig("paper/fig_error_types.pdf")
fig.savefig("out/fig_error_types.png")
plt.close(fig)

# Figure 3 (amendment 5): PJM's one-hour offset in EIA-930 and its effect on CP capture
import json  # noqa: E402

LC = json.load(open("out_realign/lag_check.json"))
PUB, RA = "#D55E00", "#0072B2"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1.05, 1]})
names = {"ERCO": "ERCOT", "NYIS": "NYISO", "ISNE": "ISO-NE"}
for b, nm in names.items():
    yrs = sorted(int(y) for y in LC[b])
    a1.plot(yrs, [LC[b][str(y)]["year"]["mae"]["0"] for y in yrs], color="#bbbbbb", lw=1, zorder=1)
    a1.annotate(nm, (yrs[-1], LC[b][str(yrs[-1])]["year"]["mae"]["0"]), xytext=(3, 0), textcoords="offset points",
                fontsize=6.5, color="#777777", va="center")
yrs = sorted(int(y) for y in LC["PJM"])
a1.plot(yrs, [LC["PJM"][str(y)]["year"]["mae"]["0"] for y in yrs], color=PUB, lw=2, marker="o", ms=3.5,
        label="PJM, as published", zorder=3)
post = [y for y in yrs if y >= 2019]
a1.plot([2018] + post, [LC["PJM"]["2018"]["year"]["mae"]["0"]] + [LC["PJM"][str(y)]["year"]["mae"]["-1"] for y in post],
        color=RA, lw=2, ls="--", marker="o", ms=3.5, label="PJM, forecast delayed 1 h", zorder=3)
a1.axvline(2019, color="#999999", lw=0.8, ls=":")
a1.text(2018.85, 4.15, "offset from\nFeb 2019", fontsize=6.5, color="#555555", va="top", ha="right")
a1.set_ylabel("Forecast MAE (% of mean load)")
a1.set_xlim(2015.6, 2027.6)
a1.set_ylim(1.5, 4.2)
a1.set_xticks([2016, 2018, 2020, 2022, 2024, 2026])
a1.grid(color="#eeeeee", lw=0.6)
a1.legend(frameon=False, fontsize=6.5, loc="lower left")
a1.set_title("(a) Day-ahead forecast error by year", fontsize=8, loc="left")
cats = [("F_ISO", 1), ("F_ISO", 3), ("F_ISObc", 1), ("F_ISObc", 3)]
clab = ["ISO, 1-h window", "ISO, 3-h window", "ISO bias-corr., 1-h", "ISO bias-corr., 3-h"]
vals = {}
for od, key in (("out", "pub"), ("out_realign", "ra")):
    Q = pd.read_csv(f"{od}/panel.csv")
    Q = Q[(Q.ba == "PJM") & (Q.mult == 5) & (Q.scale == "nomax")]
    vals[key] = [int(Q[(Q.forecaster == fc) & (Q.window == w)].hits.sum()) for fc, w in cats]
ncp = int(Q[(Q.forecaster == "F_ISO") & (Q.window == 3)].n_cp.sum())
yy = np.arange(len(cats))
for off, key, col, lab in ((-0.19, "pub", PUB, "as published"), (0.19, "ra", RA, "delayed 1 h")):
    a2.barh(yy + off, vals[key], height=0.36, color=col, label=lab, edgecolor="white", linewidth=1)
    for yv, v in zip(yy + off, vals[key]):
        a2.text(v + 0.6, yv, str(v), fontsize=6.5, va="center", color="#333333")
a2.set_yticks(yy, clab, fontsize=7)
a2.invert_yaxis()
a2.set_xlim(0, ncp + 4)
a2.set_xlabel(f"PJM CP hours caught (of {ncp})")
a2.grid(axis="x", color="#eeeeee", lw=0.6)
a2.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2)
a2.set_title("(b) Capture at equal alert budgets", fontsize=8, loc="left")
fig.tight_layout()
fig.savefig("paper/fig_pjm_offset.pdf")
fig.savefig("out_realign/fig_pjm_offset.png")
plt.close(fig)
print("figures written")
