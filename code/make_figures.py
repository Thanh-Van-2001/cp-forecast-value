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
types = ["S_bias", "S_under", "S_shrink", "S_noise", "S_ar1"]
fig, ax = plt.subplots(figsize=(6.3, 3.0))
y = np.arange(len(types))
for i, (sc, lab, col) in enumerate(pol):
    q = P[(P.scale == sc) & (P.mult == 5) & (P.window == 3)]
    v = [100 * q[q.forecaster == t].capture.mean() for t in types]
    ax.scatter(v, y + (i - 1) * 0.22, s=30, color=col, label=lab, zorder=3,
               marker=["o", "s", "^"][i], edgecolor="white", linewidth=0.6)
for k in range(len(types)):
    ax.axhline(k, color="#f0f0f0", lw=6, zorder=0)
ax.set_yticks(y, [LAB[t] for t in types])
ax.invert_yaxis()
ax.set_xlabel("Coincident peaks captured (%), all errors with the same MAE as the ISO forecast")
ax.set_xlim(40, 102)
ax.grid(axis="x", color="#eeeeee", lw=0.6)
ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=1)
fig.tight_layout()
fig.savefig("paper/fig_error_types.pdf")
fig.savefig("out/fig_error_types.png")
plt.close(fig)
print("figures written")
