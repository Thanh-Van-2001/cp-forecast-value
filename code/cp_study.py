"""Coincident-peak capture vs forecast accuracy (see PLAN.md).

Run: python -I cp_study.py <repo_root> [--check]
--check: pipeline check on ERCO, test seasons 2018-2019 only (PLAN workflow).
Writes out/panel.csv (one row per market, year, forecaster, configuration)
and out/results.json (hypotheses, summaries).
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = sys.argv[1]
CHECK = "--check" in sys.argv
# amendment 5: PJM's EIA-930 forecast is stamped one hour early relative to its
# demand in every month from February 2019 (found in a separate study of EIA-930
# recording errors); --realign-pjm delays it by one hour from that month on.
REALIGN = "--realign-pjm" in sys.argv
PJM_SHIFT_FROM = pd.Timestamp("2019-02-01")
OUT = os.path.join(ROOT, "out_check" if CHECK else ("out_realign" if REALIGN else "out"))
os.makedirs(OUT, exist_ok=True)
BAS = ["ERCO"] if CHECK else ["ERCO", "PJM", "NYIS", "ISNE"]
NCP = {"ERCO": 4, "PJM": 5, "NYIS": 1, "ISNE": 1}
SEASON = (6, 9)
FIRST_TRAIN, FIRST_TEST = 2016, 2018
A_RUN = 0.97
MULTS = [3, 5, 8]
WINDOWS = [1, 3, 5]
REAL = ["F_ISO", "F_ISObc", "F_pers", "F_week", "F_ml", "F_mix"]
SYN = ["S_bias", "S_under", "S_noise", "S_ar1", "S_shrink"]
FCS = REAL + SYN
RNG_SEED = 20261118
SCREEN = {}
SCREEN_LIST = []


def holidays(years):
    """NERC holidays (amendment 4 fix): New Year, Memorial Day, Independence
    Day, Labor Day, Thanksgiving, Christmas; a holiday on Sunday moves to
    Monday, a holiday on Saturday is not moved."""
    out = []
    for y in range(min(years), max(years) + 1):
        for d in (pd.Timestamp(y, 1, 1), pd.Timestamp(y, 7, 4), pd.Timestamp(y, 12, 25)):
            out.append(d + pd.Timedelta(days=1) if d.weekday() == 6 else d)
        may = pd.date_range(f"{y}-05-25", f"{y}-05-31")
        out.append(may[may.weekday == 0][-1])
        sep = pd.date_range(f"{y}-09-01", f"{y}-09-07")
        out.append(sep[sep.weekday == 0][0])
        nov = pd.date_range(f"{y}-11-22", f"{y}-11-28")
        out.append(nov[nov.weekday == 3][0])
    return set(pd.DatetimeIndex(out).normalize())


def load():
    d = pd.read_csv(os.path.join(ROOT, "data", "eia930_4ba.csv.gz"), parse_dates=["date", "utc_end"])
    d = d[d["hour"].between(1, 24)]
    return d


def hourly_frame(d, ba):
    g = d[d["ba"] == ba].copy()
    g = g.drop_duplicates(["date", "hour"], keep="last")
    piv = g.set_index(["date", "hour"])[["demand", "forecast"]]
    days = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
    idx = pd.MultiIndex.from_product([days, range(1, 25)], names=["date", "hour"])
    piv = piv.reindex(idx)
    if REALIGN and ba == "PJM":
        # amendment 5: forecast row one hour earlier is paired with demand from Feb 2019
        late = piv.index.get_level_values("date") >= PJM_SHIFT_FROM
        piv.loc[late, "forecast"] = piv["forecast"].shift(1)[late].values
    # amendment 3, revised by amendment 4: screen only gross recording errors
    # (EIA-930 spikes, zeros, an integer-overflow sentinel). A value is set
    # missing if it is <= 0 or lies outside [0.3, 2.0] x the centred 168-hour
    # rolling median of actual demand (the 1.5 bound of amendment 3 removed
    # genuine heat-wave peaks). Every removed June-September value is logged.
    med = piv["demand"].where(piv["demand"] > 0).rolling(168, center=True, min_periods=24).median()
    for c in ("demand", "forecast"):
        bad = (piv[c] <= 0) | (piv[c] > 2.0 * med) | (piv[c] < 0.3 * med)
        SCREEN[f"{ba}_{c}"] = int(bad.sum())
        dd = piv.index.get_level_values("date")
        summer = bad & (dd.month >= 6) & (dd.month <= 9)
        SCREEN_LIST.extend([f"{ba} {c} {k[0].date()} h{k[1]} {piv.loc[k, c]:.0f} (median {med.loc[k]:.0f})"
                            for k in piv.index[summer.values]])
        piv.loc[bad, c] = np.nan
    return piv


def make_ml(H, years, hol):
    """Expanding-window LightGBM per year: for year y train on all hours of
    earlier years (lag features only), predict year y."""
    import lightgbm as lgb
    df = H.copy()
    y = df["demand"]
    lag24 = y.shift(24)
    lag168 = y.shift(168)
    dpk = y.groupby(level="date").transform("max").shift(24)
    dates = df.index.get_level_values("date")
    X = pd.DataFrame({"lag24": lag24.values, "lag168": lag168.values, "dpk1": dpk.values,
                      "hour": df.index.get_level_values("hour"), "wd": dates.weekday, "doy": dates.dayofyear,
                      "hol": dates.normalize().isin(hol).astype(int)}, index=df.index)
    out = pd.Series(np.nan, index=df.index)
    yr = dates.year
    for t in years:
        tr = (yr < t) & X.notna().all(axis=1).values & y.notna().values
        te = (yr == t) & X.notna().all(axis=1).values
        if tr.sum() < 24 * 120 or te.sum() == 0:
            continue
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=7, verbose=-1)
        m.fit(X[tr], y[tr])
        out[te] = m.predict(X[te])
    return out


def real_forecasts(H, years, hol):
    y = H["demand"]
    F = {"F_ISO": H["forecast"]}
    err = (H["forecast"] - y)
    hour = H.index.get_level_values("hour")
    bias = err.groupby(hour).transform(lambda s: s.shift(1).rolling(28, min_periods=14).mean())
    F["F_ISObc"] = H["forecast"] - bias
    F["F_pers"] = y.shift(24)
    F["F_week"] = y.shift(168)
    F["F_ml"] = make_ml(H, years, hol)
    F["F_mix"] = (F["F_ISO"] + F["F_ml"]) / 2
    return F


def season_mask(idx, year):
    d = idx.get_level_values("date")
    return (d.year == year) & (d.month >= SEASON[0]) & (d.month <= SEASON[1])


def synthetic(H, F_iso, year, rng):
    """Synthetic forecasts for one year's season hours, each with season MAE
    equal to F_ISO's season MAE that year."""
    y = H["demand"]
    m = season_mask(H.index, year)
    ys = y[m]
    target = float(np.nanmean(np.abs(F_iso[m] - ys)))
    hour = H.index.get_level_values("hour")
    m28 = y.groupby(hour).transform(lambda s: s.shift(1).rolling(28, min_periods=14).mean())[m]
    S = {}
    S["S_bias"] = ys + target
    S["S_under"] = ys - target
    e = rng.normal(size=ys.size)
    S["S_noise"] = ys + e * target / np.mean(np.abs(e))
    a = np.empty(ys.size)
    a[0] = rng.normal()
    z = rng.normal(size=ys.size) * math.sqrt(1 - 0.9 ** 2)
    for i in range(1, ys.size):
        a[i] = 0.9 * a[i - 1] + z[i]
    S["S_ar1"] = ys + a * target / np.mean(np.abs(a))
    dev = (ys - m28)
    k = target / float(np.nanmean(np.abs(dev)))
    S["S_shrink"] = ys - k * dev
    return S, target


def cp_hours(H, ba, year, hol):
    """Actual CP (date, hour) list for the market-year; None if ISNE annual
    peak falls outside the season."""
    y = H["demand"]
    d = H.index.get_level_values("date")
    if ba == "ERCO":
        out = []
        for mo in range(SEASON[0], SEASON[1] + 1):
            s = y[(d.year == year) & (d.month == mo)]
            if s.notna().sum() == 0:
                return []
            out.append(s.idxmax())
        return out
    sm = season_mask(H.index, year)
    if ba == "PJM":
        s = y[sm].dropna()
        dd = s.index.get_level_values("date")
        ok = (dd.weekday < 5) & (~dd.normalize().isin(hol))
        s = s[ok]
        daily = s.groupby(level="date").agg(["max", "idxmax"])
        top = daily.sort_values("max", ascending=False).head(5)
        return list(top["idxmax"])
    if ba == "NYIS":
        return [y[sm].idxmax()]
    if ba == "ISNE":
        s = y[d.year == year]
        k = s.idxmax()
        return [k] if SEASON[0] <= k[0].month <= SEASON[1] else None


def eligible_days(days, ba, hol):
    days = pd.DatetimeIndex(days)
    if ba == "PJM":
        return days[(days.weekday < 5) & (~days.normalize().isin(hol))]
    return days


def period_key(day, ba):
    return day.month if ba == "ERCO" else 0


def run_policy(P, Ph, act_pk, days, ba, q, scale):
    """Online alert policy over one season. P, Ph: forecast daily peak and
    peak hour by day; act_pk: actual daily peaks. Returns alerted days and
    window centres."""
    alerts = {}
    run = {}
    for day in days:
        pk = period_key(day, ba)
        mprev = run.get(pk, -np.inf)
        p = P.get(day, np.nan)
        a = 0.0 if scale == "nomax" else A_RUN  # amendment 4: exploratory policy without running maximum
        if np.isfinite(p) and p >= max(q, a * mprev):
            alerts[day] = Ph[day]
        # update the running maximum with information known at the end of the day
        upd = p if scale in ("forecast", "nomax") else act_pk.get(day, np.nan)
        if np.isfinite(upd):
            run[pk] = max(mprev, upd)
    return alerts


def daily_peaks(series, days):
    s = series[series.index.get_level_values("date").isin(days)].dropna()
    g = s.groupby(level="date")
    return g.max().to_dict(), g.idxmax().map(lambda k: k[1] if isinstance(k, tuple) else np.nan).to_dict()


def calibrate_q(train_sets, ba, B, scale):
    """Choose the threshold level so that the policy issues on average B
    alerts per training season. train_sets: list of (P, Ph, act_pk, days)."""
    pool = np.concatenate([[p for d, p in P.items() if d in set(days)] if scale in ("forecast", "nomax")
                           else [a for d, a in A.items() if d in set(days)]
                           for P, Ph, A, days in train_sets])
    pool = pool[np.isfinite(pool)]
    if pool.size == 0:
        return np.nan
    lo, hi = 0.0, 1.0
    best = None
    for _ in range(30):
        mid = (lo + hi) / 2
        q = np.quantile(pool, mid)
        n = np.mean([len(run_policy(P, Ph, A, days, ba, q, scale)) for P, Ph, A, days in train_sets])
        best = q
        if n > B:
            lo = mid
        else:
            hi = mid
    return best


def main():
    d = load()
    allyears = sorted(d["date"].dt.year.unique())
    hol = holidays(allyears)
    last = max(y for y in allyears if ((d["date"].dt.year == y) & (d["date"].dt.month == 9)).any()
               and d[(d["date"].dt.year == y) & (d["date"].dt.month == 9)]["date"].dt.day.max() >= 30)
    test_years = list(range(FIRST_TEST, 2020 if CHECK else last + 1))
    rows, notes = [], {"last_season": int(last), "excluded": []}
    rng = np.random.default_rng(RNG_SEED)
    for ba in BAS:
        H = hourly_frame(d, ba)
        years = list(range(FIRST_TRAIN, max(test_years) + 1))
        F = real_forecasts(H, years, hol)
        # synthetic forecasts per season, assembled into full-length series
        S = {k: pd.Series(np.nan, index=H.index) for k in SYN}
        mae_iso = {}
        for yr in years:
            m = season_mask(H.index, yr)
            if H["demand"][m].isna().mean() > 0.02 or F["F_ISO"][m].isna().mean() > 0.02:
                notes["excluded"].append(f"{ba}_{yr}_missing")
                continue
            Sy, t = synthetic(H, F["F_ISO"], yr, rng)
            mae_iso[yr] = t
            for k in SYN:
                S[k][m] = Sy[k].values
        allF = {**F, **S}
        # amendment 2: thresholds act on peaks divided by the mean actual load of
        # the previous 365 days (removes load growth between training and test seasons)
        dmean = H["demand"].groupby(level="date").mean()
        abar = dmean.rolling(365, min_periods=200).mean().shift(1)
        # per season: eligible days, actual daily peaks, forecast daily peaks (raw and normalized)
        seas, raw = {}, {}
        for yr in years:
            if yr not in mae_iso:
                continue
            days = pd.date_range(f"{yr}-06-01", f"{yr}-09-30", freq="D")
            el = eligible_days(days, ba, hol)
            act_pk, _ = daily_peaks(H["demand"], el)
            norm = lambda dct: {k: v / abar.get(k, np.nan) for k, v in dct.items()}
            per_f, per_raw = {}, {}
            for f in FCS:
                P, Ph = daily_peaks(allF[f], el)
                per_f[f] = (norm(P), Ph)
                per_raw[f] = P
            seas[yr] = (el, norm(act_pk), per_f)
            raw[yr] = per_raw
        for yr in test_years:
            if yr not in seas:
                continue
            cps = cp_hours(H, ba, yr, hol)
            if cps is None:
                notes["excluded"].append(f"{ba}_{yr}_annual_peak_outside_season")
                continue
            if not cps:
                continue
            # pre-specified rule (implemented in amendment 4): exclude the market-year if a CP day
            # lacks any hour of actual demand or of the ISO forecast
            incomplete = [str(dd.date()) for dd, _ in cps
                          if H["demand"].loc[dd].isna().any() or F["F_ISO"].loc[dd].isna().any()]
            if incomplete:
                notes["excluded"].append(f"{ba}_{yr}_incomplete_CP_day_{','.join(incomplete)}")
                continue
            cp_days = {dd for dd, _ in cps}
            el, act_pk, per_f = seas[yr]
            m = season_mask(H.index, yr)
            ys = H["demand"][m]
            mean_load = float(ys.mean())
            act_raw, _ = daily_peaks(H["demand"], el)
            top5_act = set(pd.Series(act_raw).sort_values(ascending=False).head(5).index)
            train_years = [t for t in seas if t < yr]
            for f in FCS:
                P, Ph = per_f[f]
                Praw = raw[yr][f]
                # synthetic forecasts share F_ISO's MAE by construction (amendment 4: exact ties)
                mae = float(np.nanmean(np.abs(allF["F_ISO" if f in SYN else f][m] - ys)))
                top5_f = set(pd.Series(Praw).dropna().sort_values(ascending=False).head(5).index)
                tail = len(top5_act & top5_f) / 5.0
                for scale in ("forecast", "actual", "exact", "nomax"):
                    for mult in MULTS:
                        B = mult * NCP[ba]
                        if scale == "exact":
                            # amendment 2 (secondary): exactly B alerts on the highest forecast-peak
                            # days of the season (ERCOT: B/4 per month); isolates ranking quality
                            s = pd.Series(Praw).dropna()
                            if ba == "ERCO":
                                pick = []
                                for mo in range(SEASON[0], SEASON[1] + 1):
                                    sm_ = s[[k.month == mo for k in s.index]]
                                    pick += list(sm_.sort_values(ascending=False).head(B // 4).index)
                            else:
                                pick = list(s.sort_values(ascending=False).head(B).index)
                            alerts = {k: Ph[k] for k in pick}
                        else:
                            train = [(seas[t][2][f][0], seas[t][2][f][1], seas[t][1], seas[t][0]) for t in train_years]
                            q = calibrate_q(train, ba, B, scale)
                            alerts = run_policy(P, Ph, act_pk, el, ba, q, scale)
                        for w in WINDOWS:
                            half = (w - 1) // 2
                            hit = sum(1 for (dd, hh) in cps if dd in alerts and np.isfinite(alerts[dd])
                                      and abs(hh - alerts[dd]) <= half)
                            hit_day = sum(1 for dd in cp_days if dd in alerts)
                            rows.append(dict(ba=ba, year=yr, forecaster=f, scale=scale, mult=mult, window=w,
                                             n_cp=len(cps), hits=hit, capture=hit / len(cps), alerts=len(alerts),
                                             day_capture=hit_day / len(cp_days),
                                             budget=B, mae=mae, mae_pct=100 * mae / mean_load, tail=tail,
                                             n_train=len(train_years)))
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(OUT, "panel.csv"), index=False)
    notes["screened_values"] = SCREEN
    notes["screened_summer_list"] = SCREEN_LIST
    res = {"notes": notes, "hypotheses": hypotheses(P), "summary": summary(P)}
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
    print(json.dumps(res["hypotheses"], indent=1, default=float)[:3000])


def tau_by_cell(P, col):
    out = []
    for (ba, yr), g in P.groupby(["ba", "year"]):
        if g["capture"].nunique() < 2 or g[col].nunique() < 2:
            out.append(((ba, yr), np.nan))
            continue
        out.append(((ba, yr), stats.kendalltau(g[col], g["capture"], variant="b").statistic))
    return pd.Series(dict(out))


def boot_mean(x, reps=10000, seed=1):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    r = np.random.default_rng(seed)
    bs = r.choice(x, size=(reps, x.size), replace=True).mean(axis=1)
    return bs


def hypotheses(P):
    p = P[(P.scale == "forecast") & (P.mult == 5) & (P.window == 3)]
    H = {}
    t_mae = tau_by_cell(p, "mae_pct")
    bs = boot_mean(t_mae.values)
    H["H1"] = dict(mean_tau=float(np.nanmean(t_mae)), n_cells=int(t_mae.notna().sum()),
                   n_undefined=int(t_mae.isna().sum()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                   p_one_sided=float((np.sum(bs >= 0) + 1) / (bs.size + 1)))
    H["H1"]["supported"] = H["H1"]["p_one_sided"] < 0.05
    piv = p.pivot_table(index=["ba", "year"], columns="forecaster", values="capture")
    ps = {}
    for other in ("S_bias", "S_under", "S_noise"):
        x, y = piv["S_shrink"], piv[other]
        dlt = (x - y).dropna()
        ps[other] = float(stats.wilcoxon(dlt, alternative="less").pvalue) if (dlt != 0).any() else 1.0
    order = sorted(ps, key=ps.get)
    adj, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, (len(order) - i) * ps[k])
        adj[k] = min(1.0, run)
    H["H2"] = dict(p=ps, p_holm=adj, mean_capture={k: float(piv[k].mean()) for k in SYN},
                   supported=all(v < 0.05 for v in adj.values()))
    t_tail = tau_by_cell(p, "tail")
    both = pd.concat([t_tail, -t_mae], axis=1, keys=["tail", "negmae"]).dropna()
    r = np.random.default_rng(2)
    diffs = []
    for _ in range(10000):
        s = both.sample(len(both), replace=True, random_state=int(r.integers(1e9)))
        diffs.append(s["tail"].mean() - s["negmae"].mean())
    diffs = np.array(diffs)
    H["H3"] = dict(mean_tau_tail=float(t_tail.mean()), mean_tau_mae=float(t_mae.mean()), n=int(len(both)),
                   p_one_sided=float((np.sum(diffs <= 0) + 1) / (diffs.size + 1)))
    H["H3"]["supported"] = H["H3"]["p_one_sided"] < 0.05
    h4 = {}
    for ba, g in p.groupby("ba"):
        a = g[g.forecaster == "F_ISO"]
        b = g[g.forecaster == "F_ml"]
        h4[ba] = dict(iso_hits=int(a.hits.sum()), ml_hits=int(b.hits.sum()), n_cp=int(a.n_cp.sum()))
    H["H4"] = dict(per_market=h4, at_least_all=all(v["iso_hits"] >= v["ml_hits"] for v in h4.values()),
                   more_in=int(sum(v["iso_hits"] > v["ml_hits"] for v in h4.values())))
    H["H4"]["supported"] = H["H4"]["at_least_all"] and H["H4"]["more_in"] >= min(3, len(h4))
    return H


def summary(P):
    out = {}
    for (scale, mult, w), g in P.groupby(["scale", "mult", "window"]):
        key = f"{scale}_m{mult}_w{w}"
        out[key] = {f: dict(capture=float(h.capture.mean()), alerts=float(h.alerts.mean()),
                            mae_pct=float(h["mae_pct"].mean()), tail=float(h["tail"].mean()))
                    for f, h in g.groupby("forecaster")}
    return out


if __name__ == "__main__":
    main()
