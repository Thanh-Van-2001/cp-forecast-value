# Pre-specified analysis plan: is a more accurate load forecast better at catching coincident peaks?

Committed to git before the full data download and before any analysis.
One six-month EIA-930 file (2023 July to December) was downloaded to check
access and the column layout; no statistic was computed from it. Amendments
are appended at the end with dates and reasons.

## Background

Large customers in several US markets pay transmission or capacity charges
set by their own load in a few system peak hours (coincident peaks, CP):
ERCOT 4CP (the peak interval of each month, June to September), PJM 5CP
(the five highest daily peak hours on distinct non-holiday weekdays, June 1
to September 30), NYISO (the summer system peak hour) and ISO-NE (the annual
system peak hour). Customers curtail on days they expect to contain a CP.
Forecasts used for this are chosen and sold on accuracy (MAPE). Closest
prior work: Carmona, Yang and Zeng (arXiv 2407.04081, 2024) predict CP days
from ISO forecasts in NYISO and PJM, but do not compare forecasts or relate
accuracy to CP capture.

## Question

Does a more accurate day-ahead load forecast capture more coincident peaks
for the same number of curtailment alerts, and which kinds of forecast
error cost CP capture?

## Data

EIA-930 hourly balance files (public, eia.gov six-month files), balancing
authorities ERCO (ERCOT), PJM, NYIS (NYISO), ISNE (ISO-NE): "Demand (MW)"
(actual) and "Demand Forecast (MW)" (the authority's own day-ahead
forecast), local hour ending. Period: July 2015 to the latest complete
season. Hours with missing actual demand are left missing; a season of a
market is excluded if more than 2% of its season hours or any CP-candidate
day lack actual demand or forecast (reported).

CP definitions are approximated on hourly data (ERCOT uses 15-minute
intervals; this is a stated limitation):
- ERCOT: for each month June to September, the hour with the highest demand.
- PJM: the five highest daily peak hours on distinct weekdays that are not
  NERC holidays, June 1 to September 30.
- NYISO: the highest hour June 1 to September 30.
- ISO-NE: the highest hour of the calendar year (season for alerts = June 1
  to September 30; if the annual peak falls outside it, the market-year is
  reported separately and excluded from the primary analysis).

Seasons: training from 2016; test seasons 2018 to the latest complete
summer, with expanding training windows (all earlier seasons).

## Forecasters (day-ahead, for each hour of day d, using data up to the end of day d-1)

Real:
- F_ISO: the authority's day-ahead forecast as reported to EIA.
- F_ISObc: F_ISO minus its mean error over the previous 28 days (same hours).
- F_pers: the actual load of the same hour on day d-1.
- F_week: the actual load of the same hour on day d-7.
- F_ml: LightGBM on lagged loads (d-1 same hour, d-1 daily peak, d-7 same
  hour), hour, weekday, day of year and holiday flag; trained on all earlier
  years (no weather data).
- F_mix: the average of F_ISO and F_ml.

Synthetic (built from the actual load; each scaled so that its season MAE
equals F_ISO's season MAE in the same market-year):
- S_bias: actual + constant (over-forecast).
- S_under: actual - constant (under-forecast).
- S_noise: actual + independent Gaussian noise.
- S_ar1: actual + AR(1) noise with hourly autocorrelation 0.9.
- S_shrink: actual - k (actual - 28-day mean of the same hour): extremes pulled
  towards the mean (the error most forecasting models make at peaks).

## Alert policies (online, one decision per day before the day)

A day is alerted if its forecast daily peak P_d is high enough; the customer
then curtails in a window of hours centred on the forecast peak hour (3 hours
primary; 1 and 5 hours secondary). A CP hour is captured if its day was
alerted and the hour lies in the window.

- Primary, forecast-scale policy: alert if P_d >= max(q_f, a M^f_{d-1}),
  where q_f is a quantile of this forecaster's own season-day forecast peaks
  in the training seasons, M^f_{d-1} is the highest forecast peak so far in
  the current CP period (month for ERCOT, season otherwise) and a = 0.97.
  q_f is set on the training seasons so that the policy issues on average
  B alerts per season, with B = 5 x (number of CP events per season):
  ERCOT 20, PJM 25, NYISO 5, ISO-NE 5. Secondary B multipliers: 3 and 8.
- Secondary, actual-scale policy: as above, but q and the running maximum
  use actual loads of past days (what a user does when comparing a vendor
  forecast with observed peaks); here forecast bias matters.

## Outcomes

- Capture: share of a market-year's CP hours captured (primary).
- Alerts issued (reported; budgets are matched only in expectation).
- Accuracy: MAE (as % of mean season load) over season hours; also
  "tail rank score" = share of the season's top-5 actual daily peaks that are
  among the forecaster's top-5 forecast daily peaks.

## Hypotheses (primary configuration: forecast-scale policy, 3-hour window, B multiplier 5, real and synthetic forecasters together)

- H1 (accuracy predicts value). Within each market-year, Kendall tau-b between
  forecaster MAE and capture across the 11 forecasters; mean over market-years
  is negative (lower error, more capture). One-sided test by bootstrap over
  market-years (10,000 replicates), p < 0.05.
- H2 (error type matters). Among the same-MAE synthetic forecasters,
  S_shrink captures fewer CPs than S_bias, S_under and S_noise (paired
  over market-years, one-sided Wilcoxon, Holm across the three comparisons).
- H3 (tail rank beats MAE). The mean within-market-year tau-b between tail
  rank score and capture is larger in magnitude than that for MAE (paired
  bootstrap over market-years, one-sided p < 0.05).
- H4 (the operator's forecast). F_ISO captures at least as many CPs as F_ml
  in every market (pooled over test seasons; reported with exact binomial
  intervals), and more in at least 3 of 4 markets.

Secondary and robustness (no new hypotheses): window 1 and 5 hours; B
multipliers 3 and 8; actual-scale policy (expected: S_bias and S_under lose
capture there); per-market results; excluding 2020 (pandemic loads).

## Workflow

Download on titan (nice 19); a pipeline check on one market and two seasons
precedes the full run and may only fix code. All numbers in the paper come
from result files via generated macros.

## Amendment log

(none yet)
