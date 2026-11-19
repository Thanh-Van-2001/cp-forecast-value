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

### Amendment 1 (2026-11-18, pipeline check): code fix only

A summary routine referred to a pandas method instead of the "tail" column;
fixed. No analysis change.

### Amendment 2 (2026-11-18, after the pipeline check, before the full run)

What the check showed: on ERCOT 2018-2019 (the only data run), the policies
calibrated to B = 20 alerts per season on the training seasons issued 24 to
44 alerts in the test seasons, differently for each forecaster, because
ERCOT load grew between the training and test seasons. Capture would then
reflect alert counts more than forecast quality. The capture results of the
check (ERCOT 2018-2019) were also seen; this amendment is motivated by the
alert counts, not by them. Changes:

- Thresholds (both policies) act on daily peaks divided by the mean actual
  load of the previous 365 days, which removes load growth; the running
  maximum uses the same normalized peaks. Window centres and the tail rank
  score use raw forecasts (unchanged).
- Added a secondary "exact-budget" policy: each forecaster alerts on exactly
  the B days of the season with its highest forecast peaks (ERCOT: B/4 per
  month). It uses the season's forecasts ex post and is not a usable rule; it
  isolates how well a forecaster ranks days. H1-H4 are also reported for it,
  as secondary results.
- Alert counts per policy are reported next to every capture figure.

### Amendment 3 (2026-11-18, after the full run): screening of recording errors

The full run showed an implausible error for the ML forecaster in PJM from
2022 (MAE 27-79% of load). A scan of the raw data found recording errors in
EIA-930: PJM demand spikes of 417,669 MW (2019), 262,651 MW (2020) and
2,147,480,000 MW (2021, an integer-overflow sentinel), zero demand hours and
zero forecasts in NYISO, and one spike in ISO-NE (2025). The plan had no
screening rule; such values corrupt model training, lagged forecasts and,
if they fall in summer, the identification of CP hours. Change: before any
step, a demand or forecast value is set missing if it is <= 0 or lies
outside [0.5, 1.5] times the centred 168-hour rolling median of actual
demand (the season-exclusion rule then applies as planned). The full
analysis is rerun; results of the unscreened run are kept in out/v1 and the
differences are reported.

### Amendment 4 (2026-11-18, after an internal referee review of the full results)

Corrections:
- The amendment-3 screen (bounds 0.5-1.5 x weekly median) also removed
  genuine heat-wave peaks (139 ISO-NE values). Revised to remove only gross
  recording errors: <= 0, or outside [0.3, 2.0] x the centred weekly median;
  every removed June-September value is listed in results.json.
- The pre-specified rule "exclude a season if any CP-candidate day lacks
  data" had not been implemented; implemented as: a market-year is excluded
  if any CP day lacks any hour of actual demand or of the ISO forecast.
- Holidays now follow NERC rules (a Saturday holiday is not moved; the code
  had moved it to Friday).
- Synthetic forecasts are assigned exactly the ISO forecast's MAE (they are
  matched by construction; numerical noise had broken the ties in tau).

Exploratory additions (no new hypotheses):
- Day-level capture (CP day alerted, regardless of the hour window), because
  synthetic forecasts built from actual load always place the peak hour
  correctly.
- A forecast-scale policy without the running-maximum condition ("nomax"),
  in which the calibrated threshold binds and alert counts are closer to B;
  the review showed that the running maximum, not q, set the alert counts of
  the primary policy, so capture comparisons there are confounded by alerts.
- H1 on subsets excluding the weakest forecasters, a year-clustered
  bootstrap, per-market H1 for ERCOT and PJM, out-of-sample (previous
  season) tail-rank score and MAE, and exact binomial intervals for H4.
The unscreened run (out/v1) and the amendment-3 run (out/v2) are kept.

### Amendment 5 (2026-11-19, after an editorial desk rejection; logged before the rerun's results were seen)

A separate pre-specified study of EIA-930 recording errors (repository
eia930-quality, exploratory offset analysis) found that PJM's Demand Forecast
column is stamped one hour early relative to its Demand column in every
month from February 2019 (92 of 92 months; MAPE 3.6% unshifted vs 2.4%
when the forecast row one hour earlier is paired with demand, the same as
PJM's 2016-2018 level). No offset was found for ERCO, NYIS or ISNE.

Change: a corrected-data analysis (`--realign-pjm`, results in out_realign/)
delays PJM's forecast by one hour from 2019-02-01 before the screen; every
other step is unchanged (same seed, policies, tests). The pre-specified
results in out/ are kept and reported alongside. Because the offset changes
F_ISO's MAE, it also changes the MAE matched by the synthetic forecasts and
F_ISObc/F_mix for PJM. Both sets of H1-H4 outcomes are reported; neither is
chosen after seeing which is more favourable. The corrected-data analysis is
labelled as such, not as pre-specified.
