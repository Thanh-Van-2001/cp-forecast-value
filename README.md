# Coincident-peak capture vs load-forecast accuracy

Replication package for *Does a more accurate load forecast catch more
coincident peaks? A pre-specified evaluation in four US electricity markets*.

- `PLAN.md`: pre-specified analysis plan and amendment log. The git history
  shows the plan was committed before the full data download and analysis.
- `code/download_eia930.sh`: downloads the public EIA-930 six-month balance
  files (atomic writes, SHA-256 list in `out/SHA256SUMS`).
- `code/extract.py`: keeps ERCOT, PJM, NYISO and ISO-NE demand and day-ahead
  forecasts.
- `code/cp_study.py`: forecasts, same-MAE synthetic forecasts, alert policies,
  CP definitions, hypotheses H1-H4 (`--check` runs the ERCOT 2018-2019
  pipeline check).
- `code/analyze2.py`: secondary configurations; `code/make_macros.py` and
  `code/make_figures.py`: every number and figure in the paper.
- `refs/build_bib.py`: builds `paper/refs.bib` from Crossref and arXiv.
- `out/`: results (`results.json`, `panel.csv`, `secondary.json`); `out/v1/`:
  the run before amendment 3 (screening of recording errors).

## Run

```
bash code/download_eia930.sh                 # about 0.9 GB
python -I code/extract.py .
python -I code/cp_study.py .                 # about 10 minutes
python -I code/analyze2.py
python -I code/make_macros.py && python -I code/make_figures.py
```

Python 3.12 with numpy, pandas, scipy, lightgbm and matplotlib.

## Licence

MIT (code). EIA-930 data are public U.S. government data.
