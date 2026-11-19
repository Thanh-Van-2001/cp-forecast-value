"""Build refs.bib from Crossref BibTeX for DOIs (fails loudly if a DOI does
not resolve) and print each title for a by-eye check; arXiv entries are
checked against the arXiv API. Run: python build_bib.py <out.bib>"""
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "ref-check/0.1 (academic reference verification)"}
DOIS = {
    "carmona2026cp": "10.1007/s12667-026-00788-1",
    "liu2013datacenter": "10.1016/j.peva.2013.08.014",
    "ruggles2020eia": "10.1038/s41597-020-0483-x",
    "albadi2008dr": "10.1016/j.epsr.2008.04.002",
    "hong2016probabilistic": "10.1016/j.ijforecast.2015.11.011",
    "hong2014gefcom": "10.1016/j.ijforecast.2013.07.001",
    "gneiting2011point": "10.1198/jasa.2011.r10138",
    "murphy1993good": "10.1175/1520-0434(1993)008<0281:wiagfa>2.0.co;2",
    "hyndman2006another": "10.1016/j.ijforecast.2006.03.001",
    "elmachtoub2022spo": "10.1287/mnsc.2020.3922",
    "kendall1945ties": "10.1093/biomet/33.3.239",
    "wilcoxon1945individual": "10.2307/3001968",
    "efron1993bootstrap": "10.1201/9780429246593",
    "zarnikau2013response": "10.1016/j.jup.2013.04.004",
    "skiles2023peak": "10.1016/j.tej.2023.107254",
}
ARXIV = {
    "zhang2026cp": "2605.16794",
}
EXTRA = r"""
@misc{zhang2026cp,
  author = {Zhang, Qian and Zhao, Sadie and Diao, Lucy and Byers, Conleigh and Chen, Yiling and Cansever, Derya and Xie, Le},
  title = {Modeling coincident peak pricing in electricity markets: challenges and peak shaving effectiveness},
  year = {2026},
  eprint = {2605.16794},
  archivePrefix = {arXiv},
  note = {arXiv preprint arXiv:2605.16794}
}
@inproceedings{ke2017lightgbm,
  author = {Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan},
  title = {{LightGBM}: A highly efficient gradient boosting decision tree},
  booktitle = {Advances in Neural Information Processing Systems 30},
  pages = {3146--3154},
  year = {2017}
}
@article{holm1979simple,
  author = {Holm, Sture},
  title = {A simple sequentially rejective multiple test procedure},
  journal = {Scandinavian Journal of Statistics},
  volume = {6},
  number = {2},
  pages = {65--70},
  year = {1979}
}
@misc{eia930,
  author = {{U.S. Energy Information Administration}},
  title = {Hourly Electric Grid Monitor ({Form EIA-930}): six-month balance files},
  year = {2026},
  note = {Accessed November 2026}
}
@misc{puct25192,
  author = {{Public Utility Commission of Texas}},
  title = {Substantive Rule {\S}25.192: Transmission service rates},
  year = {2026},
  note = {Texas Administrative Code, Title 16, Chapter 25}
}
@misc{pjm_m19,
  author = {{PJM Interconnection}},
  title = {{PJM} Manual 19: Load Forecasting and Analysis},
  year = {2026}
}
"""
TITLE_FIX = {}
def get(url, tries=5):
    for t in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code != 429 or t == tries - 1:
                raise
            time.sleep(20 * (t + 1))


out = []
for key, doi in DOIS.items():
    bib = get(f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex")
    bib = re.sub(r"^@(\w+)\{[^,]+,", lambda m: "@" + m.group(1) + "{" + key + ",", bib.strip(), count=1)
    bib = re.sub(r"\s*url=\{[^}]*\},", "", bib)
    bib = re.sub(r",\s*month=\w+", "", bib)  # Crossref month macros are not defined in every style
    for a, b in TITLE_FIX.items():
        bib = bib.replace(a, b)
    t = re.search(r"title=\{(.*?)\},", bib)
    print(f"{key:28s} {t.group(1)[:90] if t else '??'}")
    out.append(bib)
    time.sleep(0.5)
ids = ",".join(ARXIV.values())
feed = ET.fromstring(get(f"http://export.arxiv.org/api/query?id_list={ids}&max_results=10"))
ns = {"a": "http://www.w3.org/2005/Atom"}
for e in feed.findall("a:entry", ns):
    print("arXiv", e.find("a:id", ns).text.split("/")[-1], "|", " ".join(e.find("a:title", ns).text.split())[:90],
          "|", ", ".join(a.find("a:name", ns).text for a in e.findall("a:author", ns))[:80])
open(sys.argv[1], "w", encoding="utf-8").write("\n\n".join(out) + "\n" + EXTRA)
print(len(out) + EXTRA.count("\n@"), "entries written")
