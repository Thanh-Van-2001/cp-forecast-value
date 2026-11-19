"""Plain-text abstract and keywords (macros expanded) for submission portals.
Run from the repository root: python -I code/plain_abstract.py
Writes paper/submission_tej/abstract_plain.txt (elsarticle layout of the TEJ version)."""
import json
import re

M = json.load(open("out/macros.json"))
t = open("paper/main.tex", encoding="utf-8").read()
a = t[t.index(r"\begin{abstract}") + len(r"\begin{abstract}"):t.index(r"\end{abstract}")]
kw = t[t.index(r"\begin{keyword}") + len(r"\begin{keyword}"):t.index(r"\end{keyword}")]
a = re.sub(r"\\res\{([^}]+)\}", lambda m: M[m.group(1)], a)
a = a.replace(r"$\tau=", "tau = ").replace("$", "").replace("--", "\u2013").replace(r"\%", "%")
a = re.sub(r"\s+", " ", a).strip().replace("-0.", "\u22120.")
kws = [k.strip() for k in kw.split(r"\sep")]
assert "\\" not in a and "{" not in a, a
open("paper/submission_tej/abstract_plain.txt", "w", encoding="utf-8").write(a + "\n\nKeywords: " + "; ".join(kws) + "\n")
print(len(a.split()), "words")
print(a)
print("KEYWORDS:", kws)
