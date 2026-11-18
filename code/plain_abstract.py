"""Plain-text abstract (macros expanded) for submission portals.
Run from the repository root: python -I code/plain_abstract.py"""
import json
import re

M = json.load(open("out/macros.json"))
t = open("paper/main.tex", encoding="utf-8").read()
start = t.index(r"\begin{abstract}") + len(r"\begin{abstract}")
end = t.index(r"\medskip\noindent\textbf{Keywords")
a = t[start:end]
a = re.sub(r"\\res\{([^}]+)\}", lambda m: M[m.group(1)], a)
a = a.replace(r"$\tau=", "tau = ").replace("$", "").replace("--", "\u2013").replace(r"\%", "%")
a = re.sub(r"\s+", " ", a).strip().replace("-0.", "\u22120.")
open("paper/submission/abstract_plain.txt", "w", encoding="utf-8").write(a)
kw = t[end:].split("}", 1)[1].split("\\end{abstract}")[0].strip()
print(len(a.split()), "words")
print(a)
print("KEYWORDS:", kw)
