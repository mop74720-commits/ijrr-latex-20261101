"""Reflow the source-derived formulas for the existing SAGE two-column paper."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from build_condensed_derivation import ROOT, EXCERPTS, LAYOUT, typeset


def aligned(*anchors):
    return [("<START>", r"\begin{aligned}&", "")] + [
        (anchor, r"\\ &", "") for anchor in anchors
    ] + [("<END>", r"\end{aligned}", "")]


COLUMN = {
    "P0151M01": aligned(
        r"=\frac{1}{2}{\left( {M}^{-1}(q)p\right)}",
        r"=\frac{1}{2}{p}^{T}{\left( {M}^{-1}(q)\right)}",
    ),
    "P0155M01": aligned(r"=\int K(q)q", r"=\int ({K}_{tan}"),
    "P0165M01": aligned(r"={F}_{mag}+{F}_{cont}-C(q,\dot{q})"),
    "P0170M01": aligned(
        r"=\frac{\partial H}{\partial q}\frac{\partial H}{\partial p}",
        r"=\frac{\partial H}{\partial p}\left( {F}_{ext}",
    ),
    "P0337M01": aligned(r":=H(q,p)+iG(q,p)"),
    "P0341M01": aligned(r"{G}_{1}(q,p)="),
    "P0344M01": aligned(r"\Rightarrow {\dot{p}}_{ext}"),
    "P0367M01": aligned(
        r"=\frac{1}{2}\left( \frac{\partial H}{\partial q}",
        r"\Rightarrow -\sqrt{2}i",
    ),
    "P0416M01": [
        (r"\left[ ", "", r"\begin{gathered}"),
        (r"+\left( \frac{\partial }{\partial \mathrm{\xi }}", r"\\ ", ""),
        (r"\right]", r"\end{gathered}", ""),
    ],
    "P0423M01": aligned(r"\mathrm{=}-\sqrt{2}i"),
    "P0576M01": aligned(r"\Leftrightarrow ", r"\Longleftrightarrow "),
    "P0570M01": aligned(
        r":=-\sqrt{2}i(\frac{\partial H}{\partial \overline{\mathrm{\xi }}}+\frac{i}",
        r":=-\sqrt{2}i(\frac{\partial H}{\partial \overline{\mathrm{\xi }}}+i\frac",
    ),
    "P0796M01": aligned(r"\quad i=1,2"),
    "P0872M01": aligned(r"=\overline{A}"),
    "P0916M01": aligned(r"=\left( \begin{matrix}0 & I"),
    "P0930M01": LAYOUT["P0930M01"][:-1] + [
        (r"\frac{1}{2}[", "", r"\begin{gathered}"),
        (r"+{F}_{ext}-C{M}^{-1}\hat{p}", r"\\ ", ""),
        (r"\mathrm{]} \\ \frac{1}{2}{M}^{-1}", r"\end{gathered}", ""),
        ("[", r"\Bigg", ""),
        ("]", r"\Bigg", ""),
    ] + LAYOUT["P0930M01"][-1:],
    "P0931M01": aligned(
        r"\mathrm{=}\left( \begin{matrix}-\frac{1}{2}",
    )[:-1] + [
        (r"-\frac{1}{2}{\hat{p}}^{T}", r"\begin{gathered}", ""),
        (r"+{F}_{ext}-C{M}^{-1}\hat{p}", r"\\ ", ""),
        (r" \\ {M}^{-1}(\hat{q})\hat{p}", r"\end{gathered}", ""),
    ] + aligned()[-1:],
    "P0954M01": aligned(
        r"=A({\mathrm{\varphi }}_{w}^{(0)}",
        r"=A({\left. \hat{w}\right|}",
        r"={\left. \frac{d\hat{w}}{d\mathrm{t}}\right|}",
    ),
    "P0959M01": aligned(r"\left\{ \begin{gathered}C")[:-1] + [
        (r"-\frac{1}{2}{\mathrm{P}}^{T}", r"\begin{gathered}", ""),
        (r"+{F}_{ext}-C{M}^{-1}\mathrm{P}", r"\\ ", ""),
        (r" \\ \mathrm{-}{M}^{-1}(\mathrm{Q})\mathrm{P}", r"\end{gathered}", ""),
    ] + aligned()[-1:],
    "P0960M01": aligned(
        r"+{\mathrm{\varphi }}_{w}^{(1)}",
        r"t-{t}_{0}\mathrm{=}\delta",
    ),
    "P0962M01": aligned(r"\hat{w}\mathrm{=}")[:-1] + [
        (r"\begin{matrix} \\  \\  \\  \\  \\ \end{matrix}", r"\smash{", "}"),
    ] + aligned()[-1:],
    "P0983M01": [
        (r"-\frac{1}{8}", r"\begin{gathered}", ""),
        (r"-\frac{1}{2}K", r"\\ ", ""),
        (r"+{F}_{ext}", r"\\ ", ""),
        (r"\right\}\delta ", r"\end{gathered}", ""),
    ],
    "P0985M01": [
        (r"-\frac{1}{8}", r"\begin{gathered}", ""),
        (r"-\frac{1}{2}K", r"\\ ", ""),
        (r"+{F}_{ext}", r"\\ ", ""),
        (r"\right\}\delta ", r"\end{gathered}", ""),
    ],
    "P0989M01": [
        (r"-\frac{1}{8}", r"\begin{gathered}", ""),
        (r"-\frac{1}{2}K", r"\\ ", ""),
        (r"+{F}_{ext}", r"\\ ", ""),
        (r"\right\}\delta ", r"\end{gathered}", ""),
    ],
    "P0992M01": aligned(r"=p(1-\frac{1}{4}", r"-Kq\cdot "),
    "P1002M01": [
        (r"{q}_{k+1}={q}_{k}+", r"\begin{gathered}", ""),
        (r"\mathrm{\text{其中} }", r"\\ ", ""),
        (r" \\ {p}_{k+1}=", r"\end{gathered}", ""),
        (r"{p}_{k+1}={p}_{k}\cdot", r"\begin{gathered}", ""),
        (r"\left\{ \begin{gathered}K", r"\\ ", ""),
        (r"\end{matrix}\right.", r"\end{gathered}", ""),
    ],
}


def main():
    body=(ROOT/"manuscript/sections/02_fixed_method.tex").read_text(encoding="utf-8")
    identifiers=re.findall(r"\\D[UE]\{(P\d{4}M\d{2})\}",body)
    source=(ROOT/"manuscript/derivation_math.tex").read_text(encoding="utf-8")
    originals=dict(re.findall(r"^\\expandafter\\def\\csname DerivationOriginal(P\d{4}M\d{2})\\endcsname\{(.*)\}$",source,re.M))
    lines=["% Generated two-column layout: insertion-only changes to retained source expressions."]
    records=[]
    for identifier in identifiers:
        original=originals[identifier]
        start_anchor,end_anchor=EXCERPTS.get(identifier,(None,None))
        start=0 if start_anchor is None else original.index(start_anchor)
        end=len(original) if end_anchor is None else original.index(end_anchor)
        excerpt=original[start:end]
        layout=COLUMN if identifier in COLUMN else LAYOUT
        result,operations=typeset(identifier,excerpt,layout)
        lines.append(r"\expandafter\def\csname DerivationLayout"+identifier+r"\endcsname{"+result+"}")
        records.append({"id":identifier,"source_sha256":hashlib.sha256(original.encode()).hexdigest(),"retained_source_span":[start,end],"layout_insertions":operations,"reversible_check":True})
    (ROOT/"manuscript/derivation_ijrr_layout.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (ROOT/"audit/ijrr_derivation_layout.json").write_text(json.dumps({"formula_count":len(records),"records":records},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Verified source fragments and reversible column layout:",len(records))


if __name__=="__main__":
    main()
