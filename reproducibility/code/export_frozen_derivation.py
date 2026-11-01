"""Mechanically export selected native OMML from the frozen 1025 DOCX.

This is a source transcription tool, not a mathematical repair tool.  In
particular, nested accents/subscripts, source signs, and inconsistent factors
are retained.  Unsupported mathematical nodes raise an exception instead of
being silently discarded.  Only audit/frozen_derivation.tex and .md are written.

The generated TeX contains definitions, not a manuscript.  In an amsmath /
amssymb document, use \\FrozenMath{P1002M01} inside math mode, or
\\FrozenParagraph{P1002} in text mode.  Chinese source text requires a suitable
Unicode/CJK document setup.  With XeLaTeX, use fontspec's no-math option so that
text fonts do not replace legacy math slots.  Identifiers use zero-based indexes.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import zipfile
from pathlib import Path

from lxml import etree as ET


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}
DEFAULT_SOURCE = Path("E:/1030/推理草稿5月1025.docx")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "audit"

# The selections retain the derivation route and the source passages needed to
# explain its current limits.  Including a passage does not certify its proof.
GROUPS = [
    ("physical_model", "物理模型与有限维系统", [75, 76, 77, 96, 97, 98, 99, 115,
                                                *range(120, 130), 144]),
    ("hamiltonian_balance", "动量、Hamiltonian与能量收支", list(range(149, 171))),
    ("complex_source", "复值表示原文（包含待核对环节）", list(range(336, 368)) +
     [412, 414, 415, 416, 417, 418, 421, 423, 424, 425, 429, 430]),
    ("action_angle_source", "作用角与非退化条件原文", [434, 435, 436, 437, 439,
     441, 472, 476, 477, 478, 479, 480, 483, 485, 486, 487, 488, 490, 492,
     493, 494, 495, 496, 498, 500, 501, 502, 503, 504, 505, 507, 508,
     510, 513, 514]),
    ("birkhoff_source", "Birkhoff表示及Theorem A、B原文（需条件核验）",
     list(range(792, 820)) + list(range(837, 888))),
    ("fixed_scheme", "式（30）至（36）的原始推导", list(range(908, 1006))),
]

MATH_CHARS = {
    "−": "-", "–": "-", "∂": r"\partial ", "∇": r"\nabla ",
    "α": r"\alpha ", "β": r"\beta ", "γ": r"\gamma ",
    "δ": r"\delta ", "ε": r"\varepsilon ", "ζ": r"\zeta ",
    "η": r"\eta ", "θ": r"\theta ", "κ": r"\kappa ",
    "λ": r"\lambda ", "μ": r"\mu ", "ν": r"\nu ",
    "ξ": r"\xi ", "π": r"\pi ", "ρ": r"\rho ",
    "σ": r"\sigma ", "τ": r"\tau ", "φ": r"\phi ",
    "ϕ": r"\varphi ", "ω": r"\omega ", "Δ": r"\Delta ",
    "Γ": r"\Gamma ", "Ω": r"\Omega ", "ℋ": r"\mathcal{H}",
    "ℝ": r"\mathbb{R}", "ℂ": r"\mathbb{C}", "ℤ": r"\mathbb{Z}",
    "ℕ": r"\mathbb{N}", "∞": r"\infty ", "∈": r"\in ",
    "∉": r"\notin ", "∀": r"\forall ", "∃": r"\exists ",
    "≠": r"\ne ", "≤": r"\le ", "≥": r"\ge ",
    "≈": r"\approx ", "≡": r"\equiv ", "⇒": r"\Rightarrow ",
    "⟹": r"\Longrightarrow ", "⇔": r"\Leftrightarrow ",
    "⟺": r"\Longleftrightarrow ", "→": r"\to ", "⟶": r"\longrightarrow ",
    "↦": r"\mapsto ", "⋅": r"\cdot ", "×": r"\times ",
    "∧": r"\wedge ", "⋯": r"\cdots ", "…": r"\ldots ",
    "∗": r"\ast ", "★": r"\star ", "∘": r"\circ ",
    "∶": ":", "≔": ":=", "≜": r"\triangleq ",
    "∑": r"\sum ", "∫": r"\int ", "∏": r"\prod ",
    "⊂": r"\subset ", "⊆": r"\subseteq ", "∥": r"\Vert ",
    "‖": r"\Vert ", "⟨": r"\langle ", "⟩": r"\rangle ",
    "\u2009": r"\,", "\u2003": r"\quad ", "\u2002": r"\enspace ",
    "\u2006": r"\,", "\u00a0": r"\,", "\u202f": r"\,",
    "\u2004": r"\;", "\u2005": r"\;", "\u200b": "",
}
TEXT_ESCAPES = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
    "$": r"\$", "&": r"\&", "%": r"\%", "#": r"\#",
    "_": r"\_", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}",
}


def lname(node: ET._Element) -> str:
    return ET.QName(node).localname


def namespace(node: ET._Element) -> str:
    return ET.QName(node).namespace or ""


def text_tex(value: str) -> str:
    return "".join(TEXT_ESCAPES.get(c, c) for c in value)


def math_text(value: str) -> str:
    output: list[str] = []
    prose: list[str] = []

    def flush() -> None:
        if prose:
            output.append(r"\text{" + text_tex("".join(prose)) + "}")
            prose.clear()

    for c in value:
        if c in MATH_CHARS:
            flush()
            output.append(MATH_CHARS[c])
        elif ord(c) > 127 and not c.isspace():
            # Native Chinese comments / punctuation remain exactly as source.
            prose.append(c)
        else:
            flush()
            if c in "{}$%#&_":
                output.append("\\" + c)
            elif c == "\\":
                output.append(r"\backslash ")
            elif c == "^":
                output.append(r"\text{\textasciicircum{}}")
            elif c == "~":
                output.append(r"\sim ")
            else:
                output.append(c)
    flush()
    return "".join(output)


def prop(node: ET._Element, path: str, default: str = "") -> str:
    found = node.find(path, NS)
    return default if found is None else found.get("{" + NS["m"] + "}val", default)


def part(node: ET._Element, name: str) -> str:
    found = node.find("m:" + name, NS)
    return "" if found is None else omml(found)


def delimiter(value: str) -> str:
    return {"": ".", "{": r"\{", "}": r"\}",
            "⟨": r"\langle", "⟩": r"\rangle",
            "‖": r"\Vert", "∥": r"\Vert"}.get(value, value)


def omml(node: ET._Element) -> str:
    tag = lname(node)
    if tag.endswith("Pr") or tag in {"ctrlPr", "mcs", "mc", "mcJc", "count"}:
        return ""
    if namespace(node) == NS["w"]:
        if tag == "t":
            return r"\text{" + text_tex(node.text or "") + "}"
        if tag in {"bookmarkStart", "bookmarkEnd", "proofErr"}:
            return ""
        raise ValueError(f"Unexpected Word node inside OMML: {tag}")
    if tag == "t":
        return math_text(node.text or "")
    if tag == "r":
        body = "".join(omml(c) for c in node if lname(c) != "rPr")
        style = prop(node, "m:rPr/m:sty", "i")
        script = prop(node, "m:rPr/m:scr", "roman")
        command = {"script": "mathcal", "double-struck": "mathbb",
                   "fraktur": "mathfrak", "sans-serif": "mathsf",
                   "monospace": "mathtt"}.get(script)
        if command:
            body = "\\" + command + "{" + body + "}"
        if style == "p":
            return r"\mathrm{" + body + "}"
        if style in {"b", "bi"}:
            return r"\boldsymbol{" + body + "}"
        return body
    if tag == "f":
        ftype = prop(node, "m:fPr/m:type", "bar")
        if ftype == "noBar":
            return r"\genfrac{}{}{0pt}{}{" + part(node, "num") + "}{" + part(node, "den") + "}"
        if ftype in {"lin", "skw"}:
            # Grouping preserves the numerator and denominator of inline fractions.
            return "{" + part(node, "num") + "}/{" + part(node, "den") + "}"
        return r"\frac{" + part(node, "num") + "}{" + part(node, "den") + "}"
    if tag == "sSub":
        return "{" + part(node, "e") + "}_{" + part(node, "sub") + "}"
    if tag == "sSup":
        return "{" + part(node, "e") + "}^{" + part(node, "sup") + "}"
    if tag == "sSubSup":
        return "{" + part(node, "e") + "}_{" + part(node, "sub") + "}^{" + part(node, "sup") + "}"
    if tag == "sPre":
        return "{}_{" + part(node, "sub") + "}^{" + part(node, "sup") + "}{" + part(node, "e") + "}"
    if tag == "rad":
        degree = part(node, "deg")
        index = "" if prop(node, "m:radPr/m:degHide", "0") == "1" or not degree else "[" + degree + "]"
        return r"\sqrt" + index + "{" + part(node, "e") + "}"
    if tag == "acc":
        char = prop(node, "m:accPr/m:chr", "\u0302")
        command = {"\u0302": "hat", "\u0307": "dot", "\u0308": "ddot",
                   "\u0304": "bar", "\u0305": "overline", "\u0303": "widetilde",
                   "\u20d7": "vec", "\u0306": "breve", "\u030c": "check"}.get(char)
        if command is None:
            raise ValueError(f"Unsupported accent character U+{ord(char):04X}")
        return "\\" + command + "{" + part(node, "e") + "}"
    if tag == "bar":
        command = "underline" if prop(node, "m:barPr/m:pos", "top") == "bot" else "overline"
        return "\\" + command + "{" + part(node, "e") + "}"
    if tag == "d":
        begin = delimiter(prop(node, "m:dPr/m:begChr", "("))
        end = delimiter(prop(node, "m:dPr/m:endChr", ")"))
        separator = prop(node, "m:dPr/m:sepChr", "|")
        content = (r"\middle" + delimiter(separator) + " ").join(
            omml(c) for c in node if lname(c) == "e")
        return r"\left" + begin + " " + content + r"\right" + end
    if tag == "m":
        rows = []
        for row in node.findall("m:mr", NS):
            rows.append(" & ".join(omml(c) for c in row.findall("m:e", NS)))
        return r"\begin{matrix}" + r" \\ ".join(rows) + r"\end{matrix}"
    if tag == "eqArr":
        return r"\begin{gathered}" + r" \\ ".join(
            omml(c) for c in node.findall("m:e", NS)) + r"\end{gathered}"
    if tag == "nary":
        char = prop(node, "m:naryPr/m:chr", "∫")
        operator = {"∫": r"\int", "∑": r"\sum", "∏": r"\prod",
                    "∬": r"\iint", "∭": r"\iiint", "∮": r"\oint"}.get(char)
        if operator is None:
            raise ValueError(f"Unsupported n-ary operator: {char!r}")
        lower, upper = part(node, "sub"), part(node, "sup")
        if lower and prop(node, "m:naryPr/m:subHide", "0") != "1":
            operator += "_{" + lower + "}"
        if upper and prop(node, "m:naryPr/m:supHide", "0") != "1":
            operator += "^{" + upper + "}"
        return operator + " " + part(node, "e")
    if tag in {"limUpp", "limLow"}:
        command = "overset" if tag == "limUpp" else "underset"
        return "\\" + command + "{" + part(node, "lim") + "}{" + part(node, "e") + "}"
    if tag == "groupChr":
        char = prop(node, "m:groupChrPr/m:chr", "⏟")
        command = {"⏟": "underbrace", "⏞": "overbrace"}.get(char)
        if command is None:
            if char == "\u0302" and prop(node, "m:groupChrPr/m:pos", "bot") == "top":
                return r"\hat{" + part(node, "e") + "}"
            position = prop(node, "m:groupChrPr/m:pos", "bot")
            command = "overset" if position == "bot" else "underset"
            return "\\" + command + "{" + part(node, "e") + "}{" + math_text(char) + "}"
        return "\\" + command + "{" + part(node, "e") + "}"
    if tag == "func":
        return part(node, "fName") + " " + part(node, "e")
    if tag in {"oMath", "oMathPara", "e", "num", "den", "sub", "sup",
               "deg", "lim", "box", "fName"}:
        return "".join(omml(c) for c in node)
    raise ValueError(f"Unsupported mathematical node: {tag}")


def source_text(paragraph: ET._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text() | .//m:t/text()", namespaces=NS))


def export(source: Path, output: Path) -> dict:
    source_bytes = source.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    with zipfile.ZipFile(source) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    paragraphs = root.xpath("//w:body//w:p", namespaces=NS)
    tex = ["% Mechanical transcription; no mathematical corrections.",
           f"% Source DOCX SHA256: {digest}",
           r"\providecommand{\FrozenMath}[1]{\csname FrozenMath#1\endcsname}",
           r"\providecommand{\FrozenParagraph}[1]{\csname FrozenParagraph#1\endcsname}",
           r"\providecommand{\FrozenDerivationAll}{}"]
    md = ["# 冻结推导机械转写", "", f"源文件：`{source.as_posix()}`",
          f"源文件SHA256：`{digest}`", "",
          "P编号为Word正文段落的零基索引，不是页码。公式仅转写原生OMML；",
          "重复横线、异常下标、系数和符号均按源结构保留。数学正确性另见审查报告。",
          "TeX文件只定义宏，不自动插入正文。须使用amsmath、amssymb及中文兼容排版环境；",
          "XeLaTeX使用fontspec时建议加no-math选项，以免文本字体替代数学符号槽位。",
          ""]
    emitted: set[int] = set()
    count_math = 0
    all_paragraphs: list[str] = []
    for group_id, group_title, indexes in GROUPS:
        md += [f"## {group_title}", ""]
        tex.append("% GROUP: " + group_id)
        for index in indexes:
            if index in emitted:
                continue
            paragraph = paragraphs[index]
            plain = source_text(paragraph)
            if not plain.strip():
                continue
            emitted.add(index)
            identifier = f"P{index:04d}"
            xml_hash = hashlib.sha256(ET.tostring(paragraph, method="c14n")).hexdigest()
            pieces: list[str] = []
            paragraph_math: list[tuple[str, str]] = []

            def visit(node: ET._Element) -> None:
                nonlocal count_math
                tag = lname(node)
                if namespace(node) == NS["m"] and tag == "oMath":
                    mid = identifier + f"M{len(paragraph_math)+1:02d}"
                    converted = omml(node)
                    paragraph_math.append((mid, converted))
                    pieces.append(r"\(\FrozenMath{" + mid + r"}\)")
                    count_math += 1
                    return
                if namespace(node) == NS["w"] and tag == "t":
                    pieces.append(text_tex(node.text or ""))
                    return
                if tag.endswith("Pr") or tag in {"ctrlPr", "bookmarkStart", "bookmarkEnd", "proofErr"}:
                    return
                for child in node:
                    visit(child)

            visit(paragraph)
            tex.append(f"% {identifier}; canonical paragraph XML SHA256: {xml_hash}")
            for mid, converted in paragraph_math:
                tex.append(r"\expandafter\def\csname FrozenMath" + mid + r"\endcsname{" + converted + "}")
            paragraph_tex = "".join(pieces)
            tex.append(r"\expandafter\def\csname FrozenParagraph" + identifier + r"\endcsname{" + paragraph_tex + "}")
            all_paragraphs.append(r"\par\noindent\texttt{" + identifier + r"}\quad\FrozenParagraph{" + identifier + "}")
            md += [f"### {identifier}", "", f"段落XML SHA256：`{xml_hash}`", "",
                   "原文线性文本（仅供定位，不能取代公式结构）：", "", plain, ""]
            for mid, converted in paragraph_math:
                md += [f"数学片段 `{mid}`：", "", "```latex", converted, "```", ""]
    tex.append(r"\renewcommand{\FrozenDerivationAll}{" + "\n".join(all_paragraphs) + "}")
    tex.append("")
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen_derivation.tex").write_text("\n".join(tex), encoding="utf-8", newline="\n")
    (output / "frozen_derivation.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    # Verify that reading/exporting never changed the frozen source.
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    return {"source_sha256": digest, "paragraphs": len(emitted),
            "math_fragments": count_math, "unsupported_nodes": 0,
            "outputs": [str(output / "frozen_derivation.tex"), str(output / "frozen_derivation.md")]}


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output), ensure_ascii=False, indent=2))
