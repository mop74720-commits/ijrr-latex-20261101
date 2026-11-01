"""Export only the original mathematical fragments used by the condensed section.

The authoritative DOCX, the earlier frozen macros and the delivered manuscript
are read-only. Display transformations insert layout commands; they never repair
a sign, coefficient, dependency, or mathematical assertion.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

from lxml import etree as ET

from export_frozen_derivation import NS, omml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "audit/source_1025_unchanged.docx"
BODY = ROOT / "manuscript/sections/02_derivation_condensed.tex"
EXPECTED_SOURCE = "8c62b874caaa6362b0f1c5612ddd6514aa7141454b17ef4cf5688408fe307338"
EXPECTED_FROZEN = "3e22b9c936f6734d60a66c78d0f191a3313fe2427f7b0971cde86ac9f686b883"

# Four contiguous excerpts remove only unused framing: references to omitted
# local labels, a draft Jacobian placeholder, the unused general K^-1 D term,
# and the already omitted constant-coefficient branch. The source expressions
# are kept intact in DerivationOriginal and the audit map records both bounds.
EXCERPTS = {
    "P0846M01": (r"\left\{", None),
    "P0857M01": (r"\left(", r"\mathrm{\text{，}}"),
    "P0873M01": (None, r"=\left( {A}_{\alpha }"),
    "P0994M01": (r"\mathrm{\varepsilon }=", r",\mathrm{\text{则}}"),
}

# Insertions are applied to exact substrings of a mechanically transcribed
# expression. Empty anchors at start/end allow an aligned environment wrapper.
# No replacement or deletion is allowed in this layout-only table.
LAYOUT: dict[str, list[tuple[str, str, str]]] = {
    "P0161M01": [
        ("<START>", r"\begin{aligned}&", ""),
        (r"=-\frac{1}{2}{p}^{T}\left[ \frac{\partial {M}^{-1}(q)}{\partial q}\right]p-\frac{dV(q)}{dq}", r"\\ &", ""),
        (r"=-\frac{1}{2}{p}^{T}\left[ \frac{\partial {M}^{-1}(q)}{\partial q}\right]p-K(q)q", r"\\ &", ""),
        ("<END>", r"\end{aligned}", ""),
    ],
    "P0357M01": [
        ("<START>", r"\begin{aligned}&", ""),
        (r"=-\frac{i}{2}(d\mathrm{\xi }+d\overline{\mathrm{\xi }})", r"\\ &", ""),
        (r"=-\frac{i}{2}(d\mathrm{\xi }\wedge d\mathrm{\xi }", r"\\ &", ""),
        (r"=-\frac{i}{2}(-d\mathrm{\xi }\wedge", r"\\ &", ""),
        ("<END>", r"\end{aligned}", ""),
    ],
    "P0366M01": [
        ("<START>", r"\begin{aligned}&", ""),
        (r"=\frac{1}{\sqrt{2}}\left[ \frac{\partial H}", r"\\ &", ""),
        (r":=\frac{1}{\sqrt{2}}\left[", r"\\ &", ""),
        (r"=\frac{1}{\sqrt{2}}\left[ \left( \frac{\partial H}{\partial p}-i\frac{\partial H}{\partial q}\right)-i{\mathrm{\nabla }}_{p}", r"\\ &", ""),
        ("<END>", r"\end{aligned}", ""),
    ],
    "P0930M01": [
        ("<START>", r"\begin{aligned}&", ""),
        (r"\mathrm{=}\left( \begin{matrix}\frac{1}{2}[-", r"\\ &", ""),
        ("<END>", r"\end{aligned}", ""),
    ],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def typeset(identifier: str, original: str, layout=None) -> tuple[str, list[dict]]:
    displayed = original
    operations = []
    for anchor, before, after in (LAYOUT if layout is None else layout).get(identifier, []):
        if anchor == "<START>":
            position = 0
            displayed = before + after + displayed
        elif anchor == "<END>":
            position = len(displayed)
            displayed += before + after
        else:
            assert displayed.count(anchor) == 1, (identifier, anchor)
            position = displayed.index(anchor)
            displayed = displayed[:position] + before + anchor + after + displayed[position + len(anchor):]
        operations.append({"anchor": anchor, "position": position, "before": before, "after": after})
    # Undo exact insertions, in reverse order, to verify the scientific token
    # stream is unchanged, including all original matrix separators and braces.
    recovered = displayed
    for op in reversed(operations):
        pos, before, after = op["position"], op["before"], op["after"]
        anchor = "" if op["anchor"] in {"<START>", "<END>"} else op["anchor"]
        segment = before + anchor + after
        assert recovered[pos:pos + len(segment)] == segment
        recovered = recovered[:pos] + anchor + recovered[pos + len(segment):]
    assert recovered == original, identifier
    return displayed, operations


def main() -> None:
    assert digest(SOURCE) == EXPECTED_SOURCE
    frozen_path = ROOT / "audit/frozen_derivation.tex"
    assert digest(frozen_path) == EXPECTED_FROZEN
    body = BODY.read_text(encoding="utf-8")
    ordered = re.findall(r"\\D[UE]\{(P\d{4}M\d{2})\}", body)
    assert len(ordered) == len(set(ordered)), "Repeated source fragment in condensed body"
    numbered = re.findall(r"\\DE\{(P\d{4}M\d{2})\}\{(\d+)\}", body)
    source_tex = (ROOT / "manuscript/source_derivation.tex").read_text(encoding="utf-8")
    full_macros = dict(re.findall(r"^\\expandafter\\def\\csname SourceMath(P\d{4}M\d{2})\\endcsname\{(.*)\}$", source_tex, re.M))
    frozen_macros = dict(re.findall(r"^\\expandafter\\def\\csname FrozenMath(P\d{4}M\d{2})\\endcsname\{(.*)\}$", frozen_path.read_text(encoding="utf-8"), re.M))
    with zipfile.ZipFile(SOURCE) as archive:
        tree = ET.fromstring(archive.read("word/document.xml"))
    paragraphs = tree.xpath("//w:body//w:p", namespaces=NS)
    lines = [
        "% Generated by code/build_condensed_derivation.py; original source math only.",
        "% Source DOCX SHA256: " + EXPECTED_SOURCE,
        r"\providecommand{\DerivationOriginal}[1]{\csname DerivationOriginal#1\endcsname}",
        r"\providecommand{\DerivationLayout}[1]{\csname DerivationLayout#1\endcsname}",
    ]
    records = []
    for identifier in ordered:
        paragraph_index = int(identifier[1:5])
        math_index = int(identifier[-2:]) - 1
        paragraph = paragraphs[paragraph_index]
        native_node = paragraph.xpath(".//m:oMath", namespaces=NS)[math_index]
        original = omml(native_node)
        assert original == full_macros[identifier], identifier
        if identifier in frozen_macros:
            assert original == frozen_macros[identifier], identifier
        start_anchor, end_anchor = EXCERPTS.get(identifier, (None, None))
        assert start_anchor is None or original.count(start_anchor) == 1
        assert end_anchor is None or original.count(end_anchor) == 1
        start = 0 if start_anchor is None else original.index(start_anchor)
        end = len(original) if end_anchor is None else original.index(end_anchor)
        excerpt = original[start:end]
        assert excerpt and excerpt == original[start:end]
        displayed, operations = typeset(identifier, excerpt)
        lines += [
            "% Original native mathematical fragment: " + identifier,
            r"\expandafter\def\csname DerivationOriginal" + identifier + r"\endcsname{" + original + "}",
            r"\expandafter\def\csname DerivationLayout" + identifier + r"\endcsname{" + displayed + "}",
        ]
        records.append({
            "id": identifier,
            "xml_paragraph_index": paragraph_index,
            "source_math_c14n_sha256": hashlib.sha256(ET.tostring(native_node, method="c14n")).hexdigest(),
            "latex_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
            "matches_full_transcription": True,
            "matches_frozen_if_present": True,
            "retained_source_span": [start, end],
            "whole_fragment_retained": start == 0 and end == len(original),
            "removed_source_prefix": original[:start],
            "removed_source_suffix": original[end:],
            "layout_insertions_only": operations,
            "reversible_layout_check": True,
        })
    target = ROOT / "manuscript/derivation_math.tex"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    report = {
        "source_sha256": EXPECTED_SOURCE,
        "source_unchanged": digest(SOURCE) == EXPECTED_SOURCE,
        "frozen_math_unchanged": digest(frozen_path) == EXPECTED_FROZEN,
        "mathematical_fragments": len(records),
        "numbered_equations": [{"id": identifier, "original_number": int(number)} for identifier, number in numbered],
        "body_chinese_characters": len(re.findall(r"[\u4e00-\u9fff]", body)),
        "records": records,
    }
    (ROOT / "audit/derivation_revision_map.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: report[key] for key in report if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
