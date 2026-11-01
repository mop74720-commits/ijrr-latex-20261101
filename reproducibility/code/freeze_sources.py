"""Preserve the user's source documents and their native equation markup."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    manuscript = SOURCE / "推理草稿5月1025.docx"
    paths = [manuscript, SOURCE / "论文格式表达要求.txt",
             SOURCE / "CHNN_KAM_LaTeX工程_20260730.zip"]
    paths += sorted((SOURCE / "草稿涉及文献").rglob("*.pdf"))
    paths += sorted((SOURCE / "ijrr模板").glob("*.zip"))
    records = [{"path": p.as_posix(), "bytes": p.stat().st_size,
                "sha256": digest(p)} for p in paths]
    out = ROOT / "audit"
    snapshot = out / "source_1025_unchanged.docx"
    if snapshot.exists() and digest(snapshot) != digest(manuscript):
        raise RuntimeError("Existing snapshot differs; refusing to overwrite it.")
    if not snapshot.exists():
        shutil.copyfile(manuscript, snapshot)
    with zipfile.ZipFile(manuscript) as archive:
        xml = archive.read("word/document.xml")
    tree = etree.fromstring(xml)
    paragraphs = []
    for index, paragraph in enumerate(tree.xpath("//w:body//w:p", namespaces=NS)):
        text = "".join(paragraph.xpath(".//w:t/text() | .//m:t/text()", namespaces=NS))
        if not text:
            continue
        paragraphs.append({
            "xml_paragraph_index": index,
            "plain_text_for_search_only": text,
            "colors": sorted(set(paragraph.xpath(".//w:color/@w:val", namespaces=NS))),
            "native_ooxml": etree.tostring(paragraph, encoding="unicode"),
        })
    manifest = {"source_policy": "All input files are read-only. Equations through (36) remain unchanged.",
                "snapshot_sha256": digest(snapshot), "inputs": records}
    (out / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "source_1025_paragraphs.json").write_text(json.dumps(paragraphs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"snapshot_matches_original": digest(snapshot) == digest(manuscript),
                      "source_files_hashed": len(records), "nonempty_paragraphs": len(paragraphs)}))


if __name__ == "__main__":
    main()
