"""Package the reviewed project locally without changing original inputs."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "IJRR_1025_中文审阅稿_20261030.zip"
EXCLUDED_SUFFIXES = {".aux", ".out", ".toc", ".fls", ".fdb_latexmk", ".pyc", ".nbi", ".nbc"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    required = ["manuscript/main.pdf", "manuscript/main.tex", "manuscript/source_derivation.pdf",
                "README.md", "RESULTS_中文说明.md", "audit/final_numeric_verification.json",
                "audit/final_scientific_review.md", "audit/final_source_review.md"]
    for name in required:
        assert (ROOT / name).is_file(), f"Missing final artifact: {name}"
    source = json.loads((ROOT / "audit/source_manifest.json").read_text(encoding="utf-8-sig"))
    for entry in source["inputs"]:
        assert digest(Path(entry["path"])) == entry["sha256"], f"Source changed: {entry['path']}"
    summary = json.loads((ROOT / "results/neural_learning_10000/summary.json").read_text(encoding="utf-8-sig"))
    assert summary["completed"] == 20 and not summary["failures"]
    manifest_path = ROOT / "audit/output_manifest.json"
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rel = path.relative_to(ROOT)
        if path.suffix in EXCLUDED_SUFFIXES or any(p.startswith("render_") or p == "__pycache__" for p in rel.parts):
            continue
        if path.name == "main_contact_sheet.png":
            continue
        files.append(path)
    manifest = {"scope": "Chinese IJRR review manuscript, frozen-source companion, saved experiments and audits",
                "source_input_count": len(source["inputs"]), "source_inputs_unchanged": True,
                "completed_neural_runs": 20, "files": [
                    {"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in files]}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Exclusive creation prevents accidental replacement of a previous delivery.
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files + [manifest_path]:
            archive.write(path, (Path(ROOT.name) / path.relative_to(ROOT)).as_posix())
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(files) + 1
    print(json.dumps({"archive": str(ARCHIVE), "files": len(files) + 1,
                      "bytes": ARCHIVE.stat().st_size, "sha256": digest(ARCHIVE)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
