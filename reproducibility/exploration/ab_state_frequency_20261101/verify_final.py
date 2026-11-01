"""Read-only verification of old files; writes only this exploration's QA JSON."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    manifest = read(HERE / "protected_manifest.json")
    missing, changed = [], []
    for relative, expected in manifest.items():
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
        elif digest(path) != expected:
            changed.append(relative)

    syntax = []
    for path in sorted(HERE.rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        syntax.append(path.relative_to(HERE).as_posix())

    input_checks = []
    for relative in ("experiment_a/protocol.json", "experiment_b/protocol.json"):
        protocol = read(HERE / relative)
        failures = [name for name, expected in protocol["input_sha256"].items()
                    if not (ROOT / name).is_file() or digest(ROOT / name) != expected]
        input_checks.append({"protocol": relative, "checked": len(protocol["input_sha256"]),
                             "changed_or_missing": failures})

    b_root = HERE / "experiment_b"
    completed = []
    for level in ("B030mT", "B060mT", "B120mT"):
        for seed in (17, 43, 91, 137, 211):
            dest = b_root / level / f"seed_{seed}"
            for group in ("GR", "GC"):
                record = read(dest / f"{group}_training.json")
                assert record["status"] == "completed"
                assert record["training_steps"] == 10000
                assert record["real_parameter_count"] == 673
                assert record["weights_sha256"] == digest(dest / f"{group}_weights.npz")
                assert record["data_sha256"] == digest(dest / "data.npz")
                assert record["baseline_weights_reused"] == (level == "B030mT")
                completed.append({"level": level, "seed": seed, "group": group,
                                  "reused": record["baseline_weights_reused"]})
            evaluation = read(dest / "evaluation.json")
            assert evaluation["status"] == "completed"
            assert evaluation["archive_sha256"] == digest(dest / "continuous64s.npz")
            assert len(evaluation["metrics"]) == 2 * 5 * 8
            assert {row["duration_seconds"] for row in evaluation["metrics"]} == {16, 64}
            for group in ("GR", "GC"):
                assert evaluation["weights_sha256"][group] == digest(dest / f"{group}_weights.npz")

    result = {"protected_files_checked": len(manifest), "missing": missing,
              "changed": changed, "all_protected_files_unchanged": not missing and not changed,
              "manuscript_pdf_sha256": digest(ROOT / "manuscript/main.pdf"),
              "new_scripts_ast_checked": syntax, "protocol_input_checks": input_checks,
              "b_training_records": len(completed),
              "b_new_fits": sum(not x["reused"] for x in completed),
              "b_reused_weights": sum(x["reused"] for x in completed),
              "b_completed_seed_levels": len(completed) // 2,
              "no_github_mutation_performed": True}
    assert result["all_protected_files_unchanged"], result
    assert all(not entry["changed_or_missing"] for entry in input_checks), result
    assert result["b_new_fits"] == 20 and result["b_reused_weights"] == 10
    destination = HERE / "final_verification.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
