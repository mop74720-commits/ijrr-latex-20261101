"""Run the original FE reconstruction check using the bundled unchanged source."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from package_common import (DEFAULT_MANIFEST, PACKAGE, import_file, new_run, read_json,
                            require_declared, require_verified, sha256, write_json)

EXPECTED_BEAM_SHA256 = "da1cd69ea2b8d5e6d2f5c835e85c25c7d4d9c33db99dd722f8ca8d63c5d611dd"
DESIGN = "exploration/ab_state_frequency_20261101/physical_design"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--beam-source", default=str(PACKAGE / "vendor/beam/run_beam_validation.py"))
    args = parser.parse_args()
    before = require_verified(args.manifest, full=True)
    beam = Path(args.beam_source).resolve()
    if not beam.is_relative_to(PACKAGE.resolve()):
        raise RuntimeError("Use a manifest-declared beam source inside the reproduction package")
    require_declared(beam.relative_to(PACKAGE).as_posix(), args.manifest)
    if sha256(beam) != EXPECTED_BEAM_SHA256:
        raise RuntimeError("Bundled beam source does not match the frozen original SHA-256")
    models = require_declared(f"{DESIGN}/build_models.py", args.manifest)
    check = require_declared(f"{DESIGN}/check_reconstruction.py", args.manifest)
    destination = new_run("physics_source")
    os.environ["MPLCONFIGDIR"] = str(destination / "matplotlib_cache")
    write_json(destination / "package_verification_before.json", before)
    module = import_file("build_models", models)
    # Path/output adaptation is in memory; scientific source and protocols stay intact.
    module.RAW_SOURCE = beam
    module.OUT = destination
    verifier = import_file("published_beam_reconstruction_check", check)
    verifier.main()
    after = require_verified(args.manifest, full=True)
    if before["manifest_sha256"] != after["manifest_sha256"]:
        raise RuntimeError("Package manifest changed during physical-source check")
    write_json(destination / "package_verification_after.json", after)
    result = {"status": "passed", "output_directory": str(destination),
              "beam_source_sha256": sha256(beam), "all_published_inputs_unchanged": True,
              "reconstruction": read_json(destination / "reconstruction_checks.json"),
              "historical_2409_file_audit_reexecuted": False,
              "scope": "Dimensional/FE equation reconstruction and linear frequency mesh sensitivity only; not strong-field high-fidelity or material validation."}
    write_json(destination / "runner_report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
