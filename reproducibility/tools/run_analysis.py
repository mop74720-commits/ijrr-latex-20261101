"""Recompute the frozen A/B statistics and figures in a fresh output directory."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
from package_common import (DEFAULT_MANIFEST, import_file, new_run, read_json,
                            require_declared, require_verified, sha256, write_json)

ANALYSIS = "exploration/ab_state_frequency_20261101/analysis"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()
    before = require_verified(args.manifest, full=True)
    source = require_declared(f"{ANALYSIS}/build_analysis.py", args.manifest)
    original_protocol = require_declared(f"{ANALYSIS}/analysis_protocol.json", args.manifest)
    destination = new_run("analysis")
    os.environ["MPLCONFIGDIR"] = str(destination / "matplotlib_cache")
    write_json(destination / "package_verification_before.json", before)
    module = import_file("published_ab_analysis", source)
    # All input roots remain the original package paths. Only new artifacts move.
    module.OUT = destination
    if module.protocol() != read_json(original_protocol):
        raise RuntimeError("Frozen analysis protocol differs from the original record")
    old_argv = sys.argv
    try:
        sys.argv = [str(source), "--stage", "run"]
        module.main()
    finally:
        sys.argv = old_argv
    independent = read_json(destination / "independent_statistics_verification.json")
    if independent["status"] != "passed":
        raise RuntimeError("Recomputed statistics failed independent checks")
    after = require_verified(args.manifest, full=True)
    if before["manifest_sha256"] != after["manifest_sha256"]:
        raise RuntimeError("Package manifest changed during analysis")
    write_json(destination / "package_verification_after.json", after)
    result = {"status": "passed", "output_directory": str(destination),
              "analysis_source_sha256": sha256(source), "frozen_protocol_unchanged": True,
              "all_published_inputs_unchanged": True, "independent_statistics": independent,
              "historical_2409_file_audit_reexecuted": False,
              "scope": "Recomputed saved-trajectory statistics and figures; no training or physical verification."}
    write_json(destination / "runner_report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
