"""Verify either the complete Release payload or the explicitly available subset."""
from __future__ import annotations

import argparse
import json
import sys

from package_common import DEFAULT_MANIFEST, verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--full", action="store_true", help="Require every declared file, size and SHA-256")
    modes.add_argument("--available", action="store_true", help="Verify existing files and explicitly list missing Release files")
    args = parser.parse_args()
    try:
        result = verify_manifest(args.manifest, full=args.full)
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
