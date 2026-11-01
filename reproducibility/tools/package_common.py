"""Integrity and output-path helpers for the published reproduction package."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import uuid

PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parent
DEFAULT_MANIFEST = PACKAGE / "package_manifest.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_file(relative, root=PACKAGE):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Manifest paths must be nonempty strings")
    posix, windows = PurePosixPath(relative), PureWindowsPath(relative)
    if (posix.is_absolute() or windows.drive or "\\" in relative or ":" in relative
            or ".." in posix.parts or posix.as_posix() != relative or relative == "."):
        raise ValueError(f"Unsafe package-relative manifest path: {relative!r}")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Manifest path escapes package: {relative!r}")
    return path


def _manifest_snapshot(path=DEFAULT_MANIFEST):
    path = Path(path).resolve()
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8-sig"))
    if value.get("schema_version") != 1 or not isinstance(value.get("files"), dict):
        raise ValueError("Expected schema_version=1 and a files mapping")
    if not value["files"]:
        raise ValueError("An empty file manifest cannot verify this package")
    for relative, record in value["files"].items():
        package_file(relative)
        if not isinstance(record, dict):
            raise ValueError(f"Invalid manifest record: {relative}")
        digest = record.get("sha256")
        size = record.get("bytes")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or not isinstance(size, int) or isinstance(size, bool) or size < 0):
            raise ValueError(f"Invalid SHA-256 or byte size: {relative}")
    return path, value, hashlib.sha256(payload).hexdigest()


def load_manifest(path=DEFAULT_MANIFEST):
    path, value, _ = _manifest_snapshot(path)
    return path, value


def verify_manifest(path=DEFAULT_MANIFEST, *, full):
    path, manifest, manifest_digest = _manifest_snapshot(path)
    missing, mismatches = [], []
    checked = checked_bytes = 0
    for relative, record in manifest["files"].items():
        target = package_file(relative)
        if not target.is_file():
            missing.append(relative)
            continue
        actual_size = target.stat().st_size
        actual_digest = sha256(target)
        checked += 1
        checked_bytes += actual_size
        if actual_size != record["bytes"] or actual_digest != record["sha256"]:
            mismatches.append({"path": relative, "expected_bytes": record["bytes"],
                               "actual_bytes": actual_size, "expected_sha256": record["sha256"],
                               "actual_sha256": actual_digest})
    manifest_digest_after = sha256(path)
    manifest_changed = manifest_digest_after != manifest_digest
    failed = bool(mismatches) or (full and bool(missing)) or manifest_changed
    status = "failed" if failed else ("partial" if missing else "complete")
    return {"status": status, "mode": "full" if full else "available",
            "manifest": str(path), "manifest_sha256": manifest_digest,
            "manifest_sha256_after": manifest_digest_after,
            "manifest_changed_during_verification": manifest_changed,
            "declared_files": len(manifest["files"]), "verified_present_files": checked,
            "verified_present_bytes": checked_bytes, "missing_count": len(missing),
            "missing_files": missing, "mismatched_files": mismatches,
            "full_package_verified": not missing and not mismatches and not manifest_changed,
            "scope": "Published package bytes only; not a rerun of the historical 2409-file workspace audit.",
            "partial_note": "Download and extract the complete Release archive to verify missing files." if missing else None}


def require_verified(path=DEFAULT_MANIFEST, *, full=True):
    result = verify_manifest(path, full=full)
    if result["status"] == "failed":
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result


def require_declared(relative, manifest_path=DEFAULT_MANIFEST):
    _, manifest = load_manifest(manifest_path)
    if relative not in manifest["files"]:
        raise RuntimeError(f"Required input is not declared in package manifest: {relative}")
    target = package_file(relative)
    if not target.is_file():
        raise FileNotFoundError(f"Required input is missing: {relative}")
    record = manifest["files"][relative]
    if target.stat().st_size != record["bytes"] or sha256(target) != record["sha256"]:
        raise RuntimeError(f"Required input failed byte/hash verification: {relative}")
    return target


def new_run(kind):
    root = (REPOSITORY / "reproduction_runs").resolve()
    if not root.is_relative_to(REPOSITORY.resolve()):
        raise RuntimeError("reproduction_runs escapes the repository")
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = root / f"{kind}_{stamp}_{uuid.uuid4().hex[:6]}"
    destination.mkdir(exist_ok=False)
    return destination


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def import_file(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module
