"""Read-only inspection of five original datasets and all 50 published weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np

from package_common import DEFAULT_MANIFEST, package_file, require_verified, sha256

SEEDS = (17, 43, 91, 137, 211)
AB = "exploration/ab_state_frequency_20261101/experiment_b"


def required_input(relative, records):
    """Check an input against the manifest snapshot validated by the preflight."""
    if relative not in records:
        raise RuntimeError(f"Required input is not declared in package manifest: {relative}")
    path = package_file(relative)
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing: {relative}")
    expected = records[relative]
    if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
        raise RuntimeError(f"Required input failed byte/hash verification: {relative}")
    return path


def inspect_data(relative, records):
    path = required_input(relative, records)
    with np.load(path, allow_pickle=False) as data:
        fields = {name: {"shape": list(data[name].shape), "dtype": str(data[name].dtype)} for name in data.files}
        for split, trajectories, samples in (("train", 12, 286), ("val", 4, 571)):
            assert data[f"{split}_initials"].shape == (trajectories, 4)
            for name in ("midpoints", "labels", "clean_labels", "noise"):
                array = data[f"{split}_{name}"]
                assert array.shape == (trajectories * samples, 4) and np.isfinite(array).all()
            assert np.array_equal(data[f"{split}_labels"], data[f"{split}_clean_labels"] + data[f"{split}_noise"])
            ids, counts = np.unique(data[f"{split}_trajectory_id"], return_counts=True)
            assert np.array_equal(ids, np.arange(trajectories)) and np.all(counts == samples)
        assert data["test_initials"].shape == (8, 4)
        for name in ("test_reference_states", "test_teacher_states"):
            assert data[name].shape == (8, len(data["test_times"]), 4)
            assert np.isfinite(data[name]).all()
        return {"path": relative, "fields": fields, "train_trajectories": 12,
                "train_samples_per_trajectory": 286, "validation_trajectories": 4,
                "validation_samples_per_trajectory": 571, "test_trajectories": 8,
                "test_saved_times": len(data["test_times"]), "test_state_dimension": 4,
                "label_plus_noise_identity": True}


def inspect_weight(relative, group, records):
    path = required_input(relative, records)
    with np.load(path, allow_pickle=False) as weight:
        names = ("w1", "b1", "w2", "b2", "w3", "b3", "scale", "energy_scale", "nu", "prior")
        assert set(weight.files) == set(names)
        assert all(np.isfinite(weight[name]).all() for name in names)
        real_family = group in ("R", "GR")
        expected = ((20, 4), (20,), (26, 20), (26,), (26,), (1,)) if real_family else ((32, 4), (32,), (32, 32), (32,), (32,), (1,))
        assert all(weight[name].shape == shape for name, shape in zip(names[:6], expected))
        if not real_family:
            # Export stores redundant real blocks for the complex architecture.
            # Their entry count is not the number of independent trained parameters.
            for name, columns in (("w1", 2), ("w2", 16)):
                w = weight[name]
                assert np.array_equal(w[:16, :columns], w[16:, columns:])
                assert np.array_equal(w[:16, columns:], -w[16:, :columns])
        assert weight["scale"].shape == (4,) and np.all(weight["scale"] > 0)
        assert weight["nu"].shape == (2,)
        assert bool(weight["prior"]) == group.startswith("G")
        return {"path": relative, "group": group, "status": "passed",
                "independent_training_parameters": 673,
                "exported_parameter_matrix_entries": sum(int(weight[name].size) for name in names[:6]),
                "fields": {name: {"shape": list(weight[name].shape), "dtype": str(weight[name].dtype)} for name in names}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--details", action="store_true", help="Also print shapes of all weight arrays")
    args = parser.parse_args()
    try:
        integrity = require_verified(args.manifest, full=False)
        manifest_path = Path(args.manifest).resolve()
        manifest_bytes = manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != integrity["manifest_sha256"]:
            raise RuntimeError("Package manifest changed after the integrity preflight")
        # Reuse the same validated records; do not resolve all package paths
        # again for each of the 55 required dataset/weight files.
        records = json.loads(manifest_bytes.decode("utf-8-sig"))["files"]
        datasets = [inspect_data(f"results/learning/data/seed_{seed}.npz", records) for seed in SEEDS]
        original, ab = [], []
        for seed in SEEDS:
            for group in ("R", "C", "GR", "GC"):
                original.append(inspect_weight(f"results/neural_learning_10000/seed_{seed}/{group}_weights.npz", group, records))
        for level in ("B030mT", "B060mT", "B120mT"):
            for seed in SEEDS:
                for group in ("GR", "GC"):
                    ab.append(inspect_weight(f"{AB}/{level}/seed_{seed}/{group}_weights.npz", group, records))
        if not args.details:
            original = [{key: value for key, value in row.items() if key != "fields"} for row in original]
            ab = [{key: value for key, value in row.items() if key != "fields"} for row in ab]
        if sha256(manifest_path) != integrity["manifest_sha256"]:
            raise RuntimeError("Package manifest changed during dataset/weight inspection")
        result = {"status": "passed", "read_only": True, "package_integrity": integrity,
                  "manifest_snapshot_rechecked": True,
                  "original_datasets_checked": len(datasets), "original_weights_checked": len(original),
                  "ab_weights_checked": len(ab), "datasets": datasets, "original_weights": original,
                  "ab_weights": ab, "scope": "Saved synthetic data and weight integrity/shape checks; no retraining or physical validation."}
    except (OSError, ValueError, RuntimeError, AssertionError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error) or type(error).__name__}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
