"""Independently recompute the saved fine-step metrics; never simulate or train."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

import numpy as np

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
SEEDS = [17, 43, 91, 137, 211]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def comparison(predicted, reference, tip):
    error = predicted - reference
    displacement = 1000 * np.einsum("ntj,j->nt", error[..., :2], tip)
    return (np.sqrt(np.sum(displacement**2, axis=0) / len(displacement)),
            np.sqrt(np.sum(error**2, axis=(0, 2)) / np.sum(reference**2, axis=(0, 2))))


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    protected = json.loads((OUT.parent / "protected_manifest.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and not summary["failures"]
    for path, expected in protected.items():
        assert sha(PROJECT / path) == expected, path
    for item in protocol["inputs"]:
        assert sha(PROJECT / item["path"]) == item["sha256"], item["path"]
    assert sha(OUT / "execution_script.py") == sha(OUT.parent / "fine_step.py")
    lookup, maximum_gap, continuum = {}, 0., []
    count = 0
    for divisor in [16, 32]:
        for seed in SEEDS:
            path = OUT / f"seed_{seed}/h{divisor}_trajectories.npz"
            record = json.loads((path.parent / f"h{divisor}_records.json").read_text(encoding="utf-8"))
            assert sha(path) == record["archive_sha256"]
            with np.load(path, allow_pickle=False) as case:
                assert case["predicted_states"].shape == (4, 914, 8, 4)
                assert np.isfinite(case["predicted_states"]).all()
                assert list(case["groups"]) == ["H0", "GR", "GC", "Teacher"]
                for group in ["GR", "GC"]:
                    with np.load(PROJECT / f"results/neural_learning_10000/seed_{seed}/{group}_rollout.npz", allow_pickle=False) as old:
                        assert np.array_equal(case["times_seconds"], old["times_seconds"])
                        assert np.array_equal(case["continuous_reference"], old["reference_states"])
                assert abs(case["times_seconds"][-1] - 15.992993212174294) < 1e-13
                for j, group in enumerate(case["groups"]):
                    computed = {}
                    for reference_name, prefix in [("continuous_reference", "continuous"), ("matched_teacher", "matched_teacher")]:
                        tip, state = comparison(case["predicted_states"][j], case[reference_name], case["tip_map_m"])
                        computed[prefix + "_tip_rmse_mm"] = tip
                        computed[prefix + "_state_relative_l2"] = state
                    rows = [r for r in record["trajectory_metrics"] if r["group"] == group]
                    assert len(rows) == 8
                    for metric, values in computed.items():
                        saved = np.array([r[metric] for r in rows])
                        gap = float(abs(values - saved).max())
                        maximum_gap = max(maximum_gap, gap)
                        assert gap < 2e-13
                    lookup[divisor, seed, str(group)] = {metric: statistics.mean(map(float, values))
                                                        for metric, values in computed.items()}
                    count += 8
                if divisor == 32:
                    tip, state = comparison(case["continuous_H0"], case["continuous_reference"], case["tip_map_m"])
                    continuum.append({"seed": seed, "continuous_tip_rmse_mm": statistics.mean(map(float, tip)),
                                      "continuous_state_relative_l2": statistics.mean(map(float, state))})
    for item in summary["all_models"]:
        for metric in lookup[item["divisor"], SEEDS[0], item["group"]]:
            values = [lookup[item["divisor"], seed, item["group"]][metric] for seed in SEEDS]
            assert abs(statistics.mean(values) - item["metrics"][metric]["mean"]) < 2e-13
            assert abs(statistics.stdev(values) - item["metrics"][metric]["sample_sd"]) < 2e-13
    directions = {}
    for divisor in [16, 32]:
        directions[str(divisor)] = {}
        for metric in ["continuous_tip_rmse_mm", "continuous_state_relative_l2"]:
            differences = [lookup[divisor, seed, "GR"][metric] - lookup[divisor, seed, "GC"][metric] for seed in SEEDS]
            source = next(r for r in summary["paired"] if r["divisor"] == divisor and r["contrast"] == "GR-GC" and r["metric"] == metric)
            assert sum(x > 0 for x in differences) == source["positive_direction_seed_count"]
            assert abs(statistics.mean(differences) - source["mean_difference"]) < 2e-13
            directions[str(divisor)][metric] = {"GC_direction_wins": sum(x > 0 for x in differences),
                "GR_minus_GC_each_seed": dict(zip(map(str, SEEDS), differences))}
    with (OUT / "continuous_H0_baseline.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(continuum[0]))
        writer.writeheader(); writer.writerows(continuum)
    result = {"status": "pass", "raw_trajectory_records_recomputed": count,
        "maximum_primary_metric_gap": maximum_gap, "protected_files_verified": len(protected),
        "shared_GR_GC_times_and_references_exactly_equal": True,
        "source_execution_snapshot_unchanged": True, "directional_counts": directions,
        "continuous_H0_model_bias": {metric: {"mean": statistics.mean(r[metric] for r in continuum),
                    "sample_sd": statistics.stdev(r[metric] for r in continuum)}
            for metric in ["continuous_tip_rmse_mm", "continuous_state_relative_l2"]},
        "scope": "Read-only independent recomputation. No model training or trajectory integration was repeated."}
    (OUT / "independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
