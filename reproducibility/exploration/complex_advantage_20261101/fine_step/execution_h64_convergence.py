"""Supplementary h0/64 time-convergence check; fixed networks, read-only inputs.

This does not modify the accepted h16/h32 driver, protocol, or summaries.
All four models and all five existing seeds are retained. No h128 is run.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_variable] = "1"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
sys.dont_write_bytecode = True

import numpy as np
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("accepted_fine_step_driver", OUT.parent / "fine_step.py")
fine = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fine
spec.loader.exec_module(fine)


def protected_prior_outputs():
    files = [OUT.parent / "fine_step.py"]
    files += [OUT / name for name in ["protocol.json", "execution_script.py", "original_h_verification.json",
        "h16_runtime.json", "h32_runtime.json", "trajectory_metrics.csv", "seed_metrics.csv",
        "solver_diagnostics.csv", "paired_seed_differences.csv", "paired_summary.csv",
        "refinement_vs_representation.csv", "summary.json", "protection_verification.json",
        "independent_verification.json", "continuous_H0_baseline.csv"]]
    for seed in fine.SEEDS:
        files += [OUT / f"seed_{seed}/h{divisor}_{suffix}" for divisor in [16, 32]
                  for suffix in ["records.json", "trajectories.npz"]]
    return {p.relative_to(OUT.parent).as_posix(): fine.digest(p) for p in files}


def check_prior_outputs(expected):
    changed = [name for name, sha in expected.items() if fine.digest(OUT.parent / name) != sha]
    if changed:
        raise AssertionError({"accepted_outputs_changed": changed})
    return {"files_checked": len(expected), "all_hashes_match": True}


def protocol_amendment():
    old = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    protocol = {
        "date": "2026-11-01",
        "purpose": "Additional numerical time-convergence check after h16/h32; no model selection or retraining.",
        "basis": "Observed h16-to-h32 motion still affects full-state model contrasts. All fixed original seeds and models receive h64.",
        "divisor": 64, "seeds": fine.SEEDS, "groups": fine.GROUPS,
        "grid": "Unchanged 914 output times, eight held-out trajectories per seed, endpoint 15.992993212174294 seconds.",
        "weights": "Unchanged saved 10000-update GR and GC weights; no selection or updates.",
        "algorithm": "Accepted fine_step.run_seed low-level float64 midpoint and analytic H0 Cayley trajectory.",
        "primary_metrics": fine.PRIMARY,
        "statistics": "Per-trajectory time metrics, then mean of eight per seed, paired differences and mean/sample SD over all five seeds.",
        "convergence_checks": "Compare h32-to-h64 trajectories and error changes with h64 GR-GC distances; retain h16/h32 results unchanged.",
        "work_limit": "All five seeds at h64 only; no h128.",
        "claim_limits": "Existing-data numerical check only; mixed signs and output specificity retained. No fresh-data confirmation claim.",
        "accepted_protocol_sha256": fine.digest(OUT / "protocol.json"),
        "original_protected_manifest_sha256": fine.digest(OUT.parent / "protected_manifest.json"),
        "inputs": old["inputs"],
        "accepted_outputs": protected_prior_outputs(),
    }
    path = OUT / "protocol_h64_convergence_amendment.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise AssertionError("h64 amendment/input mismatch")
    else:
        fine.save_json(path, protocol)
    return protocol


def seed_means(rows):
    keys = [k for k in rows[0] if k not in ["seed", "group", "divisor", "step_tau", "trajectory"]]
    return [{"divisor": 64, "seed": seed, "group": group,
             **{key: statistics.mean(r[key] for r in rows if r["seed"] == seed and r["group"] == group)
                for key in keys}}
            for seed in fine.SEEDS for group in fine.GROUPS]


def summarize_and_verify():
    rows, diagnostics, failures, archives = [], [], [], []
    max_independent_difference = 0.
    refinements = []
    for seed in fine.SEEDS:
        record = json.loads((OUT / f"seed_{seed}/h64_records.json").read_text(encoding="utf-8"))
        path = OUT / f"seed_{seed}/h64_trajectories.npz"
        if fine.digest(path) != record["archive_sha256"]:
            raise AssertionError("h64 archive changed")
        rows += record["trajectory_metrics"]
        diagnostics += record["solver_diagnostics"]
        failures += record["failures"]
        archives.append({"seed": seed, "path": str(path), "sha256": fine.digest(path)})
        with np.load(path, allow_pickle=False) as latest, np.load(OUT / f"seed_{seed}/h32_trajectories.npz", allow_pickle=False) as previous:
            for field in ["times_seconds", "continuous_reference", "continuous_H0", "initials", "nu", "d", "tip_map_m", "groups"]:
                if not np.array_equal(latest[field], previous[field]):
                    raise AssertionError({"seed": seed, "grid_or_reference_mismatch": field})
            reference = latest["continuous_reference"]
            tip = latest["tip_map_m"]
            groups = list(latest["groups"])
            if groups != fine.GROUPS or latest["predicted_states"].shape != (4, 914, 8, 4):
                raise AssertionError("Incomplete h64 model block")
            cross = fine.array_metrics(latest["predicted_states"][groups.index("GR")] - latest["predicted_states"][groups.index("GC")], reference, tip)
            group_errors = {}
            for group in groups:
                index = groups.index(group)
                states = latest["predicted_states"][index]
                difference = states - reference
                # Independent per-trajectory sums, not the original metric implementation.
                tip_error = np.einsum("tij,j->ti", difference[..., :2], tip) * 1000
                independent = {
                    "continuous_tip_rmse_mm": np.sqrt(np.sum(tip_error**2, axis=0) / len(reference)),
                    "continuous_state_relative_l2": np.sqrt(np.sum(difference**2, axis=(0, 2)) / np.sum(reference**2, axis=(0, 2))),
                }
                for metric, values in independent.items():
                    for trajectory in range(8):
                        saved = next(r[metric] for r in record["trajectory_metrics"] if r["group"] == group and r["trajectory"] == trajectory)
                        gap = abs(float(values[trajectory]) - saved)
                        max_independent_difference = max(max_independent_difference, gap)
                        if gap > 1e-12:
                            raise AssertionError("Independent h64 metric mismatch")
                group_errors[group] = {metric: statistics.mean(map(float, values)) for metric, values in independent.items()}
            for group in groups:
                index = groups.index(group)
                coarse, last = previous["predicted_states"][index], latest["predicted_states"][index]
                change = fine.array_metrics(coarse - last, reference, tip)
                coarse_errors = fine.array_metrics(coarse - reference, reference, tip)
                for metric in fine.PRIMARY:
                    metric_move = abs(coarse_errors[metric] - group_errors[group][metric])
                    contrast = abs(group_errors["GR"][metric] - group_errors["GC"][metric])
                    refinements.append({"seed": seed, "group": group, "metric": metric,
                        "coarse_divisor": 32, "fine_divisor": 64,
                        "coarse_error": coarse_errors[metric], "fine_error": group_errors[group][metric],
                        "trajectory_change": change[metric], "same_step_GR_GC_trajectory_distance": cross[metric],
                        "trajectory_change_below_GR_GC_distance": change[metric] < cross[metric],
                        "metric_change": metric_move, "same_step_abs_GR_GC_metric_difference": contrast,
                        "metric_change_below_GR_GC_difference": metric_move < contrast})
    if failures or len(rows) != 160:
        raise AssertionError({"failures": failures, "row_count": len(rows)})
    means = seed_means(rows)
    lookup = {(r["seed"], r["group"]): r for r in means}
    aggregate, pairs, pair_summary = [], [], []
    for group in fine.GROUPS:
        aggregate.append({"group": group, "metrics": {metric: {
            "mean": statistics.mean(lookup[seed, group][metric] for seed in fine.SEEDS),
            "sample_sd": statistics.stdev(lookup[seed, group][metric] for seed in fine.SEEDS)}
            for metric in fine.PRIMARY}})
    for first, second in [("GR", "GC"), ("H0", "GR"), ("H0", "GC")]:
        for metric in fine.PRIMARY:
            subset = []
            for seed in fine.SEEDS:
                a, b = lookup[seed, first][metric], lookup[seed, second][metric]
                row = {"seed": seed, "contrast": first + "-" + second, "metric": metric,
                       "first_error": a, "second_error": b, "difference": a - b,
                       "relative_improvement_fraction": (a - b) / max(abs(a), 1e-30)}
                pairs.append(row)
                subset.append(row)
            pair_summary.append({"contrast": first + "-" + second, "metric": metric,
                "mean_difference": statistics.mean(r["difference"] for r in subset),
                "sample_sd_difference": statistics.stdev(r["difference"] for r in subset),
                "positive_direction_seed_count": sum(r["difference"] > 0 for r in subset),
                "mean_relative_improvement_fraction": statistics.mean(r["relative_improvement_fraction"] for r in subset)})
    fine.write_csv(OUT / "h64_trajectory_metrics.csv", rows)
    fine.write_csv(OUT / "h64_seed_metrics.csv", means)
    fine.write_csv(OUT / "h64_solver_diagnostics.csv", diagnostics)
    fine.write_csv(OUT / "h64_paired_seed_differences.csv", pairs)
    fine.write_csv(OUT / "h64_paired_summary.csv", pair_summary)
    fine.write_csv(OUT / "h32_h64_refinement.csv", refinements)
    summary = {"status": "complete", "divisor": 64, "all_models": aggregate, "paired": pair_summary,
               "refinement_comparisons": refinements,
               "independent_primary_metric_trajectory_rows": 160,
               "maximum_independent_metric_gap": max_independent_difference,
               "maximum_implicit_residual": max(r["maximum_implicit_residual"] for r in diagnostics),
               "archive_hashes": archives, "failures": failures}
    fine.save_json(OUT / "h64_summary.json", summary)
    return summary


def main():
    protocol = protocol_amendment()
    before = fine.check_protected_sources()
    check_prior_outputs(protocol["accepted_outputs"])
    snapshot = OUT / "execution_h64_convergence.py"
    if snapshot.exists() and fine.digest(snapshot) != fine.digest(__file__):
        raise AssertionError("h64 execution script changed")
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes())
    fine.save_json(OUT / "environment_h64.json", {"python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "script_sha256": fine.digest(__file__), "workers": 2,
        "blas_threads": 1, "numba_threads": 1, "bytecode_writes_disabled": True,
        "utc_started": datetime.now(timezone.utc).isoformat()})
    started = time.perf_counter()
    helpers = fine.load_helpers()
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=2) as executor:
            for seed in fine.SEEDS:
                fine.run_seed(helpers, seed, 64, executor)
    fine.save_json(OUT / "h64_runtime.json", {"divisor": 64, "all_five_seed_seconds": time.perf_counter() - started})
    summary = summarize_and_verify()
    after = fine.check_protected_sources()
    prior = check_prior_outputs(protocol["accepted_outputs"])
    for item in protocol["inputs"]:
        if fine.digest(fine.PROJECT / item["path"]) != item["sha256"]:
            raise AssertionError("Accepted input changed")
    fine.save_json(OUT / "protection_h64_verification.json", {"before": before, "after": after,
        "accepted_h16_h32_outputs": prior, "experiment_input_hashes_unchanged": True,
        "all_original_sources_unchanged": True})
    print("H64_COMPLETE", json.dumps({"paired": summary["paired"], "residual": summary["maximum_implicit_residual"]}), flush=True)


if __name__ == "__main__":
    main()
