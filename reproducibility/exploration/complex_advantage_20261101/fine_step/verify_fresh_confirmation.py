"""Independent saved-array audit; run only after all fresh results are complete.

No experiment helper is imported. Source data, networks and frozen scripts are
read-only; only this auditor's JSON/CSV reports are written beside this file.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True
import numpy as np

OUT = Path(__file__).resolve().parent
HERE = OUT.parent
PROJECT = HERE.parents[1]
CONF = HERE / "confirmation"
SEEDS = [307, 419, 541, 653, 809]
DIVISORS = [32, 64]
METRICS = ["tip_rmse_mm_mean", "state_relative_l2_mean"]
MAX_GAP = 0.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(actual, expected, label, tolerance=1e-12):
    global MAX_GAP
    gap = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    MAX_GAP = max(MAX_GAP, gap)
    if gap > tolerance:
        raise AssertionError({"mismatch": label, "maximum_gap": gap})


def independently_computed_metrics(states, reference, tip):
    return difference_metrics(states - reference, reference, tip)


def difference_metrics(error, reference, tip):
    tip_mm = np.einsum("tij,j->ti", error[..., :2], tip) * 1000.
    tip_values = np.sqrt(np.sum(tip_mm * tip_mm, axis=0) / error.shape[0])
    denominator = np.sum(reference * reference, axis=(0, 2))
    if np.any(denominator <= 0):
        raise AssertionError("Zero state normalization")
    state_values = np.sqrt(np.sum(error * error, axis=(0, 2)) / denominator)
    return {"tip_rmse_mm_per_trajectory": tip_values.tolist(),
            "state_relative_l2_per_trajectory": state_values.tolist(),
            "tip_rmse_mm_mean": statistics.mean(map(float, tip_values)),
            "state_relative_l2_mean": statistics.mean(map(float, state_values))}


def compare_metrics(actual, stored, label):
    for key in actual:
        close(actual[key], stored[key], label + "/" + key)


def check_originals():
    manifest = read_json(HERE / "protected_manifest.json")
    changed = [path for path, digest in manifest.items() if sha(PROJECT / path) != digest]
    if changed:
        raise AssertionError({"original_source_hash_mismatches": changed})
    return {"files_checked": len(manifest), "all_hashes_match": True}


def audit():
    # Require every declared trajectory/record before reading any test output.
    required = [CONF / "summary.json", CONF / "frozen_nomination.json"]
    required += [CONF / f"seed_{seed}/{group}_h{divisor}.{suffix}"
                 for seed in SEEDS for group in ["GR", "GC"]
                 for divisor in DIVISORS for suffix in ["npz", "json"]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError({"results_incomplete": missing})
    input_hashes = {path: sha(path) for path in required}
    before = check_originals()
    frozen_path = CONF / "frozen_nomination.json"
    frozen = read_json(frozen_path)
    if frozen["confirmation_seeds"] != SEEDS:
        raise AssertionError("Fresh seed set differs from the frozen set")
    for registry in ["pilot_evidence_sha256", "execution_inputs_sha256"]:
        for name, digest in frozen[registry].items():
            if sha(HERE / name) != digest:
                raise AssertionError({"frozen_evidence_changed": name})
            input_hashes[HERE / name] = digest
    setting = frozen["chosen"]
    if setting is None:
        raise AssertionError("No setting was nominated")
    consumed = set(required)
    cases, trajectories, seed_rows, refinement = {}, [], [], []
    max_midpoint_residual = 0.
    for seed in SEEDS:
        directory = CONF / f"seed_{seed}"
        training_paths = [directory / f"{group}_training.json" for group in ["GR", "GC"]]
        consumed.update(training_paths)
        input_hashes.update({path: sha(path) for path in training_paths})
        training = {group: read_json(directory / f"{group}_training.json") for group in ["GR", "GC"]}
        data_path = HERE / training["GR"]["dataset_path"]
        consumed.add(data_path)
        input_hashes[data_path] = sha(data_path)
        with np.load(data_path, allow_pickle=False) as data:
            reference = data["test_reference_states"].transpose(1, 0, 2)
            times, initials, tip = data["test_times"], data["test_initials"], data["tip_map_m"]
            nu, d = data["nu"], data["d"]
            step0, stride0 = float(data["teacher_step_tau"]), int(data["teacher_sample_stride"])
            ids = np.unique(data["train_trajectory_id"])
            count = setting["train_trajectories"]
            chosen = np.random.default_rng(seed + 41000).permutation(ids)[:count] if count < 12 else ids
            selected = np.isin(data["train_trajectory_id"], chosen)
            expected_input_scale = np.maximum(np.sqrt(np.sum(data["train_midpoints"][selected]**2, axis=0) / np.sum(selected)), 1e-4)
            expected_energy_scale = max(float(np.mean(np.abs(data["train_midpoints"][selected] * data["train_labels"][selected]))), 1e-5)
            for group in ["GR", "GC"]:
                row = training[group]
                weight_path = directory / f"{group}_weights.npz"
                consumed.add(weight_path)
                input_hashes[weight_path] = sha(weight_path)
                if (row["status"] != "completed" or row["setting"] != setting
                        or row["dataset_path"] != training["GR"]["dataset_path"]
                        or row["dataset_sha256"] != sha(data_path)
                        or row["nomination_sha256"] != sha(frozen_path)
                        or row["weights_sha256"] != sha(weight_path)
                        or row["training_steps"] != setting["updates"]
                        or row["real_parameter_count"] != 673
                        or row["train_trajectory_ids"] != chosen.tolist()):
                    raise AssertionError({"training_provenance_mismatch": [seed, group]})
                with np.load(weight_path, allow_pickle=False) as weights:
                    close(weights["scale"], expected_input_scale, f"train-only scale/{seed}/{group}")
                    close(weights["energy_scale"], expected_energy_scale, f"train-only energy scale/{seed}/{group}")
                    if not bool(weights["prior"]) or not np.array_equal(weights["nu"], nu):
                        raise AssertionError("Saved prior/modal parameters changed")
            for split in ["train", "val"]:
                points = data[f"{split}_midpoints"]
                field = np.concatenate([points[:, 2:] * nu, -points[:, :2] * nu], axis=-1)
                theta = points[:, :2] @ d
                field[:, 2:] -= (np.sin(theta) - theta)[:, None] * d
                close(field, data[f"{split}_clean_labels"], f"Eq36 midpoint clean labels/{seed}/{split}", tolerance=1e-11)
            if reference.shape != (914, 8, 4) or not np.isfinite(reference).all():
                raise AssertionError("Unexpected/nonfinite fresh reference grid")
            for divisor in DIVISORS:
                baseline_path = directory / f"baselines_h{divisor}.json"
                consumed.add(baseline_path)
                input_hashes[baseline_path] = sha(baseline_path)
                baseline_json = read_json(baseline_path)
                baselines = {}
                for group in ["GR", "GC"]:
                    record = read_json(directory / f"{group}_h{divisor}.json")
                    archive_path = directory / f"{group}_h{divisor}.npz"
                    if (record["status"] != "completed" or record["seed"] != seed
                            or record["group"] != group or record["divisor"] != divisor
                            or record["dataset_sha256"] != sha(data_path)
                            or record["nomination_sha256"] != sha(frozen_path)
                            or record["weights_sha256"] != training[group]["weights_sha256"]
                            or record["trajectory_sha256"] != sha(archive_path)):
                        raise AssertionError({"rollout_provenance_mismatch": [seed, group, divisor]})
                    h, stride = step0 / divisor, stride0 * divisor
                    close(record["h_tau"], h, "step size")
                    if record["internal_steps"] != (len(times) - 1) * stride:
                        raise AssertionError("Internal step count changed")
                    max_midpoint_residual = max(max_midpoint_residual, record["maximum_implicit_residual"])
                    if record["maximum_implicit_residual"] > 2e-14:
                        raise AssertionError("Implicit residual exceeds declared tolerance")
                    with np.load(archive_path, allow_pickle=False) as saved:
                        if not np.array_equal(saved["times_seconds"], times) or not np.array_equal(saved["continuous_reference"], reference):
                            raise AssertionError("Rollout/reference times or states differ from data")
                        states = saved["predicted_states"]
                        if states.shape != reference.shape or not np.isfinite(states).all():
                            raise AssertionError("Incomplete/nonfinite model trajectory")
                        close(states[0], initials, "Model initial conditions")
                        teacher = saved["teacher"]
                        if group == "GR":
                            baselines = {"H0": saved["H0"], "Teacher": teacher}
                        elif not all(np.array_equal(saved[name], baselines[key]) for name, key in [("H0", "H0"), ("teacher", "Teacher")]):
                            raise AssertionError("GR/GC baseline trajectories differ")
                        for target, scope in [(reference, "continuous"), (teacher, "matched_teacher")]:
                            metric = independently_computed_metrics(states, target, tip)
                            compare_metrics(metric, record["metrics"][scope], f"{seed}/{group}/h{divisor}/{scope}")
                            if scope == "continuous":
                                cases[divisor, seed, group] = {"states": states.copy(), "metrics": metric}
                for group, states in baselines.items():
                    continuous = independently_computed_metrics(states, reference, tip)
                    matched = independently_computed_metrics(states, baselines["Teacher"], tip)
                    source = baseline_json["H0" if group == "H0" else "teacher"]
                    compare_metrics(continuous, source["continuous"], f"baseline/{seed}/{divisor}/{group}")
                    compare_metrics(matched, source["matched_teacher"], f"matched baseline/{seed}/{divisor}/{group}")
                    close(states[0], initials, "Baseline initial conditions")
                    cases[divisor, seed, group] = {"states": states.copy(), "metrics": continuous}
                phase = (np.arange(len(times)) * stride)[:, None] * (2 * np.arctan(h * nu / 2))[None, :]
                cosine, sine = np.cos(phase)[:, None, :], np.sin(phase)[:, None, :]
                analytic_h0 = np.concatenate([initials[None, :, :2] * cosine + initials[None, :, 2:] * sine,
                                             initials[None, :, 2:] * cosine - initials[None, :, :2] * sine], axis=-1)
                close(analytic_h0, baselines["H0"], f"Independent H0 Cayley/{seed}/{divisor}")
                for group in ["H0", "GR", "GC", "Teacher"]:
                    metric = cases[divisor, seed, group]["metrics"]
                    seed_rows.append({"seed": seed, "divisor": divisor, "group": group,
                                      **{key: metric[key] for key in METRICS}})
                    for trajectory in range(8):
                        trajectories.append({"seed": seed, "divisor": divisor, "group": group, "trajectory": trajectory,
                            "tip_rmse_mm": metric["tip_rmse_mm_per_trajectory"][trajectory],
                            "state_relative_l2": metric["state_relative_l2_per_trajectory"][trajectory]})
            cross = difference_metrics(cases[64, seed, "GR"]["states"] - cases[64, seed, "GC"]["states"], reference, tip)
            for group in ["H0", "GR", "GC", "Teacher"]:
                a, b = cases[32, seed, group], cases[64, seed, group]
                change = difference_metrics(a["states"] - b["states"], reference, tip)
                for key in METRICS:
                    refinement.append({"seed": seed, "group": group, "metric": key,
                        "trajectory_change": change[key], "same_step_GR_GC_trajectory_distance": cross[key],
                        "metric_change": abs(a["metrics"][key] - b["metrics"][key])})
    aggregate, paired, contrast_shifts = [], [], []
    root_summary = read_json(CONF / "summary.json")
    if len(root_summary["results"]) != 4:
        raise AssertionError("Incomplete root summary")
    for divisor in DIVISORS:
        for key in METRICS:
            values = {group: [cases[divisor, seed, group]["metrics"][key] for seed in SEEDS]
                      for group in ["H0", "GR", "GC", "Teacher"]}
            aggregate += [{"divisor": divisor, "metric": key, "group": group,
                           "mean": statistics.mean(numbers), "sample_sd": statistics.stdev(numbers)}
                          for group, numbers in values.items()]
            for first, second in [("GR", "GC"), ("H0", "GR"), ("H0", "GC")]:
                differences = [a - b for a, b in zip(values[first], values[second])]
                relative = [(a - b) / a for a, b in zip(values[first], values[second])]
                paired.append({"divisor": divisor, "metric": key, "contrast": first + "-" + second,
                    "seed_order": SEEDS, "differences": differences, "relative_improvements": relative,
                    "mean_difference": statistics.mean(differences), "sample_sd_difference": statistics.stdev(differences),
                    "mean_relative_improvement": statistics.mean(relative), "sample_sd_relative_improvement": statistics.stdev(relative),
                    "positive_direction_seed_count": sum(value > 0 for value in differences)})
            stored = next(row for row in root_summary["results"] if row["divisor"] == divisor and row["metric"] == key)
            gr_gc = paired[-3]
            for group in ["GR", "GC", "H0"]:
                close(statistics.mean(values[group]), stored[group + "_mean"], "summary mean")
                close(statistics.stdev(values[group]), stored[group + "_sample_sd"], "summary sample SD")
            close(gr_gc["relative_improvements"], stored["paired_relative_improvements"], "summary paired relative improvements")
            close(gr_gc["mean_relative_improvement"], stored["mean_paired_relative_improvement"], "summary mean relative improvement")
            close(gr_gc["sample_sd_relative_improvement"], stored["paired_improvement_sample_sd"], "summary paired sample SD")
            close(list(map(list, zip(values["GR"], values["GC"]))), stored["pairs"], "summary seed pairs")
            passed = gr_gc["mean_relative_improvement"] >= .05 and gr_gc["positive_direction_seed_count"] >= 4
            if gr_gc["positive_direction_seed_count"] != stored["GC_wins"] or passed != stored["pragmatic_rule_passed"]:
                raise AssertionError("Summary direction count/candidate rule differs")
    for seed in SEEDS:
        for key in METRICS:
            coarse = cases[32, seed, "GR"]["metrics"][key] - cases[32, seed, "GC"]["metrics"][key]
            fine = cases[64, seed, "GR"]["metrics"][key] - cases[64, seed, "GC"]["metrics"][key]
            contrast_shifts.append({"seed": seed, "metric": key, "GR_minus_GC_h32": coarse,
                "GR_minus_GC_h64": fine, "absolute_paired_contrast_shift": abs(fine - coarse),
                "shift_below_h64_margin": abs(fine - coarse) < abs(fine),
                "direction_changed": (coarse > 0) != (fine > 0)})
    both_pass = all(row["mean_relative_improvement"] >= .05 and row["positive_direction_seed_count"] >= 4
                    for row in paired if row["divisor"] == 64 and row["contrast"] == "GR-GC")
    if both_pass != root_summary["both_primary_outcomes_passed"]:
        raise AssertionError("Combined primary rule differs")
    after = check_originals()
    changed_during_audit = [str(path) for path, digest in input_hashes.items() if sha(path) != digest]
    if changed_during_audit:
        raise AssertionError({"audit_inputs_changed_during_read": changed_during_audit})
    return {"status": "passed", "method": "Independent raw-array sum/einsum; statistics.mean/stdev, no experiment helper imports.",
        "primary_divisor": 64, "numerical_check_divisor": 32, "seed_order": SEEDS, "frozen_setting": setting,
        "trajectory_metric_rows_recomputed": len(trajectories), "seed_metrics": seed_rows,
        "aggregate": aggregate, "paired": paired, "h32_h64_refinement": refinement,
        "h32_h64_paired_contrast_shifts": contrast_shifts, "both_primary_outcomes_passed": both_pass,
        "maximum_comparison_gap": MAX_GAP, "maximum_implicit_residual": max_midpoint_residual,
        "original_files_before": before, "original_files_after": after,
        "consumed_file_sha256": {path.relative_to(HERE).as_posix(): input_hashes[path] for path in sorted(consumed)},
        "frozen_evidence_and_all_consumed_hashes_unchanged_during_audit": True,
        "limits": ["Saved DOP853 continuous references checked against frozen data but not reintegrated by this audit.",
                   "Finite h32/h64 contrasts are numerical sensitivity checks, not certified continuous-limit bounds.",
                   "Five seeds and the pragmatic threshold do not establish population significance."]}, trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-confirmed-complete", action="store_true", required=True)
    parser.parse_args()
    report, rows = audit()
    (OUT / "fresh_confirmation_independent_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (OUT / "fresh_confirmation_independent_trajectory_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: report[key] for key in ["status", "trajectory_metric_rows_recomputed", "maximum_comparison_gap", "both_primary_outcomes_passed"]}))


if __name__ == "__main__":
    main()
