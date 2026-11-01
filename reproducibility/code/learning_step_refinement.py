"""Frozen seed-17 weight study separating integration and identification error.

All five Hamiltonians (teacher, R, C, GR, GC) use the original eight held-out
initial states and the same stored DOP853 output grid. There is no retraining
or best-model selection. The implicit solver uses preconditioned fixed-point
iterations, not Newton iterations. Existing baseline files are read only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

import learning_experiment as baseline


def main():
    threadpool_limits(1)
    out = baseline.OUT / "refinement"
    out.mkdir(parents=True, exist_ok=True)
    seed = 17
    factors = [1, 2, 4]
    data_path = baseline.OUT / "data" / f"seed_{seed}.npz"
    data = np.load(data_path, allow_pickle=False)
    model = baseline.load_model_spec()
    features = baseline.feature_module()
    original_h = float(data["teacher_step_tau"])
    original_n = int(data["teacher_nsteps"])
    original_stride = int(data["teacher_sample_stride"])
    times_tau = data["test_times_tau"]
    reference = data["test_reference_states"]
    initials = data["test_initials"]
    input_records = [{"role": "fixed test data and independent reference", "path": data_path.as_posix(), "sha256": baseline.sha256(data_path)}]
    frozen = {"teacher": (baseline.teacher_coefficients(model, features), 1., data["test_teacher_states"])}
    for name in ["R", "C", "GR", "GC"]:
        path = baseline.OUT / "fits" / f"seed_{seed}_{name}.npz"
        loaded = np.load(path, allow_pickle=False)
        frozen[name] = (loaded["polynomial_coefficients"], float(loaded["cosine_multiplier"]), loaded["predictions"])
        input_records.append({"role": "unchanged learned coefficients", "model": name, "path": path.as_posix(), "sha256": baseline.sha256(path)})
    protocol = {"seed": seed, "models": list(frozen), "step_refinement_factors": factors,
                "initial_states": "the same eight fixed held-out states from seed_17.npz",
                "prediction_duration_s": float(data["test_times"][-1]), "retraining": False,
                "reference": "the unchanged stored DOP853 reference at identical output times",
                "original_step_tau": original_h, "original_nsteps": original_n,
                "sample_alignment": "nsteps and sample_stride are multiplied by the refinement factor; step is divided by it; include final step",
                "implicit_solver": "linearly preconditioned fixed-point iteration, maximum 30 iterations; same tolerance as baseline",
                "scope": "the declared constant-mass conservative example only",
                "command": f'"{sys.executable}" -B "{Path(__file__).resolve()}"',
                "code_sha256": baseline.sha256(__file__), "baseline_code_sha256": baseline.sha256(baseline.__file__),
                "inputs": input_records}
    baseline.save_json(out / "protocol.json", protocol)
    # JIT compilation happens before timed model evaluations.
    baseline.rollout_feature_model(np.zeros(4), original_h, 1, 1, frozen["teacher"][0], 1., model)
    rows, trajectory_rows, grid_checks = [], [], []
    predictions = {}
    for factor in factors:
        h = original_h / factor
        nsteps = original_n * factor
        stride = original_stride * factor
        indices = np.arange(0, nsteps + 1, stride, dtype=int)
        if indices[-1] != nsteps:
            indices = np.r_[indices, nsteps]
        obtained_times = indices * h
        time_gap = float(np.max(np.abs(obtained_times - times_tau)))
        if obtained_times.shape != times_tau.shape or time_gap > 1e-10:
            raise AssertionError(f"Output grid mismatch at refinement {factor}: {time_gap}")
        grid_checks.append({"factor": factor, "samples": len(indices), "max_tau_grid_difference": time_gap})
        for name, (coeff, cosine, original_predictions) in frozen.items():
            path, diagnostics = baseline.rollout_feature_model(initials, h, nsteps, stride, coeff, cosine, model)
            if not np.all(np.isfinite(path)):
                raise AssertionError(f"Non-finite prediction: {name}, factor {factor}")
            reproduce_gap = float(np.max(np.abs(path - original_predictions))) if factor == 1 else None
            if factor == 1 and reproduce_gap > 1e-12:
                raise AssertionError(f"Baseline reproduction failed for {name}: {reproduce_gap}")
            values = baseline.metrics(path, reference, model)
            row = {"seed": seed, "model": name, "refinement_factor": factor, "step_tau": h,
                   "nsteps": nsteps, "sample_stride": stride, "elapsed_seconds": diagnostics["elapsed_seconds"],
                   "maximum_implicit_residual": diagnostics["maximum_implicit_residual"],
                   "mean_fixed_point_iterations_per_step": diagnostics["mean_iterations_per_step"],
                   "baseline_reproduction_max_state_gap": reproduce_gap,
                   **values}
            rows.append(row)
            predictions[(name, factor)] = path
            for trajectory_index in range(len(initials)):
                trajectory_rows.append({"seed": seed, "model": name, "refinement_factor": factor,
                                        "test_trajectory": trajectory_index,
                                        **baseline.metrics(path[trajectory_index:trajectory_index + 1], reference[trajectory_index:trajectory_index + 1], model)})
            np.savez_compressed(out / f"{name}_factor_{factor}.npz", predictions=path, test_times=data["test_times"],
                                test_times_tau=times_tau, polynomial_coefficients=coeff, cosine_multiplier=np.array(cosine),
                                step_tau=np.array(h), test_initials=initials)
            print(f"REFINEMENT_READY model={name} factor={factor} tip_rmse_mm={values['tip_rmse_mm']:.9g} phase_rmse_rad={values['phase_rmse_rad']:.9g}", flush=True)
    for row in rows:
        name, factor = row["model"], row["refinement_factor"]
        previous = next((r for r in rows if r["model"] == name and r["refinement_factor"] == factor // 2), None)
        row["tip_rmse_observed_order"] = float(np.log2(previous["tip_rmse_mm"] / row["tip_rmse_mm"])) if previous else None
        row["phase_rmse_observed_order"] = float(np.log2(previous["phase_rmse_rad"] / row["phase_rmse_rad"])) if previous else None
        row["state_rmse_observed_order"] = float(np.log2(previous["state_rmse"] / row["state_rmse"])) if previous else None
        if name != "teacher":
            gap = baseline.metrics(predictions[(name, factor)], predictions[("teacher", factor)], model)
            row["same_step_learning_tip_rmse_mm"] = gap["tip_rmse_mm"]
            row["same_step_learning_phase_rmse_rad"] = gap["phase_rmse_rad"]
        else:
            row["same_step_learning_tip_rmse_mm"] = 0.
            row["same_step_learning_phase_rmse_rad"] = 0.
    unchanged = all(baseline.sha256(item["path"]) == item["sha256"] for item in input_records)
    if not unchanged:
        raise AssertionError("An input changed during the refinement study")
    result = {"protocol": protocol, "grid_checks": grid_checks, "inputs_unchanged": unchanged, "results": rows,
              "interpretation": "Compare error reduction from step refinement at frozen weights. The three levels may demonstrate second-order integration error; they do not establish the eventual learning-error plateau or general model superiority.",
              "solver_terminology": "Residuals refer to the converged implicit equations. No Newton iteration was used."}
    baseline.write_csv(out / "metrics.csv", rows)
    baseline.write_csv(out / "trajectory_metrics.csv", trajectory_rows)
    baseline.save_json(out / "summary.json", result)
    print("REFINEMENT_COMPLETE", json.dumps({"inputs_unchanged": unchanged, "maximum_grid_gap": max(r["max_tau_grid_difference"] for r in grid_checks), "records": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
