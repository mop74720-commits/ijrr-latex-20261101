"""Fresh Eq. (36) conservative-branch data and matched real/complex readouts.

The teacher has constant mass, no damping and no input. Its Eq. (36) map is
implemented through its algebraically equivalent implicit midpoint equations,
with a separate direct matrix-form check. Learned Hamiltonians use the same
implicit-midpoint prediction rule, but their unrestricted momentum quartics
are not claimed to retain the original physical coefficient parameterization.
Only linear readouts are trained; this is not an end-to-end complex MLP.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from types import SimpleNamespace

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results" / "learning"
INPUT_NAMES = ("reference_model_spec.json", "feature_model.py", "run_learning.py")
SEEDS = [17, 43, 91, 137, 211]
J = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verified_input_paths(verify_original_sources=False):
    """Resolve project-local inputs against their recorded hashes.

    Historical absolute source/copy paths remain in the manifest as provenance.
    An available original may restore a missing copy, but is not required when
    all local copies pass verification. Original-source verification is opt-in.
    """
    inputs = OUT / "inputs"
    manifest_path = OUT / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Verified input manifest is required: {manifest_path}")
    records = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    names = [Path(record["copy"]).name for record in records]
    if len(names) != len(set(names)) or set(names) != set(INPUT_NAMES):
        raise RuntimeError(f"Unexpected or duplicate inputs in {manifest_path}")
    resolved = {}
    for record, name in zip(records, names):
        expected = record["sha256"]
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise RuntimeError(f"Invalid recorded SHA-256 for {name}")
        target = inputs / name
        if not target.exists():
            source = Path(record["source"])
            if not source.is_file():
                raise FileNotFoundError(f"Missing verified project input: {target}; original unavailable: {source}")
            if sha256(source) != expected:
                raise RuntimeError(f"Original input fails recorded SHA-256: {source}")
            inputs.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        if sha256(target) != expected:
            raise RuntimeError(f"Project input fails recorded SHA-256: {target}")
        if verify_original_sources:
            source = Path(record["source"])
            if not source.is_file():
                raise FileNotFoundError(f"Requested original-source verification, but source is unavailable: {source}")
            if sha256(source) != expected:
                raise RuntimeError(f"Original input fails recorded SHA-256: {source}")
        resolved[name] = target
    return resolved


def source_setup(verify_original_sources=False):
    return load_model_spec(verify_original_sources=verify_original_sources)


def load_model_spec(verify_original_sources=False):
    source = _verified_input_paths(verify_original_sources)["reference_model_spec.json"]
    spec = json.loads(source.read_text(encoding="utf-8-sig"))
    return SimpleNamespace(
        nu=np.array(spec["nu"], dtype=float), d=np.array(spec["slope_map_d"], dtype=float),
        omega0=float(spec["omega0_rad_s"]), energy_scale=float(spec["energy_scale_J"]),
        tip_map_m=np.array(spec["tip_map_m"], dtype=float), spec=spec,
    )


def feature_module():
    path = OUT / "inputs" / "feature_model.py"
    name = "eq36_frozen_feature_model"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def initial_conditions(seed, count):
    rng = np.random.default_rng(seed)
    radius = np.column_stack([rng.uniform(.03, .20, count), rng.uniform(.03, .16, count)])
    phase = rng.uniform(0, 2 * np.pi, (count, 2))
    return np.c_[radius * np.cos(phase), -radius * np.sin(phase)]


def teacher_rhs(tau, state, model):
    state = np.asarray(state)
    q, p = state[:2], state[2:]
    theta = model.d @ q
    return np.r_[model.nu * p, -model.nu * q - model.d * (np.sin(theta) - theta)]


def teacher_energy(states, model):
    states = np.asarray(states)
    theta = states[..., :2] @ model.d
    return .5 * np.sum(model.nu * (states[..., :2] ** 2 + states[..., 2:] ** 2), axis=-1) + 2 * np.sin(theta / 2) ** 2 - theta ** 2 / 2


def teacher_coefficients(model, module):
    coeff = np.zeros(45)
    for j in range(4):
        exp = np.zeros(4, dtype=int)
        exp[j] = 2
        row = np.flatnonzero(np.all(module.EXP == exp, axis=1))[0]
        coeff[row] = model.nu[j % 2] / 2
    return coeff


def polynomial_parts(coeff, module):
    quadratic = np.zeros((4, 4))
    for coefficient, exponent in zip(coeff[:10], module.EXP[:10]):
        indices = np.flatnonzero(exponent)
        if len(indices) == 1:
            quadratic[indices[0], indices[0]] = 2 * coefficient
        else:
            i, j = indices
            quadratic[i, j] = quadratic[j, i] = coefficient
    return quadratic, np.asarray(coeff[10:], dtype=float), np.asarray(module.EXP[10:], dtype=np.int64)


@njit(cache=False)
def nonlinear_gradient(x, quartic, exponents, cosine, d):
    powers = np.ones((4, 5))
    for j in range(4):
        for degree in range(1, 5):
            powers[j, degree] = powers[j, degree - 1] * x[j]
    result = np.zeros(4)
    for k in range(len(quartic)):
        if quartic[k] == 0.:
            continue
        for j in range(4):
            if exponents[k, j] == 0:
                continue
            value = quartic[k] * exponents[k, j]
            for s in range(4):
                value *= powers[s, exponents[k, s] - (s == j)]
            result[j] += value
    theta = d[0] * x[0] + d[1] * x[1]
    factor = cosine * (math.sin(theta) - theta)
    result[0] += factor * d[0]
    result[1] += factor * d[1]
    return result


@njit(cache=False)
def _rollout_kernel(initial, h, steps, stride, quadratic, quartic, exponents, cosine, d, cayley, inv_j):
    count = steps // stride + 1 + (1 if steps % stride else 0)
    states = np.empty((count, 4))
    states[0] = initial
    x = initial.copy()
    saved = 1
    max_residual = 0.
    total_iterations = 0
    for step in range(steps):
        base = cayley @ x
        y = base.copy()
        converged = False
        for iteration in range(30):
            midpoint = (x + y) * .5
            nonlinear = nonlinear_gradient(midpoint, quartic, exponents, cosine, d)
            updated = base + h * (inv_j @ nonlinear)
            gap = np.max(np.abs(updated - y))
            y = updated
            if gap <= 2e-14 * (1 + np.max(np.abs(y))):
                converged = True
                break
        if not converged or not np.all(np.isfinite(y)):
            raise RuntimeError("Implicit midpoint iteration did not converge")
        total_iterations += iteration + 1
        midpoint = (x + y) * .5
        grad = quadratic @ midpoint + nonlinear_gradient(midpoint, quartic, exponents, cosine, d)
        field = np.array([grad[2], grad[3], -grad[0], -grad[1]])
        max_residual = max(max_residual, np.max(np.abs(y - x - h * field)))
        x = y
        if (step + 1) % stride == 0 or step + 1 == steps:
            states[saved] = x
            saved += 1
    return states, max_residual, total_iterations / steps


def rollout_feature_model(initials, step_tau, nsteps, sample_stride, coefficients, cosine, model):
    """Return (states[trajectory,sample,4], diagnostics) on a uniform step grid.

    Initials may be [4] or [n,4]. Samples are indices 0,stride,2*stride,...,
    with nsteps always appended if needed. No input state normalization is used.
    This API evaluates analytic readouts; it does not accept a Torch module.
    """
    module = feature_module()
    quadratic, quartic, exponents = polynomial_parts(np.asarray(coefficients), module)
    inverse = np.linalg.inv(np.eye(4) - .5 * step_tau * J @ quadratic)
    cayley = inverse @ (np.eye(4) + .5 * step_tau * J @ quadratic)
    inv_j = inverse @ J
    initials = np.atleast_2d(initials)
    results, residuals, iterations = [], [], []
    start = time.perf_counter()
    for initial in initials:
        states, residual, average_iterations = _rollout_kernel(np.asarray(initial), step_tau, nsteps, sample_stride, quadratic, quartic, exponents, float(cosine), model.d, cayley, inv_j)
        results.append(states)
        residuals.append(residual)
        iterations.append(average_iterations)
    return np.asarray(results), {"maximum_implicit_residual": float(max(residuals)), "mean_iterations_per_step": float(np.mean(iterations)), "elapsed_seconds": time.perf_counter() - start}


def grid(duration_s, model, requested_step, stride):
    steps = int(np.ceil(duration_s * model.omega0 / requested_step))
    h = duration_s * model.omega0 / steps
    indices = np.arange(0, steps + 1, stride, dtype=int)
    if indices[-1] != steps:
        indices = np.r_[indices, steps]
    return h, steps, indices * h / model.omega0


def generate_data(seed, config, model):
    module = feature_module()
    coeff = teacher_coefficients(model, module)
    arrays, diagnostics = {}, {}
    for split, duration, count, offset in [("train", config["train_duration_s"], config["train_trajectories"], 0), ("val", config["val_duration_s"], config["val_trajectories"], 100000)]:
        initials = initial_conditions(seed + offset, count)
        h, steps, times = grid(duration, model, config["step_tau"], 1)
        states, diag = rollout_feature_model(initials, h, steps, 1, coeff, 1., model)
        midpoints = .5 * (states[:, 1:] + states[:, :-1])
        clean = (states[:, 1:] - states[:, :-1]) / h
        sample_indices = np.arange(0, steps, config["training_stride"])
        sampled_midpoints = midpoints[:, sample_indices].reshape(-1, 4)
        sampled_clean = clean[:, sample_indices].reshape(-1, 4)
        rng = np.random.default_rng(seed + offset + 10000)
        noise = rng.uniform(-config["noise_bound"], config["noise_bound"], sampled_clean.shape)
        arrays.update({f"{split}_initials": initials, f"{split}_midpoints": sampled_midpoints, f"{split}_labels": sampled_clean + noise, f"{split}_clean_labels": sampled_clean, f"{split}_noise": noise, f"{split}_trajectory_id": np.repeat(np.arange(count), len(sample_indices)), f"{split}_step_tau": np.array(h), f"{split}_sample_states": states[:, ::config["training_stride"]], f"{split}_sample_times_s": times[::config["training_stride"]]})
        exact = np.array([teacher_rhs(0., x, model) for x in sampled_midpoints])
        diag["midpoint_label_consistency_max"] = float(np.max(np.abs(sampled_clean - exact)))
        diagnostics[split] = diag
    test_initials = initial_conditions(seed + 200000, config["test_trajectories"])
    h, steps, times_s = grid(config["test_duration_s"], model, config["step_tau"], config["test_sample_stride"])
    teacher_states, diagnostics["test_teacher"] = rollout_feature_model(test_initials, h, steps, config["test_sample_stride"], coeff, 1., model)
    times_tau = times_s * model.omega0
    references = []
    start = time.perf_counter()
    for initial in test_initials:
        result = solve_ivp(lambda t, x: teacher_rhs(t, x, model), (0., times_tau[-1]), initial, t_eval=times_tau, method="DOP853", rtol=2e-11, atol=2e-13)
        if not result.success:
            raise RuntimeError(result.message)
        references.append(result.y.T)
    diagnostics["test_reference"] = {"method": "DOP853", "rtol": 2e-11, "atol": 2e-13, "elapsed_seconds": time.perf_counter() - start}
    arrays.update(test_initials=test_initials, test_times=times_s, test_times_tau=times_tau,
                  test_teacher_states=teacher_states, test_reference_states=np.asarray(references),
                  teacher_step_tau=np.array(h), teacher_nsteps=np.array(steps), teacher_sample_stride=np.array(config["test_sample_stride"]),
                  nu=model.nu, d=model.d, omega0=np.array(model.omega0), energy_scale_J=np.array(model.energy_scale), tip_map_m=model.tip_map_m,
                  source_spec_sha256=np.array(sha256(OUT / "inputs" / "reference_model_spec.json")), seed=np.array(seed))
    target = OUT / "data" / f"seed_{seed}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    save_json(OUT / "data" / f"seed_{seed}_diagnostics.json", diagnostics)
    print(f"DATA_READY seed={seed} path={target.as_posix()} train={len(arrays['train_labels'])} val={len(arrays['val_labels'])}", flush=True)


def jfeatures(gradients):
    return gradients[:, [2, 3, 0, 1], :] * np.array([1., 1., -1., -1.])[None, :, None]


def fit_readout(data, model, module, basis, prior, representation, ridge):
    start = time.perf_counter()
    first = 10 if prior else 0
    train = data["train_midpoints"]
    labels = data["train_labels"]
    g = module.gradients(train, model, basis, representation)
    design = jfeatures(g).reshape(-1, 45)[:, first:]
    physical_linear = np.c_[train[:, 2:] * model.nu, -train[:, :2] * model.nu]
    offset = physical_linear if prior else np.zeros_like(labels)
    response = (labels - offset).ravel()
    normalization = np.sqrt(len(response))
    mapping = block_diag(module.TRANSFORM[:10, :10], np.eye(35)) if representation == "complex" else np.eye(45)
    mapping = mapping[first:, first:]
    weights = np.linalg.lstsq(np.vstack([design / normalization, np.sqrt(ridge) * mapping]), np.r_[response / normalization, np.zeros(45 - first)], rcond=None)[0]
    coeff, cosine = module.decode_weights(weights, prior, representation, model, basis)
    common = mapping @ weights
    real_design = jfeatures(module.gradients(train, model, basis, "real")).reshape(-1, 45)[:, first:]
    consistency = np.max(np.abs(design - real_design @ mapping))
    train_prediction = (real_design @ common).reshape(-1, 4) + offset
    val = data["val_midpoints"]
    val_design = jfeatures(module.gradients(val, model, basis, representation)).reshape(-1, 45)[:, first:]
    val_offset = np.c_[val[:, 2:] * model.nu, -val[:, :2] * model.nu] if prior else np.zeros_like(val)
    val_prediction = (val_design @ weights).reshape(-1, 4) + val_offset
    return coeff, cosine, weights, {"fit_seconds": time.perf_counter() - start, "real_parameter_count": 45 - first,
        "train_mse": float(np.mean((train_prediction - labels) ** 2)),
        "train_clean_mse": float(np.mean((train_prediction - data["train_clean_labels"]) ** 2)),
        "val_mse": float(np.mean((val_prediction - data["val_labels"]) ** 2)),
        "val_clean_relative_l2": float(np.linalg.norm(val_prediction - data["val_clean_labels"]) / np.linalg.norm(data["val_clean_labels"])),
        "matched_feature_gradient_max_gap": float(consistency), "design_rank": int(np.linalg.matrix_rank(design))}


def metrics(predictions, reference, model):
    error = predictions - reference
    tip_error = error[..., :2] @ model.tip_map_m
    energy_error = (teacher_energy(predictions, model) - teacher_energy(reference, model)) * model.energy_scale
    phase_prediction = np.unwrap(np.arctan2(-predictions[..., 2:], predictions[..., :2]), axis=1)
    phase_reference = np.unwrap(np.arctan2(-reference[..., 2:], reference[..., :2]), axis=1)
    phase_error = phase_prediction - phase_reference
    return {"state_rmse": float(np.sqrt(np.mean(error ** 2))),
        "state_relative_l2": float(np.linalg.norm(error) / np.linalg.norm(reference)),
        "tip_rmse_mm": float(np.sqrt(np.mean(tip_error ** 2)) * 1e3),
        "tip_max_error_mm": float(np.max(np.abs(tip_error)) * 1e3),
        "phase_rmse_rad": float(np.sqrt(np.mean(phase_error ** 2))),
        "phase_max_error_rad": float(np.max(np.abs(phase_error))),
        "true_energy_rmse_nJ": float(np.sqrt(np.mean(energy_error ** 2)) * 1e9)}


def write_csv(path, records):
    if not records:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def fit_seed(seed, config, model):
    module = feature_module()
    basis = module.make_basis(model)
    data = np.load(OUT / "data" / f"seed_{seed}.npz", allow_pickle=False)
    records, trajectory_records = [], []
    fits_dir = OUT / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)
    for prior, representation, name in [(False, "real", "R"), (False, "complex", "C"), (True, "real", "GR"), (True, "complex", "GC")]:
        coeff, cosine, weights, stats = fit_readout(data, model, module, basis, prior, representation, config["ridge"])
        predictions, solve_stats = rollout_feature_model(data["test_initials"], float(data["teacher_step_tau"]), int(data["teacher_nsteps"]), int(data["teacher_sample_stride"]), coeff, cosine, model)
        discrete = metrics(predictions, data["test_teacher_states"], model)
        continuous = metrics(predictions, data["test_reference_states"], model)
        own_energy = np.array([[module.energy_gradient(x, coeff, cosine, model.d)[0] for x in trajectory] for trajectory in predictions])
        own_drift = np.max(np.abs(own_energy - own_energy[:, :1]), axis=1) / np.maximum(np.abs(own_energy[:, 0]), 1e-12)
        record = {"seed": seed, "model": name, "prior": prior, "representation": representation, **stats,
                  "rollout_seconds": solve_stats["elapsed_seconds"], "maximum_implicit_residual": solve_stats["maximum_implicit_residual"],
                  "mean_iterations_per_step": solve_stats["mean_iterations_per_step"], "own_energy_relative_drift_max": float(np.max(own_drift)),
                  "cosine_multiplier": float(cosine), **{"discrete_" + k: v for k, v in discrete.items()}, **{"continuous_" + k: v for k, v in continuous.items()}}
        records.append(record)
        for trajectory_index in range(len(predictions)):
            dmetrics = metrics(predictions[trajectory_index:trajectory_index + 1], data["test_teacher_states"][trajectory_index:trajectory_index + 1], model)
            cmetrics = metrics(predictions[trajectory_index:trajectory_index + 1], data["test_reference_states"][trajectory_index:trajectory_index + 1], model)
            trajectory_records.append({"seed": seed, "model": name, "test_trajectory": trajectory_index,
                                      **{"discrete_" + k: v for k, v in dmetrics.items()}, **{"continuous_" + k: v for k, v in cmetrics.items()}})
        np.savez_compressed(fits_dir / f"seed_{seed}_{name}.npz", readout_weights=weights, polynomial_coefficients=coeff,
                            cosine_multiplier=np.array(cosine), predictions=predictions, own_energy=own_energy,
                            test_times=data["test_times"], d=model.d, nu=model.nu, exponents=module.EXP)
        save_json(fits_dir / f"seed_{seed}_{name}_metrics.json", record)
        print(f"FIT_READY seed={seed} model={name} teacher_tip_rmse_mm={discrete['tip_rmse_mm']:.6g} continuous_tip_rmse_mm={continuous['tip_rmse_mm']:.6g}", flush=True)
    write_csv(fits_dir / f"seed_{seed}_metrics.csv", records)
    write_csv(fits_dir / f"seed_{seed}_trajectory_metrics.csv", trajectory_records)
    return records, trajectory_records


def summarize(config, model):
    records, trajectory_records = [], []
    for seed in config["seeds"]:
        with (OUT / "fits" / f"seed_{seed}_metrics.csv").open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                for key in row:
                    if key not in ("model", "prior", "representation"):
                        row[key] = float(row[key])
                row["seed"] = int(row["seed"])
                records.append(row)
        with (OUT / "fits" / f"seed_{seed}_trajectory_metrics.csv").open(encoding="utf-8") as stream:
            trajectory_records.extend(list(csv.DictReader(stream)))
    contrasts = []
    metric_names = ["discrete_tip_rmse_mm", "continuous_tip_rmse_mm", "discrete_state_relative_l2", "discrete_phase_rmse_rad", "fit_seconds", "rollout_seconds"]
    definitions = [("R-GR", {"R": 1, "GR": -1}), ("C-GC", {"C": 1, "GC": -1}),
                   ("R-C", {"R": 1, "C": -1}), ("GR-GC", {"GR": 1, "GC": -1}),
                   ("(GR-GC)-(R-C)", {"GR": 1, "GC": -1, "R": -1, "C": 1})]
    for seed in config["seeds"]:
        by_model = {row["model"]: row for row in records if row["seed"] == seed}
        for name, weights in definitions:
            contrasts.append({"seed": seed, "contrast": name, **{metric: sum(w * by_model[group][metric] for group, w in weights.items()) for metric in metric_names}})
    aggregates = {}
    for name, _ in definitions:
        chosen = [row for row in contrasts if row["contrast"] == name]
        aggregates[name] = {metric: {"mean": float(np.mean([row[metric] for row in chosen])), "sample_sd": float(np.std([row[metric] for row in chosen], ddof=1)), "n_independent_seeds": len(chosen)} for metric in metric_names}
    teacher_errors = []
    for seed in config["seeds"]:
        data = np.load(OUT / "data" / f"seed_{seed}.npz", allow_pickle=False)
        teacher_errors.append({"seed": seed, **metrics(data["test_teacher_states"], data["test_reference_states"], model)})
    summary = {"scope": "constant-mass conservative Eq36 teacher; trained fixed-feature linear readouts; no general complex H+iG, damping, input, contact or KAM certification claim",
               "config": config, "model_averages": {}, "five_paired_contrasts": aggregates,
               "teacher_vs_continuous_reference": teacher_errors,
               "inference": "Five independent data/noise seeds; descriptive paired means and sample SD only. Time samples are not independent replicates. CPU timings are measured local wall times, not controlled hardware benchmarks."}
    for group in ("R", "C", "GR", "GC"):
        chosen = [row for row in records if row["model"] == group]
        summary["model_averages"][group] = {metric: {"mean": float(np.mean([row[metric] for row in chosen])), "sample_sd": float(np.std([row[metric] for row in chosen], ddof=1))} for metric in metric_names + ["own_energy_relative_drift_max", "val_clean_relative_l2", "maximum_implicit_residual"]}
    write_csv(OUT / "model_metrics.csv", records)
    write_csv(OUT / "trajectory_metrics.csv", trajectory_records)
    write_csv(OUT / "paired_contrasts.csv", contrasts)
    write_csv(OUT / "teacher_discretization_metrics.csv", teacher_errors)
    save_json(OUT / "summary.json", summary)
    return summary


def verify_implementation(model):
    module = feature_module()
    coeff = teacher_coefficients(model, module)
    points = initial_conditions(61973, 20)
    h = .02
    steps, diagnostics = rollout_feature_model(points, h, 1, 1, coeff, 1., model)
    direct_gaps, structure_defects = [], []
    nmat = np.diag(model.nu)
    for initial, path in zip(points, steps):
        end = path[-1]
        mid = .5 * (initial + end)
        theta = model.d @ mid[:2]
        factor = np.sinc(theta / np.pi) - 1
        stiffness = nmat + factor * np.outer(model.d, model.d)
        epsilon = .25 * h * h * stiffness @ nmat
        momentum = np.linalg.solve(np.eye(2) + epsilon, (np.eye(2) - epsilon) @ initial[2:] - h * stiffness @ initial[:2])
        position = initial[:2] + .5 * h * model.nu * (momentum + initial[2:])
        direct_gaps.append(np.max(np.abs(end - np.r_[position, momentum])))
        hessian = np.diag(np.r_[model.nu, model.nu])
        hessian[:2, :2] += (np.cos(theta) - 1) * np.outer(model.d, model.d)
        tangent = np.linalg.solve(np.eye(4) - .5 * h * J @ hessian, np.eye(4) + .5 * h * J @ hessian)
        structure_defects.append(np.linalg.norm(tangent.T @ J @ tangent - J, ord=np.inf))
    x0 = points[0]
    interval = 4.
    reference = solve_ivp(lambda t, x: teacher_rhs(t, x, model), (0, interval), x0, method="DOP853", rtol=2e-13, atol=2e-15).y[:, -1]
    convergence = []
    for h0 in [.08, .04, .02, .01]:
        steps_count = round(interval / h0)
        path, _ = rollout_feature_model(x0, h0, steps_count, steps_count, coeff, 1., model)
        convergence.append({"step_tau": h0, "endpoint_error": float(np.linalg.norm(path[0, -1] - reference))})
    for i in range(1, len(convergence)):
        convergence[i]["observed_order"] = math.log(convergence[i - 1]["endpoint_error"] / convergence[i]["endpoint_error"], 2)
    basis = module.make_basis(model)
    real = module.gradients(points, model, basis, "real")
    complex_g = module.gradients(points, model, basis, "complex")
    transform = block_diag(module.TRANSFORM[:10, :10], np.eye(35))
    feature_gap = float(np.max(np.abs(complex_g - real @ transform)))
    coarse = solve_ivp(lambda t, x: teacher_rhs(t, x, model), (0, 40 * model.omega0), x0, t_eval=np.linspace(0, 40 * model.omega0, 1001), method="DOP853", rtol=2e-11, atol=2e-13)
    refined = solve_ivp(lambda t, x: teacher_rhs(t, x, model), (0, 40 * model.omega0), x0, t_eval=coarse.t, method="DOP853", rtol=2e-13, atol=2e-15)
    reference_gap = float(np.max(np.abs(coarse.y - refined.y)))
    result = {"direct_eq36_vs_midpoint_max_gap": float(max(direct_gaps)), "canonical_tangent_defect_max": float(max(structure_defects)),
              "matched_features_max_gap": feature_gap, "reference_40s_refinement_max_state_gap": reference_gap,
              "convergence": convergence, "single_step_solver": diagnostics,
              "claim_boundary": "This verifies the explicitly declared constant-mass conservative specialization only; it does not repair or certify the frozen general Birkhoff derivation."}
    if max(direct_gaps) > 1e-11 or max(structure_defects) > 1e-10 or feature_gap > 1e-10:
        raise AssertionError(result)
    if not all(1.8 < row["observed_order"] < 2.2 for row in convergence[-2:]):
        raise AssertionError("Unexpected conservative specialization convergence order")
    save_json(OUT / "implementation_checks.json", result)
    print("IMPLEMENTATION_CHECKS_PASSED", json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["verify", "data", "fit", "summary", "all"], default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    threadpool_limits(1)
    OUT.mkdir(parents=True, exist_ok=True)
    config = {"seeds": args.seeds, "train_trajectories": 12, "val_trajectories": 4, "test_trajectories": 8,
              "train_duration_s": 2., "val_duration_s": 4., "test_duration_s": 40., "step_tau": .01,
              "training_stride": 8, "test_sample_stride": 20, "noise_bound": 1e-6, "ridge": 1e-14,
              "state_order": ["X1", "X2", "Y1", "Y2"], "label_units": "dimensionless dz/dtau", "test_times_units": "SI seconds",
              "normalization": "No data-dependent normalization; use frozen canonical physical chart.",
              "supervision": "Midpoint finite increments generated exclusively by the conservative constant-mass Eq36 teacher, plus bounded uniform noise.",
              "validation": "Independent whole trajectories; fixed predeclared ridge, no test-set hyperparameter selection.",
              "prediction": "All fitted models use converged implicit midpoint; unrestricted learned momentum quartics are not claimed to retain the original physical coefficient form."}
    config_path = OUT / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise RuntimeError("Existing frozen configuration differs; use the recorded seed list.")
    else:
        save_json(config_path, config)
    model = source_setup()
    save_json(OUT / "environment.json", {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "numpy": np.__version__, "script_sha256": sha256(__file__), "threads": 1})
    if args.stage in ("verify", "data", "all"):
        verify_implementation(model)
    else:
        module = feature_module()
        rollout_feature_model(np.zeros(4), .01, 1, 1, teacher_coefficients(model, module), 1., model)
    if args.stage in ("data", "all"):
        for seed in args.seeds:
            if (OUT / "data" / f"seed_{seed}.npz").exists():
                print(f"DATA_EXISTS seed={seed}; preserving previous output", flush=True)
            else:
                generate_data(seed, config, model)
        save_json(OUT / "data_ready.json", {"seeds": args.seeds, "files": [{"seed": seed, "path": (OUT / "data" / f"seed_{seed}.npz").as_posix(), "sha256": sha256(OUT / "data" / f"seed_{seed}.npz")} for seed in args.seeds]})
        print("ALL_DATA_READY", flush=True)
    if args.stage in ("fit", "all"):
        for seed in args.seeds:
            fit_seed(seed, config, model)
    if args.stage in ("fit", "summary", "all"):
        result = summarize(config, model)
        print("SUMMARY_READY", json.dumps(result["model_averages"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
