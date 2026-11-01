"""Fixed-weight midpoint refinement against a quadratic physical baseline.

Uses the 10000-update neural experiment, all five seeds and the original
914 output times. This script never trains models or alters existing inputs.
Run --stage plan before --stage verify/run; --stage assets only renders outputs.
"""
from __future__ import annotations

import argparse
import ast
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import numba
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "neural_physics_refinement"
SOURCE = ROOT / "results" / "neural_learning_10000"
DATA = ROOT / "results" / "learning" / "data"
SEEDS = [17, 43, 91, 137, 211]
GROUPS = ["H0", "R", "C", "GR", "GC"]
FACTORS = [1, 2, 4, 8]
TOL = 2e-14
PROTOCOL_SCHEMA_VERSION = 2
LEGACY_PROJECT_ROOT = PureWindowsPath("E:/1030/IJRR_1025_20261030")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def protocol_input_id(value, *, legacy=False):
    """Accept canonical project paths; migrate only the recorded original root."""
    if not isinstance(value, str) or not value:
        raise RuntimeError("Protocol input path must be a nonempty string.")
    windows = PureWindowsPath(value)
    if windows.is_absolute():
        if not legacy:
            raise RuntimeError(f"Absolute input path in portable protocol: {value}")
        try:
            value = windows.relative_to(LEGACY_PROJECT_ROOT).as_posix()
        except ValueError as error:
            raise RuntimeError(f"Unknown absolute protocol input source: {value}") from error
    path = PurePosixPath(value)
    if (path.is_absolute() or windows.drive and not windows.is_absolute()
            or "\\" in value or ":" in value or ".." in path.parts
            or path.as_posix() != value or value == "."):
        raise RuntimeError(f"Invalid project-relative protocol input path: {value}")
    return path.as_posix()


def protocol_input_file(value):
    path = (ROOT / protocol_input_id(value)).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise RuntimeError(f"Protocol input escapes the project: {value}")
    return path


def normalize_protocol(previous):
    """Keep every scientific setting and hash while normalizing known old paths."""
    version = previous.get("schema_version")
    if version not in (None, PROTOCOL_SCHEMA_VERSION):
        raise RuntimeError(f"Unsupported refinement protocol schema: {version}")
    normalized = dict(previous)
    normalized["schema_version"] = PROTOCOL_SCHEMA_VERSION
    normalized["inputs"] = [
        {**item, "path": protocol_input_id(item["path"], legacy=version is None)}
        for item in previous["inputs"]
    ]
    return normalized


def plan():
    inputs = [ROOT / "code" / "train_neural_hnn.py", SOURCE / "protocol.json"]
    for seed in SEEDS:
        inputs.append(DATA / f"seed_{seed}.npz")
        for group in GROUPS[1:]:
            inputs.extend([SOURCE / f"seed_{seed}" / f"{group}_weights.npz",
                           SOURCE / f"seed_{seed}" / f"{group}_rollout.npz"])
    manifest = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)} for p in inputs]
    protocol = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "objective": "Evaluate every saved 10000-update model and a zero-parameter quadratic physical baseline under fixed-weight midpoint step refinement, without assuming any model wins.",
        "seeds": SEEDS, "models": GROUPS, "step_divisors": FACTORS,
        "network_training_updates": 10000, "training": "None; all saved weights remain fixed.",
        "reference": "Original saved DOP853 nonlinear continuous reference; additionally recompute the full nonlinear Hamiltonian teacher at each matching step.",
        "grid": "Use all 914 original saved times, including t=0, actual endpoint 15.992993212174294 s; original stride 20, refined stride 20*divisor.",
        "solver": "Implicit midpoint with the known physical quadratic part solved by its exact Cayley matrix; fixed-point solve for the remaining vector field, float64, fastmath=False.",
        "stopping_absolute_midpoint_residual": TOL, "maximum_iterations": 40,
        "baseline": "H0=0.5*sum_j nu_j*(X_j^2+Y_j^2); analytic Cayley rotations at each step, plus its exact continuous rotation.",
        "statistics": "For each trajectory compute RMSE over the same saved times; average eight trajectory RMSEs within seed; report the mean and sample SD (ddof=1) across all five seeds. Paired differences use the same seed. No significance or general superiority claim.",
        "primary_metrics": ["continuous_tip_rmse_mm", "matched_teacher_tip_rmse_mm", "continuous_state_relative_l2", "matched_teacher_state_relative_l2"],
        "secondary_metrics": ["target_energy_drift_relative", "own_energy_drift_relative", "midpoint_residual", "nonlinear_force_to_linear_force_training_L2"],
        "verification": "Independent original NumPy field comparison and finite differences; reproduce all 20 original-h saved network trajectories and all five original-h teacher trajectories before accepting summaries.",
        "retained_outputs": "Common-grid float64 trajectories only, per-trajectory and per-seed CSVs, paired differences, summary JSON, source hashes, solver diagnostics, vector PDF and 300dpi PNG; no internal substep trajectories.",
        "claim_limits": "Time samples and eight trajectories are not independent training replicates. The common output grid resolves the same stored states at all steps. Fixed-weight refinement does not reduce learned vector-field bias or prove KAM/physical calibration.",
        "inputs": manifest,
    }
    path = OUT / "protocol.json"
    if path.exists():
        original_bytes = path.read_bytes()
        previous = json.loads(original_bytes.decode("utf-8"))
        if normalize_protocol(previous) != protocol:
            raise RuntimeError("Existing protocol or input hashes differ; preserve the original experiment.")
        if previous != protocol:
            backup = path.with_name("protocol.legacy_absolute_paths.json")
            if backup.exists():
                if backup.read_bytes() != original_bytes:
                    raise RuntimeError("Legacy protocol backup differs; preserve both protocol records.")
            else:
                with backup.open("xb") as stream:
                    stream.write(original_bytes)
            save_json(path, protocol)
    else:
        save_json(path, protocol)
    return protocol


def load_seed(seed):
    with np.load(DATA / f"seed_{seed}.npz", allow_pickle=False) as source:
        arrays = {k: source[k] for k in ["test_initials", "nu", "d", "omega0", "tip_map_m", "teacher_step_tau", "teacher_sample_stride", "train_midpoints", "val_midpoints"]}
    with np.load(SOURCE / f"seed_{seed}" / "GR_rollout.npz", allow_pickle=False) as source:
        arrays.update(times=source["times_seconds"], reference=source["reference_states"], old_teacher=source["teacher_states"])
    if arrays["reference"].shape != (914, 8, 4):
        raise AssertionError(arrays["reference"].shape)
    return arrays


def weights_for(seed, group):
    with np.load(SOURCE / f"seed_{seed}" / f"{group}_weights.npz", allow_pickle=False) as source:
        w = {k: np.array(source[k], copy=True, order="C") for k in source.files}
    return (w["w1"], w["b1"], w["w2"], w["b2"], w["w3"], w["b3"],
            w["scale"], float(w["energy_scale"]), w["nu"], bool(w["prior"])), w


@njit(cache=False, nogil=True)
def neural_field(state, w1, b1, w2, b2, w3, b3, scale, escale, nu, prior):
    n1, n2 = len(b1), len(b2)
    first, second = np.empty(n1), np.empty(n2)
    for j in range(n1):
        value = b1[j]
        for k in range(4):
            value += w1[j, k] * state[k] / scale[k]
        first[j] = np.tanh(value)
    energy = b3[0]
    for j in range(n2):
        value = b2[j]
        for k in range(n1):
            value += w2[j, k] * first[k]
        second[j] = np.tanh(value)
        energy += w3[j] * second[j]
    energy *= escale
    gradient = np.zeros(4)
    for k in range(n1):
        back = 0.
        for j in range(n2):
            back += w3[j] * (1. - second[j] * second[j]) * w2[j, k]
        back *= 1. - first[k] * first[k]
        for j in range(4):
            gradient[j] += back * w1[k, j]
    for j in range(4):
        gradient[j] *= escale / scale[j]
        if prior:
            gradient[j] += state[j] * nu[j % 2]
            energy += .5 * nu[j % 2] * state[j] ** 2
    return energy, np.array([gradient[2], gradient[3], -gradient[0], -gradient[1]])


@njit(cache=False, nogil=True)
def physical_linear_field(state, nu):
    return np.array([nu[0] * state[2], nu[1] * state[3], -nu[0] * state[0], -nu[1] * state[1]])


def cayley_preconditioner(h, nu):
    linear = np.zeros((4, 4))
    linear[:2, 2:] = np.diag(nu)
    linear[2:, :2] = -np.diag(nu)
    inverse = np.linalg.inv(np.eye(4) - h * linear / 2)
    return np.ascontiguousarray(inverse @ (np.eye(4) + h * linear / 2)), np.ascontiguousarray(inverse)


@njit(cache=False, nogil=True)
def neural_rollout(initials, h, nsteps, stride, cayley, inverse, params):
    ntime, count = nsteps // stride + 1, len(initials)
    result = np.empty((ntime, count, 4))
    maximum_residual, maximum_iterations, total_iterations = 0., 0, 0
    for trajectory in range(count):
        old = initials[trajectory].copy()
        result[0, trajectory] = old
        for step in range(1, nsteps + 1):
            base = cayley @ old
            new = base.copy()
            converged = False
            for iteration in range(40):
                mid = (old + new) / 2
                field = neural_field(mid, *params)[1]
                remainder = field - physical_linear_field(mid, params[8])
                updated = base + h * (inverse @ remainder)
                gap = np.max(np.abs(updated - new))
                new = updated
                if gap < 1e-13:
                    mid = (old + new) / 2
                    residual = np.max(np.abs(new - old - h * neural_field(mid, *params)[1]))
                    if residual <= TOL:
                        converged = True
                        break
            if not converged:
                raise RuntimeError("Neural midpoint residual failed to converge")
            maximum_residual = max(maximum_residual, residual)
            maximum_iterations = max(maximum_iterations, iteration + 1)
            total_iterations += iteration + 1
            old = new
            if not np.all(np.isfinite(old)) or np.max(np.abs(old)) > 100:
                raise RuntimeError("Neural trajectory became nonfinite or exceeded bound 100")
            if step % stride == 0:
                result[step // stride, trajectory] = old
    return result, maximum_residual, maximum_iterations, total_iterations / (nsteps * count)


@njit(cache=False, nogil=True)
def teacher_rollout(initials, h, nsteps, stride, cayley, inverse, nu, d):
    count = len(initials)
    result = np.empty((nsteps // stride + 1, count, 4))
    maximum_residual, max_iterations, total = 0., 0, 0
    for trajectory in range(count):
        old = initials[trajectory].copy()
        result[0, trajectory] = old
        for step in range(1, nsteps + 1):
            base = cayley @ old
            new = base.copy()
            for iteration in range(40):
                mid = (old + new) / 2
                theta = d @ mid[:2]
                remainder = np.zeros(4)
                remainder[2:] = -d * (np.sin(theta) - theta)
                updated = base + h * (inverse @ remainder)
                gap = np.max(np.abs(updated - new))
                new = updated
                if gap < 1e-13:
                    break
            else:
                raise RuntimeError("Teacher midpoint iteration failed")
            mid = (old + new) / 2
            field = physical_linear_field(mid, nu)
            field[2:] -= d * (np.sin(d @ mid[:2]) - d @ mid[:2])
            residual = np.max(np.abs(new - old - h * field))
            if residual > TOL:
                raise RuntimeError("Teacher midpoint residual exceeds protocol")
            maximum_residual = max(maximum_residual, residual)
            max_iterations = max(max_iterations, iteration + 1)
            total += iteration + 1
            old = new
            if step % stride == 0:
                result[step // stride, trajectory] = old
    return result, maximum_residual, max_iterations, total / (nsteps * count)


def linear_states(initials, angles):
    co, si = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    return np.concatenate([co * initials[None, :, :2] + si * initials[None, :, 2:],
                           co * initials[None, :, 2:] - si * initials[None, :, :2]], axis=2)


def original_numpy_field():
    source = ROOT / "code" / "train_neural_hnn.py"
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "numpy_energy_field")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["numpy_energy_field"]


def verify_fields():
    field_original = original_numpy_field()
    records = []
    for seed in SEEDS:
        arrays = load_seed(seed)
        points = arrays["val_midpoints"][np.linspace(0, len(arrays["val_midpoints"]) - 1, 128, dtype=int)]
        for group in GROUPS[1:]:
            params, weights = weights_for(seed, group)
            independent_energy, independent_field = field_original(points, weights)
            evaluated = [neural_field(point, *params) for point in points]
            gap = float(np.max(np.abs(np.array([v[1] for v in evaluated]) - independent_field)))
            energy_gap = float(np.max(np.abs(np.array([v[0] for v in evaluated]) - independent_energy)))
            finite_gaps = []
            for point, (_, field) in zip(points[:4], evaluated[:4]):
                gradient = np.r_[-field[2:], field[:2]]
                for j in range(4):
                    shift = np.zeros(4); shift[j] = 1e-6
                    derivative = (field_original(point + shift, weights)[0] - field_original(point - shift, weights)[0]) / 2e-6
                    finite_gaps.append(abs(float(derivative) - gradient[j]))
            record = {"seed": seed, "group": group, "states_checked": len(points), "field_max_absolute_gap": gap,
                      "energy_max_absolute_gap": energy_gap, "energy_central_difference_max_gap": max(finite_gaps)}
            if gap > 2e-12 or energy_gap > 2e-13 or max(finite_gaps) > 2e-8:
                raise AssertionError(record)
            records.append(record)
    save_json(OUT / "field_verification.json", records)
    print("FIELD_CHECKS_PASSED", json.dumps({"max_field_gap": max(r["field_max_absolute_gap"] for r in records), "models": len(records)}), flush=True)
    return records


def target_energy(states, nu, d):
    theta = states[..., :2] @ d
    return .5 * np.sum(nu * (states[..., :2] ** 2 + states[..., 2:] ** 2), axis=-1) + 2 * np.sin(theta / 2) ** 2 - theta ** 2 / 2


def metrics_per_trajectory(prediction, continuous, teacher, tip, nu, d, own_energy):
    def tip_rmse(reference):
        error = (prediction - reference)[..., :2] @ tip * 1e3
        return np.sqrt(np.mean(error ** 2, axis=0))
    def state_relative(reference):
        return np.linalg.norm(prediction - reference, axis=(0, 2)) / np.maximum(np.linalg.norm(reference, axis=(0, 2)), 1e-15)
    target = target_energy(prediction, nu, d)
    scale = np.maximum(np.abs(target[0]), 1e-15)
    columns = {
        "continuous_tip_rmse_mm": tip_rmse(continuous), "matched_teacher_tip_rmse_mm": tip_rmse(teacher),
        "continuous_state_relative_l2": state_relative(continuous), "matched_teacher_state_relative_l2": state_relative(teacher),
        "target_energy_drift_relative": np.max(np.abs(target - target[0]), axis=0) / scale,
        "own_energy_drift_relative": np.max(np.abs(own_energy - own_energy[0]), axis=0) / scale,
    }
    return [{key: float(value[trajectory]) for key, value in columns.items()} for trajectory in range(prediction.shape[1])]


def run_one_network(seed, group, arrays, h, nsteps, stride, cayley, inverse):
    params, weights = weights_for(seed, group)
    started = time.perf_counter()
    states, residual, max_iter, mean_iter = neural_rollout(arrays["test_initials"], h, nsteps, stride, cayley, inverse, params)
    energy = original_numpy_field()(states, weights)[0]
    return group, states, energy, {"maximum_implicit_residual": float(residual), "maximum_iterations": int(max_iter), "mean_iterations": float(mean_iter), "elapsed_seconds": time.perf_counter() - started}


def run_experiment(workers):
    plan()
    complete = all((OUT / f"seed_{seed}" / f"h{factor}_{suffix}").exists()
                   for seed in SEEDS for factor in FACTORS for suffix in ["records.json", "trajectories.npz"])
    if complete:
        print("ALL_RESULTS_EXIST; preserving numerical outputs and their execution metadata", flush=True)
        return
    if not (OUT / "field_verification.json").exists():
        verify_fields()
    snapshot = OUT / "execution_script.py"
    if snapshot.exists() and sha256(snapshot) != sha256(__file__):
        raise RuntimeError("Partial results were generated by a different script; preserve the execution snapshot.")
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes())
    save_json(OUT / "environment.json", {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "numba": numba.__version__,
        "script_sha256": sha256(__file__), "execution_source_snapshot": snapshot.as_posix(), "workers": workers, "utc_started": datetime.now(timezone.utc).isoformat(), "fastmath": False, "threads_per_blas": 1})
    prototype = load_seed(SEEDS[0])
    params, _ = weights_for(SEEDS[0], "GR")
    hh = float(prototype["teacher_step_tau"])
    cayley, inverse = cayley_preconditioner(hh, prototype["nu"])
    neural_rollout(prototype["test_initials"][:1], hh, 1, 1, cayley, inverse, params)
    teacher_rollout(prototype["test_initials"][:1], hh, 1, 1, cayley, inverse, prototype["nu"], prototype["d"])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for seed in SEEDS:
            arrays = load_seed(seed)
            h0 = float(arrays["teacher_step_tau"]); omega0 = float(arrays["omega0"])
            base_stride = int(arrays["teacher_sample_stride"])
            ntime = len(arrays["times"]); base_steps = (ntime - 1) * base_stride
            linear_exact = linear_states(arrays["test_initials"], arrays["times"][:, None] * omega0 * arrays["nu"][None, :])
            np.savez_compressed(OUT / f"common_seed_{seed}.npz", times_seconds=arrays["times"], continuous_reference=arrays["reference"],
                                continuous_quadratic=linear_exact, initials=arrays["test_initials"], nu=arrays["nu"], d=arrays["d"], tip_map_m=arrays["tip_map_m"])
            theta = arrays["train_midpoints"][:, :2] @ arrays["d"]
            nonlinear = -(np.sin(theta) - theta)[:, None] * arrays["d"]
            linear = -arrays["train_midpoints"][:, :2] * arrays["nu"]
            force = {"seed": seed, "sample_count": len(theta), "nonlinear_to_linear_force_L2": float(np.linalg.norm(nonlinear) / np.linalg.norm(linear)), "max_abs_training_angle_rad": float(np.max(np.abs(theta)))}
            save_json(OUT / f"force_seed_{seed}.json", force)
            for factor in FACTORS:
                directory = OUT / f"seed_{seed}"; directory.mkdir(exist_ok=True)
                record_path = directory / f"h{factor}_records.json"
                if record_path.exists():
                    print(f"EXISTING seed={seed} divisor={factor}", flush=True)
                    continue
                h, nsteps, stride = h0 / factor, base_steps * factor, base_stride * factor
                expected_times = np.arange(ntime) * h * stride / omega0
                grid_gap = float(np.max(np.abs(expected_times - arrays["times"])))
                if grid_gap > 1e-13:
                    raise AssertionError("Output grid changed")
                cayley, inverse = cayley_preconditioner(h, arrays["nu"])
                teacher, teacher_residual, teacher_max_iter, teacher_mean_iter = teacher_rollout(arrays["test_initials"], h, nsteps, stride, cayley, inverse, arrays["nu"], arrays["d"])
                old_teacher_gap = float(np.max(np.abs(teacher - arrays["old_teacher"]))) if factor == 1 else None
                if factor == 1 and old_teacher_gap > 2e-10:
                    raise AssertionError({"teacher_original_h_gap": old_teacher_gap})
                quadratic = linear_states(arrays["test_initials"], (np.arange(ntime) * stride)[:, None] * (2 * np.arctan(h * arrays["nu"] / 2))[None, :])
                trajectories = {"H0": quadratic}; rows = []; diagnostics = []
                energy = .5 * np.sum(arrays["nu"] * (quadratic[..., :2] ** 2 + quadratic[..., 2:] ** 2), axis=-1)
                one_step = quadratic @ cayley.T
                mid = (quadratic + one_step) / 2
                field = np.concatenate([mid[..., 2:] * arrays["nu"], -mid[..., :2] * arrays["nu"]], axis=-1)
                quadratic_residual = float(np.max(np.abs(one_step - quadratic - h * field)))
                completed = [("H0", quadratic, energy, {"maximum_implicit_residual": quadratic_residual, "maximum_iterations": 0, "mean_iterations": 0., "elapsed_seconds": 0.})]
                futures = [executor.submit(run_one_network, seed, group, arrays, h, nsteps, stride, cayley, inverse) for group in GROUPS[1:]]
                for future in futures:
                    completed.append(future.result())
                for group, prediction, own_energy, solve_info in completed:
                    original_gap = None
                    if group != "H0" and factor == 1:
                        with np.load(SOURCE / f"seed_{seed}" / f"{group}_rollout.npz", allow_pickle=False) as old:
                            original_gap = float(np.max(np.abs(prediction - old["predicted_states"])))
                        if original_gap > 2e-8:
                            raise AssertionError({"seed": seed, "group": group, "original_h_gap": original_gap})
                    trajectories[group] = prediction
                    metrics = metrics_per_trajectory(prediction, arrays["reference"], teacher, arrays["tip_map_m"], arrays["nu"], arrays["d"], own_energy)
                    for trajectory, metric in enumerate(metrics):
                        rows.append({"seed": seed, "group": group, "step_divisor": factor, "step_tau": h, "trajectory": trajectory, **metric})
                    diagnostics.append({"seed": seed, "group": group, "step_divisor": factor, "step_tau": h, "internal_steps": nsteps,
                        "output_stride": stride, "output_count": ntime, "output_grid_max_gap_s": grid_gap, "original_h_saved_state_max_gap": original_gap, **solve_info})
                def tip_mean(first, second):
                    error = (first - second)[..., :2] @ arrays["tip_map_m"] * 1e3
                    return float(np.mean(np.sqrt(np.mean(error ** 2, axis=0))))
                controls = {"seed": seed, "step_divisor": factor, "teacher_continuous_tip_rmse_mm": tip_mean(teacher, arrays["reference"]),
                    "quadratic_integration_only_tip_rmse_mm": tip_mean(quadratic, linear_exact), "continuous_quadratic_model_bias_tip_rmse_mm": tip_mean(linear_exact, arrays["reference"]),
                    "teacher_maximum_implicit_residual": float(teacher_residual), "teacher_maximum_iterations": int(teacher_max_iter), "teacher_mean_iterations": float(teacher_mean_iter),
                    "teacher_original_h_saved_state_max_gap": old_teacher_gap}
                np.savez_compressed(directory / f"h{factor}_trajectories.npz", times_seconds=arrays["times"], matched_teacher=teacher,
                                    groups=np.array(GROUPS), predicted_states=np.stack([trajectories[group] for group in GROUPS]))
                save_json(record_path, {"trajectory_metrics": rows, "solver_diagnostics": diagnostics, "controls": controls})
                continuous_values = {group: float(np.mean([r["continuous_tip_rmse_mm"] for r in rows if r["group"] == group])) for group in GROUPS}
                print("REFINEMENT_READY", json.dumps({"seed": seed, "divisor": factor, "continuous_tip_rmse_mm": continuous_values, "max_solver_residual": max(d["maximum_implicit_residual"] for d in diagnostics)}, ensure_ascii=False), flush=True)
    for item in plan()["inputs"]:
        if sha256(protocol_input_file(item["path"])) != item["sha256"]:
            raise AssertionError("Input changed during evaluation")


def summarize():
    rows, diagnostics, controls = [], [], []
    for seed in SEEDS:
        for factor in FACTORS:
            record = json.loads((OUT / f"seed_{seed}" / f"h{factor}_records.json").read_text(encoding="utf-8"))
            rows.extend(record["trajectory_metrics"]); diagnostics.extend(record["solver_diagnostics"]); controls.append(record["controls"])
    metrics = [k for k in rows[0] if k not in ["seed", "group", "step_divisor", "step_tau", "trajectory"]]
    seed_rows = []
    for seed in SEEDS:
        for factor in FACTORS:
            for group in GROUPS:
                chosen = [r for r in rows if (r["seed"], r["step_divisor"], r["group"]) == (seed, factor, group)]
                if len(chosen) != 8:
                    raise AssertionError("Missing trajectory")
                seed_rows.append({"seed": seed, "group": group, "step_divisor": factor,
                                  **{metric: float(np.mean([r[metric] for r in chosen])) for metric in metrics}})
    aggregates = []
    for factor in FACTORS:
        for group in GROUPS:
            chosen = [r for r in seed_rows if r["group"] == group and r["step_divisor"] == factor]
            aggregates.append({"group": group, "step_divisor": factor, "n_seeds": len(chosen),
                **{metric: {"mean": float(np.mean([r[metric] for r in chosen])), "sample_sd": float(np.std([r[metric] for r in chosen], ddof=1))} for metric in metrics}})
    paired = []
    for seed in SEEDS:
        for factor in FACTORS:
            chosen = {r["group"]: r for r in seed_rows if r["seed"] == seed and r["step_divisor"] == factor}
            for group in GROUPS[1:]:
                paired.append({"seed": seed, "step_divisor": factor, "contrast": f"H0-{group}",
                    **{metric: chosen["H0"][metric] - chosen[group][metric] for metric in metrics[:4]}})
    paired_aggregates = []
    for factor in FACTORS:
        for group in GROUPS[1:]:
            chosen = [r for r in paired if r["step_divisor"] == factor and r["contrast"] == f"H0-{group}"]
            paired_aggregates.append({"step_divisor": factor, "contrast": f"H0-{group}", "n_seeds": len(chosen),
                **{metric: {"mean": float(np.mean([r[metric] for r in chosen])), "sample_sd": float(np.std([r[metric] for r in chosen], ddof=1)), "positive_seed_count": sum(r[metric] > 0 for r in chosen)} for metric in metrics[:4]}})
    force = [json.loads((OUT / f"force_seed_{seed}.json").read_text(encoding="utf-8")) for seed in SEEDS]
    summary = {"status": "complete", "seed_count": 5, "trajectories_per_seed": 8, "output_times": 914,
        "analysis_script_sha256": sha256(__file__),
        "statistic": "Time RMSE within each trajectory, then eight-trajectory mean within seed; five-seed mean and sample SD.",
        "all_models": aggregates, "paired_H0_minus_network": paired_aggregates, "force_ratios": force,
        "force_ratio_mean": float(np.mean([r["nonlinear_to_linear_force_L2"] for r in force])),
        "force_ratio_sample_sd": float(np.std([r["nonlinear_to_linear_force_L2"] for r in force], ddof=1)),
        "maximum_network_implicit_residual": max(r["maximum_implicit_residual"] for r in diagnostics),
        "maximum_teacher_implicit_residual": max(r["teacher_maximum_implicit_residual"] for r in controls),
        "maximum_original_h_network_state_gap": max(r["original_h_saved_state_max_gap"] or 0 for r in diagnostics),
        "maximum_original_h_teacher_state_gap": max(r["teacher_original_h_saved_state_max_gap"] or 0 for r in controls),
        "claim_limits": "All 20 saved models and all seeds included. No retraining. Reduced discretization error need not improve every learned model; lower teacher error does not establish unique physical-potential recovery or general complex-network superiority."}
    write_csv(OUT / "trajectory_metrics.csv", rows); write_csv(OUT / "seed_metrics.csv", seed_rows)
    write_csv(OUT / "solver_diagnostics.csv", diagnostics); write_csv(OUT / "controls.csv", controls)
    write_csv(OUT / "paired_seed_differences.csv", paired); write_csv(OUT / "nonlinear_force_ratios.csv", force)
    flat_paired = [{"step_divisor": row["step_divisor"], "contrast": row["contrast"], "metric": metric,
        "n_seeds": row["n_seeds"], **row[metric]} for row in paired_aggregates for metric in metrics[:4]]
    write_csv(OUT / "paired_summary.csv", flat_paired)
    save_json(OUT / "summary.json", summary)
    return summary


def assets(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup = {(r["group"], r["step_divisor"]): r for r in summary["all_models"]}
    def cell(group, factor, metric):
        record = lookup[group, factor][metric]
        return f"{record['mean'] * 1000:.3f} $\\pm$ {record['sample_sd'] * 1000:.3f}"
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{固定网络权重的步长加密结果。杆端误差单位为 $\mu\mathrm{m}$，报告五种子均值与样本标准差。}",
        r"\label{tab:neural_refinement}", r"\small", r"\begin{tabular}{llrrrr}", r"\toprule", r"比较对象 & 模型 & $h$ & $h/2$ & $h/4$ & $h/8$ \\", r"\midrule"]
    for metric, name in [("continuous_tip_rmse_mm", "连续参考"), ("matched_teacher_tip_rmse_mm", "同步长教师")]:
        for group in ["H0", "GR", "GC"]:
            label = r"$H_0$（无网络）" if group == "H0" else group
            lines.append(f"{name} & {label} & " + " & ".join(cell(group, f, metric) for f in FACTORS) + r" \\")
        if name == "连续参考":
            lines.append(r"\midrule")
    def state_cell(group):
        record = lookup[group, 8]["continuous_state_relative_l2"]
        return f"${record['mean']:.6f}\\pm{record['sample_sd']:.6f}$"
    note = r"\par\smallskip\footnotesize 所有模型使用相同的八条测试轨迹、914个输出时刻和实际时长15.993 s；各轨迹先计算时间RMSE，再在种子内平均。$h=0.009999813692$ 为无量纲步长。同步长教师由完整非线性Hamiltonian在相同步长下求解；连续参考保持原DOP853解。"
    note += r"最细步长的完整状态相对$L^2$误差为：$H_0$ " + state_cell("H0") + "，GR " + state_cell("GR") + "，GC " + state_cell("GC") + r"；该指标中$H_0$更低。R/C的完整结果随复现数据提供。"
    lines.extend([r"\bottomrule", r"\end{tabular}", note, r"\end{table*}", ""])
    table = ROOT / "manuscript" / "tables" / "neural_refinement.tex"
    table.write_text("\n".join(lines), encoding="utf-8")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.4, "lines.markersize": 4, "savefig.dpi": 300})
    colors = {"H0": "#333333", "R": "#E69F00", "C": "#CC79A7", "GR": "#0072B2", "GC": "#009E73"}
    markers = {"H0": "o", "R": "v", "C": "^", "GR": "s", "GC": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), layout="constrained")
    for axis, metric, title in zip(axes, ["continuous_tip_rmse_mm", "continuous_state_relative_l2"], ["(a) Tip displacement", "(b) Complete canonical state"]):
        multiplier = 1000 if metric == "continuous_tip_rmse_mm" else 100
        for group in ["H0", "GR", "GC"]:
            values = [lookup[group, f][metric] for f in FACTORS]
            means = np.array([v["mean"] * multiplier for v in values]); sds = np.array([v["sample_sd"] * multiplier for v in values])
            axis.errorbar(FACTORS, means, yerr=sds, color=colors[group], marker=markers[group], capsize=2, label=r"$H_0$ (no network)" if group == "H0" else group)
        axis.set_xscale("log", base=2); axis.set_yscale("log")
        axis.set_xticks(FACTORS, [r"$h$", r"$h/2$", r"$h/4$", r"$h/8$"])
        axis.set_xlabel("Prediction step")
        axis.set_ylabel(r"Tip RMSE ($\mu$m)" if metric == "continuous_tip_rmse_mm" else r"State relative $L^2$ error (%)")
        axis.set_title(title, loc="left"); axis.grid(True, which="major", alpha=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.10), ncol=3, fontsize=8)
    target = ROOT / "manuscript" / "figures" / "neural_physics_refinement"
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("ASSETS_READY", table.as_posix(), target.with_suffix(".pdf").as_posix(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["plan", "verify", "run", "assets", "all"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use between one and four independent model workers")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ["plan", "verify", "run", "all"]:
        plan()
    if args.stage in ["verify", "all"]:
        verify_fields()
    if args.stage in ["run", "all"]:
        run_experiment(args.workers)
    if args.stage in ["run", "assets", "all"]:
        summary = summarize(); assets(summary)
        print("COMPLETE", json.dumps({k: summary[k] for k in ["maximum_network_implicit_residual", "maximum_teacher_implicit_residual", "maximum_original_h_network_state_gap", "force_ratio_mean"]}), flush=True)


if __name__ == "__main__":
    main()
