"""Fresh amplitude/noise pilot; all writes stay in this exploration directory.

The public generate_dataset(seed, amplitude_multiplier, noise_bound, output_dir)
also supports prospective fresh-seed confirmation without running it here.
Original production modules are imported read-only and never have main called.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
sys.dont_write_bytecode = True

import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "signal_strength"
SEEDS = (17, 43, 91)
NOISES = (1e-6, 0.)
GATE = .22
STEPS = 10000
MODULES = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def protected_check():
    manifest = json.loads((HERE / "protected_manifest.json").read_text(encoding="utf-8"))
    bad = [name for name, digest in manifest.items() if not (ROOT / name).is_file() or sha256(ROOT / name) != digest]
    if bad:
        raise RuntimeError(f"Protected files changed: {bad}")
    return {"files_checked": len(manifest), "all_unchanged": True,
            "manifest_sha256": sha256(HERE / "protected_manifest.json")}


def modules():
    global MODULES
    if MODULES is None:
        loaded = []
        for name in ("learning_experiment", "train_neural_hnn", "neural_physics_refinement"):
            path = ROOT / "code" / f"{name}.py"
            spec = importlib.util.spec_from_file_location(f"signal_pilot_{name}", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            loaded.append(module)
        MODULES = tuple(loaded)
    return MODULES


def get_model():
    # source_setup has a restoration branch: explicitly forbid invoking it on
    # missing local inputs so this script can never restore/write old inputs.
    for name in ("reference_model_spec.json", "feature_model.py", "run_learning.py"):
        path = ROOT / "results" / "learning" / "inputs" / name
        if not path.is_file():
            raise FileNotFoundError(path)
    return modules()[0].source_setup()


def exact_field(points, model):
    points = np.asarray(points)
    theta = points[..., :2] @ model.d
    return np.concatenate([points[..., 2:] * model.nu,
                           -points[..., :2] * model.nu - (np.sin(theta) - theta)[..., None] * model.d], axis=-1)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def generate_dataset(seed, amplitude_multiplier, noise_bound, output_dir):
    """Re-integrate original Hamiltonian and save independent noisy labels.

    output_dir must be inside the new exploration tree. Noise seeds and the
    validation selection in train_one follow the established experiment.
    No saved trajectory is scaled. All internal teacher states and continuous
    reference evaluations on the full coarse test grid enter the angle gate.
    """
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(HERE):
        raise ValueError("Dataset output must stay inside this exploration tree")
    if amplitude_multiplier <= 0 or noise_bound < 0:
        raise ValueError("Invalid amplitude/noise")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"seed_{seed}.npz"
    diag_path = output_dir / f"seed_{seed}_diagnostics.json"
    config = {"seed": int(seed), "amplitude_multiplier": float(amplitude_multiplier),
              "noise_bound": float(noise_bound), "angle_gate_rad": GATE,
              "train_trajectories": 12, "train_duration_s": 2.,
              "validation_trajectories": 4, "validation_duration_s": 4.,
              "test_trajectories": 8, "test_requested_duration_s": 16.,
              "requested_step_tau": .01, "training_stride": 8, "test_stride": 20,
              "test_grid_rule": "Original 40s grid, prefix no later than 16s, rounded down to stride20",
              "reference": {"method": "DOP853", "rtol": 2e-11, "atol": 2e-13},
              "training_noise_seed": int(seed + 10000), "validation_noise_seed": int(seed + 110000),
              "source_spec_sha256": sha256(ROOT / "results/learning/inputs/reference_model_spec.json")}
    if path.exists() or diag_path.exists():
        if not path.exists() or not diag_path.exists():
            raise RuntimeError("Incomplete existing dataset; preserve it for investigation")
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        if diag["config"] != config or diag["dataset_sha256"] != sha256(path):
            raise RuntimeError("Existing dataset configuration/hash mismatch")
        if not diag["angle_gate_passed"]:
            raise RuntimeError(f"Angle gate exceeded for seed {seed}")
        return load_npz(path), diag
    le, _, _ = modules()
    model = get_model()
    coeff = le.teacher_coefficients(model, le.feature_module())
    arrays, diagnostics, angles = {}, {}, {}
    started = time.perf_counter()
    for split, duration, count, offset in (("train", 2., 12, 0), ("val", 4., 4, 100000)):
        initials = le.initial_conditions(seed + offset, count) * amplitude_multiplier
        h, nsteps, times = le.grid(duration, model, .01, 1)
        states, diag = le.rollout_feature_model(initials, h, nsteps, 1, coeff, 1., model)
        midpoints = (states[:, 1:] + states[:, :-1]) / 2
        clean = (states[:, 1:] - states[:, :-1]) / h
        indices = np.arange(0, nsteps, 8)
        x, y = midpoints[:, indices].reshape(-1, 4), clean[:, indices].reshape(-1, 4)
        noise = np.random.default_rng(seed + offset + 10000).uniform(-noise_bound, noise_bound, y.shape)
        angles[split] = float(np.max(np.abs(states[..., :2] @ model.d)))
        diag["midpoint_label_consistency_max"] = float(np.max(np.abs(y - exact_field(x, model))))
        diag["internal_states_angle_gate_max_rad"] = angles[split]
        diagnostics[split] = diag
        arrays.update({f"{split}_initials": initials, f"{split}_midpoints": x,
                       f"{split}_labels": y + noise, f"{split}_clean_labels": y,
                       f"{split}_noise": noise, f"{split}_continuous_field": exact_field(x, model),
                       f"{split}_trajectory_id": np.repeat(np.arange(count), len(indices)),
                       f"{split}_step_tau": np.array(h), f"{split}_sample_states": states[:, ::8],
                       f"{split}_sample_times_s": times[::8]})
    initials = le.initial_conditions(seed + 200000, 8) * amplitude_multiplier
    h, _, _ = le.grid(40., model, .01, 20)
    nsteps = int(16. * model.omega0 / h) // 20 * 20
    all_times_tau = np.arange(nsteps + 1) * h
    internal, diagnostics["test_teacher"] = le.rollout_feature_model(initials, h, nsteps, 1, coeff, 1., model)
    angles["test_teacher"] = float(np.max(np.abs(internal[..., :2] @ model.d)))
    reference, reference_angle, nfev = [], 0., 0
    for initial in initials:
        solution = solve_ivp(lambda t, x: le.teacher_rhs(t, x, model), (0., all_times_tau[-1]), initial,
                             t_eval=all_times_tau, method="DOP853", rtol=2e-11, atol=2e-13)
        if not solution.success:
            raise RuntimeError(solution.message)
        reference_angle = max(reference_angle, float(np.max(np.abs(solution.y[:2].T @ model.d))))
        reference.append(solution.y.T[::20])
        nfev += solution.nfev
    angles["test_continuous_reference"] = reference_angle
    arrays.update(test_initials=initials, test_times=all_times_tau[::20] / model.omega0,
                  test_times_tau=all_times_tau[::20], test_teacher_states=internal[:, ::20],
                  test_reference_states=np.array(reference), teacher_step_tau=np.array(h),
                  teacher_nsteps=np.array(nsteps), teacher_sample_stride=np.array(20),
                  nu=model.nu, d=model.d, omega0=np.array(model.omega0),
                  energy_scale_J=np.array(model.energy_scale), tip_map_m=model.tip_map_m,
                  seed=np.array(seed), amplitude_multiplier=np.array(amplitude_multiplier),
                  noise_bound=np.array(noise_bound), source_spec_sha256=np.array(config["source_spec_sha256"]))
    angle_passed = max(angles.values()) <= GATE
    diagnostics.update(config=config, angle_gate_max_by_split=angles, angle_gate_passed=angle_passed,
                       angle_gate_scope="Computational screening only; not a physical validity proof",
                       reference_nfev=nfev, actual_test_duration_s=float(arrays["test_times"][-1]),
                       elapsed_seconds=time.perf_counter() - started)
    np.savez_compressed(path, **arrays)
    diagnostics["dataset_sha256"] = sha256(path)
    save_json(diag_path, diagnostics)
    print(f"DATA seed={seed} amplitude={amplitude_multiplier:g} noise={noise_bound:g} max_angle={max(angles.values()):.9g} gate={angle_passed} seconds={diagnostics['elapsed_seconds']:.2f}", flush=True)
    if not angle_passed:
        raise RuntimeError(f"Angle gate exceeded, seed={seed}: {angles}; no fitting or sample truncation")
    return arrays, diagnostics


def noise_key(noise):
    return "noise_1e-6" if noise == 1e-6 else "noise_0"


def prepare():
    # Integrate once for each seed. The zero-noise condition reuses these same
    # freshly integrated states, changing only independent labels as prescribed.
    for seed in SEEDS:
        arrays, diag = generate_dataset(seed, 2., 1e-6, OUT / "data" / noise_key(1e-6))
        target = OUT / "data" / noise_key(0.)
        target.mkdir(parents=True, exist_ok=True)
        path, diag_path = target / f"seed_{seed}.npz", target / f"seed_{seed}_diagnostics.json"
        if path.exists():
            old_diag = json.loads(diag_path.read_text(encoding="utf-8"))
            if old_diag["dataset_sha256"] != sha256(path):
                raise RuntimeError("Zero-noise cached data hash mismatch")
            continue
        clean_arrays = dict(arrays)
        clean_arrays["noise_bound"] = np.array(0.)
        for split in ("train", "val"):
            clean_arrays[f"{split}_labels"] = arrays[f"{split}_clean_labels"].copy()
            clean_arrays[f"{split}_noise"] = np.zeros_like(arrays[f"{split}_noise"])
        np.savez_compressed(path, **clean_arrays)
        clean_diag = json.loads(json.dumps(diag))
        clean_diag["config"]["noise_bound"] = 0.
        clean_diag["fresh_integration_shared_with"] = str((OUT / "data" / noise_key(1e-6) / f"seed_{seed}.npz").relative_to(HERE))
        clean_diag["dataset_sha256"] = sha256(path)
        save_json(diag_path, clean_diag)
    save_json(OUT / "angle_gate.json", {"all_passed": True, "threshold_rad": GATE, "seeds": list(SEEDS),
        "scope": "Both noise conditions share newly integrated amplitude-x2 states. No samples excluded."})


def validation_score(weights, arrays):
    _, train, _ = modules()
    prediction = train.numpy_energy_field(arrays["val_midpoints"], weights)[1]
    truth = arrays["val_continuous_field"]
    # One output scale shared by GR/GC for a given seed/condition. This is
    # post-selection reporting, not replacement checkpoint selection.
    scale = np.maximum(np.sqrt(np.mean(arrays["train_labels"] ** 2, axis=0)), 1e-6)
    return {"clean_full_validation_scaled_gradient_rms": float(np.sqrt(np.mean(((prediction - truth) / scale) ** 2))),
            "clean_full_validation_gradient_relative_l2": float(np.linalg.norm(prediction - truth) / np.linalg.norm(truth)),
            "clean_full_validation_gradient_rms": float(np.sqrt(np.mean((prediction - truth) ** 2))),
            "clean_full_validation_points": len(truth), "output_component_scale": scale.tolist()}


def train_all():
    if not (OUT / "angle_gate.json").is_file():
        raise RuntimeError("Run and pass complete data screening before any fit")
    _, training, _ = modules()
    for noise in NOISES:
        for seed in SEEDS:
            data_path = OUT / "data" / noise_key(noise) / f"seed_{seed}.npz"
            arrays = load_npz(data_path)
            data_diag = json.loads(data_path.with_name(f"seed_{seed}_diagnostics.json").read_text(encoding="utf-8"))
            if sha256(data_path) != data_diag["dataset_sha256"] or not data_diag["angle_gate_passed"]:
                raise RuntimeError("Dataset hash or angle gate mismatch")
            for group in ("GR", "GC"):
                dest = OUT / noise_key(noise) / f"seed_{seed}"
                dest.mkdir(parents=True, exist_ok=True)
                row_path = dest / f"{group}_training.json"
                if row_path.exists():
                    existing = json.loads(row_path.read_text(encoding="utf-8"))
                    if existing["dataset_sha256"] != sha256(data_path):
                        raise RuntimeError("Cached fit dataset changed")
                    print(f"EXISTING seed={seed} condition={noise_key(noise)} group={group}", flush=True)
                    continue
                print(f"FIT_START seed={seed} condition={noise_key(noise)} group={group}", flush=True)
                try:
                    weights, row, history = training.train_one(seed, group, arrays, STEPS, 512)
                    np.savez_compressed(dest / f"{group}_weights.npz", **weights)
                    row.update(validation_score(weights, arrays))
                    row.update(status="completed", amplitude_multiplier=2., noise_bound=noise,
                               dataset_sha256=sha256(data_path), weights_sha256=sha256(dest / f"{group}_weights.npz"),
                               selection="Unchanged train_one: noisy validation labels when noise=1e-6, independent validation noise, fixed original 2048 subset")
                    save_json(dest / f"{group}_history.json", history)
                    save_json(row_path, row)
                    print(f"FIT_DONE seed={seed} condition={noise_key(noise)} group={group} seconds={row['training_seconds']:.2f} val_clean={row['clean_full_validation_scaled_gradient_rms']:.9g} selected={row['selected_validation_step']}", flush=True)
                except Exception as error:
                    save_json(row_path, {"status": "failed", "seed": seed, "group": group, "noise_bound": noise,
                              "dataset_sha256": sha256(data_path), "failure": repr(error), "traceback": traceback.format_exc()})
                    print(f"FIT_FAILED seed={seed} condition={noise_key(noise)} group={group}: {error}", flush=True)


def trajectory_metrics(prediction, reference, teacher, tip):
    def against(target):
        delta = prediction - target
        tip_per = np.sqrt(np.mean((delta[..., :2] @ tip * 1000) ** 2, axis=0))
        full_per = np.sqrt(np.sum(delta ** 2, axis=(0, 2)) / np.sum(target ** 2, axis=(0, 2)))
        return {"tip_rmse_mm_per_trajectory": tip_per.tolist(), "tip_rmse_mm_mean": float(np.mean(tip_per)),
                "state_relative_l2_per_trajectory": full_per.tolist(), "state_relative_l2_mean": float(np.mean(full_per))}
    return {"continuous": against(reference), "matched_teacher": against(teacher)}


def rollout_all():
    _, training, refine = modules()
    for seed in SEEDS:
        arrays = load_npz(OUT / "data" / noise_key(1e-6) / f"seed_{seed}.npz")
        h = float(arrays["teacher_step_tau"]) / 16
        nsteps, stride = int(arrays["teacher_nsteps"]) * 16, 320
        initials, nu, d = arrays["test_initials"], arrays["nu"], arrays["d"]
        cayley, inverse = refine.cayley_preconditioner(h, nu)
        times = arrays["test_times"]
        reference = np.transpose(arrays["test_reference_states"], (1, 0, 2))
        common = OUT / f"fine_common_seed_{seed}.npz"
        if common.exists():
            saved = load_npz(common)
            teacher, quadratic = saved["teacher_states"], saved["H0_states"]
            common_diag = json.loads(common.with_suffix(".json").read_text(encoding="utf-8"))
            if common_diag["trajectory_sha256"] != sha256(common):
                raise RuntimeError("Fine teacher cache hash mismatch")
        else:
            started = time.perf_counter()
            teacher, residual, max_iter, mean_iter = refine.teacher_rollout(initials, h, nsteps, stride, cayley, inverse, nu, d)
            angles = (np.arange(len(times)) * stride)[:, None] * (2 * np.arctan(h * nu / 2))[None, :]
            quadratic = refine.linear_states(initials, angles)
            np.savez_compressed(common, times_seconds=times, continuous_reference=reference,
                                teacher_states=teacher, H0_states=quadratic, initials=initials,
                                nu=nu, d=d, tip_map_m=arrays["tip_map_m"])
            common_diag = {"seed": seed, "divisor": 16, "h_tau": h, "nsteps": nsteps, "stride": stride,
                           "maximum_implicit_residual": float(residual), "maximum_iterations": int(max_iter),
                           "mean_iterations": float(mean_iter), "elapsed_seconds": time.perf_counter() - started,
                           "trajectory_sha256": sha256(common), "actual_forecast_seconds": float(times[-1]),
                           "H0": trajectory_metrics(quadratic, reference, teacher, arrays["tip_map_m"])}
            save_json(common.with_suffix(".json"), common_diag)
        print(f"FINE_COMMON seed={seed} h0/16 teacher_residual={common_diag['maximum_implicit_residual']:.3g}", flush=True)
        for noise in NOISES:
            for group in ("GR", "GC"):
                dest = OUT / noise_key(noise) / f"seed_{seed}"
                result_path = dest / f"{group}_fine.json"
                if result_path.exists():
                    print(f"EXISTING_FINE seed={seed} condition={noise_key(noise)} group={group}", flush=True)
                    continue
                row = json.loads((dest / f"{group}_training.json").read_text(encoding="utf-8"))
                if row["status"] != "completed":
                    continue
                weights_path = dest / f"{group}_weights.npz"
                weights = load_npz(weights_path)
                if row["weights_sha256"] != sha256(weights_path):
                    raise RuntimeError("Weight cache hash mismatch")
                params = tuple(np.ascontiguousarray(weights[key]) for key in ("w1", "b1", "w2", "b2", "w3", "b3", "scale")) + (float(weights["energy_scale"]), np.ascontiguousarray(weights["nu"]), bool(weights["prior"]))
                check_points = arrays["val_midpoints"][:128]
                field_gap = float(np.max(np.abs(np.array([refine.neural_field(point, *params)[1] for point in check_points]) - training.numpy_energy_field(check_points, weights)[1])))
                if field_gap > 2e-12:
                    raise AssertionError(f"JIT field export mismatch {field_gap}")
                print(f"FINE_START seed={seed} condition={noise_key(noise)} group={group}", flush=True)
                started = time.perf_counter()
                try:
                    prediction, residual, max_iter, mean_iter = refine.neural_rollout(initials, h, nsteps, stride, cayley, inverse, params)
                    trajectory_path = dest / f"{group}_fine.npz"
                    np.savez_compressed(trajectory_path, times_seconds=times, predicted_states=prediction,
                                        continuous_reference=reference, teacher_states=teacher, H0_states=quadratic)
                    result = {"status": "completed", "seed": seed, "group": group, "noise_bound": noise,
                              "divisor": 16, "h_tau": h, "nsteps": nsteps, "stride": stride,
                              "maximum_implicit_residual": float(residual), "maximum_iterations": int(max_iter),
                              "mean_iterations": float(mean_iter), "elapsed_seconds": time.perf_counter() - started,
                              "numpy_jit_field_max_gap": field_gap, "weights_sha256": sha256(weights_path),
                              "trajectory_sha256": sha256(trajectory_path),
                              "prediction_saved_grid_max_abs_angle_rad": float(np.max(np.abs(prediction[..., :2] @ d))),
                              "metrics": trajectory_metrics(prediction, reference, teacher, arrays["tip_map_m"])}
                    save_json(result_path, result)
                    print(f"FINE_DONE seed={seed} condition={noise_key(noise)} group={group} seconds={result['elapsed_seconds']:.2f} state={result['metrics']['continuous']['state_relative_l2_mean']:.9g} tip={result['metrics']['continuous']['tip_rmse_mm_mean']:.9g}", flush=True)
                except Exception as error:
                    save_json(result_path, {"status": "failed", "seed": seed, "group": group,
                              "noise_bound": noise, "failure": repr(error), "traceback": traceback.format_exc()})
                    print(f"FINE_FAILED seed={seed} condition={noise_key(noise)} group={group}: {error}", flush=True)


def moments(values):
    return {"n": len(values), "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def summary():
    result = {"scope": "Three-seed exploratory pilot; no five-seed confirmation and no paper claim", "conditions": {}}
    for noise in NOISES:
        condition = {"amplitude_multiplier": 2., "noise_bound": noise, "training": [], "fine_rollouts": [], "models": {}}
        for seed in SEEDS:
            for group in ("GR", "GC"):
                dest = OUT / noise_key(noise) / f"seed_{seed}"
                for suffix, key in (("training", "training"), ("fine", "fine_rollouts")):
                    path = dest / f"{group}_{suffix}.json"
                    if path.exists():
                        condition[key].append(json.loads(path.read_text(encoding="utf-8")))
        by_pair = {(row["seed"], row["group"]): row for row in condition["training"] if row["status"] == "completed"}
        paired_seeds = [seed for seed in SEEDS if (seed, "GR") in by_pair and (seed, "GC") in by_pair]
        ratios = [by_pair[seed, "GR"]["clean_full_validation_scaled_gradient_rms"] / by_pair[seed, "GC"]["clean_full_validation_scaled_gradient_rms"] for seed in paired_seeds]
        condition["nomination"] = {"paired_seeds": paired_seeds, "log_GR_over_GC_per_seed": np.log(ratios).tolist(),
            "mean_log_GR_over_GC": float(np.mean(np.log(ratios))) if ratios else None,
            "GC_validation_wins": int(sum(ratio > 1 for ratio in ratios)),
            "eligible_on_validation": bool(len(ratios) == 3 and np.mean(np.log(ratios)) > 0 and sum(ratio > 1 for ratio in ratios) >= 2)}
        fine = {(row["seed"], row["group"]): row for row in condition["fine_rollouts"] if row["status"] == "completed"}
        for group in ("H0", "GR", "GC"):
            rows = []
            for seed in SEEDS:
                if group == "H0":
                    path = OUT / f"fine_common_seed_{seed}.json"
                    if path.exists():
                        rows.append(json.loads(path.read_text(encoding="utf-8"))["H0"]["continuous"])
                elif (seed, group) in fine:
                    rows.append(fine[seed, group]["metrics"]["continuous"])
            if rows:
                condition["models"][group] = {key: moments([row[key] for row in rows]) for key in ("state_relative_l2_mean", "tip_rmse_mm_mean")}
        condition["paired_rollout"] = {}
        complete = [seed for seed in SEEDS if (seed, "GR") in fine and (seed, "GC") in fine]
        for metric in ("state_relative_l2_mean", "tip_rmse_mm_mean"):
            gr = [fine[seed, "GR"]["metrics"]["continuous"][metric] for seed in complete]
            gc = [fine[seed, "GC"]["metrics"]["continuous"][metric] for seed in complete]
            if complete:
                improvement = (np.array(gr) - np.array(gc)) / gr
                condition["paired_rollout"][metric] = {"seeds": complete, "GR": gr, "GC": gc,
                    "GC_relative_improvement_per_seed": improvement.tolist(),
                    "GC_mean_paired_relative_improvement": float(np.mean(improvement)),
                    "GC_wins": int(np.sum(improvement > 0))}
        result["conditions"][noise_key(noise)] = condition
    result["protection"] = protected_check()
    save_json(OUT / "summary.json", result)
    print("SUMMARY " + json.dumps({key: {"nomination": value["nomination"], "paired_rollout": value["paired_rollout"]} for key, value in result["conditions"].items()}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "train", "rollout", "summary", "all"), default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    check = protected_check()
    config = {"amplitude_multiplier": 2., "noise_bounds": list(NOISES), "seeds": list(SEEDS),
              "groups": ["GR", "GC"], "updates": STEPS, "batch_size": 512, "real_parameters": 673,
              "angle_gate_rad": GATE, "fine_divisor": 16,
              "checkpoint_selection": "Unmodified train_neural_hnn.train_one; independent matching-noise validation labels; fixed seed+3000 subset2048",
              "clean_validation": "Final full-set scaled gradient RMS only; no checkpoint selection",
              "noise_seeds": "train=seed+10000; val=seed+110000", "protected": check,
              "production_source_hashes": {name: sha256(ROOT / "code" / name) for name in ("learning_experiment.py", "train_neural_hnn.py", "neural_physics_refinement.py")}}
    path = OUT / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
        raise RuntimeError("Existing signal-strength protocol differs")
    if not path.exists():
        save_json(path, config)
    snapshot = OUT / "execution_script.py"
    if snapshot.exists() and sha256(snapshot) != sha256(__file__):
        raise RuntimeError("Script differs from preserved execution snapshot")
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes())
        save_json(OUT / "environment.json", {"utc_started": datetime.now(timezone.utc).isoformat(),
                  "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                  "threads_per_blas": 1, "script_sha256": sha256(__file__)})
    with threadpool_limits(limits=1):
        if args.stage in ("prepare", "all"):
            prepare()
        if args.stage in ("train", "all"):
            train_all()
        if args.stage in ("rollout", "all"):
            rollout_all()
        if args.stage in ("summary", "all"):
            summary()
    save_json(OUT / "protected_verification.json", protected_check())


if __name__ == "__main__":
    main()
