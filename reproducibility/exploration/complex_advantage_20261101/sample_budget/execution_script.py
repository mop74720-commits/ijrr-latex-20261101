"""Explore training-trajectory and optimization budgets without changing the paper.

The original train_one is called directly. Its fixed validation selection subset
is retained; final clean-vector-field reporting uses the full validation set.
All writes are confined to this exploration's sample_budget directory.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "sample_budget"
DATA = ROOT / "results/learning/data"
SOURCE = ROOT / "results/neural_learning_10000"
SEEDS = [17, 43, 91]
GROUPS = ["GR", "GC"]
CONDITIONS = [
    ("traj12_updates10000_saved", 12, 10000, True),
    ("traj1_updates10000", 1, 10000, False),
    ("traj3_updates10000", 3, 10000, False),
    ("traj12_updates30000", 12, 30000, False),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_seed(seed):
    with np.load(DATA / f"seed_{seed}.npz", allow_pickle=False) as source:
        return {key: np.array(source[key], copy=True) for key in source.files}


def selection(seed, arrays, count):
    ids = np.unique(arrays["train_trajectory_id"])
    order = np.random.default_rng(seed + 41000).permutation(ids)
    chosen = order[:count] if count < 12 else ids
    indices = np.flatnonzero(np.isin(arrays["train_trajectory_id"], chosen))
    return order, chosen, indices


def training_arrays(arrays, indices):
    result = dict(arrays)
    for key in ["train_midpoints", "train_labels", "train_clean_labels", "train_noise", "train_trajectory_id"]:
        result[key] = arrays[key][indices]
    return result


def verify_protected():
    protected = json.loads((HERE / "protected_manifest.json").read_text(encoding="utf-8"))
    changed = [name for name, digest in protected.items() if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    if changed:
        raise RuntimeError("Protected files changed: " + ", ".join(changed[:10]))
    return {"protected_file_count": len(protected), "changed": changed}


def plan():
    OUT.mkdir(parents=True, exist_ok=True)
    trainer = ROOT / "code/train_neural_hnn.py"
    registry = {}
    inputs = [{"path": trainer.relative_to(ROOT).as_posix(), "sha256": sha(trainer)}]
    for seed in SEEDS:
        arrays = load_seed(seed)
        inputs.append({"path": (DATA / f"seed_{seed}.npz").relative_to(ROOT).as_posix(), "sha256": sha(DATA / f"seed_{seed}.npz")})
        order, _, _ = selection(seed, arrays, 1)
        report = {"trajectory_order_rng_seed": seed + 41000, "permuted_training_trajectory_ids": order.tolist(), "subsets": {}}
        for count in [1, 3, 12]:
            _, chosen, indices = selection(seed, arrays, count)
            report["subsets"][str(count)] = {"trajectory_ids": chosen.tolist(), "row_indices": indices.tolist(), "row_count": len(indices)}
        val_indices = np.random.default_rng(seed + 3000).choice(len(arrays["val_midpoints"]), size=min(2048, len(arrays["val_midpoints"])), replace=False)
        report.update(validation_selection_indices=val_indices.tolist(), validation_selection_count=len(val_indices), validation_reporting_count=len(arrays["val_midpoints"]))
        registry[str(seed)] = report
        for group in GROUPS:
            path = SOURCE / f"seed_{seed}" / f"{group}_weights.npz"
            inputs.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)})
    protocol = {
        "purpose": "Pilot sample/optimization-budget exploration of the original GR and GC HNNs; retain unfavorable conditions and failures.",
        "seeds": SEEDS, "groups": GROUPS, "conditions": [dict(name=name, train_trajectories=count, updates=steps, reuse_existing_weights=saved) for name, count, steps, saved in CONDITIONS],
        "trainer": "Direct call to unmodified code/train_neural_hnn.py train_one and EnergyNetwork; no original main/summarize call.",
        "parameter_budget": "673 real scalars per GR/GC model, original architecture and componentwise tanh.",
        "subsampling": "Within each seed permute whole trajectory ids with seed+41000; first one and first three form nested subsets; preserve original row order inside a subset.",
        "training": "Original noisy Eq36 midpoint labels, batch 512, original train-only RMS input/output and energy scaling, Adam lr .002 then x.3 at half and three quarters of each update budget, clipping 10; no test selection.",
        "selection": "Keep original fixed seed+3000 subset of 2048 from the 2284 independent validation points. Every 100 updates choose the minimum noisy normalized validation MSE, as in the saved 12-trajectory 10k reference.",
        "reporting": "Evaluate all 2284 validation points against val_clean_labels and independently checked continuous Hamiltonian RHS. Also report learned residual versus the physical nonquadratic field after subtracting the known quadratic field.",
        "validation_metrics": ["clean_gradient_rms", "common_scale_clean_gradient_rms", "clean_field_relative_l2", "clean_field_mse", "nonlinear_residual_relative_l2", "nonlinear_momentum_relative_l2", "spurious_kinematic_residual_rms"],
        "common_reporting_scale": "For normalized reporting only use the original full-12-trajectory training-label RMS of each component (floor 1e-6), shared across all conditions within seed. Actual training retains condition-specific original train_one scaling.",
        "nomination": "Use mean_seed log(GR_clean_gradient_rms/GC_clean_gradient_rms), including the original full12/10k baseline. Eligibility requires a positive score and at least 2/3 GC wins; root selects at most one setting without using pilot test rollouts.",
        "statistics": "Three data/noise seeds are exploratory paired repeats; mean/sample SD and GC directional wins; no population significance or universal advantage claim.",
        "baseline": "Reuse original 12-trajectory 10000-update selected weights; re-evaluate on the identical full validation set. Recorded original hardware timings are not a controlled timing comparison.",
        "optional_rollout": "Frozen new weights, h0/16, original 914 output times through 15.992993212 s; original DOP853 continuous reference and analytic H0 baseline; test data never select configurations.",
        "threading": "One Python training process, torch/BLAS threads 1; other exploration workers may run, so wall times are descriptive.",
        "failure_policy": "Save exception type/message/traceback and continue remaining scheduled conditions; no silent exclusion.",
        "selection_registry": registry, "inputs": inputs,
    }
    path = OUT / "protocol.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != protocol:
            raise RuntimeError("Existing sample-budget protocol differs")
    else:
        save_json(path, protocol)
    return protocol


def field_metrics(weights, arrays, trainer, common_output_rms):
    points, clean = arrays["val_midpoints"], arrays["val_clean_labels"]
    nu, d = arrays["nu"], arrays["d"]
    theta = points[:, :2] @ d
    physical_linear = np.c_[points[:, 2:] * nu, -points[:, :2] * nu]
    physical_rhs = physical_linear.copy()
    physical_rhs[:, 2:] -= (np.sin(theta) - theta)[:, None] * d
    label_gap = float(np.max(np.abs(clean - physical_rhs)))
    if label_gap > 1e-11:
        raise AssertionError(f"Saved clean labels differ from continuous RHS: {label_gap}")
    predicted = trainer.numpy_energy_field(points, weights)[1]
    residual_target = clean - physical_linear
    error = predicted - clean
    scales = np.maximum(np.sqrt(np.mean(arrays["train_labels"] ** 2, axis=0)), 1e-6)
    result = {
        "validation_reporting_points": len(points), "clean_label_rhs_max_gap": label_gap,
        "clean_gradient_rms": float(np.sqrt(np.mean(error ** 2))),
        "common_scale_clean_gradient_rms": float(np.sqrt(np.mean((error / common_output_rms) ** 2))),
        "common_reporting_output_rms": common_output_rms.tolist(),
        "clean_field_mse": float(np.mean(error ** 2)),
        "clean_field_relative_l2": float(np.linalg.norm(error) / np.linalg.norm(clean)),
        "nonlinear_residual_relative_l2": float(np.linalg.norm(error) / np.linalg.norm(residual_target)),
        "nonlinear_momentum_relative_l2": float(np.linalg.norm(error[:, 2:]) / np.linalg.norm(residual_target[:, 2:])),
        "spurious_kinematic_residual_rms": float(np.sqrt(np.mean((predicted[:, :2] - physical_linear[:, :2]) ** 2))),
        "noisy_full_validation_normalized_mse": float(np.mean(((predicted - arrays["val_labels"]) / scales) ** 2)),
        "quadratic_baseline_field_relative_l2": float(np.linalg.norm(residual_target) / np.linalg.norm(clean)),
        "validation_nonlinear_field_l2": float(np.linalg.norm(residual_target)),
        "validation_field_error_component_rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
        "train_only_input_rms": np.maximum(np.sqrt(np.mean(arrays["train_midpoints"] ** 2, axis=0)), 1e-4).tolist(),
        "train_only_output_rms": scales.tolist(),
        "train_only_energy_scale": max(float(np.mean(np.abs(arrays["train_midpoints"] * arrays["train_labels"]))), 1e-5),
    }
    if not all(np.isfinite(value) for value in result.values() if isinstance(value, (float, int))):
        raise RuntimeError("Nonfinite validation metric")
    return result


def run_training():
    protocol = plan()
    save_json(OUT / "protected_before.json", verify_protected())
    trainer = import_file("sample_budget_original_trainer", ROOT / "code/train_neural_hnn.py")
    trainer.torch.set_num_threads(1)
    try:
        trainer.torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    environment = {"utc_start": datetime.now(timezone.utc).isoformat(), "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "torch": trainer.torch.__version__, "torch_threads": trainer.torch.get_num_threads(), "script_sha256": sha(__file__), "original_trainer_sha256": sha(ROOT / "code/train_neural_hnn.py")}
    save_json(OUT / "environment.json", environment)
    for name, count, steps, saved in CONDITIONS:
        for seed in SEEDS:
            full = load_seed(seed)
            order, chosen, indices = selection(seed, full, count)
            arrays = training_arrays(full, indices)
            common_output_rms = np.maximum(np.sqrt(np.mean(full["train_labels"] ** 2, axis=0)), 1e-6)
            pair_started = time.perf_counter()
            for group in GROUPS:
                directory = OUT / name / f"seed_{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{group}_record.json"
                if path.exists():
                    print(f"EXISTING {name} seed={seed} group={group}", flush=True)
                    continue
                started = time.perf_counter()
                print(f"FIT_START {name} seed={seed} group={group} rows={len(indices)} saved={saved}", flush=True)
                base = {"condition": name, "seed": seed, "group": group, "train_trajectories": count, "training_rows": len(indices), "training_trajectory_ids": chosen.tolist(), "training_updates": steps, "baseline_saved_weights": saved, "validation_selection_count": protocol["selection_registry"][str(seed)]["validation_selection_count"], "validation_reporting_count": len(arrays["val_midpoints"])}
                try:
                    if saved:
                        source = SOURCE / f"seed_{seed}" / f"{group}_weights.npz"
                        shutil.copyfile(source, directory / f"{group}_weights.npz")
                        with np.load(source, allow_pickle=False) as archive:
                            weights = {key: archive[key] for key in archive.files}
                        old = json.loads((SOURCE / f"seed_{seed}" / f"{group}_metrics.json").read_text(encoding="utf-8"))
                        training = {key: old[key] for key in ["real_parameter_count", "selected_validation_step", "validation_normalized_mse", "training_seconds", "numpy_torch_field_max_difference"]}
                        history = json.loads((SOURCE / f"seed_{seed}" / f"{group}_training_history.json").read_text(encoding="utf-8"))
                        base.update(weight_source=source.relative_to(ROOT).as_posix(), weight_source_sha256=sha(source))
                    else:
                        weights, training, history = trainer.train_one(seed, group, arrays, steps, 512)
                        np.savez_compressed(directory / f"{group}_weights.npz", **weights)
                    save_json(directory / f"{group}_history.json", history)
                    record = {**base, "status": "completed", **training, **field_metrics(weights, arrays, trainer, common_output_rms), "elapsed_seconds_this_evaluation": time.perf_counter() - started, "weights_sha256": sha(directory / f"{group}_weights.npz")}
                    if not saved and record["elapsed_seconds_this_evaluation"] > 120:
                        print("SLOW_FIT", name, seed, group, record["elapsed_seconds_this_evaluation"], flush=True)
                except Exception as error:
                    record = {**base, "status": "failed", "exception_type": type(error).__name__, "exception_message": str(error), "traceback": traceback.format_exc(), "elapsed_seconds_this_evaluation": time.perf_counter() - started}
                save_json(path, record)
                print("FIT_DONE", json.dumps({key: record.get(key) for key in ["condition", "seed", "group", "status", "training_seconds", "clean_field_relative_l2", "nonlinear_residual_relative_l2", "selected_validation_step"]}), flush=True)
            summarize()
            pair = [json.loads((OUT / name / f"seed_{seed}" / f"{group}_record.json").read_text(encoding="utf-8")) for group in GROUPS]
            print("PAIR_COMPLETE", json.dumps({"condition": name, "seed": seed, "elapsed_seconds": time.perf_counter() - pair_started, "statuses": {row["group"]: row["status"] for row in pair}, "GC_over_GR_clean_field_error_ratio": pair[1].get("clean_field_relative_l2", 0) / pair[0].get("clean_field_relative_l2", 1) if all(row["status"] == "completed" for row in pair) else None}), flush=True)
    save_json(OUT / "protected_after_training.json", verify_protected())


def summarize():
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(OUT.glob("traj*/seed_*/*_record.json"))]
    scalar_records = [{key: value for key, value in row.items() if not isinstance(value, (list, dict)) and key != "traceback"} for row in records]
    write_csv(OUT / "all_runs.csv", scalar_records)
    paired, aggregates = [], []
    metrics = ["clean_gradient_rms", "common_scale_clean_gradient_rms", "clean_field_relative_l2", "clean_field_mse", "nonlinear_residual_relative_l2", "nonlinear_momentum_relative_l2", "spurious_kinematic_residual_rms"]
    for name, count, steps, saved in CONDITIONS:
        for seed in SEEDS:
            lookup = {row["group"]: row for row in records if row["condition"] == name and row["seed"] == seed and row["status"] == "completed"}
            if set(lookup) != set(GROUPS):
                continue
            for metric in metrics:
                gr, gc = lookup["GR"][metric], lookup["GC"][metric]
                paired.append({"condition": name, "seed": seed, "metric": metric, "GR": gr, "GC": gc, "GR_minus_GC": gr - gc, "GC_relative_improvement": (gr - gc) / gr if gr else 0., "GC_directional_win": gc < gr})
        for metric in metrics:
            chosen = [row for row in paired if row["condition"] == name and row["metric"] == metric]
            if not chosen:
                continue
            aggregates.append({"condition": name, "metric": metric, "n_complete_pairs": len(chosen), "GC_directional_wins": sum(row["GC_directional_win"] for row in chosen),
                "GR_mean": float(np.mean([row["GR"] for row in chosen])), "GC_mean": float(np.mean([row["GC"] for row in chosen])),
                "paired_difference_mean": float(np.mean([row["GR_minus_GC"] for row in chosen])), "paired_difference_sample_sd": float(np.std([row["GR_minus_GC"] for row in chosen], ddof=1)) if len(chosen) > 1 else None,
                "paired_relative_improvement_mean": float(np.mean([row["GC_relative_improvement"] for row in chosen])), "paired_relative_improvement_sample_sd": float(np.std([row["GC_relative_improvement"] for row in chosen], ddof=1)) if len(chosen) > 1 else None})
    write_csv(OUT / "validation_paired_seed_metrics.csv", paired)
    write_csv(OUT / "validation_summary.csv", aggregates)
    nomination = []
    for name, count, steps, saved in CONDITIONS:
        chosen = [row for row in paired if row["condition"] == name and row["metric"] == "clean_gradient_rms"]
        if not chosen:
            continue
        logs = [float(np.log(row["GR"] / row["GC"])) for row in chosen]
        wins = sum(row["GC_directional_win"] for row in chosen)
        score = float(np.mean(logs))
        nomination.append({"condition": name, "n_complete_pairs": len(chosen), "mean_log_GR_over_GC_clean_gradient_rms": score, "per_seed_log_ratios": logs, "GC_wins": wins, "eligible_when_all_three_available": len(chosen) == 3 and score > 0 and wins >= 2})
    write_csv(OUT / "nomination_validation_scores.csv", [{k: v for k, v in row.items() if not isinstance(v, list)} for row in nomination])
    result = {"expected_training_fits": 18, "expected_existing_reference_evaluations": 6, "completed": sum(row["status"] == "completed" for row in records), "failures": [row for row in records if row["status"] != "completed"], "validation_comparisons": aggregates, "nomination_validation_scores": nomination, "selection_used_test_data": False, "scope": "Three-seed exploratory pilot; report all settings, including unfavorable outcomes. No validation ranking establishes test-set or population superiority."}
    save_json(OUT / "summary.json", result)
    return result


def rollouts():
    """Evaluate frozen trained weights after validation reporting is fixed."""
    helper = import_file("sample_budget_refinement_helper", ROOT / "code/neural_physics_refinement.py")
    trainer = import_file("sample_budget_rollout_original_trainer", ROOT / "code/train_neural_hnn.py")
    for name, count, steps, saved in CONDITIONS:
        if saved:
            continue
        for seed in SEEDS:
            arrays = load_seed(seed)
            with np.load(SOURCE / f"seed_{seed}/GR_rollout.npz", allow_pickle=False) as source:
                times, reference = source["times_seconds"], source["reference_states"]
            factor = 16
            h = float(arrays["teacher_step_tau"]) / factor
            stride = int(arrays["teacher_sample_stride"]) * factor
            nsteps = (len(times) - 1) * stride
            cayley, inverse = helper.cayley_preconditioner(h, arrays["nu"])
            angles = (np.arange(len(times)) * stride)[:, None] * (2 * np.arctan(h * arrays["nu"] / 2))[None, :]
            baseline = helper.linear_states(arrays["test_initials"], angles)
            def metrics(predicted):
                relative = np.linalg.norm(predicted - reference, axis=(0, 2)) / np.linalg.norm(reference, axis=(0, 2))
                tip = (predicted - reference)[..., :2] @ arrays["tip_map_m"] * 1e3
                tip_rmse = np.sqrt(np.mean(tip ** 2, axis=0))
                return {"continuous_state_relative_l2_mean": float(np.mean(relative)), "continuous_tip_rmse_mm_mean": float(np.mean(tip_rmse)), "per_trajectory_state_relative_l2": relative.tolist(), "per_trajectory_tip_rmse_mm": tip_rmse.tolist()}
            for group in GROUPS:
                directory = OUT / name / f"seed_{seed}"
                record_path = directory / f"{group}_rollout_h16.json"
                if record_path.exists():
                    continue
                training = json.loads((directory / f"{group}_record.json").read_text(encoding="utf-8"))
                if training["status"] != "completed":
                    save_json(record_path, {"condition": name, "seed": seed, "group": group, "status": "skipped_training_failed"})
                    continue
                started = time.perf_counter()
                try:
                    with np.load(directory / f"{group}_weights.npz", allow_pickle=False) as source:
                        weights = {key: np.array(source[key], copy=True, order="C") for key in source.files}
                    params = (weights["w1"], weights["b1"], weights["w2"], weights["b2"], weights["w3"], weights["b3"], weights["scale"], float(weights["energy_scale"]), weights["nu"], bool(weights["prior"]))
                    check = arrays["val_midpoints"][:128]
                    numba_field = np.array([helper.neural_field(x, *params)[1] for x in check])
                    gap = float(np.max(np.abs(numba_field - trainer.numpy_energy_field(check, weights)[1])))
                    if gap > 2e-12:
                        raise AssertionError(f"Independent field gap {gap}")
                    predicted, residual, maximum_iterations, mean_iterations = helper.neural_rollout(arrays["test_initials"], h, nsteps, stride, cayley, inverse, params)
                    np.savez_compressed(directory / f"{group}_rollout_h16.npz", times_seconds=times, predicted_states=predicted, continuous_reference=reference, quadratic_baseline=baseline)
                    record = {"condition": name, "seed": seed, "group": group, "status": "completed", "step_divisor": factor, "h_tau": h, "output_count": len(times), "internal_steps": nsteps,
                        "maximum_implicit_residual": float(residual), "maximum_iterations": int(maximum_iterations), "mean_iterations": float(mean_iterations), "independent_numpy_field_max_gap": gap,
                        "weights_sha256": sha(directory / f"{group}_weights.npz"), "prediction": metrics(predicted), "quadratic_baseline": metrics(baseline), "elapsed_seconds": time.perf_counter() - started}
                except Exception as error:
                    record = {"condition": name, "seed": seed, "group": group, "status": "failed", "exception_type": type(error).__name__, "exception_message": str(error), "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter() - started}
                save_json(record_path, record)
                print("ROLLOUT_DONE", json.dumps({key: record.get(key) for key in ["condition", "seed", "group", "status", "prediction", "elapsed_seconds"]}), flush=True)
    paths = sorted(OUT.glob("traj*/seed_*/*_rollout_h16.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    save_json(OUT / "rollout_summary.json", {"records": rows, "used_for_configuration_selection": False, "scope": "Optional diagnostic evaluation of every newly fitted pilot model, original common held-out reference and initial states; no nominated setting or confirmation claim."})
    save_json(OUT / "protected_after_rollout.json", verify_protected())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["plan", "train", "summary", "rollout"], default="plan")
    args = parser.parse_args()
    if args.stage == "plan":
        plan()
    elif args.stage == "train":
        run_training()
    elif args.stage == "summary":
        summarize()
    elif args.stage == "rollout":
        rollouts()


if __name__ == "__main__":
    main()
