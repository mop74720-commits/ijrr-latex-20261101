"""Explore h0/16 and h0/32 with ALL saved 10k GR/GC weights, read-only sources.

Only low-level pure/read functions are reused from neural_physics_refinement.
No original plan/run/summarize function is called. All new outputs are confined
to this script's sibling fine_step directory. Use Python -B to suppress pycache.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMBA_NUM_THREADS"):
    os.environ[_variable] = "1"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
sys.dont_write_bytecode = True

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = HERE / "fine_step"
SEEDS = [17, 43, 91, 137, 211]
GROUPS = ["H0", "GR", "GC", "Teacher"]
PRIMARY = ["continuous_tip_rmse_mm", "continuous_state_relative_l2"]
SOURCE = PROJECT / "code" / "neural_physics_refinement.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def load_helpers():
    specification = importlib.util.spec_from_file_location("fine_step_readonly_helpers", SOURCE)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def check_protected_sources():
    manifest = json.loads((HERE / "protected_manifest.json").read_text(encoding="utf-8"))
    failures = [path for path, expected in manifest.items()
                if not (PROJECT / path).exists() or digest(PROJECT / path) != expected]
    if failures:
        raise RuntimeError({"protected_source_hash_mismatches": failures})
    return {"file_count": len(manifest), "all_hashes_match": True,
            "manifest_sha256": digest(HERE / "protected_manifest.json")}


def own_protocol(helpers):
    inputs = [SOURCE, PROJECT / "code/train_neural_hnn.py",
              PROJECT / "results/neural_learning_10000/protocol.json"]
    for seed in SEEDS:
        inputs.append(PROJECT / f"results/learning/data/seed_{seed}.npz")
        for group in ["GR", "GC"]:
            inputs += [PROJECT / f"results/neural_learning_10000/seed_{seed}/{group}_weights.npz",
                       PROJECT / f"results/neural_learning_10000/seed_{seed}/{group}_rollout.npz"]
        inputs += [PROJECT / f"results/neural_physics_refinement/seed_{seed}/h8_trajectories.npz",
                   PROJECT / f"results/neural_physics_refinement/seed_{seed}/h8_records.json"]
    protocol = {
        "date": "2026-11-01", "purpose": "Fixed-weight fine-step exploration; retain all outcomes.",
        "seeds": SEEDS, "groups": GROUPS, "divisors": [16, 32], "training": "None",
        "weights": "All saved 10000-update GR/GC weights; no seed or output selection.",
        "grid": "Exactly the original 914 saved times, all eight held-out initials per seed, endpoint 15.992993212174294 seconds.",
        "solver": "Original low-level float64 preconditioned implicit midpoint, fastmath=False, original stopping tolerance and iteration cap.",
        "maximum_midpoint_residual": helpers.TOL,
        "comparators": "Same-grid nonlinear midpoint teacher at each step, quadratic H0 Cayley trajectory and exact continuous H0, original DOP853 nonlinear reference.",
        "reuse_whitelist": ["load_seed", "weights_for", "neural_field", "neural_rollout", "teacher_rollout",
                            "linear_states", "cayley_preconditioner", "original_numpy_field",
                            "metrics_per_trajectory", "target_energy"],
        "forbidden_calls": ["plan", "run_experiment", "summarize", "verify_fields"],
        "statistics": "Time RMSE per trajectory, mean of eight within seed, mean and sample SD over five seeds; seed-paired GR-GC and H0-network contrasts.",
        "refinement_check": "Compare h16-to-h32 trajectory changes and metric changes with same-step GR-GC differences. h8-to-h16 retained as previous-level context.",
        "work_limit": "Run h16 for all seeds first; estimate h32 cost from measured h16 runtime before launching h32.",
        "threads": "At most two neural tasks; each process BLAS and Numba threads fixed to one.",
        "claim_limits": "Existing-data exploratory comparison only. One-output advantage, mixed seed signs, and incomplete time convergence cannot establish general complex superiority.",
        "inputs": [{"path": p.relative_to(PROJECT).as_posix(), "sha256": digest(p)} for p in inputs],
    }
    path = OUT / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Fine-step protocol/input mismatch; preserve prior outputs")
    else:
        save_json(path, protocol)
    return protocol


def verify_original_h(helpers):
    records = []
    independent_field = helpers.original_numpy_field()
    for seed in SEEDS:
        arrays = helpers.load_seed(seed)
        h0 = float(arrays["teacher_step_tau"])
        stride = int(arrays["teacher_sample_stride"])
        if stride != 20 or len(arrays["times"]) != 914:
            raise AssertionError("Original output grid differs from the declared protocol")
        expected = np.arange(914) * h0 * stride / float(arrays["omega0"])
        if np.max(abs(expected - arrays["times"])) > 1e-13:
            raise AssertionError("Original h0/stride/time fields are inconsistent")
        cayley, inverse = helpers.cayley_preconditioner(h0, arrays["nu"])
        teacher, residual, _, _ = helpers.teacher_rollout(arrays["test_initials"][:2], h0,
                                                         stride * 3, stride, cayley, inverse,
                                                         arrays["nu"], arrays["d"])
        teacher_gap = float(np.max(abs(teacher - arrays["old_teacher"][:4, :2])))
        if teacher_gap > 2e-10:
            raise AssertionError({"seed": seed, "original_teacher_gap": teacher_gap})
        for group in ["GR", "GC"]:
            params, weights = helpers.weights_for(seed, group)
            if not params[-1] or not np.array_equal(params[-2], arrays["nu"]):
                raise AssertionError("Saved prior or modal parameters differ")
            predicted, residual, _, _ = helpers.neural_rollout(arrays["test_initials"][:2], h0,
                                                               stride * 3, stride, cayley,
                                                               inverse, params)
            with np.load(PROJECT / f"results/neural_learning_10000/seed_{seed}/{group}_rollout.npz",
                         allow_pickle=False) as original:
                state_gap = float(np.max(abs(predicted - original["predicted_states"][:4, :2])))
            points = arrays["val_midpoints"][np.linspace(0, len(arrays["val_midpoints"]) - 1, 32, dtype=int)]
            independent_energy, independent_vector = independent_field(points, weights)
            computed = [helpers.neural_field(x, *params) for x in points]
            field_gap = float(np.max(abs(np.array([x[1] for x in computed]) - independent_vector)))
            energy_gap = float(np.max(abs(np.array([x[0] for x in computed]) - independent_energy)))
            if state_gap > 2e-8 or field_gap > 2e-12 or energy_gap > 2e-13:
                raise AssertionError({"seed": seed, "group": group, "state_gap": state_gap,
                                      "field_gap": field_gap, "energy_gap": energy_gap})
            records.append({"seed": seed, "group": group, "h0_tau": h0, "stride": stride,
                            "initials_checked": 2, "saved_times_checked": 4,
                            "original_h_state_gap": state_gap, "vector_field_gap": field_gap,
                            "energy_gap": energy_gap, "teacher_original_h_gap": teacher_gap,
                            "maximum_midpoint_residual": float(residual)})
    save_json(OUT / "original_h_verification.json", records)
    print("ORIGINAL_H_VERIFIED", json.dumps({"models": len(records),
          "max_state_gap": max(r["original_h_state_gap"] for r in records),
          "max_field_gap": max(r["vector_field_gap"] for r in records)}), flush=True)


def run_network(helpers, seed, group, arrays, h, nsteps, stride, cayley, inverse):
    params, weights = helpers.weights_for(seed, group)
    started = time.perf_counter()
    states, residual, max_iterations, mean_iterations = helpers.neural_rollout(
        arrays["test_initials"], h, nsteps, stride, cayley, inverse, params)
    own_energy = helpers.original_numpy_field()(states, weights)[0]
    return group, states, own_energy, {"maximum_implicit_residual": float(residual),
        "maximum_iterations": int(max_iterations), "mean_iterations": float(mean_iterations),
        "elapsed_seconds": time.perf_counter() - started}


def run_seed(helpers, seed, divisor, executor):
    directory = OUT / f"seed_{seed}"
    directory.mkdir(exist_ok=True)
    record_path = directory / f"h{divisor}_records.json"
    archive_path = directory / f"h{divisor}_trajectories.npz"
    if record_path.exists():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        if archive_path.exists() and digest(archive_path) == previous["archive_sha256"]:
            print(f"RETAINED seed={seed} divisor={divisor}", flush=True)
            return
        raise RuntimeError("Prior seed archive does not match its record")
    arrays = helpers.load_seed(seed)
    h0, omega0 = float(arrays["teacher_step_tau"]), float(arrays["omega0"])
    base_stride, ntime = int(arrays["teacher_sample_stride"]), len(arrays["times"])
    h, nsteps, stride = h0 / divisor, (ntime - 1) * base_stride * divisor, base_stride * divisor
    grid_gap = float(np.max(abs(np.arange(ntime) * stride * h / omega0 - arrays["times"])))
    if grid_gap > 1e-13:
        raise AssertionError("Refined output grid changed")
    cayley, inverse = helpers.cayley_preconditioner(h, arrays["nu"])
    started = time.perf_counter()
    teacher, residual, max_iterations, mean_iterations = helpers.teacher_rollout(
        arrays["test_initials"], h, nsteps, stride, cayley, inverse, arrays["nu"], arrays["d"])
    teacher_seconds = time.perf_counter() - started
    quadratic = helpers.linear_states(arrays["test_initials"],
        (np.arange(ntime) * stride)[:, None] * (2 * np.arctan(h * arrays["nu"] / 2))[None, :])
    linear_exact = helpers.linear_states(arrays["test_initials"],
        arrays["times"][:, None] * omega0 * arrays["nu"][None, :])
    quadratic_energy = .5 * np.sum(arrays["nu"] * (quadratic[..., :2]**2 + quadratic[..., 2:]**2), axis=-1)
    next_quadratic = quadratic @ cayley.T
    midpoint_quadratic = (quadratic + next_quadratic) / 2
    quadratic_field = np.concatenate([midpoint_quadratic[..., 2:] * arrays["nu"],
                                      -midpoint_quadratic[..., :2] * arrays["nu"]], axis=-1)
    quadratic_residual = float(np.max(abs(next_quadratic - quadratic - h * quadratic_field)))
    teacher_energy = helpers.target_energy(teacher, arrays["nu"], arrays["d"])
    completed = {
        "H0": (quadratic, quadratic_energy, {"maximum_implicit_residual": quadratic_residual, "maximum_iterations": 0,
                 "mean_iterations": 0., "elapsed_seconds": 0., "solver": "analytic Cayley rotation"}),
        "Teacher": (teacher, teacher_energy, {"maximum_implicit_residual": float(residual),
                    "maximum_iterations": int(max_iterations), "mean_iterations": float(mean_iterations),
                    "elapsed_seconds": teacher_seconds, "solver": "nonlinear midpoint"}),
    }
    futures = {group: executor.submit(run_network, helpers, seed, group, arrays, h, nsteps,
                                     stride, cayley, inverse) for group in ["GR", "GC"]}
    failures = []
    for group, future in futures.items():
        try:
            _, states, energy, diagnostics = future.result()
            diagnostics["solver"] = "fixed-weight neural midpoint"
            completed[group] = states, energy, diagnostics
        except Exception as error:
            failures.append({"seed": seed, "group": group, "divisor": divisor,
                             "error": repr(error)})
    rows, solve_rows = [], []
    available = [g for g in GROUPS if g in completed]
    for group in available:
        states, own_energy, diagnostics = completed[group]
        if states.shape != arrays["reference"].shape or not np.isfinite(states).all():
            raise AssertionError("Prediction grid/finite-state check failed")
        metrics = helpers.metrics_per_trajectory(states, arrays["reference"], teacher,
                   arrays["tip_map_m"], arrays["nu"], arrays["d"], own_energy)
        # Independent sum-based verification of the four trajectory metrics.
        for reference, prefix in [(arrays["reference"], "continuous"), (teacher, "matched_teacher")]:
            difference = states - reference
            tip_error = np.einsum("ntj,j->nt", difference[..., :2], arrays["tip_map_m"]) * 1000
            tip_rmse = np.sqrt(np.sum(tip_error**2, axis=0) / ntime)
            state_rel = np.sqrt(np.sum(difference**2, axis=(0, 2)) / np.sum(reference**2, axis=(0, 2)))
            for trajectory in range(8):
                if abs(tip_rmse[trajectory] - metrics[trajectory][prefix + "_tip_rmse_mm"]) > 1e-12:
                    raise AssertionError("Independent tip metric disagrees")
                if abs(state_rel[trajectory] - metrics[trajectory][prefix + "_state_relative_l2"]) > 1e-12:
                    raise AssertionError("Independent state metric disagrees")
        predicted_phase = np.unwrap(np.angle(states[..., :2] - 1j * states[..., 2:]), axis=0)
        reference_phase = np.unwrap(np.angle(arrays["reference"][..., :2] - 1j * arrays["reference"][..., 2:]), axis=0)
        phase_rmse = np.sqrt(np.mean((predicted_phase - reference_phase)**2, axis=(0, 2)))
        for trajectory, metric in enumerate(metrics):
            rows.append({"seed": seed, "group": group, "divisor": divisor, "step_tau": h,
                         "trajectory": trajectory, **metric,
                         "continuous_modal_phase_rmse_rad": float(phase_rmse[trajectory])})
        solve_rows.append({"seed": seed, "group": group, "divisor": divisor,
                           "internal_steps_per_trajectory": nsteps, "output_stride": stride,
                           "output_count": ntime, "output_grid_max_gap_s": grid_gap, **diagnostics})
    np.savez_compressed(archive_path, times_seconds=arrays["times"],
        continuous_reference=arrays["reference"], continuous_H0=linear_exact,
        matched_teacher=teacher, initials=arrays["test_initials"], nu=arrays["nu"], d=arrays["d"],
        tip_map_m=arrays["tip_map_m"], groups=np.array(available),
        predicted_states=np.stack([completed[g][0] for g in available]),
        own_energies=np.stack([completed[g][1] for g in available]), step_tau=h, divisor=divisor)
    save_json(record_path, {"status": "complete" if not failures else "partial",
        "seed": seed, "divisor": divisor, "failures": failures, "trajectory_metrics": rows,
        "solver_diagnostics": solve_rows, "archive_sha256": digest(archive_path)})
    means = {g: statistics.mean(x["continuous_tip_rmse_mm"] for x in rows if x["group"] == g) for g in available}
    print("FINE_STEP_READY", json.dumps({"seed": seed, "divisor": divisor,
          "continuous_tip_rmse_mm": means, "neural_seconds": {g: completed[g][2]["elapsed_seconds"]
          for g in ["GR", "GC"] if g in completed}, "failures": failures}), flush=True)


def array_metrics(difference, reference, tip):
    tip_error = np.einsum("ntj,j->nt", difference[..., :2], tip) * 1000
    return {"continuous_tip_rmse_mm": float(np.mean(np.sqrt(np.mean(tip_error**2, axis=0)))),
            "continuous_state_relative_l2": float(np.mean(np.sqrt(np.sum(difference**2, axis=(0, 2))
                        / np.sum(reference**2, axis=(0, 2)))))}


def summarize_local():
    all_rows, diagnostics, failures = [], [], []
    for path in sorted(OUT.glob("seed_*/h*_records.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        all_rows += record["trajectory_metrics"]
        diagnostics += record["solver_diagnostics"]
        failures += record["failures"]
    metrics = [key for key in all_rows[0] if key not in ["seed", "group", "divisor", "step_tau", "trajectory"]]
    seed_rows = []
    for divisor in [16, 32]:
        for seed in SEEDS:
            for group in GROUPS:
                chosen = [r for r in all_rows if (r["divisor"], r["seed"], r["group"]) == (divisor, seed, group)]
                if chosen:
                    if len(chosen) != 8:
                        raise AssertionError("Incomplete trajectory block")
                    seed_rows.append({"divisor": divisor, "seed": seed, "group": group,
                                      **{m: statistics.mean(r[m] for r in chosen) for m in metrics}})
    lookup = {(r["divisor"], r["seed"], r["group"]): r for r in seed_rows}
    aggregates, pairs, paired_summary = [], [], []
    for divisor in [16, 32]:
        for group in GROUPS:
            chosen = [r for r in seed_rows if r["divisor"] == divisor and r["group"] == group]
            if chosen:
                aggregates.append({"divisor": divisor, "group": group, "n_seeds": len(chosen),
                    "metrics": {m: {"mean": statistics.mean(r[m] for r in chosen),
                        "sample_sd": statistics.stdev(r[m] for r in chosen) if len(chosen) > 1 else None}
                        for m in metrics}})
        for a, b in [("GR", "GC"), ("H0", "GR"), ("H0", "GC")]:
            for metric in metrics[:4] + ["continuous_modal_phase_rmse_rad"]:
                subset = []
                for seed in SEEDS:
                    if (divisor, seed, a) not in lookup or (divisor, seed, b) not in lookup:
                        continue
                    first, second = lookup[divisor, seed, a][metric], lookup[divisor, seed, b][metric]
                    row = {"divisor": divisor, "seed": seed, "contrast": a + "-" + b,
                           "metric": metric, "difference": first - second,
                           "relative_improvement_fraction": (first - second) / max(abs(first), 1e-30)}
                    pairs.append(row); subset.append(row)
                if subset:
                    paired_summary.append({"divisor": divisor, "contrast": a + "-" + b, "metric": metric,
                        "n_seeds": len(subset), "mean_difference": statistics.mean(r["difference"] for r in subset),
                        "sample_sd_difference": statistics.stdev(r["difference"] for r in subset) if len(subset) > 1 else None,
                        "positive_direction_seed_count": sum(r["difference"] > 0 for r in subset),
                        "mean_relative_improvement_fraction": statistics.mean(r["relative_improvement_fraction"] for r in subset)})
    refinement = []
    for seed in SEEDS:
        for coarse_divisor, fine_divisor in [(8, 16), (16, 32)]:
            coarse_path = (PROJECT / f"results/neural_physics_refinement/seed_{seed}/h8_trajectories.npz"
                           if coarse_divisor == 8 else OUT / f"seed_{seed}/h16_trajectories.npz")
            fine_path = OUT / f"seed_{seed}/h{fine_divisor}_trajectories.npz"
            if not coarse_path.exists() or not fine_path.exists():
                continue
            with np.load(coarse_path, allow_pickle=False) as coarse, np.load(fine_path, allow_pickle=False) as fine:
                coarse_names, fine_names = list(coarse["groups"]), list(fine["groups"])
                reference, tip = fine["continuous_reference"], fine["tip_map_m"]
                if not all(g in fine_names for g in ["GR", "GC"]):
                    continue
                cross = array_metrics(fine["predicted_states"][fine_names.index("GR")]
                            - fine["predicted_states"][fine_names.index("GC")], reference, tip)
                metric_cross = {m: abs(lookup[fine_divisor, seed, "GR"][m]
                                    - lookup[fine_divisor, seed, "GC"][m]) for m in PRIMARY}
                for group in ["H0", "GR", "GC"]:
                    if group not in coarse_names or group not in fine_names:
                        continue
                    previous = coarse["predicted_states"][coarse_names.index(group)]
                    latest = fine["predicted_states"][fine_names.index(group)]
                    change = array_metrics(previous - latest, reference, tip)
                    coarse_errors = array_metrics(previous - reference, reference, tip)
                    for metric in PRIMARY:
                        metric_move = abs(coarse_errors[metric] - lookup[fine_divisor, seed, group][metric])
                        refinement.append({"seed": seed, "coarse_divisor": coarse_divisor,
                            "fine_divisor": fine_divisor, "group": group, "metric": metric,
                            "trajectory_change": change[metric], "same_step_GR_GC_trajectory_distance": cross[metric],
                            "trajectory_change_below_GR_GC_distance": change[metric] < cross[metric],
                            "metric_change": metric_move, "same_step_abs_GR_GC_metric_difference": metric_cross[metric],
                            "metric_change_below_GR_GC_difference": metric_move < metric_cross[metric]})
    write_csv(OUT / "trajectory_metrics.csv", all_rows)
    write_csv(OUT / "seed_metrics.csv", seed_rows)
    write_csv(OUT / "solver_diagnostics.csv", diagnostics)
    write_csv(OUT / "paired_seed_differences.csv", pairs)
    write_csv(OUT / "paired_summary.csv", paired_summary)
    write_csv(OUT / "refinement_vs_representation.csv", refinement)
    result = {"status": "complete" if len(seed_rows) == 40 and not failures else "partial",
              "completed_seed_model_step_blocks": len(seed_rows), "failures": failures,
              "all_models": aggregates, "paired": paired_summary,
              "refinement_comparisons": refinement,
              "maximum_implicit_residual": max(r["maximum_implicit_residual"] for r in diagnostics),
              "interpretation": "Positive GR-GC means lower GC error. Seed direction counts and full-state/tip metrics must be reported together; one-output wins do not establish general superiority."}
    save_json(OUT / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--divisors", type=int, nargs="+", choices=[16, 32], default=[16])
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    helpers = load_helpers()
    with threadpool_limits(limits=1):
        protocol = own_protocol(helpers)
        before = check_protected_sources()
        if not args.summarize_only:
            snapshot = OUT / "execution_script.py"
            if snapshot.exists() and digest(snapshot) != digest(__file__):
                raise RuntimeError("Execution source changed; preserve the accepted snapshot")
            if not snapshot.exists():
                snapshot.write_bytes(Path(__file__).read_bytes())
            save_json(OUT / ("environment_h" + "_h".join(map(str, args.divisors)) + ".json"), {"python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__, "script_sha256": digest(__file__), "workers": args.workers,
                "blas_threads": 1, "numba_threads": 1, "bytecode_writes_disabled": True,
                "utc_started": datetime.now(timezone.utc).isoformat()})
            verify_original_h(helpers)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for divisor in sorted(set(args.divisors)):
                    started = time.perf_counter()
                    for seed in SEEDS:
                        run_seed(helpers, seed, divisor, executor)
                    save_json(OUT / f"h{divisor}_runtime.json", {"divisor": divisor,
                        "all_five_seed_seconds": time.perf_counter() - started,
                        "h32_estimate_from_h16_seconds": 2 * (time.perf_counter() - started) if divisor == 16 else None})
                    summary = summarize_local()
                    print("DIVISOR_COMPLETE", divisor, "blocks", summary["completed_seed_model_step_blocks"], flush=True)
        result = summarize_local()
        after = check_protected_sources()
        for item in protocol["inputs"]:
            if digest(PROJECT / item["path"]) != item["sha256"]:
                raise RuntimeError("Fine-step input changed during execution")
        save_json(OUT / "protection_verification.json", {"before": before, "after": after,
            "experiment_input_hashes_unchanged": True, "all_original_sources_unchanged": True})
        print("FINE_STEP_STATUS", result["status"], flush=True)


if __name__ == "__main__":
    main()
