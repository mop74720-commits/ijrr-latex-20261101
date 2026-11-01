"""Read raw NPZ arrays and independently verify the completed pilot statistics."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
SEEDS = (17, 43, 91)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def archive(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def same(actual, expected):
    if not np.allclose(actual, expected, rtol=2e-12, atol=2e-16):
        raise AssertionError((actual, expected))


def recompute(prediction, reference, tip):
    # Explicit trajectory loop: first time RMSE/L2, then arithmetic trajectory
    # average. No reuse of the experiment's metric or summary functions.
    tips, states = [], []
    for trajectory in range(prediction.shape[1]):
        delta = prediction[:, trajectory] - reference[:, trajectory]
        tip_error = delta[:, :2].dot(tip) * 1000
        tips.append(float(np.sqrt(np.mean(tip_error ** 2))))
        states.append(float(np.linalg.norm(delta.ravel()) / np.linalg.norm(reference[:, trajectory].ravel())))
    return {"tip_rmse_mm_per_trajectory": tips, "tip_rmse_mm_mean": float(np.mean(tips)),
            "state_relative_l2_per_trajectory": states, "state_relative_l2_mean": float(np.mean(states))}


def main():
    manifest = read(OUT.parent / "protected_manifest.json")
    for relative, expected in manifest.items():
        if digest(PROJECT / relative) != expected:
            raise AssertionError(f"Protected file changed: {relative}")
    tree = ast.parse((PROJECT / "code/train_neural_hnn.py").read_text(encoding="utf-8-sig"))
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "numpy_energy_field")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "original_numpy_energy_field", "exec"), namespace)
    field = namespace["numpy_energy_field"]
    summary = read(OUT / "summary.json")
    rows, gradient, teacher_errors, h0_errors, all_residuals, export_gaps = [], {}, [], [], [], []
    for seed in SEEDS:
        common = archive(OUT / f"fine_common_seed_{seed}.npz")
        common_record = read(OUT / f"fine_common_seed_{seed}.json")
        assert digest(OUT / f"fine_common_seed_{seed}.npz") == common_record["trajectory_sha256"]
        reference, tip = common["continuous_reference"], common["tip_map_m"]
        assert reference.shape == (914, 8, 4)
        teacher_errors.append(recompute(common["teacher_states"], reference, tip))
        h0 = recompute(common["H0_states"], reference, tip)
        h0_errors.append(h0)
        for key, value in h0.items():
            same(value, common_record["H0"]["continuous"][key])
        for condition in ("noise_1e-6", "noise_0"):
            data_path = OUT / "data" / condition / f"seed_{seed}.npz"
            data = archive(data_path)
            data_record = read(data_path.with_name(f"seed_{seed}_diagnostics.json"))
            assert digest(data_path) == data_record["dataset_sha256"] and data_record["angle_gate_passed"]
            for group in ("GR", "GC"):
                dest = OUT / condition / f"seed_{seed}"
                fit = read(dest / f"{group}_training.json")
                history = read(dest / f"{group}_history.json")
                assert fit["status"] == "completed" and fit["real_parameter_count"] == 673
                assert digest(data_path) == fit["dataset_sha256"]
                assert digest(dest / f"{group}_weights.npz") == fit["weights_sha256"]
                best = min(history, key=lambda item: item["validation_normalized_mse"])
                assert best["step"] == fit["selected_validation_step"]
                assert [item["step"] for item in history] == list(range(0, 10001, 100))
                weights = archive(dest / f"{group}_weights.npz")
                predicted_field = field(data["val_midpoints"], weights)[1]
                scale = np.maximum(np.sqrt(np.mean(data["train_labels"] ** 2, axis=0)), 1e-6)
                indices = np.random.default_rng(seed + 3000).choice(len(predicted_field), size=2048, replace=False)
                selected_mse = np.mean(((predicted_field[indices] - data["val_labels"][indices]) / scale) ** 2)
                same(selected_mse, fit["validation_normalized_mse"])
                clean_score = float(np.sqrt(np.mean(((predicted_field - data["val_continuous_field"]) / scale) ** 2)))
                same(clean_score, fit["clean_full_validation_scaled_gradient_rms"])
                gradient[condition, seed, group] = clean_score
                export_gaps.append(fit["numpy_torch_field_max_difference"])
                fine_path = dest / f"{group}_fine.npz"
                fine, record = archive(fine_path), read(dest / f"{group}_fine.json")
                assert record["status"] == "completed"
                assert digest(fine_path) == record["trajectory_sha256"]
                assert np.array_equal(fine["continuous_reference"], reference)
                assert np.array_equal(fine["times_seconds"], common["times_seconds"])
                assert np.array_equal(fine["teacher_states"], common["teacher_states"])
                for target, stored_key in (("continuous_reference", "continuous"), ("teacher_states", "matched_teacher")):
                    values = recompute(fine["predicted_states"], fine[target], tip)
                    for key, value in values.items():
                        same(value, record["metrics"][stored_key][key])
                values = recompute(fine["predicted_states"], reference, tip)
                rows.append({"condition": condition, "seed": seed, "group": group,
                             "state_relative_l2_mean": values["state_relative_l2_mean"],
                             "tip_rmse_mm_mean": values["tip_rmse_mm_mean"],
                             "clean_scaled_gradient_rms": clean_score,
                             "training_seconds": fit["training_seconds"],
                             "rollout_seconds": record["elapsed_seconds"]})
                all_residuals.append(record["maximum_implicit_residual"])
    for condition in ("noise_1e-6", "noise_0"):
        saved = summary["conditions"][condition]
        logs = [np.log(gradient[condition, seed, "GR"] / gradient[condition, seed, "GC"]) for seed in SEEDS]
        same(logs, saved["nomination"]["log_GR_over_GC_per_seed"])
        same(np.mean(logs), saved["nomination"]["mean_log_GR_over_GC"])
        for metric in ("state_relative_l2_mean", "tip_rmse_mm_mean"):
            for group in ("H0", "GR", "GC"):
                values = [row[metric] for row in rows if row["condition"] == condition and row["group"] == group] if group != "H0" else [row[metric] for row in h0_errors]
                same(np.mean(values), saved["models"][group][metric]["mean"])
                same(np.std(values, ddof=1), saved["models"][group][metric]["sample_sd"])
            paired = saved["paired_rollout"][metric]
            values = {(row["seed"], row["group"]): row[metric] for row in rows if row["condition"] == condition}
            improvements = [(values[seed, "GR"] - values[seed, "GC"]) / values[seed, "GR"] for seed in SEEDS]
            same(improvements, paired["GC_relative_improvement_per_seed"])
            same(np.mean(improvements), paired["GC_mean_paired_relative_improvement"])
            assert sum(value > 0 for value in improvements) == paired["GC_wins"]
    with (OUT / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {"passed": True, "fits_checked": len(rows), "rollouts_checked": len(rows),
             "protected_files_checked": len(manifest), "protected_files_unchanged": True,
             "raw_trajectory_metric_recomputation": "Explicit trajectory loop; time RMSE/full L2 then8 means, seed sampleSD(ddof1)",
             "checkpoint_selection_recomputed_with_original_noisy_validation_subset": True,
             "maximum_numpy_torch_field_gap": max(export_gaps), "maximum_neural_midpoint_residual": max(all_residuals),
             "teacher_h0_div16_continuous_error_per_seed": [{"seed": seed, **record} for seed, record in zip(SEEDS, teacher_errors)],
             "teacher_h0_div16_continuous_error_seed_mean": {metric: float(np.mean([record[metric] for record in teacher_errors])) for metric in ("state_relative_l2_mean", "tip_rmse_mm_mean")},
             "total_training_seconds": sum(row["training_seconds"] for row in rows),
             "total_neural_rollout_seconds": sum(row["rollout_seconds"] for row in rows),
             "summary_sha256": digest(OUT / "summary.json"), "audit_script_sha256": digest(Path(__file__))}
    (OUT / "independent_verification.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if not key.endswith("per_seed")}))


if __name__ == "__main__":
    main()
