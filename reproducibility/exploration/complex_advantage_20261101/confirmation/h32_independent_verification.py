"""Independently recompute the five fresh-seed h32 confirmation metrics."""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np

OUT = Path(__file__).resolve().parent
HERE = OUT.parent
PROJECT = HERE.parents[1]
SEEDS = (307, 419, 541, 653, 809)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def same(actual, expected):
    if not np.allclose(actual, expected, rtol=2e-12, atol=2e-16):
        raise AssertionError((actual, expected))


def calculate(prediction, reference, tip):
    tip_rmse, state_l2 = [], []
    for trajectory in range(8):
        difference = prediction[:, trajectory] - reference[:, trajectory]
        tip_difference_mm = (difference[:, :2] @ tip) * 1000
        tip_rmse.append(float(np.sqrt(np.mean(tip_difference_mm ** 2))))
        state_l2.append(float(np.linalg.norm(difference.ravel()) / np.linalg.norm(reference[:, trajectory].ravel())))
    return {"tip_rmse_mm_per_trajectory": tip_rmse, "tip_rmse_mm_mean": float(np.mean(tip_rmse)),
            "state_relative_l2_per_trajectory": state_l2, "state_relative_l2_mean": float(np.mean(state_l2))}


def moments(values):
    return {"n": len(values), "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}


def main():
    frozen = read(OUT / "frozen_nomination.json")
    assert frozen["confirmation_seeds"] == list(SEEDS)
    assert frozen["chosen"]["name"] == "traj1_updates10000"
    assert frozen["chosen"]["train_trajectories"] == 1
    assert frozen["chosen"]["amplitude_multiplier"] == 1.
    assert frozen["chosen"]["noise_bound"] == 1e-6
    for key in ("execution_inputs_sha256", "pilot_evidence_sha256"):
        for relative, expected in frozen[key].items():
            assert sha(HERE / relative) == expected
    manifest = read(HERE / "protected_manifest.json")
    for relative, expected in manifest.items():
        assert sha(PROJECT / relative) == expected
    spec = read(PROJECT / "results/learning/inputs/reference_model_spec.json")
    rows, teacher_rows, export_gaps, residuals = [], [], [], []
    for seed in SEEDS:
        dest = OUT / f"seed_{seed}"
        baseline = read(dest / "baselines_h32.json")
        common_h0 = None
        for group in ("GR", "GC"):
            training = read(dest / f"{group}_training.json")
            record = read(dest / f"{group}_h32.json")
            data_path = HERE / training["dataset_path"]
            data = load(data_path)
            assert training["dataset_sha256"] == sha(data_path) == record["dataset_sha256"]
            assert record["nomination_sha256"] == training["nomination_sha256"] == sha(OUT / "frozen_nomination.json")
            assert record["weights_sha256"] == training["weights_sha256"] == sha(dest / f"{group}_weights.npz")
            assert record["trajectory_sha256"] == sha(dest / f"{group}_h32.npz")
            assert training["status"] == record["status"] == "completed"
            assert training["real_parameter_count"] == 673
            assert training["training_steps"] == 10000
            selected = np.random.default_rng(seed + 41000).permutation(np.arange(12))[:1].tolist()
            assert training["train_trajectory_ids"] == selected
            assert np.array_equal(data["nu"], np.array(spec["nu"]))
            assert np.array_equal(data["d"], np.array(spec["slope_map_d"]))
            history = read(dest / f"{group}_history.json")
            best = min(history, key=lambda item: item["validation_normalized_mse"])
            assert best["step"] == training["selected_validation_step"]
            assert [item["step"] for item in history] == list(range(0, 10001, 100))
            trajectory = load(dest / f"{group}_h32.npz")
            reference = data["test_reference_states"].transpose(1, 0, 2)
            assert reference.shape == trajectory["predicted_states"].shape == (914, 8, 4)
            assert np.array_equal(trajectory["continuous_reference"], reference)
            assert np.array_equal(trajectory["times_seconds"], data["test_times"])
            same(trajectory["predicted_states"][0], data["test_initials"])
            same(record["h_tau"], float(data["teacher_step_tau"]) / 32)
            assert record["internal_steps"] == int(data["teacher_nsteps"]) * 32
            calculated = calculate(trajectory["predicted_states"], reference, data["tip_map_m"])
            matched = calculate(trajectory["predicted_states"], trajectory["teacher"], data["tip_map_m"])
            for category, values in (("continuous", calculated), ("matched_teacher", matched)):
                for key, value in values.items():
                    same(value, record["metrics"][category][key])
            rows.append({"seed": seed, "group": group, **calculated})
            export_gaps.append(record["field_export_gap"])
            residuals.append(record["maximum_implicit_residual"])
            if group == "GR":
                h0 = calculate(trajectory["H0"], reference, data["tip_map_m"])
                for key, value in h0.items():
                    same(value, baseline["H0"]["continuous"][key])
                rows.append({"seed": seed, "group": "H0", **h0})
                teacher = calculate(trajectory["teacher"], reference, data["tip_map_m"])
                for key, value in teacher.items():
                    same(value, baseline["teacher"]["continuous"][key])
                teacher_rows.append({"seed": seed, **teacher})
                common_h0 = trajectory["H0"]
            else:
                assert np.array_equal(common_h0, trajectory["H0"])
    models, paired = {}, {}
    for group in ("H0", "GR", "GC"):
        models[group] = {key: moments([row[key] for row in rows if row["group"] == group]) for key in ("state_relative_l2_mean", "tip_rmse_mm_mean")}
    for key in ("state_relative_l2_mean", "tip_rmse_mm_mean"):
        values = {(row["seed"], row["group"]): row[key] for row in rows}
        gr = np.array([values[seed, "GR"] for seed in SEEDS])
        gc = np.array([values[seed, "GC"] for seed in SEEDS])
        improvement = (gr - gc) / gr
        paired[key] = {"seeds": list(SEEDS), "GR": gr.tolist(), "GC": gc.tolist(),
                       "GC_relative_improvement_per_seed": improvement.tolist(),
                       "mean_paired_relative_improvement": float(np.mean(improvement)),
                       "GC_wins": int(np.sum(improvement > 0)),
                       "rule_passed": bool(np.mean(improvement) >= .05 and np.sum(improvement > 0) >= 4)}
    audit = {"passed": True, "divisor": 32, "role": "Numerical robustness check; h64 is the frozen primary level",
             "fresh_seeds": list(SEEDS), "trajectory_count_per_seed": 8,
             "statistic_order": "Per-trajectory time RMSE/state relativeL2, mean8, seed mean/sampleSD(ddof1)",
             "models": models, "paired": paired,
             "both_metrics_rule_passed_at_h32": all(row["rule_passed"] for row in paired.values()),
             "per_seed_records": rows, "teacher_continuous_error_per_seed": teacher_rows,
             "maximum_field_export_gap": max(export_gaps), "maximum_neural_midpoint_residual": max(residuals),
             "protected_files_checked": len(manifest), "protected_files_unchanged": True,
             "nomination_sha256": sha(OUT / "frozen_nomination.json"), "audit_script_sha256": sha(Path(__file__))}
    (OUT / "h32_independent_verification.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key not in ("per_seed_records", "teacher_continuous_error_per_seed")}))


if __name__ == "__main__":
    main()
