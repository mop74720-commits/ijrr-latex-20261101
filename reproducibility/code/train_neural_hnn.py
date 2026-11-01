"""Train matched-budget real/complex HNNs on equation-(36) midpoint data.

This experiment concerns the conservative branch H+i*0. Complex hidden
weights and activations do not validate the frozen dissipative decoder.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import os
import platform
import time
# Avoid two OpenMP runtimes in this Conda/PyTorch process. Do not suppress
# duplicate-runtime errors: use MKL's sequential backend instead.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "learning" / "data"
OUT = ROOT / "results" / "neural_learning"
SEEDS = [17, 43, 91, 137, 211]
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


class EnergyNetwork(nn.Module):
    def __init__(self, family, prior, scale, energy_scale, nu):
        super().__init__()
        self.family, self.prior = family, prior
        self.register_buffer("scale", torch.tensor(scale))
        self.register_buffer("energy_scale", torch.tensor(float(energy_scale)))
        self.register_buffer("nu", torch.tensor(nu))
        if family == "R":
            self.first = nn.Linear(4, 20)
            self.second = nn.Linear(20, 26)
            self.readout = nn.Linear(26, 1)
            for layer in [self.first, self.second]:
                nn.init.xavier_normal_(layer.weight, gain=.8)
                nn.init.zeros_(layer.bias)
        else:
            for name, shape in [("a1", (16, 2)), ("b1", (16, 2)),
                                ("a2", (16, 16)), ("b2", (16, 16))]:
                parameter = nn.Parameter(torch.empty(shape))
                nn.init.normal_(parameter, std=.8 / math.sqrt(sum(shape)))
                setattr(self, name, parameter)
            self.bias1r = nn.Parameter(torch.zeros(16))
            self.bias1i = nn.Parameter(torch.zeros(16))
            self.bias2r = nn.Parameter(torch.zeros(16))
            self.bias2i = nn.Parameter(torch.zeros(16))
            self.readout = nn.Linear(32, 1)
        nn.init.normal_(self.readout.weight, std=.03)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x):
        y = x / self.scale
        if self.family == "R":
            y = torch.tanh(self.first(y))
            y = torch.tanh(self.second(y))
        else:
            z = torch.complex(y[..., :2], y[..., 2:]) / math.sqrt(2)
            z = nn.functional.linear(z, torch.complex(self.a1, self.b1),
                                     torch.complex(self.bias1r, self.bias1i))
            z = torch.complex(torch.tanh(z.real), torch.tanh(z.imag))
            z = nn.functional.linear(z, torch.complex(self.a2, self.b2),
                                     torch.complex(self.bias2r, self.bias2i))
            y = torch.cat([torch.tanh(z.real), torch.tanh(z.imag)], dim=-1)
        value = self.energy_scale * self.readout(y).squeeze(-1)
        if self.prior:
            value = value + .5 * torch.sum(self.nu * (x[..., :2] ** 2 + x[..., 2:] ** 2), dim=-1)
        return value

    def field(self, x, create_graph=False):
        x = x.detach().requires_grad_(True)
        gradient = torch.autograd.grad(self(x).sum(), x, create_graph=create_graph)[0]
        return torch.cat([gradient[..., 2:], -gradient[..., :2]], dim=-1)

    def export(self):
        get = lambda x: x.detach().cpu().numpy().copy()
        if self.family == "R":
            w1, b1 = get(self.first.weight), get(self.first.bias)
            w2, b2 = get(self.second.weight), get(self.second.bias)
        else:
            a, b = get(self.a1), get(self.b1)
            w1 = np.block([[a, -b], [b, a]]) / math.sqrt(2)
            b1 = np.r_[get(self.bias1r), get(self.bias1i)]
            a, b = get(self.a2), get(self.b2)
            w2 = np.block([[a, -b], [b, a]])
            b2 = np.r_[get(self.bias2r), get(self.bias2i)]
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2,
                "w3": get(self.readout.weight).ravel(), "b3": get(self.readout.bias),
                "scale": get(self.scale), "energy_scale": get(self.energy_scale),
                "nu": get(self.nu), "prior": np.array(self.prior)}


def numpy_energy_field(state, weights):
    y = state / weights["scale"]
    first = np.tanh(y @ weights["w1"].T + weights["b1"])
    second = np.tanh(first @ weights["w2"].T + weights["b2"])
    energy = weights["energy_scale"] * (second @ weights["w3"] + weights["b3"][0])
    back = weights["w3"] * (1 - second * second)
    gradient = weights["energy_scale"] * ((back @ weights["w2"]) * (1 - first * first)) @ weights["w1"] / weights["scale"]
    if bool(weights["prior"]):
        energy = energy + .5 * np.sum(weights["nu"] * (state[..., :2] ** 2 + state[..., 2:] ** 2), axis=-1)
        gradient = gradient + state * np.r_[weights["nu"], weights["nu"]]
    field = np.concatenate([gradient[..., 2:], -gradient[..., :2]], axis=-1)
    return energy, field


def rollout(initials, h, nsteps, stride, weights):
    state = np.array(initials, dtype=float, copy=True)
    saved = [state.copy()]
    maximum_residual, iterations = 0., 0
    for step in range(1, nsteps + 1):
        new = state + h * numpy_energy_field(state, weights)[1]
        for iteration in range(25):
            nxt = state + h * numpy_energy_field((state + new) / 2, weights)[1]
            difference = float(np.max(np.abs(nxt - new)))
            new = nxt
            if difference < 1e-12:
                break
        else:
            raise RuntimeError(f"Implicit midpoint failed at step {step}, residual {difference:g}")
        residual = float(np.max(np.abs(new - state - h * numpy_energy_field((state + new) / 2, weights)[1])))
        maximum_residual = max(maximum_residual, residual)
        iterations += iteration + 1
        state = new
        if not np.isfinite(state).all() or np.max(np.abs(state)) > 100:
            raise RuntimeError(f"Rollout became nonfinite or exceeded diagnostic bound at step {step}")
        if step % stride == 0:
            saved.append(state.copy())
    return np.array(saved), maximum_residual, iterations / nsteps


def as_time_first(states, ntrajectories):
    if states.shape[1] == ntrajectories:
        return states
    if states.shape[0] == ntrajectories:
        return np.swapaxes(states, 0, 1)
    raise ValueError(f"Unrecognized trajectory array shape {states.shape}")


def target_energy(states, nu, d):
    angle = states[..., :2] @ d
    return .5 * np.sum(nu * (states[..., :2] ** 2 + states[..., 2:] ** 2), axis=-1) + 1 - np.cos(angle) - .5 * angle ** 2


def metrics(prediction, reference, teacher, weights, nu, d, tip_map):
    error = prediction - reference
    state_relative = np.linalg.norm(error, axis=(0, 2)) / np.maximum(np.linalg.norm(reference, axis=(0, 2)), 1e-15)
    tip_error = error[..., :2] @ tip_map * 1e3
    teacher_tip_error = (prediction - teacher)[..., :2] @ tip_map * 1e3
    phase_pred = np.unwrap(np.arctan2(-prediction[..., 2:], prediction[..., :2]), axis=0)
    phase_ref = np.unwrap(np.arctan2(-reference[..., 2:], reference[..., :2]), axis=0)
    initial_energy = target_energy(reference[0], nu, d)
    own_energy = numpy_energy_field(prediction, weights)[0]
    true_energy = target_energy(prediction, nu, d)
    return {
        "state_relative_l2_mean": float(np.mean(state_relative)),
        "tip_rmse_mm_mean": float(np.mean(np.sqrt(np.mean(tip_error ** 2, axis=0)))),
        "tip_rmse_mm_per_trajectory": np.sqrt(np.mean(tip_error ** 2, axis=0)).tolist(),
        "teacher_tip_rmse_mm_mean": float(np.mean(np.sqrt(np.mean(teacher_tip_error ** 2, axis=0)))),
        "max_phase_error_rad": float(np.max(np.abs(phase_pred - phase_ref))),
        "teacher_state_relative_l2_mean": float(np.mean(np.linalg.norm(prediction - teacher, axis=(0, 2)) / np.maximum(np.linalg.norm(teacher, axis=(0, 2)), 1e-15))),
        "max_own_energy_drift_relative": float(np.max(np.abs(own_energy - own_energy[0]) / np.maximum(np.abs(initial_energy), 1e-15))),
        "max_target_energy_drift_relative": float(np.max(np.abs(true_energy - true_energy[0]) / np.maximum(np.abs(initial_energy), 1e-15))),
    }


def train_one(seed, group, arrays, steps, batch_size):
    family, prior = ("R" if group in ["R", "GR"] else "C"), group.startswith("G")
    torch.manual_seed(seed + 1000)
    rng = np.random.default_rng(seed + 2000)
    x = arrays["train_midpoints"].reshape(-1, 4)
    y = arrays["train_labels"].reshape(-1, 4)
    vx = arrays["val_midpoints"].reshape(-1, 4)
    vy = arrays["val_labels"].reshape(-1, 4)
    xscale = np.maximum(np.sqrt(np.mean(x * x, axis=0)), 1e-4)
    yscale = np.maximum(np.sqrt(np.mean(y * y, axis=0)), 1e-6)
    energy_scale = max(float(np.mean(np.abs(x * y))), 1e-5)
    net = EnergyNetwork(family, prior, xscale, energy_scale, arrays["nu"])
    count = sum(p.numel() for p in net.parameters())
    assert count == 673, (group, count)
    tx, ty = torch.tensor(x), torch.tensor(y)
    val_indices = np.random.default_rng(seed + 3000).choice(len(vx), size=min(2048, len(vx)), replace=False)
    tvx, tvy = torch.tensor(vx[val_indices]), torch.tensor(vy[val_indices])
    normalization = torch.tensor(yscale)
    optimizer = torch.optim.Adam(net.parameters(), lr=.002, weight_decay=0.)
    best, best_state, best_step = float("inf"), None, None
    history = []
    started = time.perf_counter()
    for step in range(steps + 1):
        if step % 100 == 0 or step == steps:
            val = float(torch.mean(((net.field(tvx) - tvy) / normalization) ** 2).detach())
            history.append({"step": step, "validation_normalized_mse": val})
            if val < best:
                best, best_step = val, step
                best_state = {name: value.detach().clone() for name, value in net.state_dict().items()}
        if step == steps:
            break
        indices = rng.integers(0, len(tx), size=batch_size)
        optimizer.zero_grad(set_to_none=True)
        prediction = net.field(tx[indices], create_graph=True)
        loss = torch.mean(((prediction - ty[indices]) / normalization) ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss for {seed}/{group}")
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.)
        optimizer.step()
        if step in [steps // 2, 3 * steps // 4]:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= .3
    training_seconds = time.perf_counter() - started
    net.load_state_dict(best_state)
    weights = net.export()
    check_states = vx[:128]
    pytorch_field = net.field(torch.tensor(check_states)).detach().numpy()
    numpy_field = numpy_energy_field(check_states, weights)[1]
    export_error = float(np.max(np.abs(pytorch_field - numpy_field)))
    if export_error > 1e-10:
        raise AssertionError(f"Exported analytic derivative does not match PyTorch: {export_error}")
    row = {"seed": seed, "group": group, "real_parameter_count": count,
           "training_steps": steps, "selected_validation_step": best_step,
           "validation_normalized_mse": best, "training_seconds": training_seconds,
           "numpy_torch_field_max_difference": export_error}
    return weights, row, history


def summarize():
    records = []
    for path in sorted(OUT.glob("seed_*/[RCG]*_metrics.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        return
    scalar_keys = [key for key, value in records[0].items() if not isinstance(value, (list, dict))]
    with (OUT / "all_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    paired = []
    for seed in SEEDS:
        by_group = {r["group"]: r for r in records if r["seed"] == seed and r.get("status") == "completed"}
        if set(by_group) != {"R", "C", "GR", "GC"}:
            continue
        for metric in ["tip_rmse_mm_mean", "teacher_tip_rmse_mm_mean", "state_relative_l2_mean", "teacher_state_relative_l2_mean", "training_seconds"]:
            r, c, gr, gc = [by_group[g][metric] for g in ["R", "C", "GR", "GC"]]
            for contrast, value in [("R-GR", r-gr), ("C-GC", c-gc), ("R-C", r-c),
                                    ("GR-GC", gr-gc), ("(GR-GC)-(R-C)", (gr-gc)-(r-c))]:
                paired.append({"seed": seed, "metric": metric, "contrast": contrast, "difference": value})
    if paired:
        with (OUT / "paired_contrasts.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
    summary = {"run_count": len(records), "completed": sum(r.get("status") == "completed" for r in records),
               "failures": [{"seed": r["seed"], "group": r["group"], "reason": r.get("failure")} for r in records if r.get("status") != "completed"],
               "complete_paired_seeds": sorted(set(r["seed"] for r in paired)),
               "models": {}, "paired_contrasts": []}
    for group in ["R", "C", "GR", "GC"]:
        rows = [r for r in records if r["group"] == group and r.get("status") == "completed"]
        if not rows:
            continue
        summary["models"][group] = {key: {"mean": float(np.mean([r[key] for r in rows])),
                                                "sample_sd": float(np.std([r[key] for r in rows], ddof=1)) if len(rows)>1 else None}
                                      for key in ["tip_rmse_mm_mean", "teacher_tip_rmse_mm_mean", "state_relative_l2_mean", "training_seconds", "max_phase_error_rad", "max_own_energy_drift_relative"]}
    for metric in sorted(set(row["metric"] for row in paired)):
        for contrast in ["R-GR", "C-GC", "R-C", "GR-GC", "(GR-GC)-(R-C)"]:
            values = [r["difference"] for r in paired if r["metric"] == metric and r["contrast"] == contrast]
            summary["paired_contrasts"].append({"metric": metric, "contrast": contrast, "n": len(values),
                                                 "mean": float(np.mean(values)),
                                                 "sample_sd": float(np.std(values, ddof=1)) if len(values)>1 else None})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--groups", nargs="+", choices=["R", "C", "GR", "GC"], default=["R", "C", "GR", "GC"])
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-seconds", type=float, default=16.)
    parser.add_argument("--output-name", default="neural_learning")
    args = parser.parse_args()
    if not args.output_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Output name must be a single plain directory name.")
    OUT = ROOT / "results" / args.output_name
    OUT.mkdir(exist_ok=True)
    protocol = {"scope": "Conservative branch only; actual end-to-end networks, distinct from the fixed-feature baseline.",
                "groups": ["R", "C", "GR", "GC"], "seeds": SEEDS,
                "real_architecture": [4, 20, 26, 1], "complex_architecture": [2, 16, 16],
                "complex_readout": "Real linear readout of concatenated real and imaginary channels; H+i0.",
                "real_parameter_count_each": 673, "activation": "tanh (componentwise for complex hidden values)",
                "prior": "Fixed physical quadratic Hamiltonian; no extra fitted parameters.",
                "decoder": "Canonical real-coordinate gradient (source eq.12 conservative case); no use of disputed frozen Wirtinger prefactors.",
                "training_data": "Equation-(36) midpoint states/increments; entire held-out trajectories.",
                "optimizer": "Adam, lr=.002 then x.3 at 1/2 and 3/4 of updates; gradient clipping 10.",
                "training_steps": args.steps, "batch_size": args.batch_size,
                "selection": "Minimum MSE on fixed 2048-point subset of separate validation trajectories; evaluated every 100 updates.",
                "scaling": "Shared training-only RMS input/output scales; gradients retain chain rule in physical canonical coordinates.",
                "max_seconds": args.max_seconds, "rollout": "Same implicit midpoint and tolerance for all neural models.",
                "statistics": "Seed-paired contrasts; sample standard deviations are descriptive, not a population significance claim.",
                "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "torch_threads": 1, "mkl_threading_layer": os.environ["MKL_THREADING_LAYER"]}}
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError("Protocol differs from existing run; create a separately named experiment rather than overwrite.")
    else:
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    for seed in args.seeds:
        file = DATA / f"seed_{seed}.npz"
        if not file.exists():
            raise FileNotFoundError(f"Shared equation-(36) dataset is not ready: {file}")
        with np.load(file) as source:
            arrays = {key: source[key] for key in source.files}
        dest = OUT / f"seed_{seed}"
        dest.mkdir(exist_ok=True)
        for group in args.groups:
            metric_file = dest / f"{group}_metrics.json"
            if metric_file.exists():
                print(f"EXISTING {seed}/{group}", flush=True)
                continue
            weights, row, history = train_one(seed, group, arrays, args.steps, args.batch_size)
            np.savez_compressed(dest / f"{group}_weights.npz", **weights)
            (dest / f"{group}_training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            h = float(arrays["teacher_step_tau"])
            stride = int(arrays["teacher_sample_stride"])
            omega0 = float(arrays["omega0"])
            nsteps = min(int(arrays["teacher_nsteps"]), int(args.max_seconds * omega0 / h))
            nsteps = nsteps // stride * stride
            initials = arrays["test_initials"]
            ntime = nsteps // stride + 1
            reference = as_time_first(arrays["test_reference_states"], len(initials))[:ntime]
            teacher = as_time_first(arrays["test_teacher_states"], len(initials))[:ntime]
            started = time.perf_counter()
            try:
                prediction, residual, iterations = rollout(initials, h, nsteps, stride, weights)
                row.update(metrics(prediction, reference, teacher, weights, arrays["nu"], arrays["d"], arrays["tip_map_m"]))
                row.update({"status": "completed", "forecast_seconds": nsteps*h/omega0,
                            "forecast_seconds_wall": time.perf_counter()-started,
                            "max_implicit_residual": residual, "mean_fixed_point_iterations": iterations})
                np.savez_compressed(dest / f"{group}_rollout.npz", predicted_states=prediction,
                                    reference_states=reference, teacher_states=teacher,
                                    times_seconds=np.arange(ntime)*h*stride/omega0)
            except RuntimeError as error:
                row.update({"status": "rollout_failed", "failure": str(error)})
            metric_file.write_text(json.dumps(row, indent=2, allow_nan=False), encoding="utf-8")
            summarize()
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
