"""Compare three physically scalar magnetic-input discretizations of the same beam.

I: forced implicit midpoint; II: free midpoint then full potential kick;
III: half potential kick, free midpoint, half potential kick.
The held input is a parameter in fixed-input symplectic checks. The full
sampled-feedback derivative is separately checked and not called symplectic.
No weights are trained and no formula preceding (36) is changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import root
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/control_comparison"
SEEDS = [17, 43, 91, 137, 211]
METHODS = ["I", "II", "III"]
MULTIPLIERS = [4., 2., 1., .5]
KP, KD, UMAX, UAMP = 1., .5, .2, .05
RTOL, ATOL = 1e-12, 1e-14
J = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_and_data():
    spec_path = ROOT / "results/learning/inputs/reference_model_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    initials, inputs = [], [{"path": spec_path.relative_to(ROOT).as_posix(), "sha256": sha(spec_path)}]
    for seed in SEEDS:
        path = ROOT / f"results/learning/data/seed_{seed}.npz"
        with np.load(path, allow_pickle=False) as data:
            assert np.array_equal(data["nu"], np.array(spec["nu"]))
            assert np.array_equal(data["d"], np.array(spec["slope_map_d"]))
            initials.append(data["test_initials"])
            h0 = float(data["teacher_step_tau"])
        inputs.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)})
    return spec, np.concatenate(initials), h0, inputs


def plan():
    spec, initials, h0, inputs = model_and_data()
    omega0 = float(spec["omega0_rad_s"])
    steps = int(8 * omega0 / h0) // 8 * 8
    protocol = {
        "schema_version": 1,
        "object": "Same explicit conservative two-mode magnetic beam h_* and scalar physical field ratio u=By/B0.",
        "energy": ".5 sum nu*(X^2+Y^2)+1-cos(d.X)-.5*(d.X)^2",
        "input_potential": "V_u(X)=-u sin(d.X); no independent state-component actuators.",
        "methods": {"I": "Implicit midpoint of h_*+V_u with input fixed during each step.",
                    "II": "Free implicit midpoint followed by one full exact potential kick (Lie composition).",
                    "III": "Half exact potential kick, free implicit midpoint, half kick (symmetric composition)."},
        "kick": "K_s^u(X,Y)=(X,Y+s*u*d*cos(d.X)); type-2 generating function X.P-s*u*sin(d.X).",
        "open_loop": "u(tau)=.05*(1-exp(-(tau/(.25*omega0))^2))*sin(.85*tau); all schemes evaluate at each step midpoint.",
        "reference_trajectory": "One common feasible trajectory: same model from zero state under smooth open-loop input, DOP853 dense output.",
        "feedback": "u_k=clip(u_ref(tau_k)-kp*(s(x_k)-s_ref(tau_k))-kd*(v(x_k)-v_ref(tau_k)), -umax,umax), held on the full step.",
        "output": "s=sin(d.X); v=cos(d.X)*d.(nu*Y), both use dimensionless time tau.",
        "kp": KP, "kd": KD, "umax": UMAX, "reference_input_amplitude": UAMP,
        "B0_T": .03, "By_budget_T": .006,
        "budget_scope": "Budget matches the supplied model parameterization; not a calibrated real-actuator limit.",
        "seeds": SEEDS, "initials_per_seed": 8,
        "initials": "All forty original held-out learning initials; none selected by controller performance.",
        "training": "None. Explicit physical h_* is used; no learned-control, HNN compensation, or robustness claim.",
        "step_multipliers_of_teacher_h0": MULTIPLIERS, "base_h_tau": h0,
        "base_steps": steps, "duration_tau": steps * h0, "duration_s": steps * h0 / omega0,
        "common_output_stride_in_h0": 8, "samples": steps // 8 + 1,
        "closed_loop_baselines": ["No-control physical trajectory", "Independent continuous-feedback physical trajectory"],
        "reference_tolerances": {"method": "DOP853", "rtol": RTOL, "atol": ATOL},
        "tight_reference_tolerances": {"rtol": 4e-14, "atol": 2e-16},
        "solver": "Algebraic scalar reduction of forced midpoint, residual <=3e-13; exact kicks. No artificial damping.",
        "statistics": "Per trajectory over common times; eight-trajectory mean per seed; five-seed mean and sample SD ddof=1. Initial-state blocks, not five independent controller training runs.",
        "primary_metrics": ["tracking_tip_rmse_mm", "tail_half_tracking_tip_rmse_mm", "tracking_port_rmse", "continuous_controlled_flow_tip_rmse_mm", "continuous_controlled_flow_state_relative_l2"],
        "power_balance": "Use physical base energy h_* and the common held-input work u_k*(sin(theta_next)-sin(theta_old)); additionally retain each scheme subflow/quadrature work balance as a separate diagnostic.",
        "secondary_metrics": ["integral_u_squared_seconds", "max_abs_u", "saturation_fraction", "discrete_work_balance_relative", "midpoint_residual", "full_4d_symplectic_defect"],
        "order_scope": "Open-loop smooth-input orders only. Closed-loop continuous-reference error includes zero-order-hold sampling and integration, so no second-order closed-loop claim.",
        "structure_scope": "Differentiate fixed external u and full feedback u(x) separately; frozen-input symplecticity does not imply closed-loop symplecticity.",
        "failures": "Every failed solve is recorded and raised; no failed case is omitted from an accepted aggregate.",
        "inputs": inputs,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Protocol/input mismatch; preserve the existing experiment")
    else:
        save_json(path, protocol)
    return protocol, spec, initials


@njit(cache=False)
def remainder(theta):
    if abs(theta) >= .1:
        return math.sin(theta) - theta
    s = theta * theta
    return theta * s * (-1 / 6 + s * (1 / 120 + s * (-1 / 5040 + s * (1 / 362880 - s / 39916800))))


@njit(cache=False)
def energy(x, nu, d):
    theta = d @ x[:2]
    if abs(theta) < .1:
        s = theta * theta
        pot = s * s * (-1 / 24 + s * (1 / 720 + s * (-1 / 40320 + s * (1 / 3628800 - s / 479001600))))
    else:
        pot = 2 * math.sin(theta / 2) ** 2 - theta * theta / 2
    return .5 * np.sum(nu * (x[:2] ** 2 + x[2:] ** 2)) + pot


@njit(cache=False)
def port(x, nu, d):
    angle = d @ x[:2]
    return math.sin(angle), math.cos(angle) * (d @ (nu * x[2:]))


@njit(cache=False)
def feedforward(tau, omega0):
    return UAMP * (1 - math.exp(-(tau / (.25 * omega0)) ** 2)) * math.sin(.85 * tau)


@njit(cache=False)
def field(x, u, nu, d):
    theta = d @ x[:2]
    return np.concatenate((nu * x[2:], -nu * x[:2] - d * remainder(theta) + u * d * math.cos(theta)))


@njit(cache=False)
def midpoint(x, h, u, nu, d):
    a = .5 * h * nu
    inverse = 1 / (1 + a * a)
    qbase = inverse * ((1 - a * a) * x[:2] + 2 * a * x[2:])
    pbase = inverse * (-2 * a * x[:2] + (1 - a * a) * x[2:])
    theta = d @ ((x[:2] + qbase) / 2)
    response = .5 * h * (d @ (inverse * a * d))
    rhsangle = theta
    for iteration in range(20):
        nonlinear = -remainder(theta) + u * math.cos(theta)
        derivative = 1 - response * (1 - math.cos(theta) - u * math.sin(theta))
        update = (theta - rhsangle - response * nonlinear) / derivative
        theta -= update
        if abs(update) < 2e-15:
            break
    else:
        raise RuntimeError("Midpoint scalar Newton failed")
    forcing = d * (-remainder(theta) + u * math.cos(theta))
    new = np.concatenate((qbase + h * inverse * a * forcing, pbase + h * inverse * forcing))
    mid = (x + new) / 2
    residual = np.max(np.abs(new - x - h * field(mid, u, nu, d)))
    if residual > 3e-13 or not np.all(np.isfinite(new)):
        raise RuntimeError("Midpoint equation residual is invalid")
    work = h * u * port(mid, nu, d)[1]
    return new, residual, work, iteration + 1


@njit(cache=False)
def kick(x, dt, u, nu, d):
    new = x.copy()
    new[2:] += dt * u * d * math.cos(d @ x[:2])
    # Exact energy increment on a potential-kick subflow (positions fixed).
    work = .5 * np.sum(nu * (new[2:] ** 2 - x[2:] ** 2))
    return new, work


@njit(cache=False)
def step(x, h, u, method, nu, d):
    if method == 0:
        return midpoint(x, h, u, nu, d)
    if method == 1:
        free, residual, _, iterations = midpoint(x, h, 0., nu, d)
        new, work = kick(free, h, u, nu, d)
        return new, residual, work, iterations
    first, work1 = kick(x, h / 2, u, nu, d)
    free, residual, _, iterations = midpoint(first, h, 0., nu, d)
    new, work2 = kick(free, h / 2, u, nu, d)
    return new, residual, work1 + work2, iterations


@njit(cache=False)
def rollout(initials, h, steps, stride, method, mode, reference_internal, nu, d, omega0):
    # mode: 0 prescribed smooth open-loop, 1 sampled feedback, 2 no control.
    count = len(initials)
    states = np.empty((steps // stride + 1, count, 4))
    controls = np.empty((steps, count))
    diagnostics = np.zeros((count, 7))
    for trajectory in range(count):
        x = initials[trajectory].copy()
        states[0, trajectory] = x
        base_energy = energy(x, nu, d)
        work_sum, scale_work, balance, maximum_residual, max_iterations = 0., 0., 0., 0., 0
        subflow_work_sum, subflow_balance = 0., 0.
        for k in range(steps):
            tau = k * h
            if mode == 0:
                u = feedforward(tau + h / 2, omega0)
            elif mode == 1:
                value, velocity = port(x, nu, d)
                desired, desired_velocity = port(reference_internal[k], nu, d)
                command = feedforward(tau, omega0) - KP * (value - desired) - KD * (velocity - desired_velocity)
                u = min(UMAX, max(-UMAX, command))
            else:
                u = 0.
            controls[k, trajectory] = u
            new, residual, work, iterations = step(x, h, u, method, nu, d)
            if not np.all(np.isfinite(new)) or np.max(np.abs(new)) > 5:
                raise RuntimeError("Controlled state became nonfinite or exceeded diagnostic bound")
            maximum_residual = max(maximum_residual, residual)
            max_iterations = max(max_iterations, iterations)
            subflow_work_sum += work
            subflow_balance = max(subflow_balance, abs(energy(new, nu, d) - base_energy - subflow_work_sum))
            physical_work = u * (math.sin(d @ new[:2]) - math.sin(d @ x[:2]))
            work_sum += physical_work
            scale_work += abs(physical_work)
            balance = max(balance, abs(energy(new, nu, d) - base_energy - work_sum))
            x = new
            if (k + 1) % stride == 0:
                states[(k + 1) // stride, trajectory] = x
        diagnostics[trajectory] = np.array([maximum_residual, max_iterations, balance,
                                            balance / max(abs(base_energy) + scale_work, 1e-12), work_sum, subflow_balance, subflow_work_sum])
    return states, controls, diagnostics


def rhs_many(tau, flat, nu, d, omega0, mode, reference_solution):
    x = flat.reshape(-1, 4)
    theta = x[:, :2] @ d
    # Direct double-precision trigonometric evaluation provides an independent reference.
    forces = -(np.sin(theta) - theta)[:, None] * d
    if mode == 2:
        u = np.zeros(len(x))
    elif mode == 0:
        u = np.full(len(x), feedforward(tau, omega0))
    else:
        reference = reference_solution(tau)
        desired = math.sin(d @ reference[:2])
        desired_velocity = math.cos(d @ reference[:2]) * (d @ (nu * reference[2:]))
        velocity = np.cos(theta) * ((x[:, 2:] * nu) @ d)
        u = np.clip(feedforward(tau, omega0) - KP * (np.sin(theta) - desired) - KD * (velocity - desired_velocity), -UMAX, UMAX)
    forces += (u * np.cos(theta))[:, None] * d
    return np.concatenate((x[:, 2:] * nu, -x[:, :2] * nu + forces), axis=1).ravel()


def continuous(initials, times, nu, d, omega0, mode, reference_solution, tight=False):
    solution = solve_ivp(rhs_many, (0, times[-1]), initials.ravel(), method="DOP853",
                         args=(nu, d, omega0, mode, reference_solution), dense_output=True,
                         rtol=4e-14 if tight else RTOL, atol=2e-16 if tight else ATOL)
    if not solution.success:
        raise RuntimeError(solution.message)
    states = solution.sol(times).T.reshape(len(times), len(initials), 4)
    return solution, states


def trajectory_metrics(predicted, target, continuous_flow, controls, h, common_tau, spec):
    tip = np.array(spec["tip_map_m"])
    d = np.array(spec["slope_map_d"])
    theta = predicted[..., :2] @ d
    target_port = np.sin(target[..., :2] @ d)
    tracking_tip = (predicted[..., :2] - target[:, None, :2]) @ tip * 1e3
    flow_tip = (predicted - continuous_flow)[..., :2] @ tip * 1e3
    relative = np.linalg.norm(predicted - continuous_flow, axis=(0, 2)) / np.maximum(np.linalg.norm(continuous_flow, axis=(0, 2)), 1e-15)
    offset = len(common_tau) // 2
    cols = {
        "tracking_tip_rmse_mm": np.sqrt(np.mean(tracking_tip ** 2, axis=0)),
        "tail_half_tracking_tip_rmse_mm": np.sqrt(np.mean(tracking_tip[offset:] ** 2, axis=0)),
        "tracking_state_relative_l2": np.linalg.norm(predicted - target[:, None, :], axis=(0, 2)) / max(np.linalg.norm(target), 1e-15),
        "tracking_port_rmse": np.sqrt(np.mean((np.sin(theta) - target_port[:, None]) ** 2, axis=0)),
        "continuous_controlled_flow_tip_rmse_mm": np.sqrt(np.mean(flow_tip ** 2, axis=0)),
        "continuous_controlled_flow_state_relative_l2": relative,
        "integral_u_squared_seconds": np.sum(controls ** 2, axis=0) * h / spec["omega0_rad_s"],
        "max_abs_u": np.max(np.abs(controls), axis=0),
        "saturation_fraction": np.mean(np.abs(controls) >= UMAX - 1e-14, axis=0),
        "max_abs_angle_rad": np.max(np.abs(theta), axis=0),
        "max_modal_reference_radius": np.max(np.sqrt(predicted[..., :2] ** 2 + predicted[..., 2:] ** 2), axis=(0, 2)),
    }
    return [{key: float(value[j]) for key, value in cols.items()} for j in range(predicted.shape[1])]


def fixed_jacobian(x, h, u, method, nu, d):
    def mp(z, forced):
        new = midpoint(z, h, forced, nu, d)[0]
        mid = (z + new) / 2
        theta = d @ mid[:2]
        hessian = np.diag(np.r_[nu, nu])
        hessian[:2, :2] += (math.cos(theta) - 1 + forced * math.sin(theta)) * np.outer(d, d)
        a = J @ hessian
        inv = np.linalg.inv(np.eye(4) - h * a / 2)
        state_jac = inv @ (np.eye(4) + h * a / 2)
        input_jac = inv @ (h * np.r_[np.zeros(2), d * math.cos(theta)]) if forced == u else np.zeros(4)
        return new, state_jac, input_jac
    def kk(z, length):
        new = kick(z, length, u, nu, d)[0]
        a = np.eye(4)
        a[2:, :2] -= length * u * math.sin(d @ z[:2]) * np.outer(d, d)
        bu = np.r_[np.zeros(2), length * d * math.cos(d @ z[:2])]
        return new, a, bu
    if method == 0:
        return mp(x, u)
    if method == 1:
        free, a, _ = mp(x, 0.)
        new, k, bu = kk(free, h)
        return new, k @ a, bu
    first, k1, b1 = kk(x, h / 2)
    free, a, _ = mp(first, 0.)
    new, k2, b2 = kk(free, h / 2)
    return new, k2 @ a @ k1, k2 @ a @ b1 + b2


def feedback_value_gradient(x, tau, reference, nu, d, omega0):
    value, velocity = port(x, nu, d)
    desired, desired_velocity = port(reference, nu, d)
    command = feedforward(tau, omega0) - KP * (value - desired) - KD * (velocity - desired_velocity)
    if abs(command) >= UMAX:
        return float(np.clip(command, -UMAX, UMAX)), np.zeros(4), True
    theta = d @ x[:2]
    speed = d @ (nu * x[2:])
    dy = np.r_[math.cos(theta) * d, np.zeros(2)]
    dv = np.r_[-math.sin(theta) * speed * d, math.cos(theta) * nu * d]
    return command, -KP * dy - KD * dv, False


def verify_implementation(spec, initials, h0):
    nu, d, omega0 = np.array(spec["nu"]), np.array(spec["slope_map_d"]), spec["omega0_rad_s"]
    records, structure = [], []
    rng = np.random.default_rng(61231)
    for h in [h0, 4 * h0, .1]:
        for x in initials[:8]:
            u = float(rng.uniform(-UMAX, UMAX))
            candidate, residual, _, _ = midpoint(x, h, u, nu, d)
            independent = root(lambda y: y - x - h * field((x + y) / 2, u, nu, d), candidate, tol=1e-11)
            gap = float(np.max(np.abs(independent.x - candidate)))
            direct_residual = float(np.max(np.abs(independent.fun)))
            assert gap < 2e-12 and direct_residual < 2e-13
            records.append({"h": h, "u": u, "independent_state_gap": gap, "midpoint_residual": residual, "independent_residual": direct_residual})
    # Numerical gradient of the given potential, exact type-2 kick, and the
    # forced-midpoint algebra are checked before long rollouts.
    fixed_errors, feedback_errors = [], []
    for h in [4 * h0, h0]:
        for index, x in enumerate(initials[:8]):
            for method, name in enumerate(METHODS):
                u, du, saturated = feedback_value_gradient(x, 0., np.zeros(4), nu, d, omega0)
                new, a, bu = fixed_jacobian(x, h, u, method, nu, d)
                full = a + np.outer(bu, du)
                finite_fixed = np.empty((4, 4)); finite_feedback = np.empty((4, 4))
                for j in range(4):
                    delta = np.eye(4)[j] * 1e-6
                    finite_fixed[:, j] = (step(x + delta, h, u, method, nu, d)[0] - step(x - delta, h, u, method, nu, d)[0]) / 2e-6
                    up = feedback_value_gradient(x + delta, 0., np.zeros(4), nu, d, omega0)[0]
                    um = feedback_value_gradient(x - delta, 0., np.zeros(4), nu, d, omega0)[0]
                    finite_feedback[:, j] = (step(x + delta, h, up, method, nu, d)[0] - step(x - delta, h, um, method, nu, d)[0]) / 2e-6
                fg, cg = float(np.max(np.abs(finite_fixed - a))), float(np.max(np.abs(finite_feedback - full)))
                assert fg < 2e-8 and cg < 2e-8
                fixed_errors.append(fg); feedback_errors.append(cg)
                structure.append({"h": h, "method": name, "initial_index": index, "u": u, "saturated": saturated,
                                  "fixed_u_symplectic_defect_F": float(np.linalg.norm(a.T @ J @ a - J)),
                                  "feedback_symplectic_defect_F": float(np.linalg.norm(full.T @ J @ full - J)),
                                  "fixed_u_det": float(np.linalg.det(a)), "feedback_det": float(np.linalg.det(full)),
                                  "fixed_u_jacobian_finite_difference_gap": fg, "feedback_jacobian_finite_difference_gap": cg,
                                  "zero_input_scheme_state_gap": float(np.max(np.abs(step(x, h, 0., method, nu, d)[0] - midpoint(x, h, 0., nu, d)[0])))})
    write_csv(OUT / "structure_diagnostics.csv", structure)
    report = {"independent_midpoint_checks": records, "maximum_fixed_u_jacobian_gap": max(fixed_errors),
              "maximum_feedback_jacobian_gap": max(feedback_errors),
              "fixed_u_max_symplectic_defect": max(x["fixed_u_symplectic_defect_F"] for x in structure),
              "feedback_min_symplectic_defect": min(x["feedback_symplectic_defect_F"] for x in structure),
              "feedback_max_symplectic_defect": max(x["feedback_symplectic_defect_F"] for x in structure),
              "input_potential_gradient": "Analytic gradient -u*d*cos(d.X); exact kick follows type-2 generating function X.P-s*u*sin(d.X).",
              "scope": "Frozen external input and full start-sampled feedback are explicitly distinguished."}
    save_json(OUT / "implementation_verification.json", report)
    return report


def run():
    protocol, spec, initials = plan()
    if (OUT / "completed.json").exists():
        raise RuntimeError("A completed control experiment exists; use --stage assets to retain it")
    nu, d, omega0 = np.array(spec["nu"]), np.array(spec["slope_map_d"]), spec["omega0_rad_s"]
    h0, nbase = protocol["base_h_tau"], protocol["base_steps"]
    tau = np.arange(nbase // 8 + 1) * 8 * h0
    save_json(OUT / "execution_environment.json", {"script_sha256": sha(__file__), "python": sys.version,
              "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
              "interpretation": "Control gains, input bound, initial blocks and all four steps fixed before the accepted run."})
    (OUT / "execution_script.py").write_bytes(Path(__file__).read_bytes())
    verification = verify_implementation(spec, initials, h0)
    start = time.perf_counter()
    desired_solution, desired = continuous(np.zeros((1, 4)), tau, nu, d, omega0, 0, None)
    reference = desired[:, 0]
    ref_function = lambda t: desired_solution.sol(t).ravel()
    open_solution, open_reference = continuous(initials, tau, nu, d, omega0, 0, None)
    continuous_solution, closed_reference = continuous(initials, tau, nu, d, omega0, 1, ref_function)
    _, uncontrolled_reference = continuous(initials, tau, nu, d, omega0, 2, None)
    # Independently tighten all 40 trajectories, not a selected favorable case.
    _, open_tight = continuous(initials, tau, nu, d, omega0, 0, None, tight=True)
    _, closed_tight = continuous(initials, tau, nu, d, omega0, 1, ref_function, tight=True)
    tight_gap_open = float(np.max(np.abs(open_tight - open_reference)))
    tight_gap_closed = float(np.max(np.abs(closed_tight - closed_reference)))
    assert max(tight_gap_open, tight_gap_closed) < 2e-9
    np.savez_compressed(OUT / "common_reference.npz", tau=tau, time_seconds=tau / omega0, initials=initials,
                        target_states=reference, open_loop_reference=open_reference,
                        continuous_feedback_reference=closed_reference, uncontrolled_reference=uncontrolled_reference,
                        nu=nu, d=d, tip_map_m=np.array(spec["tip_map_m"]))
    rows = []
    for multiplier in MULTIPLIERS:
        h = h0 * multiplier
        steps, stride = int(round(nbase / multiplier)), int(round(8 / multiplier))
        assert steps % stride == 0 and np.max(np.abs(np.arange(steps // stride + 1) * stride * h - tau)) < 2e-13
        internal_times = np.arange(steps + 1) * h
        desired_internal = desired_solution.sol(internal_times).T
        for mode, mode_name in [(0, "open_loop"), (1, "feedback"), (2, "uncontrolled")]:
            for method, name in enumerate(METHODS if mode != 2 else ["uncontrolled"]):
                try:
                    states, controls, diagnostics = rollout(initials, h, steps, stride, method, mode, desired_internal, nu, d, omega0)
                except Exception as error:
                    save_json(OUT / "failed_case.json", {"multiplier": multiplier, "mode": mode_name, "method": name, "error": str(error)})
                    raise
                flow = open_reference if mode == 0 else closed_reference if mode == 1 else uncontrolled_reference
                metrics = trajectory_metrics(states, reference, flow, controls, h, tau, spec)
                path = OUT / f"{mode_name}_{name}_m{multiplier:g}.npz"
                np.savez_compressed(path, states=states, controls=controls, diagnostics=diagnostics,
                                    tau=tau, h_tau=h, internal_times_tau=internal_times[:-1])
                for j, values in enumerate(metrics):
                    rows.append({"mode": mode_name, "method": name, "multiplier": multiplier, "h_tau": h,
                                 "seed": SEEDS[j // 8], "trajectory": j % 8,
                                 **values, "maximum_midpoint_residual": float(diagnostics[j, 0]),
                                 "maximum_newton_iterations": int(diagnostics[j, 1]),
                                 "discrete_work_balance_absolute": float(diagnostics[j, 2]),
                                 "discrete_work_balance_relative": float(diagnostics[j, 3]),
                                 "signed_input_work": float(diagnostics[j, 4]),
                                 "subflow_quadrature_work_balance_absolute": float(diagnostics[j, 5]),
                                 "subflow_quadrature_work": float(diagnostics[j, 6])})
                print("CONTROL_READY", mode_name, name, multiplier, "tracking_mm", np.mean([m["tracking_tip_rmse_mm"] for m in metrics]), flush=True)
    write_csv(OUT / "trajectory_metrics.csv", rows)
    save_json(OUT / "reference_verification.json", {"all_40_open_tolerance_state_gap": tight_gap_open,
              "all_40_continuous_feedback_tolerance_state_gap": tight_gap_closed,
              "reference_calls": {"target": desired_solution.nfev, "open_loop": open_solution.nfev, "continuous_feedback": continuous_solution.nfev},
              "grid_max_tau_gap": 0., "seconds_for_references_and_rollouts": time.perf_counter() - start,
              "closed_loop_comparison_scope": "Errors to continuous feedback combine sample-and-hold and integrator effects; open-loop data determine discretization orders."})
    assert all(sha(ROOT / item["path"]) == item["sha256"] for item in protocol["inputs"])
    manifest = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in sorted(OUT.glob("*.npz"))]
    save_json(OUT / "completed.json", {"status": "complete", "trajectory_records": len(rows), "failures": [],
              "input_hashes_unchanged": True, "numeric_archives": manifest, "execution_script_sha256": sha(OUT / "execution_script.py")})
    return summarize()


def summarize():
    with (OUT / "trajectory_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    numbers = [k for k in rows[0] if k not in ["mode", "method", "multiplier", "h_tau", "seed", "trajectory"]]
    seed_rows, summary_rows = [], []
    for mode in ["open_loop", "feedback", "uncontrolled"]:
        for method in METHODS if mode != "uncontrolled" else ["uncontrolled"]:
            for multiplier in MULTIPLIERS:
                chosen = [r for r in rows if r["mode"] == mode and r["method"] == method and float(r["multiplier"]) == multiplier]
                assert len(chosen) == 40
                seeds = []
                for seed in SEEDS:
                    block = [r for r in chosen if int(r["seed"]) == seed]
                    assert len(block) == 8
                    item = {"mode": mode, "method": method, "multiplier": multiplier, "seed": seed,
                            **{key: float(np.mean([float(r[key]) for r in block])) for key in numbers}}
                    seeds.append(item); seed_rows.append(item)
                item = {"mode": mode, "method": method, "multiplier": multiplier,
                        "metrics": {key: {"mean": float(np.mean([s[key] for s in seeds])),
                                          "sample_sd": float(np.std([s[key] for s in seeds], ddof=1)),
                                          "maximum_over_trajectories": max(float(r[key]) for r in chosen)} for key in numbers}}
                summary_rows.append(item)
    orders = []
    for method in METHODS:
        case = [r for r in summary_rows if r["mode"] == "open_loop" and r["method"] == method]
        for coarse, fine in zip(case[:-1], case[1:]):
            orders.append({"method": method, "fine_multiplier": fine["multiplier"],
                           "tip_observed_order": math.log2(coarse["metrics"]["continuous_controlled_flow_tip_rmse_mm"]["mean"] / fine["metrics"]["continuous_controlled_flow_tip_rmse_mm"]["mean"]),
                           "state_observed_order": math.log2(coarse["metrics"]["continuous_controlled_flow_state_relative_l2"]["mean"] / fine["metrics"]["continuous_controlled_flow_state_relative_l2"]["mean"])})
    paired = []
    for multiplier in MULTIPLIERS:
        for metric in ["tracking_tip_rmse_mm", "tail_half_tracking_tip_rmse_mm", "integral_u_squared_seconds"]:
            lookup = {(r["method"], r["seed"]): r[metric] for r in seed_rows if r["mode"] in ["feedback", "uncontrolled"] and r["multiplier"] == multiplier}
            for a, b in [("uncontrolled", "I"), ("uncontrolled", "II"), ("uncontrolled", "III"), ("I", "II"), ("I", "III"), ("II", "III")]:
                differences = [lookup[a, seed] - lookup[b, seed] for seed in SEEDS]
                paired.append({"multiplier": multiplier, "metric": metric, "contrast": a + "-" + b,
                               "mean": float(np.mean(differences)), "sample_sd": float(np.std(differences, ddof=1)),
                               "positive_seed_count": int(np.sum(np.array(differences) > 0))})
    write_csv(OUT / "seed_metrics.csv", seed_rows)
    write_csv(OUT / "open_loop_orders.csv", orders)
    write_csv(OUT / "paired_summary.csv", paired)
    result = {"status": "complete", "cases": summary_rows, "open_loop_orders": orders, "paired_differences": paired,
              "scope": "Simulation-only scalar-input physical controller comparison; seed blocks are initial-condition repeats, no learned controller.",
              "script_sha256": sha(__file__)}
    save_json(OUT / "summary.json", result)
    return result


def assets(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup = {(r["mode"], r["method"], r["multiplier"]): r["metrics"] for r in summary["cases"]}
    orders = {(r["method"], r["fine_multiplier"]): r["state_observed_order"] for r in summary["open_loop_orders"]}
    def pm(obj, digits=4):
        return f"${obj['mean']:.{digits}f}\\pm{obj['sample_sd']:.{digits}f}$"
    lines = [r"\begin{table*}[t]", r"\centering\small",
             r"\caption{同一磁场预算下的控制插入比较。误差为五个初值种子块的均值与样本标准差；代表步长为$h=h_0$。}",
             r"\label{tab:control}", r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"方案 & 开环状态观察阶 & 闭环杆端RMSE (mm) & 后半时域RMSE (mm) & $\int u^2\,dt$ (s) & 最大$|B_y|$ (mT) \\", r"\midrule"]
    for method in ["uncontrolled"] + METHODS:
        mode = "uncontrolled" if method == "uncontrolled" else "feedback"
        m = lookup[mode, method, 1.]
        order = "---" if method == "uncontrolled" else f"{orders[method,.5]:.4f}"
        name = "无控制" if method == "uncontrolled" else method
        lines.append(" & ".join([name, order, pm(m["tracking_tip_rmse_mm"]), pm(m["tail_half_tracking_tip_rmse_mm"]),
                                 pm(m["integral_u_squared_seconds"], 6), f"{30*m['max_abs_u']['maximum_over_trajectories']:.4f}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip\begin{minipage}{0.97\textwidth}\footnotesize "
              r"I为步内受控中点，II为自由中点后完整势冲量，III为对称半冲量组合。闭环均使用同一标量端口反馈、增益及$|u|\leq0.2$，对应$|B_y|\leq6$ mT；各方案沿自身状态采样，实际输入序列不同。开环观察阶来自共同平滑输入及最细两档，闭环误差另含输入保持误差。种子仅划分初值，不代表独立控制器训练。\end{minipage}", r"\end{table*}"]
    (ROOT / "manuscript/tables/control.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 300})
    colors = {"I": "#0072B2", "II": "#D55E00", "III": "#009E73"}
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0), layout="constrained")
    for method in METHODS:
        values = [lookup["open_loop", method, m]["continuous_controlled_flow_state_relative_l2"]["mean"] for m in MULTIPLIERS]
        axes[0].loglog(np.array(MULTIPLIERS) * .01, values, "o-", color=colors[method], label=method)
        with np.load(OUT / f"feedback_{method}_m1.npz") as case, np.load(OUT / "common_reference.npz") as ref:
            error = (case["states"][..., :2] - ref["target_states"][:, None, :2]) @ ref["tip_map_m"] * 1e3
            axes[1].plot(ref["time_seconds"], np.sqrt(np.mean(error ** 2, axis=1)), color=colors[method], label=method)
        stats = [lookup["feedback", method, m]["tail_half_tracking_tip_rmse_mm"] for m in MULTIPLIERS]
        axes[2].errorbar(MULTIPLIERS, [s["mean"] for s in stats], yerr=[s["sample_sd"] for s in stats],
                         color=colors[method], marker="o", capsize=2, label=method)
    with np.load(OUT / "common_reference.npz") as ref:
        err = (ref["uncontrolled_reference"][..., :2] - ref["target_states"][:, None, :2]) @ ref["tip_map_m"] * 1e3
        axes[1].plot(ref["time_seconds"], np.sqrt(np.mean(err ** 2, axis=1)), "--", color="#555555", label="No control")
    axes[0].set(xlabel=r"Midpoint step $h_\tau$", ylabel="Open-loop state relative error", title="(a) Discretization accuracy")
    axes[1].set(xlabel="Time (s)", ylabel="Tip ensemble RMS (mm)", title="(b) Common reference tracking")
    axes[2].set(xlabel=r"Step relative to $h_0$", ylabel="Tail-half tip RMSE (mm)", title="(c) Closed-loop step sensitivity")
    axes[2].set_xscale("log", base=2); axes[2].set_xticks(MULTIPLIERS, ["4", "2", "1", "1/2"])
    for ax in axes:
        ax.grid(alpha=.18); ax.legend(frameon=False, fontsize=8)
    target = ROOT / "manuscript/figures/control_comparison"
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["plan", "verify", "run", "assets"], default="plan")
    args = parser.parse_args()
    protocol, spec, initials = plan()
    if args.stage == "verify":
        verify_implementation(spec, initials, protocol["base_h_tau"])
    elif args.stage == "run":
        assets(run())
    elif args.stage == "assets":
        assets(summarize())
    print("CONTROL_STAGE_COMPLETE", args.stage, flush=True)


if __name__ == "__main__":
    main()
