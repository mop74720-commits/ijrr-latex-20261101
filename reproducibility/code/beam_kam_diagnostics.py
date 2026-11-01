"""KAM-related diagnostics of the SAME conservative two-mode beam used in learning.

No new physical model, training, torus-existence theorem, or exact shadow
Hamiltonian is introduced.  I0=(X**2+Y**2)/2 labels the initial harmonic chart;
finite differences with respect to I0 are not certified action derivatives.
Run with the project's scientific Python environment. Outputs are confined to
results/beam_kam and manuscript/{tables,figures}/beam_kam.*.
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

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "beam_kam"
RAW = OUT / "raw"
SPEC = ROOT / "results" / "learning" / "inputs" / "reference_model_spec.json"
DATA = ROOT / "results" / "learning" / "data" / "seed_17.npz"
PHASES = np.array([0.37, 1.13])
RADII = np.array([[0.05, 0.05], [0.12, 0.10], [0.20, 0.16]])
ACTION_STEPS = np.array([0.0012, 0.0010])
RTOL, ATOL = 1e-12, 1e-14
TIGHT_RTOL, TIGHT_ATOL = 4e-14, 2e-16
SAMPLE_STRIDE = 4


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def initial_state(actions):
    radii = np.sqrt(2 * np.asarray(actions))
    return np.r_[radii * np.cos(PHASES), -radii * np.sin(PHASES)]


def force_remainder(theta):
    """sin(theta)-theta, using the SAME analytic function without cancellation."""
    theta = np.asarray(theta)
    # Here |theta|<.1; a stable evaluation preserves the explicit cosine model.
    polynomial = theta**3 * (-1 / 6 + theta**2 * (1 / 120 + theta**2 *
                 (-1 / 5040 + theta**2 * (1 / 362880 - theta**2 / 39916800))))
    return np.where(np.abs(theta) < .1, polynomial, np.sin(theta) - theta)


def potential_remainder(theta):
    theta = np.asarray(theta)
    polynomial = theta**4 * (-1 / 24 + theta**2 * (1 / 720 + theta**2 *
                 (-1 / 40320 + theta**2 * (1 / 3628800 - theta**2 / 479001600))))
    return np.where(np.abs(theta) < .1, polynomial,
                    2 * np.sin(theta / 2)**2 - theta**2 / 2)


def energy(states, nu, d):
    states = np.asarray(states)
    return (.5 * np.sum(nu * (states[..., :2]**2 + states[..., 2:]**2), axis=-1)
            + potential_remainder(states[..., :2] @ d))


def rhs(tau, state, nu, d):
    q, p = state[:2], state[2:]
    theta = d @ q
    return np.r_[nu * p, -nu * q - d * force_remainder(theta)]


def quartic_diagnostics(nu, d):
    hessian = -np.array([[d[0]**4 / 8, d[0]**2 * d[1]**2 / 4],
                        [d[0]**2 * d[1]**2 / 4, d[1]**4 / 8]])
    divisors = []
    for k1 in range(-32, 33):
        for k2 in range(-32, 33):
            norm = abs(k1) + abs(k2)
            if norm == 0 or norm > 32:
                continue
            quartic_support = norm <= 4 and (k1 + k2) % 2 == 0
            divisors.append({"k1": k1, "k2": k2, "l1_norm": norm,
                             "quartic_support": quartic_support,
                             "abs_k_dot_nu": abs(k1 * nu[0] + k2 * nu[1])})
    fourth = [r for r in divisors if r["quartic_support"]]
    result = {
        "reference": "I0_j=(X_j^2+Y_j^2)/2 is the harmonic reference action; not an exact action of h_*.",
        "quartic_angle_average": "-(d1^4*I1^2+4*d1^2*d2^2*I1*I2+d2^4*I2^2)/16",
        "normal_form_scope": "Leading quartic angle average in a low-order nonresonant Birkhoff normal-form expansion; not a new theorem or global integrability claim.",
        "quartic_action_hessian": hessian.tolist(),
        "quartic_action_hessian_determinant": float(np.linalg.det(hessian)),
        "quartic_action_hessian_determinant_formula": "-3*d1^4*d2^4/64",
        "quartic_action_hessian_singular_values": np.linalg.svd(hessian, compute_uv=False).tolist(),
        "quartic_active_divisor_min": min(fourth, key=lambda row: row["abs_k_dot_nu"]),
        "generic_l1_le4_divisor_min": min((r for r in divisors if r["l1_norm"] <= 4), key=lambda row: row["abs_k_dot_nu"]),
        "generic_l1_le32_divisor_min": min(divisors, key=lambda row: row["abs_k_dot_nu"]),
    }
    return result, divisors, hessian


def phase_frequency(states, sample_step):
    """Smooth weighted mean of modal phase increments, not an FFT-bin estimate."""
    z = states[:, :2] - 1j * states[:, 2:]
    increments = np.angle(np.conjugate(z[:-1]) * z[1:])
    if np.max(np.abs(increments)) >= .9 * np.pi:
        raise RuntimeError("Sampling too coarse for unambiguous modal phase increments")
    count = len(increments)
    u = (np.arange(count, dtype=float) + .5) / count
    weights = np.exp(-1 / (u * (1 - u)))
    weights /= weights.sum()
    return np.asarray(np.sum(weights[:, None] * increments, axis=0,
                             dtype=np.longdouble) / sample_step, dtype=float)


def frequency_diagnostics(states, sample_step):
    last = len(states) - 1
    middle = last // 2
    quarter = last // 4
    windows = {"full": (0, last), "first_half": (0, middle),
               "second_half": (middle, last),
               "central_half": (quarter, last - quarter)}
    frequencies = {name: phase_frequency(states[a:b + 1], sample_step)
                   for name, (a, b) in windows.items()}
    full = frequencies["full"]
    spread = np.max(np.abs(np.array(list(frequencies.values())) - full), axis=0)
    coarse = {str(stride): phase_frequency(states[::stride], sample_step * stride)
              for stride in [2, 4]}
    sample_change = np.max(np.abs(np.array(list(coarse.values())) - full), axis=0)
    radii = np.sqrt(states[:, :2]**2 + states[:, 2:]**2)
    return {
        "frequency_tau": full.tolist(),
        "window_frequencies_tau": {name: x.tolist() for name, x in frequencies.items()},
        "max_window_change_tau": spread.tolist(),
        "downsampled_frequencies_tau": {name: x.tolist() for name, x in coarse.items()},
        "max_downsampling_change_tau": sample_change.tolist(),
        "min_modal_phase_radius": radii.min(axis=0).tolist(),
        "max_modal_phase_radius": radii.max(axis=0).tolist(),
    }


def continuous_case(tag, actions, tau, sample_step, nu, d, rtol=RTOL, atol=ATOL):
    path = RAW / f"continuous_{tag}.npz"
    y0 = initial_state(actions)
    cached = False
    if path.exists():
        with np.load(path) as saved:
            matches = (saved["states"].shape == (len(tau), 4)
                       and np.array_equal(saved["tau"], tau)
                       and np.array_equal(saved["initial_state"], y0)
                       and np.array_equal(saved["nu"], nu)
                       and np.array_equal(saved["d"], d)
                       and float(saved["rtol"]) == rtol
                       and float(saved["atol"]) == atol)
            if matches:
                states = saved["states"]
                nfev = int(saved["nfev"])
                elapsed = float(saved["integration_seconds"])
                cached = True
    if not cached:
        start = time.perf_counter()
        solution = solve_ivp(rhs, (0, tau[-1]), y0, args=(nu, d), method="DOP853",
                             t_eval=tau, rtol=rtol, atol=atol)
        elapsed = time.perf_counter() - start
        if not solution.success or solution.y.shape[1] != len(tau):
            raise RuntimeError(f"DOP853 failed for {tag}: {solution.message}")
        states, nfev = solution.y.T, solution.nfev
        np.savez_compressed(path, tau=tau, states=states, initial_state=y0,
                            actions0=actions, nu=nu, d=d, rtol=rtol, atol=atol,
                            nfev=nfev, integration_seconds=elapsed)
    diag = frequency_diagnostics(states, sample_step)
    e = energy(states, nu, d)
    diag.update({"tag": tag, "actions0": np.asarray(actions).tolist(),
                 "initial_radii": np.sqrt(2 * np.asarray(actions)).tolist(),
                 "initial_phases": PHASES.tolist(), "rtol": rtol, "atol": atol,
                 "max_relative_energy_drift": float(np.max(np.abs(e - e[0])) / abs(e[0])),
                 "max_abs_tip_angle_rad": float(np.max(np.abs(states[:, :2] @ d))),
                 "nfev": nfev, "integration_seconds": elapsed,
                 "raw_file": path.relative_to(ROOT).as_posix(),
                 "raw_sha256": sha256(path)})
    print(f"CONTINUOUS {tag}: omega={diag['frequency_tau']} energy={diag['max_relative_energy_drift']:.3g} seconds={elapsed:.2f} cached={cached}", flush=True)
    return diag, states


@njit(cache=False)
def sine_remainder_scalar(theta):
    if abs(theta) >= .1:
        return math.sin(theta) - theta
    theta2 = theta * theta
    return theta * theta2 * (-1 / 6 + theta2 * (1 / 120 + theta2 *
           (-1 / 5040 + theta2 * (1 / 362880 - theta2 / 39916800))))


@njit(cache=False)
def midpoint_trajectory(y0, nu, d, h, nsteps, stride):
    """Exact algebraic reduction of the beam's implicit midpoint to one scalar."""
    states = np.empty((nsteps // stride + 1, 4))
    states[0] = y0
    q0, q1, p0, p1 = y0[0], y0[1], y0[2], y0[3]
    den0 = 1 + .25 * h * h * nu[0]**2
    den1 = 1 + .25 * h * h * nu[1]**2
    b0 = .25 * h * h * nu[0] * d[0] / den0
    b1 = .25 * h * h * nu[1] * d[1] / den1
    beta = d[0] * b0 + d[1] * b1
    max_residual = 0.
    for step in range(nsteps):
        a0 = (q0 + .5 * h * nu[0] * p0) / den0
        a1 = (q1 + .5 * h * nu[1] * p1) / den1
        eta = d[0] * a0 + d[1] * a1
        theta = eta
        for iteration in range(8):
            g = sine_remainder_scalar(theta)
            correction = (theta + beta * g - eta) / (1 + beta * (math.cos(theta) - 1))
            theta -= correction
            if abs(correction) < 2e-16:
                break
        g = sine_remainder_scalar(theta)
        qm0, qm1 = a0 - b0 * g, a1 - b1 * g
        pm0 = p0 - .5 * h * (nu[0] * qm0 + d[0] * g)
        pm1 = p1 - .5 * h * (nu[1] * qm1 + d[1] * g)
        nq0 = q0 + h * nu[0] * pm0
        nq1 = q1 + h * nu[1] * pm1
        np0 = p0 - h * (nu[0] * qm0 + d[0] * g)
        np1 = p1 - h * (nu[1] * qm1 + d[1] * g)
        tm = d[0] * (nq0 + q0) / 2 + d[1] * (nq1 + q1) / 2
        gm = sine_remainder_scalar(tm)
        r0 = nq0 - q0 - h * nu[0] * (np0 + p0) / 2
        r1 = nq1 - q1 - h * nu[1] * (np1 + p1) / 2
        r2 = np0 - p0 + h * (nu[0] * (nq0 + q0) / 2 + d[0] * gm)
        r3 = np1 - p1 + h * (nu[1] * (nq1 + q1) / 2 + d[1] * gm)
        max_residual = max(max_residual, abs(r0), abs(r1), abs(r2), abs(r3))
        q0, q1, p0, p1 = nq0, nq1, np0, np1
        if (step + 1) % stride == 0:
            j = (step + 1) // stride
            states[j, 0], states[j, 1] = q0, q1
            states[j, 2], states[j, 3] = p0, p1
    return states, max_residual


def verify_midpoint_reduction(nu, d):
    """Compare one-step scalar reduction with an independent 4D root solve."""
    from scipy.optimize import root
    records = []
    for h in [.01, .05, .1]:
        for radii in RADII:
            old = initial_state(radii**2 / 2)
            actual, _ = midpoint_trajectory(old, nu, d, h, 1, 1)
            equation = lambda new: new - old - h * rhs(0, (new + old) / 2, nu, d)
            solved = root(equation, old + h * rhs(0, old, nu, d), tol=1e-11)
            residual = float(np.max(np.abs(equation(solved.x))))
            difference = float(np.max(np.abs(actual[-1] - solved.x)))
            if residual > 2e-13 or difference > 2e-12:
                raise RuntimeError("Scalar midpoint reduction disagrees with independent root solve")
            records.append({"h": h, "radii": radii.tolist(),
                            "independent_root_residual": residual,
                            "state_max_abs_difference": difference})
    # The stable force and potential evaluations differ from the source's direct
    # trigonometric expressions only at floating-point cancellation level here.
    theta = np.linspace(-.1, .1, 1001)
    return {"midpoint_independent_4d_checks": records,
            "stable_force_vs_direct_max_abs": float(np.max(np.abs(force_remainder(theta) - (np.sin(theta) - theta)))),
            "stable_potential_vs_direct_max_abs": float(np.max(np.abs(potential_remainder(theta) - (2 * np.sin(theta / 2)**2 - theta**2 / 2)))),
            "stable_force_series_remainder_bound_abs_theta_le_0p1": .1**13 / math.factorial(13),
            "stable_potential_series_remainder_bound_abs_theta_le_0p1": .1**14 / math.factorial(14)}


def make_outputs(summary, nu, d, hessian, tau, states_base, step_cases):
    rows = []
    for diag in summary["base_cases"]:
        omega = np.array(diag["frequency_tau"])
        rows.append({"r1": diag["initial_radii"][0], "r2": diag["initial_radii"][1],
                     "I1_reference0": diag["actions0"][0], "I2_reference0": diag["actions0"][1],
                     "omega1_tau": omega[0], "omega2_tau": omega[1],
                     "omega1_minus_nu1_tau": omega[0] - nu[0], "omega2_minus_nu2_tau": omega[1] - nu[1],
                     "normal_form_shift1": float((hessian @ diag["actions0"])[0]),
                     "normal_form_shift2": float((hessian @ diag["actions0"])[1]),
                     "max_window_change": max(diag["max_window_change_tau"]),
                     "max_sampling_change": max(diag["max_downsampling_change_tau"]),
                     "max_relative_energy_drift": diag["max_relative_energy_drift"]})
    write_csv(OUT / "frequency_summary.csv", rows)
    step_rows = []
    ref = np.array(summary["base_cases"][1]["frequency_tau"])
    for item in step_cases:
        omega = np.array(item["frequency_tau"])
        step_rows.append({"h_tau": item["h_tau"], "omega1_tau": omega[0], "omega2_tau": omega[1],
                          "abs_frequency_error1_tau": abs(omega[0] - ref[0]),
                          "abs_frequency_error2_tau": abs(omega[1] - ref[1]),
                          "linear_midpoint_error1_tau": abs(2 / item["h_tau"] * np.arctan(item["h_tau"] * nu[0] / 2) - nu[0]),
                          "linear_midpoint_error2_tau": abs(2 / item["h_tau"] * np.arctan(item["h_tau"] * nu[1] / 2) - nu[1]),
                          "max_midpoint_residual": item["max_midpoint_residual"],
                          "max_relative_energy_drift": item["max_relative_energy_drift"]})
    write_csv(OUT / "step_frequency_summary.csv", step_rows)

    table = [r"\begin{table*}[t]", r"\centering\small",
             r"\caption{与学习对象相同的两模态梁：连续相位主频及其幅值依赖。}",
             r"\label{table:beam_kam}", r"\begin{tabular}{rrrr}", r"\toprule",
             r"$(r_1,r_2)$ & $(I_1^0,I_2^0)$ & $10^6(\omega_1-\nu_1)$ & $10^6(\omega_2-\nu_2)$ \\",
             r"\midrule"]
    for row in rows:
        table.append(f"$({row['r1']:.2f},{row['r2']:.2f})$ & $({row['I1_reference0']:.5f},{row['I2_reference0']:.5f})$ & "
                     f"{row['omega1_minus_nu1_tau'] * 1e6:.4f} & {row['omega2_minus_nu2_tau'] * 1e6:.4f} " + r"\\")
    table += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              r"\begin{minipage}{0.96\textwidth}\footnotesize",
              r"所有频率均对无量纲时间$\tau$计；物理角频率为$\omega_0\omega_j$。$I_j^0=r_j^2/2$仅标记谐振子参考初值，并非非线性系统的精确作用量。容差、采样及分窗检验采用$10^{-10}$的经验分辨阈值，该阈值不是严格误差界。",
              r"\end{minipage}", r"\end{table*}"]
    (ROOT / "manuscript" / "tables" / "beam_kam.tex").write_text("\n".join(table) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.7,
                         "axes.titlesize": 8.4, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "lines.linewidth": 1.2})
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.35), layout="constrained")
    colors = ["#0072B2", "#D55E00"]
    window = np.hanning(len(states_base))
    frequencies = 2 * np.pi * np.fft.rfftfreq(len(states_base), tau[1] - tau[0])
    for j, color in enumerate(colors):
        spectrum = abs(np.fft.rfft(window * states_base[:, j]))
        spectrum /= spectrum.max()
        selected = (frequencies > 0) & (frequencies < 7.3)
        axes[0].semilogy(frequencies[selected], np.maximum(spectrum[selected], 1e-10), color=color, label=f"$X_{j+1}$")
    axes[0].set(xlabel=r"Angular frequency in $\tau$", ylabel="Normalized FFT amplitude", ylim=(1e-8, 2), title="(a) Two modal spectral peaks")
    axes[0].legend(frameon=False)
    x = np.array([sum(row["actions0"]) for row in summary["base_cases"]])
    actual_shift = np.array([np.array(row["frequency_tau"]) - nu for row in summary["base_cases"]])
    predicted_shift = np.array([hessian @ row["actions0"] for row in summary["base_cases"]])
    for j, color in enumerate(colors):
        axes[1].plot(x, -actual_shift[:, j] * 1e6, "o-", color=color, markersize=3.1, label=f"Mode {j+1}")
        axes[1].plot(x, -predicted_shift[:, j] * 1e6, "--", color=color, linewidth=.8)
    axes[1].plot([], [], "--", color=".4", linewidth=.8, label="Quartic average")
    axes[1].set(xlabel=r"Reference $I_1^0+I_2^0$", ylabel=r"$10^6(\nu_j-\omega_j)$", title="(b) Resolved nonlinear detuning")
    axes[1].legend(frameon=False)
    hs = np.array([row["h_tau"] for row in step_rows])
    for j, color in enumerate(colors):
        errors = np.array([row[f"abs_frequency_error{j+1}_tau"] for row in step_rows])
        axes[2].loglog(hs, errors, "o-", color=color, markersize=3.1, label=f"Mode {j+1}")
    axes[2].axhline(abs(actual_shift[1, 1]), color=".4", linestyle=":", linewidth=.9, label="Mode 2 detuning")
    axes[2].set(xlabel=r"Midpoint step $h$ in $\tau$", ylabel=r"$|\omega_{h,j}-\omega_j|$", title="(c) Discrete frequency error")
    axes[2].legend(frameon=False, loc="lower right")
    for ax in axes:
        ax.grid(alpha=.16, which="major")
    target = ROOT / "manuscript" / "figures" / "beam_kam"
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-steps", type=int, default=102400)
    arguments = parser.parse_args()
    if arguments.base_steps % 16:
        raise ValueError("base-steps must be divisible by 16")
    RAW.mkdir(parents=True, exist_ok=True)
    spec = json.loads(SPEC.read_text(encoding="utf-8-sig"))
    nu, d = np.array(spec["nu"]), np.array(spec["slope_map_d"])
    with np.load(DATA) as frozen:
        h0 = float(frozen["teacher_step_tau"])
        if not np.array_equal(frozen["nu"], nu) or not np.array_equal(frozen["d"], d):
            raise RuntimeError("Beam specification differs from the frozen learning teacher")
    sample_step = SAMPLE_STRIDE * h0
    tau = np.arange(arguments.base_steps // SAMPLE_STRIDE + 1) * sample_step
    normal, divisors, hessian = quartic_diagnostics(nu, d)
    write_csv(OUT / "low_order_divisors.csv", divisors)
    save_json(OUT / "quartic_normal_form.json", normal)
    verification = verify_midpoint_reduction(nu, d)
    save_json(OUT / "implementation_verification.json", verification)
    protocol = {
        "model": "Exactly the conservative two-mode learning beam h_*; no damping, contact, or time-dependent field.",
        "energy": ".5 sum nu_j*(X_j^2+Y_j^2)+1-cos(d.X)-.5*(d.X)^2",
        "nu": nu.tolist(), "d": d.tolist(), "omega0_rad_s": spec["omega0_rad_s"],
        "spec_file": SPEC.relative_to(ROOT).as_posix(), "spec_sha256": sha256(SPEC),
        "learning_teacher_file": DATA.relative_to(ROOT).as_posix(), "learning_teacher_sha256": sha256(DATA),
        "base_step_tau": h0, "base_steps": arguments.base_steps,
        "duration_tau": float(tau[-1]), "duration_s": float(tau[-1] / spec["omega0_rad_s"]),
        "sample_step_tau": sample_step, "samples_per_trajectory": len(tau),
        "continuous_method": "DOP853 with uniform dense-output samples", "rtol": RTOL, "atol": ATOL,
        "stable_trigonometric_evaluation": "For |theta|<0.1, the sine/cosine remainders are evaluated by their Taylor series through degrees 11/12, with absolute remainders at most 0.1^13/13! and 0.1^14/14!; otherwise direct trigonometric evaluation is used. This evaluates the same explicit model below floating-point cancellation error.",
        "tight_rtol": TIGHT_RTOL, "tight_atol": TIGHT_ATOL,
        "phases_rad": PHASES.tolist(), "base_radii": RADII.tolist(),
        "reference_action_difference_steps": ACTION_STEPS.tolist(),
        "frequency_estimator": "Smooth exp[-1/(s(1-s))] weighted mean of increments arg(conj(z_k) z_{k+1}), z_j=X_j-iY_j.",
        "windows": "full, first half, second half, central half; downsample by 2 and 4",
        "frequency_empirical_resolution": "max(1e-10, 10*maximum observed window, sampling, and tolerance changes), in tau^-1; diagnostic threshold, not a rigorous enclosure",
        "action_warning": "I0=r^2/2 labels an initial harmonic reference chart; not exact nonlinear actions. Numerical Jacobian d omega/d I0 is a chart response, not certified KAM twist.",
        "midpoint_steps": [h0 / 2**j for j in range(6)],
        "claim_boundary": "Finite-time frequencies and leading quartic nondegeneracy only; no invariant-torus certification, infinite-order Diophantine test, L2-to-KAM implication, or exact shadow Hamiltonian.",
        "script_sha256": sha256(__file__),
        "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
    }
    save_json(OUT / "protocol.json", protocol)
    cases, states_by_tag = {}, {}
    for j, radii in enumerate(RADII):
        diag, states = continuous_case(f"base{j}", radii**2 / 2, tau, sample_step, nu, d)
        cases[diag["tag"]], states_by_tag[diag["tag"]] = diag, states
    center_actions = RADII[1]**2 / 2
    for level in [1, 2]:
        for j in range(2):
            for sign in [-1, 1]:
                actions = center_actions.copy()
                actions[j] += sign * ACTION_STEPS[j] / level
                tag = f"grad{j}_level{level}_{'minus' if sign < 0 else 'plus'}"
                diag, _ = continuous_case(tag, actions, tau, sample_step, nu, d)
                cases[tag] = diag
    tight, _ = continuous_case("center_tight", center_actions, tau, sample_step, nu, d,
                               rtol=TIGHT_RTOL, atol=TIGHT_ATOL)
    tolerance_change = abs(np.array(tight["frequency_tau"]) - np.array(cases["base1"]["frequency_tau"]))
    maximum_window = max(max(row["max_window_change_tau"]) for row in cases.values())
    maximum_sampling = max(max(row["max_downsampling_change_tau"]) for row in cases.values())
    epsilon = max(1e-10, 10 * maximum_window, 10 * maximum_sampling, 10 * max(tolerance_change))
    gradients = []
    for level in [1, 2]:
        columns = []
        for j in range(2):
            plus = np.array(cases[f"grad{j}_level{level}_plus"]["frequency_tau"])
            minus = np.array(cases[f"grad{j}_level{level}_minus"]["frequency_tau"])
            columns.append((plus - minus) / (2 * ACTION_STEPS[j] / level))
        gradients.append(np.column_stack(columns))
    gradient = gradients[-1]
    gradient_diag = {
        "coordinate": "Derivative of measured modal phase frequency with respect to initial I0_j=r_j^2/2 at fixed initial phases; not exact nonlinear actions.",
        "center_actions0": center_actions.tolist(),
        "central_difference_full_steps": gradients[0].tolist(),
        "central_difference_half_steps": gradient.tolist(),
        "max_gradient_change_on_halving": float(np.max(np.abs(gradients[0] - gradient))),
        "half_step_gradient_determinant": float(np.linalg.det(gradient)),
        "half_step_gradient_singular_values": np.linalg.svd(gradient, compute_uv=False).tolist(),
        "leading_quartic_hessian_max_difference": float(np.max(abs(gradient - hessian))),
        "empirical_frequency_resolution_tau": epsilon,
        "empirical_derivative_resolution_from_frequency": (epsilon / (ACTION_STEPS / 2)).tolist(),
        "threshold_warning": "Empirical resolution is not an interval bound; determinant is not a certified Kolmogorov nondegeneracy bound.",
    }
    step_cases = []
    center_state = initial_state(center_actions)
    for j in range(6):
        factor = 2**j
        h = h0 / factor
        started = time.perf_counter()
        states, residual = midpoint_trajectory(center_state, nu, d, h,
                                               arguments.base_steps * factor, SAMPLE_STRIDE * factor)
        if residual > 5e-13:
            raise RuntimeError("Implicit midpoint residual exceeds acceptance threshold")
        raw = RAW / f"midpoint_level{j}.npz"
        np.savez_compressed(raw, tau=tau, states=states, h_tau=h, nu=nu, d=d,
                            max_midpoint_residual=residual)
        diag = frequency_diagnostics(states, sample_step)
        e = energy(states, nu, d)
        diag.update({"h_tau": h, "nsteps": arguments.base_steps * factor,
                     "max_midpoint_residual": residual,
                     "max_relative_energy_drift": float(np.max(abs(e - e[0])) / abs(e[0])),
                     "runtime_seconds": time.perf_counter() - started,
                     "raw_file": raw.relative_to(ROOT).as_posix(), "raw_sha256": sha256(raw)})
        step_cases.append(diag)
        print(f"MIDPOINT level {j}: h={h:.9g} omega={diag['frequency_tau']} residual={residual:.3g}", flush=True)
    reference_frequency = np.array(cases["base1"]["frequency_tau"])
    errors = np.abs(np.array([row["frequency_tau"] for row in step_cases]) - reference_frequency)
    observed_orders = np.log2(errors[:-1] / errors[1:])
    summary = {"protocol": protocol, "normal_form": normal,
               "base_cases": [cases[f"base{j}"] for j in range(3)],
               "continuous_difference_cases": [row for key, row in cases.items() if key.startswith("grad")],
               "tight_center_case": tight, "center_tolerance_frequency_change_tau": tolerance_change.tolist(),
               "maximum_window_frequency_change_tau": maximum_window,
               "maximum_downsampling_frequency_change_tau": maximum_sampling,
               "empirical_frequency_resolution_tau": epsilon,
               "reference_action_chart_frequency_gradient": gradient_diag,
               "midpoint_cases": step_cases, "midpoint_frequency_observed_orders": observed_orders.tolist(),
               "fft_bin_width_tau": float(2 * np.pi / (len(tau) * sample_step)),
               "finite_divisor_scan_at_center_l1_le32": min(
                   (dict(k1=row["k1"], k2=row["k2"], abs_k_dot_omega=abs(row["k1"] * reference_frequency[0] + row["k2"] * reference_frequency[1])) for row in divisors),
                   key=lambda row: row["abs_k_dot_omega"])}
    save_json(OUT / "summary.json", summary)
    make_outputs(summary, nu, d, hessian, tau, states_by_tag["base1"], step_cases)
    print(json.dumps({"empirical_frequency_resolution_tau": epsilon, "gradient": gradient_diag,
                      "orders": observed_orders.tolist()}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
