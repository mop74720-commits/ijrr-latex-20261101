"""KAM-related numerical diagnostics for a disclosed scalar specialization of (36).

This script does not certify a KAM torus, prove nonresonance, or alter the source
equation. All parameters are dimensionless diagnostic choices, not fitted robot
parameters. Trajectories use the existing implementation of equation (36).
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import platform
import sys
import time

import numpy as np
import scipy
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from validate_equation36 import nonlinear_step, nonlinear_tangent, continuous_rhs


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "kam"
MASS_COEFFICIENT = 0.3
STIFFNESS_COEFFICIENT = 0.4
AMPLITUDES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
STEP_SIZES = [0.1, 0.05, 0.025]
HORIZON = 300.0
RESONANCE_CUTOFF = 128
J = np.array([[0., 1.], [-1., 0.]])


def energy(states):
    q, p = np.asarray(states).T
    return p * p / (2 * (1 + MASS_COEFFICIENT * q * q)) + q * q / 2 + STIFFNESS_COEFFICIENT * q ** 4 / 4


def continuous_reference(amplitude):
    """Endpoint-regularized quadrature for the declared 1-DOF energy level.

    q=A sin(theta) removes the square-root endpoint singularity in both the
    period and action integrals. This is a reference diagnostic calculation,
    not a new integrator or a replacement of equation (36).
    """
    a, b, amplitude2 = MASS_COEFFICIENT, STIFFNESS_COEFFICIENT, amplitude * amplitude

    def period_integrand(theta):
        sine2 = np.sin(theta) ** 2
        return np.sqrt((1 + a * amplitude2 * sine2) / (1 + b * amplitude2 * (1 + sine2) / 2))

    def action_integrand(theta):
        sine2 = np.sin(theta) ** 2
        return amplitude2 * np.cos(theta) ** 2 * np.sqrt(
            (1 + a * amplitude2 * sine2) * (1 + b * amplitude2 * (1 + sine2) / 2))

    period_value, period_bound = quad(period_integrand, 0, np.pi / 2, epsabs=1e-13, epsrel=1e-13)
    action_value, action_bound = quad(action_integrand, 0, np.pi / 2, epsabs=1e-13, epsrel=1e-13)
    period = 4 * period_value
    return {
        "amplitude": float(amplitude),
        "energy": float(energy(np.array([amplitude, 0.]))),
        "period": float(period),
        "omega": float(2 * np.pi / period),
        "action": float(2 * action_value / np.pi),
        "quad_period_error_estimate": float(4 * period_bound),
        "quad_action_error_estimate": float(2 * action_bound / np.pi),
    }


def action_from_energy(value):
    # Positive root of b*A^4/4 + A^2/2 = E, evaluated without subtraction.
    amp2 = 4 * value / (1 + np.sqrt(1 + 4 * STIFFNESS_COEFFICIENT * value))
    return continuous_reference(float(np.sqrt(amp2)))["action"]


def numerical_twist(amplitude, da):
    left = continuous_reference(amplitude - da)
    right = continuous_reference(amplitude + da)
    return (right["omega"] - left["omega"]) / (right["action"] - left["action"])


def independent_period_check(amplitude, period):
    def event(_time, state):
        return state[0]

    event.direction = -1
    event.terminal = False
    result = solve_ivp(
        lambda t, state: continuous_rhs(t, state, MASS_COEFFICIENT, STIFFNESS_COEFFICIENT, 0.),
        (0., 3.2 * period), np.array([amplitude, 0.]), method="DOP853",
        rtol=2e-13, atol=2e-15, max_step=period / 40, events=event,
    )
    if not result.success or len(result.t_events[0]) < 3:
        raise RuntimeError("Independent continuous-period event check failed.")
    measured = float(np.mean(np.diff(result.t_events[0])))
    if abs(measured - period) > 1e-10:
        raise AssertionError("Quadrature and independent DOP853 period disagree.")
    return measured


def smooth_phase_frequency(phase, h):
    increments = np.diff(phase)
    u = (np.arange(len(increments)) + .5) / len(increments)
    with np.errstate(under="ignore"):
        weights = np.exp(-1 / (u * (1 - u)))
    return float(np.sum(weights * increments) / (h * np.sum(weights)))


def downward_crossing_times(q, h):
    indices = np.flatnonzero((q[:-1] >= 0) & (q[1:] < 0))
    crossings = []
    for idx in indices:
        if idx < 1 or idx + 2 >= len(q):
            continue
        # Local time in units of h avoids ill-conditioning at late times.
        coefficients = np.polynomial.polynomial.polyfit([-1., 0., 1., 2.], q[idx-1:idx+3], 3)
        root = brentq(lambda value: np.polynomial.polynomial.polyval(value, coefficients), 0., 1., xtol=1e-14)
        crossings.append((idx + root) * h)
    return np.asarray(crossings)


def spectral_diagnostic(q, h):
    # Exclude the final endpoint to use exactly N=horizon/h sampled values.
    values = q[:-1] - np.mean(q[:-1])
    transform = np.fft.rfft(values * np.hanning(len(values)))
    power = np.abs(transform) ** 2
    omega = 2 * np.pi * np.fft.rfftfreq(len(values), d=h)
    peak = 1 + int(np.argmax(power[1:]))
    correction = 0.
    if 0 < peak < len(power)-1:
        left, center, right = np.log(np.maximum(power[peak-1:peak+2], np.finfo(float).tiny))
        correction = float(.5 * (left-right) / (left-2*center+right))
    peak_omega = float((peak + correction) * omega[1])
    candidates = np.flatnonzero((power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])) + 1
    peaks = sorted(candidates, key=lambda index: power[index], reverse=True)[:5]
    top_peaks = [{"omega_bin": float(omega[index]), "power_relative_to_primary": float(power[index]/power[peak])}
                 for index in peaks]
    return omega, power, peak_omega, top_peaks


def action_history(energies):
    low, high = float(np.min(energies)), float(np.max(energies))
    if high-low < 1e-14 * max(high, 1.):
        values = np.full_like(energies, action_from_energy(float(np.mean(energies))))
        return values, 0.
    nodes = np.linspace(low, high, 33)
    interpolator = PchipInterpolator(nodes, np.array([action_from_energy(value) for value in nodes]))
    values = interpolator(energies)
    sample_indices = np.unique(np.linspace(0, len(energies)-1, 51).astype(int))
    errors = [abs(float(values[index])-action_from_energy(float(energies[index]))) for index in sample_indices]
    maximum = float(max(errors))
    if maximum > 1e-10:
        raise AssertionError("Physical-action interpolation error is too large.")
    return values, maximum


def one_case(amplitude, h, reference):
    steps = round(HORIZON / h)
    assert abs(steps*h-HORIZON) < 1e-12
    states = np.empty((steps+1, 2))
    states[0] = [amplitude, 0.]
    maximum_residual = 0.
    for index in range(steps):
        states[index+1], residual = nonlinear_step(
            states[index], h, MASS_COEFFICIENT, STIFFNESS_COEFFICIENT, 0.)
        maximum_residual = max(maximum_residual, residual)
    if not np.all(np.isfinite(states)):
        raise AssertionError("Non-finite trajectory.")

    times = np.arange(steps+1) * h
    energies = energy(states)
    actions, interpolation_error = action_history(energies)
    phase = np.unwrap(np.arctan2(-states[:, 1], states[:, 0]))
    omega_phase = smooth_phase_frequency(phase, h)
    half = steps // 2
    omega_first = smooth_phase_frequency(phase[:half+1], h)
    omega_second = smooth_phase_frequency(phase[half:], h)
    crossing_times = downward_crossing_times(states[:, 0], h)
    if len(crossing_times) < 20:
        raise AssertionError("Too few independent downward crossings for the frequency diagnostic.")
    period_crossing, intercept = np.polyfit(np.arange(len(crossing_times)), crossing_times, 1)
    omega_crossing = float(2*np.pi/period_crossing)
    disagreement = abs(omega_crossing-omega_phase)
    if disagreement > 5e-5:
        raise AssertionError("Phase and crossing frequency estimators disagree.")

    sample_steps = np.unique(np.linspace(0, steps-1, 25).astype(int))
    defects = []
    for index in sample_steps:
        tangent = nonlinear_tangent(states[index], states[index+1], h,
                                    MASS_COEFFICIENT, STIFFNESS_COEFFICIENT, 0.)
        defects.append(float(np.linalg.norm(tangent.T@J@tangent-J)/np.linalg.norm(J)))
    maximum_defect = max(defects)
    if maximum_defect > 1e-10:
        raise AssertionError("Unexpected canonical-structure defect in the conservative diagnostic.")

    rho = omega_phase*h/(2*np.pi)
    integer_orders = np.arange(1, RESONANCE_CUTOFF+1)
    detunings = np.abs(integer_orders*rho-np.round(integer_orders*rho))
    worst_index = int(np.argmin(detunings))
    spectral_omega, spectral_power, fft_peak, peaks = spectral_diagnostic(states[:, 0], h)
    energy_change = energies-reference["energy"]
    action_change = actions-reference["action"]
    case_id = f"A{amplitude:.1f}_h{h:.3f}".replace(".", "p")
    np.savez_compressed(
        OUT/f"{case_id}.npz", time=times, state=states, physical_energy=energies,
        physical_action=actions, unwrapped_phase=phase, spectrum_omega=spectral_omega,
        spectrum_power=spectral_power, downward_crossing_times=crossing_times,
    )
    metrics = {
        "case": case_id, "amplitude": amplitude, "h": h, "horizon": HORIZON,
        "steps": steps, "continuous_omega": reference["omega"], "continuous_action": reference["action"],
        "omega_phase_weighted": omega_phase, "omega_crossing": omega_crossing,
        "omega_fft_interpolated": fft_peak, "fft_resolution_omega": float(2*np.pi/HORIZON),
        "omega_error_signed": omega_phase-reference["omega"],
        "omega_error_absolute": abs(omega_phase-reference["omega"]),
        "omega_estimators_disagreement": disagreement,
        "omega_first_second_window_difference": abs(omega_first-omega_second),
        "rotation_number_per_step": rho, "crossing_count": len(crossing_times),
        "max_energy_change_absolute": float(np.max(np.abs(energy_change))),
        "energy_peak_to_peak": float(np.ptp(energies)),
        "max_energy_change_relative": float(np.max(np.abs(energy_change))/reference["energy"]),
        "end_energy_change": float(energy_change[-1]),
        "max_action_change_absolute": float(np.max(np.abs(action_change))),
        "max_action_change_relative": float(np.max(np.abs(action_change))/reference["action"]),
        "end_action_change": float(action_change[-1]),
        "max_implicit_residual": maximum_residual,
        "max_sampled_canonical_defect": maximum_defect,
        "sampled_canonical_steps": len(sample_steps),
        "action_interpolation_check_error": interpolation_error,
        "finite_nonresonance_cutoff": RESONANCE_CUTOFF,
        "finite_detuning_min": float(detunings[worst_index]),
        "finite_detuning_witness_order": int(integer_orders[worst_index]),
    }
    details = {**metrics, "spectrum_largest_local_peaks": peaks,
               "crossing_fit_intercept": float(intercept),
               "trajectory_file": f"{case_id}.npz"}
    return metrics, details


def write_csv(name, rows):
    with (OUT/name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    source_file = Path(__file__).with_name("validate_equation36.py")
    protocol = {
        "model": {"M(q)": "1+0.3*q^2", "K(q)": "1+0.4*q^2", "C": 0., "Fext": 0.},
        "parameter_status": "Dimensionless diagnostic choices; not fitted physical robot parameters.",
        "amplitudes": AMPLITUDES, "step_sizes": STEP_SIZES, "horizon": HORIZON,
        "initial_state": "[amplitude, 0]", "integrator": "Imported scalar equation (36), midpoint coefficient interpretation from (35).",
        "reference": "Endpoint-regularized action/period quadrature, independently checked by DOP853 events.",
        "twist": "Symmetric amplitude finite differences, delta A=1e-4 and 5e-5; ratio delta omega/delta I.",
        "phase_frequency": "Weighted average of unwrapped atan2(-p,q) increments; weights exp(-1/(u*(1-u))).",
        "crossing_frequency": "Linear fit of cubic-interpolated downward q=0 crossing times against crossing index.",
        "spectrum": "Hann-windowed q samples; logarithmic parabolic peak interpolation; no multiple-frequency claim.",
        "finite_resonance_cutoff": RESONANCE_CUTOFF,
        "source_integrator_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "claim_scope": "Numerical structure, frequency, twist and drift diagnostics; no rigorous KAM certificate.",
    }
    (OUT/"protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    started = time.perf_counter()
    references, cases, detailed_cases = [], [], []
    for amplitude in AMPLITUDES:
        reference = continuous_reference(amplitude)
        reference["DOP853_event_period"] = independent_period_check(amplitude, reference["period"])
        reference["period_independent_check_error"] = abs(reference["period"]-reference["DOP853_event_period"])
        twist = numerical_twist(amplitude, 1e-4)
        refined_twist = numerical_twist(amplitude, 5e-5)
        reference["numerical_twist"] = refined_twist
        reference["twist_difference_refinement"] = abs(twist-refined_twist)
        references.append(reference)
        for h in STEP_SIZES:
            metrics, detail = one_case(amplitude, h, reference)
            cases.append(metrics)
            detailed_cases.append(detail)
            print(json.dumps({"completed": metrics["case"], "omega_error": metrics["omega_error_absolute"],
                              "energy_relative_change": metrics["max_energy_change_relative"]}), flush=True)
    sensitivity = []
    for amplitude in AMPLITUDES:
        rows = [case for case in cases if case["amplitude"] == amplitude]
        for coarse, fine in zip(rows[:-1], rows[1:]):
            sensitivity.append({
                "amplitude": amplitude, "coarse_h": coarse["h"], "fine_h": fine["h"],
                "observed_frequency_error_order": float(np.log(coarse["omega_error_absolute"]/fine["omega_error_absolute"])/np.log(coarse["h"]/fine["h"])),
                "energy_excursion_reduction_factor": coarse["max_energy_change_absolute"]/fine["max_energy_change_absolute"],
                "action_excursion_reduction_factor": coarse["max_action_change_absolute"]/fine["max_action_change_absolute"],
            })
    summary = {
        "status": "completed_numerical_diagnostics_only", "protocol": protocol,
        "continuous_references": references, "case_metrics": detailed_cases,
        "step_sensitivity": sensitivity,
        "summary": {
            "case_count": len(cases), "total_steps": sum(case["steps"] for case in cases),
            "elapsed_seconds": time.perf_counter()-started,
            "sampled_twist_min": min(row["numerical_twist"] for row in references),
            "sampled_twist_max": max(row["numerical_twist"] for row in references),
            "maximum_twist_refinement_change": max(row["twist_difference_refinement"] for row in references),
            "maximum_period_independent_check_error": max(row["period_independent_check_error"] for row in references),
            "maximum_implicit_residual": max(row["max_implicit_residual"] for row in cases),
            "maximum_sampled_canonical_defect": max(row["max_sampled_canonical_defect"] for row in cases),
            "maximum_phase_crossing_frequency_disagreement": max(row["omega_estimators_disagreement"] for row in cases),
            "maximum_first_second_window_frequency_difference": max(row["omega_first_second_window_difference"] for row in cases),
            "observed_frequency_order_min": min(row["observed_frequency_error_order"] for row in sensitivity),
            "observed_frequency_order_max": max(row["observed_frequency_error_order"] for row in sensitivity),
        },
        "limitations": [
            "Finite-time bounded trajectories and Fourier peaks do not prove an invariant circle or irrational rotation.",
            "Sampled positive twist is not a rigorous lower bound on an interval or an analytic neighborhood.",
            "The cutoff-128 detuning check is not an infinite-order Diophantine condition or an error-certified bound.",
            "The action reported along discrete trajectories is the original continuous Hamiltonian action evaluated at the numerical energy, not an exactly conserved discrete action.",
            "No candidate invariant-circle embedding, posterior KAM constants, or rigorous torus certificate is computed.",
            "The conservative one-degree-of-freedom result does not apply to the full driven, damped, contact-coupled robotic model.",
        ],
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    write_csv("continuous_references.csv", references)
    write_csv("case_metrics.csv", cases)
    write_csv("step_sensitivity.csv", sensitivity)
    (OUT/"results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (OUT/"README.md").write_text(
        "# 保守标量KAM相关数值诊断\n\n"
        "protocol.json保存运行前设置；results.json保存全量指标及适用边界。\n\n"
        "continuous_references.csv是连续参考周期、作用量和数值twist；case_metrics.csv是18组离散实验；step_sensitivity.csv是步长减半比较。\n\n"
        "每个NPZ包含time、state(q,p)、physical_energy、physical_action、unwrapped_phase、spectrum_omega、spectrum_power及downward_crossing_times。\n\n"
        "所有参数均为无量纲诊断设置。这些结果不构成KAM认证，也不证明全系统长期稳定。\n",
        encoding="utf-8")
    print(json.dumps(summary["summary"], indent=2), flush=True)


if __name__ == "__main__":
    run()
