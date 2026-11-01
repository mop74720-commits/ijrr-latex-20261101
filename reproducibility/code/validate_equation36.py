"""Reproduce equation (36) without replacing or correcting its mathematics.

The scalar diagnostic systems below are explicitly chosen numerical tests,
not physical parameter estimates for the author's four-field model.
"""
from pathlib import Path
import csv
import json
import platform
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
J = np.array([[0., 1.], [-1., 0.]])


def linear_step_matrix(h, mass=1., stiffness=1., damping=0.):
    """Literal scalar constant-coefficient specialization of (36), F=Delta=0."""
    eps = stiffness / mass * h * h / 4 + damping / mass * h / 2
    lower = np.array([-stiffness * h, 1 - eps]) / (1 + eps)
    upper = np.array([1., 0.]) + h / (2 * mass) * (lower + np.array([0., 1.]))
    return np.array([upper, lower])


def coefficients(q, a, beta):
    mass = 1 + a * q * q
    inverse_mass = 1 / mass
    derivative_inverse_mass = -2 * a * q / (mass * mass)
    stiffness = 1 + beta * q * q
    return inverse_mass, derivative_inverse_mass, stiffness


def nonlinear_residual(new, old, h, a, beta, damping):
    """Equation (36), midpoint coefficient interpretation inherited from (35)."""
    q, p = old
    qn, pn = new
    qm, pm = (q + qn) / 2, (p + pn) / 2
    inv_mass, dinv_mass, stiffness = coefficients(qm, a, beta)
    delta = -.5 * pm * pm * dinv_mass
    eps = stiffness * inv_mass * h * h / 4 + damping * inv_mass * h / 2
    return np.array([
        qn - q - h * inv_mass * (pn + p) / 2,
        pn - ((1 - eps) * p - stiffness * q * h + delta * h) / (1 + eps),
    ])


def continuous_rhs(t, state, a, beta, damping):
    q, p = state
    inv_mass, dinv_mass, stiffness = coefficients(q, a, beta)
    return np.array([inv_mass * p,
                     -.5 * p * p * dinv_mass - stiffness * q - damping * inv_mass * p])


def complex_jacobian(fn, state):
    cols = []
    for j in range(len(state)):
        perturbed = np.asarray(state, dtype=complex).copy()
        perturbed[j] += 1e-25j
        cols.append(np.imag(fn(perturbed)) / 1e-25)
    return np.column_stack(cols)


def nonlinear_step(old, h, a, beta, damping):
    new = np.asarray(old, dtype=float) + h * continuous_rhs(0, old, a, beta, damping)
    fn = lambda value: nonlinear_residual(value, old, h, a, beta, damping)
    for _ in range(15):
        residual = fn(new)
        if np.linalg.norm(residual, np.inf) < 2e-14:
            return new, float(np.linalg.norm(residual, np.inf))
        new -= np.linalg.solve(complex_jacobian(fn, new), residual)
    raise RuntimeError("Equation (36) implicit solve did not converge.")


def nonlinear_tangent(old, new, h, a, beta, damping):
    new_jac = complex_jacobian(lambda v: nonlinear_residual(v, old, h, a, beta, damping), new)
    old_jac = complex_jacobian(lambda v: nonlinear_residual(new, v, h, a, beta, damping), old)
    return -np.linalg.solve(new_jac, old_jac)


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run():
    OUT.mkdir(exist_ok=True)
    initial = np.array([1., 0.])
    h_values = [.4, .2, .1, .05, .025]
    horizon = 10.
    convergence, structure, histories = [], [], {}
    checks = {}
    for damping in [0., .2]:
        generator = np.array([[0., 1.], [-1., -damping]])
        exact = expm(generator * horizon) @ initial
        previous = None
        for h in h_values:
            step = linear_step_matrix(h, damping=damping)
            cayley = np.linalg.solve(np.eye(2) - h * generator / 2, np.eye(2) + h * generator / 2)
            if not np.allclose(step, cayley, rtol=0, atol=5e-16):
                raise AssertionError("Literal (36) does not match the independent midpoint equation.")
            state = np.linalg.matrix_power(step, round(horizon / h)) @ initial
            error = float(np.linalg.norm(state - exact))
            order = None if previous is None else float(np.log2(previous / error))
            convergence.append({"case": "linear_conservative" if damping == 0 else "linear_damped",
                                "h": h, "horizon": horizon, "state_error": error, "observed_order": order})
            previous = error
            ordinary = np.linalg.norm(step.T @ J @ step - J) / np.linalg.norm(J)
            birkhoff = np.linalg.norm(np.exp(damping * h) * step.T @ J @ step - J) / np.linalg.norm(J)
            exact_step = expm(generator * h)
            exact_birkhoff = np.linalg.norm(np.exp(damping * h) * exact_step.T @ J @ exact_step - J) / np.linalg.norm(J)
            structure.append({"damping": damping, "h": h, "map_determinant": float(np.linalg.det(step)),
                              "required_determinant": float(np.exp(-damping * h)),
                              "canonical_defect_relative": float(ordinary),
                              "continuous_birkhoff_defect_relative": float(birkhoff),
                              "exact_flow_birkhoff_defect_relative": float(exact_birkhoff)})
        h = .1
        step = linear_step_matrix(h, damping=damping)
        states = [initial.copy()]
        balance, maximum_residual = [], 0.
        for _ in range(1000):
            old = states[-1]
            new = step @ old
            midpoint = (old + new) / 2
            maximum_residual = max(maximum_residual, float(np.linalg.norm(new - old - h * generator @ midpoint, np.inf)))
            balance.append(float((new @ new - old @ old) / 2 + h * damping * midpoint[1] ** 2))
            states.append(new)
        states = np.array(states)
        case = "conservative" if damping == 0 else "damped"
        histories[case] = states
        checks[case] = {"max_step_residual": maximum_residual,
                        "max_discrete_power_balance_residual": float(max(abs(np.array(balance)))),
                        "max_energy_change": float(max(abs((states ** 2).sum(axis=1) / 2 - .5)))}

    nonlinear_details = []
    for a, beta, damping, name in [(.3, .4, 0., "variable_mass_conservative"),
                                   (.3, .4, .2, "variable_mass_damped")]:
        start = np.array([.7, .2])
        rhs = lambda t, x: continuous_rhs(t, x, a, beta, damping)
        reference = solve_ivp(rhs, [0, horizon], start, method="DOP853", rtol=2e-12, atol=2e-14)
        refined = solve_ivp(rhs, [0, horizon], start, method="DOP853", rtol=3e-14, atol=3e-15)
        if not reference.success or not refined.success:
            raise RuntimeError("Independent reference integration failed.")
        reference_change = float(np.linalg.norm(reference.y[:, -1] - refined.y[:, -1]))
        previous = None
        for h in h_values:
            state = start.copy()
            max_residual = 0.
            for _ in range(round(horizon / h)):
                state, residual = nonlinear_step(state, h, a, beta, damping)
                max_residual = max(max_residual, residual)
            error = float(np.linalg.norm(state - refined.y[:, -1]))
            order = None if previous is None else float(np.log2(previous / error))
            convergence.append({"case": name, "h": h, "horizon": horizon,
                                "state_error": error, "observed_order": order})
            previous = error
            nonlinear_details.append({"case": name, "h": h, "max_step_residual": max_residual,
                                      "reference_refinement_change": reference_change})
        end, _ = nonlinear_step(start, .1, a, beta, damping)
        tangent = nonlinear_tangent(start, end, .1, a, beta, damping)
        nonlinear_details.append({"case": name, "h": .1,
                                  "canonical_defect_relative": float(np.linalg.norm(tangent.T @ J @ tangent - J) / np.linalg.norm(J))})

    counter = next(r for r in structure if r["damping"] == .2 and r["h"] == .1)
    assert counter["continuous_birkhoff_defect_relative"] > 1e-5
    assert counter["exact_flow_birkhoff_defect_relative"] < 1e-13
    assert checks["conservative"]["max_energy_change"] < 1e-11
    assert checks["damped"]["max_discrete_power_balance_residual"] < 1e-13
    assert all(row["observed_order"] > 1.9 for row in convergence if row["h"] == .025)
    report = {
        "status": "diagnostics_completed; general_Birkhoff_preservation_claim_fails",
        "source_equation_changed": False,
        "test_scope": "Scalar specializations of equation (36); not a calibrated four-field robotic model.",
        "chosen_test_parameters": {"linear": {"m": 1., "k": 1., "c": [0., .2]},
                                  "variable_mass": {"M(q)": "1+0.3*q^2", "K(q)": "1+0.4*q^2", "c": [0., .2]}},
        "interpretation": "Coefficients evaluated at midpoint as in (35); Delta=-0.5*p_mid^2*d(M^-1)/dq. No original equation edited.",
        "counterexample": counter, "trajectory_checks": checks,
        "nonlinear_checks": nonlinear_details,
        "reference": "Feng and Qin, Chapter 15, printed p.630 / PDF p.647, explicitly distinguishes midpoint from K(z,t)-symplectic schemes.",
        "claim_limits": ["Second-order observations apply to these smooth midpoint specializations, not every unspecified coefficient dependence.",
                         "Discrete work balance does not imply preservation of the continuous Birkhoff two-form.",
                         "No CHNN/RHNN training or KAM certificate has been produced."],
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }
    write_csv(OUT / "convergence.csv", convergence)
    write_csv(OUT / "structure_defects.csv", structure)
    (OUT / "diagnostic_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(OUT / "diagnostic_trajectories.npz", time=np.arange(1001) * .1, **histories)
    print(json.dumps({"status": report["status"], "counterexample": counter,
                      "finest_orders": {r["case"]: r["observed_order"] for r in convergence if r["h"] == .025},
                      "all_implementation_checks_passed": True}, indent=2))


if __name__ == "__main__":
    run()
