"""Illustrative clamped-free beam verification; no COMSOL or clinical data.

Euler--Bernoulli Hermite elements have consistent translational mass.
The tip rotation is approximated by the transverse slope; x_tip=L is fixed
in this small-slope model. G=0, hence actuation is a uniform-field torque.
All simulated quantities use SI units. This is an empirical discretization
and energy-balance check, not an interval-certified continuum error bound.
"""
from pathlib import Path
import json
import time
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

OUT = Path(__file__).resolve().parent
P = dict(length_m=0.08, radius_m=0.0006, young_modulus_Pa=5.0e6,
         density_kg_m3=1100.0, dipole_A_m2=2.0e-5, bias_field_T=0.03,
         transverse_field_amplitude_T=0.006, gravity_m_s2=0.0,
         field_gradient_T_m=0.0, damping_ratio_first=0.025,
         drive_frequency_ratio=0.85, ramp_time_s=0.25,
         duration_s=8.0, output_samples=2001)
P["area_m2"] = np.pi * P["radius_m"] ** 2
P["second_moment_m4"] = np.pi * P["radius_m"] ** 4 / 4
P["bending_rigidity_N_m2"] = P["young_modulus_Pa"] * P["second_moment_m4"]
P["line_mass_kg_m"] = P["density_kg_m3"] * P["area_m2"]
ROOTS = np.array([1.875104068711961, 4.694091132974174,
                  7.854757438237613, 10.995540734875467])
EXACT_W = ROOTS ** 2 * np.sqrt(P["bending_rigidity_N_m2"] /
                               P["line_mass_kg_m"] / P["length_m"] ** 4)
P["drag_mass_factor_s_inv"] = 2 * P["damping_ratio_first"] * EXACT_W[0]
P["drive_angular_frequency_rad_s"] = P["drive_frequency_ratio"] * EXACT_W[0]
P["rayleigh_drag_coefficient_N_s_m2"] = (P["drag_mass_factor_s_inv"] *
                                               P["line_mass_kg_m"])
TIMES = np.linspace(0, P["duration_s"], P["output_samples"])
ENERGY_SCALE = P["dipole_A_m2"] * P["bias_field_T"]


def assemble(ne):
    le = P["length_m"] / ne
    ke = P["bending_rigidity_N_m2"] / le ** 3 * np.array([
        [12, 6*le, -12, 6*le], [6*le, 4*le**2, -6*le, 2*le**2],
        [-12, -6*le, 12, -6*le], [6*le, 2*le**2, -6*le, 4*le**2]])
    me = P["line_mass_kg_m"] * le / 420 * np.array([
        [156, 22*le, 54, -13*le], [22*le, 4*le**2, 13*le, -3*le**2],
        [54, 13*le, 156, -22*le], [-13*le, -3*le**2, -22*le, 4*le**2]])
    m = np.zeros((2*(ne+1), 2*(ne+1)))
    k = np.zeros_like(m)
    for e in range(ne):
        ix = np.arange(2*e, 2*e+4)
        m[np.ix_(ix, ix)] += me
        k[np.ix_(ix, ix)] += ke
    # q=(w_1, theta_1,...,w_ne,theta_ne); w_0=theta_0=0.
    m, k = m[2:, 2:], k[2:, 2:]
    evals, phi = eigh(k, m)
    if np.any(evals <= 0):
        raise RuntimeError("Clamped stiffness is not positive definite")
    return m, k, np.sqrt(evals), phi


def field(t):
    a, om, tr = (P["transverse_field_amplitude_T"],
                 P["drive_angular_frequency_rad_s"], P["ramp_time_s"])
    ramp = 1 - np.exp(-(t/tr)**2)
    ramp_dt = 2*t/tr**2*np.exp(-(t/tr)**2)
    by = a*ramp*np.sin(om*t)
    by_dt = a*(ramp_dt*np.sin(om*t)+ramp*om*np.cos(om*t))
    return by, by_dt


def run_case(ne, modes=None, rtol=2e-8, atol=2e-12):
    start = time.perf_counter()
    m, k, w, phi = assemble(ne)
    nr = len(w) if modes is None else modes
    w, phi = w[:nr], phi[:, :nr]
    b = phi[-1].copy()
    alpha = P["drag_mass_factor_s_inv"]
    mu, b0 = P["dipole_A_m2"], P["bias_field_T"]

    def rhs(t, state):
        a, v = state[:nr], state[nr:2*nr]
        th = b @ a
        by, by_dt = field(t)
        tau = mu*(by*np.cos(th)-b0*np.sin(th))
        return np.r_[v, -w*w*a-alpha*v+b*tau,
                     -mu*by_dt*np.sin(th)/ENERGY_SCALE,
                     alpha*(v@v)/ENERGY_SCALE]

    def jac(t, state):
        th = b @ state[:nr]
        by, by_dt = field(t)
        j = np.zeros((2*nr+2, 2*nr+2))
        j[:nr, nr:2*nr] = np.eye(nr)
        j[nr:2*nr, :nr] = (-np.diag(w*w) -
                                      mu*(by*np.sin(th)+b0*np.cos(th))*np.outer(b,b))
        j[nr:2*nr, nr:2*nr] = -alpha*np.eye(nr)
        j[-2, :nr] = -mu*by_dt*np.cos(th)*b/ENERGY_SCALE
        j[-1, nr:2*nr] = 2*alpha*state[nr:2*nr]/ENERGY_SCALE
        return j

    # Audit work/dissipation states are dimensionless to prevent tolerances
    # on tiny joule values from concealing an inaccurate balance.
    abs_tol = np.r_[np.full(2*nr, atol), 1e-10, 1e-10]
    sol = solve_ivp(rhs, (0, P["duration_s"]), np.zeros(2*nr+2),
                    method="Radau", t_eval=TIMES, rtol=rtol, atol=abs_tol,
                    jac=jac, max_step=0.015)
    if not sol.success:
        raise RuntimeError(sol.message)
    a, v = sol.y[:nr], sol.y[nr:2*nr]
    q = phi @ a
    qdot = phi @ v
    th = q[-1]
    by, _ = field(TIMES)
    kinetic = 0.5*np.sum(v*v, axis=0)
    elastic = 0.5*np.sum((w[:,None]*a)**2, axis=0)
    magnetic = -mu*(b0*np.cos(th)+by*np.sin(th))
    energy = kinetic+elastic+magnetic
    delta = energy-energy[0]
    work, loss = ENERGY_SCALE*sol.y[-2], ENERGY_SCALE*sol.y[-1]
    balance = delta-work+loss
    energy_activity = max(np.max(np.abs(delta)), np.max(np.abs(work)),
                          np.max(np.abs(loss)), 1e-30)
    # Sample cubic Hermite displacement and its derivative inside each element.
    all_q = np.vstack((np.zeros((2,len(TIMES))), q))
    max_slope = 0.0
    max_y = 0.0
    le = P["length_m"]/ne
    for e in range(ne):
        qe = all_q[2*e:2*e+4]
        for xi in np.linspace(0,1,9):
            nsh = np.array([1-3*xi**2+2*xi**3,
                            le*(xi-2*xi**2+xi**3),
                            3*xi**2-2*xi**3, le*(-xi**2+xi**3)])
            dsh = np.array([(-6*xi+6*xi**2)/le, 1-4*xi+3*xi**2,
                            (6*xi-6*xi**2)/le, -2*xi+3*xi**2])
            max_slope=max(max_slope,float(np.max(np.abs(dsh@qe))))
            max_y=max(max_y,float(np.max(np.abs(nsh@qe))))
    # Independent finite-difference checks of potential torque and work rate.
    test_t, test_th, eps = 0.713, 0.043, 1e-6
    def vmag(tt, theta):
        return -mu*(b0*np.cos(theta)+field(tt)[0]*np.sin(theta))
    torque_fd=-(vmag(test_t,test_th+eps)-vmag(test_t,test_th-eps))/(2*eps)
    torque=mu*(field(test_t)[0]*np.cos(test_th)-b0*np.sin(test_th))
    power_fd=(vmag(test_t+eps,test_th)-vmag(test_t-eps,test_th))/(2*eps)
    power=-mu*field(test_t)[1]*np.sin(test_th)
    metrics=dict(elements=ne, modes=nr, rtol=rtol, atol_modal=atol,
                 wall_time_s=time.perf_counter()-start, nfev=sol.nfev,
                 njev=sol.njev, tip_peak_m=float(np.max(np.abs(q[-2]))),
                 sampled_max_displacement_m=max_y,
                 sampled_max_slope_rad=max_slope,
                 magnetic_potential_torque_fd_absolute_error_Nm=float(abs(torque-torque_fd)),
                 magnetic_input_power_fd_absolute_error_W=float(abs(power-power_fd)),
                 final_input_work_J=float(work[-1]), final_drag_loss_J=float(loss[-1]),
                 final_energy_change_J=float(delta[-1]),
                 max_energy_balance_residual_J=float(np.max(np.abs(balance))),
                 energy_balance_relative_to_activity=float(np.max(np.abs(balance))/energy_activity),
                 modal_mass_orthogonality_error=float(np.max(np.abs(phi.T@m@phi-np.eye(nr)))),
                 modal_stiffness_relative_error=float(np.linalg.norm(phi.T@k@phi-np.diag(w*w),ord=np.inf)/np.max(w*w)))
    name=f"beam_{ne}elements_{nr}modes"+("_tight" if rtol<1e-9 else "")
    np.savez_compressed(OUT/(name+".npz"), time_s=TIMES,q=q,qdot=qdot,
                        tip_displacement_m=q[-2],tip_slope_rad=th,
                        energy_J=energy,input_work_J=work,drag_loss_J=loss,
                        energy_residual_J=balance,frequencies_rad_s=w)
    print(name, "elapsed",round(metrics["wall_time_s"],2),"balance",
          metrics["energy_balance_relative_to_activity"],flush=True)
    return metrics, dict(tip=q[-2],theta=th,energy_delta=delta,work=work,loss=loss,
                         residual=balance,q=q)


def error_stats(a, reference):
    err=a-reference
    scale=max(float(np.max(np.abs(reference))),1e-30)
    return dict(max_abs_m=float(np.max(np.abs(err))),
                relative_max=float(np.max(np.abs(err))/scale),
                relative_rms=float(np.sqrt(np.mean(err**2))/scale))


def main():
    results={"description":"Illustrative SI clamped-free beam FE and modal convergence; no COMSOL data",
             "parameters":P,"exact_unmagnetized_angular_frequencies_rad_s":EXACT_W.tolist(),
             "scope":["Uniform field G=0; tip actuation is a torque, with no translational magnetic force",
                      "Euler-Bernoulli small-slope, fixed x_tip=L; theta_tip approximated by transverse slope",
                      "Consistent translational mass; rotary inertia, axial extension, contact and moving-fluid forcing omitted",
                      "D=alpha M comes from distributed stationary-medium Rayleigh drag; gravity set to zero",
                      "Mesh/modal errors are empirical, not a rigorous epsilon_N or continuum safety certificate",
                      "No KAM conclusion is asserted for this driven damped experiment"],
             "mesh":[],"dynamics":[],"modal_truncation":[]}
    mesh_data={}
    for ne in [8,16,32]:
        m,k,w,phi=assemble(ne)
        load=np.zeros(2*ne); load[-1]=1e-7
        static=np.linalg.solve(k,load)
        static_exact=load[-1]*P["length_m"]**2/(2*P["bending_rigidity_N_m2"])
        results["mesh"].append(dict(elements=ne, frequencies_rad_s=w[:4].tolist(),
                     relative_frequency_error=(np.abs(w[:4]/EXACT_W-1)).tolist(),
                     static_tip_m=float(static[-2]),static_tip_exact_m=float(static_exact),
                     static_tip_relative_error=float(abs(static[-2]/static_exact-1)),
                     min_mass_eigenvalue=float(np.min(np.linalg.eigvalsh(m)))))
        met,data=run_case(ne)
        results["dynamics"].append(met)
        mesh_data[ne]=data
    ref=mesh_data[32]
    for met in results["dynamics"]:
        met["tip_error_against_32element_full"]=error_stats(mesh_data[met["elements"]]["tip"],ref["tip"])
    mode_data={}
    for modes in [2,4,8]:
        met,data=run_case(32,modes)
        met["tip_error_against_32element_full"]=error_stats(data["tip"],ref["tip"])
        met["slope_relative_max_error"]=float(np.max(np.abs(data["theta"]-ref["theta"]))/np.max(np.abs(ref["theta"])))
        results["modal_truncation"].append(met)
        mode_data[modes]=data
    met,tight=run_case(32,rtol=2e-10,atol=2e-14)
    met["tip_error_against_standard_tolerance"]=error_stats(tight["tip"],ref["tip"])
    results["solver_tolerance_check"]=met
    # Meaningful physical/implementation checks, not a claim of model validation.
    assert all(x["min_mass_eigenvalue"]>0 for x in results["mesh"])
    assert max(x["static_tip_relative_error"] for x in results["mesh"])<1e-7
    assert max(x["energy_balance_relative_to_activity"] for x in results["dynamics"])<1e-4
    assert met["tip_error_against_standard_tolerance"]["relative_max"]<1e-4
    results["checks"]={"positive_mass":True,"static_tip_moment_solution":True,
                       "dynamic_power_balance_below_1e-4_relative":True,
                       "solver_tolerance_tip_check_below_1e-4_relative":True}
    (OUT/"results.json").write_text(json.dumps(results,indent=2),encoding="utf8")

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.titlesize":10,"axes.labelsize":9,
                         "pdf.fonttype":42,"savefig.dpi":240})
    fig,axes=plt.subplots(1,3,figsize=(13.2,3.7),layout="constrained")
    colors=["#2166ac","#d6604d","#1b7837","#762a83"]
    for j in range(4):
        axes[0].loglog([x["elements"] for x in results["mesh"]],
                      [x["relative_frequency_error"][j] for x in results["mesh"]],
                      "o-",color=colors[j],label=f"Mode {j+1}")
    axes[0].set(title="a  Consistent-mass mesh refinement",xlabel="Beam elements",
                ylabel="Relative frequency error")
    axes[0].set_xticks([8,16,32],["8","16","32"])
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].legend(fontsize=8,frameon=False)
    axes[0].grid(True,which="major",alpha=.15)
    axes[1].plot(TIMES,ref["tip"]*1e3,color="#222222",lw=1.5,label="Full FE (64 modes)")
    for i,nm in enumerate([2,4,8]):
        axes[1].plot(TIMES,mode_data[nm]["tip"]*1e3,color=colors[i],
                     lw=.9,ls=["--",":","-."][i],label=f"{nm} modes")
    axes[1].set(title="b  Driven, damped magnetic response",xlabel="Time (s)",
                ylabel="Tip displacement (mm)")
    axes[1].legend(fontsize=7,frameon=False,ncol=2,loc="lower right")
    axes[2].plot(TIMES,ref["work"]*1e6,color=colors[0],label="Magnetic input work")
    axes[2].plot(TIMES,ref["loss"]*1e6,color=colors[1],label="Rayleigh dissipation")
    axes[2].plot(TIMES,ref["energy_delta"]*1e6,color="#222222",label="Energy change")
    axes[2].set(title="c  Time-dependent power balance",xlabel="Time (s)",ylabel="Energy (microjoule)")
    axes[2].legend(fontsize=7,frameon=False)
    rel=results["dynamics"][-1]["energy_balance_relative_to_activity"]
    axes[2].text(.97,.035,f"Max. relative balance residual: {rel:.1e}",
                 transform=axes[2].transAxes,va="bottom",ha="right",fontsize=7,
                 bbox={"facecolor":"white","edgecolor":"none","alpha":.85,"pad":2})
    fig.savefig(OUT/"beam_validation_3panel.png")
    fig.savefig(OUT/"beam_validation_3panel.pdf")
    plt.close(fig)
    np.savetxt(OUT/"reference_energy_balance.csv",np.column_stack([TIMES,ref["tip"],
               ref["theta"],ref["energy_delta"],ref["work"],ref["loss"],ref["residual"]]),
               delimiter=",",header="time_s,tip_m,tip_slope_rad,delta_energy_J,input_work_J,drag_loss_J,balance_residual_J",comments="")
    print("All checks PASS; results.json and three-panel PDF/PNG saved",flush=True)


if __name__=="__main__":
    main()
