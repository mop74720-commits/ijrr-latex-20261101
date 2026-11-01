"""Independent numerical checks and an eight-mode sensitivity diagnostic.

This writes only new exploration artifacts. It does not recalibrate the source
physics, change old training data, or infer material limits from unrelated rods.
"""
from pathlib import Path
import json
import time
import numpy as np
from scipy.integrate import solve_ivp
from build_models import (ROOT, OUT, build_model, project_initials, geometry_checks,
                          energy_geometry_bound, energy, sha, save)


def rhs_si(model):
    n=model['modes'];nu=model['nu'];d=model['d'];omega0=model['omega0']
    def rhs(t,state):
        x=state[:n];y=state[n:];theta=d@x
        return omega0*np.r_[nu*y,-nu*x-d*(np.sin(theta)-theta)]
    return rhs


def nonlinear_checks(states,model):
    n=model['modes'];x=np.asarray(states)[...,:n];d=model['d'];theta=x@d
    nonlinear=np.abs(np.sin(theta)-theta)*np.linalg.norm(d)
    linear=np.linalg.norm(x*model['nu'],axis=-1)
    h4_remainder=np.abs(np.sin(theta)-theta+theta**3/6)*np.linalg.norm(d)
    quartic=np.abs(theta)**3/6*np.linalg.norm(d)
    return dict(max_tip_slope=float(np.max(np.abs(theta))),
        nonlinear_to_linear_force_l2_rms=float(np.linalg.norm(nonlinear)/max(np.linalg.norm(linear),1e-30)),
        nonlinear_to_linear_force_pointwise_max=float(np.max(nonlinear/np.maximum(linear,1e-15))),
        h4_remainder_to_linear_force_l2_rms=float(np.linalg.norm(h4_remainder)/max(np.linalg.norm(linear),1e-30)),
        h4_remainder_to_quartic_force_l2_rms=float(np.linalg.norm(h4_remainder)/max(np.linalg.norm(quartic),1e-30)),
        max_theta_squared_over_20=float(np.max(theta**2)/20))


def check_hermite_extrema(model):
    rng=np.random.default_rng(20261101)
    states=rng.normal(0,.1,(13,2*model['modes']))
    exact=geometry_checks(states,model)['max_abs_slope']
    xi=np.linspace(0,1,4097)
    dense=0.
    for x in states:
        a=model['slope_polynomial_a']@x[:model['modes']]
        b=model['slope_polynomial_b']@x[:model['modes']]
        c=model['slope_polynomial_c']@x[:model['modes']]
        dense=max(dense,float(np.max(np.abs(a[:,None]*xi**2+b[:,None]*xi+c[:,None]))))
    return dict(exact_extrema=exact,dense_extrema=dense,
                exact_minus_dense=exact-dense,dense_grid_points_per_element=len(xi))


def main():
    start=time.perf_counter()
    with np.load(ROOT/'results/learning/data/seed_17.npz') as data:
        candidates=np.array(data['test_initials'])
    # Pick a deterministic substantial-energy test condition, without selecting
    # based on learned-model performance. The index is fixed before any solve.
    baseline=build_model(.03)
    index=int(np.argmax(energy(candidates,baseline)))
    initial=candidates[index:index+1]
    grids=np.linspace(0,16,8001)
    protocol=dict(duration_s=16.,output_dt_s=.002,solver='DOP853',rtol=1e-10,atol=1e-12,
        initial_selection='Highest original exact energy in seed17 test initials, independent of neural results.',
        seed=17,initial_index=index,modes_compared=[2,8],fields_T=[.03,.06,.12],
        scope='Finite-element modal-truncation sensitivity only. Not geometric/constitutive validation or a substitute for full FE convergence.')
    protocol_path=OUT/'verification_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8'))!=protocol:
        raise RuntimeError('Verification protocol changed')
    if not protocol_path.exists():save(protocol_path,protocol)
    results=[]
    for field in protocol['fields_T']:
        models={n:build_model(field,modes=n) for n in (2,8)}
        checks={};trajectories={};initial_diags={}
        for n,model in models.items():
            x0,projection=project_initials(initial,model,baseline)
            t0=time.perf_counter()
            solution=solve_ivp(rhs_si(model),(0,16),x0[0],method='DOP853',t_eval=grids,rtol=1e-10,atol=1e-12)
            if not solution.success:raise RuntimeError(solution.message)
            states=solution.y.T
            e=energy(states,model)
            mass_diag=np.max(np.abs(model['phi'].T@model['mass']@model['phi']-np.eye(n)))
            kdiag=np.max(np.abs(model['phi'].T@model['linear_stiffness']@model['phi']-np.diag(model['omega']**2)))/max(model['omega']**2)
            checks[n]=dict(projection=projection,geometry=geometry_checks(states,model),
                continuous_energy_screen=energy_geometry_bound(x0,model),
                energy_relative_max_drift=float(np.max(np.abs(e-e[0]))/max(abs(e[0]),1e-30)),
                nonlinear=nonlinear_checks(states,model),mass_orthonormality_max_error=float(mass_diag),
                relative_stiffness_diagonalization_error=float(kdiag),
                nfev=solution.nfev,elapsed_s=time.perf_counter()-t0)
            trajectories[n]=states;initial_diags[n]=x0
        m2=models[2];m8=models[8];s2=trajectories[2];s8=trajectories[8]
        q2=(s2[:,:2]*m2['modal_scale'])@m2['phi'].T
        q8=(s8[:,:8]*m8['modal_scale'])@m8['phi'].T
        v2=(s2[:,2:]*m2['modal_scale']*m2['omega'])@m2['phi'].T
        v8=(s8[:,8:]*m8['modal_scale']*m8['omega'])@m8['phi'].T
        def mass_rel_rmse(a,b):
            num=np.einsum('ti,ij,tj->',a-b,m2['mass'],a-b)
            den=np.einsum('ti,ij,tj->',b,m2['mass'],b)
            return float(np.sqrt(max(num,0)/max(den,1e-30)))
        linear_modal_energy=.5*m8['nu']*(s8[:,:8]**2+s8[:,8:]**2)
        high_fraction=np.sum(linear_modal_energy[:,2:],axis=-1)/np.maximum(np.sum(linear_modal_energy,axis=-1),1e-30)
        entry=dict(field_T=field,models={str(k):v for k,v in checks.items()},
            two_vs_eight=dict(tip_rmse_m=float(np.sqrt(np.mean((q2[:,-2]-q8[:,-2])**2))),
                full_FE_mass_displacement_relative_rms=mass_rel_rmse(q2,q8),
                full_FE_mass_velocity_relative_rms=mass_rel_rmse(v2,v8),
                highmode_linear_energy_fraction_max=float(high_fraction.max()),
                highmode_linear_energy_fraction_mean=float(high_fraction.mean())),
            hermite_extrema_check=check_hermite_extrema(m8))
        results.append(entry)
        path=OUT/f'verify_B{round(field*1000):03d}mT_2v8.npz'
        np.savez_compressed(path,time_s=grids,states_2=s2,states_8=s8,
                            initial_original=initial,initial_2=initial_diags[2],initial_8=initial_diags[8])
        print(json.dumps(dict(field_T=field,comparison=entry['two_vs_eight'],
            force_checks=checks[2]['nonlinear'],geometry_8=checks[8]['geometry']),ensure_ascii=False),flush=True)
    result=dict(protocol=protocol,protocol_sha256=sha(protocol_path),levels=results,elapsed_s=time.perf_counter()-start)
    save(OUT/'verification.json',result)


if __name__=='__main__':
    main()
