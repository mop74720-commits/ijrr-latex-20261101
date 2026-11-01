"""Full-cosine CHNN learning on the one clamped-free beam model (SI outputs)."""
from pathlib import Path
import csv,hashlib,json,time,sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from unified_model import MagneticBeam,PARAMETERS,HERE
from feature_model import (EXP,N2,TRANSFORM,SPECS,make_basis,gradients,decode_weights,
                           energy_gradient,twist)

RAW=HERE/"raw";FIG=HERE/"figures"
for p in [RAW,FIG]:p.mkdir(parents=True,exist_ok=True)
SEEDS=[17,43,91];ETA=1e-6;RIDGE=1e-14


def initial_conditions(seed,n):
    rng=np.random.default_rng(seed)
    radius=np.column_stack([rng.uniform(.03,.20,n),rng.uniform(.03,.16,n)])
    phase=rng.uniform(0,2*np.pi,(n,2))
    return np.c_[radius*np.cos(phase),-radius*np.sin(phase)]


def jfeatures(g):return g[:,[2,3,0,1],:]*np.array([1,1,-1,-1])[None,:,None]


def learned_field(tau,x,c,a,beam,gamma=0.,drive=None):
    h,g=energy_gradient(x,c,a,beam.d)
    fy=-g[:2]-gamma/beam.nu*g[2:]
    if drive is not None:
        u,_=beam.input(tau/beam.omega0,**drive)
        fy+=u*np.cos(beam.d@x[:2])*beam.d
    return np.r_[g[2:],fy]


def simulate_learned(x0,times,c,a,beam,gamma=0.,drive=None,audit=False):
    tau=np.asarray(times)*beam.omega0
    def rhs(t,s):
        x=s[:4];f=learned_field(t,x,c,a,beam,gamma,drive)
        if not audit:return f
        h,g=energy_gradient(x,c,a,beam.d)
        v=g[2:]
        u,du=beam.input(t/beam.omega0,**drive) if drive is not None else (0.,0.)
        input_power=u*np.cos(beam.d@x[:2])*(beam.d@v)
        diss=gamma*np.sum(v*v/beam.nu)
        return np.r_[f,input_power,diss]
    sol=solve_ivp(rhs,(tau[0],tau[-1]),np.r_[x0,0.,0.] if audit else x0,t_eval=tau,
                  method="DOP853",rtol=2e-10,atol=2e-12)
    if not sol.success:raise RuntimeError(sol.message)
    x=sol.y[:4].T
    if not audit:return x
    energy=np.array([energy_gradient(row,c,a,beam.d)[0] for row in x])
    return x,energy,sol.y[4],sol.y[5],energy-energy[0]-sol.y[4]+sol.y[5]


def physical_metrics(pred,ref,beam):
    qerr=(pred[:,:,:2]-ref[:,:,:2])@beam.tip_map_m
    serr=(pred[:,:,:2]-ref[:,:,:2])@beam.d
    eref=beam.energy(ref)*beam.energy_scale
    epred=beam.energy(pred)*beam.energy_scale
    tip_reference=ref[:,:,:2]@beam.tip_map_m
    return {"tip_rmse_mm":float(np.sqrt(np.mean(qerr*qerr))*1000),
            "tip_max_error_mm":float(np.max(np.abs(qerr))*1000),
            "slope_rmse_mrad":float(np.sqrt(np.mean(serr*serr))*1000),
            "slope_max_error_mrad":float(np.max(np.abs(serr))*1000),
            "true_energy_rmse_nJ":float(np.sqrt(np.mean((epred-eref)**2))*1e9),
            "tip_relative_l2":float(np.linalg.norm(qerr)/np.linalg.norm(tip_reference)),
            "state_relative_l2":float(np.linalg.norm(pred-ref)/np.linalg.norm(ref))}


def fit_labels(x,labels,beam,basis,prior,representation):
    dim=len(labels.ravel());start=N2 if prior else 0
    g=gradients(x,beam,basis,representation)
    design=jfeatures(g).reshape(-1,45)[:,start:]
    prior_c,prior_a=decode_weights(np.r_[0.,np.zeros(34)],True,"real",beam,basis)
    offset=np.array([learned_field(0,row,prior_c,0.,beam) for row in x]) if prior else np.zeros_like(labels)
    y=(labels-offset).ravel()
    mapping=(block_diag(TRANSFORM[:N2,:N2],np.eye(35)) if representation=="complex" else np.eye(45))[start:,start:]
    weights=np.linalg.lstsq(np.vstack([design/np.sqrt(dim),np.sqrt(RIDGE)*mapping]),
                            np.r_[y/np.sqrt(dim),np.zeros(45-start)],rcond=None)[0]
    coeff,cosine=decode_weights(weights,prior,representation,beam,basis)
    common=mapping@weights
    realdesign=jfeatures(gradients(x,beam,basis,"real")).reshape(-1,45)[:,start:]
    gram=realdesign.T@realdesign/dim
    loss=float(np.mean((realdesign@common-y)**2))
    eig=float(np.linalg.eigvalsh(gram)[0])
    target=np.zeros(45);target[N2]=basis["normalizer"]
    target[:N2]=prior_c[:N2]
    return coeff,cosine,weights,dict(noisy_train_mse=loss,gradient_gram_lambda_min=eig,
        coefficient_error=float(np.linalg.norm(common-target[start:])),
        max_complex_design_gap=float(np.max(np.abs(design-realdesign@mapping))),
        feature_rank=int(np.linalg.matrix_rank(design))),gram


def fit_conservative(beam,basis,test,test_times,test_x,test_field):
    allrows=[];fits={};sensitivity=[]
    train_times=np.linspace(0,4.,129)
    for seed in SEEDS:
        x0=initial_conditions(seed,12)
        states=np.array([beam.simulate(x,train_times) for x in x0])
        flat=states.reshape(-1,4)
        clean=np.array([beam.field(0,x) for x in flat])
        rng=np.random.default_rng(seed+10000);unit_noise=rng.uniform(-1,1,clean.shape)
        labels=clean+ETA*unit_noise
        np.savez_compressed(RAW/f"train_conservative_{seed}.npz",time_s=train_times,initial_states=x0,
                            states=states,clean_field=clean,noisy_field=labels,unit_noise=unit_noise)
        records={}
        for prior in [False,True]:
            for representation in ["real","complex"]:
                name=("GC" if representation=="complex" else "GR") if prior else ("BC" if representation=="complex" else "BR")
                c,a,w,stats,gram=fit_labels(flat,labels,beam,basis,prior,representation)
                predicted=np.array([simulate_learned(orbit[0],test_times,c,a,beam) for orbit in test])
                predfield=np.array([learned_field(0,x,c,a,beam) for x in test_x])
                metrics=physical_metrics(predicted,test,beam)
                noiseloss=float(np.mean((np.array([learned_field(0,x,c,a,beam) for x in flat])-clean)**2))
                tmat=twist(beam,basis,c,a)
                coefbound=(np.sqrt(stats["noisy_train_mse"])+ETA)/np.sqrt(stats["gradient_gram_lambda_min"])
                row=dict(seed=seed,model=name,prior=prior,representation=representation,
                         real_parameters=35 if prior else 45,cosine_multiplier=a,
                         heldout_field_relative_l2=float(np.linalg.norm(predfield-test_field)/np.linalg.norm(test_field)),
                         noiseless_train_mse=noiseloss,noise_component_bound=ETA,
                         coefficient_error_bound=coefbound,twist=tmat.tolist(),twist_det=float(np.linalg.det(tmat)),
                         twist_min_singular=float(np.linalg.svd(tmat,compute_uv=False)[-1]),
                         fixed_linear_chart=prior,**stats,**metrics)
                allrows.append(row);records[name]=(c,a,w,predicted)
                np.savez_compressed(RAW/f"fit_{seed}_{name}.npz",polynomial_coefficients=c,cosine_multiplier=a,
                                    readout_weights=w,predictions=predicted,gradient_gram=gram,
                                    effective_quartic=c[N2:]+a*basis["quartic"],twist=tmat,
                                    exponents=EXP,d=beam.d,nu=beam.nu,complement=basis["complement"],
                                    full_cosine_feature_normalizer=basis["normalizer"])
        fits[seed]=records
        for eta in [1e-7,1e-6,1e-5,1e-4]:
            c,a,w,stats,gram=fit_labels(flat,clean+eta*unit_noise,beam,basis,True,"complex")
            bound=(np.sqrt(stats["noisy_train_mse"])+eta)/np.sqrt(stats["gradient_gram_lambda_min"])
            tmat=twist(beam,basis,c,a);true_twist=twist(beam,basis,np.zeros(45),1.)
            sensitivity.append(dict(seed=seed,noise_bound=eta,cosine_multiplier=a,**stats,
                                    coefficient_bound=bound,twist_error_spectral=float(np.linalg.norm(tmat-true_twist,2)),
                                    twist_error_upper=np.sqrt(19)*np.linalg.norm(basis["taylor"],2)*bound,
                                    true_twist_sigma_min=float(np.linalg.svd(true_twist,compute_uv=False)[-1]),
                                    learned_twist_sigma_min=float(np.linalg.svd(tmat,compute_uv=False)[-1])))
        print(f"conservative seed {seed} complete",flush=True)
    return allrows,fits,sensitivity


def port_identification(beam,basis,fits):
    train_times=np.linspace(0,5.,161);test_times=np.linspace(0,8.,1601)
    test_initial=initial_conditions(202610071,6)
    drives=[dict(amplitude_T=a,frequency_ratio=f,phase=p) for a,f,p in
            [(0.004,.65,.15),(0.006,beam.drive_nu,.8),(0.008,.95,1.6),
             (0.006,.75,2.2),(0.008,1.05,3.5),(0.004,1.2,4.7)]]
    refs=np.array([beam.simulate(x,test_times,beam.gamma,drive) for x,drive in zip(test_initial,drives)])
    np.savez_compressed(RAW/"ports_reference.npz",time_s=test_times,states=refs,initial_states=test_initial,
                        drive_amplitude_T=[d["amplitude_T"] for d in drives],
                        drive_frequency_ratio=[d["frequency_ratio"] for d in drives],
                        drive_phase_rad=[d["phase"] for d in drives])
    rows=[]
    for seed in SEEDS:
        x0=initial_conditions(seed+500,8)
        train=np.array([beam.simulate(x,train_times,beam.gamma) for x in x0]);flat=train.reshape(-1,4)
        clean=np.array([beam.field(0,x,beam.gamma) for x in flat])
        noise=np.random.default_rng(seed+501).uniform(-ETA,ETA,clean.shape);labels=clean+noise
        np.savez_compressed(RAW/f"train_damping_{seed}.npz",time_s=train_times,states=train,
                            clean_field=clean,noisy_field=labels,noise=noise)
        c,a,w,_=fits[seed]["GC"]
        grad=np.array([energy_gradient(x,c,a,beam.d)[1] for x in flat])
        regressor=-grad[:,2:]/beam.nu
        residual=labels[:,2:]+grad[:,:2]
        gamma=max(0.,float(np.sum(regressor*residual)/np.sum(regressor**2)))
        predicted=[];energies=[];works=[];losses=[];balances=[]
        nodrag=[]
        for x,drive in zip(test_initial,drives):
            pred,en,work,loss,balance=simulate_learned(x,test_times,c,a,beam,gamma,drive,True)
            predicted.append(pred);energies.append(en);works.append(work);losses.append(loss);balances.append(balance)
            nodrag.append(simulate_learned(x,test_times,c,a,beam,0.,drive))
        predicted=np.array(predicted);nodrag=np.array(nodrag)
        metrics=physical_metrics(predicted,refs,beam);baseline=physical_metrics(nodrag,refs,beam)
        record=dict(seed=seed,model="GC with learned PSD Rayleigh head",gamma_true=beam.gamma,gamma_fitted=gamma,
                    alpha_true_s_inv=PARAMETERS["drag_mass_factor_s_inv"],alpha_fitted_s_inv=gamma*beam.omega0,
                    gamma_relative_error=abs(gamma/beam.gamma-1),damping_regression_sum_squares=float(np.sum(regressor**2)),
                    energy_balance_max_nJ=float(np.max(np.abs(balances))*beam.energy_scale*1e9),
                    max_tip_displacement_mm=float(np.max(np.abs(predicted[:,:,:2]@beam.tip_map_m))*1000),
                    without_drag_head=baseline,**metrics)
        rows.append(record)
        np.savez_compressed(RAW/f"ports_fitted_{seed}.npz",time_s=test_times,predictions=predicted,
                            without_drag=nodrag,energies_J=np.array(energies)*beam.energy_scale,
                            input_work_J=np.array(works)*beam.energy_scale,drag_loss_J=np.array(losses)*beam.energy_scale,
                            energy_balance_J=np.array(balances)*beam.energy_scale,gamma=gamma,
                            polynomial_coefficients=c,cosine_multiplier=a)
        print(f"damping/input seed {seed} complete",flush=True)
    return rows


def reduction_check():
    times=np.linspace(0,8.,1601);data={};rows=[]
    for n in [2,4,8,64]:
        beam=MagneticBeam(modes=n)
        states=beam.simulate(np.zeros(2*n),times,beam.gamma,{},rtol=2e-9,atol=2e-11)
        obs=beam.observables(states)
        data[n]=(states,obs)
        np.savez_compressed(RAW/f"reduction_{n}_modes.npz",time_s=times,states=states,**obs)
        print(f"modal reduction {n} complete",flush=True)
    ref=data[64][1]
    for n,(states,obs) in data.items():
        rows.append(dict(modes=n,tip_rmse_mm=float(np.sqrt(np.mean((obs["tip_m"]-ref["tip_m"])**2))*1000),
                         tip_max_error_mm=float(np.max(np.abs(obs["tip_m"]-ref["tip_m"]))*1000),
                         slope_max_error_mrad=float(np.max(np.abs(obs["tip_slope_rad"]-ref["tip_slope_rad"]))*1000),
                         tip_peak_mm=float(np.max(np.abs(obs["tip_m"]))*1000)))
    return rows


def main():
    start=time.perf_counter();beam=MagneticBeam();spec=beam.export(RAW);basis=make_basis(beam)
    np.savez_compressed(RAW/"feature_basis.npz",exponents=EXP,complex_transform=TRANSFORM,**basis)
    test_times=np.linspace(0,16.,1601);x0=initial_conditions(202610070,8)
    test=np.array([beam.simulate(x,test_times) for x in x0]);test_x=test[:,::10].reshape(-1,4)
    test_field=np.array([beam.field(0,x) for x in test_x])
    np.savez_compressed(RAW/"conservative_test.npz",time_s=test_times,initial_states=x0,states=test,
                        vector_field_states=test_x,clean_field=test_field)
    rows,fits,sensitivity=fit_conservative(beam,basis,test,test_times,test_x,test_field)
    ports=port_identification(beam,basis,fits)
    reduction=reduction_check()
    summary={}
    for name in ["BR","BC","GR","GC"]:
        selected=[r for r in rows if r["model"]==name];summary[name]={}
        for key in ["tip_rmse_mm","tip_max_error_mm","slope_rmse_mrad","true_energy_rmse_nJ",
                    "state_relative_l2","cosine_multiplier","heldout_field_relative_l2"]:
            vals=[r[key] for r in selected]
            summary[name][key]={"mean":float(np.mean(vals)),"sd":float(np.std(vals,ddof=1))}
    result=dict(provenance="New synthetic data generated by the declared Python consistent-mass FE/Galerkin model; not COMSOL training data.",
                model_spec=spec,protocol=dict(seeds=SEEDS,training_trajectories_per_seed=12,training_time_s=4.,
                training_samples_per_trajectory=129,test_trajectories=8,test_time_s=16.,test_samples_per_trajectory=1601,
                noise_component_bound_dimensionless=ETA,ridge=RIDGE,
                initial_radius_range_mode1=[.03,.20],initial_radius_range_mode2=[.03,.16],
                feature_class="one normalized full-cosine feature plus 34 quartic Taylor-complement features; optionally 10 quadratic features",
                conservative_target_in_class=True,
                chart_note="Only fixed-prior GR/GC preserve the declared linear chart; BR/BC twist entries alone are not Birkhoff certificates."),
                learning=rows,summary=summary,noise_sensitivity=sensitivity,ports=ports,modal_reduction=reduction,
                true_twist=twist(beam,basis,np.zeros(45),1.).tolist(),elapsed_seconds=time.perf_counter()-start)
    (HERE/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    for name,records in [("learning",rows),("noise_sensitivity",sensitivity),("ports",ports),("reduction",reduction)]:
        keys=[k for k,v in records[0].items() if not isinstance(v,(dict,list))]
        with (RAW/f"{name}.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore");writer.writeheader();writer.writerows(records)
    from plot_results import main as plot
    plot()
    print(json.dumps(dict(summary=summary,ports=ports,modal_reduction=reduction),indent=2))


if __name__=="__main__":main()
