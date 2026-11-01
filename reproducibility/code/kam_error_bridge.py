"""Quantify sampled learning errors and conservative real-arithmetic envelopes.

No KAM certificate is claimed: C0 bounds are not a substitute for the norm,
nonresonance, and constants required by a specified KAM theorem. All outputs
are new and all old weights/data are read-only.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
sys.dont_write_bytecode = True
import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/kam_error_bridge'
SEEDS = [17, 43, 91, 137, 211]
GROUPS = ['GR', 'GC']
HALFWIDTH = .25
CELLS = 16
SOBOL_POWER = 16
SOBOL_SEED = 610202611
PMAX = 6
IMIN, IMAX = .0008, .02

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: np.array(archive[k], copy=True) for k in archive.files}

def input_files():
    paths = [ROOT/'results/learning/inputs/reference_model_spec.json']
    for seed in SEEDS:
        paths.append(ROOT/f'results/learning/data/seed_{seed}.npz')
        paths += [ROOT/f'results/neural_learning_10000/seed_{seed}/{g}_weights.npz' for g in GROUPS]
    return paths

def preflight():
    records = []
    maximum = 0.
    for seed in SEEDS:
        data = load(ROOT/f'results/learning/data/seed_{seed}.npz')
        for key, value in data.items():
            if (any(s in key for s in ('midpoints','initials','states')) and value.ndim>=2
                    and value.shape[-1]==4 and value.dtype.kind in 'if'):
                v = float(np.max(np.abs(value)))
                maximum = max(maximum, v)
                records.append(dict(seed=seed, key=key, maximum_abs_state=v))
    if maximum > HALFWIDTH:
        raise RuntimeError('Predeclared box does not cover existing saved learning states; no points are removed')
    return dict(saved_state_arrays=records, maximum_abs_saved_state=maximum,
                box_halfwidth=HALFWIDTH, passed=True,
                scope='Coverage of saved teacher/reference states and midpoint inputs, not a proof of orbit enclosure between samples')

def plan():
    config = dict(schema_version=1, study='KAM quantitative error bridge without torus certification',
        seeds=SEEDS, groups=GROUPS, box=[[-HALFWIDTH,HALFWIDTH]]*4,
        endpoint_grid_nodes_per_axis=CELLS+1, midpoint_cells_per_axis=CELLS,
        sobol_power=SOBOL_POWER, sobol_seed=SOBOL_SEED, gradient_p=PMAX,
        normalized_norm='(mean_points sum_components abs(gradient_error)^p)^(1/p)',
        lebesgue_norm='box_volume^(1/p) times normalized norm',
        energy_gauge='Composite-midpoint mean of HNN-H; exact domain mean differs by the reported analytic quadrature envelope',
        envelope='Real-arithmetic absolute-weight Hessian plus nearest-grid Taylor bounds. Floating computations do not use outward-rounded intervals; no machine-certified enclosure.',
        integrable_reference='K(I)=nu.I-(d1^4 I1^2+4d1^2d2^2 I1I2+d2^4 I2^2)/16',
        action_domain=[[IMIN,IMAX]]*2, finite_resonance_l1_order=32,
        physical_perturbation='Full cosine residual minus quartic angle average, retaining oscillatory quartic terms',
        cosine_remainder='Only Taylor remainder after the full angle-dependent quartic, not a full normal-form remainder',
        quadratic_discretization='Exact midpoint modified frequency for H0 only; not a full nonlinear modified-Hamiltonian bound',
        missing_for_KAM=['A specified theorem and its required analytic/high-order norm on an action-angle domain',
                         'Corresponding bounds for physical+learning+nonlinear discretization perturbations',
                         'Threshold and chart/small-divisor constants; full nonresonance conditions',
                         'Chen Theorem11 target-integrability/generalization assumptions for direct probabilistic application'],
        training_changed=False, sample_selection='No training or model/checkpoint selection; all ten saved models reported',
        source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in input_files()})
    config['preflight']=preflight()
    p=OUT/'protocol.json'
    if p.exists():
        if json.loads(p.read_text(encoding='utf-8'))!=config:
            raise RuntimeError('Existing protocol differs; preserve it instead of silently changing the study')
    else:
        save(p,config)
    return config

def grid(points_per_axis, midpoint=False):
    spacing=2*HALFWIDTH/CELLS
    axis=(-HALFWIDTH+(np.arange(CELLS)+.5)*spacing if midpoint
          else np.linspace(-HALFWIDTH,HALFWIDTH,points_per_axis))
    return np.stack(np.meshgrid(axis,axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,4)

def residual_energy_gradient(points, weights):
    """Network residual alone; prior cancels exactly against physical H0."""
    if not bool(weights['prior']):
        raise ValueError('This audit is declared for G-prior models')
    first=np.tanh((points/weights['scale'])@weights['w1'].T+weights['b1'])
    second=np.tanh(first@weights['w2'].T+weights['b2'])
    scale=float(weights['energy_scale'])
    value=scale*(second@weights['w3']+weights['b3'][0])
    gradient=scale*((weights['w3']*(1-second**2))@weights['w2']*(1-first**2))@weights['w1']/weights['scale']
    return value,gradient

def true_residual(points,d):
    theta=points[:,:2]@d
    value=2*np.sin(theta/2)**2-theta**2/2
    gradient=np.c_[(np.sin(theta)-theta)[:,None]*d,np.zeros((len(points),2))]
    return value,gradient

def errors(points,weights,d,batch=4096):
    values,gradients=[],[]
    for start in range(0,len(points),batch):
        p=points[start:start+batch]
        value,gradient=residual_energy_gradient(p,weights)
        true,truth_grad=true_residual(p,d)
        values.append(value-true)
        gradients.append(gradient-truth_grad)
    return np.concatenate(values),np.concatenate(gradients)

def hessian_envelope(weights,d):
    # tanh' <=1 and |tanh''| <=4/(3sqrt3), valid globally on real inputs.
    a=np.abs(weights['w1'])/weights['scale'][None,:]
    b=np.abs(weights['w2'])@a
    v=np.abs(weights['w3'])
    tmax=4/(3*np.sqrt(3))
    bound=abs(float(weights['energy_scale']))*tmax*(b.T@(v[:,None]*b)+a.T@((v@np.abs(weights['w2']))[:,None]*a))
    theta_max=HALFWIDTH*np.sum(np.abs(d))
    physical=np.zeros((4,4))
    physical[:2,:2]=(1-np.cos(theta_max))*np.outer(np.abs(d),np.abs(d))
    return bound+physical

def metrics(value,gradient,gauge=0.):
    return dict(n_points=len(value), gradient_lp6_normalized=float(np.mean(np.sum(np.abs(gradient)**PMAX,axis=1))**(1/PMAX)),
        gradient_lp6_lebesgue=float((2*HALFWIDTH)**(4/PMAX)*np.mean(np.sum(np.abs(gradient)**PMAX,axis=1))**(1/PMAX)),
        gradient_rms=float(np.sqrt(np.mean(gradient**2))), gradient_component_abs_max=np.max(np.abs(gradient),axis=0).tolist(),
        energy_abs_max_shifted=float(np.max(np.abs(value-gauge))), energy_rms_shifted=float(np.sqrt(np.mean((value-gauge)**2))))

def verify_derivative(weights,d,hbound):
    points=np.random.default_rng(8819).uniform(-HALFWIDTH,HALFWIDTH,(32,4))
    value,gradient=residual_energy_gradient(points,weights)
    h=1e-5
    fder=[]
    hessian=[]
    for j in range(4):
        shift=np.zeros(4);shift[j]=h
        vp,gp=residual_energy_gradient(points+shift,weights)
        vm,gm=residual_energy_gradient(points-shift,weights)
        fder.append((vp-vm)/(2*h))
        hessian.append((gp-gm)/(2*h))
    finite=np.stack(fder,axis=-1)
    matrix=np.stack(hessian,axis=-1)
    # Physical residual Hessian bound is nonnegative, so it also bounds NN alone.
    if np.any(np.abs(matrix)>hbound[None,:,:]+1e-8):
        raise AssertionError('Sampled NN Hessian violates analytic absolute-weight bound')
    gap=float(np.max(np.abs(finite-gradient)))
    if gap>1e-8:
        raise AssertionError('Network gradient finite-difference mismatch')
    return dict(gradient_central_difference_max_gap=gap,probe_points=32,
                hessian_bound_probe_passed=True,
                scope='Sanity check of implementation, not a proof by sampling')

def reference_budget(nu,d,h0):
    d1,d2=d
    b4=-np.array([[d1**4/8,d1*d1*d2*d2/4],[d1*d1*d2*d2/4,d2**4/8]])
    a=IMAX
    theta=np.sqrt(2*a)*np.sum(np.abs(d))
    average_abs=(d1**4*a*a+4*d1*d1*d2*d2*a*a+d2**4*a*a)/16
    quartic_abs=theta**4/24
    # Vnl and the angle average are both nonpositive. The magnitude of their
    # difference is bounded by max of their separate magnitudes.
    full_physical_bound=max(quartic_abs,average_abs)
    iminmax=np.array([[IMIN,IMAX]])
    action=np.stack(np.meshgrid(np.linspace(IMIN,IMAX,9),np.linspace(IMIN,IMAX,9),indexing='ij'),axis=-1).reshape(-1,2)
    angle=np.stack(np.meshgrid(np.linspace(0,2*np.pi,33)[:-1],np.linspace(0,2*np.pi,33)[:-1],indexing='ij'),axis=-1).reshape(-1,2)
    max_phys=0.
    for i in action:
        x=np.sqrt(2*i)[None,:]*np.cos(angle)
        theta_points=x@d
        avg=-(d1**4*i[0]**2+4*d1*d1*d2*d2*i[0]*i[1]+d2**4*i[1]**2)/16
        phys=2*np.sin(theta_points/2)**2-theta_points**2/2-avg
        max_phys=max(max_phys,float(np.max(np.abs(phys))))
    resonances=[]
    for k1 in range(-32,33):
        for k2 in range(-32,33):
            if 0<abs(k1)+abs(k2)<=32:
                k=np.array([k1,k2]);linear=k@b4
                lo=float(k@nu+np.sum(np.where(linear>=0,linear*IMIN,linear*IMAX)))
                hi=float(k@nu+np.sum(np.where(linear>=0,linear*IMAX,linear*IMIN)))
                lower=0. if lo<=0<=hi else min(abs(lo),abs(hi))
                resonances.append(dict(k=k.tolist(),lower_bound=lower))
    finite=min(resonances,key=lambda row:row['lower_bound'])
    discrete=[]
    for factor in [1,2,4,8,16,32,64]:
        h=h0/factor
        modified=2*np.arctan(h*nu/2)/h
        difference=np.abs(modified-nu)
        discrete.append(dict(divisor=factor,h_tau=h,modified_quadratic_frequencies=modified.tolist(),
            frequency_abs_difference=difference.tolist(),quadratic_modified_energy_C0_raw_bound_box=float(HALFWIDTH**2*np.sum(difference)),
            scope='Exact modified quadratic H0 only; no bound for the full nonlinear numerical map'))
    return dict(action_domain=[[IMIN,IMAX]]*2, b4=b4.tolist(),determinant=float(np.linalg.det(b4)),
        eigenvalues=np.linalg.eigvalsh(b4).tolist(),minimum_singular_value=float(np.min(np.abs(np.linalg.eigvalsh(b4)))),
        reference_non_degeneracy='Exact algebraic property of the explicitly defined angle-average reference K(I); floating values are reproducible evaluations',
        angle_max_on_action_domain=float(theta),quartic_angle_average_abs_bound=float(average_abs),
        angle_dependent_quartic_abs_bound=float(quartic_abs),
        full_physical_perturbation_C0_real_arithmetic_bound=float(full_physical_bound),
        full_physical_perturbation_grid_abs_max=max_phys,
        cosine_Taylor_order6_energy_bound=float(theta**6/720),
        cosine_Taylor_order6_gradient_component_bound=(np.abs(d)*theta**5/120).tolist(),
        caution='The oscillatory quartic perturbation is retained; the Taylor order6 bound is not the full physical or normal-form remainder',
        finite_resonance_worst=finite,finite_resonance_order=32,
        finite_resonance_scope='Linear frequency interval over the full declared action rectangle, for this finite integer set only',
        discrete_quadratic_budget=discrete)

def run():
    config=plan()
    snapshot=OUT/'execution_script.py'
    if snapshot.exists() and sha(snapshot)!=sha(__file__):
        raise RuntimeError('Execution script differs from preserved snapshot')
    if not snapshot.exists(): snapshot.write_bytes(Path(__file__).read_bytes())
    source=json.loads((ROOT/'results/learning/inputs/reference_model_spec.json').read_text(encoding='utf-8-sig'))
    nu=np.array(source['nu']); d=np.array(source['slope_map_d'])
    h0=float(load(ROOT/'results/learning/data/seed_17.npz')['teacher_step_tau'])
    endpoints=grid(CELLS+1)
    midpoints=grid(CELLS,True)
    sobol=(qmc.Sobol(4,scramble=True,seed=SOBOL_SEED).random_base2(SOBOL_POWER)*2-1)*HALFWIDTH
    save(OUT/'environment.json',dict(utc_started=datetime.now(timezone.utc).isoformat(),python=sys.version,numpy=np.__version__,
        platform=platform.platform(),execution_script_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json')))
    budget=reference_budget(nu,d,h0)
    save(OUT/'reference_budget.json',budget)
    rows=[]
    for seed in SEEDS:
        for group in GROUPS:
            path=ROOT/f'results/neural_learning_10000/seed_{seed}/{group}_weights.npz'
            weights=load(path)
            hbound=hessian_envelope(weights,d)
            verification=verify_derivative(weights,d,hbound)
            ev,eg=errors(endpoints,weights,d)
            mv,mg=errors(midpoints,weights,d)
            sv,sg=errors(sobol,weights,d)
            gauge=float(np.mean(mv))
            spacing=2*HALFWIDTH/CELLS; radius=spacing/2
            mean_quadrature_bound=float(spacing**2*np.trace(hbound)/24)
            gradient_component_upper=np.max(np.abs(eg),axis=0)+radius*np.sum(hbound,axis=1)
            upper=float(np.max(np.abs(ev-gauge))+radius*np.max(np.sum(np.abs(eg),axis=1))
                        +.5*radius**2*np.sum(hbound)+mean_quadrature_bound)
            result=dict(seed=seed,group=group,weights_sha256=sha(path),box_halfwidth=HALFWIDTH,
                gradient_p=PMAX,discrete_mean_energy_shift=gauge,
                endpoint_grid=metrics(ev,eg,gauge),composite_midpoint_grid=metrics(mv,mg,gauge),
                independent_sobol=metrics(sv,sg,gauge),hessian_component_abs_bound=hbound.tolist(),
                exact_mean_minus_discrete_mean_real_arithmetic_bound=mean_quadrature_bound,
                gradient_component_domain_real_arithmetic_bound=gradient_component_upper.tolist(),
                gradient_Lp6_normalized_domain_real_arithmetic_bound=float(np.sum(gradient_component_upper**PMAX)**(1/PMAX)),
                zero_exact_mean_energy_domain_real_arithmetic_C0_bound=upper,
                physical_plus_learning_C0_bookkeeping_bound=upper+budget['full_physical_perturbation_C0_real_arithmetic_bound'],
                bound_status='Reproducible conservative envelope from real-arithmetic inequalities; not an outward-rounded interval certificate',
                KAM_threshold_epsilon_star=None,KAM_passed=None,
                required_next='Establish the specified KAM theorem norm, chart and threshold constants. A C0 bookkeeping bound alone is insufficient.',
                verification=verification)
            # Directly sampled checks are necessary sanity checks, not proofs.
            if max(result['endpoint_grid']['energy_abs_max_shifted'],result['independent_sobol']['energy_abs_max_shifted'])>upper:
                raise AssertionError('Sampled energy error exceeds proposed envelope')
            save(OUT/f'seed_{seed}_{group}.json',result)
            rows.append(result)
            print('AUDIT',seed,group,'Lp6',result['independent_sobol']['gradient_lp6_normalized'],
                  'sample_C0',result['independent_sobol']['energy_abs_max_shifted'],'envelope_C0',upper,flush=True)
    aggregates=[]
    for group in GROUPS:
        sub=[r for r in rows if r['group']==group]
        a=dict(group=group,n_seeds=len(sub))
        for key in ['gradient_lp6_normalized','energy_abs_max_shifted']:
            vals=np.array([r['independent_sobol'][key] for r in sub])
            a[key+'_mean']=float(vals.mean());a[key+'_sample_sd']=float(vals.std(ddof=1))
        for key in ['zero_exact_mean_energy_domain_real_arithmetic_C0_bound','physical_plus_learning_C0_bookkeeping_bound']:
            vals=np.array([r[key] for r in sub]);a[key+'_maximum_over_seeds']=float(vals.max())
            a[key+'_mean']=float(vals.mean())
        aggregates.append(a)
    changes=[rel for rel,digest in config['source_hashes'].items() if sha(ROOT/rel)!=digest]
    if changes: raise AssertionError('Input changed: '+str(changes))
    save(OUT/'summary.json',dict(aggregates=aggregates,records=rows,reference_budget=budget,
        source_protection=dict(files_checked=len(config['source_hashes']),all_unchanged=True),
        conclusions=dict(empiricalLp_audit=True,conservative_C0_envelope=True,interval_certified=False,
                         threshold_known=False,required_KAM_norm_bound=False,invariant_tori_certified=False)))
    with (OUT/'comparison.csv').open('w',encoding='utf-8',newline='') as stream:
        flat=[dict(seed=r['seed'],group=r['group'],gradient_lp6=r['independent_sobol']['gradient_lp6_normalized'],
            sampled_energy_error=r['independent_sobol']['energy_abs_max_shifted'],
            C0_envelope=r['zero_exact_mean_energy_domain_real_arithmetic_C0_bound'],
            total_C0_bookkeeping=r['physical_plus_learning_C0_bookkeeping_bound']) for r in rows]
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print('COMPLETE',json.dumps(aggregates),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['plan','run'],default='plan');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='plan':
        config=plan();print('PROTOCOL_SAVED',config['preflight']['maximum_abs_saved_state'])
    else: run()
