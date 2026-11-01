"""Fixed-domain KAM error bookkeeping for the new variable-mass models.

The domain and energy prior belong to the new canonical (q,p) chart. They
must not be confused with the old physical-modal Cartesian audit.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import torch
from scipy.stats import qmc

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'pullback'
spec=importlib.util.spec_from_file_location('kam_error_base',ROOT/'code/kam_error_bridge.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
OUT.mkdir(parents=True,exist_ok=True)
source=json.loads((ROOT/'results/learning/inputs/reference_model_spec.json').read_text(encoding='utf-8-sig'))
d=np.array(source['slope_map_d'])
seeds=[17,43,91,137,211]
inputs=[ROOT/'code/variable_mass_learning.py',ROOT/'code/kam_error_bridge.py',Path(__file__),HERE/'reference_budget.json']
maximum=0.
for seed in seeds:
    dp=ROOT/f'results/variable_mass_learning/data/seed_{seed}.npz'
    inputs.append(dp)
    arrays=base.load(dp)
    for key,a in arrays.items():
        if any(t in key for t in ('midpoints','initials','states')) and a.ndim>=2 and a.shape[-1]==4:
            maximum=max(maximum,float(np.max(np.abs(a))))
    for g in ['GR','GC']: inputs.append(ROOT/f'results/variable_mass_learning/seed_{seed}/{g}_weights.npz')
if maximum>.25: raise RuntimeError('Declared new-coordinate box fails preflight; no observations are removed')
config=dict(domain=[[-.25,.25]]*4,coordinates='New canonical q1,q2,p1,p2',
    transformation='X=q+q^3, Y=p/(1+3q^2)',known_prior='Pullback .5sum nu*(f(q)^2+p^2/fprime(q)^2)',
    residual='Energy network residual minus Vnl(f(q)); known pullback prior cancels exactly',
    grid_nodes=17,midpoint_cells=16,sobol_power=16,sobol_seed=610202611,p=6,
    norm='(|D|^-1 integral sum_j abs(partial_j energy_error)^6)^(1/6)',
    preflight_max_abs_saved_state=maximum,
    derivative_bound='New composed-physical-potential Hessian bound, not reused from the old chart',
    KAM_certificate=False,interval_certificate=False,inputs={str(p.relative_to(ROOT)):base.sha(p) for p in inputs})
path=OUT/'protocol.json'
if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=config:
    raise RuntimeError('Existing pullback audit protocol differs')
if not path.exists():base.save(path,config)

def truth(points):
    q=points[:,:2];f=q+q**3;s=1+3*q*q;theta=f@d
    value=2*np.sin(theta/2)**2-theta**2/2
    gradient=np.c_[(np.sin(theta)-theta)[:,None]*s*d,np.zeros((len(points),2))]
    return value,gradient

def evaluate(points,w):
    e=[];g=[]
    for i in range(0,len(points),4096):
        p=points[i:i+4096]
        value,gradient=base.residual_energy_gradient(p,w)
        tv,tg=truth(p);e.append(value-tv);g.append(gradient-tg)
    return np.concatenate(e),np.concatenate(g)

def bound(w):
    # Remove the old-chart potential Hessian contribution, retaining NN bound.
    b=base.hessian_envelope(w,np.zeros(2))
    qmax=.25;fmax=qmax+qmax**3;smax=1+3*qmax*qmax;theta=fmax*np.sum(np.abs(d))
    b[:2,:2]+=(1-np.cos(theta))*np.outer(np.abs(d)*smax,np.abs(d)*smax)
    b[np.arange(2),np.arange(2)]+=(theta-np.sin(theta))*np.abs(d)*(6*qmax)
    return b

def torch_probe(w):
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    x=torch.tensor(np.random.default_rng(48119).uniform(-.25,.25,(1024,4)),requires_grad=True)
    t={k:torch.tensor(w[k]) for k in ['w1','b1','w2','b2','w3','b3','scale','energy_scale']}
    a=torch.tanh((x/t['scale'])@t['w1'].T+t['b1'])
    a=torch.tanh(a@t['w2'].T+t['b2'])
    f=x[:,:2]+x[:,:2]**3;theta=f@torch.tensor(d)
    err=t['energy_scale']*(a@t['w3']+t['b3'][0])-(2*torch.sin(theta/2)**2-theta**2/2)
    derivative=torch.autograd.grad(err.sum(),x)[0].detach().numpy()
    nv,ng=evaluate(x.detach().numpy(),w)
    gap=float(np.max(np.abs(derivative-ng)))
    if gap>1e-12: raise AssertionError('Pullback gradient autograd mismatch')
    return gap

endpoint=base.grid(17);mid=base.grid(16,True)
sobol=(qmc.Sobol(4,scramble=True,seed=610202611).random_base2(16)*2-1)*.25
physical_budget=json.loads((HERE/'reference_budget.json').read_text(encoding='utf-8'))['full_physical_perturbation_C0_real_arithmetic_bound']
rows=[]
for seed in seeds:
    for g in ['GR','GC']:
        w=base.load(ROOT/f'results/variable_mass_learning/seed_{seed}/{g}_weights.npz')
        if float(w['mass_alpha'])!=1.:raise AssertionError('Unexpected point transform')
        derivative_gap=torch_probe(w)
        ev,eg=evaluate(endpoint,w);mv,mg=evaluate(mid,w);sv,sg=evaluate(sobol,w)
        gauge=float(mv.mean());h=.5/16;b=bound(w)
        quadrature=float(h*h*np.trace(b)/24)
        upper=float(np.max(np.abs(ev-gauge))+h/2*np.max(np.sum(np.abs(eg),axis=1))+h*h/8*np.sum(b)+quadrature)
        row=dict(seed=seed,group=g,coordinates='new(q,p)',energy_mean_shift=gauge,
            endpoint_grid=base.metrics(ev,eg,gauge),midpoint_grid=base.metrics(mv,mg,gauge),
            independent_sobol=base.metrics(sv,sg,gauge),hessian_component_abs_bound=b.tolist(),
            mean_quadrature_bound=quadrature,C0_envelope=upper,physical_plus_learning_C0=upper+physical_budget,
            gradient_autograd_probe_gap=derivative_gap,interval_certified=False,KAM_certificate=False)
        if max(row['endpoint_grid']['energy_abs_max_shifted'],row['independent_sobol']['energy_abs_max_shifted'])>upper:
            raise AssertionError('Pullback C0 envelope sanity check failed')
        base.save(OUT/f'seed_{seed}_{g}.json',row);rows.append(row)
        print('PULLBACK',seed,g,'N6',row['independent_sobol']['gradient_lp6_normalized'],'C0',upper,flush=True)
aggregates=[]
for g in ['GR','GC']:
    sub=[r for r in rows if r['group']==g];a=dict(group=g,n_seeds=5)
    for key in ['gradient_lp6_normalized','energy_abs_max_shifted']:
        vals=np.array([r['independent_sobol'][key] for r in sub])
        a[key+'_mean']=float(vals.mean());a[key+'_sample_sd']=float(vals.std(ddof=1))
    a['C0_envelope_maximum_over_seeds']=max(r['C0_envelope'] for r in sub)
    a['physical_plus_learning_C0_maximum_over_seeds']=max(r['physical_plus_learning_C0'] for r in sub)
    aggregates.append(a)
changed=[rel for rel,digest in config['inputs'].items() if base.sha(ROOT/rel)!=digest]
if changed:raise AssertionError('Changed source inputs:'+str(changed))
base.save(OUT/'summary.json',dict(aggregates=aggregates,records=rows,source_inputs_unchanged=True,
    max_gradient_autograd_probe_gap=max(r['gradient_autograd_probe_gap'] for r in rows),
    scope='New-coordinate error budget for actual newly trained variable-mass G models; no torus or interval certificate'))
print(json.dumps(aggregates,indent=2),flush=True)
