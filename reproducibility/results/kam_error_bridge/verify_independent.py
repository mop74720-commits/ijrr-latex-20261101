"""Independent autograd recalculation of the fixed-domain audit.

Does not import the experiment module or its energy/gradient implementation.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import qmc

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def archive(p):
    with np.load(p,allow_pickle=False) as a: return {k:a[k] for k in a.files}

protocol=read(HERE/'protocol.json')
summary=read(HERE/'summary.json')
inputs=[ROOT/rel for rel in protocol['source_hashes']]
inputs += [HERE/'protocol.json',HERE/'summary.json',HERE/'reference_budget.json']
inputs += [HERE/f'seed_{seed}_{g}.json' for seed in protocol['seeds'] for g in protocol['groups']]
before={str(p.relative_to(ROOT)):sha(p) for p in inputs}
source=read(ROOT/'results/learning/inputs/reference_model_spec.json')
d=torch.tensor(source['slope_map_d'])
half=.25; cells=16; h=2*half/cells

def tensorgrid(axis): return np.stack(np.meshgrid(*([axis]*4),indexing='ij'),axis=-1).reshape(-1,4)
endpoint=tensorgrid(np.linspace(-half,half,17))
mid=tensorgrid(-half+(np.arange(cells)+.5)*h)
sobol=(qmc.Sobol(4,scramble=True,seed=protocol['sobol_seed']).random_base2(protocol['sobol_power'])*2-1)*half

def torch_error(points,w):
    values=[]; gradients=[]
    tt={key:torch.tensor(w[key]) for key in ['w1','b1','w2','b2','w3','b3','scale','energy_scale']}
    for i in range(0,len(points),4096):
        x=torch.tensor(points[i:i+4096],requires_grad=True)
        y=torch.tanh((x/tt['scale'])@tt['w1'].T+tt['b1'])
        y=torch.tanh(y@tt['w2'].T+tt['b2'])
        residual=tt['energy_scale']*(y@tt['w3']+tt['b3'][0])
        theta=x[:,:2]@d
        truth=2*torch.sin(theta/2)**2-theta**2/2
        error=residual-truth
        gradient=torch.autograd.grad(error.sum(),x)[0]
        values.append(error.detach().numpy());gradients.append(gradient.numpy())
    return np.concatenate(values),np.concatenate(gradients)

checks=[]
max_metric_gap=0.
for row in summary['records']:
    seed,g=row['seed'],row['group']
    w=archive(ROOT/f'results/neural_learning_10000/seed_{seed}/{g}_weights.npz')
    ev,eg=torch_error(endpoint,w);mv,mg=torch_error(mid,w);sv,sg=torch_error(sobol,w)
    gauge=mv.mean()
    for label,value,gradient in [('endpoint_grid',ev,eg),('composite_midpoint_grid',mv,mg),('independent_sobol',sv,sg)]:
        expected=row[label]
        n6=float(np.mean(np.sum(gradient**6,axis=1))**(1/6))
        c0=float(np.max(np.abs(value-gauge)))
        for actual,target in [(n6,expected['gradient_lp6_normalized']),(c0,expected['energy_abs_max_shifted'])]:
            gap=abs(actual-target);max_metric_gap=max(max_metric_gap,gap)
            if gap>1e-12: raise AssertionError((seed,g,label,actual,target))
        if abs(n6*(.5**4)**(1/6)-expected['gradient_lp6_lebesgue'])>1e-12:
            raise AssertionError('Lebesgue norm conversion mismatch')
    aa=np.abs(w['w1'])/w['scale'][None,:]
    bb=np.abs(w['w2'])
    vv=np.abs(w['w3'])
    bound=np.zeros((4,4))
    tmax=4/(3*np.sqrt(3))
    # Sum each output path explicitly rather than the matrix formula in main.
    for out in range(len(vv)):
        slope=bb[out]@aa
        bound += abs(float(w['energy_scale']))*vv[out]*tmax*np.outer(slope,slope)
        for first in range(len(aa)):
            bound += abs(float(w['energy_scale']))*vv[out]*bb[out,first]*tmax*np.outer(aa[first],aa[first])
    dn=np.array(source['slope_map_d'])
    bound[:2,:2] += (1-np.cos(half*sum(abs(dn))))*np.outer(abs(dn),abs(dn))
    bdiff=float(np.max(np.abs(bound-np.array(row['hessian_component_abs_bound']))))
    if bdiff>1e-12: raise AssertionError('Hessian bound path sum mismatch')
    c0upper=float(np.max(np.abs(ev-gauge))+h/2*np.max(np.sum(np.abs(eg),axis=1))
                  +h*h/8*np.sum(bound)+h*h/24*np.trace(bound))
    udiff=abs(c0upper-row['zero_exact_mean_energy_domain_real_arithmetic_C0_bound'])
    if udiff>1e-12: raise AssertionError('Uniform envelope mismatch')
    checks.append(dict(seed=seed,group=g,gradient_norm_and_energy_max_passed=True,
                       hessian_path_sum_max_gap=bdiff,uniform_envelope_gap=udiff))

budget=summary['reference_budget']
sing=float(np.linalg.svd(np.array(budget['b4']),compute_uv=False).min())
if abs(sing-budget['minimum_singular_value'])>1e-15: raise AssertionError('Reference twist mismatch')
corners=np.array([[i,j] for i in [.0008,.02] for j in [.0008,.02]])
frequency=np.array(source['nu'])[None,:]+corners@np.array(budget['b4']).T
min_value=float('inf')
for i in range(-32,33):
    for j in range(-32,33):
        if 0<abs(i)+abs(j)<=32:
            vals=frequency@np.array([i,j])
            lo,hi=vals.min(),vals.max()
            lower=0. if lo<=0<=hi else float(min(abs(lo),abs(hi)))
            min_value=min(min_value,lower)
if abs(min_value-budget['finite_resonance_worst']['lower_bound'])>1e-13: raise AssertionError('Finite resonance rectangle mismatch')

# A quadratic attains the composite-midpoint mean error coefficient exactly.
coef=np.array([1.,2.,3.,4.])
exact=float(sum(coef)*half*half/3)
quadmean=float(np.mean(mid**2@coef))
quadrature_formula=h*h*(2*sum(coef))/24
if abs((exact-quadmean)-quadrature_formula)>1e-14: raise AssertionError('Quadrature coefficient check failed')

changed=[rel for rel,digest in before.items() if sha(ROOT/rel)!=digest]
if changed: raise AssertionError(changed)
for rel,digest in protocol['source_hashes'].items():
    if sha(ROOT/rel)!=digest: raise AssertionError('Original source changed: '+rel)
result=dict(status='passed',models=10,independently_evaluated_points_per_model=len(endpoint)+len(mid)+len(sobol),
            maximum_metric_difference=max_metric_gap,checks=checks,finite_resonance_corner_check_passed=True,
            mean_quadrature_quadratic_equality_passed=True,source_inputs_unchanged=True,
            audited_input_hashes=before,interval_certified=False,KAM_certificate=False)
(HERE/'independent_verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['checks','audited_input_hashes']},indent=2))
