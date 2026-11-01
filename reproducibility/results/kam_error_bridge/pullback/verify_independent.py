"""Independent autograd and derivative-envelope check of new (q,p) audit."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from scipy.stats import qmc

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=read(HERE/'protocol.json');summary=read(HERE/'summary.json')
d=np.array(read(ROOT/'results/learning/inputs/reference_model_spec.json')['slope_map_d'])
inputs={rel:sha(ROOT/rel) for rel in cfg['inputs']}
for rel,digest in cfg['inputs'].items():
    if inputs[rel]!=digest:raise AssertionError('Source input changed: '+rel)
def grid(axis):return np.stack(np.meshgrid(*([axis]*4),indexing='ij'),axis=-1).reshape(-1,4)
h=.5/16
points=[grid(np.linspace(-.25,.25,17)),grid(-.25+(np.arange(16)+.5)*h),
        (qmc.Sobol(4,scramble=True,seed=cfg['sobol_seed']).random_base2(16)*2-1)*.25]

def independent(points,w):
    tt={k:torch.tensor(w[k]) for k in ['w1','w2','w3','b1','b2','b3','scale','energy_scale']}
    values=[];gradients=[]
    for i in range(0,len(points),4096):
        x=torch.tensor(points[i:i+4096],requires_grad=True)
        a=torch.tanh((x/tt['scale'])@tt['w1'].T+tt['b1'])
        a=torch.tanh(a@tt['w2'].T+tt['b2'])
        f=x[:,:2]+x[:,:2]**3
        theta=f@torch.tensor(d)
        e=tt['energy_scale']*(a@tt['w3']+tt['b3'][0])-(2*torch.sin(theta/2)**2-theta**2/2)
        g=torch.autograd.grad(e.sum(),x)[0]
        values.append(e.detach().numpy());gradients.append(g.numpy())
    return np.concatenate(values),np.concatenate(gradients)

checks=[];max_gap=0.
for row in summary['records']:
    seed,g=row['seed'],row['group']
    wp=ROOT/f'results/variable_mass_learning/seed_{seed}/{g}_weights.npz'
    with np.load(wp,allow_pickle=False) as a:w={k:a[k] for k in a.files}
    evaluations=[independent(p,w) for p in points]
    gauge=float(evaluations[1][0].mean())
    for label,(value,gradient) in zip(['endpoint_grid','midpoint_grid','independent_sobol'],evaluations):
        n6=float(np.mean(np.sum(gradient**6,axis=1))**(1/6))
        c0=float(np.max(np.abs(value-gauge)))
        for x,y in [(n6,row[label]['gradient_lp6_normalized']),(c0,row[label]['energy_abs_max_shifted'])]:
            gap=abs(x-y);max_gap=max(max_gap,gap)
            if gap>1e-12:raise AssertionError((seed,g,label,x,y))
    aa=np.abs(w['w1'])/w['scale'][None,:];ww=np.abs(w['w2']);vv=np.abs(w['w3'])
    hh=np.zeros((4,4));t=4/(3*np.sqrt(3));esc=abs(float(w['energy_scale']))
    for j in range(len(vv)):
        gg=ww[j]@aa
        hh += esc*vv[j]*t*np.outer(gg,gg)
        for i in range(len(aa)):hh += esc*vv[j]*ww[j,i]*t*np.outer(aa[i],aa[i])
    theta=(.25+.25**3)*sum(abs(d));s=1+3*.25**2
    hh[:2,:2] += (1-np.cos(theta))*np.outer(abs(d)*s,abs(d)*s)
    for j in range(2):hh[j,j] += (theta-np.sin(theta))*abs(d[j])*6*.25
    if np.max(np.abs(hh-np.array(row['hessian_component_abs_bound'])))>1e-12:
        raise AssertionError('Composed Hessian bound mismatch')
    ev,eg=evaluations[0]
    envelope=float(np.max(np.abs(ev-gauge))+h/2*np.max(np.sum(abs(eg),axis=1))+h*h/8*hh.sum()+h*h/24*np.trace(hh))
    gap=abs(envelope-row['C0_envelope']);max_gap=max(max_gap,gap)
    if gap>1e-12:raise AssertionError('Pullback energy envelope mismatch')
    checks.append(dict(seed=seed,group=g,passed=True,C0_envelope_difference=gap))

for rel,digest in inputs.items():
    if sha(ROOT/rel)!=digest:raise AssertionError('Audit input changed during calculation')
out=dict(status='passed',models=10,independently_evaluated_points_per_model=sum(map(len,points)),
    maximum_metric_or_envelope_difference=max_gap,checks=checks,input_hashes=inputs,
    annulus_pullback_coordinate_abs_bound=.2*(1+3*.2**2),annulus_contained_in_new_box=True,
    includes_composed_potential_second_derivative_term=True,interval_certificate=False,KAM_certificate=False)
(HERE/'independent_verification.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ['checks','input_hashes']},indent=2))
