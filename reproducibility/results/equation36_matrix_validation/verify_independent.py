"""Independent energy/autograd and saved-trajectory audit; no experiment import."""
from pathlib import Path
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import hashlib
import json
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.linalg import expm
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
result=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
cases={c['name']:c for c in protocol['cases']}

def scalar_energy(x,c):
    n=c['N'];q=x[:n];p=x[n:];m=torch.tensor(c['M0'])+c['alpha']*torch.outer(q,q)
    return .5*p@torch.linalg.solve(m,p)+.5*q@torch.tensor(c['K0'])@q+c['beta']/4*(q@q)**2

def exact_field(x,c):
    # Explicit scalar energy differentiation is independent of the implementation's d(M^-1)/dq.
    t=torch.tensor(x,requires_grad=True);g=torch.autograd.grad(scalar_energy(t,c),t)[0].detach().numpy();n=c['N']
    return np.r_[g[n:],-g[:n]]

reference={}
max_error_gap=0.;max_label_gap=0.;max_tangent_gap=0.;max_ref_gap=0.;n_labels=0
inputs={};by_initial={}
for record in result['records']:
    c=cases[record['case']];n=c['N'];p=ROOT/record['trajectory_path'];assert sha(p)==record['trajectory_sha256'];inputs[record['trajectory_path']]=sha(p)
    a=np.load(p,allow_pickle=False);states=a['states'];initial=a['initial'];h=float(a['h']);T=protocol['horizon'];key=(record['case'],record['initial_index'])
    assert states.shape==(round(T/h)+1,2*n);assert a['mass0'].shape==(n,n)
    j=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])
    if key not in reference:
        if not c['alpha'] and not c['beta']:
            H=np.block([[np.array(c['K0']),np.zeros((n,n))],[np.zeros((n,n)),np.linalg.inv(c['M0'])]])
            terminal=expm(T*j@H)@initial
        else:
            sol=solve_ivp(lambda t,x:exact_field(x,c),(0,T),initial,method='DOP853',rtol=5e-14,atol=5e-16)
            assert sol.success;terminal=sol.y[:,-1]
        reference[key]=terminal
    max_ref_gap=max(max_ref_gap,float(np.max(np.abs(reference[key]-a['reference_terminal']))))
    err=float(np.linalg.norm(states[-1]-reference[key]));max_error_gap=max(max_error_gap,abs(err-record['endpoint_error']))
    by_initial.setdefault(key,[]).append((h,err))
    # All trajectories: include beginning, midpoint and final implicit steps.
    for index in sorted({0,(len(states)-2)//2,len(states)-2}):
        midpoint=(states[index]+states[index+1])/2
        gap=np.max(np.abs((states[index+1]-states[index])/h-exact_field(midpoint,c)))
        max_label_gap=max(max_label_gap,float(gap));n_labels+=1
    middle=torch.tensor((states[0]+states[1])/2,requires_grad=True)
    hessian=torch.autograd.functional.hessian(lambda x:scalar_energy(x,c),middle).detach().numpy()
    tangent=np.linalg.solve(np.eye(2*n)-h*j@hessian/2,np.eye(2*n)+h*j@hessian/2)
    max_tangent_gap=max(max_tangent_gap,float(np.max(np.abs(tangent-a['first_step_tangent']))))
    defect=float(np.linalg.norm(tangent.T@j@tangent-j,ord=np.inf))
    assert defect<1e-12

orders=[]
for key,values in by_initial.items():
    values=sorted(values,reverse=True)
    order=np.log2(values[-2][1]/values[-1][1]);orders.append(float(order));assert order>1.9

assert max_error_gap<1e-10
assert max_ref_gap<1e-10
assert max_label_gap<2e-11
assert max_tangent_gap<2e-12
for rel,digest in protocol['source_sha256'].items():assert sha(ROOT/rel)==digest

# Fully independent block midpoint map exposes incorrect matrix arithmetic.
c=cases['coupled_constant'];m=np.array(c['M0']);k=np.array(c['K0']);a=np.linalg.inv(m);h=.3;old=np.array([.7,-.3,.2,.4])
g=np.block([[np.zeros((2,2)),a],[-k,np.zeros((2,2))]])
correct=np.linalg.solve(np.eye(4)-h*g/2,(np.eye(4)+h*g/2)@old)
epsilon=h*h*k@a/4;rhs=(np.eye(2)-epsilon)@old[2:]-h*k@old[:2]
right=rhs@np.linalg.inv(np.eye(2)+epsilon)
badA=1/m;badE=h*h*k@badA/4
elementwise=np.linalg.solve(np.eye(2)+badE,(np.eye(2)-badE)@old[2:]-h*k@old[:2])
assert np.linalg.norm(right-correct[2:],np.inf)>1e-5
assert np.linalg.norm(elementwise-correct[2:],np.inf)>1e-3

report=dict(status='passed',trajectories_checked=len(result['records']),independent_reference_cases=len(reference),
    energy_autograd_label_steps_checked=n_labels,minimum_independently_recomputed_finest_order=min(orders),
    max_reference_terminal_gap=max_ref_gap,max_endpoint_metric_gap=max_error_gap,
    max_midpoint_label_autograd_gap=max_label_gap,max_implicit_tangent_autograd_gap=max_tangent_gap,
    wrong_right_inverse_gap=float(np.linalg.norm(right-correct[2:],np.inf)),wrong_elementwise_inverse_gap=float(np.linalg.norm(elementwise-correct[2:],np.inf)),
    input_sha256=inputs,protocol_sha256=sha(OUT/'protocol.json'),
    limits='Finite smooth conservative benchmarks; no arbitrary damping, feedback, constitutive mass identification or global theorem claim.')
(OUT/'independent_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))
