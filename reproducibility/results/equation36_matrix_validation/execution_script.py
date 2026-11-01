"""Equation (36): scalar nonunit mass and coupled N-by-N mass benchmarks.

Declared conservative numerical benchmarks only, not beam constitutive data.
All outputs are isolated under results/equation36_matrix_validation.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/equation36_matrix_validation'
HS=[.2,.1,.05,.025,.0125]
T=5.
CASES=[
 dict(name='scalar_m1',N=1,M0=[[1.]],K0=[[1.]],alpha=0.,beta=0.),
 dict(name='scalar_m2',N=1,M0=[[2.]],K0=[[1.]],alpha=0.,beta=0.),
 dict(name='scalar_m5',N=1,M0=[[5.]],K0=[[1.]],alpha=0.,beta=0.),
 dict(name='scalar_variable',N=1,M0=[[2.]],K0=[[1.]],alpha=.3,beta=.1),
 dict(name='coupled_constant',N=2,M0=[[2.,.3],[.3,1.5]],K0=[[1.7,.2],[.2,.9]],alpha=0.,beta=0.),
 dict(name='coupled_variable',N=2,M0=[[2.,.3],[.3,1.5]],K0=[[1.7,.2],[.2,.9]],alpha=.3,beta=.1),
]
INITIALS={1:[[.7,.2],[-.4,.5],[.2,-.6]],
          2:[[.7,-.3,.2,.4],[-.4,.5,.3,-.2],[.2,.1,-.6,.35]]}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def jac(fn,x):
    return np.column_stack([np.imag(fn(np.asarray(x,dtype=complex)+1e-25j*np.eye(len(x))[j]))/1e-25 for j in range(len(x))])

def matrices(q,case):
    n=case['N'];M=np.asarray(case['M0'])+case['alpha']*np.outer(q,q)
    A=np.linalg.solve(M,np.eye(n))
    eye=np.eye(n)
    da=np.array([-A@(case['alpha']*(np.outer(eye[i],q)+np.outer(q,eye[i])))@A for i in range(n)])
    K=np.asarray(case['K0'])+case['beta']*(q@q)*np.eye(n)
    return M,A,da,K

def energy(z,case):
    n=case['N'];q,p=z[:n],z[n:]
    _,a,_,_=matrices(q,case)
    return .5*p@a@p+.5*q@np.asarray(case['K0'])@q+case['beta']/4*(q@q)**2

def field(z,case):
    n=case['N'];q,p=z[:n],z[n:];_,a,da,k=matrices(q,case)
    delta=-.5*np.einsum('i,aij,j->a',p,da,p)
    return np.r_[a@p,delta-k@q]

def eq36_residual(new,old,h,case):
    n=case['N'];qm=(new[:n]+old[:n])/2;pm=(new[n:]+old[n:])/2
    _,a,da,k=matrices(qm,case)
    delta=-.5*np.einsum('i,aij,j->a',pm,da,pm)
    epsilon=h*h/4*k@a
    pnext=np.linalg.solve(np.eye(n)+epsilon,(np.eye(n)-epsilon)@old[n:]-h*k@old[:n]+h*delta)
    return np.r_[new[:n]-old[:n]-h/2*a@(new[n:]+old[n:]),new[n:]-pnext]

def midpoint_residual(new,old,h,case):
    return new-old-h*field((new+old)/2,case)

def step(old,h,case,residual=eq36_residual):
    new=old+h*field(old,case)
    for iteration in range(14):
        r=residual(new,old,h,case)
        if np.max(np.abs(r))<2e-14:
            return new,float(np.max(np.abs(r))),iteration+1
        new-=np.linalg.solve(jac(lambda z:residual(z,old,h,case),new),r)
    raise RuntimeError(f'Implicit solver failed: {case["name"]}, h={h}')

def plan():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(schema=1,scope='Declared mathematical benchmarks of Eq36; not calibrated beam mass laws',
        cases=CASES,initials=INITIALS,h_values=HS,horizon=T,
        state_order='q components then p components; mass M is N by N; phase space and J are2N',
        mass='M(q)=M0+alpha*q*q^T; scalar variable case is2+.3q^2',
        potential='V=.5q^TK0q+beta/4*(q^Tq)^2',
        coefficients='A=M^-1; K=K0+beta(q^Tq)I; Delta_i=-.5*p_mid^T*d_iA(q_mid)*p_mid',
        damping=0.,external_force=0.,matrix_order='epsilon=h^2*K*A/4; solve(I+epsilon,rhs) on the left',
        primary='Endpoint Euclidean error against expm/independent refined DOP853; step-halving order per fixed initial',
        geometry='Implicit Eq36 tangent via complex-step Jacobians; canonical symplectic defect',
        numerical_reference=dict(rtol=5e-13,atol=5e-15,refined_rtol=5e-14,refined_atol=5e-16),
        repetitions='Three fixed deterministic initial states; not independent stochastic seeds',
        wrong_implementations='Incorrect right inverse/order and elementwise mass reciprocal must disagree with independent block midpoint map',
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),ROOT/'manuscript/derivation_corrections.tex',OUT/'verify_independent.py']})
    p=OUT/'protocol.json'
    if p.exists() and json.loads(p.read_text(encoding='utf-8'))!=protocol:raise RuntimeError('Protocol differs; preserve prior files')
    if not p.exists():save(p,protocol)
    snapshot=OUT/'execution_script.py'
    if snapshot.exists() and sha(snapshot)!=sha(__file__):raise RuntimeError('Script differs from execution snapshot')
    if not snapshot.exists():shutil.copyfile(__file__,snapshot)
    return protocol

def reference(initial,case,tol):
    n=case['N']
    if not case['alpha'] and not case['beta']:
        A=np.linalg.solve(case['M0'],np.eye(n));k=np.asarray(case['K0']);g=np.block([[np.zeros((n,n)),A],[-k,np.zeros((n,n))]])
        return expm(T*g)@initial
    sol=solve_ivp(lambda t,z:field(z,case),(0,T),initial,method='DOP853',rtol=tol,atol=tol*.01)
    if not sol.success:raise RuntimeError(sol.message)
    return sol.y[:,-1]

def wrong_matrix_regression():
    c=CASES[4];n=2;m=np.array(c['M0']);k=np.array(c['K0']);a=np.linalg.solve(m,np.eye(n));h=.3
    old=np.array([.7,-.3,.2,.4]);eps=h*h*k@a/4;rhs=(np.eye(n)-eps)@old[n:]-h*k@old[:n]
    correct=np.linalg.solve(np.eye(n)+eps,rhs)
    wrong_right=rhs@np.linalg.solve(np.eye(n)+eps,np.eye(n))
    elementwise_a=1/m;bad_eps=h*h*k@elementwise_a/4
    wrong_elementwise=np.linalg.solve(np.eye(n)+bad_eps,(np.eye(n)-bad_eps)@old[n:]-h*k@old[:n])
    j=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])
    hessian=np.block([[k,np.zeros((n,n))],[np.zeros((n,n)),a]])
    midpoint=np.linalg.solve(np.eye(4)-h*j@hessian/2,(np.eye(4)+h*j@hessian/2)@old)
    checks=dict(commutator_norm=float(np.linalg.norm(m@k-k@m)),correct_momentum_gap=float(np.max(np.abs(correct-midpoint[n:]))),
        wrong_right_inverse_gap=float(np.max(np.abs(wrong_right-midpoint[n:]))),
        wrong_elementwise_reciprocal_gap=float(np.max(np.abs(wrong_elementwise-midpoint[n:]))))
    assert checks['commutator_norm']>.1
    assert checks['correct_momentum_gap']<1e-13
    assert checks['wrong_right_inverse_gap']>1e-5
    assert checks['wrong_elementwise_reciprocal_gap']>1e-3
    return checks

def run():
    plan();records=[];references=[];started=time.perf_counter()
    for c in CASES:
        n=c['N'];j=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])
        for index,initial in enumerate(np.array(INITIALS[n])):
            ref=reference(initial,c,5e-13);refined=reference(initial,c,5e-14)
            references.append(dict(case=c['name'],initial_index=index,initial=initial.tolist(),terminal=refined.tolist(),refinement_gap=float(np.max(np.abs(ref-refined)))))
            previous=None
            for h in HS:
                dest=OUT/'trajectories'/f'{c["name"]}_initial{index}_h{h:g}.npz'
                if dest.exists():raise RuntimeError('Existing trajectory; use summary/audit rather than overwrite')
                path=[initial.copy()];residual_max=0.;midpoint_max=0.;delta_max=0.;gap_max=0.;eig_min=float('inf');cond_max=0.;iters=0
                for _ in range(round(T/h)):
                    old=path[-1];new,res,iteration=step(old,h,c);path.append(new);residual_max=max(residual_max,res);iters=max(iters,iteration)
                    midpoint_max=max(midpoint_max,float(np.max(np.abs(midpoint_residual(new,old,h,c)))))
                    q=(new[:n]+old[:n])/2;p=(new[n:]+old[n:])/2;m,a,da,k=matrices(q,c)
                    eig=np.linalg.eigvalsh(m);eig_min=min(eig_min,float(eig.min()));cond_max=max(cond_max,float(eig.max()/eig.min()))
                    delta_max=max(delta_max,float(np.max(np.abs(-.5*np.einsum('i,aij,j->a',p,da,p)))))
                # An independently solved midpoint step and an implicit tangent at the first step.
                independent,_,_=step(initial,h,c,midpoint_residual)
                first=path[1];gap_max=float(np.max(np.abs(first-independent)))
                dn=jac(lambda z:eq36_residual(z,initial,h,c),first);do=jac(lambda z:eq36_residual(first,z,h,c),initial)
                tangent=-np.linalg.solve(dn,do);defect=float(np.linalg.norm(tangent.T@j@tangent-j,ord=np.inf))
                states=np.asarray(path);error=float(np.linalg.norm(states[-1]-refined));order=None if previous is None else float(np.log2(previous/error));previous=error
                energy0=energy(initial,c);drift=float(max(abs(energy(z,c)-energy0) for z in states))
                row=dict(case=c['name'],N=n,mass_shape=[n,n],phase_dimension=2*n,initial_index=index,h=h,steps=len(path)-1,horizon=T,
                    endpoint_error=error,observed_order=order,maximum_eq36_residual=residual_max,maximum_midpoint_residual=midpoint_max,
                    first_step_independent_gap=gap_max,first_step_symplectic_defect=defect,mass_min_eigenvalue=eig_min,mass_max_condition=cond_max,
                    maximum_delta=delta_max,maximum_energy_change=drift,maximum_newton_iterations=iters)
                dest.parent.mkdir(exist_ok=True)
                np.savez_compressed(dest,states=states,times=np.arange(len(states))*h,initial=initial,reference_terminal=refined,
                    first_step_tangent=tangent,N=np.array(n),h=np.array(h),mass0=np.asarray(c['M0']),stiffness0=np.asarray(c['K0']),alpha=np.array(c['alpha']),beta=np.array(c['beta']))
                row['trajectory_sha256']=sha(dest);row['trajectory_path']=dest.relative_to(ROOT).as_posix();records.append(row)
                print('CASE',c['name'],index,h,'err',error,'order',order,flush=True)
    assert len(records)==90
    assert all(r['observed_order']>1.9 for r in records if r['h']==HS[-1])
    assert max(r['first_step_symplectic_defect'] for r in records)<1e-11
    assert max(r['first_step_independent_gap'] for r in records)<1e-11
    save(OUT/'reference_checks.json',references)
    save(OUT/'results.json',dict(records=records,wrong_matrix_regression=wrong_matrix_regression(),elapsed_seconds=time.perf_counter()-started,protocol_sha256=sha(OUT/'protocol.json')))
    import csv
    with (OUT/'comparison.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    save(OUT/'environment.json',dict(utc=datetime.now(timezone.utc).isoformat(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),script_sha256=sha(__file__)))
    summarize()

def summarize():
    data=json.loads((OUT/'results.json').read_text(encoding='utf-8'));records=data['records'];summaries=[]
    for case in CASES:
        rows=[r for r in records if r['case']==case['name'] and r['h']==HS[-1]]
        summaries.append(dict(case=case['name'],N=case['N'],n_fixed_initials=len(rows),finest_h=HS[-1],
            maximum_endpoint_error=max(r['endpoint_error'] for r in rows),minimum_observed_order=min(r['observed_order'] for r in rows),
            maximum_symplectic_defect=max(r['first_step_symplectic_defect'] for r in rows),
            minimum_mass_eigenvalue=min(r['mass_min_eigenvalue'] for r in rows),maximum_mass_condition=max(r['mass_max_condition'] for r in rows),
            maximum_delta=max(r['maximum_delta'] for r in rows)))
    save(OUT/'summary.json',dict(cases=summaries,all_step_symplectic_defect_max=max(r['first_step_symplectic_defect'] for r in records),
        all_step_implicit_residual_max=max(r['maximum_eq36_residual'] for r in records),
        all_step_independent_gap_max=max(r['first_step_independent_gap'] for r in records),wrong_matrix_regression=data['wrong_matrix_regression']))
    labels={'scalar_m1':r'$N=1,\ m=1$','scalar_m2':r'$N=1,\ m=2$','scalar_m5':r'$N=1,\ m=5$',
            'scalar_variable':r'$N=1,\ m(q)$','coupled_constant':r'$N=2,\ M_0$','coupled_variable':r'$N=2,\ M(q)$'}
    lines=[r'\begin{table}[t]',r'\centering\small\setlength{\tabcolsep}{3pt}',
        r'\caption{式（36）的质量维度基准。$T=5$、最细步长$h=0.0125$；对每类三个固定初值报告最大终点误差、最小观察阶和最大一步辛缺陷。}',
        r'\label{tab:eq36_mass_dimension}',r'\begin{tabular}{lrrr}',r'\toprule',r'情形 & 状态误差 & 阶 & 辛缺陷\\',r'\midrule']
    for row in summaries:
        err=row['maximum_endpoint_error'];defect=row['maximum_symplectic_defect']
        def tex(value):
            a,b=f'{value:.2e}'.split('e');return f'{a}\\times10^{{{int(b)}}}'
        lines.append(labels[row['case']]+f" & ${tex(err)}$ & ${row['minimum_observed_order']:.4f}$ & ${tex(defect)}$ "+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
    (OUT/'manuscript_table.tex').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['plan','run','summary'],default='run');args=parser.parse_args()
    if args.stage=='plan':plan()
    elif args.stage=='run':run()
    else:summarize()
