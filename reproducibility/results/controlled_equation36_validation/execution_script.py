"""Separate forcing, sampling and integration errors of controlled Eq36.

Only writes results/controlled_equation36_validation. Existing experiments
are read-only. No fitting, gain selection, or test-outcome selection occurs.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
sys.dont_write_bytecode=True
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.rk import DOP853
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/controlled_equation36_validation'
MULTS=[4.,2.,1.,.5,.25]
SEEDS=[17,43,91,137,211]
POLICIES=['none','GR','GC']
METHODS=['I','II','III']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):
    with np.load(p,allow_pickle=False) as a: return {k:a[k] for k in a.files}
def module(name):
    sp=importlib.util.spec_from_file_location('cev_'+name,ROOT/'code'/f'{name}.py')
    m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);return m

ctrl=module('control_comparison')
lrc=module('learned_residual_control')
A=np.ascontiguousarray(DOP853.A)
B=np.ascontiguousarray(DOP853.B)

def plan():
    spec,initials,h0,inputs=lrc.inputs()
    bypath={v['path']:v for v in inputs}
    for name in ('learned_residual_control.py','controlled_equation36_validation.py'):
        p=ROOT/'code'/name;bypath[p.relative_to(ROOT).as_posix()]={'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)}
    steps=read(ROOT/'results/learned_residual_control/protocol.json')['base_steps']
    value=dict(schema_version=1,date='2026-11-01',scope='Same conservative two-mode beam; separate prescribed-force integration, actual sampled feedback, and frozen common-input integration.',
       dimensions={'N':2,'q':'R2','p':'R2','z':'R4','M':'diag(1/nu), 2x2','physical_input':'scalar magnetic field ratio u=By/B0','state_increment':'R4; kick=(0,h*u*d*cos(d.q))'},
       base_h_tau=h0,base_steps=steps,h_multipliers=MULTS,common_stride_h0=8,duration_seconds=steps*h0/spec['omega0_rad_s'],
       prescribed_initials='Zero plus first original test initial from seeds17,43,91; preselected without inspecting outcomes',
       held_feedback_initials='Zero for all original5 model seeds; none is one reused deterministic baseline',policies=POLICIES,methods=METHODS,
       generator_G='Shares I only after original320 independent generating-function equivalence checks; no duplicated statistical unit',
       branch_A='Smooth prescribed original u_ref(tau). Each discrete step samples u at its temporal midpoint. Independent continuous DOP853 flow.',
       branch_B='Actual state feedback sampled at start and held; compare to own continuous feedback flow. Reuse old stored zero-initial levels2,1,.5,.25, recompute errors against the current shared tight flow. Only4h0 newly simulated.',
       branch_C='For each h freeze u_k from the own continuous feedback trajectory sampled at t_k; all I/II/III share that same exogenous sequence. Compare to high-order continuous flow of exactly that held sequence. This branch is not an additional closed-loop implementation.',
       held_reference='Independent fixed-step DOP853 RK order8, two substeps per hold; four-substep refinement for every case. AdaptiveDOP853 and linear-expm single-step checks before running.',
       scalar_control={'N':1,'mass':2.,'stiffness':1.,'forcing':'.05*sin(.7*t)','h':[.2,.1,.05,.025,.0125],'horizon':5.,'initial':[.1,.2],'reference':'Augmented4x4 linear-expm exact forcing'},
       negative_control='Scalar post-step momentum increment .05*sin(.7*t_mid) without h, versus same fixed continuous forcing. Inconsistent example only, never proposed as controller. It can remain symplectic yet fail convergence.',
       structure='Fixed exogenous u map and full feedback Jacobian differentiated separately at zero and first test initial per model; local defect shrinking withh is not strict symplecticity at fixedh.',
       reporting='Endpoint and whole-path state/tip errors, h-halving observed orders, hold-reference refinement and all signs. No time-point pseudoreplication or model/gain selection.',
       inputs=list(bypath.values()))
    OUT.mkdir(parents=True,exist_ok=True);p=OUT/'protocol.json'
    if p.exists() and read(p)!=value:raise RuntimeError('Existing controlled convergence protocol changed')
    if not p.exists():save(p,value)
    snapshot=OUT/'execution_script.py'
    if snapshot.exists() and sha(snapshot)!=sha(__file__):raise RuntimeError('Execution source differs from saved script')
    if not snapshot.exists():snapshot.write_bytes(Path(__file__).read_bytes())
    return value,spec,initials

def verify_inputs(protocol):
    for v in protocol['inputs']:
        if sha(ROOT/v['path'])!=v['sha256']:raise RuntimeError('Changed input '+v['path'])

@njit(cache=False,nogil=True)
def reference_rhs(x,u,nu,d):
    result=np.empty(4);theta=d[0]*x[0]+d[1]*x[1]
    residual=math.sin(theta)-theta
    for j in range(2):
        result[j]=nu[j]*x[j+2]
        result[j+2]=-nu[j]*x[j]-d[j]*residual+u*d[j]*math.cos(theta)
    return result

@njit(cache=False,nogil=True)
def dop_step(x,dt,u,nu,d,a,b):
    stages=np.empty((12,4))
    for k in range(12):
        point=x.copy()
        for j in range(k):point+=dt*a[k,j]*stages[j]
        stages[k]=reference_rhs(point,u,nu,d)
    new=x.copy()
    for k in range(12):new+=dt*b[k]*stages[k]
    return new

@njit(cache=False,nogil=True)
def held_reference(initial,h,controls,stride,substeps,nu,d,a,b):
    states=np.empty((len(controls)//stride+1,4));states[0]=initial;x=initial.copy()
    for k in range(len(controls)):
        for sub in range(substeps):x=dop_step(x,h/substeps,controls[k],nu,d,a,b)
        if (k+1)%stride==0:states[(k+1)//stride]=x
    return states

@njit(cache=False,nogil=True)
def frozen_simulate(initial,h,controls,stride,method,nu,d):
    states=np.empty((len(controls)//stride+1,4));states[0]=initial;x=initial.copy();maxres=0.
    for k in range(len(controls)):
        x,res,_,_=ctrl.step(x,h,controls[k],method,nu,d);maxres=max(maxres,res)
        if (k+1)%stride==0:states[(k+1)//stride]=x
    return states,maxres

def error_metrics(states,reference,tip):
    diff=states-reference
    return dict(state_relative_l2=float(np.linalg.norm(diff)/max(np.linalg.norm(reference),1e-15)),
        state_absolute_rms=float(np.sqrt(np.mean(diff**2))),endpoint_state_l2=float(np.linalg.norm(diff[-1])),
        tip_rmse_mm=float(np.sqrt(np.mean((diff[:,:2]@tip*1000)**2))))

def reference_checks(protocol,spec,initials):
    nu=np.array(spec['nu']);d=np.array(spec['slope_map_d']);h0=protocol['base_h_tau'];rows=[]
    for h in (4*h0,h0/4):
        for j,x in enumerate(np.vstack([np.zeros(4),initials[[0,8,16]]])):
            for u in (-.17,0.,.11):
                fixed=dop_step(dop_step(x,h/2,u,nu,d,A,B),h/2,u,nu,d,A,B)
                sol=solve_ivp(lambda t,z:reference_rhs(z,u,nu,d),(0,h),x,method='DOP853',rtol=4e-14,atol=2e-16)
                gap=float(np.max(np.abs(fixed-sol.y[:,-1])))
                rows.append(dict(h=h,initial_index=j,u=u,adaptive_gap=gap))
    L=np.block([[np.zeros((2,2)),np.diag(nu)],[-np.diag(nu),np.zeros((2,2))]])
    gaplinear=0.
    for h in (4*h0,h0/4):
        x=initials[0]
        val=dop_step(dop_step(x,h/2,0.,nu,np.zeros(2),A,B),h/2,0.,nu,np.zeros(2),A,B)
        gaplinear=max(gaplinear,float(np.max(np.abs(val-expm(h*L)@x))))
    maxgap=max(v['adaptive_gap'] for v in rows)
    assert maxgap<2e-12 and gaplinear<2e-12,(maxgap,gaplinear)
    save(OUT/'reference_implementation_checks.json',dict(status='passed',adaptive_checks=rows,
       adaptive_max_gap=maxgap,linear_expm_max_gap=gaplinear,coefficient_source='Installed scipy.integrate._ivp.rk.DOP853',a=A.tolist(),b=B.tolist()))

def nominal_and_loops(protocol,spec):
    nu=np.array(spec['nu']);d=np.array(spec['slope_map_d']);w=spec['omega0_rad_s'];end=protocol['base_steps']*protocol['base_h_tau']
    def nominal_rhs(t,x):return np.r_[nu*x[2:],-nu*x[:2]+ctrl.feedforward(t,w)*d*np.cos(d@x[:2])]
    target=solve_ivp(nominal_rhs,(0,end),np.zeros(4),method='DOP853',rtol=4e-14,atol=2e-16,dense_output=True)
    if not target.success:raise RuntimeError(target.message)
    flows={}
    for policy in POLICIES:
        for seed in SEEDS:
            key=(seed,policy)
            if policy=='none' and seed!=SEEDS[0]:flows[key]=flows[(SEEDS[0],policy)];continue
            params=lrc.params_for(seed,policy);kind=0 if policy=='none' else 2
            sol=solve_ivp(lrc.closed_rhs,(0,end),np.zeros(4),args=(kind,params,target,nu,d,w),method='DOP853',rtol=4e-14,atol=2e-16,dense_output=True)
            if not sol.success:raise RuntimeError(sol.message)
            flows[key]=sol
    return target,flows

def branch_a(protocol,spec,initials):
    nu=np.array(spec['nu']);d=np.array(spec['slope_map_d']);w=spec['omega0_rad_s'];tip=np.array(spec['tip_map_m'])
    starts=np.vstack([np.zeros(4),initials[[0,8,16]]]);h0=protocol['base_h_tau'];base=protocol['base_steps']
    times=np.arange(base//8+1)*8*h0
    def rhs(t,flat):
        x=flat.reshape(-1,4);theta=x[:,:2]@d;u=ctrl.feedforward(t,w)
        return np.c_[x[:,2:]*nu,-x[:,:2]*nu-(np.sin(theta)-theta)[:,None]*d+u*np.cos(theta)[:,None]*d].ravel()
    solution=solve_ivp(rhs,(0,times[-1]),starts.ravel(),t_eval=times,method='DOP853',rtol=4e-14,atol=2e-16)
    if not solution.success:raise RuntimeError(solution.message)
    reference=solution.y.T.reshape(len(times),4,4)
    records=[]
    for mult in MULTS:
        h=h0*mult;nsteps=round(base/mult);stride=round(8/mult)
        for method_i,method in enumerate(METHODS):
            states,controls,diag=ctrl.rollout(starts,h,nsteps,stride,method_i,0,np.zeros((nsteps+1,4)),nu,d,w)
            p=OUT/'smooth_external'/f'hmult_{mult:g}_{method}.npz';p.parent.mkdir(exist_ok=True)
            np.savez_compressed(p,times_tau=times,states=states,reference=reference,controls=controls,initials=starts,tip_map_m=tip,diagnostics=diag)
            for j in range(4):records.append(dict(branch='smooth_external',h_multiplier=mult,method=method,initial_index=j,h_tau=h,
               trajectory_path=p.relative_to(ROOT).as_posix(),trajectory_sha256=sha(p),**error_metrics(states[:,j],reference[:,j],tip),maximum_midpoint_residual=float(diag[j,0])))
        print('SMOOTH_DONE',mult,flush=True)
    save(OUT/'smooth_external_records.json',records)

def branches_b_c(protocol,spec,initials):
    nu=np.array(spec['nu']);d=np.array(spec['slope_map_d']);w=spec['omega0_rad_s'];tip=np.array(spec['tip_map_m'])
    h0=protocol['base_h_tau'];base=protocol['base_steps'];times=np.arange(base//8+1)*8*h0
    target,flows=nominal_and_loops(protocol,spec);target_sample=target.sol(times).T
    records_b=[];records_c=[];refchecks=[];reused=[]
    for policy in POLICIES:
        seeds=[SEEDS[0]] if policy=='none' else SEEDS
        for seed in seeds:
            params=lrc.params_for(seed,policy);kind=0 if policy=='none' else 2;continuous=flows[(seed,policy)].sol(times).T
            for mult in MULTS:
                h=h0*mult;nsteps=round(base/mult);stride=round(8/mult);tt=np.arange(nsteps+1)*h
                for method_i,method in enumerate(METHODS):
                    if mult!=4.:
                        source=ROOT/f'results/learned_residual_control/zero_initial/hdiv_{1/mult:g}/seed_{seed}/{policy}_{method}.npz'
                        old=load(source);states=old['states'][:,0];controls=old['controls'][:,0];diag=old['diagnostics'][0]
                        assert np.max(np.abs(old['times_tau']-times))<1e-11
                        reference_gap=float(np.max(np.abs(old['reference']-target_sample)))
                        if reference_gap>1e-9:raise RuntimeError('Reused target mismatch')
                        sourcepath=source.relative_to(ROOT).as_posix();reused.append(dict(path=sourcepath,sha256=sha(source)))
                    else:
                        states_many,controls_many,ff,diagnostics=lrc.simulate(np.zeros((1,4)),h,nsteps,stride,method_i,kind,np.ascontiguousarray(target.sol(tt).T),nu,d,w,params)
                        states=states_many[:,0];controls=controls_many[:,0];diag=diagnostics[0];reference_gap=0.
                        source=OUT/'sampled_feedback'/f'seed_{seed}_{policy}_hmult4_{method}.npz';source.parent.mkdir(exist_ok=True)
                        np.savez_compressed(source,times_tau=times,states=states,continuous_reference=continuous,controls=controls,tip_map_m=tip)
                        sourcepath=source.relative_to(ROOT).as_posix()
                    records_b.append(dict(branch='sampled_feedback',seed=seed,policy=policy,h_multiplier=mult,method=method,h_tau=h,
                        trajectory_path=sourcepath,trajectory_sha256=sha(source),reused=(mult!=4.),
                        maximum_midpoint_residual=float(diag[0]),target_reference_recompute_gap=reference_gap,
                        **error_metrics(states,continuous,tip)))
                # One sequence from the continuous-feedback trajectory, shared by all methods.
                ref_states=np.ascontiguousarray(flows[(seed,policy)].sol(tt[:-1]).T)
                goals=np.ascontiguousarray(target.sol(tt[:-1]).T)
                sequence=np.array([lrc.command(x,t,y,kind,nu,d,w,params)[0] for x,t,y in zip(ref_states,tt[:-1],goals)])
                held2=held_reference(np.zeros(4),h,sequence,stride,2,nu,d,A,B)
                held4=held_reference(np.zeros(4),h,sequence,stride,4,nu,d,A,B)
                gap=float(np.max(np.abs(held2-held4)))
                if gap>2e-10:raise RuntimeError('Held reference refinement exceeded predeclared tolerance')
                refchecks.append(dict(seed=seed,policy=policy,h_multiplier=mult,maximum_substep_refinement_gap=gap))
                for method_i,method in enumerate(METHODS):
                    states,maxres=frozen_simulate(np.zeros(4),h,sequence,stride,method_i,nu,d)
                    p=OUT/'frozen_sequence'/f'seed_{seed}_{policy}_hmult{mult:g}_{method}.npz';p.parent.mkdir(exist_ok=True)
                    np.savez_compressed(p,times_tau=times,states=states,held_reference=held4,held_reference_sub2=held2,continuous_feedback_reference=continuous,
                        controls=sequence,tip_map_m=tip)
                    records_c.append(dict(branch='frozen_sequence',seed=seed,policy=policy,h_multiplier=mult,method=method,h_tau=h,
                       trajectory_path=p.relative_to(ROOT).as_posix(),trajectory_sha256=sha(p),maximum_midpoint_residual=float(maxres),
                       hold_error_against_continuous_feedback=error_metrics(held4,continuous,tip),**error_metrics(states,held4,tip)))
                print('HELD_DONE',seed,policy,mult,'reference_gap',gap,flush=True)
    save(OUT/'sampled_feedback_records.json',records_b);save(OUT/'frozen_sequence_records.json',records_c)
    save(OUT/'held_reference_refinement.json',dict(records=refchecks,maximum_gap=max(v['maximum_substep_refinement_gap'] for v in refchecks)))
    save(OUT/'reused_inputs.json',reused)

def scalar_and_structure(protocol,spec,initials):
    hs=[.2,.1,.05,.025,.0125];mass=2.;stiffness=1.;amp=.05;frequency=.7;end=5.;start=np.array([.1,.2]);j2=np.array([[0.,1.],[-1.,0.]])
    aug=np.array([[0.,1/mass,0.,0.],[-stiffness,0.,amp,0.],[0.,0.,0.,frequency],[0.,0.,-frequency,0.]])
    L=aug[:2,:2];rows=[]
    for h in hs:
        n=round(end/h);times=np.arange(n+1)*h;reference=np.array([(expm(t*aug)@np.r_[start,0.,1.])[:2] for t in times])
        cayley=np.linalg.solve(np.eye(2)-h*L/2,np.eye(2)+h*L/2)
        for method in METHODS+['bad_fixed_increment']:
            x=start.copy();states=[x.copy()]
            for k in range(n):
                u=amp*np.sin(frequency*(k+.5)*h)
                if method=='I':x=np.linalg.solve(np.eye(2)-h*L/2,(np.eye(2)+h*L/2)@x+h*np.array([0.,u]))
                elif method=='II':x=cayley@x+np.array([0.,h*u])
                elif method=='III':x=cayley@(x+np.array([0.,h*u/2]))+np.array([0.,h*u/2])
                else:x=cayley@x+np.array([0.,u])
                states.append(x.copy())
            p=OUT/'scalar_m2'/f'{method}_h{h:g}.npz';p.parent.mkdir(exist_ok=True)
            np.savez_compressed(p,times=times,states=states,reference=reference)
            rows.append(dict(method=method,h=h,endpoint_state_l2=float(np.linalg.norm(x-reference[-1])),state_absolute_rms=float(np.sqrt(np.mean((np.array(states)-reference)**2))),
                fixed_input_symplectic_defect=float(np.linalg.norm(cayley.T@j2@cayley-j2)),path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
    save(OUT/'scalar_m2_records.json',rows)
    nu=np.array(spec['nu']);d=np.array(spec['slope_map_d']);w=spec['omega0_rad_s'];records=[]
    # Target at zero is exactly zero; sample time0 makes the test deterministic.
    for policy in POLICIES:
        seeds=[SEEDS[0]] if policy=='none' else SEEDS
        for seed in seeds:
            params=lrc.params_for(seed,policy);kind=0 if policy=='none' else 2
            seed_i=SEEDS.index(seed)
            for initial_index,x in enumerate([np.zeros(4),initials[seed_i*8]]):
                u=lrc.command(x,0.,np.zeros(4),kind,nu,d,w,params)[0]
                eps=1e-6;eye=np.eye(4)
                du=np.array([(lrc.command(x+eps*e,0.,np.zeros(4),kind,nu,d,w,params)[0]-lrc.command(x-eps*e,0.,np.zeros(4),kind,nu,d,w,params)[0])/(2*eps) for e in eye])
                for mult in MULTS:
                    h=protocol['base_h_tau']*mult
                    for mi,method in enumerate(METHODS):
                        _,fixed,bu=ctrl.fixed_jacobian(x,h,u,mi,nu,d);full=fixed+np.outer(bu,du)
                        gap=None
                        if mult in (4.,.25):
                            numeric=np.column_stack([(ctrl.step(x+eps*e,h,lrc.command(x+eps*e,0.,np.zeros(4),kind,nu,d,w,params)[0],mi,nu,d)[0]-ctrl.step(x-eps*e,h,lrc.command(x-eps*e,0.,np.zeros(4),kind,nu,d,w,params)[0],mi,nu,d)[0])/(2*eps) for e in eye])
                            gap=float(np.max(np.abs(numeric-full)))
                            if gap>2e-8:raise RuntimeError('Full feedback chain Jacobian mismatch')
                        records.append(dict(seed=seed,policy=policy,initial_index=initial_index,h_multiplier=mult,method=method,
                           fixed_input_defect=float(np.linalg.norm(fixed.T@ctrl.J@fixed-ctrl.J)),
                           full_feedback_defect=float(np.linalg.norm(full.T@ctrl.J@full-ctrl.J)),independent_finite_difference_gap=gap))
    save(OUT/'structure_records.json',records)

def summarize():
    rows=[]
    for name in ('smooth_external','sampled_feedback','frozen_sequence'):
        data=read(OUT/f'{name}_records.json')
        keys=['method','initial_index'] if name=='smooth_external' else ['method','policy','seed']
        groups={tuple(r[k] for k in keys) for r in data}
        for group in groups:
            values=sorted([r for r in data if tuple(r[k] for k in keys)==group],key=lambda r:r['h_multiplier'],reverse=True)
            for metric in ('state_relative_l2','state_absolute_rms','tip_rmse_mm','endpoint_state_l2'):
                errors=[r[metric] for r in values]
                orders=[float(np.log2(errors[j-1]/errors[j])) for j in range(1,len(errors))]
                rows.append(dict(branch=name,**dict(zip(keys,group)),metric=metric,h_multipliers=[v['h_multiplier'] for v in values],errors=errors,orders=orders,finest_observed_order=orders[-1]))
    scalar=[]
    ss=read(OUT/'scalar_m2_records.json')
    for method in METHODS+['bad_fixed_increment']:
        values=sorted([r for r in ss if r['method']==method],key=lambda r:r['h'],reverse=True);errors=[v['endpoint_state_l2'] for v in values]
        scalar.append(dict(method=method,h=[v['h'] for v in values],endpoint_errors=errors,orders=[float(np.log2(errors[j-1]/errors[j])) for j in range(1,len(errors))]))
    structure=read(OUT/'structure_records.json')
    summary=dict(convergence=rows,scalar=scalar,structure_fixed_max=max(v['fixed_input_defect'] for v in structure),
        structure_feedback_max=max(v['full_feedback_defect'] for v in structure),
        structure_finite_difference_max=max(v['independent_finite_difference_gap'] or 0. for v in structure),
        held_reference_refinement=read(OUT/'held_reference_refinement.json')['maximum_gap'],
        scope='Observed orders for declared references; not a general controlled convergence theorem, not closed-loop symplecticity.',all_failures_retained=True)
    save(OUT/'summary.json',summary)
    for name in ('smooth_external','sampled_feedback','frozen_sequence'):
        vals=[v for v in rows if v['branch']==name and v['metric']=='state_relative_l2' and v.get('initial_index',0)==0]
        print(name,[(m,min(v['finest_observed_order'] for v in vals if v['method']==m),max(v['finest_observed_order'] for v in vals if v['method']==m)) for m in METHODS],flush=True)
    print('SCALAR',scalar,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['plan','all','smooth','held','structure','summary'],default='plan');args=parser.parse_args()
    protocol,spec,initials=plan();verify_inputs(protocol)
    if args.stage in ('all','smooth'):reference_checks(protocol,spec,initials);branch_a(protocol,spec,initials)
    if args.stage in ('all','held'):branches_b_c(protocol,spec,initials)
    if args.stage in ('all','structure'):scalar_and_structure(protocol,spec,initials)
    if args.stage in ('all','summary'):summarize()
    verify_inputs(protocol)
    save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json')))
