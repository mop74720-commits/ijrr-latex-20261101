"""Equation-(36) learning in a canonical variable-mass chart of the SAME beam.

X=f(q)=q+q^3, Y=p/f'(q). This is a coordinate benchmark, not a new
calibrated constitutive mass law. All existing files remain read-only.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
sys.dont_write_bytecode = True
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/variable_mass_learning'
SEEDS = [17,43,91,137,211]
GROUPS = ['R','C','GR','GC']
ALPHA = 1.
SPEC = ROOT / 'results/learning/inputs/reference_model_spec.json'
spec = json.loads(SPEC.read_text(encoding='utf-8-sig'))
NU = np.array(spec['nu'])
D = np.array(spec['slope_map_d'])
TIP = np.array(spec['tip_map_m'])
OMEGA0 = float(spec['omega0_rad_s'])
J = np.block([[np.zeros((2,2)),np.eye(2)],[-np.eye(2),np.zeros((2,2))]])

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def module(name):
    path = ROOT / 'code' / (name+'.py')
    s = importlib.util.spec_from_file_location('variable_mass_'+name,path)
    m = importlib.util.module_from_spec(s)
    sys.modules[s.name] = m
    s.loader.exec_module(m)
    return m

def to_physical(x):
    x=np.asarray(x)
    q,p=x[...,:2],x[...,2:]
    return np.concatenate([q+ALPHA*q**3,p/(1+3*ALPHA*q*q)],axis=-1)

def from_physical(x):
    x=np.asarray(x)
    q=x[...,:2].copy()
    for _ in range(12):
        q -= (q+ALPHA*q**3-x[...,:2])/(1+3*ALPHA*q*q)
    return np.concatenate([q,x[...,2:]*(1+3*ALPHA*q*q)],axis=-1)

def physical_field(x):
    q,p=x[...,:2],x[...,2:]
    theta=q@D
    return np.concatenate([NU*p,-NU*q-(np.sin(theta)-theta)[...,None]*D],axis=-1)

def model_energy(x, magnetic=True):
    y=to_physical(x)
    result=.5*np.sum(NU*(y[...,:2]**2+y[...,2:]**2),axis=-1)
    if magnetic:
        theta=y[...,:2]@D
        result += 2*np.sin(theta/2)**2-.5*theta**2
    return result

def model_field(x, magnetic=True):
    x=np.asarray(x)
    q,p=x[...,:2],x[...,2:]
    f=q+ALPHA*q**3
    s=1+3*ALPHA*q*q
    inv=NU/s**2
    dinv=-12*ALPHA*NU*q/s**3
    force=s*NU*f
    if magnetic:
        theta=f@D
        force=force+s*D*(np.sin(theta)-theta)[...,None]
    return np.concatenate([inv*p,-.5*dinv*p*p-force],axis=-1)

@njit(cache=False,nogil=True)
def teacher_field(x,nu,d,magnetic):
    result=np.empty(4)
    theta=0.
    for j in range(2): theta += d[j]*(x[j]+ALPHA*x[j]**3)
    residual=np.sin(theta)-theta if magnetic else 0.
    for j in range(2):
        q,p=x[j],x[j+2]
        s=1+3*ALPHA*q*q
        inv=nu[j]/s**2
        dinv=-12*ALPHA*nu[j]*q/s**3
        result[j]=inv*p
        result[j+2]=-.5*dinv*p*p-s*(nu[j]*(q+ALPHA*q**3)+d[j]*residual)
    return result

@njit(cache=False,nogil=True)
def teacher_rollout(initials,h,nsteps,stride,nu,d,magnetic=True):
    output=np.empty((len(initials),nsteps//stride+1,4))
    max_residual=0.
    max_iterations=0
    for trajectory in range(len(initials)):
        old=initials[trajectory].copy()
        output[trajectory,0]=old
        for step in range(1,nsteps+1):
            new=old+h*teacher_field(old,nu,d,magnetic)
            converged=False
            for iteration in range(50):
                updated=old+h*teacher_field((old+new)/2,nu,d,magnetic)
                gap=np.max(np.abs(updated-new))
                new=updated
                if gap<2e-14:
                    converged=True
                    break
            if not converged: raise RuntimeError('Variable-mass teacher implicit iteration failed')
            residual=np.max(np.abs(new-old-h*teacher_field((old+new)/2,nu,d,magnetic)))
            max_residual=max(max_residual,residual)
            max_iterations=max(max_iterations,iteration+1)
            old=new
            if step%stride==0: output[trajectory,step//stride]=old
    return output,max_residual,max_iterations

_trainer=None
_base_numpy=None

def numpy_energy_field(x,w):
    base=dict(w)
    base['prior']=np.array(False)
    energy,field=_base_numpy(x,base)
    if bool(w['prior']):
        energy=energy+model_energy(x,False)
        field=field+model_field(x,False)
    return energy,field

def trainer():
    global _trainer,_base_numpy
    if _trainer is not None: return _trainer
    t=module('train_neural_hnn')
    base=t.EnergyNetwork
    _base_numpy=t.numpy_energy_field
    class PullbackEnergyNetwork(base):
        def __init__(self,family,prior,scale,energy_scale,nu):
            super().__init__(family,False,scale,energy_scale,nu)
            self.physical_prior=bool(prior)
        def forward(self,x):
            out=super().forward(x)
            if self.physical_prior:
                q,p=x[...,:2],x[...,2:]
                f=q+ALPHA*q**3
                s=1+3*ALPHA*q*q
                out=out+.5*t.torch.sum(self.nu*(f*f+p*p/(s*s)),dim=-1)
            return out
        def export(self):
            result=super().export()
            result['prior']=np.array(self.physical_prior)
            result['mass_alpha']=np.array(ALPHA)
            return result
    t.EnergyNetwork=PullbackEnergyNetwork
    t.numpy_energy_field=numpy_energy_field
    _trainer=t
    return t

# Reuse only the existing compiled network derivative, with its quadratic prior
# disabled. The known pullback prior is then differentiated independently above.
refinement=module('neural_physics_refinement')
base_neural_field=refinement.neural_field

def params_for(w):
    return tuple(np.ascontiguousarray(w[k]) for k in ('w1','b1','w2','b2','w3','b3','scale'))+(float(w['energy_scale']),np.ascontiguousarray(w['nu']),False)

@njit(cache=False,nogil=True)
def learned_field(x,params,prior):
    energy,field=base_neural_field(x,*params)
    if prior: field=field+teacher_field(x,params[8],np.zeros(2),False)
    return field

@njit(cache=False,nogil=True)
def neural_rollout(initials,h,nsteps,stride,params,prior):
    output=np.empty((len(initials),nsteps//stride+1,4))
    maximum_residual=0.
    max_iterations=0
    for trajectory in range(len(initials)):
        old=initials[trajectory].copy()
        output[trajectory,0]=old
        for step in range(1,nsteps+1):
            new=old+h*learned_field(old,params,prior)
            converged=False
            for iteration in range(50):
                updated=old+h*learned_field((old+new)/2,params,prior)
                gap=np.max(np.abs(updated-new))
                new=updated
                if gap<2e-14:
                    converged=True
                    break
            if not converged: raise RuntimeError('Variable-mass learned midpoint iteration failed')
            residual=np.max(np.abs(new-old-h*learned_field((old+new)/2,params,prior)))
            maximum_residual=max(maximum_residual,residual)
            max_iterations=max(max_iterations,iteration+1)
            old=new
            if not np.all(np.isfinite(old)) or np.max(np.abs(old))>100:
                raise RuntimeError('Variable-mass learned trajectory diverged')
            if step%stride==0: output[trajectory,step//stride]=old
    return output,maximum_residual,max_iterations

def protocol():
    cfg=dict(schema=1,scope='Canonical coordinate benchmark of the same conservative two-mode beam; not a calibrated constitutive variable-mass law.',
             transform='X_j=q_j+q_j^3, Y_j=p_j/(1+3q_j^2); alpha=1 fixed before generation',
             inverse_mass='diag(nu_j/(1+3q_j^2)^2)',delta='-.5*p_mid^2*d(M^-1)/dq at midpoint',
             labels='New Eq36 implicit midpoint trajectories; midpoint state to adjacent increment/h plus uniform noise1e-6',
             prior='Pullback known kinetic+linear elastic energy .5sum nu*(p^2/fprime^2+f^2)',
             state_order=['q1','q2','p1','p2'],evaluation_coordinates='Original physical modal canonical X,Y for full state and tip',
             seeds=SEEDS,groups=GROUPS,real_parameters=673,updates=10000,batch_size=512,
             optimizer='Original Adam protocol .002, x.3 at budget half and three quarters; clip10',
             checkpoint='Original fixed seed+3000 validation subset2048 every100updates; independent noisy labels',
             durations_s=dict(train=2.,validation=4.,test_prefix=16.),train_trajectories=12,validation_trajectories=4,test_trajectories=8,
             initial_conditions='All corresponding original seed train/val/test initial conditions, transformed to this chart; no outcome selection',
             step_tau='Original test teacher h0 approx.01; train/val endpoint-aligned approx.01',training_stride=8,test_stride=20,
             prediction_divisors=[1,4],primary='All five paired contrasts for continuous and matching-step teacher tip/state errors; no superiority threshold selection',
             source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (SPEC,ROOT/'code/train_neural_hnn.py',ROOT/'code/neural_physics_refinement.py',Path(__file__))})
    p=OUT/'protocol.json'
    if p.exists() and json.loads(p.read_text(encoding='utf-8'))!=cfg:
        raise RuntimeError('Variable-mass protocol changed')
    if not p.exists(): save(p,cfg)
    return cfg

def verify():
    rng=np.random.default_rng(760031)
    points=rng.uniform(-.2,.2,(32,4))
    roundtrip=float(np.max(np.abs(from_physical(to_physical(points))-points)))
    chain_gaps=[]
    direct_gaps=[]
    symplectic=[]
    delta_max=0.
    for x in points:
        q,p=x[:2],x[2:]
        s=1+3*q*q
        field=model_field(x)
        chain=np.r_[s*field[:2],field[2:]/s-p*6*q*field[:2]/s**2]
        chain_gaps.append(np.max(np.abs(chain-physical_field(to_physical(x)))))
        h=.02
        states,_,_=teacher_rollout(x[None,:],h,1,1,NU,D)
        end=states[0,-1]
        mid=(x+end)/2
        q,p=mid[:2],mid[2:]
        s=1+3*q*q
        n=NU/s**2
        dinv=-12*NU*q/s**3
        delta=-.5*p*p*dinv
        delta_max=max(delta_max,float(np.max(np.abs(delta))))
        f=q+q**3
        theta=f@D
        ratio=(np.sin(theta)-theta)/theta if abs(theta)>1e-12 else 0.
        stiffness=np.diag(s)@(np.diag(NU)+ratio*np.outer(D,D))@np.diag(1+q*q)
        eps=h*h/4*stiffness@np.diag(n)
        pn=np.linalg.solve(np.eye(2)+eps,(np.eye(2)-eps)@x[2:]-h*stiffness@x[:2]+h*delta)
        qn=x[:2]+h/2*n*(pn+x[2:])
        direct_gaps.append(np.max(np.abs(np.r_[qn,pn]-end)))
        columns=[]
        for j in range(4):
            z=mid.astype(complex)
            z[j]+=1e-25j
            columns.append(np.imag(model_field(z))/1e-25)
        jac=np.array(columns).T
        tangent=np.linalg.solve(np.eye(4)-h/2*jac,np.eye(4)+h/2*jac)
        symplectic.append(np.linalg.norm(tangent.T@J@tangent-J,ord=np.inf))
    initial=points[0]
    ref=solve_ivp(lambda t,x:model_field(x),(0.,4.),initial,method='DOP853',rtol=3e-13,atol=3e-15)
    if not ref.success: raise RuntimeError(ref.message)
    convergence=[]
    for h in (.08,.04,.02,.01):
        path,_,_=teacher_rollout(initial[None,:],h,round(4/h),round(4/h),NU,D)
        error=float(np.linalg.norm(path[0,-1]-ref.y[:,-1]))
        row=dict(h=h,error=error)
        if convergence: row['observed_order']=float(np.log2(convergence[-1]['error']/error))
        convergence.append(row)
    result=dict(canonical_roundtrip_gap=roundtrip,continuous_chain_rule_gap=float(max(chain_gaps)),
                direct_eq36_matrix_vs_midpoint_gap=float(max(direct_gaps)),max_tangent_symplectic_defect=float(max(symplectic)),
                nonzero_delta_max=delta_max,convergence=convergence,scope='Conservative canonical pullback coordinate test only')
    assert roundtrip<1e-14 and max(chain_gaps)<1e-14 and max(direct_gaps)<1e-12 and max(symplectic)<1e-12
    assert delta_max>1e-4 and all(1.8<r['observed_order']<2.2 for r in convergence[-2:])
    save(OUT/'implementation_verification.json',result)
    print('IMPLEMENTATION_VERIFIED',json.dumps(result),flush=True)

def generate_data():
    protocol()
    for seed in SEEDS:
        dest=OUT/'data'/f'seed_{seed}.npz'
        if dest.exists():
            meta=json.loads(dest.with_suffix('.json').read_text(encoding='utf-8'))
            if sha(dest)!=meta['dataset_sha256']: raise RuntimeError('Dataset checksum mismatch')
            continue
        original=ROOT/'results/learning/data'/f'seed_{seed}.npz'
        source=load(original)
        arrays={}
        diagnostics={}
        started=time.perf_counter()
        for split,duration,count,offset in [('train',2.,12,0),('val',4.,4,100000)]:
            initials=from_physical(source[f'{split}_initials'])
            nsteps=int(np.ceil(duration*OMEGA0/.01))
            h=duration*OMEGA0/nsteps
            path,res,its=teacher_rollout(initials,h,nsteps,1,NU,D)
            mids=(path[:,1:]+path[:,:-1])/2
            clean=np.diff(path,axis=1)/h
            indices=np.arange(0,nsteps,8)
            x=mids[:,indices].reshape(-1,4)
            y=clean[:,indices].reshape(-1,4)
            noise=np.random.default_rng(seed+offset+10000).uniform(-1e-6,1e-6,y.shape)
            arrays.update({f'{split}_initials':initials,f'{split}_midpoints':x,f'{split}_labels':y+noise,
                           f'{split}_clean_labels':y,f'{split}_noise':noise,f'{split}_trajectory_id':np.repeat(np.arange(count),len(indices)),
                           f'{split}_step_tau':np.array(h)})
            invmass=NU/(1+3*x[:,:2]**2)**2
            delta=6*NU*x[:,:2]*x[:,2:]**2/(1+3*x[:,:2]**2)**3
            diagnostics[split]=dict(maximum_implicit_residual=float(res),max_iterations=int(its),
                midpoint_label_gap=float(np.max(np.abs(y-model_field(x)))),mass_diagonal_min=np.min(1/invmass,axis=0).tolist(),
                mass_diagonal_max=np.max(1/invmass,axis=0).tolist(),delta_max=float(np.max(np.abs(delta))))
        h=float(source['teacher_step_tau'])
        nsteps=int(16.*OMEGA0/h)//20*20
        times_tau=np.arange(nsteps//20+1)*20*h
        initials=from_physical(source['test_initials'])
        teacher,res,its=teacher_rollout(initials,h,nsteps,20,NU,D)
        reference=[]
        for initial in initials:
            r=solve_ivp(lambda t,x:model_field(x),(0.,times_tau[-1]),initial,t_eval=times_tau,method='DOP853',rtol=2e-11,atol=2e-13)
            if not r.success: raise RuntimeError(r.message)
            reference.append(r.y.T)
        arrays.update(test_initials=initials,test_times=times_tau/OMEGA0,test_times_tau=times_tau,
            test_teacher_states=teacher,test_reference_states=np.array(reference),teacher_step_tau=np.array(h),teacher_nsteps=np.array(nsteps),
            teacher_sample_stride=np.array(20),nu=NU,d=D,omega0=np.array(OMEGA0),tip_map_m=TIP,mass_alpha=np.array(ALPHA))
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,**arrays)
        save(dest.with_suffix('.json'),dict(seed=seed,dataset_sha256=sha(dest),source_initials_sha256=sha(original),
             diagnostics=diagnostics,test_max_implicit_residual=float(res),elapsed_seconds=time.perf_counter()-started,
             test_seconds=float(arrays['test_times'][-1]),protocol_sha256=sha(OUT/'protocol.json')))
        print('DATA_DONE',seed,time.perf_counter()-started,diagnostics,flush=True)

def train_all():
    protocol()
    t=trainer()
    for seed in SEEDS:
        arrays=load(OUT/'data'/f'seed_{seed}.npz')
        for group in GROUPS:
            dest=OUT/f'seed_{seed}'
            dest.mkdir(parents=True,exist_ok=True)
            record=dest/f'{group}_training.json'
            if record.exists():
                r=json.loads(record.read_text(encoding='utf-8'))
                if r['status']!='completed' or r['weights_sha256']!=sha(dest/f'{group}_weights.npz'):
                    raise RuntimeError('Existing training failed or changed')
                continue
            print('FIT_START',seed,group,flush=True)
            try:
                weights,row,history=t.train_one(seed,group,arrays,10000,512)
                np.savez_compressed(dest/f'{group}_weights.npz',**weights)
                prediction=numpy_energy_field(arrays['val_midpoints'],weights)[1]
                row.update(status='completed',weights_sha256=sha(dest/f'{group}_weights.npz'),
                    dataset_sha256=sha(OUT/'data'/f'seed_{seed}.npz'),protocol_sha256=sha(OUT/'protocol.json'),
                    full_clean_validation_relative_l2=float(np.linalg.norm(prediction-arrays['val_clean_labels'])/np.linalg.norm(arrays['val_clean_labels'])))
                save(dest/f'{group}_history.json',history)
                save(record,row)
                print('FIT_DONE',seed,group,row['training_seconds'],row['full_clean_validation_relative_l2'],flush=True)
            except Exception as error:
                save(record,dict(seed=seed,group=group,status='failed',failure=repr(error)))
                raise

def metrics(prediction,reference,teacher):
    out={}
    for prefix,target in [('continuous',reference),('teacher',teacher)]:
        error=prediction-target
        state=np.sqrt(np.sum(error**2,axis=(1,2))/np.sum(target**2,axis=(1,2)))
        tip=np.sqrt(np.mean((error[...,:2]@TIP*1000)**2,axis=1))
        out.update({prefix+'_state_relative_l2_mean':float(state.mean()),prefix+'_tip_rmse_mm_mean':float(tip.mean()),
                    prefix+'_state_per_trajectory':state.tolist(),prefix+'_tip_mm_per_trajectory':tip.tolist()})
    return out

def evaluate():
    protocol()
    trainer()
    for seed in SEEDS:
        a=load(OUT/'data'/f'seed_{seed}.npz')
        reference=to_physical(a['test_reference_states'])
        for divisor in (1,4):
            h=float(a['teacher_step_tau'])/divisor
            nsteps=int(a['teacher_nsteps'])*divisor
            stride=20*divisor
            teacher,res,its=teacher_rollout(a['test_initials'],h,nsteps,stride,NU,D)
            baseline,baseres,_=teacher_rollout(a['test_initials'],h,nsteps,stride,NU,D,False)
            physical_teacher=to_physical(teacher)
            save(OUT/f'seed_{seed}'/f'baselines_h{divisor}.json',dict(H0=metrics(to_physical(baseline),reference,physical_teacher),teacher_residual=float(res),baseline_residual=float(baseres)))
            for group in GROUPS:
                dest=OUT/f'seed_{seed}'
                rowpath=dest/f'{group}_h{divisor}.json'
                weights_path=dest/f'{group}_weights.npz'
                if rowpath.exists():
                    old=json.loads(rowpath.read_text(encoding='utf-8'))
                    if old['weights_sha256']!=sha(weights_path) or old['trajectory_sha256']!=sha(dest/f'{group}_h{divisor}.npz'):
                        raise RuntimeError('Cached prediction changed')
                    continue
                weights=load(weights_path)
                params=params_for(weights)
                points=a['val_midpoints'][:128]
                independent=np.array([learned_field(x,params,bool(weights['prior'])) for x in points])
                gap=float(np.max(np.abs(independent-numpy_energy_field(points,weights)[1])))
                if gap>2e-12: raise RuntimeError('Compiled network derivative differs')
                start=time.perf_counter()
                prediction,residual,iterations=neural_rollout(a['test_initials'],h,nsteps,stride,params,bool(weights['prior']))
                physical=to_physical(prediction)
                path=dest/f'{group}_h{divisor}.npz'
                np.savez_compressed(path,times_seconds=a['test_times'],predicted_chart_states=prediction,
                    predicted_physical_states=physical,continuous_physical_reference=reference,physical_teacher=physical_teacher,
                    physical_H0=to_physical(baseline))
                row=dict(seed=seed,group=group,divisor=divisor,status='completed',elapsed_seconds=time.perf_counter()-start,
                    weights_sha256=sha(weights_path),trajectory_sha256=sha(path),numpy_numba_field_gap=gap,
                    maximum_implicit_residual=float(residual),max_iterations=int(iterations),**metrics(physical,reference,physical_teacher))
                save(rowpath,row)
                print('EVAL_DONE',seed,group,divisor,row['continuous_tip_rmse_mm_mean'],row['continuous_state_relative_l2_mean'],flush=True)

def summary():
    rows=[]
    paired=[]
    aggregate=[]
    for seed in SEEDS:
        for divisor in (1,4):
            current={}
            for group in GROUPS:
                p=OUT/f'seed_{seed}'/f'{group}_h{divisor}.json'
                if p.exists():
                    row=json.loads(p.read_text(encoding='utf-8'))
                    rows.append(row)
                    current[group]=row
            if len(current)!=4: continue
            for metric in ('continuous_tip_rmse_mm_mean','teacher_tip_rmse_mm_mean','continuous_state_relative_l2_mean','teacher_state_relative_l2_mean'):
                r,c,gr,gc=[current[g][metric] for g in GROUPS]
                for name,value in [('R-GR',r-gr),('C-GC',c-gc),('R-C',r-c),('GR-GC',gr-gc),('(GR-GC)-(R-C)',(gr-gc)-(r-c))]:
                    paired.append(dict(seed=seed,divisor=divisor,metric=metric,contrast=name,difference=value))
    for divisor in (1,4):
        for group in GROUPS:
            values=[r for r in rows if r['divisor']==divisor and r['group']==group]
            if len(values)!=5: continue
            record=dict(divisor=divisor,group=group,n=5)
            for key in ('continuous_tip_rmse_mm_mean','teacher_tip_rmse_mm_mean','continuous_state_relative_l2_mean'):
                v=[r[key] for r in values]
                record[key]=dict(mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1)))
            aggregate.append(record)
    contrast_summary=[]
    for divisor in (1,4):
        for metric in ('continuous_tip_rmse_mm_mean','teacher_tip_rmse_mm_mean','continuous_state_relative_l2_mean','teacher_state_relative_l2_mean'):
            for contrast in ('R-GR','C-GC','R-C','GR-GC','(GR-GC)-(R-C)'):
                values=[r['difference'] for r in paired if r['divisor']==divisor and r['metric']==metric and r['contrast']==contrast]
                if len(values)!=5: continue
                contrast_summary.append(dict(divisor=divisor,metric=metric,contrast=contrast,n=5,mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)),positive_seeds=int(np.sum(np.array(values)>0))))
    for name,records in [('paired_contrasts.csv',paired),('all_predictions.csv',rows)]:
        if records:
            keys=[k for k in records[0] if not isinstance(records[0][k],(dict,list))]
            with (OUT/name).open('w',newline='',encoding='utf-8') as stream:
                w=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(records)
    save(OUT/'summary.json',dict(prediction_count=len(rows),expected_predictions=40,models=aggregate,paired_contrasts=contrast_summary,
        scope='Same beam in a nonlinear canonical chart; conservative H+i0; no empirical superiority or physically calibrated variable mass claim',
        protocol_sha256=sha(OUT/'protocol.json')))
    print('SUMMARY',len(rows),'predictions',len(paired),'seedwise contrasts',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['verify','data','train','evaluate','summary','all'],default='all')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage in ('verify','all'): verify()
    if args.stage in ('data','all'): generate_data()
    if args.stage in ('train','all'): train_all()
    if args.stage in ('evaluate','all'): evaluate()
    if args.stage in ('summary','all'): summary()
