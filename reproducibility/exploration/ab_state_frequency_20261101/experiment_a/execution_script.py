"""A: fixed-weight state/frequency checks; writes only this exploration tree."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
sys.dont_write_bytecode=True
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'experiment_a'
OLD=ROOT/'exploration/complex_advantage_20261101/fine_step'
SEEDS=[17,43,91,137,211]
GROUPS=['H0','H4','GR','GC','Teacher']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def module(name):
    spec=importlib.util.spec_from_file_location('exp_a_'+name,ROOT/'code'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
REFINE=module('neural_physics_refinement')

def weights_for(seed,group):
    w=load(ROOT/f'results/neural_learning_10000/seed_{seed}/{group}_weights.npz')
    params=tuple(np.ascontiguousarray(w[k]) for k in ('w1','b1','w2','b2','w3','b3','scale'))+(float(w['energy_scale']),np.ascontiguousarray(w['nu']),bool(w['prior']))
    return params,w

@njit(cache=False,nogil=True)
def batch_field(flat,kind,params,nu,d):
    result=np.empty_like(flat)
    for n in range(len(flat)//4):
        x=flat[n*4:n*4+4]
        if kind==2:
            result[n*4:n*4+4]=REFINE.neural_field(x,*params)[1]
        else:
            theta=d[0]*x[0]+d[1]*x[1]
            if kind==0:
                # Analytically identical cosine remainder, stable near zero.
                if abs(theta)<.1:
                    a=theta*theta
                    remainder=theta**3*(-1/6+a*(1/120+a*(-1/5040+a*(1/362880-a/39916800))))
                else:remainder=np.sin(theta)-theta
            else: remainder=-theta**3/6
            for j in range(2):
                result[n*4+j]=nu[j]*x[j+2]
                result[n*4+j+2]=-nu[j]*x[j]-d[j]*remainder
    return result

def batch_solve(initials,kind,params,nu,d,times_tau,rtol=1e-11,atol=1e-13):
    """Return (time,trajectory,4) states and diagnostics; no files written."""
    initials=np.ascontiguousarray(initials);nu=np.ascontiguousarray(nu);d=np.ascontiguousarray(d)
    started=time.perf_counter()
    fun=lambda tau,flat:batch_field(flat,kind,params,nu,d)
    result=solve_ivp(fun,(float(times_tau[0]),float(times_tau[-1])),initials.ravel(),t_eval=times_tau,method='DOP853',rtol=rtol,atol=atol)
    if not result.success:raise RuntimeError(result.message)
    states=result.y.T.reshape(-1,len(initials),4)
    if not np.all(np.isfinite(states)):raise RuntimeError('Nonfinite continuous trajectory')
    return states,{'nfev':result.nfev,'seconds':time.perf_counter()-started,'rtol':rtol,'atol':atol}

def exact_linear(initials,nu,times_tau):
    phase=times_tau[:,None,None]*nu[None,None,:]
    q=initials[None,:,:2]*np.cos(phase)+initials[None,:,2:]*np.sin(phase)
    p=initials[None,:,2:]*np.cos(phase)-initials[None,:,:2]*np.sin(phase)
    return np.concatenate([q,p],axis=-1)

def phase_frequency(states,times_tau):
    """Smooth phase-increment frequency; result (trajectory,mode), per tau."""
    z=states[...,:2]-1j*states[...,2:]
    increments=np.angle(np.conjugate(z[:-1])*z[1:])
    if np.max(np.abs(increments))>=.9*np.pi:raise RuntimeError('Phase sampling ambiguous')
    step=np.diff(times_tau)
    if not np.allclose(step,step[0],rtol=1e-8,atol=1e-12):raise RuntimeError('Nonuniform phase grid')
    u=(np.arange(len(step))+.5)/len(step)
    window=np.exp(-1/(u*(1-u)));window/=window.sum()
    return np.asarray(np.sum(window[:,None,None]*increments,axis=0,dtype=np.longdouble)/step[0],float)

def metric_rows(prediction,reference,times_tau,tip):
    """Primary full-state and both modal frequencies, plus diagnostics."""
    error=prediction-reference
    state=np.sqrt(np.sum(error**2,axis=(0,2))/np.sum(reference**2,axis=(0,2)))
    tip_error=error[...,:2]@tip*1000
    tip_rmse=np.sqrt(np.mean(tip_error**2,axis=0))
    frequency=phase_frequency(prediction,times_tau);true_frequency=phase_frequency(reference,times_tau)
    middle=len(times_tau)//2
    first=phase_frequency(prediction[:middle+1],times_tau[:middle+1])
    second=phase_frequency(prediction[middle:],times_tau[middle:])
    true_first=phase_frequency(reference[:middle+1],times_tau[:middle+1])
    true_second=phase_frequency(reference[middle:],times_tau[middle:])
    radius=np.sqrt(prediction[...,:2]**2+prediction[...,2:]**2)
    rows=[]
    for trajectory in range(prediction.shape[1]):
        row={'trajectory':trajectory,'state_relative_l2':float(state[trajectory]),'tip_rmse_mm':float(tip_rmse[trajectory])}
        for j in range(2):
            suffix=str(j+1)
            row['frequency_'+suffix]=float(frequency[trajectory,j])
            row['reference_frequency_'+suffix]=float(true_frequency[trajectory,j])
            row['frequency_abs_error_'+suffix]=float(abs(frequency[trajectory,j]-true_frequency[trajectory,j]))
            row['frequency_relative_error_'+suffix]=float(abs(frequency[trajectory,j]/true_frequency[trajectory,j]-1))
            row['frequency_half_window_shift_'+suffix]=float(abs(second[trajectory,j]-first[trajectory,j]))
            row['reference_half_window_shift_'+suffix]=float(abs(true_second[trajectory,j]-true_first[trajectory,j]))
            row['half_window_frequency_error_'+suffix]=float(max(abs(first[trajectory,j]-true_first[trajectory,j]),abs(second[trajectory,j]-true_second[trajectory,j])))
            row['radius_min_'+suffix]=float(radius[:,trajectory,j].min())
            row['radius_max_'+suffix]=float(radius[:,trajectory,j].max())
            row['cycles_'+suffix]=float(frequency[trajectory,j]*(times_tau[-1]-times_tau[0])/(2*np.pi))
        rows.append(row)
    return rows

def plan():
    inputs={}
    for seed in SEEDS:
        for divisor in (32,64):
            for suffix in ('trajectories.npz','records.json'):
                p=OLD/f'seed_{seed}/h{divisor}_{suffix}';inputs[p.relative_to(ROOT).as_posix()]=sha(p)
        for group in ('GR','GC'):
            p=ROOT/f'results/neural_learning_10000/seed_{seed}/{group}_weights.npz';inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    for rel in ('code/neural_physics_refinement.py','code/train_neural_hnn.py','results/learning/inputs/reference_model_spec.json'):
        inputs[rel]=sha(ROOT/rel)
    protocol={'scope':'A only, fixed original physical model/weights/all40held-out initials; no manuscript or GitHub changes',
              'seeds':SEEDS,'groups':GROUPS,'reuse':'All16s h32/h64 saved outputs unchanged; no old driver rerun',
              'long_duration_requested_seconds':64,'long_grid':'Original output stride20 extended without changing spacing; end no later than64s',
              'continuous_integrator':{'method':'DOP853 batch32dimension','rtol':1e-11,'atol':1e-13},
              'refinement':{'trajectory_per_seed':0,'rtol':5e-13,'atol':5e-15,'models':['H4','GR','GC','Teacher']},
              'H4':'H0-(d.X)^4/24, not the angle-averaged normal form',
              'frequency':'Smooth exp[-1/(s(1-s))] weighted phase increments of X-iY, full/half windows; tau frequency',
              'primary':['state_relative_l2','frequency_abs_error_1','frequency_abs_error_2'],
              'comparisons':['GR-H0','GC-H0','GR-H4','GC-H4','GC-GR'],
              'exploratory_success_rule':'All3primary metrics require>=4/5 directional seed wins and mean paired relative improvement>=5%; distinguish numerical-resolution checks from success, no significance claim',
              'statistics':'First per trajectory over time, mean8withinseed, then mean/sampleSD over5seeds; no timepoint pseudoreplication',
              'input_sha256':inputs}
    path=OUT/'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=protocol:raise RuntimeError('Existing protocol/input mismatch')
    else:save(path,protocol)
    return protocol

def check_inputs(protocol):
    changed=[p for p,h in protocol['input_sha256'].items() if sha(ROOT/p)!=h]
    if changed:raise RuntimeError('Source changed '+str(changed))
    return {'files_checked':len(protocol['input_sha256']),'all_unchanged':True}

def verify_fields():
    trainer=module('train_neural_hnn');gaps=[]
    for seed in SEEDS:
        a=load(OLD/f'seed_{seed}/h64_trajectories.npz');points=a['continuous_reference'][::100].reshape(-1,4)
        for group in ('GR','GC'):
            params,w=weights_for(seed,group)
            independent=trainer.numpy_energy_field(points,w)[1]
            compiled=batch_field(points.ravel(),2,params,a['nu'],a['d']).reshape(-1,4)
            gap=float(np.max(np.abs(independent-compiled)));assert gap<2e-12
            gaps.append({'seed':seed,'group':group,'field_gap':gap})
    save(OUT/'field_verification.json',{'status':'passed','records':gaps})

def reused_metrics():
    rows=[]
    for seed in SEEDS:
        for divisor in (32,64):
            p=OLD/f'seed_{seed}/h{divisor}_trajectories.npz';a=load(p)
            record=json.loads((OLD/f'seed_{seed}/h{divisor}_records.json').read_text(encoding='utf-8'))
            assert sha(p)==record['archive_sha256'] and not record['failures']
            times=a['times_seconds']*float(json.loads((ROOT/'results/learning/inputs/reference_model_spec.json').read_text(encoding='utf-8-sig'))['omega0_rad_s'])
            for group,states in zip(a['groups'],a['predicted_states']):
                for row in metric_rows(states,a['continuous_reference'],times,a['tip_map_m']):
                    rows.append({'seed':seed,'group':str(group),'phase':'saved16s','divisor':divisor,**row})
    save(OUT/'saved16s_trajectory_metrics.json',rows)
    return rows

def long_runs():
    spec=json.loads((ROOT/'results/learning/inputs/reference_model_spec.json').read_text(encoding='utf-8-sig'))
    omega0=float(spec['omega0_rad_s'])
    for seed in SEEDS:
        dest=OUT/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
        path=dest/'continuous64s.npz';record_path=dest/'continuous64s.json'
        if path.exists() or record_path.exists():
            if not path.exists() or not record_path.exists():raise RuntimeError('Incomplete continuous cache')
            record=json.loads(record_path.read_text(encoding='utf-8'));assert record['archive_sha256']==sha(path)
            print('EXISTING64',seed,flush=True);continue
        a=load(OLD/f'seed_{seed}/h64_trajectories.npz')
        h0=float(a['step_tau'])*64;dt=h0*20
        count=int(np.floor(64*omega0/dt));times=np.arange(count+1)*dt
        params,_=weights_for(seed,'GR')
        predictions={};diagnostics={};refined={}
        for group,kind in (('Teacher',0),('H4',1),('GR',2),('GC',2)):
            if kind==2:params,_=weights_for(seed,group)
            print('CONTINUOUS_START',seed,group,flush=True)
            predictions[group],diagnostics[group]=batch_solve(a['initials'],kind,params,a['nu'],a['d'],times)
            tight,diag=batch_solve(a['initials'][:1],kind,params,a['nu'],a['d'],times,5e-13,5e-15)
            gap=float(np.max(np.abs(predictions[group][:,0]-tight[:,0])))
            freq_gap=np.abs(phase_frequency(predictions[group][:,:1],times)-phase_frequency(tight,times))[0]
            refined[group]={'state_max_gap':gap,'frequency_abs_gap':freq_gap.tolist(),'tight_solver':diag}
            print('CONTINUOUS_DONE',seed,group,diagnostics[group]['seconds'],gap,flush=True)
        predictions['H0']=exact_linear(a['initials'],a['nu'],times)
        states=np.stack([predictions[g] for g in GROUPS])
        np.savez_compressed(path,times_tau=times,times_seconds=times/omega0,groups=np.array(GROUPS),predicted_states=states,
                            continuous_reference=predictions['Teacher'],initials=a['initials'],nu=a['nu'],d=a['d'],tip_map_m=a['tip_map_m'])
        rows=[]
        for group in GROUPS:
            for row in metric_rows(predictions[group],predictions['Teacher'],times,a['tip_map_m']):
                rows.append({'seed':seed,'group':group,'phase':'continuous64s','divisor':0,**row})
        save(record_path,{'seed':seed,'status':'completed','actual_duration_seconds':float(times[-1]/omega0),
                         'diagnostics':diagnostics,'reference_refinement':refined,'metrics':rows,'archive_sha256':sha(path),
                         'scope':'64s continuous Hamiltonian flows, not 64s midpoint trajectories; no weights changed'})

def write_csv(path,rows):
    import csv
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def summarize():
    rows=json.loads((OUT/'saved16s_trajectory_metrics.json').read_text(encoding='utf-8'))
    for seed in SEEDS:
        p=OUT/f'seed_{seed}/continuous64s.json'
        if p.exists():rows+=json.loads(p.read_text(encoding='utf-8'))['metrics']
    metrics=['state_relative_l2','tip_rmse_mm','frequency_abs_error_1','frequency_abs_error_2','half_window_frequency_error_1','half_window_frequency_error_2']
    seed_rows=[]
    for phase,div in sorted({(r['phase'],r['divisor']) for r in rows}):
        for seed in SEEDS:
            for group in GROUPS:
                block=[r for r in rows if r['phase']==phase and r['divisor']==div and r['seed']==seed and r['group']==group]
                if not block:continue
                seed_rows.append({'phase':phase,'divisor':div,'seed':seed,'group':group,**{k:float(np.mean([r[k] for r in block])) for k in metrics}})
    aggregates=[];contrasts=[]
    for phase,div in sorted({(r['phase'],r['divisor']) for r in seed_rows}):
        for group in GROUPS:
            block=[r for r in seed_rows if r['phase']==phase and r['divisor']==div and r['group']==group]
            if block:aggregates.append({'phase':phase,'divisor':div,'group':group,'n':len(block),**{k:{'mean':float(np.mean([r[k] for r in block])),'sample_sd':float(np.std([r[k] for r in block],ddof=1))} for k in metrics}})
        for candidate,baseline in [('GR','H0'),('GC','H0'),('GR','H4'),('GC','H4'),('GC','GR')]:
            for metric in metrics:
                pairs=[]
                for seed in SEEDS:
                    rr={r['group']:r for r in seed_rows if r['phase']==phase and r['divisor']==div and r['seed']==seed}
                    if candidate in rr and baseline in rr:
                        a,b=rr[candidate][metric],rr[baseline][metric]
                        pairs.append({'seed':seed,'candidate':a,'baseline':b,'relative_improvement':1-a/b if b>1e-30 else None})
                if pairs and all(p['relative_improvement'] is not None for p in pairs):
                    improvements=[p['relative_improvement'] for p in pairs]
                    contrasts.append({'phase':phase,'divisor':div,'candidate':candidate,'baseline':baseline,'metric':metric,'n':len(pairs),
                                      'wins':sum(x>0 for x in improvements),'mean_paired_relative_improvement':float(np.mean(improvements)),
                                      'rule_passed':bool(len(pairs)==5 and sum(x>0 for x in improvements)>=4 and np.mean(improvements)>=.05),'pairs':pairs})
    save(OUT/'summary.json',{'aggregates':aggregates,'contrasts':contrasts,'scope':'Exploratory paired diagnostics, no significance/universal advantage claim; numerical floor must additionally be checked.'})
    write_csv(OUT/'trajectory_metrics.csv',rows);write_csv(OUT/'seed_metrics.csv',seed_rows)
    write_csv(OUT/'paired_comparisons.csv',[{k:v for k,v in c.items() if k!='pairs'} for c in contrasts])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['plan','run','summary'],default='run');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);protocol=plan();check_inputs(protocol)
    save(OUT/'environment.json',{'python':sys.version,'platform':platform.platform(),'script_sha256':sha(__file__),'started_utc':datetime.now(timezone.utc).isoformat()})
    with threadpool_limits(limits=1):
        if args.stage=='run':
            verify_fields();reused_metrics();long_runs();summarize()
        elif args.stage=='summary':summarize()
    save(OUT/'input_protection.json',check_inputs(protocol))

if __name__=='__main__':main()
