"""Independent B audit and CSV export; imports no experiment implementation."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
sys.dont_write_bytecode=True
import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
ROOT=BASE.parents[1]
SEEDS=[17,43,91,137,211]
LEVELS=['B030mT','B060mT','B120mT']
METRICS=['state_relative_l2','frequency_abs_error_1','frequency_abs_error_2','tip_rmse_mm',
         'frequency_relative_error_1','frequency_relative_error_2',
         'frequency_abs_error_1_rad_s','frequency_abs_error_2_rad_s']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv_save(p,rows):
    with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def energy(x,w):
    z=np.tanh((x/w['scale'])@w['w1'].T+w['b1'])
    z=np.tanh(z@w['w2'].T+w['b2'])
    value=w['energy_scale']*(z@w['w3']+w['b3'][0])
    if bool(w['prior']):value+=.5*np.sum(x*x*np.tile(w['nu'],2),axis=-1)
    return value

def neural_field(x,w):
    """Scalar exported real-block network, independently reverse-differentiated."""
    z1=np.tanh(np.matmul(x/w['scale'],w['w1'].T)+w['b1'])
    z2=np.tanh(np.matmul(z1,w['w2'].T)+w['b2'])
    v=(1-z2*z2)*w['w3']
    v=(v@w['w2'])*(1-z1*z1)
    grad=(v@w['w1'])/w['scale']*w['energy_scale']
    if bool(w['prior']):grad+=x*np.tile(w['nu'],2)
    return np.concatenate((grad[...,2:],-grad[...,:2]),axis=-1)

def physical_field(x,nu,d,quartic=False):
    q,p=x[...,:2],x[...,2:];theta=q@d
    residual=-theta**3/6 if quartic else np.sin(theta)-theta
    return np.concatenate((p*nu,-q*nu-residual[...,None]*d),axis=-1)

def frequency(states,times):
    """Independent real dot/cross phase increments; no complex-angle helper."""
    x0,y0=states[:-1,...,:2],states[:-1,...,2:]
    x1,y1=states[1:,...,:2],states[1:,...,2:]
    angles=np.arctan2(y0*x1-x0*y1,x0*x1+y0*y1)
    assert np.max(np.abs(angles))<.9*np.pi
    dt=np.diff(times);assert np.allclose(dt,dt[0],rtol=1e-8,atol=1e-12)
    u=(np.arange(len(dt))+.5)/len(dt)
    w=np.exp(-1/(u*(1-u)));w=w/w.sum()
    return np.asarray(np.einsum('t,tjm->jm',w.astype(np.longdouble),angles.astype(np.longdouble))/dt[0],float)

def checked_sources():
    frozen=read(OUT/'protocol.json')['input_sha256']
    source_changed=[p for p,h in frozen.items() if sha(ROOT/p)!=h]
    protected=read(BASE/'protected_manifest.json')
    protected_changed=[p for p,h in protected.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    assert not source_changed and not protected_changed,(source_changed,protected_changed)
    return dict(frozen_files_checked=len(frozen),protected_files_checked=len(protected),
                changed_frozen_files=source_changed,changed_protected_files=protected_changed)

def audit_fits(require_all):
    rows=[];spec=read(ROOT/'results/learning/inputs/reference_model_spec.json')
    baseline=load(BASE/'physical_design/B030mT.npz')
    for level in LEVELS:
        model=load(BASE/f'physical_design/{level}.npz')
        for seed in SEEDS:
            dest=OUT/level/f'seed_{seed}';data=load(dest/'data.npz');meta=read(dest/'data.json')
            original=load(ROOT/f'results/learning/data/seed_{seed}.npz')
            assert sha(dest/'data.npz')==meta['data_sha256']
            assert np.allclose(data['nu'],model['nu'],rtol=0,atol=2e-14)
            assert np.allclose(data['d'],model['d'],rtol=0,atol=2e-14)
            overlap=model['phi'].T@model['mass']@baseline['phi']
            max_projection_gap=0.;max_label_gap=0.
            for split,ntraj,nsample in [('train',12,286),('val',4,571)]:
                x=data[split+'_midpoints'];y=data[split+'_clean_labels'];noise=data[split+'_noise']
                assert x.shape==y.shape==noise.shape==(ntraj*nsample,4)
                assert np.array_equal(noise,original[split+'_noise'])
                assert np.array_equal(data[split+'_labels'],y+noise)
                rhs=physical_field(x,data['nu'],data['d'])
                max_label_gap=max(max_label_gap,float(np.max(np.abs(y-rhs))))
                assert max_label_gap<5e-11
                assert np.array_equal(data[split+'_trajectory_id'],np.repeat(np.arange(ntraj),nsample))
            for split in ('train','val','test'):
                old=original[split+'_initials'];q=(old[:,:2]*baseline['modal_scale'])@overlap.T
                v=(old[:,2:]*baseline['modal_scale']*baseline['omega'])@overlap.T
                projected=np.concatenate((q/model['modal_scale'],v/model['modal_scale']/model['omega']),axis=-1)
                max_projection_gap=max(max_projection_gap,float(np.max(np.abs(projected-data[split+'_initials']))))
            assert max_projection_gap<5e-13
            gate=meta['energy_geometry_screen'];assert gate['continuous_model_energy_screen_passed']
            for group in ('GR','GC'):
                record=dest/f'{group}_training.json'
                if not record.exists():
                    if require_all:raise RuntimeError(f'Missing completed fit {level}/{seed}/{group}')
                    continue
                fit=read(record);w=load(dest/f'{group}_weights.npz');history=read(dest/f'{group}_history.json')
                assert fit['status']=='completed' and sha(dest/f'{group}_weights.npz')==fit['weights_sha256']
                assert fit['real_parameter_count']==673 and fit['training_steps']==10000
                assert fit['data_sha256']==sha(dest/'data.npz')
                if level=='B030mT':assert sha(dest/f'{group}_weights.npz')==sha(ROOT/f'results/neural_learning_10000/seed_{seed}/{group}_weights.npz')
                trainx,trainy=data['train_midpoints'],data['train_labels']
                assert np.array_equal(w['scale'],np.maximum(np.sqrt(np.mean(trainx**2,axis=0)),1e-4))
                ys=np.maximum(np.sqrt(np.mean(trainy**2,axis=0)),1e-6)
                vi=np.random.default_rng(seed+3000).choice(len(data['val_midpoints']),min(2048,len(data['val_midpoints'])),replace=False)
                valfield=neural_field(data['val_midpoints'],w)
                mse=float(np.mean(((valfield[vi]-data['val_labels'][vi])/ys)**2))
                mse_gap=abs(mse-fit['validation_normalized_mse']);assert mse_gap<1e-14
                best=min(history,key=lambda x:x['validation_normalized_mse'])
                assert best['step']==fit['selected_validation_step'] and abs(best['validation_normalized_mse']-mse)<1e-14
                points=data['val_midpoints'][::max(1,len(valfield)//37)][:37]
                grad=np.empty_like(points)
                for j in range(4):
                    z=points.astype(complex);z[:,j]+=1e-24j
                    grad[:,j]=np.imag(energy(z,w))/1e-24
                complex_field=np.concatenate((grad[:,2:],-grad[:,:2]),axis=-1)
                gradient_gap=float(np.max(np.abs(neural_field(points,w)-complex_field)));assert gradient_gap<2e-13
                clean_relative=float(np.linalg.norm(valfield-data['val_clean_labels'])/np.linalg.norm(data['val_clean_labels']))
                assert abs(clean_relative-fit['clean_validation_relative_l2'])<1e-13
                rows.append(dict(level=level,seed=seed,group=group,baseline_weights_reused=fit['baseline_weights_reused'],
                                 midpoint_rhs_maximum_gap=max_label_gap,physical_projection_maximum_gap=max_projection_gap,
                                 complex_step_field_maximum_gap=gradient_gap,checkpoint_validation_mse_gap=mse_gap,
                                 data_sha256=sha(dest/'data.npz'),weights_sha256=sha(dest/f'{group}_weights.npz')))
    return rows

def audit_metrics():
    trajectories=[];checks=[]
    for level in LEVELS:
        for seed in SEEDS:
            dest=OUT/level/f'seed_{seed}';record=read(dest/'evaluation.json');archive=load(dest/'continuous64s.npz');data=load(dest/'data.npz')
            assert record['status']=='completed' and sha(dest/'continuous64s.npz')==record['archive_sha256']
            assert archive['predicted_states'].shape==(5,12801,8,4)
            times=archive['times_tau'];seconds=archive['times_seconds'];omega=float(data['omega0'])
            assert np.array_equal(times,seconds*omega)
            groups=archive['groups'].tolist();states=archive['predicted_states'];reference=states[groups.index('Teacher')]
            max_metric_gap=0.;max_endpoint_gap=0.
            for duration in (16,64):
                mask=seconds<=duration+1e-12;t=times[mask];ref=reference[mask];freq_ref=frequency(ref,t)
                for group,whole in zip(groups,states):
                    p=whole[mask];delta=p-ref
                    state=np.sqrt(np.sum(delta*delta,axis=(0,2))/np.sum(ref*ref,axis=(0,2)))
                    tip=np.sqrt(np.mean((delta[...,:2]@data['tip_map_m']*1000)**2,axis=0))
                    freq=frequency(p,t);freqerr=np.abs(freq-freq_ref);relative=np.abs(freq/freq_ref-1)
                    for trajectory in range(8):
                        row=dict(level=level,field_T=int(level[1:4])/1000,seed=seed,group=group,duration_seconds=duration,
                                 trajectory=trajectory,omega0_rad_s=omega,state_relative_l2=float(state[trajectory]),tip_rmse_mm=float(tip[trajectory]))
                        original=next(r for r in record['metrics'] if r['duration_seconds']==duration and r['group']==group and r['trajectory']==trajectory)
                        for j in range(2):
                            suffix=str(j+1)
                            row['frequency_'+suffix]=float(freq[trajectory,j])
                            row['reference_frequency_'+suffix]=float(freq_ref[trajectory,j])
                            row['frequency_abs_error_'+suffix]=float(freqerr[trajectory,j])
                            row['frequency_relative_error_'+suffix]=float(relative[trajectory,j])
                            row['frequency_abs_error_'+suffix+'_rad_s']=float(freqerr[trajectory,j]*omega)
                        for key in ('state_relative_l2','tip_rmse_mm','frequency_1','frequency_2','frequency_abs_error_1','frequency_abs_error_2','frequency_relative_error_1','frequency_relative_error_2'):
                            max_metric_gap=max(max_metric_gap,abs(row[key]-original[key]))
                        trajectories.append(row)
            assert max_metric_gap<1e-10
            # Independent short restarts probe every long-flow group at three times.
            # This checks the integrated field, not merely the output arithmetic.
            for group,whole in zip(groups,states):
                if group=='H0':
                    q0,p0=archive['test_initials'][:,:2],archive['test_initials'][:,2:]
                    ang=times[:,None,None]*data['nu'];c,s=np.cos(ang),np.sin(ang)
                    exact=np.concatenate((q0*c+p0*s,p0*c-q0*s),axis=-1)
                    gap=float(np.max(np.abs(exact-whole)));assert gap<2e-13
                    max_endpoint_gap=max(max_endpoint_gap,gap);continue
                if group in ('GR','GC'):
                    w=load(dest/f'{group}_weights.npz');fun=lambda t,x:neural_field(x,w)
                else:fun=lambda t,x:physical_field(x,data['nu'],data['d'],quartic=group=='H4')
                for start in (0,6400,12780):
                    stop=start+20
                    sol=solve_ivp(fun,(times[start],times[stop]),whole[start,0],method='DOP853',rtol=5e-13,atol=5e-15,t_eval=[times[stop]])
                    assert sol.success
                    gap=float(np.max(np.abs(sol.y[:,-1]-whole[stop,0])))
                    max_endpoint_gap=max(max_endpoint_gap,gap);assert gap<2e-9
            theta=reference[...,:2]@data['d']
            linear=reference[...,:2]*data['nu']
            residual=(np.sin(theta)-theta)[...,None]*data['d']
            remainder=(np.sin(theta)-theta+theta**3/6)[...,None]*data['d']
            checks.append(dict(level=level,seed=seed,independent_metric_maximum_gap=max_metric_gap,
                               independent_short_restart_endpoint_maximum_gap=max_endpoint_gap,
                               continuous64s_nonlinear_to_linear_force_l2=float(np.linalg.norm(residual)/np.linalg.norm(linear)),
                               continuous64s_H4_remainder_to_linear_force_l2=float(np.linalg.norm(remainder)/np.linalg.norm(linear)),
                               continuous64s_maximum_abs_tip_slope=float(np.max(np.abs(theta))),archive_sha256=sha(dest/'continuous64s.npz')))
    seeds=[]
    for level in LEVELS:
        for seed in SEEDS:
            for duration in (16,64):
                for group in ('H0','H4','GR','GC','Teacher'):
                    selected=[r for r in trajectories if r['level']==level and r['seed']==seed and r['duration_seconds']==duration and r['group']==group]
                    assert len(selected)==8
                    row={k:selected[0][k] for k in ('level','field_T','seed','group','duration_seconds','omega0_rad_s')}
                    row.update({m:float(np.mean([r[m] for r in selected])) for m in METRICS});seeds.append(row)
    aggregates=[];comparisons=[]
    for level in LEVELS:
        for duration in (16,64):
            for group in ('H0','H4','GR','GC','Teacher'):
                selected=[r for r in seeds if r['level']==level and r['duration_seconds']==duration and r['group']==group]
                for metric in METRICS:
                    values=np.array([r[metric] for r in selected])
                    aggregates.append(dict(level=level,duration_seconds=duration,group=group,metric=metric,mean=float(values.mean()),sample_sd=float(values.std(ddof=1))))
            for candidate,baseline in [('GR','H0'),('GC','H0'),('GR','H4'),('GC','H4'),('GC','GR')]:
                rule=[]
                for metric in METRICS:
                    pairs=[]
                    for seed in SEEDS:
                        c=next(r[metric] for r in seeds if r['level']==level and r['duration_seconds']==duration and r['seed']==seed and r['group']==candidate)
                        b=next(r[metric] for r in seeds if r['level']==level and r['duration_seconds']==duration and r['seed']==seed and r['group']==baseline)
                        pairs.append((c,b))
                    pairs=np.array(pairs);improvements=1-pairs[:,0]/pairs[:,1]
                    passed=bool(np.sum(improvements>0)>=4 and improvements.mean()>=.05)
                    if metric in METRICS[:3]:rule.append(passed)
                    comparisons.append(dict(level=level,duration_seconds=duration,candidate=candidate,baseline=baseline,metric=metric,
                                            candidate_wins=int(np.sum(improvements>0)),mean_paired_relative_improvement=float(improvements.mean()),
                                            mean_absolute_error_reduction=float(np.mean(pairs[:,1]-pairs[:,0])),passed_metric_rule=passed))
                comparisons.append(dict(level=level,duration_seconds=duration,candidate=candidate,baseline=baseline,metric='ALL_THREE_PRIMARY',
                                        candidate_wins='',mean_paired_relative_improvement='',mean_absolute_error_reduction='',passed_metric_rule=all(rule)))
    csv_save(OUT/'independent_metrics_trajectory.csv',trajectories)
    csv_save(OUT/'independent_metrics_seed.csv',seeds)
    csv_save(OUT/'independent_aggregates.csv',aggregates)
    csv_save(OUT/'independent_comparisons.csv',comparisons)
    return checks,comparisons

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['fits','all'],default='all');args=parser.parse_args()
    with threadpool_limits(limits=1):
        sources=checked_sources();fits=audit_fits(require_all=args.stage=='all')
        if args.stage=='all':metrics,comparisons=audit_metrics()
        else:metrics=[];comparisons=[]
        report=dict(status='passed',stage=args.stage,source_checks=sources,completed_fits=len(fits),expected_fits=30,
                    independent_fit_checks=fits,independent_flow_checks=metrics,
                    combined_primary_rules=[x for x in comparisons if x['metric']=='ALL_THREE_PRIMARY'],
                    numerical_scope='Independent local restarts at0,32,63.9s; first-trajectory tight64s refinements recorded by experiment. Not a global certified error bound.',
                    physical_scope='Original declared approximate slope-angle constant-dipole model; no material calibration or geometric/high-mode physical validation.')
        save(OUT/f'independent_audit_{args.stage}.json',report)
        print(json.dumps(dict(status='passed',stage=args.stage,completed_fits=len(fits),completed_seed_flows=len(metrics),source_checks=sources)))
