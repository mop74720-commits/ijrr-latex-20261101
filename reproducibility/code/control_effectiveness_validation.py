"""Independent held-out force checks and paired control-effect diagnostics.

All writes are confined to results/control_effectiveness_validation. Existing
learning/control archives are read only; no training or gain selection occurs.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
sys.dont_write_bytecode=True
import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/control_effectiveness_validation'
OLD=ROOT/'results/learned_residual_control'
SEEDS=[17,43,91,137,211]
POLICIES=['none','oracle','GR','GC']
METHODS=['I','II','III']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):
    with np.load(p,allow_pickle=False) as a: return {k:a[k] for k in a.files}
def save(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def module(name):
    spec=importlib.util.spec_from_file_location('cev_'+name,ROOT/'code'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def registry():
    paths=[ROOT/'code/learned_residual_control.py',ROOT/'code/control_comparison.py',ROOT/'code/neural_physics_refinement.py',ROOT/'code/train_neural_hnn.py',ROOT/'results/learning/inputs/reference_model_spec.json']
    paths+=sorted((OLD).rglob('*.npz'))+sorted((OLD).rglob('*.json'))
    for seed in SEEDS:
        paths.append(ROOT/f'results/learning/data/seed_{seed}.npz')
        for g in ['GR','GC']: paths.append(ROOT/f'results/neural_learning_10000/seed_{seed}/{g}_weights.npz')
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def plan():
    OUT.mkdir(parents=True,exist_ok=True)
    old=read(OLD/'protocol.json')
    config=dict(schema=1,created_before_new_evaluation=True,seeds=SEEDS,policies=POLICIES,methods=METHODS,
        common_points='Each original held-out continuous test trajectory:64 rounded equidistant saved indices from t=0 through the last saved time no later than the old control endpoint. All40 trajectories,2560 points, no performance selection.',
        common_prefix_seconds=old['duration_seconds'],point_count_per_trajectory=64,
        own_closed_loop_points='64 rounded equidistant saved step-start indices, excluding terminal state; every old40initials case,all policies/methods/h. Recompute force estimates at those states; separately use actual held input vs baseline feedback at the same state.',
        force_statistics='Per trajectory component/time RMS over64points, average8 within seed, then mean/sampleSD over5seeds; point values not independent repeats.',
        effectiveness='Reuse all old360case/1200trajectory archives; independently recompute errors/input effort before paired seed differences. No new training,gain choice,case selection or model change.',
        zero_initial_repeats='none/oracle curves are identical across5weight blocks and have1effective response. GR/GC have5model realizations,not5initial-condition replicates.',
        source_sha256=sha(Path(__file__)),old_inputs=registry())
    p=OUT/'protocol.json'
    if p.exists():
        existing=read(p)
        if existing!=config: raise RuntimeError('Frozen diagnostic protocol or source inputs changed')
    else: save(p,config)
    amendment=dict(before_matched_reference_computation=True,reason='Isolate nominal-model mismatch from the original large initial tracking error;retain all old results.',
        selected_initials='All40 original held-out initial states; each nominal reference starts from its matching plant initial.',
        nominal_reference='H0 plus same u_ref; DOP853 rtol2e-12 atol2e-14; first trajectory of each seed independently refined to rtol5e-14 atol1e-16.',
        plant='Unchanged H*',policies=POLICIES,methods=METHODS,step_divisors=[1,2],feedback='Original kp1,kd.5,total budget .2,weights unchanged',
        status='Mechanism-isolation diagnostic,not fresh-seed confirmation;never replaces original common-zero-reference results',
        primary='Paired same-seed tracking tip/state versus none and oracle,all cases retained',
        source_sha256=sha(Path(__file__)))
    p=OUT/'matched_initial_reference_amendment.json'
    if p.exists() and read(p)!=amendment: raise RuntimeError('Matched-reference amendment changed')
    if not p.exists(): save(p,amendment)
    return config,old

def verify_inputs(config):
    bad=[p for p,h in config['old_inputs'].items() if sha(ROOT/p)!=h]
    if bad:raise RuntimeError('Original inputs changed: '+str(bad[:8]))

def net_field(x,w):
    y=x/w['scale'];a=np.tanh(y@w['w1'].T+w['b1']);b=np.tanh(a@w['w2'].T+w['b2'])
    grad=w['energy_scale']*((w['w3']*(1-b*b))@w['w2']*(1-a*a))@w['w1']/w['scale']
    if bool(w['prior']):grad+=x*np.r_[w['nu'],w['nu']]
    return np.concatenate([grad[...,2:],-grad[...,:2]],axis=-1)

def point_metrics(points,policy,nu,d,weights):
    theta=points[...,:2]@d
    truth=-(np.sin(theta)-theta)[...,None]*d
    b=np.cos(theta)[...,None]*d
    if policy=='none': estimated=np.zeros_like(points)
    elif policy=='oracle': estimated=np.concatenate([np.zeros_like(truth),truth],axis=-1)
    else:
        estimated=net_field(points,weights)-np.concatenate([nu*points[...,2:],-nu*points[...,:2]],axis=-1)
    ff=-np.sum(b*estimated[...,2:],axis=-1)/np.sum(b*b,axis=-1)
    rem=truth+b*ff[...,None]
    orth=estimated[...,2:]+b*ff[...,None]
    # All callers use [trajectory,selectedtime,component].
    rms=lambda a:np.sqrt(np.mean(a*a,axis=(1,2)))
    per=dict(true_force_rms=rms(truth),remaining_force_rms=rms(rem),learned_force_error_rms=rms(estimated[...,2:]-truth),
        unrealizable_kinematic_rms=rms(estimated[...,:2]),unrealizable_orthogonal_force_rms=rms(orth),
        max_abs_ff=np.max(np.abs(ff),axis=1))
    return {k:v.tolist() for k,v in per.items()},ff,truth,b

def moments(values):
    a=np.array(values,dtype=float)
    return dict(mean=float(a.mean()),sample_sd=float(a.std(ddof=1)) if len(a)>1 else 0.,n=len(a))

def aggregate(rows,keys,metric_names):
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    output=[]
    for k,rs in groups.items():
        rr=dict(zip(keys,k));rr['seed_count']=len(rs)
        rr['metrics']={m:moments([r[m] for r in rs]) for m in metric_names}
        output.append(rr)
    return output

def trajectory_metrics(states,reference,controls,h,omega,tip):
    if reference.ndim==2: reference=np.repeat(reference[:,None,:],states.shape[1],axis=1)
    err=states-reference
    projected=np.einsum('tji,i->tj',err[:,:,:2],tip)*1000
    norm=np.sqrt(np.sum(reference*reference,axis=(0,2)))
    return dict(tracking_tip_rmse_mm=np.sqrt(np.mean(projected**2,axis=0)),
        tail_half_tracking_tip_rmse_mm=np.sqrt(np.mean(projected[len(states)//2:]**2,axis=0)),
        tracking_state_relative_l2=np.sqrt(np.sum(err*err,axis=(0,2)))/np.maximum(norm,1e-15),
        integral_u_squared_seconds=np.sum(controls*controls,axis=0)*h/omega,
        max_abs_u=np.max(np.abs(controls),axis=0))

def paired(rows):
    groups={}
    for r in rows:groups.setdefault((r['case'],r['step_divisor'],r['method'],r['seed']),{})[r['policy']]=r
    output=[]
    comparisons=[('none','oracle'),('none','GR'),('none','GC'),('GR','GC')]
    for (case,divisor,method,seed),g in groups.items():
        for a,b in comparisons:
            if a not in g or b not in g:continue
            for metric in ('tracking_tip_rmse_mm','tracking_state_relative_l2','integral_u_squared_seconds'):
                av,bv=g[a][metric],g[b][metric]
                output.append(dict(case=case,step_divisor=divisor,method=method,seed=seed,contrast=a+'-'+b,metric=metric,
                    difference=av-bv,relative_improvement=(av-bv)/av if av!=0 else 0.))
    aggregates=aggregate(output,['case','step_divisor','method','contrast','metric'],['difference','relative_improvement'])
    for a in aggregates:
        values=[r for r in output if all(r[k]==a[k] for k in ['case','step_divisor','method','contrast','metric'])]
        a['positive_seed_blocks']=sum(r['difference']>0 for r in values)
        a['interpretation']='positive means latter policy has smaller metric; descriptive,not significance'
        if a['case']=='zero_initial':a['repetition_note']='none/oracle baseline reuses one physical response; learned comparisons vary trained model'
    return output,aggregates

def analyze():
    config,old=plan();spec=read(ROOT/'results/learning/inputs/reference_model_spec.json')
    nu=np.array(spec['nu']);d=np.array(spec['slope_map_d']);omega=spec['omega0_rad_s']
    common_rows=[];own_rows=[];tracking_rows=[];largest_gap=0.;archives=0;traj_count=0
    for seed in SEEDS:
        data=load(ROOT/f'results/learning/data/seed_{seed}.npz')
        eligible=np.flatnonzero(data['test_times']<=config['common_prefix_seconds']+1e-13)
        indices=eligible[np.rint(np.linspace(0,len(eligible)-1,64)).astype(int)]
        assert len(np.unique(indices))==64
        points=data['test_reference_states'][:,indices,:]
        dest=OUT/'common_points'/f'seed_{seed}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,points=points,source_time_indices=indices,times_seconds=data['test_times'][indices])
        for policy in POLICIES:
            w=load(ROOT/f'results/neural_learning_10000/seed_{seed}/{policy}_weights.npz') if policy in ['GR','GC'] else None
            values,ff,truth,b=point_metrics(points,policy,nu,d,w)
            row=dict(seed=seed,policy=policy,point_count=512,actual_endpoint_seconds=float(data['test_times'][indices[-1]]),
                per_trajectory=values,**{k:float(np.mean(v)) for k,v in values.items()},
                max_abs_theta=float(np.max(np.abs(points[...,:2]@d))),
                max_modal_radius=float(np.max(np.sqrt(points[...,:2]**2+points[...,2:]**2))))
            common_rows.append(row)
    # Read every existing case for independent metric recomputation.
    for case in ['forty_initials','zero_initial']:
        for p in sorted((OLD/case).glob('hdiv_*/seed_*/*.json')):
            r=read(p);a=load(p.with_suffix('.npz'));archives+=1;traj_count+=a['states'].shape[1]
            if sha(p.with_suffix('.npz'))!=r['trajectory_sha256']:raise RuntimeError('Old control trajectory changed')
            vals=trajectory_metrics(a['states'],a['reference'],a['controls'],r['h_tau'],omega,a['tip_map_m'])
            for j,original in enumerate(r['per_trajectory']):
                for k,v in vals.items():largest_gap=max(largest_gap,abs(float(v[j])-original[k]))
            row=dict(case=case,seed=r['seed'],policy=r['policy'],method=r['method'],step_divisor=r['step_divisor'],
                **{k:float(np.mean(v)) for k,v in vals.items()},saturation_fraction=float(np.mean(a['diagnostics'][:,-1])),
                max_abs_compensation=float(np.max(np.abs(a['compensation']))))
            tracking_rows.append(row)
            if case!='forty_initials':continue
            selected=np.rint(np.linspace(0,len(a['states'])-2,64)).astype(int)
            points=a['states'][selected].transpose(1,0,2)
            policy=r['policy'];seed=r['seed']
            w=load(ROOT/f'results/neural_learning_10000/seed_{seed}/{policy}_weights.npz') if policy in ['GR','GC'] else None
            pm,ff,truth,b=point_metrics(points,policy,nu,d,w)
            # Recorded controls are the held inputs actually chosen at these step starts.
            step_indices=selected*r['stride'];applied=a['controls'][step_indices].T
            target=a['reference'][selected];times=a['times_tau'][selected]
            th=points[...,:2]@d;thr=target[:,:2]@d
            velocity=np.cos(th)*np.sum(d*nu*points[...,2:],axis=-1)
            vr=np.cos(thr)*np.sum(d*nu*target[:,2:],axis=-1)
            uref=.05*(1-np.exp(-(times/(.25*omega))**2))*np.sin(.85*times)
            base=uref[None,:]-(np.sin(th)-np.sin(thr)[None,:])-.5*(velocity-vr[None,:])
            expected=np.clip(base+ff,-.2,.2)
            gap=float(np.max(np.abs(applied-expected)))
            if gap>2e-12:raise RuntimeError('Recorded held input disagrees with independently reconstructed policy')
            effective=applied-np.clip(base,-.2,.2)
            after=truth+b*effective[...,None]
            pm['remaining_force_with_total_budget_rms']=np.sqrt(np.mean(after**2,axis=(1,2))).tolist()
            row=dict(seed=seed,policy=policy,method=r['method'],step_divisor=r['step_divisor'],point_count=len(selected)*points.shape[0],
                per_trajectory=pm,**{k:float(np.mean(v)) for k,v in pm.items()},reconstructed_command_max_gap=gap,
                max_abs_theta=float(np.max(np.abs(th))),max_modal_radius=float(np.max(np.sqrt(points[...,:2]**2+points[...,2:]**2))))
            own_rows.append(row)
    assert archives==360 and traj_count==1200,(archives,traj_count)
    if largest_gap>2e-12:raise RuntimeError('Independent old-metric mismatch')
    metrics=list(common_rows[0]['per_trajectory'])
    common_agg=aggregate(common_rows,['policy'],metrics)
    own_agg=aggregate(own_rows,['policy','method','step_divisor'],metrics+['remaining_force_with_total_budget_rms'])
    diffs,contrasts=paired(tracking_rows)
    effective=[]
    for div in [.5,1,2,4]:
        for method in METHODS:
            for policy in ['none','oracle']:
                arrays=[load(OLD/f'zero_initial/hdiv_{div:g}/seed_{seed}/{policy}_{method}.npz')['states'] for seed in SEEDS]
                same=all(np.array_equal(arrays[0],x) for x in arrays[1:])
                assert same
                effective.append(dict(case='zero_initial',step_divisor=div,method=method,policy=policy,saved_blocks=5,effective_physical_responses=1,identical=same))
    summary=dict(common_test=common_agg,own_closed_loop=own_agg,tracking_pairs=contrasts,
        original_cases_independently_recomputed=archives,original_trajectories_independently_recomputed=traj_count,
        maximum_recomputed_metric_gap=largest_gap,zero_baseline_repetition=effective,
        scope='Test-force projections at common saved held-out states; own-state residuals are distribution-dependent. No gain/weight selection. No general control superiority claim.')
    save(OUT/'common_test_force.json',dict(records=common_rows,aggregates=common_agg,protocol_sha256=sha(OUT/'protocol.json')))
    save(OUT/'own_closed_loop_force.json',dict(records=own_rows,aggregates=own_agg))
    save(OUT/'old_tracking_recomputed.json',dict(records=tracking_rows,paired_records=diffs,aggregates=contrasts))
    save(OUT/'summary.json',summary);verify_inputs(config)
    print('ANALYSIS_COMPLETE',archives,traj_count,'metric_gap',largest_gap,flush=True)

def matched_references(lrc,config,old,spec):
    h0=old['base_h_tau'];nbase=old['base_steps'];nu=np.array(spec['nu']);d=np.array(spec['slope_map_d']);omega=spec['omega0_rad_s'];end=nbase*h0
    fine_times=np.arange(2*nbase+1)*h0/2
    for seed in SEEDS:
        dest=OUT/'matched_references'/f'seed_{seed}.npz';diag=dest.with_suffix('.json')
        if dest.exists():
            m=read(diag)
            if m['trajectory_sha256']!=sha(dest):raise RuntimeError('Matched reference changed')
            continue
        initials=load(ROOT/f'results/learning/data/seed_{seed}.npz')['test_initials']
        trajectories=[];tight_gap=0.;started=time.perf_counter()
        def rhs(t,x):return np.r_[nu*x[2:],-nu*x[:2]+lrc.reference_input(t,omega)*d*np.cos(d@x[:2])]
        for j,x in enumerate(initials):
            sol=solve_ivp(rhs,(0,end),x,t_eval=fine_times,method='DOP853',rtol=2e-12,atol=2e-14)
            if not sol.success:raise RuntimeError(sol.message)
            trajectories.append(sol.y.T)
            if j==0:
                times=fine_times[::16]
                tight=solve_ivp(rhs,(0,end),x,t_eval=times,method='DOP853',rtol=5e-14,atol=1e-16)
                if not tight.success:raise RuntimeError(tight.message)
                tight_gap=float(np.max(np.abs(tight.y.T-sol.y.T[::16])))
                if tight_gap>2e-9:raise RuntimeError('Matched nominal reference not resolved')
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,reference_fine=np.stack(trajectories,axis=1),times_tau=fine_times,initials=initials)
        save(diag,dict(seed=seed,trajectory_sha256=sha(dest),first_reference_refinement_max_gap=tight_gap,elapsed_seconds=time.perf_counter()-started))
        print('MATCHED_REFERENCE',seed,'seconds',time.perf_counter()-started,'gap',tight_gap,flush=True)

def matched():
    config,old=plan();spec=read(ROOT/'results/learning/inputs/reference_model_spec.json');lrc=module('learned_residual_control')
    nu=np.array(spec['nu']);d=np.array(spec['slope_map_d']);omega=spec['omega0_rad_s'];tip=np.array(spec['tip_map_m'])
    matched_references(lrc,config,old,spec)
    for divisor in [1,2]:
        h=old['base_h_tau']/divisor;nsteps=old['base_steps']*divisor;stride=8*divisor
        for seed in SEEDS:
            a=load(OUT/f'matched_references/seed_{seed}.npz');ref=a['reference_fine'][::2] if divisor==1 else a['reference_fine']
            target=ref[::stride];times=np.arange(len(target))*stride*h
            for policy in POLICIES:
                kind=0 if policy=='none' else 1 if policy=='oracle' else 2;params=lrc.params_for(seed,policy)
                for method_i,method in enumerate(METHODS):
                    dest=OUT/f'matched_initial/hdiv_{divisor}/seed_{seed}';path=dest/f'{policy}_{method}.json'
                    if path.exists():
                        oldr=read(path)
                        if oldr['trajectory_sha256']!=sha(path.with_suffix('.npz')):raise RuntimeError('Matched trajectory changed')
                        continue
                    started=time.perf_counter();states=[];controls=[];ff=[];diagnostics=[]
                    for j,x in enumerate(a['initials']):
                        s,u,f,diag=lrc.simulate(x[None,:],h,nsteps,stride,method_i,kind,np.ascontiguousarray(ref[:,j]),nu,d,omega,params)
                        states.append(s[:,0]);controls.append(u[:,0]);ff.append(f[:,0]);diagnostics.append(diag[0])
                    states=np.stack(states,axis=1);controls=np.stack(controls,axis=1);ff=np.stack(ff,axis=1);diagnostics=np.array(diagnostics)
                    vals=trajectory_metrics(states,target,controls,h,omega,tip)
                    dest.mkdir(parents=True,exist_ok=True);npz=path.with_suffix('.npz')
                    np.savez_compressed(npz,states=states,reference=target,controls=controls,compensation=ff,diagnostics=diagnostics,times_tau=times,initials=a['initials'],tip_map_m=tip)
                    r=dict(case='matched_initial',seed=seed,policy=policy,method=method,step_divisor=divisor,h_tau=h,
                        status='completed',per_trajectory={k:v.tolist() for k,v in vals.items()},
                        **{k:float(np.mean(v)) for k,v in vals.items()},saturation_fraction=float(np.mean(diagnostics[:,-1])),
                        maximum_implicit_residual=float(np.max(diagnostics[:,0])),max_abs_compensation=float(np.max(np.abs(ff))),
                        elapsed_seconds=time.perf_counter()-started,trajectory_sha256=sha(npz),protocol_sha256=sha(OUT/'matched_initial_reference_amendment.json'))
                    save(path,r);print('MATCHED_CASE',divisor,seed,policy,method,r['tracking_tip_rmse_mm'],flush=True)
    summarize_matched();verify_inputs(config)

def summarize_matched():
    rows=[];gap=0.
    spec=read(ROOT/'results/learning/inputs/reference_model_spec.json');omega=spec['omega0_rad_s']
    for p in sorted((OUT/'matched_initial').glob('hdiv_*/seed_*/*.json')):
        r=read(p);a=load(p.with_suffix('.npz'))
        if sha(p.with_suffix('.npz'))!=r['trajectory_sha256']:raise RuntimeError('Matched cache changed')
        m=trajectory_metrics(a['states'],a['reference'],a['controls'],r['h_tau'],omega,a['tip_map_m'])
        for k,v in m.items():gap=max(gap,float(np.max(np.abs(v-r['per_trajectory'][k]))))
        row={k:r[k] for k in ['case','step_divisor','method','policy','seed','saturation_fraction','max_abs_compensation']}
        row.update({k:float(v.mean()) for k,v in m.items()});rows.append(row)
    assert len(rows)==120,len(rows)
    diffs,contrasts=paired(rows)
    agg=aggregate(rows,['case','step_divisor','method','policy'],['tracking_tip_rmse_mm','tracking_state_relative_l2','integral_u_squared_seconds','max_abs_u'])
    save(OUT/'matched_initial_summary.json',dict(records=rows,aggregates=agg,paired_records=diffs,paired_aggregates=contrasts,
        mechanism_diagnostic_not_fresh_confirmation=True,previous_common_zero_reference_results_retained=True,
        maximum_metric_recomputation_gap=gap))
    print('MATCHED_COMPLETE',len(rows),'cases',flush=True)

def verify():
    config,_=plan();verify_inputs(config)
    own=read(OUT/'own_closed_loop_force.json');common=read(OUT/'common_test_force.json')
    assert len(common['records'])==20 and len(own['records'])==120
    assert all(r['point_count']==512 for r in common['records']+own['records'])
    oracle=[r for r in common['records']+own['records'] if r['policy']=='oracle']
    assert max(r['remaining_force_rms'] for r in oracle)<1e-14
    assert max(r['reconstructed_command_max_gap'] for r in own['records'])<2e-12
    summary=read(OUT/'summary.json');assert summary['original_trajectories_independently_recomputed']==1200
    if (OUT/'matched_initial_summary.json').exists():
        summarize_matched()
        matched_summary=read(OUT/'matched_initial_summary.json')
        assert matched_summary['maximum_metric_recomputation_gap']<2e-12
    save(OUT/'independent_verification.json',dict(status='passed',original_cases_checked=360,original_trajectories_checked=1200,
        common_probe_points=2560,own_closed_loop_state_sets=120,matched_cases_checked=120 if (OUT/'matched_initial_summary.json').exists() else 0,
        original_inputs_unchanged=True,maximum_original_metric_gap=summary['maximum_recomputed_metric_gap'],
        caveat='Metrics independently recomputed from archived trajectories; matched simulations reuse original validated step implementation,not an independent integrator proof.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['plan','analyze','matched','verify','all'],default='all');a=p.parse_args()
    plan()
    if a.stage in ['analyze','all']:analyze()
    if a.stage in ['matched','all']:matched()
    if a.stage in ['verify','all']:verify()
