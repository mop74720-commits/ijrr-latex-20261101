"""B: conditional physical-bias sweep; never writes manuscript or old outputs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
sys.dont_write_bytecode=True
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'experiment_b'
PD=HERE/'physical_design'
SEEDS=[17,43,91,137,211]
FIELDS=[.03,.06,.12]
GROUPS=['H0','H4','GR','GC','Teacher']
SPEC=json.loads((ROOT/'results/learning/inputs/reference_model_spec.json').read_text(encoding='utf-8-sig'))

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
P=import_file('experiment_b_physical',PD/'build_models.py')
A=import_file('experiment_b_continuous',HERE/'experiment_a.py')
TRAINER=None
MODELS=None

def trainer():
    global TRAINER
    if TRAINER is None:
        TRAINER=import_file('experiment_b_original_trainer',ROOT/'code/train_neural_hnn.py')
        TRAINER.torch.set_num_threads(1)
    return TRAINER

def model_list():
    global MODELS
    if MODELS is not None:return MODELS
    design=json.loads((PD/'final_design.json').read_text(encoding='utf-8'))
    assert [v['field_T'] for v in design['levels']]==FIELDS
    MODELS=[]
    for field in FIELDS:
        model=P.build_model(field)
        name=f'B{round(1000*field):03d}mT';model['name']=name
        stored=load(PD/(name+'.npz'))
        for key in ('nu','d','omega','modal_scale','tip_map_m'):
            assert np.max(np.abs(model[key]-stored[key]))<1e-12,(name,key)
        if field==.03:
            # Preserve the exact existing experiment's float coefficients. The
            # FE reconstruction was separately verified to roundoff precision.
            model['nu']=np.asarray(SPEC['nu'],float)
            model['d']=np.asarray(SPEC['slope_map_d'],float)
            model['omega0']=float(SPEC['omega0_rad_s'])
            model['tip_map_m']=np.asarray(SPEC['tip_map_m'],float)
            model['energy_scale_J']=float(SPEC['energy_scale_J'])
        MODELS.append(model)
    return MODELS

def project(original,model):
    projected,diag=P.project_initials(original,model)
    if model['field_T']==.03:
        assert np.max(np.abs(projected-original))<2e-13
        projected=np.array(original,copy=True)
    return projected,diag

def plan():
    inputs={}
    for rel in ('code/train_neural_hnn.py','code/neural_physics_refinement.py','results/learning/inputs/reference_model_spec.json'):
        inputs[rel]=sha(ROOT/rel)
    for path in (Path(__file__),HERE/'experiment_a.py',PD/'build_models.py',PD/'final_protocol.json',PD/'final_design.json'):
        inputs[path.relative_to(ROOT).as_posix()]=sha(path)
    for model in model_list():inputs[(PD/(model['name']+'.npz')).relative_to(ROOT).as_posix()]=sha(PD/(model['name']+'.npz'))
    for seed in SEEDS:
        for rel in (f'results/learning/data/seed_{seed}.npz',*[f'results/neural_learning_10000/seed_{seed}/{group}_weights.npz' for group in ('GR','GC')]):inputs[rel]=sha(ROOT/rel)
    protocol=dict(scope='Conditional B0 pressure test within the original constant-dipole/full-cosine-of-approximate-slope model. No manuscript or GitHub writes.',
        fields_T=FIELDS,seeds=SEEDS,models=GROUPS,train_trajectories=12,validation_trajectories=4,test_trajectories=8,
        durations_seconds={'training':2.,'validation':4.,'evaluation':64.,'short_prefix':16.},
        Eq36='Constant mass conservative midpoint; regenerate each non-baseline field with its reconstructed nu,d,omega0,E0.',
        pairing='Same original full-FE displacement and velocity, mass projection to field-specific2mode basis; report projection defect.',
        sample_count='Exactly original per-trajectory counts,286train/571val; project original stride8 midpoint time fractions onto new integration grid. Baseline copied unchanged.',
        noise='Same paired canonical uniform[-1e-6,1e-6] array per seed/split across fields; not same SI noise; report equivalent physical and baseline-coordinate RMS.',
        architecture='Existing GR/GC each673 real parameters; identical10k Adam updates,batch512,original checkpoint subset2048 every100updates.',
        baseline='Re-use original10 selected weights and data after exact coefficient/noise/initial checks. Baseline float coefficients anchored to original spec after FE reconstruction roundoff checks.',
        H4='H0-(d.X)^4/24 with exact physical Taylor coefficient; no fitted parameters.',
        evaluation='Continuous DOP853 batch32dim,rtol1e-11 atol1e-13; common physical sampling dt=.005s,0..64s. H0 exact flow.',
        numerical_refinement='Preselected first test trajectory per seed/field/method,rtol5e-13 atol5e-15; retain state/frequency sensitivity.',
        geometry='All train/val/test initials must pass finite-FE all-space conservative energy slope screen<.15 before any fit. Supplied-state exact spatial slope/curvature checks also reported. Not outward interval or physical calibration.',
        physical_limitation='theta approximately slope omits geometric terms of the same quartic order as retained magnetic residual; constant-dipole field admissibility and full nonlinear/high-mode physical accuracy are not established.',
        primary=['64s continuous complete canonical state relativeL2','64s mode1 absolute frequency error','64s mode2 absolute frequency error'],
        secondary=['16s state','tipRMSE','half-window frequency stability','mode radii','nonlinear/linear force ratio','H4 truncation error','noiseSNR','training cost'],
        exploratory_rule='For each prescribed comparison,all3primary metrics need>=4/5 directional seed wins and mean paired relative improvement>=5%; not significance or population claim. Numerical uncertainty must be smaller than paired gains.',
        comparisons=['GR/H0','GC/H0','GR/H4','GC/H4','GC/GR'],statistics='Per trajectory then mean8withinseed then seed mean/sampleSD; no timepoint pseudoreplication.',input_sha256=inputs)
    path=OUT/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==protocol,'Existing B protocol differs'
    else:
        save(path,protocol)
        shutil.copyfile(__file__,OUT/'execution_script.py')
        save(OUT/'environment.json',dict(utc=datetime.now(timezone.utc).isoformat(),python=sys.version,numpy=np.__version__,script_sha256=sha(__file__)))
    return protocol

def check_sources(protocol):
    changed=[rel for rel,h in protocol['input_sha256'].items() if sha(ROOT/rel)!=h]
    if changed:raise RuntimeError('Frozen B input changed '+str(changed))
    return dict(files_checked=len(protocol['input_sha256']),all_unchanged=True)

def field(points,model):
    theta=points[...,:2]@model['d']
    return np.concatenate([points[...,2:]*model['nu'],-points[...,:2]*model['nu']-(np.sin(theta)-theta)[...,None]*model['d']],axis=-1)

def params_for(weights):
    return tuple(np.ascontiguousarray(weights[k]) for k in ('w1','b1','w2','b2','w3','b3','scale'))+(float(weights['energy_scale']),np.ascontiguousarray(weights['nu']),bool(weights['prior']))

def signal_metrics(arrays,model):
    x=arrays['train_midpoints'];theta=x[:,:2]@model['d'];linear=-x[:,:2]*model['nu'];nonlinear=-(np.sin(theta)-theta)[:,None]*model['d'];quartic=theta[:,None]**3*model['d']/6
    noise=arrays['train_noise'];noise_p=noise[:,2:]
    basis,_=project(np.eye(4),model);equivalent=noise@np.linalg.inv(basis)*(model['omega0']/SPEC['omega0_rad_s'])
    tip_noise=(noise[:,:2]@model['tip_map_m'])*model['omega0']
    accel_noise=noise_p*model['modal_scale']*model['omega']*model['omega0']
    return dict(nonlinear_to_linear_force_l2=float(np.linalg.norm(nonlinear)/np.linalg.norm(linear)),
        H4_remainder_to_linear_force_l2=float(np.linalg.norm(quartic-nonlinear)/np.linalg.norm(linear)),
        nonlinear_to_noise_force_l2=float(np.linalg.norm(nonlinear)/np.linalg.norm(noise_p)),
        nonlinear_force_component_rms=float(np.sqrt(np.mean(nonlinear**2))),label_noise_component_rms=float(np.sqrt(np.mean(noise_p**2))),
        noise_equivalent_baseline_canonical_derivative_component_rms=np.sqrt(np.mean(equivalent**2,axis=0)).tolist(),
        SI_tip_velocity_label_noise_rms_m_s=float(np.sqrt(np.mean(tip_noise**2))),
        SI_modal_acceleration_label_noise_component_rms=np.sqrt(np.mean(accel_noise**2,axis=0)).tolist(),max_abs_tip_slope=float(np.max(np.abs(theta))))

def prepare():
    protocol=plan();screens=[]
    for model in model_list():
        for seed in SEEDS:
            path=OUT/model['name']/f'seed_{seed}'/'data.npz';meta=path.with_suffix('.json')
            if path.exists():
                row=json.loads(meta.read_text(encoding='utf-8'));assert row['data_sha256']==sha(path);screens.append(row);continue
            old=load(ROOT/f'results/learning/data/seed_{seed}.npz');arrays={};diagnostics={};all_initials=[]
            for split,duration,offset in (('train',2.,0),('val',4.,100000)):
                initials,projection=project(old[split+'_initials'],model);all_initials.append(initials)
                if model['field_T']==.03:
                    for key,value in old.items():
                        if key.startswith(split+'_'):arrays[key]=value
                    clean_gap=float(np.max(np.abs(arrays[split+'_clean_labels']-field(arrays[split+'_midpoints'],model))))
                    diagnostics[split]=dict(reused=True,projection=projection,midpoint_label_rhs_gap=clean_gap)
                    assert clean_gap<5e-11
                    continue
                nsteps=int(np.ceil(duration*model['omega0']/.01));h=duration*model['omega0']/nsteps
                cayley,inverse=A.REFINE.cayley_preconditioner(h,model['nu'])
                states,residual,maxit,meanit=A.REFINE.teacher_rollout(initials,h,nsteps,1,cayley,inverse,model['nu'],model['d'])
                mids=(states[1:]+states[:-1])/2;clean=(states[1:]-states[:-1])/h
                old_nsteps=round(duration*SPEC['omega0_rad_s']/float(old[split+'_step_tau']))
                old_indices=np.arange(0,old_nsteps,8)
                expected=len(old[split+'_midpoints'])//len(initials);assert len(old_indices)==expected
                indices=np.rint((old_indices+.5)/old_nsteps*nsteps-.5).astype(int)
                assert len(np.unique(indices))==expected and indices.min()>=0 and indices.max()<nsteps
                x=mids[indices].transpose(1,0,2).reshape(-1,4);y=clean[indices].transpose(1,0,2).reshape(-1,4)
                noise=np.random.default_rng(seed+offset+10000).uniform(-1e-6,1e-6,y.shape)
                assert np.array_equal(noise,old[split+'_noise'])
                arrays.update({split+'_initials':initials,split+'_midpoints':x,split+'_labels':y+noise,split+'_clean_labels':y,
                    split+'_noise':noise,split+'_trajectory_id':np.repeat(np.arange(len(initials)),expected),split+'_step_tau':np.array(h),split+'_midpoint_step_indices':indices})
                gap=float(np.max(np.abs(y-field(x,model))));assert gap<5e-11
                diagnostics[split]=dict(reused=False,projection=projection,maximum_implicit_residual=float(residual),maximum_iterations=int(maxit),mean_iterations=float(meanit),midpoint_label_rhs_gap=gap,
                    samples_per_trajectory=expected,maximum_midpoint_time_fraction_difference=float(np.max(np.abs((indices+.5)/nsteps-(old_indices+.5)/old_nsteps))),geometry=P.geometry_checks(states,model))
            initials,projection=project(old['test_initials'],model);all_initials.append(initials)
            gate=P.energy_geometry_bound(np.concatenate(all_initials),model)
            assert gate['continuous_model_energy_screen_passed'],'Stop entire level/seed before fitting'
            arrays.update(test_initials=initials,nu=model['nu'],d=model['d'],omega0=np.array(model['omega0']),tip_map_m=model['tip_map_m'],energy_scale_J=np.array(model['energy_scale_J']))
            path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
            row=dict(seed=seed,level=model['name'],field_T=model['field_T'],data_sha256=sha(path),source_data_sha256=sha(ROOT/f'results/learning/data/seed_{seed}.npz'),diagnostics=diagnostics,
                test_projection=projection,energy_geometry_screen=gate,signals=signal_metrics(arrays,model))
            save(meta,row);screens.append(row)
            print('B_DATA',model['name'],seed,json.dumps(row['signals']),flush=True)
    save(OUT/'screening.json',dict(all_passed=True,records=screens,physical_validity_certified=False))
    save(OUT/'protected_after_prepare.json',check_sources(protocol))

def train():
    protocol=plan();screen=json.loads((OUT/'screening.json').read_text(encoding='utf-8'));assert screen['all_passed']
    t=trainer()
    for model in model_list():
        for seed in SEEDS:
            dest=OUT/model['name']/f'seed_{seed}';arrays=load(dest/'data.npz')
            for group in ('GR','GC'):
                record=dest/f'{group}_training.json'
                if record.exists():
                    row=json.loads(record.read_text(encoding='utf-8'));assert row['status']=='completed' and row['weights_sha256']==sha(dest/f'{group}_weights.npz');continue
                print('B_FIT_START',model['name'],seed,group,flush=True)
                if model['field_T']==.03:
                    source=ROOT/f'results/neural_learning_10000/seed_{seed}'
                    weights=load(source/f'{group}_weights.npz');assert np.array_equal(weights['nu'],model['nu'])
                    row=json.loads((source/f'{group}_metrics.json').read_text(encoding='utf-8'))
                    history=json.loads((source/f'{group}_training_history.json').read_text(encoding='utf-8'))
                    shutil.copyfile(source/f'{group}_weights.npz',dest/f'{group}_weights.npz')
                    reused=True
                else:
                    weights,row,history=t.train_one(seed,group,arrays,10000,512)
                    np.savez_compressed(dest/f'{group}_weights.npz',**weights);reused=False
                pred=t.numpy_energy_field(arrays['val_midpoints'],weights)[1]
                points=arrays['val_midpoints'][:128];compiled=A.batch_field(points.ravel(),2,params_for(weights),model['nu'],model['d']).reshape(-1,4)
                gap=float(np.max(np.abs(compiled-t.numpy_energy_field(points,weights)[1])));assert gap<2e-12
                row.update(seed=seed,group=group,status='completed',level=model['name'],baseline_weights_reused=reused,
                    clean_validation_relative_l2=float(np.linalg.norm(pred-arrays['val_clean_labels'])/np.linalg.norm(arrays['val_clean_labels'])),numpy_jit_field_gap=gap,
                    data_sha256=sha(dest/'data.npz'),weights_sha256=sha(dest/f'{group}_weights.npz'),protocol_sha256=sha(OUT/'protocol.json'))
                save(dest/f'{group}_history.json',history);save(record,row)
                print('B_FIT_DONE',model['name'],seed,group,'reused',reused,'seconds',row['training_seconds'],flush=True)
    save(OUT/'protected_after_train.json',check_sources(protocol))

def evaluate():
    protocol=plan();trainer()
    for model in model_list():
        for seed in SEEDS:
            dest=OUT/model['name']/f'seed_{seed}';record=dest/'evaluation.json';archive=dest/'continuous64s.npz'
            if record.exists():
                old=json.loads(record.read_text(encoding='utf-8'));assert old['status']=='completed' and old['archive_sha256']==sha(archive);continue
            arrays=load(dest/'data.npz');times_seconds=np.arange(12801)*.005;times=times_seconds*model['omega0']
            weights={g:load(dest/f'{g}_weights.npz') for g in ('GR','GC')};predictions={};diagnostics={};refined={}
            for group,kind in (('Teacher',0),('H4',1),('GR',2),('GC',2)):
                params=params_for(weights[group if group in weights else 'GR'])
                pred,diag=A.batch_solve(arrays['test_initials'],kind,params,model['nu'],model['d'],times)
                tight,td=A.batch_solve(arrays['test_initials'][:1],kind,params,model['nu'],model['d'],times,5e-13,5e-15)
                predictions[group]=pred;diagnostics[group]=diag
                refined[group]=dict(maximum_state_gap=float(np.max(np.abs(pred[:,:1]-tight))),modal_frequency_gap=np.abs(A.phase_frequency(pred[:,:1],times)-A.phase_frequency(tight,times))[0].tolist(),diagnostics=td)
                print('B_FLOW',model['name'],seed,group,diag['seconds'],flush=True)
            predictions['H0']=A.exact_linear(arrays['test_initials'],model['nu'],times)
            geometries={g:P.geometry_checks(predictions[g],model) for g in GROUPS}
            rows=[]
            for duration in (16,64):
                mask=times_seconds<=duration+1e-12
                for group in GROUPS:
                    for row in A.metric_rows(predictions[group][mask],predictions['Teacher'][mask],times[mask],model['tip_map_m']):
                        rows.append(dict(level=model['name'],field_T=model['field_T'],seed=seed,group=group,duration_seconds=duration,**row))
            np.savez_compressed(archive,times_seconds=times_seconds,times_tau=times,groups=np.array(GROUPS),predicted_states=np.array([predictions[g] for g in GROUPS]),test_initials=arrays['test_initials'],nu=model['nu'],d=model['d'],tip_map_m=model['tip_map_m'])
            save(record,dict(status='completed',level=model['name'],seed=seed,archive_sha256=sha(archive),metrics=rows,diagnostics=diagnostics,refinement=refined,geometry=geometries,
                protocol_sha256=sha(OUT/'protocol.json'),weights_sha256={g:sha(dest/f'{g}_weights.npz') for g in weights}))
    save(OUT/'protected_after_evaluate.json',check_sources(protocol))

def summary():
    aggregates=[];comparisons=[];records=[]
    metrics=('state_relative_l2','frequency_abs_error_1','frequency_abs_error_2','tip_rmse_mm')
    for model in model_list():
        seed_values={}
        for seed in SEEDS:
            p=OUT/model['name']/f'seed_{seed}'/'evaluation.json'
            if not p.exists():continue
            record=json.loads(p.read_text(encoding='utf-8'));assert record['status']=='completed';records.append(record)
            for duration in (16,64):
                for group in GROUPS:
                    selected=[x for x in record['metrics'] if x['duration_seconds']==duration and x['group']==group]
                    seed_values[(seed,duration,group)]={metric:float(np.mean([r[metric] for r in selected])) for metric in metrics}
        for duration in (16,64):
            for group in GROUPS:
                for metric in metrics:
                    values=[seed_values[(s,duration,group)][metric] for s in SEEDS if (s,duration,group) in seed_values]
                    if len(values)!=5:continue
                    aggregates.append(dict(level=model['name'],duration_seconds=duration,group=group,metric=metric,mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)),seed_values=values))
            for candidate,baseline in (('GR','H0'),('GC','H0'),('GR','H4'),('GC','H4'),('GC','GR')):
                for metric in metrics:
                    pairs=[[seed_values[(s,duration,candidate)][metric],seed_values[(s,duration,baseline)][metric]] for s in SEEDS if (s,duration,candidate) in seed_values]
                    if len(pairs)!=5:continue
                    values=np.array(pairs);improvement=1-values[:,0]/values[:,1]
                    comparisons.append(dict(level=model['name'],duration_seconds=duration,candidate=candidate,baseline=baseline,metric=metric,
                        candidate_wins=int(np.sum(improvement>0)),mean_paired_relative_improvement=float(improvement.mean()),paired_relative_improvements=improvement.tolist(),
                        passed_exploratory_rule=bool(np.sum(improvement>0)>=4 and improvement.mean()>=.05)))
    save(OUT/'summary.json',dict(completed_seed_levels=len(records),expected_seed_levels=15,aggregates=aggregates,comparisons=comparisons,
        scope='Conditional approximate-model sweep. No physical-admissibility or generic CHNN-superiority proof.',physical_warning='Slope-angle geometric approximation affects the same quartic order. H4 is the simple parameter-free competing approximation.'))
    print('B_SUMMARY',len(records),'of15seedlevels',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('plan','prepare','train','evaluate','summary','all'),default='plan');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        if args.stage=='plan':plan()
        if args.stage in ('prepare','all'):prepare()
        if args.stage in ('train','all'):train()
        if args.stage in ('evaluate','all'):evaluate()
        if args.stage in ('summary','all'):summary()
