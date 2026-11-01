"""Independent saved-artifact audit; does not import the experiment module."""
from pathlib import Path
import csv
import hashlib
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/variable_mass_learning'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def physical(x):
    q,p=x[...,:2],x[...,2:]
    return np.concatenate([q+q**3,p/(1+3*q*q)],axis=-1)
def h0(x,nu):
    x=physical(x)
    return .5*np.sum(nu*(x[...,:2]**2+x[...,2:]**2),axis=-1)
def true_energy(x,nu,d):
    theta=physical(x)[...,:2]@d
    return h0(x,nu)+2*np.sin(theta/2)**2-.5*theta**2
def network_energy(x,w):
    z=x/w['scale']
    z=np.tanh(z@w['w1'].T+w['b1'])
    z=np.tanh(z@w['w2'].T+w['b2'])
    energy=w['energy_scale']*(z@w['w3']+w['b3'][0])
    if bool(w['prior']): energy=energy+h0(x,w['nu'])
    return energy
def complex_gradient_field(x,fun):
    columns=[]
    for j in range(4):
        perturbed=x.astype(complex)
        perturbed[...,j]+=1e-25j
        columns.append(np.imag(fun(perturbed))/1e-25)
    grad=np.stack(columns,axis=-1)
    return np.concatenate([grad[...,2:],-grad[...,:2]],axis=-1)
def metrics(pred,target,tip):
    error=pred-target
    state=np.sqrt(np.sum(error**2,axis=(1,2))/np.sum(target**2,axis=(1,2)))
    output=np.sqrt(np.mean((error[...,:2]@tip*1000)**2,axis=1))
    return float(state.mean()),float(output.mean())

def main():
    cfg=read(OUT/'protocol.json')
    inputs={}
    for rel,digest in cfg['source_sha256'].items():
        assert sha(ROOT/rel)==digest,rel
    summary=read(OUT/'summary.json')
    assert summary['prediction_count']==40
    max_label_gap=0.
    max_checkpoint_gap=0.
    max_metric_gap=0.
    max_prior_scale_gap=0.
    rows=[]
    metrics_by_key={}
    for seed in cfg['seeds']:
        dataset=OUT/'data'/f'seed_{seed}.npz'
        a=load(dataset)
        meta=read(dataset.with_suffix('.json'))
        assert sha(dataset)==meta['dataset_sha256']
        original=ROOT/'results/learning/data'/f'seed_{seed}.npz'
        assert sha(original)==meta['source_initials_sha256']
        old=load(original)
        for split in ('train','val'):
            assert np.max(np.abs(physical(a[f'{split}_initials'])-old[f'{split}_initials']))<2e-15
            points=a[f'{split}_midpoints']
            exact=complex_gradient_field(points,lambda x:true_energy(x,a['nu'],a['d']))
            label_gap=float(np.max(np.abs(exact-a[f'{split}_clean_labels'])))
            max_label_gap=max(max_label_gap,label_gap)
            assert label_gap<1e-11
            assert np.max(np.abs(a[f'{split}_noise']))<=1e-6
            assert np.max(np.abs(a[f'{split}_labels']-a[f'{split}_clean_labels']-a[f'{split}_noise']))<2e-15
        assert np.max(np.abs(physical(a['test_initials'])-old['test_initials']))<2e-15
        for group in cfg['groups']:
            dest=OUT/f'seed_{seed}'
            weightpath=dest/f'{group}_weights.npz'
            w=load(weightpath)
            train=read(dest/f'{group}_training.json')
            assert train['status']=='completed' and train['real_parameter_count']==673 and train['training_steps']==10000
            assert train['weights_sha256']==sha(weightpath) and train['dataset_sha256']==sha(dataset)
            assert bool(w['prior'])==group.startswith('G') and float(w['mass_alpha'])==1.
            scale=np.maximum(np.sqrt(np.mean(a['train_midpoints']**2,axis=0)),1e-4)
            max_prior_scale_gap=max(max_prior_scale_gap,float(np.max(np.abs(scale-w['scale']))))
            assert np.array_equal(scale,w['scale'])
            if group in ('C','GC'):
                for name,n,m in [('w1',16,2),('w2',16,16)]:
                    value=w[name]
                    assert np.max(np.abs(value[:n,:m]-value[n:,m:]))<1e-15
                    assert np.max(np.abs(value[:n,m:]+value[n:,:m]))<1e-15
            indices=np.random.default_rng(seed+3000).choice(len(a['val_midpoints']),size=min(2048,len(a['val_midpoints'])),replace=False)
            output_scale=np.maximum(np.sqrt(np.mean(a['train_labels']**2,axis=0)),1e-6)
            predicted=complex_gradient_field(a['val_midpoints'][indices],lambda x:network_energy(x,w))
            mse=float(np.mean(((predicted-a['val_labels'][indices])/output_scale)**2))
            gap=abs(mse-train['validation_normalized_mse'])
            max_checkpoint_gap=max(max_checkpoint_gap,gap)
            assert gap<max(1e-20,abs(mse)*1e-9)
            history=read(dest/f'{group}_history.json')
            best=min(history,key=lambda row:row['validation_normalized_mse'])
            assert best['step']==train['selected_validation_step']
            for divisor in (1,4):
                p=dest/f'{group}_h{divisor}.npz'
                archive=load(p)
                row=read(p.with_suffix('.json'))
                assert sha(p)==row['trajectory_sha256'] and sha(weightpath)==row['weights_sha256']
                assert archive['predicted_physical_states'].shape==(8,914,4)
                assert np.array_equal(archive['times_seconds'],a['test_times'])
                physical_pred=physical(archive['predicted_chart_states'])
                assert np.max(np.abs(physical_pred-archive['predicted_physical_states']))<2e-15
                assert np.array_equal(physical(a['test_reference_states']),archive['continuous_physical_reference'])
                if divisor==1:
                    assert np.array_equal(physical(a['test_teacher_states']),archive['physical_teacher'])
                metric_values={}
                for prefix,target in [('continuous',archive['continuous_physical_reference']),('teacher',archive['physical_teacher'])]:
                    state,tip=metrics(physical_pred,target,a['tip_map_m'])
                    for key,val in [(prefix+'_state_relative_l2_mean',state),(prefix+'_tip_rmse_mm_mean',tip)]:
                        max_metric_gap=max(max_metric_gap,abs(row[key]-val))
                        assert abs(row[key]-val)<1e-12
                        metric_values[key]=val
                if group=='R':
                    baseline=read(dest/f'baselines_h{divisor}.json')['H0']
                    for prefix,target in [('continuous',archive['continuous_physical_reference']),('teacher',archive['physical_teacher'])]:
                        state,tip=metrics(archive['physical_H0'],target,a['tip_map_m'])
                        assert abs(baseline[prefix+'_state_relative_l2_mean']-state)<1e-12
                        assert abs(baseline[prefix+'_tip_rmse_mm_mean']-tip)<1e-12
                metrics_by_key[(seed,divisor,group)]=metric_values
                inputs[str(p.relative_to(ROOT))]=sha(p)
                inputs[str(p.with_suffix('.json').relative_to(ROOT))]=sha(p.with_suffix('.json'))
                rows.append(dict(seed=seed,group=group,divisor=divisor,**metric_values))
    for record in summary['paired_contrasts']:
        values=[]
        for seed in cfg['seeds']:
            r,c,gr,gc=[metrics_by_key[(seed,record['divisor'],g)][record['metric']] for g in cfg['groups']]
            contrasts={'R-GR':r-gr,'C-GC':c-gc,'R-C':r-c,'GR-GC':gr-gc,'(GR-GC)-(R-C)':(gr-gc)-(r-c)}
            values.append(contrasts[record['contrast']])
        assert abs(float(np.mean(values))-record['mean'])<1e-12
        assert abs(float(np.std(values,ddof=1))-record['sample_sd'])<1e-12
        assert int(np.sum(np.array(values)>0))==record['positive_seeds']
    assert len(summary['paired_contrasts'])==40
    assert all(sha(ROOT/rel)==digest for rel,digest in inputs.items())
    result=dict(status='passed',fits_checked=20,predictions_checked=40,trajectories_checked=320,baseline_trajectories_checked=80,contrast_aggregates_checked=40,
                independent_method='Complex-step gradient of the exported tanh energy and analytic pullback H0; saved NPZ recomputation without importing experiment code',
                max_label_gradient_gap=max_label_gap,max_checkpoint_mse_gap=max_checkpoint_gap,max_metric_gap=max_metric_gap,
                max_training_input_scale_gap=max_prior_scale_gap,input_hashes=inputs,protocol_sha256=sha(OUT/'protocol.json'),
                audit_script_sha256=sha(__file__))
    (OUT/'independent_verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (OUT/'independently_recomputed_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('INDEPENDENT_VERIFICATION_PASSED',json.dumps({k:v for k,v in result.items() if k!='input_hashes'}),flush=True)

if __name__=='__main__': main()
