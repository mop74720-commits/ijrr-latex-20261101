"""Read-only numerical reconstruction; never imports the experiment module."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import sys
import numpy as np
import scipy
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def metrics(x,y,tip):
    d=x-y
    total=sum(float(v*v) for v in d.ravel())
    norm=sum(float(v*v) for v in y.ravel())
    projected=d[:,:2]@tip*1000
    return dict(state_relative_l2=math.sqrt(total/max(norm,1e-30)),state_absolute_rms=math.sqrt(total/d.size),
        endpoint_state_l2=math.sqrt(sum(float(v*v) for v in d[-1])),tip_rmse_mm=math.sqrt(statistics.mean(float(v*v) for v in projected)))

protocol=read(OUT/'protocol.json');summary=read(OUT/'summary.json')
sources={v['path']:v['sha256'] for v in protocol['inputs']}
for v in read(OUT/'reused_inputs.json'):sources[v['path']]=v['sha256']
before={p:sha(ROOT/p) for p in sources}
assert before==sources
gap=0.;record_count=0;paths={};fresh_held=0;old_reference_gap=0.
spec=read(ROOT/'results/learning/inputs/reference_model_spec.json')
tip=np.array(spec['tip_map_m'])
for branch in ('smooth_external','sampled_feedback','frozen_sequence'):
    rows=read(OUT/f'{branch}_records.json')
    for row in rows:
        p=ROOT/row['trajectory_path'];assert sha(p)==row['trajectory_sha256'];paths[row['trajectory_path']]=sha(p)
        a=load(p)
        if branch=='smooth_external':
            x=a['states'][:,row['initial_index']];y=a['reference'][:,row['initial_index']]
        elif branch=='sampled_feedback':
            x=a['states']
            if x.ndim==3:x=x[:,0]
            # Newly generated4h0 caches contain the shared tight continuous reference.
            # Reused old archives bind a separate continuous flow through this sidecar.
            if row['reused']:
                path=ROOT/f"results/learned_residual_control/continuous_zero/seed_{row['seed']}/{row['policy']}.npz"
                yy=load(path);oldref=yy['states']
                newpath=OUT/'frozen_sequence'/f"seed_{row['seed']}_{row['policy']}_hmult{row['h_multiplier']:g}_I.npz"
                y=load(newpath)['continuous_feedback_reference']
                old_reference_gap=max(old_reference_gap,float(np.max(np.abs(oldref-y))))
                assert np.max(np.abs(oldref-y))<1e-9
            else:y=a['continuous_reference']
        else:
            x=a['states'];y=a['held_reference'];fresh_held+=1
            refine=float(np.max(np.abs(a['held_reference']-a['held_reference_sub2'])))
            assert refine<2e-10
            for key,value in metrics(a['held_reference'],a['continuous_feedback_reference'],a['tip_map_m']).items():
                assert abs(value-row['hold_error_against_continuous_feedback'][key])<2e-12
        vals=metrics(x,y,a.get('tip_map_m',tip))
        local=max(abs(vals[k]-row[k]) for k in vals)
        gap=max(gap,local);assert local<2e-12,(branch,local)
        record_count+=1
        if branch=='frozen_sequence':
            # Direct Hamiltonian midpoint residual on retained output windows.
            # Full internal controls are saved, whereas states are output-subsampled;
            # only shared sequence identity can be checked across method archives.
            for method in ('I','II','III'):
                sibling=OUT/'frozen_sequence'/f"seed_{row['seed']}_{row['policy']}_hmult{row['h_multiplier']:g}_{method}.npz"
                aa=load(sibling)
                assert np.array_equal(aa['controls'],a['controls'])
                assert np.array_equal(aa['held_reference'],a['held_reference'])

scalar=read(OUT/'scalar_m2_records.json')
mass=2.;amp=.05;freq=.7
aug=np.zeros((4,4));aug[0,1]=1/mass;aug[1,0]=-1;aug[1,2]=amp;aug[2,3]=freq;aug[3,2]=-freq
for row in scalar:
    a=load(ROOT/row['path']);assert sha(ROOT/row['path'])==row['sha256']
    expected=np.array([(expm(t*aug)@np.array([.1,.2,0.,1.]))[:2] for t in a['times']])
    assert np.max(np.abs(expected-a['reference']))<1e-14
    error=float(np.sqrt(np.sum((a['states'][-1]-expected[-1])**2)))
    assert abs(error-row['endpoint_state_l2'])<1e-13

# Independently derive each h-halving rate from archived errors.
for v in summary['convergence']:
    errors=v['errors'];rates=[math.log(errors[i-1]/errors[i],2) for i in range(1,len(errors))]
    assert max(abs(x-y) for x,y in zip(rates,v['orders']))<1e-12
for v in summary['scalar']:
    errors=v['endpoint_errors'];rates=[math.log(errors[i-1]/errors[i],2) for i in range(1,len(errors))]
    assert max(abs(x-y) for x,y in zip(rates,v['orders']))<1e-12

assert all(sha(ROOT/p)==h for p,h in before.items())
output=dict(status='passed',records_checked=record_count,scalar_exact_trajectories_checked=len(scalar),
    distinct_numerical_archives=len(paths),frozen_method_archives_checked=fresh_held,
    exact_metric_reconstruction_max_gap=gap,old_loose_vs_shared_tight_reference_max_gap=old_reference_gap,
    input_hashes_unchanged=True,inputs_checked=len(before),input_hashes=before,
    held_reference_refinement_max=summary['held_reference_refinement'],
    environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
    scope='Finite numerical checks and observed rates, not a rigorous global error bound.')
(OUT/'independent_verification.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in output.items() if k!='input_hashes'},ensure_ascii=False,indent=2))
