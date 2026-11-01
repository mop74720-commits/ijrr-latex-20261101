"""Read-only numerical audit, independent of the analysis implementation."""
from pathlib import Path
import hashlib,json,os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SEEDS=[17,43,91,137,211]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def energy(x,w):
    a=np.tanh((x/w['scale'])@w['w1'].T+w['b1'])
    a=np.tanh(a@w['w2'].T+w['b2'])
    out=w['energy_scale']*(a@w['w3']+w['b3'][0])
    if bool(w['prior']):out+=.5*np.sum(x*x*np.r_[w['nu'],w['nu']],axis=-1)
    return out
def exact_learned_field(x,w):
    original_shape=x.shape;x=x.reshape(-1,4)
    gradient=np.empty_like(x)
    for j in range(4):
        p=x.astype(complex);p[:,j]+=1e-25j
        gradient[:,j]=energy(p,w).imag/1e-25
    return np.c_[gradient[:,2:],-gradient[:,:2]].reshape(original_shape)

def main():
    spec=read(ROOT/'results/learning/inputs/reference_model_spec.json')
    nu=np.array(spec['nu']);d=np.array(spec['slope_map_d']);omega=spec['omega0_rad_s']
    source=read(HERE/'protocol.json');before={p:sha(ROOT/p) for p in source['old_inputs']}
    assert before==source['old_inputs']
    common=read(HERE/'common_test_force.json');own=read(HERE/'own_closed_loop_force.json')
    maximum_force_gap=0.;cases=0
    for typ,rows in [('common',common['records']),('own',own['records'])]:
        for r in rows:
            if typ=='common':points=load(HERE/f'common_points/seed_{r["seed"]}.npz')['points']
            else:
                p=ROOT/f'results/learned_residual_control/forty_initials/hdiv_{r["step_divisor"]:g}/seed_{r["seed"]}/{r["policy"]}_{r["method"]}.npz'
                a=load(p);idx=np.rint(np.linspace(0,len(a['states'])-2,64)).astype(int);points=a['states'][idx].transpose(1,0,2)
            theta=points[...,:2]@d;truth=-(np.sin(theta)-theta)[...,None]*d;b=np.cos(theta)[...,None]*d
            if r['policy']=='none':estimate=np.zeros_like(points)
            elif r['policy']=='oracle':estimate=np.concatenate([np.zeros_like(truth),truth],axis=-1)
            else:
                w=load(ROOT/f'results/neural_learning_10000/seed_{r["seed"]}/{r["policy"]}_weights.npz')
                estimate=exact_learned_field(points,w)-np.concatenate([nu*points[...,2:],-nu*points[...,:2]],axis=-1)
            ff=-np.einsum('tki,tki->tk',b,estimate[...,2:])/np.einsum('tki,tki->tk',b,b)
            rms=lambda x:np.sqrt(np.sum(x*x,axis=(1,2))/(x.shape[1]*x.shape[2]))
            independently=dict(true_force_rms=rms(truth),remaining_force_rms=rms(truth+b*ff[...,None]),
                learned_force_error_rms=rms(estimate[...,2:]-truth),unrealizable_kinematic_rms=rms(estimate[...,:2]),
                unrealizable_orthogonal_force_rms=rms(estimate[...,2:]+b*ff[...,None]),max_abs_ff=np.max(abs(ff),axis=1))
            for k,x in independently.items():maximum_force_gap=max(maximum_force_gap,float(np.max(np.abs(x-r['per_trajectory'][k]))))
            cases+=1
    assert cases==140 and maximum_force_gap<2e-12,(cases,maximum_force_gap)
    largest=0.;count=0;matched=0
    for folder in [ROOT/'results/learned_residual_control/forty_initials',ROOT/'results/learned_residual_control/zero_initial',HERE/'matched_initial']:
        for p in folder.glob('hdiv_*/seed_*/*.json'):
            r=read(p);a=load(p.with_suffix('.npz'));assert sha(p.with_suffix('.npz'))==r['trajectory_sha256']
            states=a['states'];target=a['reference']
            if target.ndim==2:target=target[:,None,:]
            err=states-target
            tip=np.einsum('tji,i->tj',err[:,:,:2],a['tip_map_m'])*1e3
            denom=np.sqrt(np.sum(target*target,axis=(0,2)))
            x=dict(tracking_tip_rmse_mm=np.sqrt(np.sum(tip*tip,axis=0)/len(tip)),
                tracking_state_relative_l2=np.sqrt(np.sum(err*err,axis=(0,2)))/np.maximum(denom,1e-15),
                integral_u_squared_seconds=np.sum(a['controls']*a['controls'],axis=0)*r['h_tau']/omega)
            if r['case']=='matched_initial':
                for k,v in x.items():largest=max(largest,float(np.max(abs(v-r['per_trajectory'][k]))))
                matched+=states.shape[1]
            else:
                for j,row in enumerate(r['per_trajectory']):
                    for k,v in x.items():largest=max(largest,abs(float(v[j])-row[k]))
            count+=states.shape[1]
    assert matched==960 and count==2160 and largest<2e-12,(matched,count,largest)
    after={p:sha(ROOT/p) for p in source['old_inputs']};assert before==after
    report=dict(status='passed',force_state_sets_checked=cases,force_gradient_method='Complex-step differentiation of exported scalar energy; no import of analysis code',
        maximum_force_metric_gap=maximum_force_gap,trajectory_metrics_checked=count,matched_trajectories_checked=matched,
        maximum_trajectory_metric_gap=largest,original_input_hashes_unchanged=True,
        original_input_count=len(before),limitations='Archive and energy differentiation audit; no claim of independent physical calibration or KAM/control stability proof.')
    (HERE/'independent_verification_full.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
