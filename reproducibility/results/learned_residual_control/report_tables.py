"""Create compact report tables by independently reading retained metrics."""
from pathlib import Path
import json,csv,hashlib,math
import numpy as np
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
summary=read(HERE/'summary.json')
assert summary['records_count']==360,summary['records_count']
verification=read(HERE/'independent_verification.json')
assert verification['trajectories_checked']==1200
projection=read(HERE/'residual_projection_verification.json')
full=[]
for r in summary['aggregates']:
 if r['case']!='forty_initials':continue
 row={k:r[k] for k in ['case','step_divisor','policy','method']}
 if row['method']=='I':row['display_method']='I/G'
 else:row['display_method']=row['method']
 for k in ['tracking_tip_rmse_mm','tracking_state_relative_l2','integral_u_squared_seconds','saturation_fraction','max_abs_u','max_abs_compensation']:
  row[k+'_mean']=r['metrics'][k]['mean'];row[k+'_sample_sd']=r['metrics'][k]['sample_sd']
 pp=[x for x in projection['records'] if x['policy']==r['policy']]
 for k in ['true_momentum_residual_rms','after_compensation_momentum_residual_rms','unactuated_kinematic_rms','orthogonal_momentum_component_rms']:
  row[k+'_mean']=float(np.mean([x[k] for x in pp]));row[k+'_sample_sd']=float(np.std([x[k] for x in pp],ddof=1))
 full.append(row)
with (HERE/'manuscript_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(full[0]));writer.writeheader();writer.writerows(full)
contrasts=[]
for div in [1,2]:
 for method in ['I','II','III']:
  for left,right in [('none','oracle'),('none','GR'),('none','GC'),('GR','GC')]:
   for metric in ['tracking_tip_rmse_mm','tracking_state_relative_l2','integral_u_squared_seconds']:
    rows=[]
    for seed in [17,43,91,137,211]:
     base=HERE/'forty_initials'/f'hdiv_{div}'/f'seed_{seed}'
     a=read(base/f'{left}_{method}.json')['mean'][metric]
     b=read(base/f'{right}_{method}.json')['mean'][metric]
     rows.append(a-b)
    contrasts.append(dict(step_divisor=div,method=method,contrast=left+'-'+right,metric=metric,
     mean=float(np.mean(rows)),sample_sd=float(np.std(rows,ddof=1)),right_lower_blocks=sum(x>0 for x in rows),seedwise=rows))
orders=[]
for policy in ['none','oracle','GR','GC']:
 for method in ['I','II','III']:
  select=sorted([r for r in summary['aggregates'] if r['case']=='zero_initial' and r['policy']==policy and r['method']==method],key=lambda r:r['step_divisor'])
  errs=[r['metrics']['continuous_loop_state_l2']['mean'] for r in select]
  orders.append(dict(policy=policy,method=method,divisors=[r['step_divisor'] for r in select],
   continuous_loop_state_l2=errs,observed_orders=[math.log(errs[j]/errs[j+1],2) for j in range(3)],
   scope='Complete sampled-feedback numerical response versus each matching continuous-feedback flow; includes input hold, not an order theorem.'))
zero=[r for r in summary['aggregates'] if r['case']=='zero_initial']
save(HERE/'manuscript_tables.json',dict(forty_initials=full,paired_contrasts=contrasts,zero_initial_response=zero,closed_loop_observed_orders=orders,
 repetition='GR/GC depend on five saved model fits; forty starts blocked8 per fit. Zero-start none/oracle are identical baseline responses reused, not5independent physical replicates.',
 projection_domain='Full original validation midpoints, not controlled test trajectories',independent_verification=verification))
print('40 initial I/G h0:')
for r in full:
 if r['step_divisor']==1 and r['method']=='I':
  print(r['policy'],r['tracking_tip_rmse_mm_mean'],r['tracking_tip_rmse_mm_sample_sd'],r['tracking_state_relative_l2_mean'],r['integral_u_squared_seconds_mean'])
print('h0 I/G paired:')
for r in contrasts:
 if r['step_divisor']==1 and r['method']=='I' and r['metric']=='tracking_tip_rmse_mm':print(r)
print('zero I orders:',[(r['policy'],r['observed_orders']) for r in orders if r['method']=='I'])
