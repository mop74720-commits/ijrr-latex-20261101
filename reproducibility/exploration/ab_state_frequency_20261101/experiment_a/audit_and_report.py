"""Independent trajectory/phase audit and plain-language report for A."""
from pathlib import Path
import hashlib
import json
import statistics
import shutil
import numpy as np

OUT=Path(__file__).resolve().parent
HERE=OUT.parent
ROOT=HERE.parents[1]
OLD=ROOT/'exploration/complex_advantage_20261101/fine_step'
SEEDS=[17,43,91,137,211]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def frequency(states,times):
    phase=np.unwrap(np.arctan2(-states[...,2:],states[...,:2]),axis=0)
    step=np.diff(times);x=(np.arange(len(step))+.5)/len(step)
    weights=np.exp(-1/(x*(1-x)))
    return np.einsum('t,tnj->nj',weights/weights.sum(),np.diff(phase,axis=0))/float(np.mean(step))
def primary(pred,ref,times,tip):
    delta=pred-ref
    state=np.linalg.norm(delta,axis=(0,2))/np.linalg.norm(ref,axis=(0,2))
    tip=np.sqrt(np.mean((np.einsum('tni,i->tn',delta[...,:2],tip)*1000)**2,axis=0))
    freq=np.abs(frequency(pred,times)-frequency(ref,times))
    return {'state_relative_l2':state,'tip_rmse_mm':tip,'frequency_abs_error_1':freq[:,0],'frequency_abs_error_2':freq[:,1]}

protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
for rel,digest in protocol['input_sha256'].items():assert sha(ROOT/rel)==digest,rel
environment=json.loads((OUT/'environment.json').read_text(encoding='utf-8'))
assert sha(HERE/'experiment_a.py')==environment['script_sha256']
snapshot=OUT/'execution_script.py'
if snapshot.exists():assert sha(snapshot)==environment['script_sha256']
else:shutil.copy2(HERE/'experiment_a.py',snapshot)
omega0=json.loads((ROOT/'results/learning/inputs/reference_model_spec.json').read_text(encoding='utf-8-sig'))['omega0_rad_s']
saved_rows=json.loads((OUT/'saved16s_trajectory_metrics.json').read_text(encoding='utf-8'))
metric_gap=0.;frequency_exact_H0_gap=0.;records_checked=0;resolution=[];refinement=[]
all_archives={}
for seed in SEEDS:
    long_path=OUT/f'seed_{seed}/continuous64s.npz';long=load(long_path)
    record=json.loads(long_path.with_suffix('.json').read_text(encoding='utf-8'))
    assert sha(long_path)==record['archive_sha256']
    all_archives[long_path.relative_to(ROOT).as_posix()]=sha(long_path)
    frequency_exact_H0_gap=max(frequency_exact_H0_gap,float(np.max(np.abs(frequency(long['predicted_states'][0],long['times_tau'])-long['nu']))))
    for group,pred in zip(long['groups'],long['predicted_states']):
        independently=primary(pred,long['continuous_reference'],long['times_tau'],long['tip_map_m'])
        rows=[r for r in record['metrics'] if r['group']==group]
        assert len(rows)==8
        for key,values in independently.items():
            gap=float(np.max(np.abs(values-np.array([r[key] for r in rows]))));metric_gap=max(metric_gap,gap)
            assert gap<2e-10,(seed,group,key,gap)
        records_checked+=8
    for group,r in record['reference_refinement'].items():
        refinement.append({'seed':seed,'group':group,**r})
    for div in (32,64):
        path=OLD/f'seed_{seed}/h{div}_trajectories.npz';a=load(path)
        all_archives[path.relative_to(ROOT).as_posix()]=sha(path)
        times=a['times_seconds']*omega0
        assert np.max(np.abs(times-long['times_tau'][:len(times)]))<2e-12
        for group,pred in zip(a['groups'],a['predicted_states']):
            independently=primary(pred,a['continuous_reference'],times,a['tip_map_m'])
            rows=[r for r in saved_rows if r['seed']==seed and r['divisor']==div and r['group']==group]
            for key,values in independently.items():
                gap=float(np.max(np.abs(values-np.array([r[key] for r in rows]))));metric_gap=max(metric_gap,gap)
                assert gap<2e-10,(seed,div,group,key,gap)
            records_checked+=8
            if group in ('GR','GC'):
                continuous=long['predicted_states'][list(long['groups']).index(group),:len(times)]
                truth=long['continuous_reference'][:len(times)]
                integration=primary(pred,continuous,times,a['tip_map_m'])
                model=primary(continuous,truth,times,a['tip_map_m'])
                resolution.append({'seed':seed,'group':str(group),'divisor':div,
                    **{key:{'integration_seed_mean':float(values.mean()),'continuous_model_seed_mean':float(model[key].mean()),
                            'integration_exceeds_model_trajectories':int(np.sum(values>model[key]))} for key,values in integration.items()}})

summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
primary_keys=['state_relative_l2','frequency_abs_error_1','frequency_abs_error_2']
decisions=[]
for candidate,baseline in [('GR','H0'),('GC','H0'),('GR','H4'),('GC','H4'),('GC','GR')]:
    selected=[r for r in summary['contrasts'] if r['phase']=='continuous64s' and r['candidate']==candidate and r['baseline']==baseline and r['metric'] in primary_keys]
    assert len(selected)==3
    decisions.append({'candidate':candidate,'baseline':baseline,'all_primary_rule_passed':all(r['rule_passed'] for r in selected),
                      'wins':{r['metric']:r['wins'] for r in selected},'paired_relative_improvements':{r['metric']:r['mean_paired_relative_improvement'] for r in selected}})

shifts=[]
for key in primary_keys:
    for seed in SEEDS:
        def contrast(div):
            a=load(OLD/f'seed_{seed}/h{div}_trajectories.npz');t=a['times_seconds']*omega0
            values={str(g):primary(p,a['continuous_reference'],t,a['tip_map_m'])[key].mean() for g,p in zip(a['groups'],a['predicted_states']) if g in ('GR','GC')}
            return values['GR']-values['GC']
        first,last=contrast(32),contrast(64)
        shifts.append({'seed':seed,'metric':key,'contrast_h32_GR_minus_GC':float(first),'contrast_h64_GR_minus_GC':float(last),
                       'same_direction':bool(first*last>0),'contrast_shift_smaller_than_h64_gap':bool(abs(last-first)<abs(last))})

audit={'status':'passed','trajectory_metric_records_independently_checked':records_checked,'independent_frequency':'Unwrapped atan2 phase differences; implementation does not import experiment_a',
       'maximum_primary_metric_gap':metric_gap,'exact_H0_frequency_max_gap':frequency_exact_H0_gap,'original_input_hashes_unchanged':True,
       'input_files_checked':len(protocol['input_sha256']),'new_and_reused_archive_hashes':all_archives,
       'refinement_scope':'Prespecified first trajectory per seed, not an all-trajectory error certificate',
       'max_tight_state_gap':max(r['state_max_gap'] for r in refinement),'max_tight_frequency_gap':max(max(r['frequency_abs_gap']) for r in refinement),
       'numerical_resolution':resolution,'h32_to_h64_contrast_shifts':shifts,'decisions':decisions}
save(OUT/'independent_verification.json',audit)
save(OUT/'resolution_analysis.json',{'rows':resolution,'refinement':refinement,'contrast_shifts':shifts})

def table(rows):
    lines=['| 模型 | 完整状态相对L2 | 模态1频差 | 模态2频差 |','|---|---:|---:|---:|']
    for r in rows:
        lines.append('|'+str(r['group'])+'|'+'|'.join(f"{r[k]['mean']:.8g} ± {r[k]['sample_sd']:.3g}" for k in primary_keys)+'|')
    return '\n'.join(lines)

lines=['# 实验A 固定物理对象与网络权重的状态及频率核对','',
       '本实验只新增探索记录，未修改论文、原始结果、物理参数或已选网络权重，也未上传GitHub。五个原种子及全部40条测试初值均保留。','',
       '## 主要判断','',
       '在64秒连续Hamiltonian流对照中，GR与GC相对纯二次H0的完整状态及两个模态频率均有5/5种子改善；两网络却在这三个主指标上全部5/5劣于无训练的四次解析近似H4。GC相对GR的完整状态、模态1频率、模态2频率方向获胜数分别为4/5、5/5、2/5，没有通过三个主指标同时稳定改善的预设探索判据。','',
       '因此，原步长确实曾掩盖网络相对H0的部分收益，但当前物理对象下尚无相对简单四次解析模型的实际预测优势，亦不能由均值较小宣称GC在两模态长期频率上稳定优于GR。','',
       '## 两类时间实验','',
       '- 16秒h0/32、h0/64：直接复用已保存中点轨迹，各914个共同输出时刻，终点15.992993秒；不运行旧驱动，也不重写旧摘要。',
       '- 长时连续流：将原输出间隔延伸至不晚于64秒的共同网格，实际终点63.989489818秒；分别积分完整余弦真值、四次H4、GR、GC，H0采用精确解析旋转。该项用于检查中点离散偏差趋零时的模型表现，不冒充64秒h0/64推进。',
       '- 频率取原正则坐标X-iY的平滑加权相位增量，逐轨迹/逐模态计算后平均；报告全窗、半窗、周期数和半径，不以FFT频点精度代替微小频差。下表频差单位为无量纲角频率rad/tau；乘omega0/(2pi)可换算Hz。','',
       '## 64秒连续流结果','',table([r for r in summary['aggregates'] if r['phase']=='continuous64s']), '',
       '## 16秒中点步长结果','']
for div in (32,64):
    lines+=['### h0/'+str(div),'',table([r for r in summary['aggregates'] if r['phase']=='saved16s' and r['divisor']==div]),'']
lines+=['## 配对判断与数值分辨','',
        '| 候选/基线 | 全状态胜数 | 模态1频差胜数 | 模态2频差胜数 | 三主指标规则同时通过 |',
        '|---|---:|---:|---:|---|']
for d in decisions:
    lines.append('|'+d['candidate']+'/'+d['baseline']+'|'+'|'.join(str(d['wins'][k])+'/5' for k in primary_keys)+'|'+('是' if d['all_primary_rule_passed'] else '否')+'|')
lines+=['',
        '规则是在计算前固定的描述性探索标准：每项至少4/5种子方向改善，平均种子内相对改善至少5%，并要求三个主指标同时成立。它不是显著性检验或普遍优越性证明。','',
        'h32到h64的完整状态对比曾改变部分种子的方向；逐模态频差也须比较离散移动与模型差。resolution_analysis.json给出每种子在同一16秒前缀上，中点轨迹相对对应连续网络流的积分误差，以及该连续网络相对真值的模型偏差。完整状态和频率都未仅凭“步长更小”自动判定充分收敛。','',
        f"预先固定的每种子首条轨迹加密，最大状态变化{audit['max_tight_state_gap']:.4g}，最大频率变化{audit['max_tight_frequency_gap']:.4g} rad/tau；该检查不是所有轨迹的严格误差界。独立atan2解卷绕与主实现复数相位增量计算的主指标最大差{metric_gap:.4g}。",'',
        'h0/64相对同一网络连续流的16秒积分误差：完整状态均值约3.114e-5，低于GR/GC对应模型偏差均值6.323e-4/4.605e-4；模态1频率离散偏差约2.034e-9，远小于模型偏差。模态2频率离散偏差仍约4.737e-7，GR的2/40和GC的5/40轨迹上它大于相应模型频率偏差，不能声称全部轨迹的该模态已充分消除积分影响。64秒连续流对照因而是本轮判断模型能力的重要补充。','',
        '## 定义与文件','',
        'H4=H0-(d^T X)^4/24，保留完整四次Taylor势项，不使用角平均正规形替代动力学。网络均为原10k已选GR/GC；未更换训练预算、初值、物理参数或评价输出。','',
        '`protocol.json`保存预设及旧输入哈希；`saved16s_trajectory_metrics.json`、`seed_*/continuous64s.npz/json`保存全部轨迹与频率诊断；`summary.json`、`trajectory_metrics.csv`、`seed_metrics.csv`、`paired_comparisons.csv`保存汇总；`independent_verification.json`和`resolution_analysis.json`保存独立审计与数值预算。','',
        '复算：在原科学环境中运行 `python -B exploration/ab_state_frequency_20261101/experiment_a.py --stage run`，随后运行本目录 `audit_and_report.py`。所有输出只写本新探索目录。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:audit[k] for k in ('status','trajectory_metric_records_independently_checked','maximum_primary_metric_gap','max_tight_state_gap','max_tight_frequency_gap')},ensure_ascii=False))
