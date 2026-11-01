"""Independent A/B seed-level statistics and numeric figures; no paper edits."""
from pathlib import Path
from itertools import product
from math import comb
import argparse
import csv
import hashlib
import json
import os

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

OUT=Path(__file__).resolve().parent
HERE=OUT.parent
ROOT=HERE.parents[1]
A=HERE/'experiment_a'
B=HERE/'experiment_b'
SEEDS=[17,43,91,137,211]
GROUPS=['H0','H4','GR','GC']
FIELDS=[.03,.06,.12]
CONTRASTS=[('GR','H0'),('GC','H0'),('GR','H4'),('GC','H4'),('GC','GR')]
METRICS=['state_relative_l2','frequency_error_si_1','frequency_error_si_2']
COLORS={'H0':'#555555','H4':'#77933C','GR':'#2471A3','GC':'#D97721'}
RESAMPLES=np.array(list(product(range(5),repeat=5)))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def read_csv(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}

def protocol():
    value={'scope':'A/B analysis bundle only. No manuscript/GitHub edits; no experiment or checkpoint selection.',
           'primary_families':['A continuous63.989489818s','B30mT continuous64s','B60mT continuous64s','B120mT continuous64s'],
           'repetition_unit':'Five paired seeds; average8trajectory metrics withinseed first. Do not pool A/B30mT repeated baselines or timepoints.',
           'contrasts':[list(x) for x in CONTRASTS],'primary_metrics':METRICS,'family_size':15,
           'inference':'Exact paired two-sided sign test, zeros excluded from sign count; no symmetry or normality assumption. Holm within each15test family at alpha.05.',
           'minimum_raw_p_n5':.0625,'bootstrap':'Enumerate5^5=3125 ordered seed resamples with replacement, paired withincontrast; percentile2.5/97.5 of means and mean paired relative improvements. Empiricalseed intervals, n5 highly unstable; not physical uncertainty or numerical certificate.',
           'effect_sizes':'Mean/median paired absolute reduction and perseed relative improvement, directional wins, sign balance.',
           'plot_uncertainty':'95% enumerated seed-bootstrap intervals; all5 seed points shown. Tables also include sampleSD(ddof1).',
           'units':'Full state relativeL2 within each declared canonical chart; frequency absolute error multiplied by the actual omega0 to physical rad/s. Across-field improvements are computed withinfield pairs.',
           'A_h32_h64':'Descriptive sensitivity only; no extra p-values or pooled hypotheses. A/B horizons kept explicitly separate.',
           'B_evidence_gate':'Require all15 level-seed evaluations completed and baseline groups/8trajectories/64s windows present; never generate B conclusions from incomplete results.',
           'figures':['01_models_log_errors','02_B_paired_relative_improvement','03_A_step_sensitivity']}
    path=OUT/'analysis_protocol.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==value
    else:save(path,value)
    return value

def bounds(values):
    values=np.asarray(values,float)
    samples=values[RESAMPLES].mean(axis=1)
    return [float(x) for x in np.percentile(samples,[2.5,97.5])]

def sign_p(differences):
    differences=np.asarray(differences,float)
    nonzero=differences[differences!=0]
    n=len(nonzero);positive=int(np.sum(nonzero>0));negative=n-positive
    if n==0:return 1.,0,0,0
    tail=sum(comb(n,j) for j in range(min(positive,negative)+1))/2**n
    return min(1.,2*tail),positive,negative,n

def holm(pvalues):
    pvalues=np.asarray(pvalues,float);order=np.argsort(pvalues,kind='stable');result=np.empty(len(pvalues));previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,(len(pvalues)-rank)*pvalues[index]);result[index]=min(previous,1.)
    return result.tolist()

def read_all():
    sources={};rows=[];sets=[];signal_rows=[]
    spec_path=ROOT/'results/learning/inputs/reference_model_spec.json';spec=json.loads(spec_path.read_text(encoding='utf-8-sig'));omega0=spec['omega0_rad_s']
    sources[str(spec_path.relative_to(ROOT))]=sha(spec_path)
    path=A/'trajectory_metrics.csv';sources[str(path.relative_to(ROOT))]=sha(path)
    for record in read_csv(path):
        if record['phase']!='continuous64s' or record['group'] not in GROUPS:continue
        row=dict(experiment='A',field_T=.03,dataset='A',seed=int(record['seed']),trajectory=int(record['trajectory']),group=record['group'],
                 duration_seconds=63.98948981826144,state_relative_l2=float(record['state_relative_l2']),omega0=float(omega0))
        for j in (1,2):row[f'frequency_error_si_{j}']=float(record[f'frequency_abs_error_{j}'])*omega0
        rows.append(row)
    sets.append({'dataset':'A','experiment':'A','field_T':.03,'duration_seconds':63.98948981826144,'label':'A: original field, 63.989 s'})
    for seed in SEEDS:
        path=A/f'seed_{seed}/continuous64s.npz';sources[str(path.relative_to(ROOT))]=sha(path)
        for div in (32,64):
            path=ROOT/f'exploration/complex_advantage_20261101/fine_step/seed_{seed}/h{div}_trajectories.npz'
            sources[str(path.relative_to(ROOT))]=sha(path)
    incomplete=[]
    for field in FIELDS:
        name=f'B{round(field*1000):03d}mT';sets.append({'dataset':name,'experiment':'B','field_T':field,'duration_seconds':64.,'label':f'B: {field*1000:.0f} mT, 64 s'})
        for seed in SEEDS:
            directory=B/name/f'seed_{seed}';path=directory/'evaluation.json'
            if not path.exists():incomplete.append(str(path));continue
            record=json.loads(path.read_text(encoding='utf-8'))
            if record['status']!='completed':raise RuntimeError('Failed B case: '+str(path))
            archive=directory/'continuous64s.npz';assert sha(archive)==record['archive_sha256'];sources[str(path.relative_to(ROOT))]=sha(path);sources[str(archive.relative_to(ROOT))]=sha(archive)
            data_path=directory/'data.npz';data=load(data_path);omega=float(data['omega0']);sources[str(data_path.relative_to(ROOT))]=sha(data_path)
            path=directory/'data.json';metadata=json.loads(path.read_text(encoding='utf-8'));sources[str(path.relative_to(ROOT))]=sha(path)
            signal_rows.append({'dataset':name,'field_T':field,'seed':seed,**metadata['signals']})
            for r in record['metrics']:
                if r['duration_seconds']!=64 or r['group'] not in GROUPS:continue
                row=dict(experiment='B',field_T=field,dataset=name,seed=seed,trajectory=int(r['trajectory']),group=r['group'],duration_seconds=64.,state_relative_l2=float(r['state_relative_l2']),omega0=omega)
                for j in (1,2):row[f'frequency_error_si_{j}']=float(r[f'frequency_abs_error_{j}'])*omega
                rows.append(row)
    if incomplete:raise RuntimeError('B incomplete; no figures/claims generated: '+str(incomplete))
    for ds in sets:
        for seed in SEEDS:
            for group in GROUPS:
                block=[r for r in rows if r['dataset']==ds['dataset'] and r['seed']==seed and r['group']==group]
                assert len(block)==8 and sorted(r['trajectory'] for r in block)==list(range(8)),(ds,seed,group)
                assert all(np.isfinite(r[m]) and r[m]>0 for r in block for m in METRICS)
    return rows,sets,sources,signal_rows

def aggregate(rows,sets):
    seeds=[];descriptive=[];comparisons=[]
    for ds in sets:
        name=ds['dataset']
        for seed in SEEDS:
            for group in GROUPS:
                block=[r for r in rows if r['dataset']==name and r['seed']==seed and r['group']==group]
                seeds.append({**ds,'seed':seed,'group':group,**{metric:float(np.mean([r[metric] for r in block])) for metric in METRICS}})
        for group in GROUPS:
            block=[r for r in seeds if r['dataset']==name and r['group']==group]
            for metric in METRICS:
                values=np.array([r[metric] for r in block]);lo,hi=bounds(values)
                descriptive.append({**ds,'group':group,'metric':metric,'n_seeds':5,'mean':float(values.mean()),'sample_sd':float(values.std(ddof=1)),'seed_bootstrap_lo95':lo,'seed_bootstrap_hi95':hi})
        family=[]
        for candidate,baseline in CONTRASTS:
            for metric in METRICS:
                cc=np.array([next(r[metric] for r in seeds if r['dataset']==name and r['group']==candidate and r['seed']==seed) for seed in SEEDS])
                bb=np.array([next(r[metric] for r in seeds if r['dataset']==name and r['group']==baseline and r['seed']==seed) for seed in SEEDS])
                differences=bb-cc;relative=1-cc/bb;p,wins,losses,n=sign_p(differences);dlo,dhi=bounds(differences);rlo,rhi=bounds(relative)
                family.append({**ds,'candidate':candidate,'baseline':baseline,'metric':metric,'n_seeds':5,'nonzero_signs':n,'wins':wins,'losses':losses,
                    'mean_paired_absolute_reduction':float(differences.mean()),'sample_sd_paired_absolute_reduction':float(differences.std(ddof=1)),
                    'median_paired_absolute_reduction':float(np.median(differences)),'absolute_reduction_bootstrap_lo95':dlo,'absolute_reduction_bootstrap_hi95':dhi,
                    'mean_paired_relative_improvement':float(relative.mean()),'sample_sd_paired_relative_improvement':float(relative.std(ddof=1)),
                    'median_paired_relative_improvement':float(np.median(relative)),'relative_improvement_bootstrap_lo95':rlo,'relative_improvement_bootstrap_hi95':rhi,
                    'direction_balance':float((wins-losses)/n) if n else 0.,'sign_test_two_sided_p':p,'exploratory_direction_rule':bool(wins>=4 and relative.mean()>=.05),
                    'seed_candidate_values':cc.tolist(),'seed_baseline_values':bb.tolist(),'seed_relative_improvements':relative.tolist()})
        adjusted=holm([r['sign_test_two_sided_p'] for r in family]);assert len(family)==15
        for row,p in zip(family,adjusted):row['holm_adjusted_p']=p;row['holm_reject_05']=bool(p<.05)
        comparisons+=family
    return seeds,descriptive,comparisons

def phase_freq(states,times):
    angle=np.unwrap(np.arctan2(-states[...,2:],states[...,:2]),axis=0);n=len(times)-1;s=(np.arange(n)+.5)/n;w=np.exp(-1/(s*(1-s)));w/=w.sum()
    return np.einsum('t,tnj->nj',w,np.diff(angle,axis=0))/float(np.mean(np.diff(times)))

def a_step_metrics():
    rows=[];old=ROOT/'exploration/complex_advantage_20261101/fine_step'
    for seed in SEEDS:
        long=load(A/f'seed_{seed}/continuous64s.npz');omega=long['times_tau'][-1]/long['times_seconds'][-1]
        for div in (32,64):
            archive=load(old/f'seed_{seed}/h{div}_trajectories.npz');times=archive['times_seconds']*omega;ref=archive['continuous_reference']
            selected={str(g):p for g,p in zip(archive['groups'],archive['predicted_states']) if g in GROUPS}
            for group,pred in selected.items():
                difference=pred-ref;state=np.linalg.norm(difference,axis=(0,2))/np.linalg.norm(ref,axis=(0,2));freq=np.abs(phase_freq(pred,times)-phase_freq(ref,times))*omega
                rows.append({'seed':seed,'group':group,'step':f'h0/{div}','state_relative_l2':float(state.mean()),'frequency_error_si_1':float(freq[:,0].mean()),'frequency_error_si_2':float(freq[:,1].mean())})
        count=len(archive['times_seconds']);times=long['times_tau'][:count];ref=long['continuous_reference'][:count]
        for group,pred in zip(long['groups'],long['predicted_states']):
            if group not in GROUPS:continue
            pred=pred[:count];difference=pred-ref;state=np.linalg.norm(difference,axis=(0,2))/np.linalg.norm(ref,axis=(0,2));freq=np.abs(phase_freq(pred,times)-phase_freq(ref,times))*omega
            rows.append({'seed':seed,'group':str(group),'step':'ODE16s','state_relative_l2':float(state.mean()),'frequency_error_si_1':float(freq[:,0].mean()),'frequency_error_si_2':float(freq[:,1].mean())})
    return rows

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':9,'axes.titlesize':10,'figure.dpi':140,'savefig.dpi':220,'pdf.fonttype':42,'svg.fonttype':'none'})

LABELS=['Complete state relative L2','Mode 1 frequency error (rad/s)','Mode 2 frequency error (rad/s)']
def export(fig,name):
    for ext in ('png','pdf','svg'):fig.savefig(OUT/'figures'/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)

def figure_models(seeds,descriptive,sets):
    fig,axes=plt.subplots(4,3,figsize=(11.7,11.),constrained_layout=True)
    for i,ds in enumerate(sets):
        for j,metric in enumerate(METRICS):
            ax=axes[i,j]
            for index,group in enumerate(GROUPS):
                vals=np.array([r[metric] for r in seeds if r['dataset']==ds['dataset'] and r['group']==group]);row=next(r for r in descriptive if r['dataset']==ds['dataset'] and r['group']==group and r['metric']==metric)
                ax.scatter(index+np.linspace(-.12,.12,5),vals,color=COLORS[group],s=19,alpha=.8,zorder=3)
                mean=row['mean'];ax.errorbar(index,mean,yerr=[[mean-row['seed_bootstrap_lo95']],[row['seed_bootstrap_hi95']-mean]],fmt='D',color='black',markersize=3.5,capsize=3,zorder=4)
            ax.set_yscale('log');ax.set_xticks(range(4),GROUPS);ax.grid(axis='y',alpha=.2);ax.set_title(ds['label']);ax.set_ylabel(LABELS[j])
    fig.suptitle('Model error on fixed held-out trajectories\nDots: five seed means; diamonds / whiskers: mean and empirical 95% seed-bootstrap interval',fontsize=12)
    export(fig,'figure-01-model-errors')

def figure_strength(comparisons):
    fig,axes=plt.subplots(3,3,figsize=(11.7,8.8),sharex=True,constrained_layout=True)
    for row_index,baseline in enumerate(['H0','H4','GR']):
        candidates=['GR','GC'] if baseline!='GR' else ['GC']
        for j,metric in enumerate(METRICS):
            ax=axes[row_index,j]
            for candidate in candidates:
                block=sorted([r for r in comparisons if r['experiment']=='B' and r['baseline']==baseline and r['candidate']==candidate and r['metric']==metric],key=lambda r:r['field_T'])
                xx=np.array([r['field_T']*1000 for r in block]);mean=np.array([r['mean_paired_relative_improvement']*100 for r in block]);lo=np.array([r['relative_improvement_bootstrap_lo95']*100 for r in block]);hi=np.array([r['relative_improvement_bootstrap_hi95']*100 for r in block])
                ax.plot(xx,mean,'o-',label=candidate,color=COLORS[candidate],markersize=4);ax.fill_between(xx,lo,hi,color=COLORS[candidate],alpha=.14)
                for x,r in zip(xx,block):ax.scatter(x+np.linspace(-1,1,5),np.array(r['seed_relative_improvements'])*100,s=9,alpha=.45,color=COLORS[candidate])
            ax.axhline(0,color='black',linewidth=.8);ax.set_yscale('symlog',linthresh=5,linscale=.8)
            # Keep zero readable: automatic symlog ticks crowd +/-1 against zero.
            ticks=[-10.**k for k in range(8,0,-1)]+[0.,10.,100.]
            limits=ax.get_ylim();ax.yaxis.set_major_locator(FixedLocator([t for t in ticks if limits[0]<=t<=limits[1]]));ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value,position:'0' if value==0 else ('$'+('-' if value<0 else '')+r'10^{'+str(round(np.log10(abs(value))))+'}$')))
            ax.set_title(['State error','Mode 1 frequency error','Mode 2 frequency error'][j]);ax.set_ylabel(f'vs {baseline}: paired improvement (%)');ax.grid(axis='y',alpha=.2);ax.set_xticks([30,60,120]);ax.legend(frameon=False,fontsize=8)
            if row_index==2:ax.set_xlabel('Bias magnetic field (mT)')
    fig.suptitle('B: Within-field paired improvements on 64 s continuous flows\nPositive is better; symmetric-log y scale; shaded bands are empirical 95% seed-bootstrap intervals',fontsize=12)
    export(fig,'figure-02-strength-paired-improvements')

def figure_steps(rows):
    fig,axes=plt.subplots(1,3,figsize=(11.7,3.5),constrained_layout=True)
    steps=['h0/32','h0/64','ODE16s']
    for j,metric in enumerate(METRICS):
        ax=axes[j]
        for group in GROUPS:
            x=[];means=[]
            for i,step in enumerate(steps):
                vals=np.array([r[metric] for r in rows if r['group']==group and r['step']==step])
                if not len(vals):continue
                x.append(i);means.append(vals.mean());ax.scatter(i+np.linspace(-.08,.08,5),vals,s=11,color=COLORS[group],alpha=.45)
            ax.plot(x,means,'o-',color=COLORS[group],label=group,markersize=3.5)
        ax.set_yscale('log');ax.set_xticks(range(3),['h0/32','h0/64','ODE flow']);ax.set_ylabel(LABELS[j]);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=8)
    fig.suptitle('A: Same 15.993 s output prefix, unchanged weights\nH4: continuous-flow comparison',fontsize=11)
    export(fig,'figure-03-A-step-sensitivity')

def reports(descriptive,comparisons,sets,sources,signal_rows,a_steps):
    table=['| Dataset | Model | State relative L2 | Mode1 freq error (rad/s) | Mode2 freq error (rad/s) |','|---|---|---:|---:|---:|']
    for ds in sets:
        for group in GROUPS:
            rr={r['metric']:r for r in descriptive if r['dataset']==ds['dataset'] and r['group']==group}
            table.append('|'+ds['label']+'|'+group+'|'+'|'.join(f"{rr[m]['mean']:.7g} ± {rr[m]['sample_sd']:.3g}" for m in METRICS)+'|')
    numeric='\n'.join(table)
    signals=['| B field | nonlinear/linear force L2 | H4 remainder/linear force L2 | nonlinear/label-noise force L2 |',
             '|---|---:|---:|---:|']
    for field in FIELDS:
        block=[r for r in signal_rows if r['field_T']==field]
        entries=[]
        for key in ('nonlinear_to_linear_force_l2','H4_remainder_to_linear_force_l2','nonlinear_to_noise_force_l2'):
            values=np.array([r[key] for r in block]);entries.append(f'{values.mean():.6g} ± {values.std(ddof=1):.3g}')
        signals.append(f'|{field*1000:.0f} mT|'+'|'.join(entries)+'|')
    signal_table='\n'.join(signals)
    stats=['# Statistical appendix','',
           '独立重复单位为5个配对种子。每条轨迹先沿时间计算原始误差，种子内平均8条测试轨迹；没有把轨迹时间点、模态或同一基线的重复使用当成新独立样本。A长时为63.989489818秒，B为64秒；A/B30mT共享原权重与初值，不合并为10个独立种子。','',
           '每个实验/场强的主推断family包含5个预定候选/基线对比×3个主指标=15项；A h32/h64只是描述性数值敏感性检查，不追加挑选式p值。频差已由每条记录实际omega0转为SI rad/s，不将不同场强的raw rad/tau混作同单位。完整状态为各场强声明正则尺度中的相对L2；跨场趋势以场内配对量解释。','',
           '使用双侧精确配对符号检验，零差剔除后计算非零符号数。零假设为配对差正负概率各1/2；不要求差值正态或对称。n=5极小，正态检验缺乏判断力，本分析预先选非参数检验而非先看显著性再选择方法。Holm调整在各15项family内部执行，可容许对比间相关。即使5/5同方向，未调整最小双侧p=.0625，不能在.05水平宣布显著；4/5或5/5只是方向证据。','',
           '列出均值、样本SD、配对绝对差、逐种子相对改善、方向胜数与符号平衡作为效应量。95%区间枚举5^5=3125次有放回种子重抽样，候选和基线保持同索引配对，取均值分布2.5%及97.5%分位。它只是这5个随机种子的经验bootstrap区间，n=5高度不稳定，不是模型物理不确定度、全域泛化界、严格频率误差界，也未对区间作多重比较校正。','',numeric,'',
           '## Signal and noise limits','',signal_table,'',
           '上述比率来自每个场强、每个种子的全部训练样本上力向量的L2范数之比，并非逐时刻比值的平均。非线性信号增强不自动保证相对标签噪声更可辨；还需同时看nonlinear/noise。各场强采用实际omega0转换时间与频率，数据中的噪声亦提供SI杆端速度及模态加速度映射。三场强的状态图使用各自的正则归一化，不能把同一个无量纲标签噪声幅值解释为相同SI强度。B物理严格有效性尚未认证；数值表现的结论限定于指定近似两模态模型，不外推到实际梁。','',
           'H4使用已给定nu、d和已知cos势的Taylor精确系数，无需拟合；GR/GC仅共享H0先验，非线性残差由数据学习。因此H4对照回答的是本已知物理对象下相对简单解析模型的实际收益，并非同等结构知识下的表示能力定理比较。保持该基线和预定义规则，不把知识前提差异作为事后移除不利基线的理由。','',
           '完整60项推断见 `comparisons.json/csv`，逐种子值见 `seed_metrics.csv`，原始逐轨迹量及SI转换见 `trajectory_metrics.csv`，信号/噪声原始值见 `signal_noise_seed.csv`。未删除不利种子、未按结果调模型、未修改主文。']
    (OUT/'stats-appendix.md').write_text('\n'.join(stats)+'\n',encoding='utf-8')
    successful=[]
    for ds in sets:
        rr=[r for r in comparisons if r['dataset']==ds['dataset'] and r['candidate']=='GC' and r['baseline']=='GR']
        if all(r['exploratory_direction_rule'] for r in rr):successful.append(ds['label'])
    success_text='、'.join(successful) if successful else '没有条件'
    lines=['# A/B strict analysis bundle','',
           '本包评价细步长/连续流以及磁场分级下，神经残差是否同时改善完整状态和两模态频率。正文、原实验、权重与GitHub不在本包修改范围。B的物理解释仍受近似斜率/几何截断与高模态条件限制，不能把数值包络通过解释成物理模型已经校准。','',numeric,'',
           f'GC相对GR的三指标联合方向探索规则由以下条件达到：{success_text}。这个正向发现应保留；其他条件未达到并不抹去它。同一分析仍保留H0/H4全部对照，是否相对H4获益见逐项结果。','',
           '## Statistical interpretation','',
           f"60个预定义主检验中，Holm校正后p<.05的项数为{sum(r['holm_reject_05'] for r in comparisons)}。n=5的双侧符号检验最小原始p=.0625，方向全胜也不等于统计显著。探索规则为至少4/5种子同方向、逐种子相对改善均值至少5%，须三主指标同时满足；它并非显著性或物理有效性证明。完整bootstrap区间与效应量见统计附录。",'',
           'H4使用已知cos势Taylor精确系数及给定nu、d，不需拟合；GR/GC只共享H0先验并由数据学习残差。H4比较检验已知物理对象上神经模型的实用收益，不是同等知识前提的表达能力定理。','',
           '## Training force signal and noise','',signal_table,'',
           '这是整个训练集的力向量L2范数之比。场强、时间尺度、正则坐标与SI噪声映射同时变化；数值趋势须与信号相对标签噪声的大小一起解释，不能归因于单个网络表示。B严格物理有效性未认证。','',
           '## Claim candidates','']
    for ds in sets:
        for candidate,baseline in [('GR','H4'),('GC','H4'),('GC','GR')]:
            rr=[r for r in comparisons if r['dataset']==ds['dataset'] and r['candidate']==candidate and r['baseline']==baseline]
            wins={r['metric']:r['wins'] for r in rr};all_rule=all(r['exploratory_direction_rule'] for r in rr)
            lines += [f"- Claim: {ds['label']}，{candidate}相对{baseline}的方向证据。",f"  - Source evidence: comparisons.csv；三指标胜数{wins}。",f"  - Allowed wording: 在这5个种子的指定时域/模型内，{'达到' if all_rule else '未达到'}三主指标共同方向改善的探索判据。",'  - Forbidden stronger wording: 显著优越、普遍优越、实物系统优势或严格连续极限认证。','  - Uncertainty: seed bootstrap不稳定；数值与物理适用性需另看原实验审计。',"  - Next check: 保持基线与误差单位一致，依据独立物理有效性证据再决定论文主张。",'  - Decision: '+('keep' if all_rule else 'weaken')]
    lines+=['','图的读法与解释检查清单见 `figure-catalog.md`。本文件是分析证据包，不是论文Results段落或总实验总结。']
    (OUT/'analysis-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    directions=[]
    for ds in sets:
        for candidate in ('GR','GC'):
            rr=[next(r for r in comparisons if r['dataset']==ds['dataset'] and r['candidate']==candidate and r['baseline']=='H4' and r['metric']==m) for m in METRICS]
            directions.append(f"{ds['label']} 的 {candidate}/H4 三指标胜数为 "+'/'.join(str(r['wins']) for r in rr)+'（各5种子）')
    strengths=[]
    for field in FIELDS:
        rr=[next(r for r in comparisons if r['experiment']=='B' and r['field_T']==field and r['candidate']=='GC' and r['baseline']=='GR' and r['metric']==m) for m in METRICS]
        strengths.append(f'{field*1000:.0f} mT 的 GC/GR 三指标逐种子相对改善均值为 '+', '.join(f"{r['mean_paired_relative_improvement']*100:.2f}%" for r in rr))
    step_observations=[]
    for group in ('H0','GR','GC'):
        state_means=[np.mean([r['state_relative_l2'] for r in a_steps if r['group']==group and r['step']==step]) for step in ('h0/32','h0/64','ODE16s')]
        step_observations.append(f"{group} 全状态均值随 h0/32→h0/64→ODE16s 为 "+'→'.join(f'{x:.5g}' for x in state_means))
    catalog=f'''# 图表目录与解释

## Figure 01：完整状态与两模态频差

- 文件：`figures/figure-01-model-errors.png/pdf/svg`。
- 目的：在三主指标上同时比较 H0、H4、GR、GC，避免把杆端投影作为网络优势的唯一依据。
- 来源：A 原始逐轨迹 CSV、全部15个 B `evaluation.json`、`seed_metrics.csv`。
- 图注要求：A 为63.989489818秒，B 为64秒，分行呈现。每个彩点是一个种子的8条测试轨迹均值；黑菱形为5种子算术均值；误差棒为枚举3125个种子重抽样得到的经验95%区间。频差已转SI rad/s；三列纵轴均为对数。
- 实际观察：{'; '.join(directions)}。
- 含义：相对 H0 的误差下降须与相对更简单 H4 的结果同时解读；不同评价量方向一致时才有较强的探索方向证据。H4使用给定nu、d及已知cos势Taylor精确系数，无拟合；GR/GC只共享H0先验并由数据学残差。因此本图比较已知物理对象下的实用收益，不是同等知识前提的表示能力定理。
- 决策影响：主张须限定到明确基线、三项主指标及本次种子/时域，不以选定输出或最好种子宣称通用优势。
- 检查清单：Teacher 的恒零误差不放在对数轴；A/B30mT不合并为10个独立种子；区间并非物理不确定度；B严格物理有效性尚未认证。

## Figure 02：磁场分级与配对相对改善

- 文件：`figures/figure-02-strength-paired-improvements.png/pdf/svg`。
- 目的：检查30、60、120mT下，各自场内配对收益是否同时在完整状态和两模态频差出现。
- 来源：`comparisons.json/csv`，固定三场强，不按结果挑选场强。
- 图注要求：正值表示候选模型误差较小。曲线为逐种子 `1-Ecandidate/Ebaseline` 的均值，浅色点为全部5种子的比值，阴影为经验95%种子bootstrap区间。对称对数轴保留相对极小H4误差产生的巨大负值，并保留零点。场内频差的相对比值不受SI转换影响。
- 实际观察：{'; '.join(strengths)}。
- 解释限制：须同时查看 `signal_noise_seed.csv` 中全训练集力向量 L2 比值 nonlinear/linear、H4 remainder/linear、nonlinear/noise。增强非线性与增强相对噪声可辨性是不同的问题。三场强时间尺度、正则坐标和SI噪声映射均不同，完整状态的跨场比较采用场内相对量。
- 决策影响：只有在两种解析基线、全部三项主指标和数值误差检查下仍有一致收益，才可保留条件模型的表示优势候选主张；严格物理适用性不能由此图认证。
- 检查清单：family固定；未丢弃巨大负比值；区间不是物理置信界；n=5不标显著性星号；4/5方向及5%阈值只是探索规则。

## Figure 03：A 固定模型的步长敏感性

- 文件：`figures/figure-03-A-step-sensitivity.png/pdf/svg`。
- 目的：区分中点离散误差与固定学习Hamiltonian本身的连续流误差。
- 来源：原始h32/h64 NPZ以及同一15.992993212秒输出前缀的连续模型轨迹；`a_step_seed_metrics.csv`。
- 图注要求：物理对象、权重、初值和输出时域完全相同；H4仅有连续流点，不补造其h32/h64中点结果。点为5种子均值，线连接算术平均；纵轴为对数。ODE16s和长时ODE64s不混用。
- 实际观察：{'; '.join(step_observations)}。
- 含义：步长减小后剩余差异须由连续模型对照和两模态频率共同判断，不能仅由单个离散步长的误差排名推断稳定优势。
- 决策影响：若排名随步长或评价量变化，保留条件陈述；若简单H4更准，应同时呈现。
- 检查清单：未合并不同时间窗；未把有限64秒窗口外推到严格无限时不变频率；未由近似数值收敛宣称实物有效性。
'''
    (OUT/'figure-catalog.md').write_text(catalog,encoding='utf-8')

def independent_check(rows,seeds,comparisons):
    # Recalculate means via Python sums and p-values by enumerating all sign strings.
    import statistics
    max_gap=0.
    for seedrow in seeds:
        block=[r for r in rows if r['dataset']==seedrow['dataset'] and r['seed']==seedrow['seed'] and r['group']==seedrow['group']]
        for metric in METRICS:max_gap=max(max_gap,abs(seedrow[metric]-statistics.mean(r[metric] for r in block)))
    for row in comparisons:
        c=np.array(row['seed_candidate_values']);b=np.array(row['seed_baseline_values']);diff=b-c;n=int(np.sum(diff!=0));k=int(np.sum(diff>0));observed=min(k,n-k)
        signs=list(product((0,1),repeat=n));p=sum(min(sum(v),n-sum(v))<=observed for v in signs)/len(signs) if n else 1.
        assert abs(row['sign_test_two_sided_p']-p)<1e-15
        assert not row['holm_reject_05']
        boots=np.array([sum((1-c/b)[list(idx)])/5 for idx in product(range(5),repeat=5)])
        lo,hi=np.percentile(boots,[2.5,97.5]);assert abs(lo-row['relative_improvement_bootstrap_lo95'])<1e-10*max(1,abs(lo));assert abs(hi-row['relative_improvement_bootstrap_hi95'])<1e-10*max(1,abs(hi))
    for dataset in sorted({r['dataset'] for r in comparisons}):
        family=[r for r in comparisons if r['dataset']==dataset];ordered=sorted(family,key=lambda r:r['sign_test_two_sided_p'])
        for rank,row in enumerate(ordered):
            adjusted=min(1.,max((15-j)*ordered[j]['sign_test_two_sided_p'] for j in range(rank+1)))
            assert abs(adjusted-row['holm_adjusted_p'])<1e-15
    raw_gap={m:0. for m in METRICS};checked=0
    for dataset in ['A']+[f'B{round(f*1000):03d}mT' for f in FIELDS]:
        for seed in SEEDS:
            path=(A if dataset=='A' else B/dataset)/f'seed_{seed}/continuous64s.npz'
            archive=load(path);reference=archive['predicted_states'][list(archive['groups']).index('Teacher')]
            omega=float(archive['times_tau'][-1]/archive['times_seconds'][-1]);reference_frequency=phase_freq(reference,archive['times_tau'])
            for group,prediction in zip(archive['groups'],archive['predicted_states']):
                if group not in GROUPS:continue
                delta=prediction-reference
                state=np.sqrt(np.sum(delta*delta,axis=(0,2))/np.sum(reference*reference,axis=(0,2)))
                frequency=np.abs(phase_freq(prediction,archive['times_tau'])-reference_frequency)*omega
                for trajectory in range(8):
                    row=next(r for r in rows if r['dataset']==dataset and r['seed']==seed and r['group']==group and r['trajectory']==trajectory)
                    expected=[state[trajectory],frequency[trajectory,0],frequency[trajectory,1]]
                    for metric,value in zip(METRICS,expected):raw_gap[metric]=max(raw_gap[metric],abs(row[metric]-value))
                    checked+=1
    assert max_gap<1e-12 and max(raw_gap.values())<1e-11
    save(OUT/'independent_statistics_verification.json',{'status':'passed','seed_outcome_values_checked':len(seeds)*3,'comparisons_checked':len(comparisons),'maximum_mean_gap':max_gap,'raw_archive_trajectory_records_checked':checked,'raw_archive_metric_maximum_abs_gap':raw_gap,'sign_test':'Independent enumeration of 2^n outcomes','bootstrap':'Independent Python enumeration of 3125 seed tuples','holm':'Independent ordered closed-form reconstruction per 15-test family','no_holm_rejections':True})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['plan','run'],default='plan');args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);protocol()
    if args.stage=='plan':print('Analysis protocolfixed; noBfigures orclaims beforeall15evaluationscomplete.');return
    rows,sets,sources,signal_rows=read_all();seeds,descriptive,comparisons=aggregate(rows,sets)
    independent_check(rows,seeds,comparisons)
    for rel,digest in sources.items():assert sha(ROOT/rel)==digest
    write_csv(OUT/'trajectory_metrics.csv',rows);write_csv(OUT/'seed_metrics.csv',seeds);write_csv(OUT/'descriptive.csv',descriptive)
    write_csv(OUT/'comparisons.csv',[{k:v for k,v in r.items() if not isinstance(v,list)} for r in comparisons]);save(OUT/'comparisons.json',comparisons)
    write_csv(OUT/'signal_noise_seed.csv',signal_rows)
    a_steps=a_step_metrics();write_csv(OUT/'a_step_seed_metrics.csv',a_steps)
    (OUT/'figures').mkdir(exist_ok=True);style();figure_models(seeds,descriptive,sets);figure_strength(comparisons);figure_steps(a_steps);reports(descriptive,comparisons,sets,sources,signal_rows,a_steps)
    for rel,digest in sources.items():assert sha(ROOT/rel)==digest
    save(OUT/'provenance.json',{'script_sha256':sha(__file__),'protocol_sha256':sha(OUT/'analysis_protocol.json'),'input_sha256':sources,'raw_rows':len(rows),'seed_rows':len(seeds),'inference_families':4,'hypotheses_per_family':15,'units':'frequencySIrad/s','all_original_inputs_unchanged':True})
    print('ANALYSIS_COMPLETE',len(rows),len(seeds),len(comparisons))

if __name__=='__main__':main()
