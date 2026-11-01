from pathlib import Path
import csv,json
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def csvwrite(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
force=read(HERE/'common_test_force.json')
matched=read(HERE/'matched_initial_summary.json')
names={'none':'无补偿','oracle':'精确残差','GR':'GR','GC':'GC'}
records=[]
for a in force['aggregates']:
    row={'policy':a['policy']}
    for k,v in a['metrics'].items():row[k+'_mean']=v['mean'];row[k+'_sample_sd']=v['sample_sd']
    records.append(row)
csvwrite(HERE/'common_test_force.csv',records)
records=[]
for a in matched['aggregates']:
    row={k:a[k] for k in ['case','step_divisor','method','policy','seed_count']}
    for k,v in a['metrics'].items():row[k+'_mean']=v['mean'];row[k+'_sample_sd']=v['sample_sd']
    records.append(row)
csvwrite(HERE/'matched_initial_comparison.csv',records)
records=[]
for a in matched['paired_aggregates']:
    row={k:a[k] for k in ['case','step_divisor','method','contrast','metric','seed_count','positive_seed_blocks']}
    for k,v in a['metrics'].items():row[k+'_mean']=v['mean'];row[k+'_sample_sd']=v['sample_sd']
    records.append(row)
csvwrite(HERE/'matched_initial_paired.csv',records)

def pm(v,scale=1,digits=3):return f"${v['mean']*scale:.{digits}f}\\pm{v['sample_sd']*scale:.{digits}f}$"
force_lines=[r'\begin{table}[t]',r'\centering\small\setlength{\tabcolsep}{4pt}',
r'\caption{共同独立测试状态上的残差投影。40条原测试轨迹各取64点；先计算逐轨迹分量RMS，再在种子内和种子间汇总。数值乘$10^6$，无量纲。}',
r'\label{tab:control_test_force}',r'\begin{tabular}{lrrr}',r'\toprule',r'补偿 & 补偿后力残差 & 运动学分量 & 正交力分量 \\',r'\midrule']
for policy in names:
    a=next(a for a in force['aggregates'] if a['policy']==policy)['metrics']
    vals=[pm(a[k],1e6) for k in ['remaining_force_rms','unrealizable_kinematic_rms','unrealizable_orthogonal_force_rms']]
    force_lines.append(names[policy]+' & '+' & '.join(vals)+r' \\')
force_lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
(HERE/'manuscript_force_table.tex').write_text('\n'.join(force_lines),encoding='utf-8')
lines=[r'\begin{table}[t]',r'\centering\small',
r'\caption{初值匹配名义参考下的补偿有效性诊断：步内中点I及等价生成函数G。杆端RMSE单位为$\mu\mathrm m$，报告五个初值块的均值$\pm$样本标准差。}',
r'\label{tab:matched_initial_effectiveness}',r'\begin{tabular}{lrr}',r'\toprule',r'补偿 & $h_0$ & $h_0/2$ \\',r'\midrule']
for policy in names:
    vals=[next(a for a in matched['aggregates'] if a['policy']==policy and a['method']=='I' and a['step_divisor']==div)['metrics']['tracking_tip_rmse_mm'] for div in [1,2]]
    lines.append(names[policy]+' & '+' & '.join(pm(v,1000) for v in vals)+r' \\')
lines.extend([r'\bottomrule',r'\end{tabular}',r'\par\smallskip\begin{minipage}{\columnwidth}\footnotesize 各条名义参考与真实对象从相同测试初值出发，输入、反馈增益、预算及权重保持不变。该诊断隔离初始偏差，原公共零初态参考结果仍保留。两档步长中GR、GC相对无补偿均在5/5块改善；$h_0/2$下平均种子内相对改善为$9.85\%$、$10.52\%$。\end{minipage}',r'\end{table}',''])
(HERE/'manuscript_matched_table.tex').write_text('\n'.join(lines),encoding='utf-8')
print('Created independent-test and matched-reference tables plus complete CSVs')
