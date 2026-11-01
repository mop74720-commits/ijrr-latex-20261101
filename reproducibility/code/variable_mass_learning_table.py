"""Render a compact manuscript table from the audited aggregate JSON."""
from pathlib import Path
import hashlib
import json
import numpy as np

OUT=Path(__file__).resolve().parents[1]/'results/variable_mass_learning'

def number(value):
    token=f'{value:.5g}'
    if 'e' not in token: return token
    first,power=token.split('e')
    return first+r'\times10^{'+str(int(power))+'}'

def main():
    s=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    audit=json.loads((OUT/'independent_verification.json').read_text(encoding='utf-8'))
    assert audit['status']=='passed' and s['prediction_count']==40
    body=[]
    for contrast in ('R-GR','C-GC','R-C','GR-GC','(GR-GC)-(R-C)'):
        values=[]
        for metric in ('teacher_tip_rmse_mm_mean','continuous_tip_rmse_mm_mean'):
            row=next(r for r in s['paired_contrasts'] if r['divisor']==4 and r['metric']==metric and r['contrast']==contrast)
            values.append('$'+number(row['mean']*1000)+r'\pm'+number(row['sample_sd']*1000)+'$')
        body.append('$'+contrast+'$ & '+' & '.join(values)+r' \\')
    table='\n'.join([
        r'\begin{table*}[t]',r'\centering\small',
        r'\caption{变质量正则坐标检验的五项种子内配对差（预测步长$h_0/4$，杆端RMSE差，单位$\mu\mathrm m$）。}',
        r'\label{tab:variable_mass_paired}',r'\begin{tabular}{lrr}',r'\toprule',
        r'配对差 & 对同步长教师 & 对连续参考 \\',r'\midrule',*body,r'\bottomrule',r'\end{tabular}',
        r'\par\smallskip\begin{minipage}{0.96\textwidth}\footnotesize 每个种子内先平均8条独立测试轨迹的时间域RMSE，再构造五项配对差；表列5个种子的均值$\pm$样本标准差。正值表示前组误差较大。误差在原物理模态坐标中评价；该实验为同一梁的非线性正则坐标检验。\end{minipage}',
        r'\end{table*}',''])
    (OUT/'manuscript_table.tex').write_text(table,encoding='utf-8')
    values=dict(models=s['models'],paired_contrasts=s['paired_contrasts'],H0=[],units=dict(tip='mm',state='dimensionless relative L2'),
                protocol_sha256=audit['protocol_sha256'],summary_sha256=hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest())
    for divisor in (1,4):
        records=[]
        for seed in (17,43,91,137,211):
            path=OUT/f'seed_{seed}'/f'baselines_h{divisor}.json'
            records.append(json.loads(path.read_text(encoding='utf-8'))['H0'])
        row=dict(divisor=divisor)
        for key in ('continuous_tip_rmse_mm_mean','continuous_state_relative_l2_mean','teacher_tip_rmse_mm_mean'):
            vector=[r[key] for r in records]
            row[key]=dict(mean=float(np.mean(vector)),sample_sd=float(np.std(vector,ddof=1)))
        values['H0'].append(row)
    (OUT/'manuscript_values.json').write_text(json.dumps(values,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    for row in values['models']:
        print(row['divisor'],row['group'],'tip_um',row['continuous_tip_rmse_mm_mean']['mean']*1000,'state',row['continuous_state_relative_l2_mean']['mean'],'teacher_tip_um',row['teacher_tip_rmse_mm_mean']['mean']*1000)
    print('MANUSCRIPT_TABLE_READY',flush=True)

if __name__=='__main__': main()
