"""Build the Chinese B report from independently audited saved results."""
from pathlib import Path
import csv
import json
import numpy as np

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
LEVELS=['B030mT','B060mT','B120mT'];SEEDS=[17,43,91,137,211]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csv_read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def fmt(x):return f'{x:.4g}'
aggregates=csv_read(OUT/'independent_aggregates.csv')
comparisons=csv_read(OUT/'independent_comparisons.csv')
audit=read(OUT/'independent_audit_all.json');assert audit['status']=='passed'
design=read(BASE/'physical_design/final_design.json')

def stat(level,duration,group,metric,factor=1):
    r=next(x for x in aggregates if x['level']==level and int(x['duration_seconds'])==duration and x['group']==group and x['metric']==metric)
    return fmt(float(r['mean'])*factor)+' ± '+fmt(float(r['sample_sd'])*factor)

lines=['# 实验 B：实际偏置磁场参数的条件压力测试','',
'本次仅新增独立实验文件；没有把结果写入论文，也没有提交或发布到 GitHub。三档场强、五个配对种子、训练与评价协议在首次新增训练前已锁定。',
'',
'**适用范围：以下是原先声明的“常偶极、线性小斜率梁、以杆端斜率近似转角”的既定模型内结果。斜率筛查通过，不等于真实材料的磁响应和高阶几何模型已经验证。原近似转角所遗漏的几何项与所保留的磁四次项同阶，因此这个实验不能单独完成“真实梁物理适用性已获证明”的前提，也不能据此否定真实梁上 CHNN 的一般能力。**',
'',
'## 固定设计与公平性','',
'- 场强为 30、60、120 mT；每档重新装配含偏置磁刚度的32单元一致质量有限元模型，再重新计算两模态、线性频率和正则变量缩放。没有直接给原非线性残差乘放大系数。',
'- 从原来的完整有限元位移和速度出发，以新场强的振型作质量正交投影；投影误差保留在数据记录，三档物理初态只是近似配对。',
'- 训练12条轨迹×2 s，验证4条×4 s，每条训练/验证均固定286/571个式（36）中点监督观测，测试8条×64 s，并报告16 s前缀。新增场强用同一式（36）的常质量保守中点特例重新生成数据。',
'- 5个种子为17、43、91、137、211。GR/GC均673个实参数、10000次Adam更新、batch512，按原验证规则选权重；没有挑种子或再搜索参数。30 mT的10份原权重经参数、数据、checkpoint核验后原样复用，60/120 mT新增20次训练。',
'- 各档复用相同的无量纲均匀噪声数组（幅度10⁻⁶）；因为正则缩放不同，这并不等于相同的有量纲测量噪声。换算量已经逐档逐种子保存。',
'- H0是已知二次物理能量；H4=H0−(dᵀX)⁴/24是原余弦势的真实四次Taylor截断，没有可训练参数。神经网络只获得H0先验和同档训练标签，没有获得真实四次/更高项。',
'- H4直接使用给定ν、d与原余弦势的精确Taylor系数，GR/GC则从数据学习残差。因此H4是已知物理参数下的实用预测基线；此比较没有证明相同结构先验下谁的表达能力更强，H4胜出也不表示网络不能表示该残差。',
'- 全部预测用连续DOP853求解，rtol=10⁻¹¹、atol=10⁻¹³，共同物理采样间隔0.005 s。H0用精确线性流。连续参考是本档原余弦模型，不是离散教师轨迹。',
'- GR/GC在此都是实值能量的Hamiltonian向量场，GC使用复权重/复隐藏层；本次没有验证非零虚部的耗散复Hamiltonian。',
'',
'## 信号强度、噪声与全梁筛查','',
'训练数据中的非线性/线性力比是整批向量L2范数之比，不对线性力接近零的瞬时比值求平均。下表范围覆盖全部五个种子。',
'',
'| 场强 | 真非线性/线性力 | H4遗漏力/线性力 | 非线性力/噪声 | SI杆端速度标签噪声RMS(m/s) |',
'|---|---:|---:|---:|---:|']
for level in LEVELS:
    meta=[read(OUT/level/f'seed_{s}/data.json') for s in SEEDS]
    def span(key):
        values=[r['signals'][key] for r in meta];return fmt(min(values))+'–'+fmt(max(values))
    lines.append(f'| {int(level[1:4])} mT | {span("nonlinear_to_linear_force_l2")} | {span("H4_remainder_to_linear_force_l2")} | {span("nonlinear_to_noise_force_l2")} | {span("SI_tip_velocity_label_noise_rms_m_s")} |')
lines+=['',
'增加实际偏置磁场确实增强了原模型中的非线性残差，但其相对于线性力仍很弱。不能把“偏置变大”写成“已进入强非线性、神经网络应当占优的区域”。H4本来已经捕捉了绝大部分可辨认残差，是必须保留的强解析竞争模型。',
'',
'在独立的64 s连续参考测试集上，按每种子8条轨迹整体L2计算的比值如下；范围覆盖五个种子。',
'',
'| 场强 | 测试真非线性/线性力 | 测试H4遗漏力/线性力 |',
'|---|---:|---:|']
for level in LEVELS:
    rows=[x for x in audit['independent_flow_checks'] if x['level']==level]
    def span_test(key):return fmt(min(x[key] for x in rows))+'–'+fmt(max(x[key] for x in rows))
    lines.append(f'| {int(level[1:4])} mT | {span_test("continuous64s_nonlinear_to_linear_force_l2")} | {span_test("continuous64s_H4_remainder_to_linear_force_l2")} |')
lines += [
'',
'| 场强 | 位移投影损失上限 | 速度投影损失上限 | 全梁连续保守流斜率能量包络 |',
'|---|---:|---:|---:|',
'| 30 mT | 6.56×10⁻¹⁶ | 6.36×10⁻¹⁶ | 0.114768 |',
'| 60 mT | 0.44584% | 0.44689% | 0.112302 |',
'| 120 mT | 1.29403% | 1.29702% | 0.108659 |',
'',
'门槛0.15仅为透明的小斜率计算筛查。全梁空间极值检查使用每单元Hermite多项式驻点，不把杆端斜率当成全梁斜率。连续保守流能量包络通过，但计算未使用外向舍入区间算术；神经网络预测另做保存时刻的全梁空间检查。详见相邻physical_design/README.md。',
'',
'## 全状态与两个模态频率结果','',
'先逐轨迹统计，再在同一种子内平均8条轨迹，最后对5个种子报告均值±样本标准差。全状态误差是在各场自身正则坐标中计算的相对L2；跨场的坐标缩放不同，因此跨场重点比较相对于同档物理基线的配对改善。频率表统一换算为物理单位rad/s，避免把不同ω0下的rad/τ直接比较。频率是有限时间平滑相位增量诊断，不是不变环面的严格频率证明。',
'']
for duration in (16,64):
    lines += [f'### {duration} s','',
              '| 场强 | 模型 | 全状态相对L2(%) | 模态1绝对频差(rad/s) | 模态2绝对频差(rad/s) | 杆端RMSE(μm，次要) |',
              '|---|---|---:|---:|---:|---:|']
    for level in LEVELS:
        for group in ('H0','H4','GR','GC'):
            lines.append(f'| {int(level[1:4])} mT | {group} | {stat(level,duration,group,"state_relative_l2",100)} | {stat(level,duration,group,"frequency_abs_error_1_rad_s")} | {stat(level,duration,group,"frequency_abs_error_2_rad_s")} | {stat(level,duration,group,"tip_rmse_mm",1000)} |')
    lines.append('')
lines+=['## 预先固定的64 s配对判断','',
'每个主指标均需至少4/5种子方向有利，且平均逐种子相对改善至少5%；全状态、模态1频差和模态2频差必须同时通过。该规则是本次探索性门槛，不是统计显著性或总体保证。下表“通过”仅指方向与幅度条件；任何最终优势解释还需数值不确定性明显更小。',
'',
'| 场强 | 比较 | 全状态胜种子/5；平均相对改善 | 模态1胜种子/5；平均相对改善 | 模态2胜种子/5；平均相对改善 | 三主指标同时满足 |',
'|---|---|---:|---:|---:|---|']
for level in LEVELS:
    for candidate,baseline in [('GR','H0'),('GC','H0'),('GR','H4'),('GC','H4'),('GC','GR')]:
        selected=[r for r in comparisons if r['level']==level and int(r['duration_seconds'])==64 and r['candidate']==candidate and r['baseline']==baseline]
        cells=[]
        for metric in ('state_relative_l2','frequency_abs_error_1','frequency_abs_error_2'):
            r=next(x for x in selected if x['metric']==metric)
            cells.append(r['candidate_wins']+'/5；'+fmt(float(r['mean_paired_relative_improvement'])*100)+'%')
        allr=next(x for x in selected if x['metric']=='ALL_THREE_PRIMARY')
        passed=allr['passed_metric_rule']=='True'
        lines.append(f'| {int(level[1:4])} mT | {candidate}/{baseline} | '+ ' | '.join(cells)+' | '+('是' if passed else '否')+' |')
lines+=['',
'## 独立核验与限制','',
f'- 不导入实验A/B实现的独立审计检查了全部{audit["completed_fits"]}份权重及15个场强—种子数据/流记录；{audit["source_checks"]["protected_files_checked"]}个原保护文件和{audit["source_checks"]["frozen_files_checked"]}个冻结输入均未变化。',
'- 独立重算了式（36）标签的连续RHS、物理初值投影、训练/验证点数、逐元素噪声、checkpoint最小验证误差与complex-step能量梯度。',
'- 从保存的原始64 s轨迹独立重算全状态、杆端和实数dot/cross相位频率。对每档每种子每种模型，在0、32、63.9 s处用独立NumPy向量场作0.1 s连续短段重启，核对保存流的端点。',
'- 每档每种子每个非线性模型的预定第1条测试轨迹还以更紧rtol=5×10⁻¹³、atol=5×10⁻¹⁵重积分64 s，保存状态与频率敏感性。它们是数值诊断，不是所有测试轨迹的全局误差证明。极小的H4频率差应同时参考此数值底噪。',
'- 两模态与八模态独立16 s诊断的杆端差随场强约为0.00440、3.07749、8.91831 μm；120 mT的质量范数速度RMS相对差约5.087%。这些子空间误差不能靠两模态网络预测精度消除。',
'- 增强磁场同时改变了线性谱、模态投影、归一化尺度和有量纲标签噪声，故这里不是把唯一的“非线性系数”孤立改变的因果实验；逐档GR/GC/H0/H4比较保持相同对象和信息。',
'',
'下表取每档五个种子、四种数值积分模型中，预定第1条轨迹的更紧64 s积分敏感性最大值；频率已换算rad/s。这些是观测到的数值敏感性，不是误差上界。最后一列覆盖全部五种预测/参考模型、八条测试轨迹和保存时刻。',
'',
'| 场强 | 更紧积分最大正则状态分量差 | 模态1最大频率差(rad/s) | 模态2最大频率差(rad/s) | 保存时刻全梁斜率最大值 |',
'|---|---:|---:|---:|---:|']
for level in LEVELS:
    omega=float(np.load(OUT/level/'seed_17/data.npz')['omega0'])
    records=[read(OUT/level/f'seed_{s}/evaluation.json') for s in SEEDS]
    refined=[v for record in records for v in record['refinement'].values()]
    slope=max(v['max_abs_slope'] for record in records for v in record['geometry'].values())
    lines.append(f'| {int(level[1:4])} mT | {fmt(max(v["maximum_state_gap"] for v in refined))} | {fmt(max(v["modal_frequency_gap"][0] for v in refined)*omega)} | {fmt(max(v["modal_frequency_gap"][1] for v in refined)*omega)} | {fmt(slope)} |')
lines += [
'',
'## 文件与复算','',
'- protocol.json：训练前锁定协议与输入哈希；execution_script.py为执行脚本快照。',
'- BxxxmT/seed_x/data.npz及data.json：式（36）数据、噪声、物理投影、全梁能量筛查与信号强度。',
'- GR/GC_weights.npz、training.json、history.json：全部30份配对权重、选点记录与成本。',
'- continuous64s.npz：每档每种子的H0/H4/GR/GC/Teacher原始连续状态；evaluation.json：指标、几何和更紧积分敏感性。',
'- independent_metrics_trajectory.csv、independent_metrics_seed.csv：独立核验后的逐轨迹/逐种子指标，含rad/s与相对频差。',
'- independent_aggregates.csv、independent_comparisons.csv、independent_audit_all.json：汇总、配对判断和审计证据。',
'',
'复算入口为上一级experiment_b.py的--stage prepare/train/evaluate/summary；锁定输入变化会报错，不覆盖原始结果。先用科学Python运行independent_audit.py --stage all，再运行report_results.py重建本报告。跨机器时需自行选用已有NumPy/SciPy/PyTorch/Numba环境；脚本没有安装依赖或更改系统设置。',
'']
primary=[r for r in comparisons if r['metric']=='ALL_THREE_PRIMARY' and int(r['duration_seconds'])==64]
yes=lambda candidate,baseline:sum(r['passed_metric_rule']=='True' for r in primary if r['candidate']==candidate and r['baseline']==baseline)
ratios=[]
for level in LEVELS:
    lookup=lambda group:float(next(r['mean'] for r in aggregates if r['level']==level and int(r['duration_seconds'])==64 and r['group']==group and r['metric']=='state_relative_l2'))
    ratios.extend(lookup(group)/lookup('H4') for group in ('GR','GC'))
summary=['## 完成结果','',
f'全部三档、五种子、20次新增训练及15组64 s连续评价已完成。按训练前固定的三主指标方向/幅度门槛，GR相对于H0通过{yes("GR","H0")}/3档，GC相对于H0通过{yes("GC","H0")}/3档；GC相对于GR通过{yes("GC","GR")}/3档。',
f'相对于H4，GR通过{yes("GR","H4")}/3档、GC通过{yes("GC","H4")}/3档。六个“场强×网络”比较中，神经网络的64 s全状态种子均值误差约为H4的{fmt(min(ratios))}–{fmt(max(ratios))}倍。应同时保留“相对二次先验的学习收益”和“相对简单四次解析模型的实用差距”，不能只选其中一个结果。',
'五个种子在各场强复用相同物理初态作为投影目标；三档不能视为15个独立训练种子。上面的通过计数是预定探索性规则，不代表显著性，也不代替物理适用性验证。','']
position=lines.index('## 固定设计与公平性');lines[position:position]=summary
(OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
print('Wrote',OUT/'README.md')
