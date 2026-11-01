# 学习误差与KAM条件的定量联系

本补充计算域级误差及保守包络，不给出严格KAM阈值或不变环面证书。正式稿件建议使用**本轮新变质量权重**的 `pullback/` 结果；目录根保留原常质量权重的完整独立审计，两者坐标与域不同，不能直接横比。

## 新变质量模型

`pullback/protocol.json` 在性能计算前固定新正则域 `D_(q,p)=[−.25,.25]^4`。所有保存训练/验证/测试状态及中点输入均在域内，最大绝对分量 `.19922915`。坐标变换为 `X=q+q^3, Y=p/(1+3q^2)`；G先验是原二次物理主部的拉回，真值和网络的已知主部严格抵消。学习误差是网络残差减去 `Vnl(q+q^3)`。

审计包含5种子GR/GC共10模型，没有训练或检查点选择。使用17^4端点网格、16^4复合中点网格及65536独立扰动Sobol点。范数为

`N6=(|D|^-1 integral_D sum_j |partial_j e|^6 dx)^(1/6)`。

内层是分量6范数，原训练仍为二次损失。该后验指标不能自动满足Chen2022的训练/分布/泛化假设。未归一化的Lebesgue L6量需乘 `|D|^(1/6)=.629960525`。

| 量 | GR | GC |
|---|---:|---:|
| N6，种子均值±样本SD | (8.660±1.540)e−5 | (8.121±1.068)e−5 |
| 对齐后采样能量最大误差，种子均值±样本SD | (2.512±.597)e−5 | (1.944±.371)e−5 |
| 学习C0保守包络，全部5种子最大值 | 4.04664e−4 | 2.82305e−4 |
| 物理+学习C0账面包络，全部5种子最大值 | 4.10652e−4 | 2.88294e−4 |

包络由网络显式权重的Hessian绝对值界、最近网格点Taylor展开及均值求积误差构造，物理残差Hessian重新计入 `diag[(sin(theta)−theta)d_j f''(q_j)]`。这些是实数算术不等式下的可复算保守包络；浮点未作向外区间舍入，不能称机器区间认证。

## 可积比较参考

角平均 `K(I)=nu.I+bar h4(I)` 明确定义为比较参考，I仍取原物理模态坐标的谐振子作用量。域为 `[.0008,.02]^2`，参考半径`.04–.20`，其拉回坐标绝对分量不超过`.224<.25`，故包含在新审计域中。K的Hessian常数，最小奇异值5.80643e−4。完整目标的可积性没有因此得到证明。

`H*−K` 保留角依赖四次项，C0包络5.98892e−6；仅余弦Taylor六阶项界为2.39336e−9，两者不能混淆。有限整数集合0<|k|1≤32上，全参考作用量矩形的频率区间下界约.152123，不代表全整数格Diophantine证书。

KAM定理需要具体规定的解析/高阶范数、变换图册及Jacobian界、非共振和阈值epsilon_*。新旧C0预算均不替代这些条件，也不提供Chen定理11的概率环面结论。二次主部精确Cayley修改频率另列于 `reference_budget.json`，它不是完整非线性格式的修改Hamiltonian界。

## 文件与复算

- `manuscript_subsection.tex`：供第4节追加的紧凑片段，表格使用新变质量模型结果。
- `pullback/summary.json`：全部新变质量模型数据及种子统计。
- `pullback/independent_verification.json`：独立Torch自动微分重算全部模型、全部点与链式Hessian上界。
- 根目录 `summary.json`、`independent_verification.json`：原常质量权重审计，完整保留。
- `../../audit/kam_error_bridge_notes.md`：解析包络、来源条件及旧坐标结果的详细说明。

从工程根运行：

```powershell
python -B code/kam_error_bridge.py --stage plan
python -B code/kam_error_bridge.py --stage run
python -B results/kam_error_bridge/verify_independent.py
python -B results/kam_error_bridge/pullback_audit.py
python -B results/kam_error_bridge/pullback/verify_independent.py
```

所有旧输入只读并进行运行前后哈希核验；新审计的前置条件是新变质量数据与全部10份GR/GC权重已经生成。没有改变式（36）之前的推导。
