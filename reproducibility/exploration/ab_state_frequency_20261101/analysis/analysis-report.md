# A/B strict analysis bundle

本包评价细步长/连续流以及磁场分级下，神经残差是否同时改善完整状态和两模态频率。正文、原实验、权重与GitHub不在本包修改范围。B的物理解释仍受近似斜率/几何截断与高模态条件限制，不能把数值包络通过解释成物理模型已经校准。

| Dataset | Model | State relative L2 | Mode1 freq error (rad/s) | Mode2 freq error (rad/s) |
|---|---|---:|---:|---:|
|A: original field, 63.989 s|H0|0.005769117 ± 0.00107|0.0001160249 ± 1.95e-05|0.0002168881 ± 4.47e-05|
|A: original field, 63.989 s|H4|9.927613e-07 ± 3.52e-07|2.029762e-08 ± 6.56e-09|3.680031e-08 ± 1.36e-08|
|A: original field, 63.989 s|GR|0.002522465 ± 0.00107|3.928705e-05 ± 1.64e-05|9.641705e-05 ± 3.73e-05|
|A: original field, 63.989 s|GC|0.001841539 ± 0.00116|2.693892e-05 ± 1.15e-05|7.233053e-05 ± 5.56e-05|
|B: 30 mT, 64 s|H0|0.005769783 ± 0.00107|0.0001160249 ± 1.95e-05|0.0002168881 ± 4.47e-05|
|B: 30 mT, 64 s|H4|9.928758e-07 ± 3.52e-07|2.029763e-08 ± 6.56e-09|3.68003e-08 ± 1.36e-08|
|B: 30 mT, 64 s|GR|0.002522756 ± 0.00107|3.928705e-05 ± 1.64e-05|9.641705e-05 ± 3.73e-05|
|B: 30 mT, 64 s|GC|0.001841751 ± 0.00116|2.693892e-05 ± 1.15e-05|7.233053e-05 ± 5.56e-05|
|B: 60 mT, 64 s|H0|0.01017187 ± 0.00191|0.0001994691 ± 3.39e-05|0.0003853835 ± 7.89e-05|
|B: 60 mT, 64 s|H4|1.636933e-06 ± 5.81e-07|3.280752e-08 ± 1.07e-08|6.094984e-08 ± 2.24e-08|
|B: 60 mT, 64 s|GR|0.004415451 ± 0.00181|6.288362e-05 ± 2.28e-05|0.0001687451 ± 6.21e-05|
|B: 60 mT, 64 s|GC|0.002870416 ± 0.00158|3.969841e-05 ± 1.5e-05|0.0001154707 ± 7.34e-05|
|B: 120 mT, 64 s|H0|0.01608301 ± 0.00307|0.0003012015 ± 5.29e-05|0.0006172696 ± 0.000125|
|B: 120 mT, 64 s|H4|2.288748e-06 ± 8.24e-07|4.423042e-08 ± 1.48e-08|8.586183e-08 ± 3.18e-08|
|B: 120 mT, 64 s|GR|0.006767704 ± 0.00226|7.529306e-05 ± 2.94e-05|0.0002711479 ± 8.95e-05|
|B: 120 mT, 64 s|GC|0.003913026 ± 0.00203|5.707911e-05 ± 1.72e-05|0.0001542497 ± 9.51e-05|

GC相对GR的三指标联合方向探索规则由以下条件达到：B: 120 mT, 64 s。这个正向发现应保留；其他条件未达到并不抹去它。同一分析仍保留H0/H4全部对照，是否相对H4获益见逐项结果。

## Statistical interpretation

60个预定义主检验中，Holm校正后p<.05的项数为0。n=5的双侧符号检验最小原始p=.0625，方向全胜也不等于统计显著。探索规则为至少4/5种子同方向、逐种子相对改善均值至少5%，须三主指标同时满足；它并非显著性或物理有效性证明。完整bootstrap区间与效应量见统计附录。

H4使用已知cos势Taylor精确系数及给定nu、d，不需拟合；GR/GC只共享H0先验并由数据学习残差。H4比较检验已知物理对象上神经模型的实用收益，不是同等知识前提的表达能力定理。

## Training force signal and noise

| B field | nonlinear/linear force L2 | H4 remainder/linear force L2 | nonlinear/label-noise force L2 |
|---|---:|---:|---:|
|30 mT|1.33071e-05 ± 2.7e-06|3.23848e-09 ± 1.39e-09|7.68571 ± 1.89|
|60 mT|2.33049e-05 ± 4.88e-06|5.33242e-09 ± 2.36e-09|9.49096 ± 2.41|
|120 mT|3.61598e-05 ± 7.68e-06|7.3268e-09 ± 3.3e-09|10.3711 ± 2.67|

这是整个训练集的力向量L2范数之比。场强、时间尺度、正则坐标与SI噪声映射同时变化；数值趋势须与信号相对标签噪声的大小一起解释，不能归因于单个网络表示。B严格物理有效性未认证。

## Claim candidates

- Claim: A: original field, 63.989 s，GR相对H4的方向证据。
  - Source evidence: comparisons.csv；三指标胜数{'state_relative_l2': 0, 'frequency_error_si_1': 0, 'frequency_error_si_2': 0}。
  - Allowed wording: 在这5个种子的指定时域/模型内，未达到三主指标共同方向改善的探索判据。
  - Forbidden stronger wording: 显著优越、普遍优越、实物系统优势或严格连续极限认证。
  - Uncertainty: seed bootstrap不稳定；数值与物理适用性需另看原实验审计。
  - Next check: 保持基线与误差单位一致，依据独立物理有效性证据再决定论文主张。
  - Decision: weaken
- Claim: A: original field, 63.989 s，GC相对H4的方向证据。
  - Source evidence: comparisons.csv；三指标胜数{'state_relative_l2': 0, 'frequency_error_si_1': 0, 'frequency_error_si_2': 0}。
  - Allowed wording: 在这5个种子的指定时域/模型内，未达到三主指标共同方向改善的探索判据。
  - Forbidden stronger wording: 显著优越、普遍优越、实物系统优势或严格连续极限认证。
  - Uncertainty: seed bootstrap不稳定；数值与物理适用性需另看原实验审计。
  - Next check: 保持基线与误差单位一致，依据独立物理有效性证据再决定论文主张。
  - Decision: weaken
- Claim: A: original field, 63.989 s，GC相对GR的方向证据。
  - Source evidence: comparisons.csv；三指标胜数{'state_relative_l2': 4, 'frequency_error_si_1': 5, 'frequency_error_si_2': 2}。
  - Allowed wording: 在这5个种子的指定时域/模型内，未达到三主指标共同方向改善的探索判据。
  - Forbidden stronger wording: 显著优越、普遍优越、实物系统优势或严格连续极限认证。
  - Uncertainty: seed bootstrap不稳定；数值与物理适用性需另看原实验审计。
  - Next check: 保持基线与误差单位一致，依据独立物理有效性证据再决定论文主张。
  - Decision: weaken
- Claim: B: 30 mT, 64 s，GR相对H4的方向证据。
  - Source evidence: comparisons.csv；三指标胜数{'state_relative_l2': 0, 'frequency_error_si_1': 0, 'frequency_error_si_2': 0}。
  - Allowed wording: 在这5个种子的指定时域/模型内，未达到三主指标共同方向改善的探索判据。
  - Forbidden stronger wording: 显著优越、普遍优越、实物系统优势或严格连续极限认证。
  - Uncertainty: seed bootstrap不稳定；数值与物理适用性需另看原实验审计。
  - Next check: 保持基线与误差单位一致，依据独立物理有效性证据再决定论文主张。
  - Decision: weaken
- Claim: B: 30 mT, 64 s，GC相对H4的方向证据。
  - Source evidence: comparisons.csv；三指标胜数{'state_relative_l2': 0, 'frequency_error_si_1': 0, 'frequency_error_si_2': 0}。
  - Allowed wording: 在这5个种子的指定时域/模型内，未达到三主指标共同方向改善的探索判据。
  - Forbidden stronger wording: 显著优越、普遍优越、实物系统优势或严格连续极限认证。
  - Uncertainty: seed bootstrap不稳定；数值与物理适用性需另看原实验审计。
  - Next check: 保持基线与误差单位一致，依据独立物理有效性证据再决定论文主张。
  - Decision: weaken
- Claim: B: 30 mT, 64 s，GC相对GR的方向证据。
  - Source evidence: comparisons.csv；三指标胜数{'state_relative_l2': 4, 'frequency_error_si_1': 5, 'frequency_error_si_2': 2}。
  - Allowed wording: 在这5个种子的指定时域/模型内，未达到三主指标共同方向改善的探索判据。
  - Forbidden stronger wording: 显著优越、普遍优越、实物系统优势或严格连续极限认证。
  - Uncertainty: seed bootstrap不稳定；数值与物理适用性需另看原实验审计。
  - Next check: 保持基线与误差单位一致，依据独立物理有效性证据再决定论文主张。
  - Decision: weaken
- Claim: B: 60 mT, 64 s，GR相对H4的方向证据。
  - Source evidence: comparisons.csv；三指标胜数{'state_relative_l2': 0, 'frequency_error_si_1': 0, 'frequency_error_si_2': 0}。
  - Allowed wording: 在这5个种子的指定时域/模型内，未达到三主指标共同方向改善的探索判据。
  - Forbidden stronger wording: 显著优越、普遍优越、实物系统优势或严格连续极限认证。
  - Uncertainty: seed bootstrap不稳定；数值与物理适用性需另看原实验审计。
  - Next check: 保持基线与误差单位一致，依据独立物理有效性证据再决定论文主张。
  - Decision: weaken
- Claim: B: 60 mT, 64 s，GC相对H4的方向证据。
  - Source evidence: comparisons.csv；三指标胜数{'state_relative_l2': 0, 'frequency_error_si_1': 0, 'frequency_error_si_2': 0}。
  - Allowed wording: 在这5个种子的指定时域/模型内，未达到三主指标共同方向改善的探索判据。
  - Forbidden stronger wording: 显著优越、普遍优越、实物系统优势或严格连续极限认证。
  - Uncertainty: seed bootstrap不稳定；数值与物理适用性需另看原实验审计。
  - Next check: 保持基线与误差单位一致，依据独立物理有效性证据再决定论文主张。
  - Decision: weaken
- Claim: B: 60 mT, 64 s，GC相对GR的方向证据。
  - Source evidence: comparisons.csv；三指标胜数{'state_relative_l2': 5, 'frequency_error_si_1': 5, 'frequency_error_si_2': 2}。
  - Allowed wording: 在这5个种子的指定时域/模型内，未达到三主指标共同方向改善的探索判据。
  - Forbidden stronger wording: 显著优越、普遍优越、实物系统优势或严格连续极限认证。
  - Uncertainty: seed bootstrap不稳定；数值与物理适用性需另看原实验审计。
  - Next check: 保持基线与误差单位一致，依据独立物理有效性证据再决定论文主张。
  - Decision: weaken
- Claim: B: 120 mT, 64 s，GR相对H4的方向证据。
  - Source evidence: comparisons.csv；三指标胜数{'state_relative_l2': 0, 'frequency_error_si_1': 0, 'frequency_error_si_2': 0}。
  - Allowed wording: 在这5个种子的指定时域/模型内，未达到三主指标共同方向改善的探索判据。
  - Forbidden stronger wording: 显著优越、普遍优越、实物系统优势或严格连续极限认证。
  - Uncertainty: seed bootstrap不稳定；数值与物理适用性需另看原实验审计。
  - Next check: 保持基线与误差单位一致，依据独立物理有效性证据再决定论文主张。
  - Decision: weaken
- Claim: B: 120 mT, 64 s，GC相对H4的方向证据。
  - Source evidence: comparisons.csv；三指标胜数{'state_relative_l2': 0, 'frequency_error_si_1': 0, 'frequency_error_si_2': 0}。
  - Allowed wording: 在这5个种子的指定时域/模型内，未达到三主指标共同方向改善的探索判据。
  - Forbidden stronger wording: 显著优越、普遍优越、实物系统优势或严格连续极限认证。
  - Uncertainty: seed bootstrap不稳定；数值与物理适用性需另看原实验审计。
  - Next check: 保持基线与误差单位一致，依据独立物理有效性证据再决定论文主张。
  - Decision: weaken
- Claim: B: 120 mT, 64 s，GC相对GR的方向证据。
  - Source evidence: comparisons.csv；三指标胜数{'state_relative_l2': 5, 'frequency_error_si_1': 4, 'frequency_error_si_2': 4}。
  - Allowed wording: 在这5个种子的指定时域/模型内，达到三主指标共同方向改善的探索判据。
  - Forbidden stronger wording: 显著优越、普遍优越、实物系统优势或严格连续极限认证。
  - Uncertainty: seed bootstrap不稳定；数值与物理适用性需另看原实验审计。
  - Next check: 保持基线与误差单位一致，依据独立物理有效性证据再决定论文主张。
  - Decision: keep

图的读法与解释检查清单见 `figure-catalog.md`。本文件是分析证据包，不是论文Results段落或总实验总结。
