# 保守标量KAM相关数值诊断

protocol.json保存运行前设置；results.json保存全量指标及适用边界。

continuous_references.csv是连续参考周期、作用量和数值twist；case_metrics.csv是18组离散实验；step_sensitivity.csv是步长减半比较。

每个NPZ包含time、state(q,p)、physical_energy、physical_action、unwrapped_phase、spectrum_omega、spectrum_power及downward_crossing_times。

所有参数均为无量纲诊断设置。这些结果不构成KAM认证，也不证明全系统长期稳定。
