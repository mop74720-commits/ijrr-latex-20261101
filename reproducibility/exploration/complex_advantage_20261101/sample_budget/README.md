# 样本量与训练预算 pilot

使用原式（36）两模态保守数据、原 `train_one` 和 `EnergyNetwork`，保留 GR/GC 各673个实参数。种子为17、43、91。新增训练18次，并重评估6份原12轨迹、10k更新的保存权重；全部完成，无失败。

每个种子按 `seed+41000` 将12个训练轨迹编号打乱，第一条与前三条构成嵌套子集，分别训练10k更新；另用全12轨迹训练30k更新。输入、输出和能量尺度均由相应训练子集按原训练函数计算。每对模型共享训练数据、噪声、初始种子和预算。

权重选择继续使用原规则：每100次更新在固定2048点验证子集上检查带噪声归一化MSE。最终报告使用全部2284点独立验证集及干净标签；干净标签与连续Hamiltonian右端的一致性逐次检查。`clean_gradient_rms` 是未缩放的四分量梯度误差RMS；正则解码器的正负置换使它也等于向量场误差RMS。`common_scale_clean_gradient_rms` 则使用原全12轨迹训练输出RMS逐分量归一化，同一个种子的所有条件共用该尺度。

全局候选选择以总协议规定的 **scaled full-validation** 指标为准；本目录的raw分数仅标记为 `raw diagnostic, not global nomination`。本子任务没有选择候选，也没有读取pilot测试性能。

| 条件 | scaled平均log(GR/GC) | GC方向胜数 |
|---|---:|---:|
| 原12轨迹、10k保存权重 | 0.161277 | 2/3 |
| 1轨迹、10k | 0.629264 | 3/3 |
| 3轨迹、10k | 0.193562 | 2/3 |
| 12轨迹、30k | -0.049864 | 2/3 |

1轨迹条件在这三个种子的干净验证误差中给出较强的复网络信号。3轨迹和30k条件均保留了不利种子；30k的平均scaled分数为负，说明增加训练预算并不保证复值相对优势。上述均为探索验证结果，尚不构成独立测试或普遍优势结论。

所有权重位于 `traj*/seed_*/{GR,GC}_weights.npz`，旁边保存完整历史和记录。`all_runs.csv` 包含24次评估，`validation_paired_seed_metrics.csv` 和 `validation_summary.csv` 提供各指标逐种子配对及汇总；`scaled_validation_diagnostics.json` 为全局协调器提供共同尺度分数。非线性余项相对误差在每份记录中单独保存。

`execution_script.py` 保存实际训练时的脚本，SHA256为 `3aec83aab5a17b8f9d71dabadba4401c46b526f90f872db525a2272a9cf6af02`。训练结束后仅澄清raw诊断表名和全局提名元数据；稳定的主脚本SHA256为 `661a94bbc807c35c963646a5e79824351870b7f10165268b324bafd0b4aee19f`。初始协议保存在 `protocol.initial_execution.json`。

`verification.json` 记录：24份保存权重独立复算的raw/scaled验证RMS与原记录差为0；训练专用输入尺度差为0；最佳步与保存历史一致；453个原稿、程序和实验保护文件未变。审计环境在导入NumPy/PyTorch前设置 `MKL_THREADING_LAYER=SEQUENTIAL`，未使用重复OpenMP运行时的绕过选项。

重现命令（科学Python，原结果保持保护）：

```powershell
& "D:/Apps/Miniconda3/python.exe" -B "E:/1030/IJRR_1025_20261030/exploration/complex_advantage_20261101/sample_budget.py" --stage plan
& "D:/Apps/Miniconda3/python.exe" -B "E:/1030/IJRR_1025_20261030/exploration/complex_advantage_20261101/sample_budget.py" --stage train
```

以上为首次执行命令。已完成目录使用 `--stage summary` 重建本探索汇总，以保留原训练环境记录；复算训练应在单独隔离的副本中进行。可选的 `--stage rollout` 使用已冻结权重及低层中点推进，在原914时刻执行 $h_0/16$ 诊断，输出仍只在本目录。此pilot交付时尚未执行该可选测试。
