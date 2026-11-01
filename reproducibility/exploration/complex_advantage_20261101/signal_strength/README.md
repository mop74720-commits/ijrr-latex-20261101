# 初态幅值与标签噪声 pilot

仅在此目录写入探索数据；原 `code/`、`results/`、`manuscript/` 由上层 `protected_manifest.json` 校验，不修改。所有结果为 3 种子探索结果，不能代替预设 5 个新种子的确认。

固定原两模态 Hamiltonian、物理参数、673 个实参数、GR/GC 架构和 `train_neural_hnn.train_one`。初态幅值乘 2 后重新积分非线性中点方程，保留训练 12 条 ×2 s、验证 4 条 ×4 s、测试 8 条 ×15.992993 s。测试输出网格为原 40 s 网格的约 16 s 前缀。独立连续参考由 DOP853 产生，容差 `rtol=2e-11, atol=2e-13`。

两种条件为均匀标签噪声 ±1e-6 与 0；相同幅值域的新轨迹在两噪声条件共享。训练、验证噪声分别由 `seed+10000`、`seed+110000` 产生。保持原权重选择：每 100 更新在原固定 2048 个验证点上比较，验证标签噪声与本条件相同。完整干净验证向量场仅在训练结束后评分，不用于更换检查点。

所有内部中点轨迹和连续参考在完整粗网格上的 `max|d·X|≤0.22 rad` 是计算压力筛选规则，不是完整物理有效性验证。超限不截断样本，不开展拟合。`data_verification.json` 另验证了初态倍增、参数不变、噪声种子独立以及新非线性轨迹不同于旧轨迹直接倍增。

每个模型训练 10000 次，batch=512，种子 17/43/91。`*_training.json` 保存训练时间、原检查点指标、完整干净验证指标与数据/权重哈希。复值网络仍是复隐层加实标量 Hamiltonian 读出；没有在本探索中引入非零虚部物理模型。

`*_fine.npz` 保留 h0/16 的所有共同输出时刻上的预测、DOP853 参考、同一步长非线性教师与 H0。`*_fine.json` 记录每条轨迹的时间 RMSE、完整状态相对 L2，然后对 8 条求均值；`summary.json` 再对种子均值报告 mean/sample SD。H0/GR/GC 使用相同初态和输出网格。测试轨迹不用于配置挑选；配置提名仅采用上层协议规定的完整干净验证指标。

执行：

```powershell
& "D:/Apps/Miniconda3/python.exe" -B "E:/1030/IJRR_1025_20261030/exploration/complex_advantage_20261101/signal_strength.py" --stage all
```

也可分别使用 `prepare`、`train`、`rollout`、`summary`。已有数据、权重和执行快照按哈希复用；不覆盖不同配置。失败记录与未通过门槛的数据保留供检查。

可复用函数：`generate_dataset(seed, amplitude_multiplier, noise_bound, output_dir)`，输出必须在本次探索树内。它能为预先固定的后续确认设置生成新数据，本 pilot 未提前运行确认种子。
