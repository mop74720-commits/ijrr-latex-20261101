# 训练、数据、权重与实验复现材料

发布日期：2026-11-01。这里补齐已有科研材料，不修改论文，也不表示尚未完成的验证已经完成。

## 文件状态

| 用户列出的项目 | 现在提供的位置 | 状态与范围 |
|---|---|---|
| 原训练代码、实验脚本 | [code](code/)、[历史来源入口](legacy_sources/learning/) | 当前代码及必要来源依赖均已提供 |
| 神经网络权重 | [主论文训练](results/neural_learning_10000/)、[A/B](exploration/ab_state_frequency_20261101/experiment_b/) | 主论文 R/C/GR/GC × 5 种子共20份；B另有30份，其中10份复用主权重、20份新增 |
| 原训练、验证、测试数据 | [learning/data](results/learning/data/) | 5份完整NPZ，含标签、噪声、初值及测试教师/连续参考状态，不只是汇总表 |
| A/B新增实验代码与结果 | [A/B总报告](exploration/ab_state_frequency_20261101/README.md)、完整Release | 3档场强、5种子、全部权重、原始64秒轨迹及统计/图表已提供 |
| 真实强场梁高保真验证 | [状态与边界](VALIDATION_STATUS.md) | **尚未完成**；已有模态/网格敏感性和历史跨求解器材料不能替代该验证 |

仓库可直接浏览代码、JSON/CSV报告、图表、主数据和权重。完整科学数据含约1.12 GiB的代码/结果/探索记录，较大的NPZ轨迹通过同仓库Release提供：

- [Release下载页](https://github.com/mop74720-commits/ijrr-latex-20261101/releases/tag/research-artifacts-20261101)
- [完整复现ZIP](https://github.com/mop74720-commits/ijrr-latex-20261101/releases/download/research-artifacts-20261101/full-reproducibility-20261101.zip)
- [ZIP的SHA-256校验值](https://github.com/mop74720-commits/ijrr-latex-20261101/releases/download/research-artifacts-20261101/SHA256SUMS.txt)

ZIP保留 `reproducibility/` 前缀，解压到仓库根目录即可补齐大轨迹。已有同名文件应相同，可先核对SHA-256，避免覆盖自己的修改。包内完整清单为 [package_manifest.json](package_manifest.json)，包含每个文件的字节数和SHA-256；不包含清单自身。

## 先检查现有数据和结果

从仓库根目录执行；工具也支持从任意工作目录使用绝对路径调用：

```text
python -B reproducibility/tools/verify_package.py --available
python -B reproducibility/tools/inspect_training_data.py
```

第一条明确区分已下载文件的校验和完整包校验。仅克隆仓库时大轨迹缺失是预期状态，不能把 `partial` 解释成完整复现已通过。下载解压完整ZIP后运行：

```text
python -B reproducibility/tools/verify_package.py --full
python -B reproducibility/tools/run_analysis.py
python -B reproducibility/tools/check_physics_source.py
```

统计与物理入口先后校验发布包，并将新结果写入仓库根目录的 `reproduction_runs/` 唯一子目录，不覆盖已发布结果。物理入口只核对原有限元方程和线性频率网格敏感性，不是强场实物/高阶物理验证。详细说明见 [tools/README.md](tools/README.md)。

## 数据与权重是什么

- `results/learning/data/seed_{17,43,91,137,211}.npz` 是主论文实际使用的合成数据，由式（36）的保守两模态中点特例生成。数据保留训练和验证标签/干净标签/噪声，以及测试教师和连续参考；不是实测梁或COMSOL原始导出。
- `results/neural_learning_10000/` 是10000次更新的20组端到端网络结果。`results/neural_learning/` 的2500步先导和 `results/learning/fits/` 的解析特征基线属于不同阶段，不能混用。
- B 的 `experiment_b/BxxxmT/seed_x/data.npz` 为每档重建后的教师数据，`continuous64s.npz` 保存H0/H4/GR/GC/Teacher完整状态，`*_weights.npz` 是原选权重。A保留原权重并依赖随包附带的 `complex_advantage_20261101/fine_step/` 中 h32/h64 轨迹。
- `code/`、`results/`、`exploration/` 均按原相对层级保存。`vendor/beam/run_beam_validation.py` 是原有限元源的未修改副本；`legacy_sources/learning/` 保存历史入口所需5个脚本/依赖文件。没有把论文文献全文或原始Word草稿一起公开。

## 从头训练与历史审计

主训练脚本默认2500步。重训10000步时，在 `reproducibility/` 目录运行并使用新输出名：

```text
python -B code/train_neural_hnn.py --steps 10000 --batch-size 512 --max-seconds 16 --output-name neural_learning_retrain
```

这会重新训练R/C/GR/GC五种子，不是读取旧权重，耗时较长。原主结果输出名为 `neural_learning_10000`。不同Python、BLAS或数值库可能产生微小差异，应保留新运行记录，不要修改历史协议来绕过一致性检查。环境参考 [environment-observed.json](environment-observed.json) 和各原实验协议；没有自动安装依赖或改变系统配置。

原脚本、输入、数据、权重和科学结果按原字节保存。历史报告中的“未上传GitHub”描述其当时的执行边界，本次发布状态以本页为准。原 `protected_manifest.json` / `verify_final.py` 的2409文件保护检查针对原开发工作区，包含未公开的稿件历史；它不是本发布包的文件范围。发布工具使用明确的新清单，不把历史保护记录冒充跨机器重新完成的核验。

## A/B结果边界

120 mT时GC相对GR的全状态、两个模态频率通过预设探索规则，但五种子证据尚弱，第二频率改善区间跨零，统计检验未显著。各场强下H4仍优于网络。B仍是既定小斜率、恒偶极近似模型内的条件测试，不构成真实强场梁高保真验证。详见 [A/B总报告](exploration/ab_state_frequency_20261101/README.md) 和 [文献依据](exploration/ab_state_frequency_20261101/literature_basis.md)。
