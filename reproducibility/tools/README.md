# 发布包的核验与隔离复算

这些工具使用各自文件位置解析包根目录，可以从任意工作目录运行。默认清单为 `reproducibility/package_manifest.json`，也可用 `--manifest` 指定。原科学脚本、历史协议、权重与结果保持原字节；新运行产物写入仓库根目录 `reproduction_runs/` 的唯一新子目录。

```text
python -B /path/to/repository/reproducibility/tools/verify_package.py --available
python -B /path/to/repository/reproducibility/tools/inspect_training_data.py
```

`--available` 校验已下载文件，逐项列出未下载文件，并报告 `partial`；它不代表完整 Release 已核验。哈希或字节数不符始终失败。训练检查只读五份原数据与20份主权重、30份AB权重，输出字段尺寸、数据类型和数量，不输出原数组。

下载完整 Release ZIP 并按原层级解压到仓库根目录后，可执行：

```text
python -B /path/to/repository/reproducibility/tools/verify_package.py --full
python -B /path/to/repository/reproducibility/tools/run_analysis.py
python -B /path/to/repository/reproducibility/tools/check_physics_source.py
```

两个复算入口都会在运行前后要求完整包通过字节数与 SHA-256 校验。统计入口调用未改动的 `analysis/build_analysis.py`，只重定向输出目录，重新计算原始保存轨迹的种子统计、独立检查和图。没有重新训练。

物理源入口校验 `vendor/beam/run_beam_validation.py` 的锁定 SHA-256，在内存适配原 `build_models.RAW_SOURCE` 和输出路径后执行原 `check_reconstruction.py`。它只核对有限元/有量纲方程及线性频率网格敏感性，不构成真实强场梁高保真验证。

发布包完整性检查与原开发工作区的2409文件保护审计是不同范围。历史审计原文件作为证据保留；这些工具明确记录没有重新执行该历史总保护审计。不同 Python/数值库版本可能产生微小浮点差异，应使用新运行目录保存，不能改写历史协议以绕过检查。
