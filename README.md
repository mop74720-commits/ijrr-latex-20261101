# 磁驱柔性杆实复哈密顿学习论文

采用 SAGE Harvard 模板的中文 IJRR 审阅稿，当前 PDF 为16页。

- [阅读当前论文](main.pdf)
- [LaTeX 主文件](main.tex)
- 正文章节位于 `sections/`，表格位于 `tables/`，插图位于 `figures/`。
- `audit/frozen_derivation.tex` 为主稿实际引用的公式依赖，已随工程提供。

## 编译

使用 XeLaTeX 和 BibTeX。工程保留主稿字体设置，需要 Times New Roman、Arial、Courier New、SimSun 和 Microsoft YaHei，以及 `fontspec`、`xeCJK` 等 LaTeX 包。SAGE 类文件和参考文献样式已包含在仓库内。

PowerShell 下，在仓库根目录执行：

```powershell
./build.ps1
```

也可使用命令：

```text
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

在 LaTeX 编辑器中将 `main.tex` 设为主文档，编译器选择 XeLaTeX。

## 代码、数据、权重与新增 A/B 实验

已补充 [复现材料总入口](reproducibility/README.md)：原训练/实验脚本、20份主训练权重、5份训练/验证/测试数据，以及A/B新增代码、权重、统计与图表。较大的完整轨迹归档位于同仓库 [Release](https://github.com/mop74720-commits/ijrr-latex-20261101/releases/tag/research-artifacts-20261101)，逐文件SHA-256见发布清单。

[真实强场梁高保真验证仍未完成](reproducibility/VALIDATION_STATUS.md)。此次上传没有修改论文，也没有把已有模态/网格敏感性检查写成强场高保真验证。

## 文件范围

仓库根目录包含当前主稿实际依赖的 LaTeX 源码、书目、模板、图表和 PDF；`reproducibility/` 及 Release 提供科研复现材料。历史稿件预览、编译缓存和参考文献原文不在本次发布范围内。

为使工程独立编译，仅调整了主文件中一处公式依赖路径；正文、数学内容、图表和所附 PDF 保持本地当前版本。逐文件校验值见 `source_manifest.json`。
