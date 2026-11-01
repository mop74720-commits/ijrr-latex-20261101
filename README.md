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

## 文件范围

此仓库包含当前主稿实际依赖的 LaTeX 源码、书目、模板、图表和 PDF。历史预览、编译缓存、训练数据、模型权重和参考文献原文不在本仓库中。

为使工程独立编译，仅调整了主文件中一处公式依赖路径；正文、数学内容、图表和所附 PDF 保持本地当前版本。逐文件校验值见 `source_manifest.json`。
