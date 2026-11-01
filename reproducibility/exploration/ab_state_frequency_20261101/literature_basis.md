# 实验 B 的文献依据与物理边界

检索与核对日期：2026-11-01。此文件只服务独立探索，不改论文，不上传仓库。

## 文献能确定什么

1. **Richter, M., Venkiteswaran, V. K., and Misra, S. (2021).** *Multi-Point Orientation Control of Discretely-Magnetized Continuum Manipulators*. IEEE Robotics and Automation Letters, 6(2), 3607–3614. [作者机构全文](https://surgicalroboticslab.nl/wp-content/uploads/2021/02/richter21-ral.pdf)。其模型用截面姿态确定磁矩方向，式 (10) 的磁力矩为磁矩与场的叉积；磁场模型经过测量校准。它支持“实际姿态、场分布和对象参数必须进入适用性判断”，没有给出本工程磁体的通用场强上限。

2. **Sano, T. G., Pezzulla, M., and Reis, P. M. (2022).** *A Kirchhoff-like theory for hard magnetic rods under geometrically nonlinear deformation in three dimensions*. Journal of the Mechanics and Physics of Solids, 160, 104739. [DOI](https://doi.org/10.1016/j.jmps.2021.104739)，[作者预印本](https://arxiv.org/abs/2106.15189)，[EPFL 作者全文](https://www.epfl.ch/labs/flexlab/wp-content/uploads/2022/05/99_Sano_JMPS_MagneticRods.pdf)。该工作采用几何非线性杆，讨论高矫顽力材料的固定磁化及被忽略的长程磁相互作用。因此固定偶极矩和几何近似需要对象依据，不能仅通过小斜率检查加以保证。已核对作者全文及期刊、预印本记录；全文印刷第 2 页引言首段涉及高矫顽力，第 4 页 §2.2 说明磁化及能量假设，第 5 页 §3 开始杆的降维。

3. **Kim, Y., Yuk, H., Zhao, R., Chester, S. A., and Zhao, X. (2018).** *Printing ferromagnetic domains for untethered fast-transforming soft materials*. Nature, 558, 274–279. [期刊记录](https://www.nature.com/articles/s41586-018-0185-0)，[作者机构全文](https://zhaolab.stanford.edu/sites/g/files/sbiybj21256/files/media/file/s41586-018-0185-0.pdf)。正文图 1–3 及相关段落使用 **200 mT 致动场**；这是 NdFeB 颗粒/硅胶复合材料的实验先例。文中的 **50 mT 打印场**、**约 2.7 T 磁化脉冲**与致动场用途不同。该论文只说明百毫特斯拉量级存在实验装置先例，不能证明本工程未注明材料的端部磁偶极在 120 mT 下满足恒磁矩假设。

## 由此采取的实验范围

最终预先锁定 30、60、120 mT，保持梁材料、几何和端部偶极参数，逐档重建磁刚度、广义特征值、振型和正则缩放。该范围称为**原近似模型内的磁场参数压力测试**。没有把文献中的 200 mT 直接当作本对象允许上限，也没有把非线性残差独立乘以任意系数。

本工程缺少磁体材料、矫顽力、退磁曲线、磁体尺寸、实测场图及对应实验标定，故目前不能声称已从文献确定本对象的严格物理场强上限。选择的全梁斜率门槛 0.15 是明确披露的数值筛查条件，不是文献材料常数。

## 本次自己的运动学核对

原模型使用端部斜率 \(s=w_x(L)\) 近似转角，同时保留 \(-\mu B\cos s\)。若只把磁矩方向改为图形曲线的真实切向角 \(\theta=\arctan s\)，则

\[
\mu B(1-\cos s)=\mu B(s^2/2-s^4/24+O(s^6)),
\]
\[
\mu B[1-\cos(\arctan s)]=\mu B(s^2/2-3s^4/8+O(s^6)).
\]

四次系数相差 9 倍。这是对现有近似的代数诊断，不是上述文献直接给出的结论，也不是一套完整几何非线性杆推导。它不否定原模型的线性主部，但说明小斜率本身不能证明其保留的四次磁残差具有物理高阶准确性。本次没有偷偷替换原模型。

此外，预先选定轨迹的 2/8 模态敏感性检查随场强出现可见差异，详见 `physical_design/verification.json`。因此两模态连续参考是**所选有限维近似的数值参考**，不能称为真实梁的精确真值。

## 小斜率域内的补充代数诊断

此项在协议锁定后计算，只用于解释非线性强度，不改变场强、训练、评价或成功判据。它不是文献定理或论文新增推导。

令 \(D=\operatorname{diag}(\nu)\)、\(\theta=d^TX\)、\(\rho=d^TD^{-1}d\)。原弹性刚度正定，经当前正则缩放得到 \(D-dd^T\succ0\)，故 \(\rho<1\)。由 Cauchy–Schwarz 及正弦 Taylor 余项界，

\[
|\theta|\le\sqrt{\rho}\,\|DX\|_{D^{-1}},\qquad
\frac{\|d(\sin\theta-\theta)\|_{D^{-1}}}{\|DX\|_{D^{-1}}}
\le\rho\theta^2/6.
\]

这里 \(\|v\|_{D^{-1}}=\sqrt{v^TD^{-1}v}\)，零状态处取连续延拓。三档的 \(\rho\) 分别为 0.06775963、0.12243907、0.20369800。即使使用整个 \(|\theta|\le0.15\) 筛查域，加权力比上界也仅分别为 **0.02541%、0.04591%、0.07639%**。这解释了为什么在原模型的小斜率域内，单靠提高场强并不能使相对磁非线性任意增强。

这是点态的加权范数界，与训练记录中的欧氏 L2 力比定义不同，不能混用。它也不直接限制长期相位误差或证明神经网络没有收益。

## 结论措辞边界

- 可以判断：在已锁定近似模型、数据预算、训练方法和场强档位下，神经残差相对 H0、H4 的收益以及 GC 相对 GR 的一致性。
- 暂不能判断：在已验证的强非线性真实梁上，CHNN 是否具有或缺乏一般优势。
- 不以网络结果反向放宽场强、斜率或初值范围；不删除不利种子，不把未解决的物理适用性写成已通过。
