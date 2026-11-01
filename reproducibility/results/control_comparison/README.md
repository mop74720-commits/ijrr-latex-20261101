# 同一两模态梁的三种控制插入实验

使用学习梁相同的nu、d和完整余弦势，比较显式物理模型上的控制离散，没有训练新网络。输入u=By/B0是一个横向磁场，控制势为−u sin(d·X)，同时作用于两个模态。

- I：完整受控隐式中点。
- II：自由中点后施加完整精确势冲量。
- III：半势冲量、自由中点、半势冲量。

势冲量K_s^u(X,Y)=(X,Y+s u d cos(d·X))由二类生成函数X·P−s u sin(d·X)给出。零输入时三者回到同一自由步；线性化极限中I与III重合，本弱非线性模型中也近乎重合。

## 协议、参考和统计

正式计算前保存protocol.json；protocol_preflight.json仅用于短时实现预检，未用于长轨迹。正式功残差采用基能量h_*与保持输入的功u_k[sinθ_next−sinθ_old]，子流/求积功另列。

五个原种子块各8条完整测试初值，共40条，跟踪同一公共零初值可行参考。参考与反馈为：

```text
u_ref(tau)=0.05*(1-exp(-(tau/(0.25*omega0))^2))*sin(0.85*tau)
s(x)=sin(d·X), v(x)=cos(d·X)*d·(nu*Y)
u_k=clip(u_ref(tau_k)−[s(x_k)−s_ref]−0.5[v(x_k)−v_ref], −0.2,0.2)
```

按各方案自身状态在步起点采样并保持u_k。增益和预算相同，实际控制序列不同。kp=1、kd=0.5针对无量纲时间，不直接当作SI速度增益。B0=0.03T，上限对应6mT，属于模型仿真预算，尚非实物限幅标定。

实际时域7.994744909秒，h0=0.009999813692（约0.875848ms）；四档4h0、2h0、h0、h0/2使用1142个共同输出时刻。开环采用同一平滑外加输入；闭环与无控制和连续反馈参考比较。相对连续反馈的误差含采样保持和积分效应，不能据此断言闭环二阶。

每轨迹先求时间RMSE，每块平均8条，再报告5块均值和样本标准差。种子只是初值分组，不是控制器训练或硬件重复；没有按测试表现调节增益。

## 原初值实验与追加诊断

40初值开环误差受自由模态相位误差主导，II也显示接近二阶的表观斜率。原结果全部保留。在记录input_response_amendment.json后，对已生成公共参考的同一零初值、同一输入、全部三方案和四档步长追加确定性响应诊断。

12条响应中，I/III最细两档状态观察阶约2.0000，II由1.1041、1.0503趋至1.0245。追加诊断不作新增种子计数。公共零初值参考收紧容差后的最大状态变化7.51e−13。原40初值结果不构成II一般二阶保证。

## 结果与边界

h0杆端跟踪RMSE（mm，均值±样本SD）：无控制约1.3312±0.1723，I为0.7453±0.1271，II为0.7486±0.1275，III为0.7453±0.1271。后半时域分别约1.3320、0.5212、0.5260和0.5212mm；完整状态相对跟踪误差分别约3.3027、1.6033、1.6104和1.6033。

I与III近乎相同，无实际优劣结论。全部档位max|u|=0.181287（约5.439mT），没有饱和，不提供抗饱和证据。积分u²dt是控制努力，单位s，并非物理能耗。闭环参考模态半径可达0.228，超过训练初值上限0.20；这里使用显式物理能量，不声称网络外推已验证。

48组结构检查覆盖首个种子块8初值、2步长、3映射。固定外生u的完整4D辛缺陷≤4.72e−16；计入反馈du/dx后约0.00540–0.02106。全部正式轨迹最大中点残差小于1.3e−16。不提供闭环保辛、稳定性或安全域证明。

## 数据与复算

protocol.json记录6份原输入的项目相对路径和哈希；common_reference.npz包含目标、初值、开环/连续反馈/无控制参考。28组open_loop、feedback、uncontrolled档案保留状态、全部内部步控制值和功/求解诊断。

trajectory_metrics.csv、seed_metrics.csv、paired_summary.csv分别含1120轨迹、140初值块和72配对。open_loop_orders.csv保留原40初值观察阶；input_response_*保留追加协议、12条响应及阶数。implementation_verification.json和structure_diagnostics.csv记录独立四维root与解析/差分Jacobian核查。

completed.json、execution_environment.json、execution_script.py记录原始数值档案及真实执行快照，历史哈希保留。input_response_execution_script.py单独记录追加响应执行时的代码。

```powershell
python -B "./code/control_comparison.py" --stage plan
python -B "./code/control_comparison.py" --stage verify
python -B "./code/control_comparison.py" --stage run
python -B "./code/control_comparison.py" --stage input-response
python -B "./code/control_comparison.py" --stage assets
python -B "./audit/verify_control_comparison.py"
```

已有completed.json时run拒绝覆盖；现工程可用input-response复核保存响应、用assets重画。从新数值目录重建时保留正式协议和追加amendment记录，再按命令执行。依赖版本见execution_environment.json。

独立复算报告为audit/verify_control_comparison.json。范围限于低维模型及规定输入，没有验证额外扰动、完整四场、接触、高模态、实物或学习控制器。
