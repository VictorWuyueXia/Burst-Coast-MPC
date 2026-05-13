# Wake Sleep MPC 交接报告

## 0. 本交接文档的目的

本文档给出 wake sleep 模型预测控制基准的当前实现计划。后续开发应将其视为面向下一阶段的工程实施说明。

当前目标是为欠驱动倒立摆建立一个仅使用 MPC 的确定性基准。强化学习与学习得到的 Q 终端值暂缓实现。控制器仍应保留 wake sleep 结构，使后续可在不改变仿真器或控制器接口的前提下加入学习型终端值。

控制器概念如下：

1. 在决策时刻，求解一段短时主动 burst 控制序列。
2. burst 结束后，施加一个简单的 coast 指令。
3. coast 期间不运行 MPC，除非 early wake 触发。
4. 控制器通过 burst 占总预测时域的比例选择 burst 与 coast 的划分。
5. 当前阶段使用固定总预测时域。

## 1. 当前仓库适配情况

当前仓库主要包含规划与规格文档，尚不是完整实现的源码树。相关文档为：

1. `docs/Plan.md`
2. `docs/wake_sleep_mpc_dev_spec.md`

现有规格使用两个时域变量：

1. Burst horizon `N_b`
2. Wake horizon `N_w`

该规格还包含面向未来的学习型 Q critic 终端项。当前开发阶段应将其修改为更简单的仅 MPC 形式：

1. 使用一个固定总时域 `N`。
2. 使用 burst 到总时域的划分比例 `lambda`。
3. 由 `N` 与 `lambda` 计算 `N_b` 和 `N_c`。
4. 在 MPC 候选评估中显式 rollout burst 与 coast 两个阶段。
5. 暂不使用学习型 Q critic。

这样既保留原始 wake sleep 架构，又能降低第一个基准的实现与调试难度。

## 2. 开发目标

构建一个具有显式 burst 与 coast 行为的确定性欠驱动摆杆 swing up 基准。

该基准应展示：

1. 扭矩饱和条件下的能量泵入。
2. 相位感知的时机选择。
3. 紧凑 burst 后接 coast 的控制结构。
4. 不依赖强化学习的 MPC 时域划分选择。
5. 当预测误差或约束风险过大时触发 early wake。

第一版实现不应包含模型失配、过程噪声、学习型 Q 值、神经网络或分段线性预测。这些内容应在确定性非线性 MPC 成功工作后再加入。

## 3. 状态、动力学与约定

使用物理状态：

```text
x = [theta, omega]^T
```

其中：

1. `theta` 为摆角，单位为弧度。
2. `omega` 为角速度，单位为弧度每秒。

采用如下约定：

```text
theta = 0      下垂平衡点
theta = pi     直立目标
```

直立目标状态为：

```text
x_star = [pi, 0]^T
```

使用非线性连续动力学：

```text
dot(theta) = omega
I dot(omega) = - m g l sin(theta) - b omega + u
```

其中：

1. `I` 为绕转轴的转动惯量。
2. `m` 为摆杆质量。
3. `g` 为重力加速度。
4. `l` 为转轴到质心的距离。
5. `b` 为粘性阻尼系数。
6. `u` 为标量扭矩输入。

重要实现说明：

当前仓库规格将角加速度方程中的重力项写为正号。若采用能量定义 `E = 0.5 I omega^2 + m g l (1 - cos(theta))`，且目标为 `theta = pi`，物理一致的符号应为负号：

```text
I dot(omega) = - m g l sin(theta) - b omega + u
```

在信任任何仿真结果前，应先修正该符号。

第一版工作实现中，仿真器积分与 MPC 预测均使用 RK4：

```text
x_next = f_d(x, u, dt) = RK4(x, u, dt)
```

其中 `dt` 为仿真步长。

## 4. 输入约束与欠驱动条件

标量输入集合为：

```text
U = { u : |u| <= u_max }
```

其中 `u_max` 为最大可用扭矩。

欠驱动基准条件为：

```text
u_max < m g l
```

这表示执行器无法在任意角度直接静态抵抗重力保持摆杆。成功 swing up 必须依赖时机选择、能量泵入与相位利用。

## 5. 能量与相位特征

原始动力学状态仍为 `theta, omega`。能量与相位是用于代价、诊断以及后续 CLBF 构造的特征映射。

定义动能：

```text
K(x) = 0.5 I omega^2
```

定义势能，并令下垂平衡点处势能为零：

```text
V(x) = m g l (1 - cos(theta))
```

定义总机械能：

```text
E(x) = K(x) + V(x)
```

直立目标能量为：

```text
E_star = 2 m g l
```

定义归一化能量误差：

```text
e_E(x) = (E(x) - E_star) / E_star
```

使用归一化能量误差而非原始能量误差，因为它能改善求解器尺度。

### 5.1 相位代理

第一版实现不要使用真正的 action angle 坐标，而应使用相位代理。

选择正的速度尺度 `omega_s`。一个实用选择为：

```text
omega_s = sqrt(m g l / I)
```

定义：

```text
a(theta) = sin(theta / 2)
b(omega) = omega / omega_s
r(x) = sqrt(a(theta)^2 + b(omega)^2 + eps_phi^2)
```

其中 `eps_phi` 为小的正则化常数。

定义相位代理分量：

```text
c_phi(x) = a(theta) / r(x)
s_phi(x) = b(omega) / r(x)
```

在直立目标处，期望相位代理近似为：

```text
c_phi = 1
s_phi = 0
```

定义相位代理误差：

```text
e_phi(x) = [c_phi(x) - 1, s_phi(x)]^T
```

该相位代理不是状态坐标，不应用于传播动力学。它只是一个平滑特征，用来告知代价函数：摆杆是否正以正确的时机与速度方向接近直立能量水平。

## 6. 能量门控

当摆杆能量严重偏离目标时，相位不应主导代价。使用能量门控：

```text
w_E(x) = exp( - e_E(x)^2 / sigma_E^2 )
```

其中 `sigma_E` 为正参数，用于控制系统必须多接近直立能量，才会强烈加权相位与局部稳定化项。

解释如下：

1. 若 `|e_E|` 较大，则 `w_E` 较小，代价主要要求控制器泵入能量。
2. 若 `|e_E|` 较小，则 `w_E` 接近一，相位与局部稳定化变得重要。

## 7. 局部直立误差

定义 wrap 后的直立角度误差：

```text
e_theta(x) = atan2(sin(theta - pi), cos(theta - pi))
```

定义局部直立误差向量：

```text
e_loc(x) = [e_theta(x), omega / omega_s]^T
```

为获得更平滑的 SQP 行为，代价也可使用三角惩罚项，例如：

```text
1 + cos(theta)
sin(theta)
```

而不是完全依赖角度 wrapping。

## 8. 与 CLBF 兼容的能量相位值函数候选

定义能量相位值函数候选：

```text
V_ep(x) =
    q_E e_E(x)^2
  + q_phi w_E(x) e_phi(x)^T Q_phi e_phi(x)
  + q_loc w_E(x) e_loc(x)^T Q_loc e_loc(x)
```

其中：

1. `q_E > 0` 为能量误差标量权重。
2. `q_phi >= 0` 为相位标量权重。
3. `q_loc >= 0` 为局部直立稳定化标量权重。
4. `Q_phi` 为相位代理误差的正定矩阵。
5. `Q_loc` 为局部直立误差的正定矩阵。
6. `w_E(x)` 为能量门控。

这是主要状态代价，并且后续可作为 Lyapunov 候选。

实现建议：

1. 从 `q_phi = 0` 且 `q_loc = 0` 开始，先调试能量泵入。
2. 加入 `q_loc`，调试直立稳定化。
3. 加入 `q_phi`，调试相位时机。
4. 不要同时调节所有权重。

## 9. Burst 与 coast 阶段代价

定义归一化输入：

```text
s = u / u_max
```

对主动 burst 步，使用：

```text
ell_b(x, u, u_prev) =
    V_ep(x)
  + rho_sat (1 - (u / u_max)^2)^2
  + rho_du ((u - u_prev) / u_max)^2
```

其中：

1. `rho_sat >= 0` 权衡对近饱和 burst 控制的偏好。
2. `rho_du >= 0` 权衡输入平滑性。
3. `u_prev` 为上一时刻已施加输入。

饱和吸引项是平滑的，适合 SQP。它鼓励 `u` 接近 `+u_max` 或 `-u_max`，同时不引入二元变量。

不要一开始就使用硬下界：

```text
|u| >= eta_u u_max
```

因为它是非凸的，并会产生不连通可行域。

对 coast 步，使用：

```text
ell_c(x, u_h) =
    V_ep(x)
  + rho_c (u_h / u_max)^2
```

其中：

1. `u_h` 为 coast 指令。
2. `rho_c >= 0` 为 coast 输入惩罚权重。

第一版中，在配置标志后实现两种 coast 模式：

1. `hold_last`: `u_h = v_{N_b - 1}`
2. `zero`: `u_h = 0`

若当前研究意图是测试 coast 期间的常值作动，先使用 `hold_last`。使用 `zero` 作为消融实验。

## 10. 总时域与划分比例

使用一个总预测时域：

```text
N
```

其中 `N` 为预测窗口内的离散仿真步数。

使用划分比例：

```text
lambda in [0, 1]
```

其中 `lambda` 为 burst 到总时域的比例。

对固定候选 `lambda`，定义：

```text
N_b = max(1, round(lambda N))
N_c = N - N_b
```

其中：

1. `N_b` 为离散步意义下的 burst horizon。
2. `N_c` 为离散步意义下的 coast horizon。
3. `N = N_b + N_c`。

重要求解器说明：

不要把 rounding 操作放入 SQP。应在 SQP 外枚举候选比例。

使用有限的划分候选集合：

```text
Lambda = {0.10, 0.15, 0.20, 0.25, 0.33, 0.50}
```

第一版可使用更小集合：

```text
Lambda = {0.10, 0.20, 0.30, 0.40}
```

对每个 `lambda`，优化器看到的都是固定的平滑问题。

## 11. 单个划分候选的 MPC 问题

在决策时刻 `k`，测得状态为：

```text
x_tk
```

对一个候选划分比例 `lambda`，求解：

```text
J_star(x_tk, lambda) =
min over v_0, ..., v_{N_b - 1}
[
    sum_{i = 0}^{N_b - 1} ell_b(x_i, v_i, v_{i - 1})
  + sum_{j = 0}^{N_c - 1} ell_c(x_{N_b + j}, u_h)
  + ell_f(x_N)
]
```

其中：

1. `v_i` 为局部索引 `i` 处优化得到的 burst 输入。
2. `x_i` 为局部索引 `i` 处的预测状态。
3. `u_h` 为 coast 指令。
4. `ell_f` 为终端代价。

使用终端代价：

```text
ell_f(x_N) = q_f V_ep(x_N)
```

其中 `q_f > 0` 为终端代价权重。

约束如下：

```text
x_0 = x_tk
```

Burst 动力学：

```text
x_{i + 1} = f_d(x_i, v_i, dt)
for i = 0, ..., N_b - 1
```

Coast 指令：

```text
u_h = v_{N_b - 1}
```

或对 zero coast：

```text
u_h = 0
```

Coast 动力学：

```text
x_{N_b + j + 1} = f_d(x_{N_b + j}, u_h, dt)
for j = 0, ..., N_c - 1
```

输入约束：

```text
|v_i| <= u_max
for i = 0, ..., N_b - 1
```

状态约束：

```text
x_i in X
for i = 0, ..., N
```

随后选择划分比例：

```text
lambda_star = argmin over lambda in Lambda of J_star(x_tk, lambda)
```

施加：

```text
u_{t_k + i} = v_i_star
for i = 0, ..., N_b_star - 1
```

然后 coast：

```text
u_{t_k + i} = u_h_star
for i = N_b_star, ..., N - 1
```

除非 early wake 被触发。

## 12. SQP 可解性建议

第一版实现应使用 single shooting：

1. 决策变量仅为 burst 输入 `v_i`。
2. 预测状态由前向仿真生成。
3. 对二维摆杆状态而言，这可以保持较低的 NLP 维度。

仅当 single shooting 变得不稳定，或路径约束过紧时，再使用 multiple shooting。

保持优化问题对 SQP 友好：

1. 在求解器外枚举 `lambda`。
2. 在每个子问题内固定 `N_b` 与 `N_c`。
3. 使用平滑代价。
4. 避免在目标函数中使用绝对值。
5. 避免在连续优化器中使用 `max`、`min`、`if` 与 `round`。
6. 使用归一化变量。
7. 用前一解 warm start。
8. 启用新代价项时使用 continuation。

建议 continuation 顺序：

1. 设置 `rho_sat = 0`、`q_phi = 0`、`q_loc = 0`，仅调试能量代价。
2. 启用 `q_loc`。
3. 启用 `rho_sat`。
4. 启用 `q_phi`。
5. 加入 early wake。
6. 加入输入边界之外的其他约束。
7. 只有在确定性版本成功后，再加入噪声与模型失配。

## 13. 自然频率与总预测时域

使用下垂平衡点附近的被动小角度自然频率初始化总预测时域。

无阻尼自然频率为：

```text
omega_n = sqrt(m g l / I)
```

其中 `omega_n` 的单位为弧度每秒。

小角度自然周期为：

```text
T_0 = 2 pi / omega_n
```

若摆杆可视为距离 `l` 处的点质量，则：

```text
I = m l^2
omega_n = sqrt(g / l)
T_0 = 2 pi sqrt(l / g)
```

使用总预测时间：

```text
T_N = alpha_T T_0
```

其中 `alpha_T` 为时域倍数。

推荐初始取值：

```text
alpha_T in {0.25, 0.50, 0.75, 1.00}
```

从以下值开始：

```text
alpha_T = 0.50
```

然后：

```text
N = round(T_N / dt)
```

当 `l = 1 m` 且 `dt = 0.02 s` 时：

```text
omega_n = sqrt(9.81) = 3.13 rad/s
T_0 = 2.01 s
T_N = 1.00 s
N = 50
```

自然频率仅是初始尺度。大幅摆动不会保持小角度自然周期。靠近直立 separatrix 时，被动周期会显著变长。控制输入也会改变实际轨迹时机，尤其当反馈改变等效刚度或阻尼时。因此，自然周期应只用于初始化时域，而不应永久决定时域。

## 14. Early wake 触发器

coast 期间，将观测状态与预测状态进行比较。

令：

```text
x_hat_{t_k + i}
```

为执行期间的观测状态。

令：

```text
x_exec_star_{i | k}
```

为所选 burst 加 coast rollout 给出的预测执行状态。

定义预测偏差：

```text
d_i = (x_hat_{t_k + i} - x_exec_star_{i | k})^T S_x (x_hat_{t_k + i} - x_exec_star_{i | k})
```

其中 `S_x` 为正定偏差加权矩阵。

若满足以下条件，则触发 early wake：

```text
d_i > delta_x
```

其中 `delta_x` 为允许的预测偏差。

若任一状态约束裕度接近违反，也触发 early wake：

```text
h_x(x_hat_{t_k + i}) > - delta_h
```

其中：

1. `h_x(x) <= 0` 定义状态约束。
2. `delta_h > 0` 为安全裕度。

仅 MPC 阶段不要使用 Q degradation 作为触发条件，因为此时尚无学习型 Q critic。

## 15. 后续 CLBF 扩展

当前值函数候选为：

```text
V_ep(x)
```

后续可通过加入以下条件将其转化为控制 Lyapunov 条件：

```text
V_ep(f_d(x, u)) - V_ep(x) <= - alpha_V V_ep(x) + s_V
```

其中：

1. `alpha_V` 为正的下降率。
2. `s_V >= 0` 为松弛变量。

对安全性，定义 barrier function：

```text
h(x) >= 0
```

其中 `h(x)` 在安全集内部为正。

使用离散 barrier 条件：

```text
h(f_d(x, u)) >= (1 - alpha_h) h(x) - s_h
```

其中：

1. `alpha_h` 为 barrier 速率。
2. `s_h >= 0` 为松弛变量。

不要在第一版工作实现中加入 CLBF 约束。当前实现只需让代价与特征设计兼容未来扩展。

## 16. 建议代码结构

当前仓库包含规划文档。建议加入如下源码结构：

```text
wake_sleep_mpc/
    config/
        pendulum_default.yaml
    src/
        wake_sleep_mpc/
            __init__.py
            dynamics/
                pendulum.py
                integrators.py
            costs/
                energy_phase.py
            controllers/
                split_ratio_mpc.py
            simulation/
                rollout.py
                experiment_runner.py
            logging/
                records.py
                metrics.py
    tests/
        test_energy_consistency.py
        test_rk4_pendulum.py
        test_cost_zero_near_goal.py
        test_split_ratio_dimensions.py
        test_mpc_single_step.py
    docs/
        Plan.md
        wake_sleep_mpc_dev_spec.md
        handoff_split_ratio_mpc.md
```

最小模块职责如下：

1. `pendulum.py`  
   实现连续动力学、输入约束、能量与自然频率。

2. `integrators.py`  
   实现 RK4。

3. `energy_phase.py`  
   实现 `V_ep`、`ell_b`、`ell_c` 与 `ell_f`。

4. `split_ratio_mpc.py`  
   实现划分比例枚举与 SQP 调用。

5. `rollout.py`  
   执行所选 burst 与 coast 序列。

6. `experiment_runner.py`  
   运行确定性 episode。

7. `records.py`  
   保存状态、输入、能量、所选划分比例、求解器状态与 early wake 事件。

8. `metrics.py`  
   计算成功率、代价、约束违反、控制稀疏性与计算时间。

## 17. 调参前的最低测试

在尝试调参前，先实现这些测试。

### 测试 1：能量一致性

当 `u = 0` 且 `b = 0` 时，短时仿真中的能量应近似保持常数。

期望：

```text
max |E(t) - E(0)| < tolerance
```

若该测试失败，检查重力项符号。

### 测试 2：输入边界

MPC 结果必须满足：

```text
|u_i| <= u_max
```

对所有 burst 输入均成立。

### 测试 3：目标附近代价

在：

```text
theta = pi
omega = 0
```

处，`V_ep` 应接近零。

### 测试 4：下垂静止代价

在：

```text
theta = 0
omega = 0
```

处，`V_ep` 应为正。

### 测试 5：划分比例维度

对每个 `lambda`，验证：

```text
N_b >= 1
N_c >= 0
N_b + N_c = N
```

### 测试 6：确定性重放

给定相同初始状态与配置，仿真器和 MPC 应产生可复现轨迹。

## 18. 初始配置建议

从以下配置开始：

```text
dt: 0.02
g: 9.81
m: 1.0
l: 1.0
I: 1.0
b: 0.05
u_max: 0.7 * m * g * l

alpha_T: 0.5
split_ratios: [0.10, 0.20, 0.30, 0.40]

sigma_E: 0.5
eps_phi: 1.0e-6

q_E: 1.0
q_phi: 0.0
q_loc: 0.0
q_f: 5.0

rho_sat: 0.0
rho_du: 1.0e-3
rho_c: 1.0e-3

coast_mode: hold_last
solver: SQP
shooting: single
```

能量泵入工作后，改为：

```text
q_loc: 1.0
rho_sat: 0.05
q_phi: 0.1
```

然后重新调参。

## 19. 实现里程碑

### 里程碑 1：动力学与能量

1. 按修正后的符号约定实现摆杆动力学。
2. 实现 RK4。
3. 实现能量函数。
4. 通过能量一致性测试。

### 里程碑 2：代价函数

1. 实现 `V_ep`。
2. 实现 burst、coast 与终端代价。
3. 通过目标点与下垂静止代价测试。

### 里程碑 3：固定划分 MPC

1. 设置一个固定划分比例。
2. 用 SQP 求解 burst 序列。
3. rollout coast 段。
4. 验证输入约束。

### 里程碑 4：划分比例枚举

1. 加入 `Lambda`。
2. 对每个划分候选求解一个子问题。
3. 选择代价最低的候选。
4. 记录随时间变化的所选划分比例。

### 里程碑 5：闭环 wake sleep 执行

1. 施加 burst 指令。
2. 施加 coast 指令。
3. 只在下一个计划 wake 时刻重新求解。
4. 加入 early wake 触发器。

### 里程碑 6：基准指标

记录：

1. 最终成功或失败。
2. 进入目标集的时间。
3. 保持在目标集内的时间。
4. 总代价。
5. 主动控制步比例。
6. 饱和控制步比例。
7. MPC 调用次数。
8. 平均与最大求解时间。
9. 约束违反次数。
10. early wake 次数。

## 20. 暂缓特性

在确定性仅 MPC 版本工作前，不要实现以下内容：

1. 学习型 Q critic。
2. 神经网络终端值。
3. 分段线性预测。
4. 过程噪声。
5. 模型失配。
6. 鲁棒 tube MPC。
7. CLBF 约束。
8. 航天器基准。
9. 多节点通信。
10. 实时并行 MPC。

## 21. 最终实现原则

控制器应写成终端代价接口后续可替换的形式。

当前仅 MPC 终端评分：

```text
terminal_score = ell_f(x_N)
```

未来学习型 Q 终端评分：

```text
terminal_score = alpha_Q Q_theta(x_{N_b}, u_h, N_c)
```

因此，应将终端评分实现为可互换组件：

```text
TerminalScorer.evaluate(...)
```

这能保持当前确定性基准与未来 wake sleep RL 扩展兼容。
