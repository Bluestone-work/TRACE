# 2025–2026 多无人机搜索、追逃、围捕与视觉跟踪调研

检索日期：2026-11-04。通过 arXiv API 检索候选论文，并核对下列论文的 arXiv HTML 正文。
年份区分首次预印本与正式发表；表中 OPEN 是 2025 年 RA-L 工作，预印本首发于 2024 年。
本记录不是完整系统综述，也不将预印本等同于已通过同行评审的工作。

## 核心结论

1. 多机围捕并不普遍使用 RGB。很多方法首先解决“给定目标状态，如何协作”，感知输入是位置、速度、相对方位或经过可见性掩码的状态向量。
2. 有限 FOV、遮挡和 camera 字样，不足以说明方法运行了真实图像检测器。必须检查目标状态来源。
3. 部分系统使用 LiDAR，另一些使用 RGB 检测器加状态估计。即使策略网络输入图像，也可能是仿真直接生成的深度图和语义分割掩码。
4. 需要区分跟踪、距离阈值意义上的捕获，以及真实接触/载网捕获。目标在视野中或被某一追踪者靠近，不自动意味着已形成包围或完成物理捕获。

## 重点论文

### 1. OPEN：Online Planning for Multi-UAV Pursuit-Evasion in Unknown Environments Using Deep Reinforcement Learning

- 年份：RA-L 2025；arXiv 首发 2024。
- 论文：https://arxiv.org/abs/2409.15866
- 正文：https://arxiv.org/html/2409.15866
- 项目：https://sites.google.com/view/pursuit-evasion-rl
- 方法：MAPPO；LSTM 逃逸者轨迹预测；多头注意力编码自身、队友、障碍物信息；自适应环境生成；两阶段奖励优化以改善动作平滑性。
- 观测：自身姿态和速度、真实/预测的目标相对位置、队友相对位置、最近障碍物相对位置；未被任何机发现的目标位置用固定值掩码。
- 动作：总推力和三轴机体系角速度 CTBR，之后由低层 PID 转成电机输入。
- 实验范围：V-F 明确写出三架真实 Crazyflie 2.1，追踪机由动捕提供状态；逃逸者是虚拟目标，在被判定可见时将其真实状态输入网络。
- 判断：是非常相关的多机策略与动力学基线，不能当成完成 RGB 非合作目标检测的实机系统。
- 对本项目：可借鉴目标预测、注意力、课程生成、策略平滑；不必为了借鉴协同机制立刻放弃当前速度动作接口。

核对位置：IV-A、IV-B、IV-C、IV-D、V-F。

### 2. Decentralized End-to-End Multi-AAV Pursuit Using Predictive Spatio-Temporal Observation via Deep Reinforcement Learning（PSTO）

- 年份：2026。
- 论文：https://arxiv.org/abs/2603.24238
- 正文：https://arxiv.org/html/2603.24238
- 项目：https://hitsz-mas.github.io/psto-aav-pursuit/
- 方法：将 LiDAR 障碍距离图与“目标未来轨迹＋队友未来轨迹”的意图热力图投影到同一个机体参考球面网格，双分支 CNN 提取特征，MAPPO 训练，共享 actor 分散执行。
- 目标预测：LSTM；不可见时使用先前预测维持连续性。队友通过通信共享运动状态，短期轨迹按速度外推。
- 动作：平面追逃中的二维高层机体系速度命令参数。
- 协作：共享奖励包含接近目标、拉开包围角、障碍安全、时间和终止奖励；课程从 2 对 1 逐步扩展至 4 对 1。
- 实验范围：有户外 2 对 1 实机；IV-E 明确说相机仅用于展示。环境几何来自 LiDAR，自身定位来自 Swarm-LIO；可见目标的相对状态来自共享 Swarm-LIO 定位结果，作为目标检测的替代。并将半径 9 m 的虚拟圆形边界加入 LiDAR 点云。
- 判断：真实 LiDAR 几何与学习策略结合，但目标检测链路有替代输入，不能称为真实 RGB 围捕。
- 对本项目：与 NavRL 的距离图、CNN、速度动作接口接近。可保留其“空间对齐＋目标预测＋队友信息”的思路，把目标状态来源换成我们的 RGB 估计。

核对位置：III-A、III-B、III-C、IV-E。尤其不能只根据摘要的“raw LiDAR end-to-end”推断目标检测全部由 LiDAR 完成。

### 3. Visibility-Aware Cooperative Tracking with Decentralized LiDAR-Based Aerial Swarms（Swarm-Tracker）

- 年份：arXiv 2025。
- 论文：https://arxiv.org/abs/2512.01280
- 正文：https://arxiv.org/html/2512.01280
- 代码：https://github.com/hku-mars/Swarm-Tracker
- 方法：LiDAR 目标观测与协同估计；SSDF 表达三维遮挡；FOV 对齐代价；防机间遮挡与安全距离；受静电势启发的三维分布代价；运动学搜索前端和时空 SE(3) 优化后端。
- 控制：轨迹规划加 on-manifold MPC，并非强化学习策略。
- 实验范围：四机在森林中跟踪目标无人机，另有人体跟踪等实验；使用 Mid360/Avia 等异构 LiDAR、Swarm-LIO、自主机载计算和 UDP 通信。
- 感知假设：IX-A 明确写出目标贴有高反射标记，目标观测通过 UDP 在队友间共享。目标本身不交换状态。
- 判断：是真实多机协同跟踪系统，但主感知是 LiDAR，且实验使用反光标记，不能等同于无标记 RGB 识别。
- 对本项目：最值得借鉴“至少一机可见”“避免队友互相遮挡”“三维观测方向分散”的目标；可以作为不依赖学习的强规划基线。

核对位置：III、IV–VII、IX-A、IX-B。

### 4. Dynamics and Control of Vision-Aided Multi-UAV-tethered Netted System Capturing Non-Cooperative Target

- 年份：2025。
- 论文：https://arxiv.org/abs/2506.03297
- 正文：https://arxiv.org/html/2506.03297
- 方法：mySim 多体动力学；弹簧阻尼绳网；碰撞模型；多旋翼控制；MAPPO 协同捕获。
- 视觉链路：固定相机和双轴云台相机；VINS-MONO 结合视觉与 IMU 估计自身位姿；DETR + ResNet-101 检测目标；通过检测框和多机信息估计目标状态；状态估计再输入协同决策。
- 判断：确实涉及渲染相机图像、目标检测器和控制闭环；结构是“感知→估计→决策→控制”，不是 RGB 像素直接输出四电机。
- 实验范围：论文主要以仿真验证多体动力学、视觉估计及载网捕获，没有据此确认实机完成完整捕获闭环。
- 对本项目：很接近当前 RGB 色标检测→状态估计→跟踪控制的模块化结构。后续可替换检测器与策略，而无需同时重做物理和感知。

核对位置：2.1、2.2、2.3、3、算法 2–4。文中基于检测框面积的距离近似是该文的建模处理，不应直接照搬为一般单目测距定律。

### 5. Intercepting an Agile Target with Net-Carrying Drones using Competitive Multi-Agent Reinforcement Learning

- 年份：2026。
- 论文：https://arxiv.org/abs/2607.05939
- 正文：https://arxiv.org/html/2607.05939
- 方法：追逐团队和逃逸者双方训练，MAPPO + Prioritized Fictitious Self-Play（PFSP）；对手来自历史策略池，降低只适应当前对手及灾难性遗忘的问题。
- 观测：自身速度、姿态矩阵、其他飞行器相对位置与相对速度，以及环境边界距离。
- 动作：CTBR；高保真旋翼/低层控制仿真；JAX 并行训练。
- 协作行为：三名追逐者分散包夹、封堵边界、错开捕获尝试；成功条件包含载网捕获和碰撞约束。
- 判断：不是 RGB 感知方法。正文将高速逃逸者状态估计列为范围之外；结果主要是仿真。
- 对本项目：当 A 需要从随机导航升级为真正对抗逃逸时，历史对手池和自博弈值得借鉴。不能只让 B 对固定的 NavRL 路径分布过拟合。

核对位置：III、IV 的观测与动作、IV-E、V。

### 6. Memory-State Critic for Asymmetric Actor-Critic with Application to Vision-Based Pursuit-Evasion

- 年份：2026。
- 论文：https://arxiv.org/abs/2610.03830
- 正文：https://arxiv.org/html/2610.03830
- 代码：https://github.com/LouetteArthur/MemoryStateCritic
- 平台：IsaacLab + skrl；双 Crazyflie，有开放场地和带缺口遮挡墙的场地。
- 任务：1 个学习追逐者对 1 个从固定启发式策略池采样的逃逸者。不是多追逐者围捕。
- actor 输入：64×64 两通道图像——深度图和二值目标分割掩码——加上次动作；CNN + GRU 形成记忆。
- critic 输入：训练时的特权状态以及 actor 记忆，记忆进入 critic 的路径停止梯度；部署不需要 critic。
- 动作：25 Hz CTBR，500 Hz PD 内环；推进旋翼动力学。
- 判断：是图像观测的追逃 RL，但分割掩码由仿真提供，不包含真实 RGB 检测器的误检漏检；实机部署属于后续工作。
- 对本项目：非常适合参考 B 从 SEARCH/TRACK/COAST 规则切换到带记忆的学习策略。需要另外解决 RGB 检测、分割或图像特征学习。

核对位置：4.1、4.2、7、Appendix A。

### 7. AT-Drone: Benchmarking Adaptive Teaming in Multi-Drone Pursuit

- 年份：2025。
- 论文：https://arxiv.org/abs/2502.09762
- 项目：https://sites.google.com/view/at-drone
- 重点：与未事先共同训练的陌生队友协作；提供不同追逐者/逃逸者/障碍配置、算法库与评估协议。
- 观测：可感知范围内目标的相对位置和方位、最近障碍信息、队友的相对位置和方位；不可见实体掩码。
- 动作：连续角向转向调整，不是完整四旋翼电机动作。
- 实机：Crazyflie + FZMotion 动捕 + Crazyswarm，策略在边缘设备运行；状态来自动捕。
- 判断：适合团队协作、可扩展性与陌生队友评估，不能据此推断完成了无人机机载 RGB 识别。

核对位置：环境与 Deployment 部分。

### 8. AgilePE: Autonomous UAV Pursuit-Evasion via Self-Play Reinforcement Learning

- 年份：2026。
- 论文：https://arxiv.org/abs/2608.14135
- 正文：https://arxiv.org/html/2608.14135
- 方法：PFSP、自博弈、历史对手池、CTBR 和硬件响应/通信延迟建模。
- 观测：自身状态和过去动作；对手状态时间窗口按圆锥视场和遮挡规则掩码。
- 实机：1 对 1，动捕提供 120 Hz 位姿，机载 Jetson 输出 CTBR；正文结尾明确将视觉感知和多机扩展列为后续工作。
- 判断：“存在视场限制”与“采用图像输入”是不同事情，这篇是很清楚的例子。

核对位置：观测定义、Real-World Deployment、Limitations。

## 其他相关工作

- [Cooperative Search and Track of Rogue Drones using Multiagent Reinforcement Learning（2025）](https://arxiv.org/abs/2501.10413)：独立 Q-learning 和 difference reward；根据四个区域内目标/队友数量与距离决策。探测按距离半径定义，二维离散移动，主要为仿真。适合研究“搜索覆盖与追踪的团队奖励”，不是原始视觉输入工作。
- [Evader-Agnostic Team-Based Pursuit Strategies in Partially-Observable Environments（2025）](https://arxiv.org/abs/2511.05812)：高空观察者与低空拦截者协作；level-k 策略训练，再在线识别对手类型并选择响应策略。论文描述相机与 FOV，但学习实验建立在网格化部分可观测仿真上，不能直接视为完整图像检测系统。

## 对当前 NavRL 双机项目的建议

当前版本实际是：A 使用固定 NavRL 导航策略执行随机目标；B 使用有已知尺寸色标的 RGB 检测、运动滤波与传统跟踪控制。它是后续学习的功能基线，不等同于自主无标记围捕。

建议保留现有速度动作、Lee 控制器和旋翼动力学，分阶段升级：

1. 视觉与记忆：将色标检测替换成无人机检测器或学习图像特征；显式输出置信度、观测时间和缺失掩码；使用 GRU 或目标轨迹历史处理遮挡。
2. 多机与信息共享：先 2 追 1，再 3 追 1；共享带时间戳的目标估计及不确定性、队友自身状态，控制器不读取 A 真值。
3. 协同训练：MAPPO 的集中训练、分散执行可作起点。critic 在仿真训练时可读取真实状态；actor 必须只使用部署时真实可得的感知与通信信息。
4. 协作目标：除了接近目标，还要奖励团队可见性、分散观测角度、重发现和避免相互遮挡；把“持续跟踪”“形成包围”“进入捕获半径”作为不同评价指标。
5. 对手多样性：先保留 NavRL 作为一种目标行为，再加入多种启发式及历史学习对手，最后考虑 PFSP；避免只在固定目标策略上取得成功。
6. 同速条件：两队使用相同速度、加速度/动力学能力上限。分别报告有界场地与开放场地，不能把依赖边界、障碍和初始包围的成功解释成开放空间必然可追上目标。

优先阅读顺序：PSTO（与 NavRL 接口接近）→ Memory-State Critic（视觉与遮挡记忆）→ Swarm-Tracker（多机可见性和实机工程）→ OPEN（协作与课程学习）→ 载网自博弈论文（真正对抗与低层敏捷控制）。
