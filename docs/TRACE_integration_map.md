# TRACE-Explorer 四个创新点 → MARVEL 代码挂载点

下面每一行的 `文件:行号` 都在 `upstream/MARVEL/` 里核对过（commit `318c2a6`）。
结论：四个创新点都有现成的挂载位置，不需要重写框架。

## 为什么是 MARVEL

MARVEL（ICRA 2025，新加坡国立 MARMot Lab）与 TRACE-Explorer 的骨架同构：

| TRACE 设想 | MARVEL 现状 |
|---|---|
| 多机去中心化策略 | 6 架 agent，各自前向，权重共享 |
| 任务图 + 注意力 | 已知自由空间稀疏图 + Transformer encoder/decoder |
| 长时域任务价值（非贪心 frontier） | QNet 对候选节点出 Q 值，SAC 学折扣回报 |
| CTDE 中心化 critic | QNet 吃 `all_agent_indices`，对全队动作做 cross-attention |
| 去冗余 | `calculate_overlap_reward` 算独有感知占比 |
| 受限感知 | FoV 120°/垂直 60°、带 yaw 速率的无人机运动学 |

它本身就是"把探索任务抽象成图 + 图注意力 + MARL"这条路线的开源实现，差的正好是 TRACE 要加的四件事。

## 创新点一：Dynamic Topology–Interaction Dual Graph

**任务节点特征 `x_m^t`** — `utils/agent.py:202`
```python
node_inputs = np.concatenate((all_node_coords, node_utility, node_guidepost,
                              node_occupancy, node_highest_utility_angles), axis=1)
```
现有 6 维 = 相对坐标(2) + utility + guidepost + occupancy + 最优视角。
TRACE 要加的 `Risk_m^t / Comm_m^t / Age_m / IG_m` 就拼在这里，同时把
`parameter.py:NODE_INPUT_DIM` 从 6 改成新维度（`test_parameter.py` 里也有一份）。
注意上游每个特征都做了归一化（`agent.py:199-201`），新特征照着做。

**拓扑邻接 `ρ_ij`** — `utils/node_manager.py:338` `update_neighbor_nodes()`，
邻接矩阵在 `get_all_node_graph()`（`node_manager.py:72`）里产出，经
`agent.py:180` 传给网络成为 `edge_mask`。Adaptive Contiguity 的 hop 距离从这里取。

**Local Interaction Graph `G_i^I`** — 这是唯一需要新建的数据结构。
队友位置已经全量可见：`agent.py:178 update_planning_state(robot_locations)`，
`utils/multi_agent_worker.py` 每步把 `robot_locations / headings` 发给所有 agent。
UAV–UAV 边（`Δp, Δv, Overlap, MsgAge, CommQuality`）在这里构造；
UAV–Obstacle 边需要先有动态障碍（见下方"缺口"）。

## 创新点二：Adaptive Contiguity

**注意力 logits** — `utils/model.py:39-44`
```python
U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
U = self.tanh_clipping * torch.tanh(U)
if mask is not None:
    U = U.masked_fill(mask == 1, -1e8)
attention = torch.log_softmax(U, dim=-1)
```
`U` 就是论文里的 `e_im^learn`。TRACE 的
`e_im = e_im^learn + e_im^topo + e_im^risk` 只需在 `tanh` 之后、`masked_fill` 之前
加两个偏置项（`-β_c[(ρ-1)_+]^2` 和 `-β_r·Risk`）。
`SingleHeadAttention` 的签名要多收一个 bias 张量，调用点在
`model.py:198 PolicyNet` 和 `model.py:286 QNet` 的 decoder 里。

## 创新点三：Foresighted Task-Value Learning

**已经有一半** — `utils/model.py:346 QNet.output_q()` 对当前节点的邻居候选输出 Q 值，
`driver.py:172-300` 用 SAC 的 TD bootstrap 训练（`GAMMA=1`，`parameter.py:73`）。
这已经是"预测未来探索回报"而不是即时 information gain。

TRACE 要改的是**预测目标和时间尺度**：
- 目标从"下一个路点"换成"task unit 的 `Q_{i,m}^exp`"；
- 把 `GAMMA=1` 改成 `<1` 的有限时域 `H`；
- 奖励项 `utils/multi_agent_worker.py:166-193`（`utility_reward + trajectory_reward + team_reward`）
  里补 `r_explore` 的定义。

## 创新点四：Counterfactual Marginal Exploration Utility

**掩码机制已经写好了** — `utils/model.py:377-378`
```python
agent_mask = all_agent_indices == current_index
global_state_action_feature, _ = self.agent_decoder(current_state_feature,
                                                   all_agent_action_features, agent_mask)
```
中心化 critic 已经在对"全队动作特征"做 cross-attention，并且已经有按 agent 掩码的能力。
`u_i^t = V(s_t) − V(M_i(s_t))` 的实现方式：同一批数据跑两次 critic，第二次把
agent `i` 的 token 连同它独有的地图证据一起 mask 掉，差值即 `u_i`。

**特权全局状态** — `utils/ground_truth_node_manager.py`（432 行）已经提供真值地图侧的
节点管理，`parameter.py:TRAIN_ALGO=3` 就是 "MAAC + Ground Truth" 模式。
`u_i` 里"删掉 UAV i 独有的地图证据"可以借这套真值信息算。

**手工版的前身** — `utils/agent.py:374 calculate_overlap_reward()` 已经在算
"本机独有感知格数 / 本机总感知格数"的平方。TRACE 的 `u_i` 是它的学习化替代，
可以直接拿它做 ablation 对照。

## 当前缺口（MARVEL 没有、TRACE 需要的）

按工作量从小到大：

1. **通信约束**：MARVEL 是全通信——`utils/env.py` 只有一份共享 belief，
   队友位置每步全可见。→ 用 `upstream/IR2/` 补：它有信号强度模型
   （`ss_realistic_model.py`）和稀疏间歇连通下的多机探索。
2. **动态障碍**：`utils/env.py` 里 grep 不到任何 dynamic/moving/velocity 障碍，
   地图是静态 PNG 占据栅格。`Risk_m^t`、`TTC_ik`、UAV–Obstacle 边都依赖这个，
   需要自己往 env 里加动态物体。
3. **连通分量级任务抽象**：MARVEL 的节点是 `NODE_RESOLUTION=4.0m` 的均匀稀疏图，
   不是 C²-Explorer 的 Connected Component Labeling 任务单元。
   TRACE 的 task unit 要在 `node_manager.py` 上做一层聚合。
4. **3D**：MARVEL 是 2D 占据栅格上的无人机（固定高度 `DRONE_HEIGHT=2`，
   FoV 带俯仰），不是真 3D 体素探索。先在 2D 验证四个创新点，再决定是否上 3D。
5. **Safety Shield（CBF/HOCBF）**：没有。属于部署层，验证阶段可以最后做。
