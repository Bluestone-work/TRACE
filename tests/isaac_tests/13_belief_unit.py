"""MultiAgentBelief 单测：纯 torch，不需要 Isaac Sim。

    python isaac_tests/13_belief_unit.py [--device cpu]
"""
import argparse, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trace.mapping.belief import CommCfg, MultiAgentBelief
from trace.mapping.occupancy_map import VoxelMapCfg

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
dev = a.device
ok = []


def check(name, cond, extra=""):
    ok.append(bool(cond))
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}", flush=True)


def fake_rays(E, A, R, centers, dev, radius=3.0):
    """构造一组从 centers 出发、朝水平均匀方向、全部打空的光线。

    打空（hit=inf）+ 给 ray_dirs，会把 max_range 内清成 free，
    于是每机在自己周围清出一个圆盘 —— 刚好用来造"各机看到不同区域"。
    """
    ang = torch.linspace(0, 6.28318, R, device=dev)
    dirs = torch.stack([ang.cos(), ang.sin(), torch.zeros_like(ang)], dim=-1)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    org = torch.as_tensor(centers, dtype=torch.float32, device=dev).view(E, A, 3)
    hits = torch.full((E, A, R, 3), float("inf"), device=dev)
    return org, hits, dirs.view(1, 1, R, 3).expand(E, A, R, 3).contiguous()


print("=== 1. 各机 belief 相互独立 ===", flush=True)
E, A, R = 2, 3, 64
mcfg = VoxelMapCfg(size=(20.0, 20.0, 4.0), origin=(-10.0, -10.0, 0.0),
                   resolution=0.5, max_range=4.0)
b = MultiAgentBelief(E, A, mcfg, CommCfg(enabled=False), device=dev)
centers = [[[-6, -6, 1.0], [0, 0, 1.0], [6, 6, 1.0]],
           [[-6, 6, 1.0], [0, 0, 1.0], [6, -6, 1.0]]]
org, hits, dirs = fake_rays(E, A, R, centers, dev)
b.update_from_rays(org, hits, dirs)

kpa = b.known_per_agent()
check("每机都观测到体素", bool((kpa > 0).all()), f"known_per_agent={kpa.tolist()}")
own_k = b.own.known_mask.view(E, A, -1)
check("agent0 与 agent2 的已知集合不同",
      not bool(torch.equal(own_k[0, 0], own_k[0, 2])))
team_cnt = b.team_known_count()
check("团队覆盖 >= 任一单机覆盖", bool((team_cnt >= kpa.max(dim=1).values).all()),
      f"team={team_cnt.tolist()}")
check("团队覆盖 <= 各机覆盖之和", bool((team_cnt <= kpa.sum(dim=1) + 1e-6).all()))

print("=== 2. 边际覆盖贡献 ===", flush=True)
marg = b.marginal_coverage()
check("边际贡献非负", bool((marg >= 0).all()), f"marginal={marg[0].tolist()}")
check("边际贡献 <= 自身覆盖", bool((marg <= kpa + 1e-6).all()))
for i in range(A):
    without = b.coverage_without(i)
    lhs = team_cnt - without
    check(f"marginal[{i}] == team - team_without_{i}",
          bool(torch.allclose(lhs, marg[:, i], atol=1e-4)),
          f"{lhs[0].item():.0f} vs {marg[0, i].item():.0f}")

print("=== 3. 单机时边际 == 团队覆盖 ===", flush=True)
b1 = MultiAgentBelief(1, 1, mcfg, CommCfg(enabled=False), device=dev)
o1, h1, d1 = fake_rays(1, 1, R, [[[0, 0, 1.0]]], dev)
b1.update_from_rays(o1, h1, d1)
check("A=1 时 marginal == team_known",
      bool(torch.allclose(b1.marginal_coverage()[:, 0], b1.team_known_count(), atol=1e-4)),
      f"{b1.marginal_coverage()[0,0].item():.0f} vs {b1.team_known_count()[0].item():.0f}")

print("=== 4. 通信关闭时 recv 恒为空、fused == own ===", flush=True)
check("recv 证据全为 0", bool((b.recv_ev == 0).all()))
fused = b.fused()
check("fused 的已知集合 == own 的已知集合",
      bool(torch.equal(fused.known_mask, b.own.known_mask)))
check("msg_age 拉满", bool((b.msg_age == b.comm.max_age).all()),
      f"age={b.msg_age[0,0].tolist()}")
check("peer_valid 全 False", not bool(b.peer_valid.any()))

print("=== 5. 通信开启后证据会流动 ===", flush=True)
b2 = MultiAgentBelief(E, A, mcfg,
                      CommCfg(enabled=True, range_m=100.0, every=1, drop_prob=0.0, delay_steps=2),
                      device=dev)
b2.update_from_rays(org, hits, dirs)
own_before = b2.own.known_mask.view(E, A, -1).sum(dim=2).clone()
pos = org.clone()
vel = torch.zeros_like(pos)
for _ in range(3):
    b2.step_comm(pos, vel)
fused2 = b2.fused()
fk = fused2.known_mask.view(E, A, -1).sum(dim=2)
check("fused 覆盖 > 自身 own 覆盖（收到了他机地图）",
      bool((fk > own_before).all()), f"own={own_before[0].tolist()} fused={fk[0].tolist()}")
check("own 本身没被通信污染（来源记录完整）",
      bool(torch.equal(b2.own.known_mask.view(E, A, -1).sum(dim=2), own_before)))
check("fused 覆盖 == 团队覆盖（全连通且无丢包）",
      bool((fk[:, 0] == b2.team_known_count()).all()),
      f"fused0={fk[0,0].item()} team={b2.team_known_count()[0].item()}")
check("peer_valid 已置位", bool(b2.peer_valid[:, 0, 1].all()))
check("收到时 msg_age == delay_steps", bool((b2.msg_age[:, 0, 1] == 2).all()),
      f"age={b2.msg_age[0,0,1].item()}")

print("=== 6. 超出通信距离则断链 ===", flush=True)
b3 = MultiAgentBelief(1, 2, mcfg, CommCfg(enabled=True, range_m=3.0, every=1, drop_prob=0.0),
                      device=dev)
far = torch.tensor([[[-8.0, 0.0, 1.0], [8.0, 0.0, 1.0]]], device=dev)
o3, h3, d3 = fake_rays(1, 2, R, far.tolist(), dev)
b3.update_from_rays(o3, h3, d3)
deliv = b3.step_comm(far, torch.zeros_like(far))
check("距离 16m > 通信半径 3m，无投递", not bool(deliv.any()))
near = torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]], device=dev)
deliv2 = b3.step_comm(near, torch.zeros_like(near))
check("靠近后有投递", bool(deliv2.any()), f"delivered={int(deliv2.sum())}")

print("=== 7. max 融合的幂等性（重传不涨置信度）===", flush=True)
ev_once = b3.recv_ev.clone()
for _ in range(5):
    b3.step_comm(near, torch.zeros_like(near))
check("重复投递同一张图后 recv 证据不变",
      bool(torch.equal(ev_once, b3.recv_ev)))

print("=== 8. reset 只清指定 env ===", flush=True)
b4 = MultiAgentBelief(2, 2, mcfg, CommCfg(enabled=True, range_m=100.0, every=1, drop_prob=0.0),
                      device=dev)
o4, h4, d4 = fake_rays(2, 2, R, [[[-3, 0, 1.], [3, 0, 1.]], [[-3, 0, 1.], [3, 0, 1.]]], dev)
b4.update_from_rays(o4, h4, d4)
b4.step_comm(o4, torch.zeros_like(o4))
k_before = b4.known_per_agent().clone()
b4.reset(torch.tensor([0], device=dev))
k_after = b4.known_per_agent()
check("env0 已清零", bool((k_after[0] == 0).all()), f"{k_after[0].tolist()}")
check("env1 未受影响", bool(torch.equal(k_after[1], k_before[1])), f"{k_after[1].tolist()}")
check("env0 的 recv 已清零", bool((b4.recv_ev.view(2, 2, -1)[0] == 0).all()))
check("env1 的 recv 保留", bool((b4.recv_ev.view(2, 2, -1)[1] > 0).any()))
check("env0 msg_age 复位", bool((b4.msg_age[0] == b4.comm.max_age).all()))

print("=== 9. stepwise_marginal 守恒且不重复奖励历史 ===", flush=True)
b5 = MultiAgentBelief(2, 3, mcfg, CommCfg(enabled=False), device=dev)
prev_team = b5.team_known_count().clone()
b5.update_from_rays(org, hits, dirs)
team_after = b5.team_known_count()
sm = b5.stepwise_marginal(commit=True)
check("Σ_i stepwise_marginal == Δ|team_known|",
      bool(torch.allclose(sm.sum(dim=1), team_after - prev_team, atol=1e-3)),
      f"Σ={sm.sum(dim=1).tolist()} Δteam={(team_after - prev_team).tolist()}")
check("stepwise_marginal 非负", bool((sm >= 0).all()), f"{sm[0].tolist()}")
sm2 = b5.stepwise_marginal(commit=True)
check("没有新观测时边际为 0（历史覆盖不被反复奖励）",
      bool((sm2 == 0).all()), f"第二次调用={sm2[0].tolist()}")

print("=== 10. 同一步被多机同时发现则均分 ===", flush=True)
b6 = MultiAgentBelief(1, 2, mcfg, CommCfg(enabled=False), device=dev)
same = [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]            # 两机完全重合
o6, h6, d6 = fake_rays(1, 2, R, same, dev)
b6.update_from_rays(o6, h6, d6)
sm6 = b6.stepwise_marginal(commit=True)
check("两机完全重合时各得一半",
      bool(torch.allclose(sm6[:, 0], sm6[:, 1], atol=1e-3))
      and bool(torch.allclose(sm6.sum(dim=1), b6.team_known_count(), atol=1e-3)),
      f"marginal={sm6[0].tolist()} team={b6.team_known_count()[0].item():.0f}")
check("完全重合时累计独占覆盖为 0（与边际是两个口径）",
      bool((b6.exclusive_coverage() == 0).all()),
      f"exclusive={b6.exclusive_coverage()[0].tolist()}")

print("=== 11. 过期障碍能被清除（覆盖式投递 + 一手优先）===", flush=True)
# 造一面墙：agent1 的光线打到 x=+2 的实体上
b7 = MultiAgentBelief(1, 2, mcfg,
                      CommCfg(enabled=True, range_m=100.0, every=1, drop_prob=0.0, delay_steps=0),
                      device=dev)
pos7 = torch.tensor([[[-6.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], device=dev)
Rw = 8
org7 = pos7
dirs7 = torch.zeros((1, 2, Rw, 3), device=dev); dirs7[..., 0] = 1.0
hits7 = torch.full((1, 2, Rw, 3), float("inf"), device=dev)
hits7[0, 1, :, 0] = 2.0; hits7[0, 1, :, 1] = 0.0; hits7[0, 1, :, 2] = 1.0   # agent1 打到 x=2
for _ in range(6):                                  # 多打几次，越过 p_occ
    b7.update_from_rays(org7, hits7, dirs7)
idx, _ = b7.own.world_to_index(torch.tensor([[2.0, 0.0, 1.0]], device=dev))
ix, iy, iz = (int(idx[0, 0]), int(idx[0, 1]), int(idx[0, 2]))
check("agent1 自己图上该体素是 occupied",
      bool(b7.own.occupied_mask[1, ix, iy, iz]),
      f"log_odds={float(b7.own.log_odds[1, ix, iy, iz]):.3f}")
b7.step_comm(pos7, torch.zeros_like(pos7))
f7 = b7.fused()
check("agent0 通过通信也看到该障碍", bool(f7.occupied_mask[0, ix, iy, iz]))
# 障碍消失：agent1 改为打空（同方向射穿）
hits7[0, 1, :, :] = float("inf")
for _ in range(10):
    b7.update_from_rays(org7, hits7, dirs7)
check("agent1 自己图上障碍已清除（不再 occupied）",
      not bool(b7.own.occupied_mask[1, ix, iy, iz]),
      f"log_odds={float(b7.own.log_odds[1, ix, iy, iz]):.3f}")
b7.step_comm(pos7, torch.zeros_like(pos7))
f7b = b7.fused()
check("agent0 收到新报告后也清除了过期障碍（max 融合下会残留）",
      not bool(f7b.occupied_mask[0, ix, iy, iz]),
      f"fused log_odds={float(f7b.log_odds[0, ix, iy, iz]):.3f}")
check("清除后覆盖来源记录仍保持（观测过就是观测过，单调）",
      bool(b7.own.known_mask[1, ix, iy, iz]) and bool(f7b.known_mask[0, ix, iy, iz]))

print("=== 12. 一手证据优先于二手 ===", flush=True)
b8 = MultiAgentBelief(1, 2, mcfg,
                      CommCfg(enabled=True, range_m=100.0, every=1, drop_prob=0.0, delay_steps=0),
                      device=dev)
# agent0 必须真的够得到 x=2 那个体素：量程只有 4 m，所以放在 x=-1（距离 3 m）。
# 上一版把它放在 x=-6（距离 8 m），它的 own 一直是 unknown，
# 按设计就该回落到二手证据，所以那个断言本身不成立。
pos8 = torch.tensor([[[-1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], device=dev)
hits8 = torch.full((1, 2, Rw, 3), float("inf"), device=dev)
hits8[0, 1, :, 0] = 2.0; hits8[0, 1, :, 1] = 0.0; hits8[0, 1, :, 2] = 1.0
for _ in range(6):
    b8.update_from_rays(pos8, hits8, dirs7)       # agent1 报占据
b8.step_comm(pos8, torch.zeros_like(pos8))
hits8[0, 0, :, :] = float("inf")                  # agent0 自己射穿 -> 自己认为是 free
for _ in range(4):
    b8.update_from_rays(pos8, hits8, dirs7)
f8 = b8.fused()
ok.append(bool(b8.own.known_mask[0, ix, iy, iz]))
print(f"  [{'OK ' if ok[-1] else 'FAIL'}] 前提：agent0 确实观测到了该体素  "
      f"own={float(b8.own.log_odds[0,ix,iy,iz]):.3f}", flush=True)
check("agent0 自己观测为 free 时，不被二手的占据证据顶回去",
      not bool(f8.occupied_mask[0, ix, iy, iz]),
      f"own={float(b8.own.log_odds[0,ix,iy,iz]):.3f} "
      f"recv={float(b8.recv_ev[0,ix,iy,iz]):.3f} fused={float(f8.log_odds[0,ix,iy,iz]):.3f}")

print(f"\n{'[DONE] 全部通过' if all(ok) else '[FAIL] 有失败项'}：{sum(ok)}/{len(ok)}", flush=True)
sys.exit(0 if all(ok) else 1)
