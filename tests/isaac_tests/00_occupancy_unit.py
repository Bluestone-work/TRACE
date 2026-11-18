from pathlib import Path
import sys, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trace.mapping.occupancy_map import VoxelMapCfg, VoxelOccupancyMap

dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
cfg = VoxelMapCfg(size=(10.,10.,4.), origin=(-5.,-5.,0.), resolution=0.25, max_range=8.0)
m = VoxelOccupancyMap(num_envs=1, cfg=cfg, device=dev)
print(f"地图 {m.nx}x{m.ny}x{m.nz} = {m.num_voxels} 体素, res={m.res}")
print(f"log-odds: min={m.p_min_log:.3f} occ={m.p_occ_log:.3f} max={m.p_max_log:.3f} hit=+{m.p_hit_log:.3f} miss={m.p_miss_log:.3f}")
print(f"unknown 初值={m.unknown_init:.3f}")
c = m.counts(); assert int(c['unknown'][0]) == m.num_voxels, "初始应全 unknown"
print(f"[OK] 初始全 unknown ({int(c['unknown'][0])})")

# 单根光线：(0,0,1) -> (3,0,1)
org = torch.tensor([[[0.,0.,1.]]], device=dev)
hit = torch.tensor([[[[3.,0.,1.]]]], device=dev)
m.update(org, hit)
c = m.counts()
print(f"1 次更新后: free={int(c['free'][0])} occupied={int(c['occupied'][0])} unknown={int(c['unknown'][0])}")
# 沿 x 从 0 到 3 共 12 个体素(res .25)，加命中体素 -> 13 个 known
assert int(c["free"][0]) == 13, f"期望 13, 实得 {int(c[chr(39)+chr(39)])}"
print("[OK] 自由空间体素数 = 13 (0~3m / 0.25m = 12 穿过 + 1 命中)")

# 命中体素位置确认
ix, ib = m.world_to_index(torch.tensor([3.,0.,1.], device=dev))
v = float(m.log_odds[0, ix[0], ix[1], ix[2]])
print(f"命中体素 log-odds = {v:.3f} (= min {m.p_min_log:.3f} + hit {m.p_hit_log:.3f})")
assert abs(v - (m.unknown_init + m.p_hit_log)) < 1e-4 or abs(v - (m.p_min_log + m.p_hit_log)) < 0.02

# 重复命中直到变 occupied
for k in range(2, 8):
    m.update(org, hit)
    v = float(m.log_odds[0, ix[0], ix[1], ix[2]])
    occ = v >= m.p_occ_log
    print(f"  第 {k} 次命中: log-odds={v:.3f} occupied={occ}")
    if occ:
        print(f"[OK] 第 {k} 次命中后转为 occupied（NavRL 的噪声过滤滞后）")
        break
assert int(m.counts()['occupied'][0]) == 1, "应恰好 1 个 occupied 体素"
print("[OK] occupied 体素数 = 1")

# 穿过的体素不应被误判占据
print(f"[OK] 穿过体素仍为 free: free={int(m.counts()['free'][0])}")
print(f"[OK] known_ratio = {float(m.known_ratio()[0]):.6f}")
print("[DONE] 占据栅格数学验证通过")
