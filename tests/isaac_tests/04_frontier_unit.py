from pathlib import Path
"""frontier 提取单元测试（合成地图，结果可手算核对）。"""
import sys, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trace.mapping.occupancy_map import VoxelMapCfg, VoxelOccupancyMap
from trace.mapping.frontier import FrontierCfg, frontier_mask, label_components, extract_task_nodes

dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
cfg = VoxelMapCfg(size=(4., 4., 1.), origin=(0., 0., 0.), resolution=0.5)
m = VoxelOccupancyMap(1, cfg, device=dev)
print(f"地图 {m.nx}x{m.ny}x{m.nz} = {m.num_voxels} 体素, res={m.res}")

# 两块自由空间（z=0 层），中间 x=3,4 保持 unknown 隔开
m.log_odds[0, 0:3, :, 0] = m.p_min_log
m.log_odds[0, 5:8, :, 0] = m.p_min_log
c = m.counts()
print(f"构造后: free={int(c['free'][0])} unknown={int(c['unknown'][0])}")
assert int(c['free'][0]) == 48, "应有 2x(3x8)=48 个 free 体素"

fcfg = FrontierCfg(adjacency=6, cluster=26, min_cluster_voxels=1)
fm = frontier_mask(m, fcfg)
nf = int(fm[0].sum())
print(f"[OK] frontier 体素数 = {nf}")
# z=1 整层 unknown，所以 z=0 的每个 free 体素上方都有 unknown -> 全是 frontier
assert nf == 48, f"期望 48, 实得 {nf}"

labels, conv = label_components(fm, fcfg)
ids = torch.unique(labels[labels > 0])
print(f"[OK] 连通分量数 = {ids.numel()}, 标签传播收敛 = {conv}")
assert ids.numel() == 2, f"期望 2 个分量（被 unknown 隔开），实得 {ids.numel()}"

nodes = extract_task_nodes(m, labels, fcfg, max_nodes=8)
nv = nodes['n_voxels'][0][:2]
ctr = nodes['centroid'][0][:2]
pos = nodes['pos'][0][:2]
print(f"[OK] num_nodes = {int(nodes['num_nodes'][0])}")
print(f"[OK] 各簇体素数 = {[int(x) for x in nv]}")
print(f"[OK] 簇质心 centroid = {[[round(float(v),3) for v in c] for c in ctr]}")
print(f"[OK] 代表点 pos      = {[[round(float(v),3) for v in p] for p in pos]}")
assert int(nv[0]) == 24 and int(nv[1]) == 24, "两簇各应 24 体素"
xs = sorted([float(c[0]) for c in ctr])
assert abs(xs[0] - 0.75) < 1e-4 and abs(xs[1] - 3.25) < 1e-4, f"质心 x 应为 0.75/3.25, 实得 {xs}"
for c in ctr:
    assert abs(float(c[1]) - 2.0) < 1e-4 and abs(float(c[2]) - 0.25) < 1e-4
print("[OK] 质心与手算一致 (x=0.75/3.25, y=2.0, z=0.25)")
# 代表点必须落在 frontier 体素上（质心对环形簇会落在已探明区，见 docs）
for p in pos:
    ip, _ = m.world_to_index(p)
    assert bool(fm[0, ip[0], ip[1], ip[2]]), f"代表点 {p} 不在 frontier 上"
print("[OK] 两个代表点都落在 frontier 体素上")
print(f"[OK] n_unknown (IG 代理) = {[int(x) for x in nodes['n_unknown'][0][:2]]}")
print(f"[OK] bbox_min[0] = {[round(float(v),3) for v in nodes['bbox_min'][0][0]]}")
print(f"[OK] bbox_max[0] = {[round(float(v),3) for v in nodes['bbox_max'][0][0]]}")

# min_cluster_voxels 过滤
f2 = FrontierCfg(adjacency=6, cluster=26, min_cluster_voxels=30)
l2, _ = label_components(fm, f2)
print(f"[OK] min_cluster_voxels=30 过滤后分量数 = {torch.unique(l2[l2>0]).numel()} (两簇各 24, 应全被丢弃)")
assert torch.unique(l2[l2 > 0]).numel() == 0
print("[DONE] frontier 提取验证通过")
