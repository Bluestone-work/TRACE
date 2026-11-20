import torch

from trace.pursuit.observations import generate_trajectory_heatmap, world_to_body


def _reference_world_to_body(v, q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q.unbind(-1)
    qc = torch.stack((w, -x, -y, -z), -1)
    t = 2 * torch.cross(qc[..., 1:], v, dim=-1)
    return v + qc[..., 0:1] * t + torch.cross(qc[..., 1:], t, dim=-1)


def _reference_heatmap(observer_pos, observer_quat, trajectories, signs, gammas):
    # Direct scalar translation of official env.py lines 729-782.
    b, p, t = trajectories.shape[:3]
    v, h = 24, 480
    vd, hd = 6, 120
    rel = _reference_world_to_body(trajectories - observer_pos.unsqueeze(2), observer_quat.unsqueeze(2))
    x, y, z = rel.unbind(-1)
    az = (torch.atan2(y, x) * 180 / torch.pi) % 360
    el = torch.atan2(z, torch.sqrt((x * x + y * y).clamp_min(1e-12))) * 180 / torch.pi
    iv = ((el + 10) / ((20 + 10) / (v - 1))).round().long().clamp(0, v - 1) // 4
    ih = (torch.floor(az / 0.75).long() % h) // 4
    idx = iv * hd + ih
    values = signs * (10 / torch.linalg.vector_norm(rel, dim=-1).clamp_min(1e-6)) * gammas
    pos = torch.full((b, p, vd * hd), -torch.inf)
    neg = torch.full_like(pos, torch.inf)
    pos.scatter_reduce_(2, idx, torch.where(values > 0, values, -torch.inf), reduce="amax", include_self=False)
    neg.scatter_reduce_(2, idx, torch.where(values < 0, values, torch.inf), reduce="amin", include_self=False)
    pos[torch.isinf(pos)] = 0
    neg[torch.isinf(neg)] = 0
    return (pos + neg).view(b, p, vd, hd)


def test_world_to_body_and_heatmap_parity():
    observer_pos = torch.tensor([[[0.0, 0.0, 2.0]]])
    observer_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    trajectories = torch.tensor([[[[2.0, 0.0, 2.0], [1.0, 1.0, 2.0], [0.0, 2.0, 2.0]]]])
    signs = torch.tensor([[[2.0, 1.5, -2.0]]])
    gammas = torch.tensor([[[1.0, 0.95, 1.0]]])
    vectors = torch.tensor([[[1.0, 2.0, 0.0]]])
    assert torch.allclose(world_to_body(vectors, observer_quat), _reference_world_to_body(vectors, observer_quat))
    expected = _reference_heatmap(observer_pos, observer_quat, trajectories, signs, gammas)
    actual = generate_trajectory_heatmap(observer_pos, observer_quat, trajectories, signs, gammas)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
