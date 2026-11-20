import torch

from trace.pursuit.evader import evader_policy


def _norm(v):
    n = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    return torch.where(n > 1e-6, v / (n + 1e-6), torch.zeros_like(v))


def official_evader_reference(pursuers, evader, hits=None):
    pursuers = pursuers.unsqueeze(0)
    evader = evader.unsqueeze(0)
    center = torch.zeros_like(evader)
    vec_center = center - evader
    vec_center[..., 2] = 0
    dist_wall = 9.0 - torch.linalg.vector_norm(vec_center[..., :2], dim=-1)
    flee = evader.unsqueeze(1) - pursuers
    flee[..., 2] = 0
    dp = torch.linalg.vector_norm(flee, dim=-1)
    strength = torch.clamp(18.0 / (dp.clamp(min=0.1) ** 2 + 1e-6), max=50.0)
    pursuer_force = (_norm(flee) * strength.unsqueeze(-1)).sum(1)
    obstacle_force = torch.zeros_like(evader)
    if hits is not None:
        rel = hits.unsqueeze(0) - evader.unsqueeze(1)
        d = torch.linalg.vector_norm(rel, dim=-1)
        hit_center_dist = torch.linalg.vector_norm((hits.unsqueeze(0) - center.unsqueeze(1))[..., :2], dim=-1)
        mask = (d < 2.0) & torch.isfinite(d) & (hit_center_dist < 7.5)
        obstacle_force = (-_norm(rel) * torch.where(mask, 1.0 / d.clamp_min(0.01) ** 2, torch.zeros_like(d)).unsqueeze(-1)).sum(1)
    near = dist_wall < 2.2
    effective = torch.zeros_like(evader)
    desired = pursuer_force + obstacle_force * 0.1
    effective[~near] = desired[~near]
    if near.any():
        out = -_norm(vec_center[near])
        dprod = (desired[near] * out).sum(-1, keepdim=True)
        projected = desired[near] - torch.clamp(dprod, min=0) * out
        effective[near] = _norm(_norm(projected) + (-out) * 0.1)
    speed = torch.linalg.vector_norm(effective, dim=-1, keepdim=True)
    final = torch.where(near.unsqueeze(-1), torch.full_like(speed, 1.7), speed.clamp(max=1.7))
    target = _norm(effective) * final
    target[..., 2] += (2.0 - evader[..., 2]) * 2.0
    return target[0]


def test_evader_policy_matches_official_force_model():
    pursuers = torch.tensor([[-2.0, 1.0, 2.0], [-2.0, -1.0, 2.0]])
    evader = torch.tensor([0.0, 0.0, 2.0])
    hits = torch.tensor([[0.6, 0.0, 2.0], [10.0, 0.0, 2.0]])
    expected = official_evader_reference(pursuers, evader, hits)
    actual = evader_policy(pursuers, evader, obstacle_hits=hits)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
