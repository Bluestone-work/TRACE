import math

import torch

from trace.pursuit.action import beta_action_to_velocity
from trace.pursuit.algorithm.mappo import PSTOMAPPO


def official_action_reference(action, quat=None, max_speed=0.8):
    angle = math.pi * (1.0 - 2.0 * action[..., 0])
    length = 0.9 + 0.1 * action[..., 1]
    body = torch.stack(
        (length * torch.cos(angle) * max_speed, length * torch.sin(angle) * max_speed, torch.zeros_like(length)),
        dim=-1,
    )
    if quat is None:
        return body
    q = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * body[..., 0] - s * body[..., 1], s * body[..., 0] + c * body[..., 1], body[..., 2]), -1)


def test_beta_action_body_and_world_parity():
    action = torch.tensor([[0.0, 0.0], [0.25, 0.5], [0.75, 1.0], [1.0, 0.3]])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], [1.0, 0.0, 0.0, 0.0], [math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)]])
    expected_body = official_action_reference(action)
    expected_world = official_action_reference(action, quat)
    assert torch.allclose(beta_action_to_velocity(action), expected_body, atol=1e-6, rtol=1e-6)
    assert torch.allclose(beta_action_to_velocity(action, quat), expected_world, atol=1e-6, rtol=1e-6)
    assert torch.allclose(PSTOMAPPO.action_to_body_velocity(action), expected_body, atol=1e-6, rtol=1e-6)
    assert torch.allclose(PSTOMAPPO.action_to_velocity(action, quat), expected_world, atol=1e-6, rtol=1e-6)
