import torch

from trace.pursuit.rewards import official_reward


def test_reward_terms_match_official_formula_without_contacts():
    pursuer_pos = torch.tensor([[[-2.0, 1.0, 2.0], [-2.0, -1.0, 2.0]]])
    pursuer_vel = torch.tensor([[[0.4, 0.0, 0.0], [0.2, 0.1, 0.0]]])
    evader_pos = torch.tensor([[0.0, 0.0, 2.0]])
    evader_vel = torch.tensor([[0.3, 0.0, 0.0]])
    lidar = torch.full((1, 2, 6, 120), 0.0)
    reward, captured, escaped, done, terms = official_reward(
        pursuer_pos,
        pursuer_vel,
        evader_pos,
        lidar,
        evader_vel=evader_vel,
        step=100,
        pursuer_collision=torch.tensor([False]),
        evader_collision=torch.tensor([False]),
    )
    assert reward.shape == captured.shape == escaped.shape == done.shape == (1,)
    assert torch.isfinite(reward).all()
    assert not captured.item() and not escaped.item() and not done.item()
    # Formation and coordination are nonzero in this nondegenerate geometry;
    # a constant-zero placeholder would fail this check.
    assert abs(float(terms["formation"])) > 1e-6
    assert abs(float(terms["coordination"])) > 1e-6


def test_escape_is_or_of_min_and_max_distance():
    pos = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]])
    vel = torch.zeros_like(pos)
    near = torch.tensor([[7.6, 0.0, 2.0]])
    _, _, escaped, done, _ = official_reward(pos, vel, near, torch.zeros(1, 2, 6, 120), pursuer_collision=torch.tensor([False]), evader_collision=torch.tensor([False]))
    assert escaped.item() and done.item()
