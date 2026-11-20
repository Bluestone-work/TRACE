import torch

from trace.pursuit.rewards import official_reward


def test_capture_and_collision_are_termination_conditions():
    pursuer = torch.tensor([[[0.0, 0.0, 2.0], [4.0, 0.0, 2.0]]])
    evader = torch.tensor([[0.4, 0.0, 2.0]])
    reward, captured, escaped, done, _ = official_reward(
        pursuer,
        torch.zeros_like(pursuer),
        evader,
        torch.zeros(1, 2, 6, 120),
        pursuer_collision=torch.tensor([False]),
        evader_collision=torch.tensor([False]),
    )
    assert captured.item() and not escaped.item() and done.item()
    assert reward.item() > 1000


def test_below_bound_is_termination():
    pursuer = torch.tensor([[[0.0, 0.0, 0.4], [4.0, 0.0, 2.0]]])
    evader = torch.tensor([[2.0, 0.0, 2.0]])
    _, captured, escaped, done, terms = official_reward(
        pursuer,
        torch.zeros_like(pursuer),
        evader,
        torch.zeros(1, 2, 6, 120),
        pursuer_collision=torch.tensor([False]),
        evader_collision=torch.tensor([False]),
    )
    assert not captured.item() and not escaped.item() and done.item()
    assert terms["below_bound"].item()
