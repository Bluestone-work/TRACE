"""Export the released NavRL checkpoint to a NumPy-only inference bundle."""

from pathlib import Path
import sys
import numpy as np
import torch

QUICK = Path(__file__).resolve().parents[1] / "quick-demos"
sys.path.insert(0, str(QUICK))
from ppo import PPO  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec  # noqa: E402


def main() -> None:
    device = torch.device("cpu")
    observation = CompositeSpec({
        "state": UnboundedContinuousTensorSpec((8,)),
        "lidar": UnboundedContinuousTensorSpec((1, 36, 4)),
        "direction": UnboundedContinuousTensorSpec((1, 3)),
        "dynamic_obstacle": UnboundedContinuousTensorSpec((1, 5, 10)),
    })
    obs = CompositeSpec({"agents": CompositeSpec({"observation": observation})}).expand(1)
    act = CompositeSpec({"agents": CompositeSpec({"action": UnboundedContinuousTensorSpec((3,))})}).expand(1, 3)
    policy = PPO(obs, act, device)
    ckpt = QUICK / "ckpts/navrl_checkpoint.pt"
    policy.load_state_dict(torch.load(ckpt, map_location=device))
    output = Path(__file__).resolve().parent / "navrl_checkpoint_numpy.npz"
    state = {k.replace('.', '__'): v.detach().cpu().numpy() for k, v in policy.state_dict().items()}
    np.savez_compressed(output, **state)
    print(output)


if __name__ == "__main__":
    main()
