"""One-batch PPO training smoke test for the Isaac Sim 6 migration.

This deliberately uses one native Isaac Sim environment and a short rollout.
It verifies the pieces that are needed before scaling training out: the
Isaac6 observation contract, the original NavRL policy, GAE/return creation,
and one differentiable PPO update.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch
from isaacsim import SimulationApp


def _disable_isaac_fast_importer() -> list[object]:
    """Make torch submodules resolve from torch_site during NavRL imports."""
    previous = list(sys.meta_path)
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not (
            getattr(finder, "__name__", "") == "FastFinder"
            or getattr(finder, "__module__", "").startswith(
                "omni.ext._impl.fast_importer"
            )
        )
    ]
    return previous


def _observation_tensordict(obs: dict[str, object], device: torch.device):
    from tensordict import TensorDict

    return TensorDict(
        {
            "agents": {
                "observation": {
                    "state": torch.as_tensor(obs["state"], device=device).reshape(1, 8),
                    "lidar": torch.as_tensor(obs["lidar"], device=device).reshape(1, 1, 36, 4),
                    "direction": torch.as_tensor(obs["direction"], device=device).reshape(1, 1, 3),
                    "dynamic_obstacle": torch.as_tensor(
                        obs["dynamic_obstacle"], device=device
                    ).reshape(1, 1, 5, 10),
                }
            }
        },
        batch_size=[1],
        device=device,
    )


def _collect_rollout(policy, env, horizon: int, device: torch.device):
    from tensordict import TensorDict

    obs = env.reset()
    transitions = []
    for _ in range(horizon):
        current = _observation_tensordict(obs, device)
        transition = policy(current)
        # Rollout tensors are PPO targets; keep the update graph local to each
        # minibatch and avoid retaining the policy forward graph across steps.
        transition = transition.detach()
        action = transition["agents", "action"].detach().cpu().numpy().reshape(3)
        next_obs, reward, terminated, _ = env.step(action)
        transition.set(
            "next",
            TensorDict(
                {
                    "agents": {
                        "observation": {
                            "state": torch.as_tensor(next_obs["state"], device=device).reshape(1, 8),
                            "lidar": torch.as_tensor(next_obs["lidar"], device=device).reshape(
                                1, 1, 36, 4
                            ),
                            "direction": torch.as_tensor(next_obs["direction"], device=device).reshape(
                                1, 1, 3
                            ),
                            "dynamic_obstacle": torch.as_tensor(
                                next_obs["dynamic_obstacle"], device=device
                            ).reshape(1, 1, 5, 10),
                        },
                        "reward": torch.tensor([reward], device=device, dtype=torch.float32),
                    },
                    "terminated": torch.tensor([terminated], device=device, dtype=torch.bool),
                },
                batch_size=[1],
                device=device,
            ),
        )
        transitions.append(transition)
        obs = env.reset() if terminated else next_obs

    # One environment and horizon time steps.  The original PPO implementation
    # expects [num_envs, num_steps], so stack on dimension 1.
    return TensorDict.stack(transitions, dim=1)


def _ppo_update(policy, data, num_minibatches: int = 2) -> dict[str, float]:
    """The original NavRL PPO update, kept local to avoid old OmniDrones APIs."""
    with torch.no_grad():
        # The rollout has [num_envs, num_steps] batch dimensions.  The
        # feature extractor itself operates on one time slice, so vectorize
        # it across the leading environment dimension as the original PPO
        # implementation does.
        next_data = torch.vmap(policy.feature_extractor)(data["next"])
        next_values = policy.critic(next_data)["state_value"]

    rewards = data["next", "agents", "reward"]
    dones = data["next", "terminated"]
    if rewards.ndim == data["state_value"].ndim - 1:
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)
    values = policy.value_norm.denormalize(data["state_value"])
    next_values = policy.value_norm.denormalize(next_values)
    advantages, returns = policy.gae(rewards, dones, values, next_values)
    print(
        "ISAAC6_TRAIN_GAE_SHAPES"
        f" rewards={tuple(rewards.shape)} values={tuple(values.shape)}"
        f" next={tuple(next_values.shape)} returns={tuple(returns.shape)}",
        flush=True,
    )
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-7)
    policy.value_norm.update(returns)
    data.set("adv", advantages)
    data.set("ret", policy.value_norm.normalize(returns))

    infos = []
    flat = data.reshape(-1)
    usable = (flat.shape[0] // num_minibatches) * num_minibatches
    if usable == 0:
        raise RuntimeError("rollout is shorter than the requested minibatch count")
    indices = torch.randperm(usable, device=flat.device).reshape(num_minibatches, -1)
    for minibatch_indices in indices:
        td = flat[minibatch_indices]
        policy.feature_extractor(td)
        action_dist = policy.actor.get_dist(td)
        log_probs = action_dist.log_prob(td["agents", "action_normalized"])
        entropy_loss = -0.001 * action_dist.entropy().mean()
        ratio = torch.exp(log_probs - td["sample_log_prob"]).unsqueeze(-1)
        advantage = td["adv"]
        surr1 = advantage * ratio
        surr2 = advantage * ratio.clamp(0.9, 1.1)
        actor_loss = -torch.minimum(surr1, surr2).mean() * policy.action_dim

        old_value = td["state_value"]
        value = policy.critic(td)["state_value"]
        value_clipped = old_value + (value - old_value).clamp(-0.1, 0.1)
        critic_loss = torch.maximum(
            policy.critic_loss_fn(td["ret"], value_clipped),
            policy.critic_loss_fn(td["ret"], value),
        )
        loss = entropy_loss + actor_loss + critic_loss
        policy.feature_extractor_optim.zero_grad()
        policy.actor_optim.zero_grad()
        policy.critic_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(policy.critic.parameters(), 5.0)
        policy.feature_extractor_optim.step()
        policy.actor_optim.step()
        policy.critic_optim.step()
        infos.append((actor_loss.detach(), critic_loss.detach(), entropy_loss.detach()))

    mean_info = torch.stack([torch.stack(item) for item in infos]).mean(0)
    return {
        "actor_loss": float(mean_info[0]),
        "critic_loss": float(mean_info[1]),
        "entropy_loss": float(mean_info[2]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    print("ISAAC6_TRAIN_START", flush=True)
    app = SimulationApp({"headless": True})
    env = None
    try:
        quick = Path(__file__).resolve().parents[1] / "quick-demos"
        sys.path.insert(0, str(quick))
        previous_meta_path = _disable_isaac_fast_importer()
        try:
            from agent import Agent
        finally:
            sys.meta_path[:] = previous_meta_path

        from navrl6_env import Isaac6NavEnv

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        env = Isaac6NavEnv()
        print("ISAAC6_TRAIN_ENV_OK", flush=True)
        agent = Agent(device)
        print("ISAAC6_TRAIN_POLICY_OK", flush=True)
        data = _collect_rollout(agent.policy, env, args.steps, device)
        print(f"ISAAC6_TRAIN_ROLLOUT_OK shape={tuple(data.batch_size)}", flush=True)
        stats = _ppo_update(agent.policy, data)
        checkpoint = Path(__file__).resolve().with_name("navrl6_smoke_checkpoint.pt")
        torch.save(agent.policy.state_dict(), checkpoint)
        print(
            "ISAAC6_TRAIN_OK"
            f" rollout_shape={tuple(data.batch_size)}"
            f" actor_loss={stats['actor_loss']:.5f}"
            f" critic_loss={stats['critic_loss']:.5f}"
            f" checkpoint={checkpoint.name}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        print(f"ISAAC6_TRAIN_ERROR {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
