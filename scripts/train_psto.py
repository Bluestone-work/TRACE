#!/usr/bin/env python3
"""Run a real, short PSTO MAPPO update in Isaac Sim PhysX."""
import argparse
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np


def load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def choose_checkpoint(args, experiment, seed):
    if args.checkpoint:
        return Path(args.checkpoint)
    checkpoint = Path(experiment.get("output_directory", "results/psto")) / f"seed_{seed}" / "checkpoint.pt"
    if not checkpoint.exists():
        return checkpoint
    base = checkpoint.parent
    index = 1
    while (base.parent / f"{base.name}_run{index}").exists():
        index += 1
    return base.parent / f"{base.name}_run{index}" / checkpoint.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/psto/trace_native.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--total_frames", type=int)
    parser.add_argument("--rollout_steps", type=int)
    parser.add_argument("--max_episode_steps", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--training_epoch_num", type=int)
    parser.add_argument("--num_minibatches", type=int)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    env_cfg, algo_cfg = config["environment"], config["mappo"]
    lstm_cfg, experiment_cfg = config["lstm"], config["experiment"]
    seed = config.get("seed", 4348) if args.seed is None else args.seed
    num_envs = env_cfg.get("num_envs", 1) if args.num_envs is None else args.num_envs
    total_frames = algo_cfg.get("total_frames", 32) if args.total_frames is None else args.total_frames
    rollout_steps = algo_cfg.get("rollout_steps", 8) if args.rollout_steps is None else args.rollout_steps
    max_episode_steps = env_cfg.get("max_episode_steps", 280) if args.max_episode_steps is None else args.max_episode_steps
    ppo_epochs = algo_cfg.get("ppo_epochs", 4) if args.training_epoch_num is None else args.training_epoch_num
    num_minibatches = algo_cfg.get("num_minibatches", 1) if args.num_minibatches is None else args.num_minibatches
    checkpoint = choose_checkpoint(args, experiment_cfg, seed)
    if num_envs != 1:
        raise SystemExit("The physical TRACE adapter currently supports --num_envs 1 only.")
    random.seed(seed)
    np.random.seed(seed)

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": bool(args.headless), "multi_gpu": False})
    env = None
    result = {"status": "blocked", "config": str(config_path), "seed": seed}
    try:
        import torch
        from trace.pursuit.algorithm.mappo import MAPPOConfig, PSTOMAPPO
        from trace.pursuit.env import PSTOPursuitEnv
        torch.manual_seed(seed)
        cuda_seeded = False
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            cuda_seeded = True
        scheduler_steps = max(int(np.ceil(total_frames / max(rollout_steps, 1))), 1)
        policy_config = MAPPOConfig(
            gamma=algo_cfg.get("gamma", .995), gae_lambda=algo_cfg.get("gae_lambda", .95),
            clip_ratio=algo_cfg.get("clip_param", .1), value_loss_coef=algo_cfg.get("value_loss_coef", 1.),
            warmup_steps=algo_cfg.get("warmup_steps", 0), scheduler=algo_cfg.get("scheduler", "cosine"),
            scheduler_steps=scheduler_steps, training_epoch_num=ppo_epochs, num_minibatches=num_minibatches,
            entropy_coef=algo_cfg.get("entropy_coef", .01), actor_lr=algo_cfg.get("actor_lr", 6e-5),
            critic_lr=algo_cfg.get("critic_lr", 6e-5), encoder_lr=algo_cfg.get("encoder_lr", 6e-5),
            sl_lr=algo_cfg.get("lstm_lr", algo_cfg.get("sl_lr", 8e-5)), lstm_update_frequency=algo_cfg.get("lstm_update_frequency", 1),
            lstm_observation_drop_prob=lstm_cfg.get("observation_dropout", .25),
            lstm_occlusion_prob=lstm_cfg.get("continuous_occlusion_probability", .35),
            lstm_max_occlusion_len=lstm_cfg.get("max_occlusion_length", 4))
        env = PSTOPursuitEnv(max_episode_steps=max_episode_steps, seed=seed,
            physics_dt=env_cfg["physics_dt"], policy_hz=env_cfg["policy_hz"],
            num_obstacles=env_cfg.get("num_obstacles", 18),
            randomize_obstacles=env_cfg.get("randomize_obstacles", False),
            randomize_resets=env_cfg.get("randomize_resets", False))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = PSTOMAPPO(config=policy_config, device=device)
        env.set_policy_lstm(policy.lstm)
        metadata = {"experiment_name": config["experiment_name"], "config": str(config_path), "seed": seed,
            "git_sha": git_sha(), "physics_dt": env_cfg["physics_dt"], "policy_hz": env_cfg["policy_hz"],
            "policy_dt": env_cfg["policy_dt"], "episode_duration": env_cfg["episode_duration"],
            "max_episode_steps": max_episode_steps, "frame_skip": env_cfg.get("frame_skip"),
            "num_envs": num_envs, "total_frames": total_frames, "rollout_steps": rollout_steps,
            "ppo_epochs": ppo_epochs, "num_minibatches": num_minibatches, "python": sys.version,
            "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "cuda_seeded": cuda_seeded}
        try:
            import omni.kit.app
            metadata["isaac_sim_version"] = omni.kit.app.get_app().get_build_version()
        except Exception:
            metadata["isaac_sim_version"] = "unavailable"
        print(json.dumps(metadata, indent=2), flush=True)
        resume_progress = {}
        if args.resume:
            if not checkpoint.exists():
                raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
            resume_progress = policy.load_model_weights(checkpoint, load_optimizers=True, restore_rng=True)
            if resume_progress.get("env_state") is not None:
                env.load_state_dict(resume_progress["env_state"])

        obs = env.observe() if args.resume and resume_progress.get("env_state") is not None else env.reset()
        frames = int(resume_progress.get("frames", 0)) if args.resume else 0
        updates = int(resume_progress.get("updates", policy._update_count)) if args.resume else 0
        terminated_count = truncated_count = 0
        logs = []
        while frames < total_frames:
            images, states, centrals, actions = [], [], [], []
            log_probs, values, next_values, rewards = [], [], [], []
            terminateds, episode_ends = [], []
            histories, futures, masks, dts, supervised_valid = [], [], [], [], []
            for _ in range(min(rollout_steps, total_frames - frames)):
                image = torch.as_tensor(obs["fused_image"], device=policy.device)
                state = torch.as_tensor(obs["state"], device=policy.device)
                central = torch.as_tensor(obs["central_state"], device=policy.device)
                with torch.no_grad():
                    action, log_prob, value = policy.act(image, state, central)
                next_obs, reward, done, info = env.step(action.cpu().numpy(), actions_are_normalized=True)
                with torch.no_grad():
                    if info["terminated"]:
                        next_value = torch.zeros(1, device=policy.device)
                    else:
                        next_central = torch.as_tensor(next_obs["central_state"], device=policy.device)
                        next_value = policy.critic(policy.critic_encoder(next_central)).reshape(1)
                images.append(image.cpu()); states.append(state.cpu()); centrals.append(central.cpu())
                actions.append(action.cpu()); log_probs.append(log_prob.cpu()); values.append(value.reshape(1).cpu())
                next_values.append(next_value.cpu()); rewards.append(float(reward))
                terminateds.append(float(info["terminated"])); episode_ends.append(float(info["episode_end"]))
                histories.append(torch.as_tensor(obs["lstm_history"], dtype=torch.float32))
                futures.append(torch.as_tensor(obs["future_ground_truth"], dtype=torch.float32))
                masks.append(torch.as_tensor(obs["lstm_history_mask"], dtype=torch.bool))
                dts.append(torch.as_tensor(obs["lstm_history_dt"], dtype=torch.float32))
                supervised_valid.append(bool(obs["lstm_target_available"]))
                terminated_count += int(info["terminated"]); truncated_count += int(info["truncated"])
                frames += 1
                obs = env.reset() if done else next_obs
            rollout = {"image": torch.stack(images)[:, None], "state": torch.stack(states)[:, None],
                "central": torch.stack(centrals)[:, None], "action": torch.stack(actions)[:, None],
                "log_prob": torch.stack(log_probs)[:, None], "value": torch.stack(values)[:, None],
                "reward": torch.tensor(rewards, dtype=torch.float32)[:, None],
                "terminated": torch.tensor(terminateds, dtype=torch.float32)[:, None],
                "episode_end": torch.tensor(episode_ends, dtype=torch.float32)[:, None],
                "next_value": torch.stack(next_values)[:, None]}
            rollout.update({"history": torch.stack(histories), "future": torch.stack(futures),
                "history_mask": torch.stack(masks), "agent_dt": torch.stack(dts),
                "supervised_valid": torch.as_tensor(supervised_valid, dtype=torch.bool)})
            log = policy.update(rollout); policy.step_schedulers(); updates += 1
            logs.append({"frames": frames, "update": updates, **log}); policy.save_state(checkpoint, progress={"frames":frames,"updates":updates,"env_state":env.state_dict()})
            print(json.dumps(logs[-1]), flush=True)
        result = {**metadata, "status": "pass", "frames": frames, "updates": updates,
            "terminated_steps": terminated_count, "truncated_steps": truncated_count,
            "checkpoint": str(checkpoint), "logs": logs}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result.update({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        checkpoint.parent.joinpath("config_snapshot.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        if env is not None:
            env.close()
        app.close()
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
