#!/usr/bin/env python3
"""Independent, gradient-free evaluation of a physical PSTO checkpoint."""
import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np


def load_config(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/psto/trace_native.yaml")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max_steps", type=int)
    ap.add_argument("--out", default="results/psto_eval/eval.json")
    ap.add_argument("--randomize", action="store_true", help="randomize evaluation positions and obstacle layout")
    args = ap.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    env_cfg = config["environment"]
    max_steps = env_cfg.get("max_episode_steps", 280) if args.max_steps is None else args.max_steps
    out = Path(args.out)
    if out.exists():
        i = 1
        while out.with_name(f"{out.stem}_run{i}{out.suffix}").exists():
            i += 1
        out = out.with_name(f"{out.stem}_run{i}{out.suffix}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": bool(args.headless), "multi_gpu": False})
    env = None
    result = {"status": "blocked", "config": str(config_path), "seed": args.seed}
    try:
        import torch
        from trace.pursuit.algorithm.mappo import PSTOMAPPO
        from trace.pursuit.env import PSTOPursuitEnv
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = PSTOMAPPO(device=device)
        policy.load_model_weights(args.checkpoint, load_optimizers=False)
        policy.eval()
        env = PSTOPursuitEnv(max_episode_steps=max_steps, seed=args.seed,
            physics_dt=env_cfg["physics_dt"], policy_hz=env_cfg["policy_hz"],
            num_obstacles=env_cfg.get("num_obstacles", 18), randomize_resets=args.randomize,
            randomize_obstacles=args.randomize)
        env.set_policy_lstm(policy.lstm)
        rows = []
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            obs = env.reset(seed=episode_seed)
            total = 0.; visibility = 0.; tf_ade = []; tf_fde = []; oo_ade = []; oo_fde = []
            terminal = {}; done = False; lost_at = None; reacq = []
            true_positions = [env._root()[0][2].copy()]
            prediction_queue = []
            for step in range(max_steps):
                with torch.no_grad():
                    action, _, _ = policy.act(torch.as_tensor(obs["fused_image"], device=device),
                        torch.as_tensor(obs["state"], device=device), torch.as_tensor(obs["central_state"], device=device), deterministic=True)
                pred_tf = pred_oo = None
                with torch.no_grad():
                    if np.asarray(obs["teacher_history_mask"]).any():
                        th = torch.as_tensor(obs["teacher_history"], device=device)[None]
                        tm = torch.as_tensor(obs["teacher_history_mask"], device=device)[None]
                        td = torch.as_tensor(obs["teacher_history_dt"], device=device)[None]
                        pred_tf = policy.lstm(torch.cat((th, td), -1), mask=tm).view(10, 3).cpu().numpy()
                    if np.asarray(obs["history_mask"]).any():
                        oh = torch.as_tensor(obs["history"], device=device)[None]
                        om = torch.as_tensor(obs["history_mask"], device=device)[None]
                        od = torch.as_tensor(obs["history_dt"], device=device)[None]
                        pred_oo = policy.lstm(torch.cat((oh, od), -1), mask=om).view(10, 3).cpu().numpy()
                prediction_queue.append((len(true_positions) - 1, pred_tf, pred_oo))
                nxt, reward, done, info = env.step(action.cpu().numpy(), actions_are_normalized=True)
                total += reward; visibility += info["visibility_ratio"]; terminal = info
                true_positions.append(env._root()[0][2].copy())
                while prediction_queue and len(true_positions) - 1 - prediction_queue[0][0] >= 10:
                    start, pred_tf_i, pred_oo_i = prediction_queue.pop(0)
                    gt = np.asarray(true_positions[start + 1:start + 11], np.float32)
                    if pred_tf_i is not None:
                        tf_ade.append(float(np.linalg.norm(pred_tf_i - gt, axis=-1).mean())); tf_fde.append(float(np.linalg.norm(pred_tf_i[-1] - gt[-1])))
                    if pred_oo_i is not None:
                        oo_ade.append(float(np.linalg.norm(pred_oo_i - gt, axis=-1).mean())); oo_fde.append(float(np.linalg.norm(pred_oo_i[-1] - gt[-1])))
                visible = info["visibility_ratio"] > 0.
                if not visible and lost_at is None: lost_at = step
                if visible and lost_at is not None:
                    reacq.append((step - lost_at) / env.scene.policy_dt); lost_at = None
                obs = nxt
                if done: break
            rows.append({"episode": episode, "seed": episode_seed, "steps": step + 1, "reward": total,
                "capture": bool(terminal.get("capture", False)), "escaped": bool(terminal.get("escaped", False)),
                "collision": bool(terminal.get("collision", False)), "visibility_ratio": visibility / (step + 1),
                "reacquisition_time_s": float(np.mean(reacq)) if reacq else None,
                "teacher_forced_ade": float(np.mean(tf_ade)) if tf_ade else None,
                "teacher_forced_fde": float(np.mean(tf_fde)) if tf_fde else None,
                "observation_only_ade": float(np.mean(oo_ade)) if oo_ade else None,
                "observation_only_fde": float(np.mean(oo_fde)) if oo_fde else None})
        def mean(key):
            values = [r[key] for r in rows if r[key] is not None]
            return float(np.mean(values)) if values else None
        result = {"status": "pass", "config": str(config_path), "git_sha": git_sha(), "seed": args.seed,
            "episodes": len(rows), "capture_success_rate": float(np.mean([r["capture"] for r in rows])),
            "average_capture_time_s": float(np.mean([r["steps"] * env.scene.policy_dt for r in rows if r["capture"]])) if any(r["capture"] for r in rows) else None,
            "escape_rate": float(np.mean([r["escaped"] for r in rows])), "collision_rate": float(np.mean([r["collision"] for r in rows])),
            "target_visibility_ratio": float(np.mean([r["visibility_ratio"] for r in rows])),
            "reacquisition_time_s": mean("reacquisition_time_s"), "teacher_forced_ade": mean("teacher_forced_ade"),
            "teacher_forced_fde": mean("teacher_forced_fde"), "observation_only_ade": mean("observation_only_ade"),
            "observation_only_fde": mean("observation_only_fde"), "episode_reward": float(np.mean([r["reward"] for r in rows])),
            "episodes_detail": rows}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result.update({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        out.with_name(out.stem + "_config_snapshot.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        if env is not None: env.close()
        app.close()
    if result["status"] != "pass": raise SystemExit(1)


if __name__ == "__main__": main()
