from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from npc_rl.baseline.baseline_agent import BaselineAgent, BaselineConfig


def _reset_env(env, seed: int):
    out = env.reset(seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}


def _step_env(env, action: int):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated), bool(truncated), info
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward), bool(done), False, info
    raise ValueError(f"Unexpected env.step return format: {type(out)} len={len(out) if isinstance(out, tuple) else 'NA'}")


def run_episode(env, agent: BaselineAgent, seed: int, max_steps: int):
    obs, info = _reset_env(env, seed)
    total_reward = 0.0
    steps = 0
    shots = 0
    hits = 0
    terminated = False
    truncated = False
    last_info = info if isinstance(info, dict) else {}

    while not (terminated or truncated) and steps < max_steps:
        valid_moves = None
        if isinstance(last_info, dict) and "valid_moves" in last_info:
            vm = last_info["valid_moves"]
            if isinstance(vm, (list, tuple)) and all(isinstance(x, int) for x in vm):
                valid_moves = tuple(vm)

        action = agent.act(obs, valid_moves=valid_moves)
        obs, reward, terminated, truncated, info = _step_env(env, action)

        total_reward += reward
        steps += 1
        if isinstance(info, dict):
                last_info = info

        # simple terminal reward metric
       
    win = 1 if reward > 0 else 0
    accuracy = float(hits / shots) if shots > 0 else 0.0


    return {
        "seed": seed,
        "agent": "baseline",
        "win": win,
        "return": total_reward,
        "steps": steps,
        "shots": shots,
        "hits": hits,
        "accuracy": accuracy,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results_baseline")
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--seed_start", type=int, default=10000)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--retreat_threshold", type=float, default=0.34)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed_start + i for i in range(args.n_eval)]
    (outdir / "eval_seeds.txt").write_text("\n".join(map(str, seeds)), encoding="utf-8")

    cfg = BaselineConfig(retreat_threshold=args.retreat_threshold)
    agent = BaselineAgent(cfg)
    from npc_rl.env.shootergrid_env import ShooterGridEnv
    env = ShooterGridEnv()

  


    rows = []
    for s in seeds:
        rows.append(run_episode(env, agent, seed=s, max_steps=args.max_steps))

    if len(rows)==0:
        print("No episodes collected")
        return

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "episodes_baseline.csv", index=False)

    summary = df.agg({
        "win": ["mean", "count"],
        "return": ["mean", "std"],
        "steps": ["mean", "std"],
        "accuracy": ["mean"],
    })

    summary.to_csv(outdir / "metrics_baseline.csv")

    print("Wrote:", outdir / "episodes_baseline.csv")
    print("Wrote:", outdir / "metrics_baseline.csv")


if __name__ == "__main__":
    main()

