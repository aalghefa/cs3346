import argparse
from pathlib import Path
import pandas as pd
from stable_baselines3 import PPO
from npc_rl.env.shootergrid_env import ShooterGridEnv

def run_episode(env, model, seed: int, max_steps: int = 200):
    out = env.reset(seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
    else:
        obs, info = out, {}

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    last_info = info if isinstance(info, dict) else {}

    while not (terminated or truncated) and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        step_out = env.step(int(action))

        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
        else:
            obs, reward, done, info = step_out
            terminated, truncated = bool(done), False

        total_reward += float(reward)
        steps += 1
        if isinstance(info, dict):
            last_info = info

    # Prefer explicit env result tag if present; else infer from last reward.
    if isinstance(last_info, dict) and "result" in last_info:
        win = 1 if last_info["result"] == "win" else 0
    else:
        win = 1 if float(reward) > 0 else 0

    return {"seed": seed, "agent": "ppo", "win": win, "return": total_reward, "steps": steps}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="ppo_model/ppo_shootergrid.zip")
    ap.add_argument("--outdir", default="results_ppo")
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--seed_start", type=int, default=10000)
    ap.add_argument("--max_steps", type=int, default=200)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.model_path)
    env = ShooterGridEnv()

    seeds = [args.seed_start + i for i in range(args.n_eval)]
    (outdir / "eval_seeds.txt").write_text("\n".join(map(str, seeds)), encoding="utf-8")

    rows = [run_episode(env, model, seed=s, max_steps=args.max_steps) for s in seeds]
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "episodes_ppo.csv", index=False)

    metrics = {
        "n": [len(df)],
        "win_rate": [df["win"].mean()],
        "mean_return": [df["return"].mean()],
        "std_return": [df["return"].std(ddof=1)],
        "mean_steps": [df["steps"].mean()],
        "std_steps": [df["steps"].std(ddof=1)],
    }
    pd.DataFrame(metrics).to_csv(outdir / "metrics_ppo.csv", index=False)

    print("Wrote:", outdir / "episodes_ppo.csv")
    print("Wrote:", outdir / "metrics_ppo.csv")

if __name__ == "__main__":
    main()
