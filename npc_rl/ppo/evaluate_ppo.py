# npc_rl/ppo/evaluate_ppo.py

from stable_baselines3 import PPO
from npc_rl.env.shootergrid_env import ShooterGridEnv

def evaluate(model_path, episodes=20):
    model = PPO.load(model_path)
    env = ShooterGridEnv(seed=10)

    wins = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # episode ended, check result
        if "result" in info and info["result"] == "win":
            wins += 1

    print(f"PPO win rate: {wins}/{episodes} = {wins/episodes:.2f}")

if __name__ == "__main__":
    evaluate("ppo_model/ppo_shootergrid")
