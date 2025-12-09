#Jonathan Kang
#CS3346
# npc_rl/env/run_env_test.py

import os
from npc_rl.env.shootergrid_env import ShooterGridEnv
import numpy as n

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.yaml")

def main():
    env = ShooterGridEnv(config_path)

    for ep in range(3):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            a1 = env.action_space.sample()
            a2 = env.action_space.sample()
            obs, reward, done, _, _ = env.step((a1, a2))
            total_reward += reward

        print(f"Episode {ep + 1} total reward: {total_reward}")


if __name__ == "__main__":
    env = ShooterGridEnv()
    obs, _ = env.reset()
    for _ in range(20):
        action = env.sample_action()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(info)
            break
