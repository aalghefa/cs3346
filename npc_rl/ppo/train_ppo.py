
import os
from stable_baselines3 import PPO
from npc_rl.ppo.ppo_agent import create_ppo_model

SAVE_PATH = "ppo_model"

def main():
    model = create_ppo_model(seed=1)
    model.learn(total_timesteps=300000)
    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save(f"{SAVE_PATH}/ppo_shootergrid")
    print("Training complete, model saved.")

if __name__ == "__main__":
    main()
