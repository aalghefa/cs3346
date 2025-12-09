
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from npc_rl.env.shootergrid_env import ShooterGridEnv

def make_env(seed):
    def _init():
        env = ShooterGridEnv(seed=seed)
        env.reset(seed=seed)
        return env
    return _init

def create_ppo_model(seed=0):
    env = DummyVecEnv([make_env(seed)])
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
    )
    return model
