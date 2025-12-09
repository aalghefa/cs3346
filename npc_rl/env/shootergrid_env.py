#Jonathan Kang
#CS3346
# npc_rl/env/shootergrid_env.py

import os
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ShooterGridEnv(gym.Env):
    """
    Simple 2-player grid combat environment.

    Actions (for each player):
        0 = stay
        1 = move up
        2 = move down
        3 = move left
        4 = move right
        5 = shoot

    Observation:
        [p1_x, p1_y, p1_hp, p2_x, p2_y, p2_hp]
    """

    metadata = {"render_modes": []}

    def __init__(self, config_path=None, seed=None):
        super().__init__()

        if config_path is None:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(this_dir, "config.yaml")

        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.grid_size = int(self.cfg["grid_size"])
        self.max_steps = int(self.cfg["max_steps"])
        self.shoot_prob = float(self.cfg["shoot_probability"])
        base_seed = int(self.cfg["seed"])

        if seed is not None:
            base_seed = seed

        self.rng = np.random.default_rng(base_seed)

        # --- REQUIRED by Gym ---
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size,
            shape=(6,),
            dtype=np.int32
        )

        self.action_space = spaces.Discrete(6)

        self.reset()


    # ------------- helpers -------------

    def _move(self, pos, action):
        x, y = pos
        if action == 1:   # up
            y -= 1
        elif action == 2: # down
            y += 1
        elif action == 3: # left
            x -= 1
        elif action == 4: # right
            x += 1

        x = np.clip(x, 0, self.grid_size - 1)
        y = np.clip(y, 0, self.grid_size - 1)
        return np.array([x, y], dtype=np.int32)

    def _line_of_sight(self, a, b):
        """Very simple LOS: same row or column."""
        return a[0] == b[0] or a[1] == b[1]

    def _get_obs(self):
        return np.array(
            [*self.p1_pos, self.p1_hp, *self.p2_pos, self.p2_hp],
            dtype=np.int32,
        )

    # ------------- gym API -------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # re-seed RNG if a seed is passed
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.p1_pos = np.array([0, 0], dtype=np.int32)
        self.p2_pos = np.array(
            [self.grid_size - 1, self.grid_size - 1],
            dtype=np.int32,
        )
        self.p1_hp = 3
        self.p2_hp = 3

        return self._get_obs(), {}

    def step(self, actions):
        # actions is a single int for our agent
        a1 = int(actions)
        a2 = self.sample_enemy()

        reward = 0.0
        info = {}
        terminated = False

        # distance BEFORE movement
        prev_dist = np.linalg.norm(self.p1_pos - self.p2_pos)

        # movement
        self.p1_pos = self._move(self.p1_pos, a1)
        self.p2_pos = self._move(self.p2_pos, a2)

        # 1) small reward for actually moving (not camping)
        if a1 in [1, 2, 3, 4]:
            reward += 0.01

        # 2) SHOOTING (agent)
        if a1 == 5 and self._line_of_sight(self.p1_pos, self.p2_pos):
            # small bonus for shooting while in LOS
            reward += 0.1
            if self.rng.random() < self.shoot_prob:
                # big bonus for a hit
                self.p2_hp -= 1
                reward += 1.0

        # 3) SHOOTING (enemy)
        if a2 == 5 and self._line_of_sight(self.p2_pos, self.p1_pos):
            if self.rng.random() < self.shoot_prob:
                self.p1_hp -= 1
                reward -= 0.5  # getting hit is bad

        # 4) distance AFTER movement: encourage closing distance
        dist = np.linalg.norm(self.p1_pos - self.p2_pos)
        if dist < prev_dist:
            reward += 0.05

        # 5) tiny survival reward to keep exploring
        reward += 0.001

        # 6) terminal conditions + result flag
        if self.p1_hp <= 0:
            reward += -1.0        # add on top of shaping
            terminated = True
            info["result"] = "loss"
        elif self.p2_hp <= 0:
            reward += 1.0         # add on top of shaping
            terminated = True
            info["result"] = "win"

        # 7) step limit -> draw if nobody died
        self.steps += 1
        if self.steps >= self.max_steps and not terminated:
            terminated = True
            info["result"] = "draw"

        obs = self._get_obs()
        return obs, reward, terminated, False, info

    def sample_action(self):
        return self.action_space.sample()

    def sample_enemy(self):
        return self.action_space.sample()
