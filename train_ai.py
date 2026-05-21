import argparse
import os
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from farming_ai import FARMING_ACTION_COUNT, FARMING_MODEL_NAME, FARMING_OBSERVATION_SIZE
from rl_env import DiepEnv

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_STEM = SCRIPT_DIR / FARMING_MODEL_NAME
MODEL_PATH = MODEL_STEM.with_suffix(".zip")
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"


class GymDiepEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, curriculum=True):
        super().__init__()
        self.env = DiepEnv(curriculum=curriculum)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(FARMING_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(FARMING_ACTION_COUNT)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(int(action))
        truncated = bool(info.pop("truncated", False))
        return np.array(obs, dtype=np.float32), float(reward), done, truncated, info


def parse_args():
    parser = argparse.ArgumentParser(description="Train the maze farming bot policy.")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total PPO timesteps.")
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel environments.")
    parser.add_argument("--fresh", action="store_true", help="Start a new farming model instead of continuing it.")
    parser.add_argument("--no-curriculum", action="store_true", help="Train only on full maze episodes.")
    parser.add_argument("--progress", action="store_true", help="Show the optional SB3 progress bar.")
    return parser.parse_args()


def build_model(vec_env):
    return PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=512,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        learning_rate=2.5e-4,
        clip_range=0.2,
        target_kl=0.04,
        policy_kwargs={"net_arch": [128, 128]},
    )


if __name__ == "__main__":
    args = parse_args()
    print(f"Starting farming AI training: {FARMING_MODEL_NAME}")
    print(f"timesteps={args.timesteps}, n_envs={args.n_envs}, curriculum={not args.no_curriculum}")

    vec_env = make_vec_env(
        lambda: GymDiepEnv(curriculum=not args.no_curriculum),
        n_envs=args.n_envs,
    )

    if MODEL_PATH.exists() and not args.fresh:
        print(f"Continuing existing model: {MODEL_PATH.name}")
        model = PPO.load(str(MODEL_PATH), env=vec_env)
    else:
        print("Creating a new v3 model.")
        model = build_model(vec_env)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // args.n_envs, 1),
        save_path=str(CHECKPOINT_DIR),
        name_prefix=FARMING_MODEL_NAME,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback, progress_bar=args.progress)
    model.save(str(MODEL_STEM))
    print(f"Training finished and saved: {MODEL_PATH.name}")
