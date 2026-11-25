import sys
from pathlib import Path
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from disaster_env import DisasterEnv

env = DisasterEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64
)

model.learn(total_timesteps=100000)

model.save("outputs/ppo_disaster_agent")