import sys
from pathlib import Path
from stable_baselines3 import DQN

sys.path.insert(0, str(Path(__file__).resolve().parent))
from disaster_env import DisasterEnv

env = DisasterEnv()

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,
    buffer_size=50000
)

model.learn(total_timesteps=100000)

model.save("outputs/dqn_disaster_agent")