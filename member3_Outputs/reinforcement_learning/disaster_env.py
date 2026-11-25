import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class DisasterEnv(gym.Env):
    """
    Gymnasium Reinforcement Learning Environment for Disaster Emergency Resource Allocation.
    Coordinates logistics (ambulances, buses, boats) to mitigate casualty and displacement risk across zones.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_path=None, max_steps=50):
        super(DisasterEnv, self).__init__()

        if data_path is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.abspath(os.path.join(curr_dir, "..", ".."))
            candidates = [
                os.path.join(base_dir, "member3_Outputs", "data", "final_zone_risk.csv"),
                os.path.join(base_dir, "CV_Spatial_Outputs", "03_disaster_resource_db.csv")
            ]
            for c in candidates:
                if os.path.exists(c) and os.path.getsize(c) > 0:
                    data_path = c
                    break

        if data_path and os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
        else:
            # Fallback synthetic environment state
            self.df = pd.DataFrame({
                "zone_id": [f"Z{i}" for i in range(10)],
                "fused_risk": np.random.uniform(0.1, 0.9, 10),
                "injured_total_Ik": np.random.randint(50, 500, 10),
                "ambulances_Ak": np.random.randint(5, 30, 10)
            })

        self.num_zones = len(self.df)
        self.max_steps = max_steps
        self.current_step = 0

        # Action: Select one priority zone to dispatch emergency response fleet
        self.action_space = spaces.Discrete(self.num_zones)

        # Observation: Feature vector per zone [fused_risk, normalized_injured, available_resources]
        obs_dim = self.num_zones * 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.state = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        risk = self.df["fused_risk"].values if "fused_risk" in self.df.columns else np.random.uniform(0.2, 0.8, self.num_zones)
        inj = self.df["injured_total_Ik"].values if "injured_total_Ik" in self.df.columns else np.ones(self.num_zones) * 100
        inj_norm = (inj / (inj.max() + 1e-5)).astype(np.float32)
        res = np.ones(self.num_zones, dtype=np.float32) * 0.5

        self.zone_risks = risk.astype(np.float32)
        self.zone_inj = inj_norm.astype(np.float32)
        self.zone_res = res.astype(np.float32)

        obs = np.concatenate([self.zone_risks, self.zone_inj, self.zone_res]).astype(np.float32)
        self.state = obs
        return obs, {}

    def step(self, action):
        self.current_step += 1
        zone_idx = int(action)

        # Dispatch relief: reduce injury burden and risk for chosen zone
        dispatched_benefit = self.zone_risks[zone_idx] * 0.35 + self.zone_inj[zone_idx] * 0.40
        self.zone_inj[zone_idx] = max(0.0, self.zone_inj[zone_idx] - 0.25)
        self.zone_risks[zone_idx] = max(0.05, self.zone_risks[zone_idx] - 0.15)
        self.zone_res[zone_idx] = min(1.0, self.zone_res[zone_idx] + 0.2)

        # Remaining penalty across all unaddressed high risk zones
        penalty = 0.1 * np.mean(self.zone_risks)
        reward = float(dispatched_benefit - penalty)

        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = np.concatenate([self.zone_risks, self.zone_inj, self.zone_res]).astype(np.float32)
        self.state = obs
        return obs, reward, terminated, truncated, {"dispatched_zone": zone_idx}

    def render(self):
        print(f"Step: {self.current_step} | Mean Zone Risk: {np.mean(self.zone_risks):.4f}")

