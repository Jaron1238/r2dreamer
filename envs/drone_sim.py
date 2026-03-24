import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DroneSimEnv(gym.Env):
    """Minimal gymnasium adapter for AirSim/Gazebo-style drone control.

    This environment is intentionally lightweight and can be replaced with
    a real simulator bridge in phase 3.
    """

    metadata = {"render_modes": []}

    def __init__(self, config):
        super().__init__()
        h = int(getattr(config.model, "img_height", 256))
        w = int(getattr(config.model, "img_width", 256))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(int(config.model.act_dim),), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0.0, high=1.0, shape=(1, h, w, 2), dtype=np.float32),
                "is_first": spaces.Box(low=0, high=1, shape=(1,), dtype=np.bool_),
                "speed": spaces.Box(low=-100.0, high=100.0, shape=(1, 1), dtype=np.float32),
            }
        )
        self._step = 0

    def _dummy_obs(self, is_first=False):
        h, w = self.observation_space["image"].shape[1:3]
        depth = np.random.rand(1, h, w, 1).astype(np.float32)
        diff = np.zeros_like(depth)
        return {
            "image": np.concatenate([depth, diff], axis=-1),
            "is_first": np.array([is_first], dtype=np.bool_),
            "speed": np.zeros((1, 1), dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return self._dummy_obs(is_first=True), {}

    def step(self, action):
        self._step += 1
        # Placeholder reward: forward-speed bonus + survival + tiny exploration term.
        forward_speed = float(np.clip(action[0], -1.0, 1.0))
        reward = 0.7 * max(0.0, forward_speed) + 0.2 + 0.1 * np.random.rand()
        terminated = False
        truncated = self._step >= 1000
        return self._dummy_obs(is_first=False), float(reward), terminated, truncated, {}
