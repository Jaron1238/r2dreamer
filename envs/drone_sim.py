import gymnasium as gym
import numpy as np
from gymnasium import spaces

from reward import DroneRewardFunction


class DroneSimEnv(gym.Env):
    """Deterministic corridor simulator with depth observations.

    The environment emits depth+diff images `(H, W, 2)` and supports an
    injectable reward function.
    """

    metadata = {"render_modes": []}

    def __init__(self, config, reward_fn=None):
        super().__init__()
        h = int(getattr(config.model, "img_height", 256))
        w = int(getattr(config.model, "img_width", 256))
        act_dim = int(getattr(config.model, "act_dim", 4))

        self._h = h
        self._w = w
        self._dt = 0.1
        self._max_steps = 1000
        self._max_speed = 1.0
        self._max_yaw_rate = 1.0
        self._corridor_half_width = 2.0
        self._collision_radius = 0.35
        self._obstacle_spawn_step = 2.5

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0.0, high=1.0, shape=(h, w, 2), dtype=np.float32),
                "is_first": spaces.Box(low=0, high=1, shape=(1,), dtype=np.bool_),
                "speed": spaces.Box(low=-self._max_speed, high=self._max_speed, shape=(1, 1), dtype=np.float32),
            }
        )

        self.reward_fn = reward_fn if reward_fn is not None else DroneRewardFunction(max_speed=self._max_speed)
        self.phase = int(np.clip(int(getattr(config, "phase", 0)), 0, 2))
        if hasattr(self.reward_fn, "set_phase"):
            self.reward_fn.set_phase(self.phase)

        self._step = 0
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._velocity = np.zeros(2, dtype=np.float32)
        self._prev_action = np.zeros(act_dim, dtype=np.float32)
        self._prev_depth = np.ones((h, w, 1), dtype=np.float32)

        self._obstacles = self._build_obstacles()

    def _build_obstacles(self):
        # Deterministic slalom corridor.
        obstacles = []
        x = 8.0
        direction = 1.0
        for _ in range(24):
            y = 0.9 * direction
            obstacles.append({"x": x, "y": y, "dir": -direction})
            x += self._obstacle_spawn_step
            direction *= -1.0
        return obstacles

    def _nearest_obstacle_ahead(self):
        ahead = [o for o in self._obstacles if o["x"] >= self._x]
        if not ahead:
            return {"x": self._x + 100.0, "y": 0.0, "dir": 0.0}
        return min(ahead, key=lambda o: o["x"])

    def _compute_collision(self):
        for obs in self._obstacles:
            dx = obs["x"] - self._x
            if abs(dx) > self._collision_radius:
                continue
            if abs(self._y - obs["y"]) <= self._collision_radius:
                return True
        if abs(self._y) > self._corridor_half_width:
            return True
        return False

    def _render_depth(self):
        nearest = self._nearest_obstacle_ahead()
        dx = max(0.0, nearest["x"] - self._x)
        col = np.linspace(-1.0, 1.0, self._w, dtype=np.float32)
        world_y = self._y + col * 1.4
        wall_clearance = np.clip((self._corridor_half_width - np.abs(world_y)) / self._corridor_half_width, 0.0, 1.0)
        obs_shape = np.exp(-((world_y - nearest["y"]) ** 2) / 0.12).astype(np.float32)
        obs_factor = np.clip(dx / 6.0, 0.0, 1.0)
        depth_line = np.clip(np.minimum(wall_clearance, (1.0 - obs_shape) * obs_factor), 0.0, 1.0)
        depth = np.repeat(depth_line[None, :, None], self._h, axis=0)
        return depth.astype(np.float32), nearest

    def _obs(self, is_first=False):
        depth, nearest = self._render_depth()
        diff = np.clip(depth - self._prev_depth, -1.0, 1.0)
        diff = (diff + 1.0) * 0.5
        image = np.concatenate([depth, diff], axis=-1).astype(np.float32)
        self._prev_depth = depth
        speed = np.array([[float(np.linalg.norm(self._velocity))]], dtype=np.float32)
        obs = {
            "image": image,
            "is_first": np.array([is_first], dtype=np.bool_),
            "speed": speed,
        }
        info = {
            "obstacle_direction": float(nearest["dir"]),
            "obstacle_dx": float(max(0.0, nearest["x"] - self._x)),
            "position": np.array([self._x, self._y], dtype=np.float32),
            "yaw": float(self._yaw),
        }
        return obs, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._velocity = np.zeros(2, dtype=np.float32)
        self._prev_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self._prev_depth = np.ones((self._h, self._w, 1), dtype=np.float32)
        if hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset()
        return self._obs(is_first=True)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._step += 1

        throttle = float((action[0] + 1.0) * 0.5 * self._max_speed)
        yaw_rate = float(action[1] * self._max_yaw_rate) if action.shape[0] > 1 else 0.0

        self._yaw += yaw_rate * self._dt
        heading = np.array([np.cos(self._yaw), np.sin(self._yaw)], dtype=np.float32)
        self._velocity = heading * throttle
        self._x += float(self._velocity[0] * self._dt)
        self._y += float(self._velocity[1] * self._dt)

        collision = self._compute_collision()
        terminated = bool(collision)
        truncated = self._step >= self._max_steps

        reward_parts = self.reward_fn(
            position_xy=np.array([self._x, self._y], dtype=np.float32),
            velocity_xy=self._velocity,
            yaw=float(self._yaw),
            action=action,
            prev_action=self._prev_action,
            collision=collision,
        )
        self._prev_action = action.copy()

        obs, info = self._obs(is_first=False)
        info.update(
            {
                "r_explore": float(reward_parts.r_explore),
                "r_vel": float(reward_parts.r_vel),
                "r_survival": float(reward_parts.r_survival),
                "r_smooth": float(reward_parts.r_smooth),
                "collision": bool(reward_parts.collision),
            }
        )
        return obs, float(reward_parts.total), terminated, truncated, info
