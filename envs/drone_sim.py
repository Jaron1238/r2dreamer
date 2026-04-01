import gymnasium as gym
import numpy as np
import cv2
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
        self._prev_depth = None

        self._obstacles = self._build_obstacles()
        self._colosseum = ColosseumBridge(config)
        if self._colosseum.enabled:
            self._colosseum.connect()

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
        if self._colosseum.enabled:
            image, speed, info = self._colosseum.read_observation(self._h, self._w, is_first=is_first)
            return {"image": image, "is_first": np.array([is_first], dtype=np.bool_), "speed": speed}, info
        depth, nearest = self._render_depth()
        if self._prev_depth is None:
            diff = np.zeros_like(depth)
        else:
            diff = np.clip(depth - self._prev_depth, -1.0, 1.0)
            diff = (diff + 1.0) * 0.5
        image = np.concatenate([depth, diff], axis=-1).astype(np.float32)
        self._prev_depth = depth.copy()
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
            "clearance": float(np.median(depth[..., 0])),
            "clearance_ema": float(np.median(depth[..., 0])),
        }
        return obs, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._colosseum.enabled:
            self._colosseum.reset()
            return self._obs(is_first=True)
        self._step = 0
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._velocity = np.zeros(2, dtype=np.float32)
        self._prev_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self._prev_depth = None
        if hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset()
        return self._obs(is_first=True)

    def step(self, action):
        if self._colosseum.enabled:
            return self._colosseum.step(action, self.reward_fn, self._max_steps)
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._step += 1

        roll_cmd = float(action[0]) if action.shape[0] > 0 else 0.0
        pitch_cmd = float(action[1]) if action.shape[0] > 1 else 0.0
        yaw_rate = float(action[2] * self._max_yaw_rate) if action.shape[0] > 2 else (
            float(action[1] * self._max_yaw_rate) if action.shape[0] > 1 else 0.0
        )
        throttle = float((action[3] + 1.0) * 0.5 * self._max_speed) if action.shape[0] > 3 else float(
            (action[0] + 1.0) * 0.5 * self._max_speed
        )

        self._yaw += yaw_rate * self._dt
        body_vel = np.array([pitch_cmd, roll_cmd], dtype=np.float32) * throttle
        rot = np.array(
            [[np.cos(self._yaw), -np.sin(self._yaw)], [np.sin(self._yaw), np.cos(self._yaw)]],
            dtype=np.float32,
        )
        self._velocity = rot @ body_vel
        self._x += float(self._velocity[0] * self._dt)
        self._y += float(self._velocity[1] * self._dt)

        collision = self._compute_collision()
        terminated = bool(collision)
        truncated = self._step >= self._max_steps

        obs, info = self._obs(is_first=False)
        reward_parts = self.reward_fn(
            position_xy=np.array([self._x, self._y], dtype=np.float32),
            velocity_xy=self._velocity,
            yaw=float(self._yaw),
            action=action,
            prev_action=self._prev_action,
            collision=collision,
            clearance=info.get("clearance"),
            clearance_ema=info.get("clearance_ema"),
        )
        self._prev_action = action.copy()
        info.update(
            {
                "r_explore": float(reward_parts.r_explore),
                "r_vel": float(reward_parts.r_vel),
                "r_survival": float(reward_parts.r_survival),
                "r_smooth": float(reward_parts.r_smooth),
                "r_height": float(reward_parts.r_height),
                "collision": bool(reward_parts.collision),
            }
        )
        return obs, float(reward_parts.total), terminated, truncated, info


class ColosseumBridge:
    def __init__(self, config):
        env_cfg = getattr(config, "env", config)
        colosseum_cfg = getattr(env_cfg, "colosseum", None)
        self.enabled = bool(getattr(colosseum_cfg, "enabled", False))
        self.host = str(getattr(colosseum_cfg, "host", "127.0.0.1"))
        self.vehicle_name = str(getattr(colosseum_cfg, "vehicle_name", ""))
        self._client = None
        self._step = 0
        model_cfg = getattr(config, "model", config)
        self.obs_h = int(getattr(model_cfg, "img_height", 256))
        self.obs_w = int(getattr(model_cfg, "img_width", 256))
        self._prev_depth = None
        self._prev_action = np.zeros(int(getattr(model_cfg, "act_dim", 4)), dtype=np.float32)
        self.down_camera_name = str(getattr(colosseum_cfg, "down_camera_name", "downward"))
        self.clearance_ema_alpha = float(getattr(colosseum_cfg, "clearance_ema_alpha", 0.2))
        self.max_clearance = float(getattr(colosseum_cfg, "max_clearance", 20.0))
        self._clearance_ema = None

    def connect(self):
        if not self.enabled:
            return
        try:
            import airsim

            self._client = airsim.MultirotorClient(ip=self.host)
            self._client.confirmConnection()
            self._client.enableApiControl(True, vehicle_name=self.vehicle_name)
            self._client.armDisarm(True, vehicle_name=self.vehicle_name)
        except Exception:
            self.enabled = False
            self._client = None

    def reset(self):
        if self._client is None:
            return
        self._client.reset()
        self._client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self._client.armDisarm(True, vehicle_name=self.vehicle_name)
        self._step = 0
        self._prev_depth = None
        self._prev_action = np.zeros_like(self._prev_action)
        self._clearance_ema = None

    def _yaw_from_quat(self, q) -> float:
        x, y, z, w = float(q.x_val), float(q.y_val), float(q.z_val), float(q.w_val)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def read_observation(self, h, w, *, is_first):
        import airsim

        req = airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, pixels_as_float=True)
        img = self._client.simGetImages([req], vehicle_name=self.vehicle_name)[0]
        if img.height <= 0 or img.width <= 0 or len(img.image_data_float) == 0:
            depth = np.ones((h, w, 1), dtype=np.float32)
        else:
            depth = np.array(img.image_data_float, dtype=np.float32).reshape(img.height, img.width, 1)
            depth = np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0)
            p2, p98 = np.percentile(depth, [2, 98])
            depth = np.clip((depth - p2) / max(float(p98 - p2), 1e-6), 0.0, 1.0)
        depth = cv2.resize(depth[..., 0], (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
        if self._prev_depth is None:
            diff = np.zeros_like(depth)
        else:
            diff = np.clip(depth - self._prev_depth, -1.0, 1.0)
            diff = (diff + 1.0) * 0.5
        self._prev_depth = depth.copy()
        image = np.concatenate([depth, diff], axis=-1).astype(np.float32)
        kin = self._client.simGetGroundTruthKinematics(vehicle_name=self.vehicle_name)
        vel = np.array([kin.linear_velocity.x_val, -kin.linear_velocity.z_val], dtype=np.float32)
        speed = np.array([[float(np.linalg.norm(vel))]], dtype=np.float32)
        yaw = self._yaw_from_quat(kin.orientation)
        clearance = self._read_downward_clearance()
        if self._clearance_ema is None:
            self._clearance_ema = clearance
        else:
            self._clearance_ema = (1.0 - self.clearance_ema_alpha) * self._clearance_ema + self.clearance_ema_alpha * clearance
        info = {
            "position": np.array([kin.position.x_val, -kin.position.z_val], dtype=np.float32),
            "yaw": yaw,
            "speed": speed,
            "clearance": float(clearance),
            "clearance_ema": float(self._clearance_ema),
            "terrain_rel_height": float(self._clearance_ema),
        }
        return image, speed, info

    def _read_downward_clearance(self) -> float:
        import airsim

        req = airsim.ImageRequest(self.down_camera_name, airsim.ImageType.DepthPlanar, pixels_as_float=True)
        img = self._client.simGetImages([req], vehicle_name=self.vehicle_name)[0]
        if img.height <= 0 or img.width <= 0 or len(img.image_data_float) == 0:
            return self.max_clearance
        depth = np.array(img.image_data_float, dtype=np.float32)
        depth = np.nan_to_num(depth, nan=self.max_clearance, posinf=self.max_clearance, neginf=0.0)
        valid = depth[(depth > 0.05) & np.isfinite(depth)]
        if valid.size == 0:
            return self.max_clearance
        return float(np.clip(np.median(valid), 0.0, self.max_clearance))

    def step(self, action, reward_fn, max_steps):
        import airsim

        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        roll_rate = float(action[0]) if action.shape[0] > 0 else 0.0
        pitch_rate = float(action[1]) if action.shape[0] > 1 else 0.0
        yaw_rate = float(action[2]) if action.shape[0] > 2 else 0.0
        throttle = float((action[3] + 1.0) * 0.5) if action.shape[0] > 3 else float((action[0] + 1.0) * 0.5)
        throttle = float(np.clip(throttle, 0.0, 1.0))
        self._client.moveByAngleRatesThrottleAsync(
            roll_rate, pitch_rate, yaw_rate, throttle, 0.1, vehicle_name=self.vehicle_name
        ).join()
        kin = self._client.simGetGroundTruthKinematics(vehicle_name=self.vehicle_name)
        vel = np.array([kin.linear_velocity.x_val, -kin.linear_velocity.z_val], dtype=np.float32)
        collision = bool(self._client.simGetCollisionInfo(vehicle_name=self.vehicle_name).has_collided)
        yaw = self._yaw_from_quat(kin.orientation)
        obs, speed, info = self.read_observation(self.obs_h, self.obs_w, is_first=False)
        reward_parts = reward_fn(
            position_xy=np.array([kin.position.x_val, -kin.position.z_val], dtype=np.float32),
            velocity_xy=vel,
            yaw=yaw,
            action=action,
            prev_action=self._prev_action,
            collision=collision,
            clearance=info.get("clearance"),
            clearance_ema=info.get("clearance_ema"),
        )
        self._prev_action = action.copy()
        self._step += 1
        terminated = collision
        truncated = self._step >= max_steps
        info.update(
            {
                "r_explore": float(reward_parts.r_explore),
                "r_vel": float(reward_parts.r_vel),
                "r_survival": float(reward_parts.r_survival),
                "r_smooth": float(reward_parts.r_smooth),
                "r_height": float(reward_parts.r_height),
                "collision": bool(reward_parts.collision),
            }
        )
        return {"image": obs, "is_first": np.array([False], dtype=np.bool_), "speed": speed}, float(reward_parts.total), terminated, truncated, info
