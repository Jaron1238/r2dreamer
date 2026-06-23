import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces

from reward import DroneRewardFunction

class DroneSimEnv(gym.Env):
    

    metadata = {"render_modes": []}

    def __init__(self, config, reward_fn=None):
        super().__init__()
        h = int(getattr(config.model, "img_height", 256))
        w = int(getattr(config.model, "img_width", 256))
        act_dim = int(getattr(config.model, "act_dim", 4))

        self._h = h
        self._w = w
        self._dt = 1.0 / 60.0  # 60 FPS
        self._max_steps = 1000
        self._max_speed = 1.0
        self._max_yaw_rate = 1.0
        self._corridor_half_width = 2.0
        self._collision_radius = 0.35
        self._obstacle_spawn_step = 2.5

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        
        
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0.0, high=1.0, shape=(h, w, 6), dtype=np.float32),
                "is_first": spaces.Box(low=0, high=1, shape=(1,), dtype=np.bool_),
                "speed": spaces.Box(low=-self._max_speed, high=self._max_speed, shape=(1, 1), dtype=np.float32),
            }
        )

        self.reward_fn = reward_fn if reward_fn is not None else DroneRewardFunction(max_speed=self._max_speed)
        # Bug #9 fix: decouple reward curriculum phase from global training phase.
        # config.phase is the Dreamer training phase (1-4, always >=3 here),
        # which would always clip to 2 and freeze the curriculum at its most aggressive preset.
        # Use an independent config.env.reward_curriculum_phase (0-2, default 0) instead.
        env_cfg = getattr(config, "env", None)
        _curriculum_phase = int(getattr(env_cfg, "reward_curriculum_phase", 0) if env_cfg is not None else 0)
        self.reward_curriculum_phase = int(np.clip(_curriculum_phase, 0, 2))
        if hasattr(self.reward_fn, "set_phase"):
            self.reward_fn.set_phase(self.reward_curriculum_phase)

        self._step = 0
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._velocity = np.zeros(2, dtype=np.float32)
        self._prev_action = np.zeros(act_dim, dtype=np.float32)
        self._prev_rgb = None   

        self._obstacles = self._build_obstacles()
        self._colosseum = ColosseumBridge(config)
        if self._colosseum.enabled:
            self._colosseum.connect()

    def _build_obstacles(self):
        
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

    def _compute_expert_evasion(self, nearest: dict, dx: float) -> np.ndarray:
        

        act = np.zeros(self.action_space.shape[0], dtype=np.float32)
        if dx < 1e-3:
            return act

        
        
        lateral_err = float(nearest["y"] - self._y)
        evasion_lateral = -np.clip(lateral_err / self._corridor_half_width, -1.0, 1.0)

        
        
        fwd_scale = float(np.clip(dx / (self._obstacle_spawn_step * 2.0), 0.0, 1.0))

        n = act.shape[0]
        if n >= 4:
            act[0] = evasion_lateral   
            act[1] = fwd_scale         
            act[2] = 0.0               
            act[3] = 0.0               
        elif n == 2:
            act[0] = fwd_scale
            act[1] = evasion_lateral
        elif n == 1:
            act[0] = evasion_lateral

        return np.clip(act, -1.0, 1.0)

    def _render_rgb(self):
        

        nearest = self._nearest_obstacle_ahead()
        dx = max(0.0, nearest["x"] - self._x)

        H, W = self._h, self._w
        # Normalised column coordinates in [-1, 1]
        col = np.linspace(-1.0, 1.0, W, dtype=np.float32)
        # Normalised row coordinates in [-1, 1]  (top = -1, bottom = +1)
        row = np.linspace(-1.0, 1.0, H, dtype=np.float32)
        col_grid, row_grid = np.meshgrid(col, row)          # (H, W)

        # ── Background: sky (top) / ground (bottom) ────────────────────────
        sky_mask = row_grid < 0.0                           # upper half
        r = np.where(sky_mask, 0.53 - 0.1 * row_grid, 0.45 + 0.15 * row_grid).astype(np.float32)
        g = np.where(sky_mask, 0.81 - 0.1 * row_grid, 0.30 - 0.10 * row_grid).astype(np.float32)
        b = np.where(sky_mask, 0.98 - 0.05 * row_grid, 0.15).astype(np.float32)

        # ── Corridor walls ─────────────────────────────────────────────────
        # Map pixel column to world-Y (lateral position in corridor).
        world_y = self._y + col_grid * 1.4
        wall_dist = self._corridor_half_width - np.abs(world_y)     # (H, W)
        wall_mask = wall_dist < 0.35
        r[wall_mask] = 0.20
        g[wall_mask] = 0.20
        b[wall_mask] = 0.22

        # ── Obstacle blob ─────────────────────────────────────────────────
        # Blob lateral centre in normalised column space
        obs_col_centre = float((nearest["y"] - self._y) / 1.4)     # in [-1, 1]
        # Blob size grows as dx → 0; disappears beyond 8 m
        blob_radius_col = float(np.clip((1.0 - dx / 8.0) * 0.25, 0.0, 0.35))
        blob_radius_row = blob_radius_col * 1.6
        blob_dist = np.sqrt(
            ((col_grid - obs_col_centre) / max(blob_radius_col, 1e-6)) ** 2
            + ((row_grid - 0.1) / max(blob_radius_row, 1e-6)) ** 2
        )
        obs_mask = (blob_dist < 1.0) & (dx < 8.0)
        # Smooth blob edge via a cosine falloff
        alpha_obs = np.clip((1.0 - blob_dist), 0.0, 1.0) * obs_mask
        r = r * (1.0 - alpha_obs) + 0.85 * alpha_obs
        g = g * (1.0 - alpha_obs) + 0.15 * alpha_obs
        b = b * (1.0 - alpha_obs) + 0.10 * alpha_obs

        # ── Motion-blur stripe: subtle horizontal smear in yaw direction ──
        yaw_shift = int(np.clip(self._yaw * W * 0.05, -W // 8, W // 8))
        if yaw_shift != 0:
            r = np.roll(r, yaw_shift, axis=1) * 0.3 + r * 0.7
            g = np.roll(g, yaw_shift, axis=1) * 0.3 + g * 0.7
            b = np.roll(b, yaw_shift, axis=1) * 0.3 + b * 0.7

        rgb = np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0).astype(np.float32)  # (H, W, 3)
        return rgb, nearest

    def _obs(self, is_first=False):
        if self._colosseum.enabled:
            image, speed, info, _stale = self._colosseum.read_observation(self._h, self._w, is_first=is_first)  # Bug #10: is_stale ignored in reset path
            return {"image": image, "is_first": np.array([is_first], dtype=np.bool_), "speed": speed}, info

        rgb, nearest = self._render_rgb()                    # (H, W, 3) float32 [0,1]

        # Temporal diff: zero-padded on the first frame, clamped to [0,1]
        if self._prev_rgb is None:
            diff = np.zeros_like(rgb)
        else:
            diff = np.clip((rgb - self._prev_rgb + 1.0) * 0.5, 0.0, 1.0)  # centre at 0.5

        # 6-channel image: [R,G,B, dR,dG,dB]  — matches RSSM shapes["image"]=(H,W,6)
        image = np.concatenate([rgb, diff], axis=-1).astype(np.float32)
        self._prev_rgb = rgb.copy()

        speed = np.array([[float(np.linalg.norm(self._velocity))]], dtype=np.float32)
        obs = {
            "image": image,
            "is_first": np.array([is_first], dtype=np.bool_),
            "speed": speed,
        }
        obs_dx = float(max(0.0, nearest["x"] - self._x))
        # Use mean luminance of RGB as a proxy for "clearance" signal
        clearance = float(np.clip(obs_dx / 8.0, 0.0, 1.0))
        info = {
            "obstacle_direction": float(nearest["dir"]),
            "obstacle_dx": obs_dx,
            "position": np.array([self._x, self._y], dtype=np.float32),
            "yaw": float(self._yaw),
            "clearance": clearance,
            "clearance_ema": clearance,
            "expert_evasion": self._compute_expert_evasion(nearest, obs_dx),
            "expert_active": float(obs_dx < 2.0),
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
        self._prev_rgb = None
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

# ──────────────────────────────────────────────────────────────────────────────
# Shared Memory IPC — ersetzt alle airsim-RPC-Calls.
# Layout muss identisch mit ShmIPCLayout.h in Colosseum sein.
# ──────────────────────────────────────────────────────────────────────────────
import ctypes
import mmap as _mmap

_IPC_MAGIC  = 0xC0105EAD
_SHM_NAME   = "/colosseum_ipc"
_SEM_UE     = "/colosseum_ue_wrote"   # UE postet nach jedem Frame
_SEM_PY     = "/colosseum_py_wrote"   # Python postet nach jedem Kommando
_SHM_SIZE   = 32 * 1024 * 1024        # 32 MB
_IMAGE_OFF  = 4096                     # Byte-Offset des RGB-Bildes im SHM

# Kommando-Flags (identisch mit COLOSSEUM_CMD_* in ShmIPCLayout.h)
_CMD_NONE  = 0
_CMD_RESET = 1
_CMD_ESTOP = 2

class _IPCHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic",    ctypes.c_uint32),
        ("version",  ctypes.c_uint32),
        ("seq_ue",   ctypes.c_uint64),
        ("seq_py",   ctypes.c_uint64),
        ("sim_time", ctypes.c_double),
        ("ue_ready", ctypes.c_uint8),
        ("py_ready", ctypes.c_uint8),
        ("_pad",     ctypes.c_uint8 * 6),
    ]

class _SensorBlock(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("pos_x",        ctypes.c_float),
        ("pos_y",        ctypes.c_float),
        ("pos_z",        ctypes.c_float),
        ("vel_x",        ctypes.c_float),
        ("vel_y",        ctypes.c_float),
        ("vel_z",        ctypes.c_float),
        ("ori_w",        ctypes.c_float),
        ("ori_x",        ctypes.c_float),
        ("ori_y",        ctypes.c_float),
        ("ori_z",        ctypes.c_float),
        ("has_collided", ctypes.c_uint8),
        ("_pad",         ctypes.c_uint8 * 3),
        ("cam_width",    ctypes.c_uint32),
        ("cam_height",   ctypes.c_uint32),
        ("depth_median", ctypes.c_float),
    ]

class _CommandBlock(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("roll_rate",  ctypes.c_float),
        ("pitch_rate", ctypes.c_float),
        ("yaw_rate",   ctypes.c_float),
        ("throttle",   ctypes.c_float),
        ("duration",   ctypes.c_float),
        ("flag",       ctypes.c_uint8),
        ("_pad",       ctypes.c_uint8 * 3),
    ]

class _IPCBlock(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("header",  _IPCHeader),
        ("sensors", _SensorBlock),
        ("command", _CommandBlock),
    ]

class ColosseumBridge:
    """Verbindet r2dreamer mit Colosseum via POSIX Shared Memory.

    Ersetzt alle airsim-RPC-Aufrufe durch Zero-Copy IPC:
      • UE schreibt Sensordaten + RGB-Bild → postet SEM_UE
      • Python liest Daten → schreibt Kommando → postet SEM_PY
      • UE liest Kommando → führt Steuerung aus

    macOS-Hinweis: Nur Named Semaphores (sem_open) werden verwendet.
    sem_init() ist auf macOS ein Stub und darf NICHT benutzt werden.
    """

    def __init__(self, config):
        col = getattr(config, "colosseum", None)
        self.enabled             = bool(getattr(col, "enabled", False)) if col is not None else False
        self._shm_name           = getattr(col, "shm_name", _SHM_NAME) if col is not None else _SHM_NAME
        self._sem_ue_name        = getattr(col, "sem_ue",   _SEM_UE)   if col is not None else _SEM_UE
        self._sem_py_name        = getattr(col, "sem_py",   _SEM_PY)   if col is not None else _SEM_PY
        self.max_clearance       = float(getattr(col, "max_clearance",      20.0) if col is not None else 20.0)
        self.clearance_ema_alpha = float(getattr(col, "clearance_ema_alpha", 0.2) if col is not None else 0.2)
        self.obs_h = int(getattr(config.model, "img_height", 256))
        self.obs_w = int(getattr(config.model, "img_width",  256))
        # Runtime state — populated in connect()
        self._mm          = None
        self._ipc         = None
        self._sem_ue_obj  = None
        self._sem_py_obj  = None
        self._step        = 0
        self._prev_rgb    = None
        self._clearance_ema = None
        self._last_seq_ue = 0
        self._prev_action = np.zeros(4, dtype=np.float32)
        # Bug #10: stale-frame tracking
        self._stale_count = 0
        self._max_stale_frames = 5

    def connect(self, retries=10, delay=2.0):
        """Wartet bis UE den SHM-Block erstellt hat, dann verbinden."""
        if not self.enabled:
            return
        try:
            import posix_ipc
        except ImportError:
            print("[ColosseumBridge] FEHLER: posix_ipc fehlt → pip install posix-ipc")
            self.enabled = False
            return

        for attempt in range(retries):
            try:
                # flags=0: nur öffnen, UE ist der Creator
                shm = posix_ipc.SharedMemory(self._shm_name, flags=0)
                mm  = _mmap.mmap(shm.fd, _SHM_SIZE, access=_mmap.ACCESS_WRITE)
                shm.close_fd()

                ipc = _IPCBlock.from_buffer(mm)
                if ipc.header.magic != _IPC_MAGIC:
                    mm.close()
                    raise ValueError(
                        f"IPC Magic falsch: 0x{ipc.header.magic:08X} "
                        f"(erwartet 0x{_IPC_MAGIC:08X})"
                    )

                # Semaphoren öffnen (bereits von UE erstellt)
                sem_ue = posix_ipc.Semaphore(self._sem_ue_name)
                sem_py = posix_ipc.Semaphore(self._sem_py_name)

                self._mm         = mm
                self._ipc        = ipc
                self._sem_ue_obj = sem_ue
                self._sem_py_obj = sem_py
                print(f"[ColosseumBridge] Verbunden via '{self._shm_name}' "
                      f"(IPC v{ipc.header.version})")
                return

            except Exception as e:
                print(f"[ColosseumBridge] Warte auf UE... ({attempt + 1}/{retries}): {e}")
                import time
                time.sleep(delay)

        print("[ColosseumBridge] Verbindung fehlgeschlagen — Bridge deaktiviert.")
        self.enabled = False

    def disconnect(self):
        """Ressourcen freigeben. SHM wird NICHT gelöscht (Sache von UE)."""
        for attr in ("_sem_ue_obj", "_sem_py_obj"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        if self._mm is not None:
            try:
                self._mm.close()
            except Exception:
                pass
            self._mm = None
        self._ipc = None

    def reset(self):
        """Sendet Reset-Kommando an UE und setzt lokalen Zustand zurück."""
        if self._ipc is None:
            return
        self._write_command(0, 0, 0, 0, 0.1, flag=_CMD_RESET)
        # Veraltete SEM_UE-Posts aus dem vorherigen Episode-End wegdrainieren,
        # damit read_observation nach dem Reset nicht sofort mit alten Daten zurückkehrt.
        try:
            while True:
                self._sem_ue_obj.acquire(timeout=0.0)
        except Exception:
            pass
        self._step          = 0
        self._prev_rgb      = None
        self._clearance_ema = None
        self._last_seq_ue   = 0
        self._prev_action[:] = 0.0

    # ── Beobachtung lesen ─────────────────────────────────────────────────────

    def read_observation(self, h: int, w: int, *, is_first: bool):
        """Wartet auf neuen UE-Frame und gibt (image6ch, speed, info) zurück."""
        if self._ipc is None:
            return self._fallback_obs(h, w, is_first)  # 4-tuple with is_stale=True

        # Auf neuen Frame warten (max. 100 ms)
        try:
            self._sem_ue_obj.acquire(timeout=0.1)
        except Exception:
            return self._fallback_obs(h, w, is_first)

        seq = self._ipc.header.seq_ue
        if seq == self._last_seq_ue:
            # UE hat noch keinen neuen Frame geschrieben — stale Frame vermeiden
            return self._fallback_obs(h, w, is_first)
        self._last_seq_ue = seq

        s        = self._ipc.sensors
        sim_time = self._ipc.header.sim_time

        # ── RGB-Bild aus SHM lesen ─────────────────────────────────────────
        cw, ch = int(s.cam_width), int(s.cam_height)
        if cw > 0 and ch > 0:
            n = cw * ch * 3
            self._mm.seek(_IMAGE_OFF)
            raw = np.frombuffer(self._mm.read(n), dtype=np.uint8).reshape(ch, cw, 3)
            rgb = cv2.resize(raw, (w, h),
                             interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        else:
            rgb = np.zeros((h, w, 3), dtype=np.float32)

        # ── Temporal-Diff ──────────────────────────────────────────────────
        if is_first or self._prev_rgb is None:
            diff = np.zeros_like(rgb)
        else:
            diff = np.clip((rgb - self._prev_rgb + 1.0) * 0.5, 0.0, 1.0)
        self._prev_rgb = rgb.copy()

        image = np.concatenate([rgb, diff], axis=-1).astype(np.float32)  # (H,W,6)

        # ── Kinematik & Clearance ──────────────────────────────────────────
        vel = np.array([s.vel_x, s.vel_y], dtype=np.float32)
        speed = np.array([[float(np.linalg.norm(vel))]], dtype=np.float32)

        # Yaw aus Quaternion
        w_q, x_q, y_q, z_q = s.ori_w, s.ori_x, s.ori_y, s.ori_z
        siny = 2.0 * (w_q * z_q + x_q * y_q)
        cosy = 1.0 - 2.0 * (y_q * y_q + z_q * z_q)
        yaw  = float(np.arctan2(siny, cosy))

        clearance = float(np.clip(s.depth_median, 0.0, self.max_clearance))
        if self._clearance_ema is None:
            self._clearance_ema = clearance
        else:
            alpha = self.clearance_ema_alpha
            self._clearance_ema = (1.0 - alpha) * self._clearance_ema + alpha * clearance

        info = {
            "position":          np.array([s.pos_x, s.pos_y], dtype=np.float32),
            "yaw":               yaw,
            "speed":             speed,
            "clearance":         clearance,
            "clearance_ema":     float(self._clearance_ema),
            "terrain_rel_height": float(self._clearance_ema),
            "sim_time":          sim_time,
        }
        return image, speed, info, False  # is_stale=False — fresh frame

    # ── Schritt ausführen ─────────────────────────────────────────────────────

    def step(self, action, reward_fn, max_steps):
        """Sendet Steuerkommando und liest resultierenden Zustand."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        roll_rate  = float(action[0]) if action.shape[0] > 0 else 0.0
        pitch_rate = float(action[1]) if action.shape[0] > 1 else 0.0
        yaw_rate   = float(action[2]) if action.shape[0] > 2 else 0.0
        throttle   = float((action[3] + 1.0) * 0.5) if action.shape[0] > 3 \
                     else float((action[0] + 1.0) * 0.5)
        throttle   = float(np.clip(throttle, 0.0, 1.0))

        self._write_command(roll_rate, pitch_rate, yaw_rate, throttle, 0.1)

        obs_img, speed, info, is_stale = self.read_observation(self.obs_h, self.obs_w, is_first=False)

        # Bug #10: don't reward or read sensors from stale/frozen SHM data
        if is_stale:
            self._stale_count += 1
            obs = {"image": obs_img, "is_first": np.array([False], dtype=np.bool_), "speed": speed}
            truncated = self._stale_count > self._max_stale_frames
            return obs, 0.0, False, truncated, {**info, "stale": True}
        self._stale_count = 0

        s         = self._ipc.sensors
        collision = bool(s.has_collided)
        vel       = np.array([s.vel_x, s.vel_y], dtype=np.float32)
        pos       = np.array([s.pos_x, s.pos_y], dtype=np.float32)

        # Yaw aus Quaternion (gleiche Formel wie in read_observation)
        w_q, x_q, y_q, z_q = s.ori_w, s.ori_x, s.ori_y, s.ori_z
        siny = 2.0 * (w_q * z_q + x_q * y_q)
        cosy = 1.0 - 2.0 * (y_q * y_q + z_q * z_q)
        yaw  = float(np.arctan2(siny, cosy))

        reward_parts = reward_fn(
            position_xy=pos,
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

        info.update({
            "r_explore":  float(reward_parts.r_explore),
            "r_vel":      float(reward_parts.r_vel),
            "r_survival": float(reward_parts.r_survival),
            "r_smooth":   float(reward_parts.r_smooth),
            "r_height":   float(reward_parts.r_height),
            "collision":  bool(reward_parts.collision),
        })

        obs = {
            "image":    obs_img,
            "is_first": np.array([False], dtype=np.bool_),
            "speed":    speed,
        }
        terminated = collision
        truncated  = self._step >= max_steps
        return obs, float(reward_parts.total), terminated, truncated, info

    # ── Interne Helfer ────────────────────────────────────────────────────────

    def _write_command(self, roll, pitch, yaw, throttle, duration,
                       flag: int = _CMD_NONE):
        """Schreibt Kommando in SHM und signalisiert UE."""
        if self._ipc is None:
            return
        self._ipc.header.py_ready = 0  # erst löschen, dann Felder schreiben
        c = self._ipc.command
        c.roll_rate  = float(roll)
        c.pitch_rate = float(pitch)
        c.yaw_rate   = float(yaw)
        c.throttle   = float(throttle)
        c.duration   = float(duration)
        c.flag       = int(flag)
        self._ipc.header.py_ready = 1
        self._sem_py_obj.release()

    def _fallback_obs(self, h: int, w: int, is_first: bool):
        """Fallback: schwarzes Bild wenn SHM nicht erreichbar. Bug #10: returns 4-tuple with is_stale=True."""
        image = np.zeros((h, w, 6), dtype=np.float32)
        speed = np.array([[0.0]], dtype=np.float32)
        info  = {
            "position": np.zeros(2, dtype=np.float32),
            "yaw": 0.0, "speed": speed,
            "clearance": self.max_clearance,
            "clearance_ema": self.max_clearance,
            "terrain_rel_height": self.max_clearance,
        }
        return image, speed, info, True  # is_stale=True
