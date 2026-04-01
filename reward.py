from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardBreakdown:
    r_explore: float
    r_vel: float
    r_survival: float
    r_smooth: float
    r_height: float
    collision: bool

    @property
    def total(self) -> float:
        if self.collision:
            return -1.0
        return self.r_explore + self.r_vel + self.r_survival + self.r_smooth + self.r_height


class DroneRewardFunction:
    """Curriculum reward for drone corridor navigation.

    Terms:
    - r_explore: grid-cell novelty bonus
    - r_vel: alignment of velocity and heading, normalized by max_speed
    - r_survival: +0.01 per timestep
    - r_smooth: penalty for abrupt action changes
    - collision: hard -1.0 terminal penalty
    """

    def __init__(
        self,
        max_speed: float = 1.0,
        grid_size: float = 0.5,
        min_clearance: float = 0.6,
        target_clearance: float = 1.5,
        high_clearance_margin: float = 1.5,
    ):
        self.max_speed = float(max(max_speed, 1e-6))
        self.grid_size = float(max(grid_size, 1e-3))
        self.min_clearance = float(max(min_clearance, 0.05))
        self.target_clearance = float(max(target_clearance, self.min_clearance))
        self.high_clearance_margin = float(max(high_clearance_margin, 0.1))
        self.phase = 0
        self.visited_cells = set()

        self._phase_weights = {
            0: {"explore": 0.08, "vel": 0.25, "survival": 1.0, "smooth": 0.02, "height_low": 0.25, "height_high": 0.02},
            1: {"explore": 0.05, "vel": 0.45, "survival": 1.0, "smooth": 0.05, "height_low": 0.35, "height_high": 0.03},
            2: {"explore": 0.02, "vel": 0.60, "survival": 1.0, "smooth": 0.08, "height_low": 0.45, "height_high": 0.04},
        }

    def set_phase(self, phase: int):
        self.phase = int(np.clip(phase, 0, 2))

    def reset(self):
        self.visited_cells.clear()

    def _grid_novelty(self, position_xy: np.ndarray) -> float:
        key = tuple(np.floor(position_xy / self.grid_size).astype(np.int64).tolist())
        if key in self.visited_cells:
            return 0.0
        self.visited_cells.add(key)
        return 1.0

    def __call__(
        self,
        *,
        position_xy: np.ndarray,
        velocity_xy: np.ndarray,
        yaw: float,
        action: np.ndarray,
        prev_action: np.ndarray,
        collision: bool,
        clearance: float | None = None,
        clearance_ema: float | None = None,
    ) -> RewardBreakdown:
        weights = self._phase_weights[self.phase]

        heading = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
        vel = np.asarray(velocity_xy, dtype=np.float32)
        aligned_speed = float(np.dot(vel, heading) / self.max_speed)
        r_vel = float(weights["vel"] * np.clip(aligned_speed, -1.0, 1.0))

        novelty = self._grid_novelty(np.asarray(position_xy, dtype=np.float32))
        r_explore = float(weights["explore"] * novelty)

        r_survival = float(0.01 * weights["survival"])

        delta_action = np.asarray(action, dtype=np.float32) - np.asarray(prev_action, dtype=np.float32)
        smooth_penalty = float(np.mean(np.square(delta_action)))
        r_smooth = float(-weights["smooth"] * smooth_penalty)
        r_height = 0.0
        c_raw = None if clearance is None else float(clearance)
        c_ema = c_raw if clearance_ema is None else float(clearance_ema)
        if c_raw is not None and c_ema is not None:
            low_deficit = max(0.0, self.min_clearance - min(c_raw, c_ema))
            below_target = max(0.0, self.target_clearance - c_ema)
            high_excess = max(0.0, c_ema - (self.target_clearance + self.high_clearance_margin))
            r_height = float(
                -weights["height_low"] * (1.5 * low_deficit + 0.25 * below_target)
                -weights["height_high"] * high_excess
            )

        return RewardBreakdown(
            r_explore=r_explore,
            r_vel=r_vel,
            r_survival=r_survival,
            r_smooth=r_smooth,
            r_height=r_height,
            collision=bool(collision),
        )
