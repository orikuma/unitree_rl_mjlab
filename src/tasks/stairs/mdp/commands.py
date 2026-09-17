"""Position-aware velocity command for stair climbing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from src.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs import ManagerBasedRlEnv


class StairVelocityCommand(UniformVelocityCommand):
  """Command forward motion, then smoothly stop on the top landing."""

  cfg: StairVelocityCommandCfg

  def __init__(self, cfg: StairVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.vel_command_b[env_ids, 0] = self.cfg.forward_speed
    self.vel_command_b[env_ids, 1:] = 0.0

  def _update_command(self) -> None:
    relative_x = (
      self.robot.data.root_link_pos_w[:, 0] - self._env.scene.env_origins[:, 0]
    )
    distance = self.cfg.goal_x - relative_x
    speed_scale = torch.clamp(
      distance / self.cfg.slowdown_distance, min=0.0, max=1.0
    )
    self.vel_command_b[:, 0] = self.cfg.forward_speed * speed_scale
    self.vel_command_b[:, 1] = 0.0
    self.vel_command_b[:, 2] = torch.clamp(
      -self.cfg.heading_control_stiffness * self.robot.data.heading_w,
      min=-self.cfg.max_yaw_rate,
      max=self.cfg.max_yaw_rate,
    )

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
  ) -> None:
    """Keep the task command autonomous when using the Viser viewer.

    The generic velocity-command GUI assumes every velocity axis has a
    positive configurable range. This task intentionally fixes lateral speed
    to zero and derives forward speed from distance to the landing, so manual
    sliders do not apply.
    """
    del name, server, get_env_idx


@dataclass(kw_only=True)
class StairVelocityCommandCfg(UniformVelocityCommandCfg):
  goal_x: float
  forward_speed: float = 0.25
  slowdown_distance: float = 0.35
  max_yaw_rate: float = 0.5

  def build(self, env: ManagerBasedRlEnv) -> StairVelocityCommand:
    return StairVelocityCommand(self, env)
