"""Reward terms for stair climbing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_stair_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track planar velocity without penalizing the vertical motion needed to climb."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.sum(
    torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1
  )
  return torch.exp(-error / std**2)


def goal_progress(
  env: ManagerBasedRlEnv,
  goal_x: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward progress along the staircase while saturating at the landing."""
  asset: Entity = env.scene[asset_cfg.name]
  relative_x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  return torch.clamp(relative_x / goal_x, min=0.0, max=1.0)


def lateral_deviation(
  env: ManagerBasedRlEnv,
  half_width: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize drifting toward the sides of the staircase."""
  asset: Entity = env.scene[asset_cfg.name]
  relative_y = asset.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
  return torch.square(relative_y / half_width)


def undesired_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Count non-foot links in contact with the terrain."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float().sum(dim=1)


def goal_reached(env: ManagerBasedRlEnv, term_name: str) -> torch.Tensor:
  """Provide a one-step bonus when the stable goal termination fires."""
  return env.termination_manager.get_term(term_name).float()
