"""Termination conditions for stair climbing."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class stable_at_goal:
  """Terminate successfully after the robot remains stable on the landing."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self._stable_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    goal_x: float,
    landing_x: float,
    half_width: float,
    foot_half_width: float,
    stable_duration: float,
    max_speed: float,
    max_tilt: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    relative_pos = asset.data.root_link_pos_w - env.scene.env_origins
    foot_pos = (
      asset.data.site_pos_w[:, asset_cfg.site_ids]
      - env.scene.env_origins.unsqueeze(1)
    )
    speed = torch.norm(asset.data.root_link_lin_vel_w, dim=1)
    tilt = torch.acos(
      torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    )
    stable = (
      (relative_pos[:, 0] >= goal_x)
      & (torch.abs(relative_pos[:, 1]) <= half_width)
      & (foot_pos[:, :, 0] >= landing_x).all(dim=1)
      & (torch.abs(foot_pos[:, :, 1]) <= foot_half_width).all(dim=1)
      & (speed <= max_speed)
      & (tilt <= max_tilt)
    )
    self._stable_steps = torch.where(stable, self._stable_steps + 1, 0)
    required_steps = math.ceil(stable_duration / env.step_dt)
    return self._stable_steps >= required_steps

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._stable_steps[env_ids] = 0


def left_stair_corridor(
  env: ManagerBasedRlEnv,
  half_width: float,
  max_forward_x: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate attempts that leave the staircase corridor or overshoot it."""
  asset: Entity = env.scene[asset_cfg.name]
  relative_pos = asset.data.root_link_pos_w - env.scene.env_origins
  return (torch.abs(relative_pos[:, 1]) > half_width) | (
    relative_pos[:, 0] > max_forward_x
  )


def illegal_terrain_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float,
) -> torch.Tensor:
  """Terminate when a non-foot link bears substantial terrain contact force."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  return (torch.norm(sensor.data.force, dim=-1) > force_threshold).any(dim=1)
