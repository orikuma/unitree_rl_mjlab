"""Observations specific to the stair-climbing goal."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def stair_goal(
  env: ManagerBasedRlEnv,
  goal_x: float,
  half_width: float,
  base_height: float,
  target_elevation: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return normalized distance-to-goal, lateral error, and achieved height."""
  asset: Entity = env.scene[asset_cfg.name]
  relative_pos = asset.data.root_link_pos_w - env.scene.env_origins
  remaining_x = torch.clamp((goal_x - relative_pos[:, 0]) / goal_x, -1.0, 1.0)
  lateral_error = torch.clamp(relative_pos[:, 1] / half_width, -1.0, 1.0)
  height = torch.clamp(
    (relative_pos[:, 2] - base_height) / target_elevation, 0.0, 1.5
  )
  return torch.stack((remaining_x, lateral_error, height), dim=-1)
