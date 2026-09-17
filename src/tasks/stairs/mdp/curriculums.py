"""Performance-based terrain curriculum for stair climbing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def stair_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  success_term: str,
) -> torch.Tensor:
  """Advance on success and step back after an unsuccessful episode."""
  terrain = env.scene.terrain
  assert terrain is not None
  assert terrain.terrain_origins is not None

  succeeded = env.termination_manager.get_term(success_term)[env_ids]
  levels = terrain.terrain_levels[env_ids]
  levels = torch.where(succeeded, levels + 1, levels - 1)
  levels = torch.clamp(levels, 0, terrain.max_terrain_level - 1)
  terrain.terrain_levels[env_ids] = levels
  terrain.env_origins[env_ids] = terrain.terrain_origins[
    levels, terrain.terrain_types[env_ids]
  ]
  return torch.mean(terrain.terrain_levels.float())

