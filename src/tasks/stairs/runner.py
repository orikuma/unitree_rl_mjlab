"""Preserve stair curriculum progress with the existing PPO and export runner."""

import torch

from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .mdp import stair_state
from .terrains import TARGET_LEVEL


class StairOnPolicyRunner(VelocityOnPolicyRunner):
  def save(self, path: str, infos=None):
    env = self.env.unwrapped
    if "stairs" in env.cfg.curriculum:
      state = stair_state(env)
      infos = {
        **(infos or {}),
        "stairs_curriculum": {
          "frontier": state.frontier.detach().cpu().clone(),
          "failures": state.failures.detach().cpu().clone(),
        },
      }
    super().save(path, infos)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict | None:
    infos = super().load(path, load_cfg, strict, map_location)
    env = self.env.unwrapped
    if load_cfg is not None or "stairs" not in env.cfg.curriculum or not infos:
      return infos
    saved = infos.get("stairs_curriculum")
    if saved is None:
      return infos

    frontier = torch.as_tensor(saved["frontier"], device=env.device, dtype=torch.long)
    failures = torch.as_tensor(saved["failures"], device=env.device, dtype=torch.long)
    if frontier.ndim != 1 or failures.shape != frontier.shape or frontier.numel() == 0:
      raise ValueError("Checkpoint stair curriculum must contain matching nonempty vectors.")
    if ((frontier < 1) | (frontier > TARGET_LEVEL) | (failures < 0)).any():
      raise ValueError("Checkpoint stair curriculum contains invalid levels or failure counts.")

    # Preserve all entries for equal batch sizes; evenly repeat or subsample
    # both vectors together when the resumed run uses a different environment count.
    indices = torch.div(
      torch.arange(env.num_envs, device=env.device) * frontier.numel(),
      env.num_envs,
      rounding_mode="floor",
    )
    state = stair_state(env)
    state.frontier.copy_(frontier[indices])
    state.failures.copy_(failures[indices])
    # The reset curriculum selects new rows from the restored frontier. Its
    # initial-reset guard prevents the discarded episode from changing mastery.
    state.initialized.zero_()
    state.stamp = env.common_step_counter
    self.env.reset()
    return infos
