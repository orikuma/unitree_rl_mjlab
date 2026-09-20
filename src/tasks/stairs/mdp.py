"""Stair-specific signals; physics, actions and PPO remain in the base task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .terrains import FIRST_RISER_X, STAIR_STAGES, STAIR_WIDTH, TARGET_LEVEL

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class StairState:
  """One update per control step, shared by rewards, termination and evaluation.

  Terrain rows/origins are captured at reset: the curriculum moves the next
  episode's origin before reset events run. Completed metrics must describe the
  old episode, not the terrain selected for its replacement.
  """

  def __init__(self, env: ManagerBasedRlEnv):
    self.env = env
    self.stamp = env.common_step_counter
    n, device = env.num_envs, env.device
    self.level = torch.zeros(n, dtype=torch.long, device=device)
    self.frontier = torch.ones_like(self.level)
    self.failures = torch.zeros_like(self.level)
    self.elapsed = torch.zeros_like(self.level)
    self.origins = torch.zeros(n, 3, device=device)
    self.dimensions = torch.tensor(
      [(s.num_steps, s.height, s.depth) for s in STAIR_STAGES], device=device
    )
    self.highest_step = torch.zeros_like(self.level)
    self.forward_distance = torch.zeros(n, device=device)
    self.top_contact_age = torch.full((n, 2), float("inf"), device=device)
    self.hold_time = torch.zeros(n, device=device)
    self.forbidden_time = torch.zeros(n, device=device)
    self.riser_steps = torch.zeros(n, device=device)
    self.velocity_error_sum = torch.zeros(n, device=device)
    self.success = torch.zeros(n, dtype=torch.bool, device=device)
    self.failure = torch.zeros_like(self.success)
    self.forbidden_contact = torch.zeros_like(self.success)
    self.initialized = torch.zeros_like(self.success)
    self.snapshot_valid = torch.zeros_like(self.success)
    self.last_episode = {
      name: torch.zeros(n, device=device)
      for name in (
        "success", "failure", "timeout", "highest_step", "forward_distance",
        "duration_s", "level", "forbidden_contact", "riser_contact_fraction", "velocity_rmse",
      )
    }
    self.foot_ids, _ = env.scene["robot"].find_sites(("left_foot", "right_foot"))

  def update(self):
    env = self.env
    if self.stamp == env.common_step_counter:
      return
    self.stamp = env.common_step_counter
    self.elapsed += self.initialized.long()
    robot = env.scene["robot"]
    position = robot.data.root_link_pos_w - self.origins
    self.forward_distance = torch.maximum(self.forward_distance, position[:, 0])
    command = env.command_manager.get_command("twist")[:, :2]
    self.velocity_error_sum += (horizontal_velocity(env) - command).square().sum(-1)
    num_steps, height, depth = self.dimensions[self.level].unbind(-1)
    landing_x = FIRST_RISER_X + torch.where(num_steps > 0, num_steps, 5) * depth
    top_z = num_steps * height

    contact = env.scene["stair_foot_surfaces"].data
    # Four strongest contacts per foot. A riser contact is not a landing.
    points = contact.pos.reshape(env.num_envs, 2, 4, 3) - self.origins[:, None, None]
    normals = contact.normal.reshape(env.num_envs, 2, 4, 3)
    forces = contact.force.reshape(env.num_envs, 2, 4, 3).norm(dim=-1)
    found = contact.found.reshape(env.num_envs, 2, 4) > 0
    loaded = found & (forces > 5.0)
    surface = torch.floor((points[..., 0] - FIRST_RISER_X) / depth[:, None, None]) + 1
    surface = torch.minimum(surface.clamp(min=0), num_steps[:, None, None])
    expected_z = surface * height[:, None, None]
    # Normal is primary (foot) -> secondary (terrain), hence downward on tops.
    support = (
      loaded & (normals[..., 2] < -0.7)
      & ((points[..., 2] - expected_z).abs() < (height * 0.25).clamp(0.003, 0.02)[:, None, None])
      & (points[..., 1].abs() < STAIR_WIDTH / 2 + 0.02)
    )
    supported_step = torch.where(support, surface, 0).amax(dim=(1, 2)).long()
    self.highest_step = torch.maximum(self.highest_step, supported_step)
    on_top = (support & (surface == num_steps[:, None, None])).any(dim=-1)
    self.top_contact_age = torch.where(on_top, 0.0, self.top_contact_age + env.step_dt)
    self.riser_steps += (loaded & (normals[..., 2].abs() < 0.5)).any(dim=(1, 2))

    forbidden = env.scene["non_foot_terrain_contact"].data
    touching = (forbidden.force.norm(dim=-1) > 20.0).any(dim=-1)
    self.forbidden_contact |= touching
    self.forbidden_time = torch.where(touching, self.forbidden_time + env.step_dt, 0.0)
    upright = -robot.data.projected_gravity_b[:, 2]
    outside = (
      (position[:, 1].abs() > STAIR_WIDTH / 2 - 0.05)
      | (position[:, 0] < -0.4) | (position[:, 0] > landing_x + 1.3)
    )
    self.failure = self.initialized & (
      (upright < math.cos(math.radians(65))) | outside | (self.forbidden_time >= 0.08)
    )
    feet_z = robot.data.site_pos_w[:, self.foot_ids, 2] - self.origins[:, 2, None]
    # Allow alternating support and heel overhang, but not flying past the goal.
    stable_top = (
      (position[:, 0] > landing_x + 0.10)
      & (position[:, 2] > top_z + 0.5)
      & (feet_z >= top_z[:, None] - 0.03).all(dim=1)
      & (self.top_contact_age < 0.75).all(dim=1)
      & (self.top_contact_age < 0.06).any(dim=1)
      & (upright > math.cos(math.radians(35)))
      & ~touching & ~self.failure
    )
    self.hold_time = torch.where(stable_top, self.hold_time + env.step_dt, 0.0)
    self.success = self.initialized & (self.hold_time >= 1.0) & ~self.failure

  def reset(self, env_ids, fixed_level=None):
    env = self.env
    valid = self.initialized[env_ids] & (self.elapsed[env_ids] > 0)
    self.snapshot_valid[env_ids] = valid
    values = {
      "success": self.success,
      "failure": self.failure,
      "timeout": env.termination_manager.get_term("time_out") & ~self.success & ~self.failure,
      "highest_step": self.highest_step,
      "forward_distance": self.forward_distance,
      "duration_s": self.elapsed * env.step_dt,
      "level": self.level,
      "forbidden_contact": self.forbidden_contact,
      "riser_contact_fraction": self.riser_steps / self.elapsed.clamp(min=1),
      "velocity_rmse": (self.velocity_error_sum / self.elapsed.clamp(min=1)).sqrt(),
    }
    for name, value in values.items():
      self.last_episode[name][env_ids] = value[env_ids].float()
    terrain = env.scene.terrain
    self.level[env_ids] = terrain.terrain_levels[env_ids] if fixed_level is None else fixed_level
    self.origins[env_ids] = env.scene.env_origins[env_ids]
    for buffer in (
      self.elapsed, self.highest_step, self.forward_distance, self.hold_time,
      self.forbidden_time, self.riser_steps, self.velocity_error_sum, self.success, self.failure,
      self.forbidden_contact,
    ):
      buffer[env_ids] = 0
    self.top_contact_age[env_ids] = float("inf")
    self.initialized[env_ids] = True


def stair_state(env: ManagerBasedRlEnv) -> StairState:
  if not hasattr(env, "_stairs_state"):
    env._stairs_state = StairState(env)
  return env._stairs_state


def reset_stair_state(env, env_ids, fixed_level=None):
  stair_state(env).reset(env_ids, fixed_level)


def completed(env):
  state = stair_state(env)
  state.update()
  return state.success


def failed(env):
  state = stair_state(env)
  state.update()
  return state.failure


def time_out(env):
  # A real terminal event on the last step must not also trigger PPO's timeout
  # bootstrap. Success, physical failure and artificial truncation are disjoint.
  return (env.episode_length_buf >= env.max_episode_length) & ~completed(env) & ~failed(env)


def horizontal_velocity(env):
  """World velocity rotated by yaw only; pitch cannot turn ascent into progress."""
  robot = env.scene["robot"]
  yaw = robot.data.heading_w
  velocity = robot.data.root_link_lin_vel_w
  c, s = yaw.cos(), yaw.sin()
  return torch.stack((c * velocity[:, 0] + s * velocity[:, 1],
                      -s * velocity[:, 0] + c * velocity[:, 1]), dim=-1)


def track_horizontal_velocity(env, std=0.2):
  command = env.command_manager.get_command("twist")[:, :2]
  tracking = torch.exp(-(horizontal_velocity(env) - command).square().sum(-1) / std**2)
  standing = torch.exp(-command.square().sum(-1) / std**2)
  return tracking - standing


def yaw_error(env):
  command = env.command_manager.get_command("twist")[:, 2]
  return (env.scene["robot"].data.root_link_ang_vel_w[:, 2] - command).square()


def forward_progress(env):
  speed = env.command_manager.get_command("twist")[:, 0].clamp(min=0.1)
  rate = env.scene["robot"].data.root_link_lin_vel_w[:, 0] / speed
  # Retain the full backward cost: symmetric clipping could reward repeated
  # slow forward motion followed by a fast retreat to the same position.
  return torch.minimum(rate, torch.ones_like(rate))


def completion_bonus(env):
  return completed(env).float() / env.step_dt


def failure_cost(env):
  return failed(env).float() / env.step_dt


def upper_body_deviation(env, asset_cfg):
  robot = env.scene[asset_cfg.name]
  return (robot.data.joint_pos[:, asset_cfg.joint_ids]
          - robot.data.default_joint_pos[:, asset_cfg.joint_ids]).square().mean(-1)


def task_state(env):
  """Critic-only episode state needed to predict completion and failure."""
  state = stair_state(env)
  return torch.cat((state.hold_time[:, None], state.top_contact_age.clamp(max=1.0),
                    state.forbidden_time[:, None]), dim=1)


def terrain_curriculum(env, env_ids, flat_fraction=0.2, easier_fraction=0.2):
  """Per-environment promotion with on-policy flat/easier-terrain rehearsal."""
  state = stair_state(env)
  ids = env_ids
  finished_frontier = (
    state.initialized[ids] & (state.elapsed[ids] > 0)
    & (state.level[ids] == state.frontier[ids])
  )
  success = finished_frontier & state.success[ids]
  failure = finished_frontier & ~state.success[ids]
  state.failures[ids] = torch.where(success, 0, state.failures[ids] + failure.long())
  down = state.failures[ids] >= 3
  state.frontier[ids] = (state.frontier[ids] + success.long() - down.long()).clamp(1, TARGET_LEVEL)
  state.failures[ids] = torch.where(down, 0, state.failures[ids])
  draw = torch.rand(len(ids), device=env.device)
  # Before any promotion, 30% flat + 70% shallow five-step staircases.
  flat_p = torch.where(state.frontier[ids] == 1, 0.3, flat_fraction)
  easier = (1 + torch.rand(len(ids), device=env.device) * (state.frontier[ids] - 1)).long()
  levels = torch.where(draw < flat_p, 0,
    torch.where(draw < flat_p + easier_fraction, easier, state.frontier[ids]))
  terrain = env.scene.terrain
  terrain.terrain_levels[ids] = levels
  terrain.env_origins[ids] = terrain.terrain_origins[levels, terrain.terrain_types[ids]]
  return {"frontier": state.frontier.float().mean(), "sampled_level": terrain.terrain_levels.float().mean()}


class EpisodeStatistics:
  """Log actual completed episodes, rather than step averages of success pulses."""

  def __init__(self, cfg, env):
    self.env = env

  def __call__(self, env):
    return stair_state(env).highest_step.float()

  def reset(self, env_ids):
    state = stair_state(self.env)
    ids = env_ids[state.snapshot_valid[env_ids]]
    if len(ids) == 0:
      return
    logs = self.env.extras["log"]
    for key, value in state.last_episode.items():
      logs[f"Stairs/{key}"] = value[ids].mean()
    logs["Stairs/episodes"] = len(ids)
    for level in range(len(STAIR_STAGES)):
      mask = state.last_episode["level"][ids] == level
      if mask.any():
        logs[f"Stairs/level_{level}/success"] = state.last_episode["success"][ids][mask].mean()
        logs[f"Stairs/level_{level}/episodes"] = mask.sum()
