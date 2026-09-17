"""Unitree G1 five-step stair-climbing environment."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  RayCastSensorCfg,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.tasks.stairs import mdp
from src.tasks.stairs.terrains import BoxFiveStepStairsTerrainCfg
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from src.tasks.velocity.mdp import UniformVelocityCommandCfg

NUM_STEPS = 5
TARGET_STEP_HEIGHT = 0.17
TARGET_STEP_DEPTH = 0.17
STAIR_HALF_WIDTH = 0.60
CORRIDOR_HALF_WIDTH = 0.50
GOAL_X = 2.90


def _terrain_generator_cfg(play: bool) -> TerrainGeneratorCfg:
  stairs = BoxFiveStepStairsTerrainCfg(
    proportion=1.0,
    stages=(
      (0.05, 0.30),
      (0.08, 0.28),
      (0.12, 0.25),
      (0.14, 0.22),
      (TARGET_STEP_HEIGHT, TARGET_STEP_DEPTH),
    ),
    num_steps=NUM_STEPS,
    stair_width=2.0 * STAIR_HALF_WIDTH,
    landing_start_x=3.0,
    spawn_x=0.4,
  )
  if play:
    return TerrainGeneratorCfg(
      seed=42,
      size=(5.0, 2.0),
      # The scan extends behind spawn; keep the same outer floor as training.
      border_width=1.0,
      num_rows=1,
      num_cols=1,
      curriculum=False,
      difficulty_range=(1.0, 1.0),
      sub_terrains={"five_step_stairs": stairs},
      add_lights=True,
    )
  return TerrainGeneratorCfg(
    seed=42,
    size=(5.0, 2.0),
    border_width=1.0,
    num_rows=len(stairs.stages),
    num_cols=4,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    sub_terrains={"five_step_stairs": stairs},
    add_lights=True,
  )


def unitree_g1_stairs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 task used to learn five-step stair ascent."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  # Stair edges produce more simultaneous contacts than the generic rough task.
  cfg.sim.nconmax = 256

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = _terrain_generator_cfg(play)
  cfg.scene.terrain.max_init_terrain_level = None if play else 1

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.pattern = GridPatternCfg(size=(1.6, 0.8), resolution=0.05)
      sensor.debug_vis = play

  non_foot_contact = ContactSensorCfg(
    name="non_foot_terrain_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r".*_collision",
      entity="robot",
      exclude=(r"^(left|right)_foot[1-7]_collision$",),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (non_foot_contact,)

  cfg.commands["twist"] = mdp.StairVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1000.0, 1000.0),
    heading_command=False,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.0, 0.30),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(-0.5, 0.5),
    ),
    goal_x=GOAL_X,
    forward_speed=0.25,
    slowdown_distance=0.35,
    heading_control_stiffness=1.5,
    max_yaw_rate=0.5,
    debug_vis=play,
  )

  reset_base = cfg.events["reset_base"]
  reset_base.params["pose_range"] = {
    "x": (-0.05, 0.05),
    "y": (-0.10, 0.10),
    "z": (0.0, 0.0),
    "yaw": (-0.05, 0.05),
  }
  cfg.events.pop("push_robot", None)
  cfg.events.pop("randomize_terrain", None)
  cfg.events["foot_friction"].params["ranges"] = (0.6, 1.2)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.03, 0.03),
    1: (-0.03, 0.03),
    2: (-0.03, 0.03),
  }

  goal_observation = ObservationTermCfg(
    func=mdp.stair_goal,
    params={
      "goal_x": GOAL_X,
      "half_width": CORRIDOR_HALF_WIDTH,
      "base_height": 0.8,
      "target_elevation": NUM_STEPS * TARGET_STEP_HEIGHT,
    },
  )
  cfg.observations["actor"].terms["stair_goal"] = goal_observation
  cfg.observations["critic"].terms["stair_goal"] = goal_observation
  cfg.observations["actor"].terms["height_scan"].noise = Unoise(
    n_min=-0.02, n_max=0.02
  )

  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=mdp.track_stair_velocity,
    weight=1.5,
    params={"command_name": "twist", "std": math.sqrt(0.20)},
  )
  cfg.rewards["goal_progress"] = RewardTermCfg(
    func=mdp.goal_progress,
    weight=2.0,
    params={"goal_x": GOAL_X},
  )
  cfg.rewards["lateral_deviation"] = RewardTermCfg(
    func=mdp.lateral_deviation,
    weight=-1.0,
    params={"half_width": CORRIDOR_HALF_WIDTH},
  )
  cfg.rewards["non_foot_contact"] = RewardTermCfg(
    func=mdp.undesired_contact,
    weight=-0.5,
    params={"sensor_name": non_foot_contact.name},
  )
  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=envs_mdp.joint_torques_l2,
    weight=-1.0e-5,
  )
  cfg.rewards["goal_reached"] = RewardTermCfg(
    func=mdp.goal_reached,
    weight=500.0,
    params={"term_name": "goal_reached"},
  )
  cfg.rewards.pop("foot_clearance", None)
  cfg.rewards.pop("foot_gait", None)

  cfg.terminations["left_stair_corridor"] = TerminationTermCfg(
    func=mdp.left_stair_corridor,
    params={"half_width": CORRIDOR_HALF_WIDTH, "max_forward_x": GOAL_X + 0.75},
  )
  cfg.terminations["illegal_terrain_contact"] = TerminationTermCfg(
    func=mdp.illegal_terrain_contact,
    params={"sensor_name": non_foot_contact.name, "force_threshold": 100.0},
  )
  cfg.terminations["goal_reached"] = TerminationTermCfg(
    func=mdp.stable_at_goal,
    time_out=True,
    params={
      "goal_x": GOAL_X,
      "landing_x": 3.0 - 0.4,
      "half_width": CORRIDOR_HALF_WIDTH,
      "foot_half_width": STAIR_HALF_WIDTH,
      "stable_duration": 0.5,
      "max_speed": 0.25,
      "max_tilt": math.radians(20.0),
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot")
      ),
    },
  )

  cfg.episode_length_s = 20.0
  if play:
    cfg.curriculum = {}
  else:
    cfg.curriculum = {
      "stair_levels": CurriculumTermCfg(
        func=mdp.stair_levels,
        params={"success_term": "goal_reached"},
      )
    }

  return cfg
