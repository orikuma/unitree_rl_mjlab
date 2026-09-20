"""One G1 policy learns flat walking and progressively harder five-step stairs."""

from copy import deepcopy

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg

from src.tasks.stairs import mdp
from src.tasks.stairs.terrains import (
  BoxFiveStepStairsTerrainCfg, STAIR_STAGES, TARGET_LEVEL, TERRAIN_SIZE,
)
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg


def unitree_g1_stairs_env_cfg(play=False, level=None, robust=False):
  if play and level is None:
    level = TARGET_LEVEL
  if level is not None and not 0 <= level <= TARGET_LEVEL:
    raise ValueError(f"Stair level must be in [0, {TARGET_LEVEL}]")
  # Start from the training config: rough play overrides randomize the terrain.
  cfg = unitree_g1_rough_env_cfg()
  cfg.episode_length_s = 20.0
  cfg.scene.num_envs = 1 if play else 1024
  cfg.sim.nconmax = 256
  cfg.sim.contact_sensor_maxmatch = 1024
  stages = STAIR_STAGES if level is None else (STAIR_STAGES[level],)
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    seed=42, size=TERRAIN_SIZE, border_width=1.0,
    num_rows=len(stages), num_cols=4 if level is None else 1,
    curriculum=True, difficulty_range=(0.0, 1.0),
    sub_terrains={"stairs": BoxFiveStepStairsTerrainCfg(stages=stages)},
    add_lights=True,
  )
  cfg.scene.terrain.max_init_terrain_level = 0

  for sensor in cfg.scene.sensors:
    if sensor.name == "terrain_scan":
      # Scan from the torso IMU so rays start above the 85 cm top platform even
      # when approaching from the ground. The observations remain relative.
      sensor.frame.type = "site"
      sensor.frame.name = "imu_in_torso"
      sensor.pattern = GridPatternCfg(size=(1.6, 0.8), resolution=0.05)
      sensor.include_geom_groups = (0,)
      sensor.debug_vis = play
  cfg.scene.sensors += (
    ContactSensorCfg(
      name="stair_foot_surfaces",
      primary=ContactMatch(mode="body", pattern=r"^(left|right)_ankle_roll_link$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force", "pos", "normal"), reduce="maxforce", num_slots=4,
    ),
    ContactSensorCfg(
      name="non_foot_terrain_contact",
      primary=ContactMatch(
        mode="geom", pattern=r".*_collision", entity="robot",
        exclude=(r"^(left|right)_foot[1-7]_collision$",),
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"), reduce="netforce", num_slots=1,
    ),
  )

  command = cfg.commands["twist"]
  command.resampling_time_range = (20.0, 20.0)
  command.ranges.lin_vel_x = (0.25, 0.4)
  command.ranges.lin_vel_y = (0.0, 0.0)
  command.ranges.ang_vel_z = (-0.5, 0.5)
  command.ranges.heading = (0.0, 0.0)
  command.heading_command = True
  command.heading_control_stiffness = 1.5
  command.rel_heading_envs = 1.0
  command.rel_standing_envs = 0.0
  command.init_velocity_prob = 0.0
  command.debug_vis = play
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.2, 0.2), "y": (-0.08, 0.08), "yaw": (-0.08, 0.08),
  }
  # The default startup randomization is unnecessarily broad for acquisition.
  cfg.events.pop("push_robot")
  if robust:
    cfg.events["foot_friction"].params["ranges"] = (0.6, 1.2)
    cfg.events["base_com"].params["ranges"] = {axis: (-0.02, 0.02) for axis in range(3)}
  else:
    cfg.events["foot_friction"].params["ranges"] = (0.9, 0.9)
    cfg.events.pop("base_com")
    cfg.events.pop("encoder_bias")
  # Must run after reset_base; reads no derived robot state after reset writes.
  cfg.events["stair_state"] = EventTermCfg(
    func=mdp.reset_stair_state, mode="reset", params={"fixed_level": level},
  )

  for name, group in cfg.observations.items():
    group.terms = deepcopy(group.terms)
    group.terms.pop("phase")
    group.history_length = None
    for key, term in group.terms.items():
      term.history_length = 5 if name == "actor" and key in (
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
      ) else 0
      if term.noise is not None and not robust:
        term.noise.n_min *= 0.25
        term.noise.n_max *= 0.25
    group.terms["height_scan"].noise = (
      UniformNoiseCfg(n_min=-0.02 if robust else -0.005, n_max=0.02 if robust else 0.005)
      if name == "actor" else None
    )
  cfg.observations["actor"].enable_corruption = not play
  cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(func=mdp.horizontal_velocity)
  cfg.observations["critic"].terms["stair_state"] = ObservationTermCfg(func=mdp.task_state)

  for name in ("pose", "foot_gait", "foot_clearance", "stand_still", "is_terminated", "track_angular_velocity"):
    cfg.rewards.pop(name)
  cfg.rewards.update({
    "track_linear_velocity": RewardTermCfg(func=mdp.track_horizontal_velocity, weight=1.5, params={"std": 0.2}),
    "forward_progress": RewardTermCfg(func=mdp.forward_progress, weight=0.3),
    "yaw_error": RewardTermCfg(func=mdp.yaw_error, weight=-0.2),
    "upper_body": RewardTermCfg(
      func=mdp.upper_body_deviation, weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*shoulder.*", r".*elbow.*", r".*wrist.*"))},
    ),
    "completed": RewardTermCfg(func=mdp.completion_bonus, weight=5.0),
    "failed": RewardTermCfg(func=mdp.failure_cost, weight=-5.0),
  })
  cfg.rewards["body_orientation_l2"].weight = -0.5
  cfg.terminations = {
    "completed": TerminationTermCfg(func=mdp.completed),
    "failed": TerminationTermCfg(func=mdp.failed),
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }
  cfg.curriculum = {} if level is not None else {
    "stairs": CurriculumTermCfg(func=mdp.terrain_curriculum),
  }
  cfg.metrics["supported_step"] = MetricsTermCfg(func=mdp.EpisodeStatistics)
  return cfg
