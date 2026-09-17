"""PPO configuration for Unitree G1 stair climbing."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def unitree_g1_stairs_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Reuse the proven G1 locomotion PPO settings under a separate experiment."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_stairs"
  cfg.run_name = "five_steps_17cm"
  return cfg

