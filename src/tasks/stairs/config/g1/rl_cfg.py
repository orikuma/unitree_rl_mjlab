"""Keep the main G1 PPO implementation and observation normalization."""

from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def unitree_g1_stairs_ppo_runner_cfg():
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_stairs_perceptive"
  cfg.logger = "tensorboard"
  return cfg
