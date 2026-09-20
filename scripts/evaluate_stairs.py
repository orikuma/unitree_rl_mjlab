"""Evaluate complete stair ascents on fixed terrain levels and held-out seeds.

Example:
  python scripts/evaluate_stairs.py --checkpoint-file logs/.../model_5000.pt
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import mjlab
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.utils.torch import configure_torch_backends

import src.tasks  # noqa: F401; populate the task registry
from src.tasks.stairs.config.g1.env_cfgs import unitree_g1_stairs_env_cfg
from src.tasks.stairs.mdp import stair_state
from src.tasks.stairs.terrains import STAIR_STAGES, TARGET_LEVEL

TASK = "Unitree-G1-Stairs"


@dataclass(frozen=True)
class EvalConfig:
  checkpoint_file: Path | None = None
  agent: Literal["trained", "zero"] = "trained"
  num_envs: int = 32
  episodes: int = 100
  """Number of completed episodes per seed and level, balanced across environments."""
  seeds: tuple[int, ...] = (123, 2026, 3456)
  levels: tuple[int, ...] = (TARGET_LEVEL,)
  device: str | None = None
  observation_noise: float = 0.0
  """Multiplier of configured actor observation noise; zero disables corruption."""
  robust: bool = False
  """Enable the task's optional domain randomization."""
  sample_actions: bool = False
  output: Path = Path("/tmp/g1_stairs_evaluation.json")


def summarize(records: list[dict]) -> dict:
  """Binomial Wilson interval, plus metrics that distinguish partial ascents."""
  n = len(records)
  successes = sum(int(r["success"]) for r in records)
  interval = None
  if n:
    p, z = successes / n, 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    interval = [max(0.0, center - half), min(1.0, center + half)]
  return {
    "episodes": n,
    "successes": successes,
    "failures": sum(int(r["failure"]) for r in records),
    "timeouts": sum(int(r["timeout"]) for r in records),
    "success_rate": successes / n if n else None,
    "success_rate_wilson_95": interval,
    "step_reached_counts": {
      str(step): sum(r["highest_step"] >= step for r in records)
      for step in range(1, 6)
    },
    "forbidden_contact_episodes": sum(bool(r["forbidden_contact"]) for r in records),
    "means": {
      key: sum(float(r[key]) for r in records) / n if n else None
      for key in ("duration_s", "forward_distance", "riser_contact_fraction", "velocity_rmse")
    },
  }


def evaluate_condition(cfg: EvalConfig, seed: int, level: int, device: str) -> dict:
  env_cfg = unitree_g1_stairs_env_cfg(play=True, level=level, robust=cfg.robust)
  env_cfg.seed = seed
  env_cfg.scene.num_envs = min(cfg.num_envs, cfg.episodes)
  env_cfg.curriculum = {}
  actor_obs = env_cfg.observations["actor"]
  actor_obs.enable_corruption = cfg.observation_noise > 0
  noise_settings = {}
  for name, term in actor_obs.terms.items():
    if term.noise is None:
      continue
    # All task noise terms are additive uniform noise. Fail explicitly if this changes.
    if not isinstance(term.noise, UniformNoiseCfg) or term.noise.operation != "add":
      raise TypeError(f"Unsupported evaluation noise for {name}: {term.noise}")
    for bound in ("n_min", "n_max"):
      value = getattr(term.noise, bound)
      scaled = (
        tuple(x * cfg.observation_noise for x in value)
        if isinstance(value, tuple)
        else value * cfg.observation_noise
      )
      setattr(term.noise, bound, scaled)
    noise_settings[name] = {"min": term.noise.n_min, "max": term.noise.n_max}

  agent_cfg = load_rl_cfg(TASK)
  agent_cfg.seed = seed
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  try:
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    policy = None
    if cfg.agent == "trained":
      runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
      runner = runner_cls(env, asdict(agent_cfg), device=device)
      runner.load(
        str(cfg.checkpoint_file), load_cfg={"actor": True}, strict=True, map_location=device
      )
      # Evaluation mode freezes the actor's observation normalization statistics.
      policy = runner.get_inference_policy(device=device)
    env.seed(seed)
    obs, _ = env.reset()
    state = stair_state(raw_env)
    n = env.num_envs
    quotas = [cfg.episodes // n + int(i < cfg.episodes % n) for i in range(n)]
    completed = [0] * n
    records: list[dict] = []
    # Each environment must finish its own quota: fast failures cannot displace
    # slower successes from the evaluation sample.
    max_steps = (env.max_episode_length + 1) * max(quotas)
    with torch.inference_mode():
      for _ in range(max_steps):
        actions = (
          policy(obs, stochastic_output=cfg.sample_actions)
          if policy is not None
          else torch.zeros((n, env.num_actions), device=device)
        )
        obs, _, dones, _ = env.step(actions)
        if policy is not None:
          policy.reset(dones)
        # step() auto-resets completed episodes. The task reset event preserves
        # their terminal metrics in last_episode before clearing live state.
        for i in dones.nonzero(as_tuple=False).flatten().tolist():
          if completed[i] >= quotas[i]:
            continue
          record = {key: value[i].item() for key, value in state.last_episode.items()}
          record.update(seed=seed, env_id=i, episode=completed[i])
          records.append(record)
          completed[i] += 1
        if len(records) == cfg.episodes:
          break
    return {
      "seed": seed,
      "level": level,
      "stair_dimensions_m": asdict(STAIR_STAGES[level]),
      "complete": len(records) == cfg.episodes,
      "expected_episodes": cfg.episodes,
      "environment_quotas": quotas,
      "environment_completed": completed,
      "configuration": {
        "num_envs": n,
        "episode_length_s": env_cfg.episode_length_s,
        "step_dt": raw_env.step_dt,
        "max_steps": max_steps,
        "actor_noise_bounds": noise_settings,
        "commands": {name: repr(command) for name, command in env_cfg.commands.items()},
        "events": list(env_cfg.events),
        "terrain": repr(env_cfg.scene.terrain.terrain_generator),
        "actor_observation_terms": {
          name: {"history_length": term.history_length, "scale": term.scale}
          for name, term in actor_obs.terms.items()
        },
      },
      "summary": summarize(records),
      "episodes": records,
    }
  finally:
    raw_env.close()


def main():
  cfg = tyro.cli(EvalConfig, config=mjlab.TYRO_FLAGS)
  if cfg.num_envs < 1 or cfg.episodes < 1 or not cfg.seeds or not cfg.levels:
    raise ValueError("num_envs and episodes must be positive; seeds and levels must be nonempty")
  if len(set(cfg.seeds)) != len(cfg.seeds) or len(set(cfg.levels)) != len(cfg.levels):
    raise ValueError("seeds and levels must not contain duplicates")
  if not math.isfinite(cfg.observation_noise) or cfg.observation_noise < 0:
    raise ValueError("observation_noise must be finite and nonnegative")
  if any(level < 0 or level > TARGET_LEVEL for level in cfg.levels):
    raise ValueError(f"levels must be between 0 and {TARGET_LEVEL}")
  if cfg.agent == "trained" and cfg.checkpoint_file is None:
    raise ValueError("--checkpoint-file is required for --agent trained")
  if cfg.agent == "zero" and (cfg.checkpoint_file is not None or cfg.sample_actions):
    raise ValueError("The zero agent does not use checkpoints or stochastic actions")
  checkpoint = None
  if cfg.checkpoint_file is not None:
    with cfg.checkpoint_file.open("rb") as stream:
      checkpoint = {
        "path": str(cfg.checkpoint_file.resolve()),
        "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
      }
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  report = {
    "task": TASK,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "checkpoint": checkpoint,
    "configuration": {**asdict(cfg), "device": device},
    "complete": False,
    "stages": [],
  }
  cfg.output.parent.mkdir(parents=True, exist_ok=True)
  for level in cfg.levels:
    for seed in cfg.seeds:
      stage = evaluate_condition(cfg, seed, level, device)
      report["stages"].append(stage)
      cfg.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
      print(f"[evaluation] level={level}, seed={seed}: {stage['summary']}")
      if not stage["complete"]:
        raise RuntimeError(f"Evaluation did not meet per-environment quotas; see {cfg.output}")
  report["by_level"] = {
    str(level): summarize([
      record for stage in report["stages"] if stage["level"] == level
      for record in stage["episodes"]
    ]) for level in cfg.levels
  }
  report["complete"] = True
  cfg.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
  print(f"[evaluation] Report written to {cfg.output.resolve()}")


if __name__ == "__main__":
  main()
