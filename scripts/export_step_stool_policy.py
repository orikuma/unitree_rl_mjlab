import sys
import shutil
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import onnx
import torch
import yaml

import src.tasks

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import (
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
)


task_id = "Unitree-G1-Tracking-StepStool"

checkpoint = Path(sys.argv[1]).resolve()
motion_file = Path(sys.argv[2]).resolve()

output_dir = Path(
    "deploy/robots/g1/config/policy/mimic/"
    "g1_step_stool_up_150mm"
)

exported_dir = output_dir / "exported"
params_dir = output_dir / "params"

exported_dir.mkdir(parents=True, exist_ok=True)
params_dir.mkdir(parents=True, exist_ok=True)

device = "cuda:0"

# 学習時の環境と方策設定をロード
cfg = load_env_cfg(task_id, play=True)
cfg.scene.num_envs = 1
cfg.commands["motion"].motion_file = str(motion_file)

agent_cfg = load_rl_cfg(task_id)

assert agent_cfg.clip_actions is None

env = ManagerBasedRlEnv(
    cfg=cfg,
    device=device,
)

vec_env = RslRlVecEnvWrapper(
    env,
    clip_actions=agent_cfg.clip_actions,
)

runner_cls = load_runner_cls(task_id)

runner = runner_cls(
    vec_env,
    asdict(agent_cfg),
    device=device,
)

runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
)

# 指定したcheckpointをONNXに変換
runner.export_policy_to_onnx(
    str(exported_dir),
    "policy.onnx",
)

robot = env.scene["robot"]
action = env.action_manager.get_term("joint_pos")

joint_names = list(robot.joint_names)

assert len(joint_names) == 29
assert list(action.target_names) == joint_names

# Unitree G1のモータ順序
motor_names = []

for side in ("left", "right"):
    motor_names.extend(
        f"{side}_{name}_joint"
        for name in (
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee",
            "ankle_pitch",
            "ankle_roll",
        )
    )

motor_names.extend(
    f"waist_{name}_joint"
    for name in ("yaw", "roll", "pitch")
)

for side in ("left", "right"):
    motor_names.extend(
        f"{side}_{name}_joint"
        for name in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    )

assert joint_names == motor_names

model = env.sim.mj_model

# 学習環境のアクチュエータからPDゲインを取得
kp = []
kd = []

for name in joint_names:
    joint_id = model.joint(f"robot/{name}").id

    ids = np.flatnonzero(
        (model.actuator_trnid[:, 0] == joint_id)
        & (
            model.actuator_trntype
            == mujoco.mjtTrn.mjTRN_JOINT
        )
    )

    assert len(ids) == 1

    actuator_id = ids[0]

    kp.append(
        float(model.actuator_gainprm[actuator_id, 0])
    )
    kd.append(
        float(-model.actuator_biasprm[actuator_id, 2])
    )


def to_vector(value):
    if isinstance(value, torch.Tensor):
        return value[0].cpu().tolist()
    return [float(value)] * 29


default_joint_pos = (
    robot.data.default_joint_pos[0].cpu().tolist()
)

action_scale = to_vector(action.scale)
action_offset = to_vector(action.offset)

# 既存のMimic用設定をテンプレートとして利用
template = Path(
    "deploy/robots/g1/config/policy/mimic/"
    "dance1_subject2/params/deploy.yaml"
)

with template.open() as f:
    deploy_cfg = yaml.safe_load(f)

deploy_cfg["step_dt"] = float(env.step_dt)
deploy_cfg["joint_ids_map"] = list(range(29))
deploy_cfg["stiffness"] = kp
deploy_cfg["damping"] = kd
deploy_cfg["default_joint_pos"] = default_joint_pos

deploy_cfg["actions"]["JointPositionAction"][
    "scale"
] = action_scale

deploy_cfg["actions"]["JointPositionAction"][
    "offset"
] = action_offset

# 学習時の観測順序を確認
assert list(
    env.observation_manager.active_terms["actor"]
) == [
    "command",
    "motion_anchor_ori_b",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
]

with (params_dir / "deploy.yaml").open("w") as f:
    yaml.safe_dump(
        deploy_cfg,
        f,
        sort_keys=False,
    )

shutil.copy2(
    motion_file,
    params_dir / motion_file.name,
)

# ONNXの入出力を確認
onnx_model = onnx.load(
    str(exported_dir / "policy.onnx")
)
onnx.checker.check_model(onnx_model)

for item in onnx_model.graph.input:
    print("Input:", item)

for item in onnx_model.graph.output:
    print("Output:", item)

print("Export completed:", output_dir)

env.close()
