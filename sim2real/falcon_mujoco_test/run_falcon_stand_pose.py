"""FALCON stands still in MuJoCo for a fixed time, starting from a given full-body standing pose.

Reuses the headless harness in run_falcon_mujoco.py (FALCON's own deploy policy + sim2real simulator,
DDS replaced by in-process hand-over, policy at 50 Hz, physics at 200 Hz). Differences to that harness:
  * the initial joint configuration comes from a pose JSON instead of DEFAULT_DOF_ANGLES;
  * the arm IK upper-body controller is disabled and ref_upper_dof_pos is set directly to the pose's
    14 arm joints (absolute targets, residual action on top as in the deploy code);
  * the lower body gets a stance command (command_stand = 0, zero velocity, base height 0.75, waist 0)
    for the whole run;
  * no video; the full qpos trajectory is saved to an npz log instead.

Usage (from FALCON/sim2real, fcreal conda env):
    PYTHONDONTWRITEBYTECODE=1 python falcon_mujoco_test/run_falcon_stand_pose.py \
        --pose /path/to/stand_pose.json --log /path/to/C4_falcon.npz
"""

import argparse
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_falcon_mujoco import (  # noqa: E402
    CONFIG,
    CONTROL_DT,
    MODEL,
    SIM2REAL,
    OfflineLocoManipPolicy,
    OfflineSimulator,
    set_loco_command,
)

NUM_BODY_JOINTS = 29
ARM_SLICE = slice(15, 29)  # MuJoCo / motor order: 12 legs, 3 waist, 7 left arm, 7 right arm


def body_name(m, b):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""


def robot_self_contacts(m, d_scratch, qpos):
    """[body1, body2, dist] for every contact between two robot bodies at this configuration."""
    d_scratch.qpos[:] = qpos
    mujoco.mj_kinematics(m, d_scratch)
    mujoco.mj_collision(m, d_scratch)
    out = []
    for i in range(d_scratch.ncon):
        c = d_scratch.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b1 == 0 or b2 == 0:  # floor / world
            continue
        out.append([body_name(m, b1), body_name(m, b2), float(c.dist)])
    return out


def torso_tilt_deg(m, d_scratch, qpos):
    d_scratch.qpos[:] = qpos
    mujoco.mj_kinematics(m, d_scratch)
    z = d_scratch.xmat[m.body("torso_link").id].reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(np.clip(z[2], -1.0, 1.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", required=True, help="pose JSON with a 'body' dict of 29 joint angles")
    ap.add_argument("--log", required=True, help="output npz path")
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--model_path", default=str(MODEL))
    args = ap.parse_args()
    pose = json.load(open(args.pose))["body"]
    log_path = Path(args.log).resolve()
    model_path = str(Path(args.model_path).resolve())

    os.chdir(SIM2REAL)  # paths in the deploy configs are relative to sim2real/
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["use_upper_body_controller"] = False  # nothing may overwrite ref_upper_dof_pos

    policy = OfflineLocoManipPolicy(config, model_path, rl_rate=50, policy_action_scale=0.25)
    sim = OfflineSimulator(config, policy)
    m, d = sim.mj_model, sim.mj_data
    decimation = int(round(CONTROL_DT / sim.sim_dt))

    # Initial configuration in MuJoCo joint order (the deploy dof_names mislabel the hip order, so the
    # model's own joint names are used here).
    jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    body_joints = [n for n in jnames if m.jnt_type[jnames.index(n)] != mujoco.mjtJoint.mjJNT_FREE]
    assert len(body_joints) == NUM_BODY_JOINTS and all(m.jnt_qposadr[jnames.index(n)] == 7 + i
                                                       for i, n in enumerate(body_joints))
    q0 = np.array([pose[n] for n in body_joints])
    sim.reset_standing(q0)  # identity quat, x = y = 0, lowest foot contact sphere 1 mm above the floor
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    # Commands (constant for the whole run)
    set_loco_command(policy, None)  # stance: command_stand = 0, zero lin/ang velocity
    policy.base_height_command[0, 0] = config["DESIRED_BASE_HEIGHT"]
    policy.waist_dofs_command[:] = 0.0
    policy.ref_upper_dof_pos[0] = q0[ARM_SLICE]
    assert policy.upper_body_controller is None and policy.residual_upper_body_action

    policy._handle_start_policy()
    sim.robot_bridge.PublishLowState()

    d_scratch = mujoco.MjData(m)
    n_ticks = int(round(args.duration / CONTROL_DT))
    ts, qposes, targets, contacts = [0.0], [d.qpos.copy()], [], [robot_self_contacts(m, d_scratch, d.qpos)]

    def arm_targets():
        mc = policy.command_sender.low_cmd.motor_cmd
        return np.array([mc[i].q for i in range(ARM_SLICE.start, ARM_SLICE.stop)])

    fell = False
    for k in range(n_ticks):
        policy.policy_action()
        targets.append(arm_targets())
        for _ in range(decimation):
            sim.sim_step()
        ts.append(round((k + 1) * CONTROL_DT, 6))
        qposes.append(d.qpos.copy())
        contacts.append(robot_self_contacts(m, d_scratch, d.qpos))
        if not fell and (d.qpos[2] < 0.45 or torso_tilt_deg(m, d_scratch, d.qpos) > 60.0):
            fell = True
            print(f"!!! fall at t={ts[-1]:.2f}s")
    policy.policy_action()  # command that would follow the last state (logged only, not simulated)
    targets.append(arm_targets())

    qpos = np.array(qposes)
    meta = dict(
        controller="C4 FALCON",
        sim_dt=sim.sim_dt,
        control_dt=CONTROL_DT,
        decimation=decimation,
        policy=model_path,
        config=str(Path(args.config).resolve()),
        commands=dict(command_stand=0, lin_vel=[0.0, 0.0], ang_vel=0.0,
                      base_height=float(policy.base_height_command[0, 0]), waist_dofs=[0.0, 0.0, 0.0],
                      ref_upper_dof_pos=q0[ARM_SLICE].tolist()),
        initial_state="stand pose joints, identity quat, x=y=0, lowest foot contact sphere 1 mm above floor, qvel=0",
        deviations_from_native_harness=[
            "arm IK upper-body controller disabled (use_upper_body_controller=False); ref_upper_dof_pos set "
            "directly to the pose's arm joints instead of IK of EE waypoints",
            "initial joint configuration = unified stand pose instead of DEFAULT_DOF_ANGLES "
            "(DEFAULT_DOF_ANGLES, i.e. observation/action offsets, unchanged)",
            "constant stance command for the whole run (no scripted walking / EE segments)",
            "no video rendering; qpos logged at the policy rate",
        ],
        same_as_native_harness=[
            "elastic band disabled, policy started at t=0",
            "DDS replaced by in-process lowstate/lowcmd hand-over; state published before each physics step",
        ],
        fell=fell,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(log_path, t=np.array(ts), qpos=qpos, xml=str(Path(config["ROBOT_SCENE"]).resolve()),
             arm_joint_names=np.array(body_joints[ARM_SLICE]), arm_pd_target=np.array(targets),
             self_contacts=json.dumps(contacts), meta=json.dumps(meta))

    # Summary
    t2 = qpos[-1]
    print(f"Wrote {log_path}")
    print(f"fell={fell}  pelvis z t0={qpos[0, 2]:.4f}  t{ts[-1]:.2f}={t2[2]:.4f}  "
          f"xy drift={np.linalg.norm(t2[:2] - qpos[0, :2]) * 100:.2f} cm  "
          f"torso tilt={torso_tilt_deg(m, d_scratch, t2):.2f} deg")
    dq = np.abs(t2[7:] - q0)
    print(f"max|q-pose| at end: arms {dq[ARM_SLICE].max():.4f} rad ({body_joints[15 + dq[ARM_SLICE].argmax()]}), "
          f"legs+waist {dq[:15].max():.4f} rad ({body_joints[dq[:15].argmax()]})")
    pairs = {}
    for step in contacts:
        for b1, b2, dist in step:
            key = tuple(sorted((b1, b2)))
            pairs[key] = min(pairs.get(key, np.inf), dist)
    print("self-contacts (min dist over run):", {f"{a}|{b}": round(v, 4) for (a, b), v in pairs.items()} or "none")


if __name__ == "__main__":
    main()
