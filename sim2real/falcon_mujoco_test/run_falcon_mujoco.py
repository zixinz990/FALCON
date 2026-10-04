"""Headless FALCON sim2sim in MuJoCo with scripted locomotion + EE commands, rendered to mp4.

The policy side is FALCON's own deploy code (rl_policy/loco_manip/loco_manip.py: observation history,
arm IK from EE waypoints, residual upper-body action) and the physics side is sim_env/loco_manip.py
(FALCON's G1 scene, 200 Hz, PD torques from the lowcmd kp/kd, effort-limit clipping). Only the DDS
transport and the keyboard/joystick threads are replaced: lowstate/lowcmd are handed over in-process
and the policy runs synchronously every 4 physics steps (50 Hz), so the run is deterministic.

Usage (from FALCON/sim2real, fcreal conda env):
    MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 python falcon_mujoco_test/run_falcon_mujoco.py \
        --out falcon_mujoco_test/falcon_mujoco_demo.mp4
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2
import mujoco
import numpy as np
import yaml

SIM2REAL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIM2REAL.parent))

from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_  # noqa: E402

from sim2real.rl_policy.loco_manip.loco_manip import LocoManipPolicy  # noqa: E402
from sim2real.sim_env.loco_manip import LocoManipSimulator  # noqa: E402
from sim2real.utils.comm.command_sender import UnitreeCommandSender  # noqa: E402
from sim2real.utils.comm.state_processor import UnitreeStateProcessor  # noqa: E402
from sim2real.utils.sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge  # noqa: E402

CONFIG = SIM2REAL / "config/g1/g1_29dof_falcon.yaml"
MODEL = SIM2REAL / "models/falcon/g1_29dof.onnx"

CONTROL_DT = 0.02  # rl_rate = 50 Hz


# ---------------------------------------------------------------------------
# In-process replacements for the DDS endpoints
# ---------------------------------------------------------------------------
class InProcessStateProcessor(UnitreeStateProcessor):
    """UnitreeStateProcessor fed directly by the simulator instead of an rt/lowstate subscriber."""

    def _init_sdk_components(self):
        pass


class InProcessCommandSender(UnitreeCommandSender):
    """UnitreeCommandSender that keeps the filled LowCmd for the simulator instead of publishing rt/lowcmd."""

    def _init_sdk_components(self):
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.InitUnitreeLowCmd()

    def send_command(self, cmd_q, cmd_dq, cmd_tau, dof_pos_latest=None):
        self._fill_motor_commands(self.low_cmd.motor_cmd, cmd_q, cmd_dq, cmd_tau)


class OfflineLocoManipPolicy(LocoManipPolicy):
    """LocoManipPolicy without DDS and input threads; commands are set by the script."""

    def _init_sdk_components(self):
        pass

    def _init_communication_components(self):
        self.state_processor = InProcessStateProcessor(self.config)
        self.command_sender = InProcessCommandSender(self.config)

    def _init_input_device(self):
        self.use_joystick = False


class InProcessBridge(UnitreeSdk2Bridge):
    """UnitreeSdk2Bridge whose lowstate 'publisher' is the policy's state handler and whose lowcmd is
    the policy's command buffer."""

    def __init__(self, mj_model, mj_data, config, policy):
        self.policy = policy
        super().__init__(mj_model, mj_data, config)

    def _init_sdk_components(self):
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.low_state_puber = SimpleNamespace(Write=self.policy.state_processor.LowStateHandler_hg)
        self.crc = SimpleNamespace(Crc=lambda msg: 0)

    @property
    def low_cmd(self):
        return self.policy.command_sender.low_cmd


class OfflineSimulator(LocoManipSimulator):
    """LocoManipSimulator without the passive viewer, DDS factory and real-time sim thread."""

    def __init__(self, config, policy):
        self.config = config
        self.init_config()
        self.mj_model = mujoco.MjModel.from_xml_path(config["ROBOT_SCENE"])
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        self.base_id = self.mj_model.body(config.get("BASE_BODY_NAME", "pelvis")).id
        self.elastic_band = ElasticBand()
        self.elastic_band.enable = False  # start standing on the ground, as in training resets
        self.band_attached_link = self.mj_model.body(config.get("BAND_ATTACHED_LINK", "torso_link")).id
        self.robot_bridge = InProcessBridge(self.mj_model, self.mj_data, config, policy)
        self.EE_xfrc = 0
        self.t = 0
        self.left_hand_link_name = config.get("left_hand_link_name", "left_hand_link")
        self.right_hand_link_name = config.get("right_hand_link_name", "right_hand_link")

    def reset_standing(self, q_default):
        """Default joint angles with the foot contact spheres just touching the floor."""
        m, d = self.mj_model, self.mj_data
        mujoco.mj_resetData(m, d)
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[7:] = q_default
        mujoco.mj_forward(m, d)
        lowest = min(d.geom_xpos[g, 2] - m.geom_size[g, 0] for g in range(m.ngeom)
                     if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and m.geom_contype[g])
        d.qpos[2] -= lowest - 0.001
        mujoco.mj_forward(m, d)


# ---------------------------------------------------------------------------
# Command script
# ---------------------------------------------------------------------------
# Arm key poses (shoulder pitch/roll/yaw, elbow), the 4 joints per arm the deploy IK solves for. The EE
# waypoints are taken from FK of these configurations so that they are reachable.
EE_KEYPOSES = {
    "both_hands_forward": dict(left=[-1.05, 0.15, 0.0, 0.45], right=[-1.05, -0.15, 0.0, 0.45]),
    "right_hand_up": dict(right=[-2.5, -0.25, 0.0, 0.25]),
    "arms_spread": dict(left=[-0.15, 1.25, 0.0, 0.25], right=[-0.15, -1.25, 0.0, 0.25]),
}
FWD_SPEED, SLOW_SPEED = 0.8, 0.5


def build_schedule(seed=0, ee_seg=2.4):
    """List of (t_start, t_end, kind, params, label)."""
    rng = np.random.default_rng(seed)
    sched = [
        (0.0, 1.0, "loco", dict(), "Stand"),
        (1.0, 3.2, "loco", dict(vel=(FWD_SPEED, 0.0, 0.0)), f"Walk forward (vx {FWD_SPEED} m/s)"),
        (3.2, 5.2, "loco", dict(vel=(0.0, SLOW_SPEED, 0.0)), f"Side-step left (vy {SLOW_SPEED} m/s)"),
        (5.2, 7.6, "loco", dict(vel=(-SLOW_SPEED, 0.0, 0.0)), f"Walk backward (vx -{SLOW_SPEED} m/s)"),
        (7.6, 9.6, "loco", dict(vel=(0.0, -SLOW_SPEED, 0.0)), f"Side-step right (vy -{SLOW_SPEED} m/s)"),
    ]
    # random walk: random 45-deg-binned direction + random turning rate per segment
    t, t_end = 9.6, 12.9
    seg_len = (t_end - t) / 3
    dirs = np.arange(8) * math.pi / 4
    dir_names = ["fwd", "fwd-left", "left", "back-left", "back", "back-right", "right", "fwd-right"]
    for _ in range(3):
        k = int(rng.integers(0, 8))
        turn = float(rng.uniform(-0.9, 0.9))
        speed = FWD_SPEED if k == 0 else SLOW_SPEED
        vel = (speed * math.cos(dirs[k]), speed * math.sin(dirs[k]), turn)
        sched.append((t, t + seg_len, "loco", dict(vel=vel),
                      f"Random walk: {dir_names[k]} {speed} m/s, turn {turn:+.2f} rad/s"))
        t += seg_len
    sched.append((12.9, 13.6, "loco", dict(), "Stand"))
    # EE commands while standing
    t = 13.6
    for pose, label in [("both_hands_forward", "EE: both hands forward"), ("right_hand_up", "EE: right hand up"),
                        ("arms_spread", "EE: arms spread"), ("start", "EE: back to start pose")]:
        sched.append((t, t + ee_seg, "ee", dict(pose=pose), label))
        t += ee_seg
    return sched


def set_loco_command(policy, vel):
    """vel=None -> stance (command_stand 0), else walk with (vx, vy, wz) in the base frame."""
    if vel is None:
        policy.stand_command[0, 0] = 0
        policy.lin_vel_command[0] = 0.0
        policy.ang_vel_command[0, 0] = 0.0
    else:
        policy.stand_command[0, 0] = 1
        policy.lin_vel_command[0] = vel[:2]
        policy.ang_vel_command[0, 0] = vel[2]


def set_ee_waypoints(policy, left, right):
    policy.EE_left_x, policy.EE_left_y, policy.EE_left_z = left
    policy.EE_right_x, policy.EE_right_y, policy.EE_right_z = right
    policy.update_waypoints()


def keypose_waypoints(ik, start, pose):
    """EE waypoints (pelvis frame) of a key pose; an arm without a key configuration keeps its start waypoint."""
    if pose == "start":
        return start
    cfg = EE_KEYPOSES[pose]
    q = np.concatenate([cfg.get("left", np.zeros(4)), cfg.get("right", np.zeros(4))])
    L, R = ik.get_end_effector_poses(q)
    return (L.translation.copy() if "left" in cfg else start[0],
            R.translation.copy() if "right" in cfg else start[1])


class EEFrames:
    """The IK's L_ee/R_ee frames (elbow joint + fixed offset) evaluated on the simulated robot."""

    def __init__(self, ik, mj_model):
        rm = ik.reduced_robot.model
        self.body_ids, self.offsets = [], []
        for fid in (ik.L_hand_id, ik.R_hand_id):
            frame = rm.frames[fid]
            self.body_ids.append(mj_model.body(rm.names[frame.parentJoint].replace("_joint", "_link")).id)
            self.offsets.append(frame.placement.translation.copy())
        self.pelvis = mj_model.body("pelvis").id

    def world(self, d):
        return np.array([d.xpos[b] + d.xmat[b].reshape(3, 3) @ o for b, o in zip(self.body_ids, self.offsets)])

    def pelvis_to_world(self, d, p):
        return d.xpos[self.pelvis] + d.xmat[self.pelvis].reshape(3, 3) @ p

    def in_pelvis(self, d):
        R = d.xmat[self.pelvis].reshape(3, 3)
        return (self.world(d) - d.xpos[self.pelvis]) @ R


def calc_heading(quat_wxyz):
    w, x, y, z = quat_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def add_sphere(scene, pos, rgba, radius=0.035):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]), np.asarray(pos, dtype=np.float64),
                        np.eye(3).reshape(-1), np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def add_arrow(scene, start, end, rgba, width=0.02):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(start, dtype=np.float64),
                         np.asarray(end, dtype=np.float64))
    scene.ngeom += 1


def draw_text(img, lines, org=(18, 36), scale=0.8):
    y = org[1]
    for i, (txt, color) in enumerate(lines):
        s = scale if i == 0 else scale * 0.8
        cv2.putText(img, txt, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, txt, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, s, color, 2, cv2.LINE_AA)
        y += int(36 * s + 6)


class FFmpegWriter:
    def __init__(self, path, width, height, fps):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-threads", "4",
             str(path)],
            stdin=subprocess.PIPE)

    def append(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "falcon_mujoco_demo.mp4"))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--model_path", default=str(MODEL))
    ap.add_argument("--duration", type=float, default=None, help="default: end of the command schedule")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None, help="optional per-tick JSON log path")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    log_path = Path(args.log).resolve() if args.log else None
    model_path = str(Path(args.model_path).resolve())

    os.chdir(SIM2REAL)  # paths in the deploy configs are relative to sim2real/
    with open(args.config) as f:
        config = yaml.safe_load(f)

    policy = OfflineLocoManipPolicy(config, model_path, rl_rate=50, policy_action_scale=0.25)
    sim = OfflineSimulator(config, policy)
    decimation = int(round(CONTROL_DT / sim.sim_dt))
    sim.reset_standing(policy.default_dof_angles)
    ik = policy.upper_body_controller
    ee = EEFrames(ik, sim.mj_model)
    ee_start = (policy.waypoints_left[0].translation.copy(), policy.waypoints_right[0].translation.copy())

    schedule = build_schedule(args.seed)
    duration = args.duration if args.duration is not None else schedule[-1][1]
    for s in schedule:
        print(f"  [{s[0]:5.2f}, {s[1]:5.2f})  {s[4]}")

    m, d = sim.mj_model, sim.mj_data
    renderer = writer = None
    if not args.no_video:
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, args.width)
        m.vis.global_.offheight = max(m.vis.global_.offheight, args.height)
        renderer = mujoco.Renderer(m, height=args.height, width=args.width)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = 3.4, 150.0, -14.0
        cam.lookat[:] = [0, 0, 0.72]
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = FFmpegWriter(out, args.width, args.height, args.fps)

    policy._handle_start_policy()  # ']' in the policy terminal
    sim.robot_bridge.PublishLowState()
    log = []
    n_ticks = int(round(duration / CONTROL_DT))
    next_frame_t = 0.0
    wall0 = time.time()
    fell = False
    cur_seg = None

    for k in range(n_ticks):
        t = k * CONTROL_DT
        seg = next(s for s in schedule if s[0] <= t + 1e-9 < s[1] or s is schedule[-1])
        kind, p, label = seg[2], seg[3], seg[4]
        if seg is not cur_seg:
            cur_seg = seg
            set_loco_command(policy, p.get("vel"))
            if kind == "ee":
                set_ee_waypoints(policy, *keypose_waypoints(ik, ee_start, p["pose"]))

        policy.policy_action()

        for _ in range(decimation):
            sim.sim_step()
            if writer is not None and d.time >= next_frame_t - 1e-9:
                next_frame_t += 1.0 / args.fps
                pel = d.xpos[ee.pelvis]
                cam.lookat[:] = 0.9 * cam.lookat + 0.1 * np.array([pel[0], pel[1], 0.72])
                renderer.update_scene(d, camera=cam)
                sc = renderer.scene
                vel = p.get("vel")
                if vel is not None and math.hypot(vel[0], vel[1]) > 1e-6:
                    h = calc_heading(d.qpos[3:7])
                    mv = np.array([vel[0] * math.cos(h) - vel[1] * math.sin(h),
                                   vel[0] * math.sin(h) + vel[1] * math.cos(h), 0.0])
                    mv /= np.linalg.norm(mv)
                    st = np.array([pel[0], pel[1], 0.03]) + 0.25 * mv
                    add_arrow(sc, st, st + 0.7 * mv, [1.0, 0.55, 0.0, 0.95], 0.04)
                if kind == "ee":
                    act = ee.world(d)
                    for i, (tgt, col) in enumerate(((ik.current_L_tf, [0.1, 0.9, 0.2, 0.8]),
                                                    (ik.current_R_tf, [0.95, 0.2, 0.2, 0.8]))):
                        add_sphere(sc, ee.pelvis_to_world(d, tgt), col, 0.045)
                        add_sphere(sc, act[i], [1, 1, 1, 0.9], 0.02)
                frame = renderer.render().copy()
                lines = [(f"FALCON sim2sim (MuJoCo)   t = {d.time:5.2f} s", (255, 255, 255)),
                         (f"Command: {label}", (255, 210, 60))]
                if kind == "ee":
                    lines.append(("Upper body: EE waypoints -> arm IK -> ref_upper_dof_pos   "
                                  "green/red = L/R target, white = actual", (180, 230, 255)))
                elif vel is None:
                    lines.append(("Lower body: stance (command_stand = 0)", (180, 230, 255)))
                else:
                    lines.append((f"Lower body: command_lin_vel = ({vel[0]:+.2f}, {vel[1]:+.2f}), "
                                  f"command_ang_vel = {vel[2]:+.2f}   orange arrow = commanded direction",
                                  (180, 230, 255)))
                draw_text(frame, lines)
                writer.append(frame)

        # ----------------------------------------------- logging
        pel = d.qpos[0:3].copy()
        entry = dict(t=round(t + CONTROL_DT, 3), seg=label, pelvis=pel.tolist(), heading=calc_heading(d.qpos[3:7]))
        if kind == "ee":
            act_rel = ee.in_pelvis(d)
            tgt = np.array([ik.current_L_tf, ik.current_R_tf])
            entry["ee_err"] = np.linalg.norm(act_rel - tgt, axis=1).tolist()
            wp = np.array([policy.waypoints_left[0].translation, policy.waypoints_right[0].translation])
            entry["ee_move"] = np.linalg.norm(wp - np.array(ee_start), axis=1).tolist()
        log.append(entry)
        if pel[2] < 0.45 and not fell:
            fell = True
            print(f"!!! robot fell at t={t:.2f}s (pelvis z={pel[2]:.3f})")

    if writer is not None:
        writer.close()
        renderer.close()
        print(f"Wrote {out}")
    wall = time.time() - wall0
    print(f"Simulated {duration:.1f}s in {wall:.1f}s wall, fell={fell}")
    summarize(log, schedule)
    if log_path:
        with open(log_path, "w") as f:
            json.dump(log, f)


def summarize(log, schedule):
    print("\nPer-segment summary (displacement expressed in the robot's heading frame at segment start):")
    print(f"{'segment':48s} {'fwd[m]':>8s} {'left[m]':>8s} {'dyaw[deg]':>9s} {'min z':>6s} "
          f"{'EE err L/R[cm]':>15s} {'EE target shift L/R[cm]':>24s}")
    for s in schedule:
        ents = [e for e in log if s[0] < e["t"] <= s[1] + 1e-9]
        if not ents:
            continue
        p0 = np.array(ents[0]["pelvis"])
        p1 = np.array(ents[-1]["pelvis"])
        h0 = ents[0]["heading"]
        dp = p1 - p0
        fwd = dp[0] * math.cos(h0) + dp[1] * math.sin(h0)
        left = -dp[0] * math.sin(h0) + dp[1] * math.cos(h0)
        dyaw = math.degrees((ents[-1]["heading"] - h0 + math.pi) % (2 * math.pi) - math.pi)
        minz = min(e["pelvis"][2] for e in ents)
        ee = mv = ""
        if "ee_err" in ents[-1]:
            tail = 100 * np.mean([e["ee_err"] for e in ents[-10:]], axis=0)
            ee = f"{tail[0]:.1f} / {tail[1]:.1f}"
            shift = 100 * np.array(ents[-1]["ee_move"])
            mv = f"{shift[0]:.1f} / {shift[1]:.1f}"
        print(f"{s[4]:48s} {fwd:8.2f} {left:8.2f} {dyaw:9.1f} {minz:6.3f} {ee:>15s} {mv:>24s}")


if __name__ == "__main__":
    main()
